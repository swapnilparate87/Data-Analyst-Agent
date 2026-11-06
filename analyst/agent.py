"""LLM analyst: turns a question into SQL, KPIs, charts and a written answer.

The model runs on NVIDIA's API (Nemotron 30B, the default when NVIDIA_API_KEY is set) or locally
via Ollama (gemma2:2b). ANALYST_PROVIDER picks one; the app can also switch per conversation.
Small models are unreliable at writing analytical SQL and at free-form tool calling,
so the model's job is kept narrow, with structured outputs forcing valid JSON:

  1. analyse - the query analyser (analyst/query.py, no model) splits the message into questions
               and finds each one's intent, places, period and topic
  2. route   - only questions the analyser can't settle go to the model's router
  3a. report / compare - prewritten queries compute the KPIs, charts and facts, with benchmarks
               (analyst/reports.py); the model only writes the narrative
  3b. direct - SQL from a code template, or written by the model (failed queries go back to it to
               fix); then the answer is written from the real result
"""

import json
import os
import re
from dataclasses import dataclass, field

import duckdb
import ollama
import pandas as pd
from openai import OpenAI

from analyst import db, query, reports
from analyst.query import DATA_COVERAGE, DATA_WORDS, DIRECT_START, REPORT_WORDS

# Provider -> default model. "nvidia" runs on NVIDIA's API, "ollama" runs locally.
MODELS = {"nvidia": "nvidia/nemotron-3.5-lightning-30b-a3b", "ollama": "gemma2:2b"}
PROVIDER = os.getenv("ANALYST_PROVIDER", "nvidia" if os.getenv("NVIDIA_API_KEY") else "ollama")
MODEL = os.getenv("ANALYST_MODEL", MODELS[PROVIDER])
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Nemotron is a reasoning model: thinking makes each call 10-100x slower, so it's off by default.
NVIDIA_THINKING = os.getenv("ANALYST_THINKING", "false").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = int(os.getenv("ANALYST_NUM_CTX", "8192"))
REQUEST_TIMEOUT = float(os.getenv("ANALYST_TIMEOUT", "180"))  # seconds per model call
SQL_FIX_ATTEMPTS = 2
PREVIEW_ROWS = 20
HISTORY_TURNS = 3  # earlier Q&A pairs kept for follow-up questions

ROUTE_PROMPT = """You are a data analyst for Olist, a Brazilian e-commerce marketplace (2016-2018).

Tables (DuckDB SQL dialect):
{schema}

How to query:
- Query FROM sales for almost everything. Each row is ONE item sold, already joined to its order, \
customer, product category (English, snake_case like 'health_beauty'), seller and the order's \
review score. Use payments only for payment methods and reviews only for review text.
- Products have NO names. A question about products groups BY product_id (not category) and \
shows ANY_VALUE(category) AS category next to it.
- "Best/most selling" means the most items sold: COUNT(*) rows. "Top by revenue" means SUM(price).
- "How many orders" is always COUNT(DISTINCT order_id), never COUNT(*). Customers: \
COUNT(DISTINCT customer_unique_id). Items sold: COUNT(*).
- Revenue: SUM(price), excluding order_status 'canceled' and 'unavailable'. Use this status \
filter ONLY when computing revenue.
- Late deliveries: delivered_late is true. Delivery time: AVG(delivery_days).
- Cities are lowercase without accents ('sao paulo', 'rio de janeiro'); states are two-letter \
codes ('SP', 'RJ'). "City" or "state" means the customer's: customer_city, customer_state. Use \
seller_city/seller_state only when the question asks where sellers are.
- Add a filter only for what the question names (a year, a city, a category...). For a year use \
YEAR(order_purchase_timestamp) = <year>.
- Rankings use GROUP BY, ORDER BY the measure, and LIMIT (LIMIT 1 for a single answer). Return \
the name column and the measure. Sort DESC for most/top/highest/best/slowest (slowest delivery = \
most delivery_days) and ASC for least/lowest/worst/fewest/fastest.

Query shapes (replace the <placeholders> with what the question asks):
- Ranking: SELECT <group column>, <measure> AS <alias> FROM sales GROUP BY <group column> \
ORDER BY <alias> DESC LIMIT <n>
- One total: SELECT <measure> AS <alias> FROM sales WHERE <only the filters the question names>
- A share in percent: SELECT 100.0 * COUNT(DISTINCT order_id) FILTER (WHERE <condition>) / \
COUNT(DISTINCT order_id) AS <alias> FROM sales

The data covers ONLY: orders, items, revenue and prices, freight, product categories and IDs, \
customer cities and states, sellers, delivery times and late deliveries, payments and review \
scores. It has no product names, profit or costs, marketing, stock, employees, competitors, or \
anything outside Olist.

Classify the user's question:
- answer_type "out_of_scope": the question can't be answered from this data, e.g. general \
knowledge, other companies, weather, news, coding help, advice, predictions, or data the \
database doesn't have (profit, marketing spend). Leave sql and entity_value empty, entity_type "none".
- answer_type "report": asks about one specific category, city, state, seller or product \
("how is X doing", "report on X", "tell me about X", "analyze X"), or for an overview of the \
whole marketplace. Set entity_type and entity_value (the name as the user wrote it; \
entity_type "none" and entity_value "" for the whole marketplace). Leave sql empty.
- answer_type "direct": anything else, e.g. a count, a top-N list or a comparison. Write ONE \
DuckDB SELECT in sql that returns exactly what was asked ("how many" means a count), with \
short snake_case column aliases. Set entity_type "none".

The sql field holds only the query: no comments, no explanation, no second statement."""

FIX_PROMPT = """This DuckDB query failed.

Question: {question}
SQL:
{sql}
Error: {error}

Available columns in `sales`: {columns}
Products have no names: use product_id. Aggregates need GROUP BY on the other selected columns.

Return a corrected single SELECT query FROM sales that uses only these columns, with no comments."""

SQL_PROMPT = """Write ONE DuckDB SELECT query FROM sales that answers: {question}
It must return exactly what was asked ("how many" means a count) and add no filters the question \
doesn't ask for. Use short snake_case aliases. Only the query: no comments or explanation.{hint}"""

ENTITY_HINT = "\nThe question mentions the {entity_type} stored as {value}: filter with {column} = {value}."

OUT_OF_SCOPE = ("I don't know. I don't have data to answer that: I can only answer questions about "
                f"Olist's Brazilian e-commerce data from {DATA_COVERAGE}: orders, revenue, product "
                "categories, customer cities and states, sellers, deliveries, payments and reviews.")

NOT_IN_DATA = ("I don't have enough data to answer that. The dataset has no profit, margin or cost "
               "figures, product names, marketing, inventory or staff data, and can't forecast the future.")
NO_MATCH = (f"I don't have enough data to answer that: nothing matched. The data covers {DATA_COVERAGE}; "
            "check the spelling of any names, or open **SQL used** below to see what was searched.")

# Numbers, but not the digits inside IDs like 6560211a19b4.
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![A-Za-z]*\d)")
MONEY_COLUMN = re.compile(r"revenue|price|freight|value|total|sales|spent|paid|payment|amount|aov|ticket", re.I)

REPORT_PROMPT = """You are a data analyst for Olist, a Brazilian e-commerce marketplace.
The user asked: "{question}"
This is a {kind} on {subject}, focused on {focus}.

Facts, all computed from the data. "vs" gives a comparison; "better"/"worse" says which way is good:
{facts}

Write markdown with exactly these three parts:
**Answer:** one or two sentences that directly answer the user's question.
**Key findings**
- 3 or 4 bullets. Each one makes a comparison (against the marketplace, the rank among peers, the \
previous period, or between the items in a breakdown) and says what it means for the business. \
Don't just repeat a figure.
**Recommendation:** one concrete next step that addresses the weakest result.

Use only numbers that appear in the facts. No title, no emojis, under 180 words."""

DIRECT_PROMPT = """You are a data analyst for Olist, a Brazilian e-commerce marketplace. \
Money is in Brazilian reais (R$).

Question: {question}

SQL that was run:
{sql}

Result ({row_count} rows{shown}):
{rows}

Answer the question in 1-3 plain sentences using only values from this result, formatting \
money as R$ with thousands separators (only for money: not for counts, scores or days). Products \
have no names, so name a product by its category and product_id. Don't describe the query or mention row counts, and \
don't add any number that isn't in the result. No headings, no emojis."""

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_type": {"type": "string", "enum": ["report", "direct", "out_of_scope"]},
        "entity_type": {"type": "string", "enum": ["category", "city", "state", "seller", "product", "none"]},
        "entity_value": {"type": "string"},
        "sql": {"type": "string"},
    },
    "required": ["answer_type", "entity_type", "entity_value", "sql"],
}
SQL_SCHEMA = {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}


MESSAGES = {"out_of_scope": OUT_OF_SCOPE, "not_in_data": NOT_IN_DATA}


class ModelOutputError(Exception):
    """The local model returned output we couldn't use (e.g. broken JSON)."""


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in NUMBER.findall(text)]


def unsupported_numbers(text: str, values: list[float]) -> list[float]:
    """Numbers in the text that match none of the values. Allows rounding (within 1%) and values
    written in thousands or millions ("R$ 151 thousand"); ignores small whole numbers ("top 3")."""
    return [n for n in numbers_in(text) if not (n <= 10 and n == int(n))
            and not any(abs(v / scale - n) <= 0.01 * max(abs(n), 1) for v in values for scale in (1, 1e3, 1e6))]


def _grounded(answer: str, df: pd.DataFrame, question: str) -> bool:
    """True if every number in the answer is in the result (or the question), and R$ is only used
    when the result has a money column (models write "R$ 28.2" for delivery days)."""
    if "R$" in answer and not any(MONEY_COLUMN.search(str(c)) for c in df.columns):
        return False
    values = [float(v) for v in df.head(PREVIEW_ROWS).select_dtypes("number").to_numpy().ravel() if pd.notna(v)]
    return not unsupported_numbers(answer, values + numbers_in(question))


def _clean_sql(sql: str) -> str:
    """Keep only the first statement and drop SQL comments the model added."""
    sql = re.sub(r"--[^\n]*", "", sql.replace("\\n", "\n")).strip()
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error:
        return sql  # let run_query report the syntax error to the fix loop
    return statements[0].query.strip() if statements else sql


def _fmt(value) -> str:
    if pd.api.types.is_integer(value):
        return f"{int(value):,}"
    if pd.api.types.is_float(value):
        return f"{value:,.2f}"
    if isinstance(value, pd.Timestamp):
        return f"{value:%Y-%m-%d}" if value == value.normalize() else f"{value:%Y-%m-%d %H:%M}"
    return str(value)


@dataclass
class Chart:
    chart_type: str
    title: str
    data: pd.DataFrame
    x: str | None
    y: str | None
    color: str | None


@dataclass
class Report:
    text: str = ""
    kpis: list[dict] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    queries: list[tuple[str, str, pd.DataFrame | str]] = field(default_factory=list)  # (purpose, sql, df or error)
    parts: list[tuple[str, "Report"]] = field(default_factory=list)  # (question, answer) when a message has several
    facts: list[str] = field(default_factory=list)  # what a report's narrative was written from


class Analyst:
    def __init__(self, con: duckdb.DuckDBPyConnection | None = None, provider: str = PROVIDER):
        self.con = con or db.connect()
        self.provider = provider
        self.model = MODEL if provider == PROVIDER else MODELS[provider]  # ANALYST_MODEL applies to PROVIDER
        if provider == "nvidia":  # a missing key fails on the first call, with the API's error
            self.client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=os.getenv("NVIDIA_API_KEY") or "missing",
                                 timeout=REQUEST_TIMEOUT)
        else:
            self.client = ollama.Client(host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT)
        # Only the tables the model should query: showing raw `products` (product_name_lenght...)
        # made models invent a product_name column.
        schema = "\n".join(line for line in db.schema_text(self.con).splitlines()
                           if line.startswith(("sales(", "payments(", "reviews(")))
        self.route_prompt = ROUTE_PROMPT.format(schema=schema)
        self.sales_columns = ", ".join(self.con.execute("SELECT * FROM sales LIMIT 0").df().columns)
        self.history: list[tuple[str, str]] = []  # (question, answer) for follow-ups
        self.last_parts: list[query.Part] = []  # the previous message, as analysed

    def ask(self, question: str, on_step=None) -> Report:
        """Answer one message, which may hold several questions. `on_step(str)` gets progress messages."""
        step = on_step or (lambda _: None)
        step("Understanding the question")
        parts = self.plan(question)
        if len(parts) > 1:
            step(f"Found {len(parts)} questions: " + " | ".join(p.text for p in parts))
        answers = []
        for part in parts:
            try:
                answers.append(self.run(part, step))
            except ModelOutputError:
                answers.append(Report(text=f"The model (`{self.model}`) returned an unusable response. "
                                           "Try rephrasing the question."))
        report = answers[0] if len(parts) == 1 else Report(parts=[(p.text, r) for p, r in zip(parts, answers)])
        text = report.text or "\n\n".join(f"{q}: {r.text}" for q, r in report.parts)
        self.history = (self.history + [(question, text)])[-HISTORY_TURNS:]
        self.last_parts = parts
        return report

    def plan(self, question: str) -> list[query.Part]:
        """The query analyser: split the message and decide where each question goes."""
        return query.analyze(self.con, question, self.last_parts)

    def run(self, part: query.Part, step=lambda _: None) -> Report:
        """Answer one analysed question."""
        if part.intent == "unsure":
            self.route_with_model(part)
        empty = next((p for p in part.periods if p.get("empty")), None)
        if part.intent in MESSAGES:
            return Report(text=MESSAGES[part.intent])
        if part.intent == "missing_entity":
            return Report(text=part.query)
        if empty:
            return Report(text=f"I don't have enough data for {empty['label']}: the data covers {DATA_COVERAGE}.")
        if part.intent == "compare":
            return self._compare(part, step)
        if part.intent == "report":
            return self._report(part, step)
        return self._direct(part.query, part.sql, step)

    def route_with_model(self, part: query.Part) -> None:
        """For questions the analyser can't settle from keywords, the model decides (it sees the
        conversation so far). Sets part.intent, and part.sql or part.entities."""
        q = part.query
        answer = self._chat(q, ROUTE_SCHEMA, system=self.route_prompt, history=True, max_tokens=400)
        # The model sometimes rejects real questions, so "out of scope" counts only without data words.
        if answer["answer_type"] == "out_of_scope" and not DATA_WORDS.search(q):
            part.intent = "out_of_scope"
            return
        # A "report" with no recognisable entity and no report wording is usually a misrouted list or
        # count ("which 10 sellers..."); "what is the average review in X" is narrow even though it names X.
        if not (answer["answer_type"] == "report" and (
                REPORT_WORDS.search(q) or (not DIRECT_START.search(q) and (part.entities or answer["entity_type"] != "none")))):
            part.intent, part.sql = "direct", answer["sql"]
            return
        part.intent = "report"
        entity_type, name = answer["entity_type"], answer["entity_value"].strip()
        if part.entities or entity_type == "none" or not name:
            return
        # A name the analyser didn't recognise, e.g. a typo ("Sao Paolo"): match it to the data.
        value, suggestions = reports.resolve(self.con, entity_type, name)
        if value:
            part.entities = [(entity_type, value)]
        else:
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            part.intent, part.query = "missing_entity", f"I couldn't find a {entity_type} matching **{name}** in the data.{hint}"

    def _report(self, part: query.Part, step) -> Report:
        report = Report()
        # The report is about the most specific thing named; the others narrow it ("health_beauty in SP").
        order = {"seller": 0, "product": 1, "category": 2, "city": 3, "state": 4}
        entities = sorted(part.entities, key=lambda e: order[e[0]])
        entity_type, value = entities[0] if entities else ("none", None)
        filters = entities[1:]
        subject = "the whole Olist marketplace" if value is None else f"the {entity_type} **{value}**"
        subject += "".join(f" in {v}" for _, v in filters)
        if part.period:
            subject += f", {part.period['label']}"
        step(f"Running {part.focus} report queries for {subject}")
        report.kpis, report.charts, report.facts, report.queries = reports.build(
            self.con, entity_type, value, focus=part.focus, period=part.period, filters=filters)
        step("Writing findings")
        title = f"{part.focus.title()} report on {subject}" if part.focus != "sales" else f"Report on {subject}"
        report.text = f"### {title}\n\n{self.write_narrative(part.text, subject, report.facts, part.focus)}"
        return report

    def _compare(self, part: query.Part, step) -> Report:
        report = Report()
        types = [t for t, _ in part.entities]
        compared = next((t for t in dict.fromkeys(types) if types.count(t) >= 2), None)
        if compared:
            shared = [e for e in part.entities if e[0] != compared]
            # "SP and RJ in 2017 and 2018": one subject per place and year
            periods = query.align_years(part.periods) if len(part.periods) > 1 else [part.period]
            subjects = [{"label": v + (f" {p['label']}" if len(periods) > 1 else ""), "filters": [(t, v)] + shared,
                         "period": p} for t, v in part.entities if t == compared for p in periods]
        else:  # comparing periods ("2017 vs 2018")
            subjects = [{"label": p["label"], "filters": part.entities, "period": p}
                        for p in query.align_years(part.periods)]
        if len(subjects) < 2:  # e.g. "what about MG?" after a comparison: nothing to compare with
            return self._report(part, step)
        names = " vs ".join(s["label"] for s in subjects)
        step(f"Comparing {names}")
        report.kpis, report.charts, report.facts, report.queries = reports.build_compare(
            self.con, subjects, focus=part.focus)
        step("Writing findings")
        narrative = self.write_narrative(part.text, names, report.facts, part.focus, kind="comparison")
        report.text = f"### Comparison: {names}\n\n{narrative}"
        return report

    def write_narrative(self, question: str, subject: str, facts: list[str], focus: str = "sales",
                        kind: str = "report") -> str:
        prompt = REPORT_PROMPT.format(question=question, subject=subject, focus=focus, kind=kind,
                                      facts="\n".join(f"- {f}" for f in facts))
        return self._chat(prompt, max_tokens=550)

    def _direct(self, question: str, sql: str, step) -> Report:
        report = Report()
        step("Running query")
        sql, df, report.queries = self.retrieve(question, sql, on_fix=lambda: step("Fixing the query"))
        if df is None:
            report.text = "I couldn't write a working query for this question. Try rephrasing it more specifically."
            return report
        if len(df) > 1:
            report.charts.append(Chart("table", "Result", df, None, None, None))
        step("Writing answer")
        report.text = self.write_answer(question, sql, df)
        return report

    def retrieve(self, question: str, sql: str = "", on_fix=None) -> tuple[str, pd.DataFrame | None, list]:
        """Get the data for a direct question: finish the SQL, run it, and have the model fix it if
        it fails. Returns (final sql, result or None, [(purpose, sql, df or error)])."""
        queries = []
        detected = reports.detect(self.con, question)
        if template := reports.template_sql(self.con, question):
            sql = template  # plain ranking or count: built in code, no model SQL needed
        elif detected or not sql.strip():
            # Name the stored value and its column; small models otherwise guess casing or the wrong column.
            entity_type, value = detected or (None, None)
            hint = ENTITY_HINT.format(entity_type=entity_type, value=reports.sql_str(value), column=reports.ENTITIES[entity_type]) \
                if detected else ""
            sql = self._chat(SQL_PROMPT.format(question=question, hint=hint), SQL_SCHEMA,
                             system=self.route_prompt, max_tokens=400)["sql"]
            if detected:
                sql = reports.fix_literal(sql, value)
        for attempt in range(SQL_FIX_ATTEMPTS + 1):
            sql = _clean_sql(sql)
            try:
                df = db.run_query(self.con, sql)
                queries.append(("Query", sql, df))
                return sql, self._with_category(df), queries
            except db.QueryError as e:
                queries.append(("Query", sql, str(e)))
                if attempt == SQL_FIX_ATTEMPTS:
                    break
                if on_fix:
                    on_fix()
                fix = FIX_PROMPT.format(question=question, sql=sql, error=e, columns=self.sales_columns)
                fixed = self._chat(fix, SQL_SCHEMA, max_tokens=400)["sql"]
                if fixed.strip() == sql.strip():
                    break  # the model can't fix it; don't loop on the same SQL
                sql = fixed
        return sql, None, queries

    def _with_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Products have no names, so add each product's category; otherwise the model makes one up."""
        if "product_id" not in df.columns or "category" in df.columns or df.empty:
            return df
        ids = df["product_id"].dropna().unique().tolist()
        categories = self.con.execute(
            "SELECT DISTINCT product_id, category FROM sales WHERE product_id IN (SELECT UNNEST(?))", [ids]
        ).df()
        df = df.merge(categories, on="product_id", how="left")
        cols = list(df.columns)
        cols.insert(cols.index("product_id") + 1, cols.pop(cols.index("category")))
        return df[cols]

    def write_answer(self, question: str, sql: str, df: pd.DataFrame) -> str:
        """Turn a query result into the reply text."""
        if df.empty or (df.shape == (1, 1) and df.iat[0, 0] in (0, None)):
            return NO_MATCH
        if df.shape == (1, 1):
            # A single value needs no model: small models tend to garble units ("R$ 3.91 thousand").
            return f"**Answer: {_fmt(df.iat[0, 0])}**"
        rows = df.head(PREVIEW_ROWS).to_json(orient="records", date_format="iso", double_precision=2)
        shown = f", first {PREVIEW_ROWS} shown" if len(df) > PREVIEW_ROWS else ""
        prompt = DIRECT_PROMPT.format(question=question, sql=sql, row_count=len(df), shown=shown, rows=rows)
        answer = self._chat(prompt, max_tokens=250)
        if _grounded(answer, df, question):
            return answer
        if len(df) == 1:
            return "  \n".join(f"**{col}**: {_fmt(df.iat[0, i])}" for i, col in enumerate(df.columns))
        # The model's sentence had numbers that aren't in the result: list the rows instead.
        lines = [f"{i}. **{_fmt(row.iloc[0])}**: " + ", ".join(f"{col} {_fmt(v)}" for col, v in row.iloc[1:].items())
                 for i, (_, row) in enumerate(df.head(10).iterrows(), 1)]
        more = f"\n\n…and {len(df) - 10} more rows in the table above." if len(df) > 10 else ""
        return "\n".join(lines) + more

    def _chat(self, prompt: str, schema: dict | None = None, system: str | None = None,
              history: bool = False, max_tokens: int = 400):
        """Plain text when schema is None, else a dict parsed from schema-constrained JSON.

        Small models can fall into repetition loops (especially inside JSON strings), so every
        call has a hard output cap and a repetition penalty.
        """
        messages = [{"role": "system", "content": system}] if system else []
        if history:
            for q, a in self.history:
                messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": prompt})
        if self.provider == "nvidia":
            fmt = {"response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}}
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0, **(fmt if schema else {}),
                max_tokens=max_tokens + (4000 if NVIDIA_THINKING else 0),  # thinking counts toward the cap
                extra_body={"chat_template_kwargs": {"enable_thinking": NVIDIA_THINKING}},
            )
            choice = response.choices[0]
            content, truncated = (choice.message.content or "").strip(), choice.finish_reason == "length"
        else:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=schema,
                options={"num_ctx": NUM_CTX, "temperature": 0, "num_predict": max_tokens, "repeat_penalty": 1.15},
            )
            content, truncated = response.message.content.strip(), response.done_reason == "length"
        if schema is None:
            return content + ("…" if truncated else "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Models sometimes write the query and then keep going inside the JSON string (comments,
            # repeats) until the token cap cuts it off. The query at the start is still usable.
            if schema.get("required") == ["sql"]:
                m = re.search(r'"sql"\s*:\s*"(.*?);', content, re.DOTALL)
                if m:
                    return {"sql": m.group(1).replace('\\"', '"')}
            raise ModelOutputError(content[:200]) from e
