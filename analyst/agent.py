"""LLM analyst: turns a question into SQL, KPIs, charts and a written answer.

The model runs locally via Ollama (default gemma2:2b) or, with ANALYST_PROVIDER=nvidia, on
NVIDIA's API (default Nemotron 30B). Small models are unreliable at writing analytical SQL and at
free-form tool calling, so the model's job is kept narrow, with structured outputs forcing valid JSON:

  1. route  - the model classifies the question: a report about an entity (category, city,
              state, seller, product, whole marketplace) or a direct question with one SQL query
  2a. report - the entity name is matched to a real value and prewritten queries compute the
              KPIs, charts and key facts (analyst/reports.py); the model only writes the narrative
  2b. direct - we run the model's SQL (failed queries go back to it to fix), then the model
              answers from the real result
"""

import json
import os
import re
from dataclasses import dataclass, field

import duckdb
import ollama
import pandas as pd
from openai import OpenAI

from analyst import db, reports

PROVIDER = os.getenv("ANALYST_PROVIDER", "ollama")  # "ollama" (local) or "nvidia" (NVIDIA API)
MODEL = os.getenv("ANALYST_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b" if PROVIDER == "nvidia" else "gemma2:2b")
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

ENTITY_HINT = "\nThe question mentions the {entity_type} stored as '{value}': filter with {column} = '{value}'."

OUT_OF_SCOPE = ("I don't know. I don't have data to answer that: I can only answer questions about "
                "Olist's Brazilian e-commerce data from Sep 2016 to Oct 2018 (orders, revenue, product "
                "categories, customer cities and states, sellers, deliveries, payments and reviews).")

NOT_IN_DATA = ("I don't have enough data to answer that. The dataset has no profit, margin or cost "
               "figures, product names, marketing, inventory or staff data, and can't forecast the future.")
# Things people ask about that sound like sales data but aren't in it. Small models answer these
# anyway with whatever column looks closest (revenue passed off as profit), so catch them first.
MISSING_DATA = re.compile(
    r"\b(profits?|profitab\w*|margins?|net income|expenses?|marketing|advertis\w*|inventory|stock levels?|"
    r"employees?|staff|salar(y|ies)|competitors?|forecast\w*|predict\w*|next (month|year))\b"
    r"|\bnames?\b.*\bproducts?\b|\bproducts?\b.*\bnames?\b",
    re.IGNORECASE,
)

# Words that tie a question to the data. Questions without them (and without a known city, state
# or category) are treated as out of scope even when the model tries to write SQL for them.
DATA_WORDS = re.compile(
    r"\b(orders?|ordered|revenue|sales|sold|sell(ers?|ing)?|buy(ers?|ing)?|bought|purchases?|customers?|"
    r"categor(y|ies)|products?|items?|deliver(y|ies|ed)|late|shipping|freight|reviews?|ratings?|"
    r"payments?|paid|installments?|boleto|credit card|vouchers?|prices?|olist|marketplace|"
    r"months?|monthly)\b",
    re.IGNORECASE,
)
# Short follow-ups that lean on the previous answer ("and in 2017?", "what about RJ?").
FOLLOW_UP = re.compile(r"^\s*(and|what about|how about|same|now|only|just|compare|vs|versus)\b", re.IGNORECASE)

# Words that signal a report rather than a narrow question, and openings of narrow questions.
REPORT_WORDS = re.compile(
    r"\b(report|overview|summary|summari[sz]e|analy[sz]e|analysis|tell me about|perform|doing|dashboard|insights?)\b",
    re.IGNORECASE,
)
DIRECT_START = re.compile(
    r"^\s*(how (many|much)|what('s| is| was| are| were)|which|who|when|list|count|top \d+)\b", re.IGNORECASE
)
# Numbers, but not the digits inside IDs like 6560211a19b4.
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![A-Za-z]*\d)")

REPORT_PROMPT = """You are a data analyst for Olist, a Brazilian e-commerce marketplace.
Write the narrative for a report on {subject}.

Facts (all computed from the data):
{facts}

Write markdown: a "**Key findings**" list of 3-5 short bullets that interpret these facts \
(strengths, weaknesses, concentration, trend), then a one-line "**Takeaway:**". Use only numbers \
from the facts; do not invent or estimate figures. No title, no emojis, under 150 words."""

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


class ModelOutputError(Exception):
    """The local model returned output we couldn't use (e.g. broken JSON)."""


def _grounded(answer: str, df: pd.DataFrame, question: str) -> bool:
    """True if every number in the answer matches a value in the result (or the question).

    Allows rounding (within 1%) and values written in thousands or millions ("R$ 151 thousand").
    Also rejects R$ when no result column is money (models write "R$ 28.2" for delivery days).
    """
    money = re.compile(r"revenue|price|freight|value|total|sales|spent|paid|payment|amount|aov|ticket", re.I)
    if "R$" in answer and not any(money.search(str(c)) for c in df.columns):
        return False
    values = [float(v) for v in df.head(PREVIEW_ROWS).select_dtypes("number").to_numpy().ravel() if pd.notna(v)]
    values += [float(n.replace(",", "")) for n in NUMBER.findall(question)]
    for token in NUMBER.findall(answer):
        n = float(token.replace(",", ""))
        if n <= 10 and n == int(n):
            continue  # small counts like "top 3" or "5 stars"
        if not any(abs(v / scale - n) <= 0.01 * max(abs(n), 1)
                   for v in values for scale in (1, 1e3, 1e6)):
            return False
    return True


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


class Analyst:
    def __init__(self, con: duckdb.DuckDBPyConnection | None = None):
        self.con = con or db.connect()
        if PROVIDER == "nvidia":
            self.client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=os.getenv("NVIDIA_API_KEY"), timeout=REQUEST_TIMEOUT)
        else:
            self.client = ollama.Client(host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT)
        # Only the tables the model should query: showing raw `products` (product_name_lenght...)
        # made models invent a product_name column.
        schema = "\n".join(line for line in db.schema_text(self.con).splitlines()
                           if line.startswith(("sales(", "payments(", "reviews(")))
        self.route_prompt = ROUTE_PROMPT.format(schema=schema)
        self.sales_columns = ", ".join(self.con.execute("SELECT * FROM sales LIMIT 0").df().columns)
        self.history: list[tuple[str, str]] = []  # (question, answer) for follow-ups

    def ask(self, question: str, on_step=None) -> Report:
        """Answer one question. `on_step(str)` is called with progress messages."""
        step = on_step or (lambda _: None)
        step("Understanding the question")
        try:
            route = self.classify(question)
            if route["kind"] == "out_of_scope":
                report = Report(text=OUT_OF_SCOPE)
            elif route["kind"] == "not_in_data":
                report = Report(text=NOT_IN_DATA)
            elif route["kind"] == "report":
                report = self._report(question, route["entity_type"], route["entity_value"], step)
            else:
                report = self._direct(question, route["sql"], step)
        except ModelOutputError:
            report = Report(text=f"The local model (`{MODEL}`) returned an unusable response. Try rephrasing the question.")
        self.history = (self.history + [(question, report.text)])[-HISTORY_TURNS:]
        return report

    # Each step below is a separate component so it can be evaluated on its own (see evals/).

    def classify(self, question: str) -> dict:
        """Scope check and routing. Returns {kind, entity_type, entity_value, sql} where kind is
        "out_of_scope", "not_in_data", "report" or "direct"."""
        route = {"kind": "out_of_scope", "entity_type": "none", "entity_value": "", "sql": ""}
        # Scope check in two layers. Keywords first: the model will happily write SQL for "who won
        # the World Cup", so a question that mentions nothing in the data (and isn't a follow-up
        # like "and in 2017?") is refused without a model call. Then the model's own judgement,
        # trusted only when no data word backs the question (it sometimes rejects real ones).
        detected = reports.detect(self.con, question)
        data_words = bool(DATA_WORDS.search(question))
        follow_up = bool(self.history and FOLLOW_UP.search(question))
        has_report_words = bool(REPORT_WORDS.search(question))  # "give me an overview": let the model judge
        if not (data_words or detected or follow_up or has_report_words):
            return route
        if MISSING_DATA.search(question):
            return {**route, "kind": "not_in_data"}
        answer = self._chat(question, ROUTE_SCHEMA, system=self.route_prompt, history=True, max_tokens=400)
        if answer["answer_type"] == "out_of_scope" and not data_words:
            return route
        # A "report" with no recognisable entity and no report wording is usually a misrouted
        # list/count question ("which 10 sellers..."), so answer it directly instead.
        # Likewise "what is the average review in X" is a narrow question even though it names X.
        is_report = answer["answer_type"] == "report" and (
            has_report_words
            or (not DIRECT_START.search(question) and (detected or answer["entity_type"] != "none"))
        )
        return {"kind": "report" if is_report else "direct", "entity_type": answer["entity_type"],
                "entity_value": answer["entity_value"].strip(), "sql": answer["sql"]}

    def _report(self, question: str, entity_type: str, raw_value: str, step) -> Report:
        report = Report()
        value = None
        # Small models often miss or mislabel the entity, so prefer a name found verbatim in the question.
        detected = reports.detect(self.con, question)
        if detected:
            entity_type, value = detected
        elif entity_type != "none":
            value, suggestions = reports.resolve(self.con, entity_type, raw_value)
            if value is None:
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                report.text = f"I couldn't find a {entity_type} matching **{raw_value}** in the data.{hint}"
                return report

        subject = "the whole Olist marketplace" if value is None else f"the {entity_type} **{value}**"
        step(f"Running report queries for {subject}")
        report.kpis, report.charts, facts, report.queries = reports.build(self.con, entity_type, value)

        step("Writing findings")
        report.text = f"### Report on {subject}\n\n{self.write_narrative(subject, facts)}"
        return report

    def write_narrative(self, subject: str, facts: list[str]) -> str:
        prompt = REPORT_PROMPT.format(subject=subject, facts="\n".join(f"- {f}" for f in facts))
        return self._chat(prompt, max_tokens=450)

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
        # If the question names a real city/state/category, regenerate the SQL with the exact
        # stored value and column; small models otherwise guess casing or the wrong column.
        detected = reports.detect(self.con, question)
        template = reports.template_sql(self.con, question)
        if template:
            sql = template  # plain ranking or count: built in code, no model SQL needed
        elif detected or not sql.strip():
            hint = ""
            if detected:
                entity_type, value = detected
                hint = ENTITY_HINT.format(entity_type=entity_type, value=value, column=reports.ENTITIES[entity_type][0])
            prompt = SQL_PROMPT.format(question=question, hint=hint)
            sql = self._chat(prompt, SQL_SCHEMA, system=self.route_prompt, max_tokens=400)["sql"]
            if detected:
                sql = reports.fix_literal(sql, detected[1])
        elif self.history:
            # A follow-up ("and in 2017?") reuses the name as the user typed it earlier ('Curitiba').
            earlier = reports.detect(self.con, self.history[-1][0])
            if earlier:
                sql = reports.fix_literal(sql, earlier[1])
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
            return ("I don't have enough data to answer that: nothing matched. The data covers "
                    "Sep 2016 to Oct 2018; check the spelling of any names, or open **SQL used** "
                    "below to see what was searched.")
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
        if PROVIDER == "nvidia":
            extra = {"response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}} if schema else {}
            response = self.client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0, **extra,
                max_tokens=max_tokens + (4000 if NVIDIA_THINKING else 0),  # thinking counts toward the cap
                extra_body={"chat_template_kwargs": {"enable_thinking": NVIDIA_THINKING}},
            )
            content, truncated = (response.choices[0].message.content or "").strip(), response.choices[0].finish_reason == "length"
        else:
            response = self.client.chat(
                model=MODEL,
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
