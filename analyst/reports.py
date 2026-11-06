"""Everything computed in code rather than by the model: finding and matching the city, state or
category a question names, SQL for common ranking and counting questions, and report and
comparison queries (KPIs, charts and the facts the model interprets).

Small local models write unreliable SQL, so reports never depend on model-written queries:
the numbers, KPIs and key facts here are all computed in code.
"""

import difflib
import functools
import re
import unicodedata

import duckdb
import pandas as pd

from analyst import db

ACTIVE = "order_status NOT IN ('canceled', 'unavailable')"
MIN_ORDERS = 30  # averages and shares over fewer orders are noise ("2 orders, avg review 1.0")

# entity_type -> column in `sales`
ENTITIES = {"category": "category", "city": "customer_city", "state": "customer_state",
            "seller": "seller_id", "product": "product_id"}

STATE_NAMES = {
    "acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM", "bahia": "BA", "ceara": "CE",
    "distrito federal": "DF", "espirito santo": "ES", "goias": "GO", "maranhao": "MA",
    "mato grosso": "MT", "mato grosso do sul": "MS", "minas gerais": "MG", "para": "PA",
    "paraiba": "PB", "parana": "PR", "pernambuco": "PE", "piaui": "PI", "rio de janeiro": "RJ",
    "rio grande do norte": "RN", "rio grande do sul": "RS", "rondonia": "RO", "roraima": "RR",
    "santa catarina": "SC", "sao paulo": "SP", "sergipe": "SE", "tocantins": "TO",
}
# Real city names that are also common words in questions ("SP sales" is not the city of Sales).
CITY_WORDS = {"sales", "central"}

_LATE = ("ROUND(100.0 * COUNT(DISTINCT order_id) FILTER (WHERE delivered_late) / NULLIF(COUNT(DISTINCT order_id) "
         "FILTER (WHERE order_delivered_customer_date IS NOT NULL), 0), 1)")
_STARS = ("ROUND(100.0 * COUNT(DISTINCT order_id) FILTER (WHERE review_score {}) / "
          "NULLIF(COUNT(DISTINCT order_id) FILTER (WHERE review_score IS NOT NULL), 0), 1)")
# metric -> (SQL over the scoped rows, label, format, which direction is better)
METRICS = {
    "revenue": ("ROUND(SUM(price), 2)", "Revenue", "money", None),
    "orders": ("COUNT(DISTINCT order_id)", "Orders", "int", None),
    "items_sold": ("COUNT(*)", "Items sold", "int", None),
    "customers": ("COUNT(DISTINCT customer_unique_id)", "Customers", "int", None),
    "avg_order_value": ("ROUND(SUM(price) / NULLIF(COUNT(DISTINCT order_id), 0), 2)", "Avg order value", "money", "high"),
    "avg_review": ("ROUND(AVG(review_score), 2)", "Avg review", "score", "high"),
    "five_star_pct": (_STARS.format(">= 4.5"), "5-star reviews", "pct", "high"),
    "one_star_pct": (_STARS.format("<= 1.5"), "1-star reviews", "pct", "low"),
    "avg_delivery_days": ("ROUND(AVG(delivery_days), 1)", "Avg delivery time", "days", "low"),
    "late_pct": (_LATE, "Late deliveries", "pct", "low"),
    "days_early": ("ROUND(AVG(date_diff('day', order_delivered_customer_date, order_estimated_delivery_date)), 1)",
                   "Arrives ahead of estimate", "days", "high"),
    "avg_freight": ("ROUND(AVG(freight_value), 2)", "Avg freight per item", "money", "low"),
    "sellers": ("COUNT(DISTINCT seller_id)", "Active sellers", "int", None),
    "revenue_per_seller": ("ROUND(SUM(price) / NULLIF(COUNT(DISTINCT seller_id), 0), 2)", "Revenue per seller",
                           "money", "high"),
}


# --- Names in questions -------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def sql_str(value) -> str:
    """A SQL string literal: names like "santa barbara d'oeste" contain quotes."""
    return "'" + str(value).replace("'", "''") + "'"


@functools.cache  # the data is read-only, so each list is loaded once per connection
def _distinct(con, column: str) -> list:
    return [v for (v,) in con.execute(f"SELECT DISTINCT {column} FROM sales WHERE {column} IS NOT NULL").fetchall()]


def resolve(con: duckdb.DuckDBPyConnection, entity_type: str, value: str) -> tuple[str | None, list[str]]:
    """Match a user-typed name to a real value. Returns (match, suggestions)."""
    target = _norm(value)
    if not target:
        return None, []
    if entity_type == "state":
        target = _norm(STATE_NAMES.get(target, target))
    by_norm = {_norm(v): v for v in _distinct(con, ENTITIES[entity_type])}
    if target in by_norm:
        return by_norm[target], []
    if close := difflib.get_close_matches(target, list(by_norm), n=1, cutoff=0.8):
        return by_norm[close[0]], []
    if containing := sorted((n for n in by_norm if target in n), key=len):
        return by_norm[containing[0]], []
    return None, [by_norm[s] for s in difflib.get_close_matches(target, list(by_norm), n=3, cutoff=0.5)]


def fix_literal(sql: str, value: str) -> str:
    """Rewrite string literals that spell `value` differently ('Curitiba', 'São Paulo') to the stored value."""
    target = _norm(value)
    return re.sub(r"'([^']*)'", lambda m: sql_str(value) if _norm(m.group(1)) == target else m.group(0), sql)


def _matches(con, question: str) -> list[tuple]:
    """Non-overlapping (start, end, entity_type, value) matches, best first: longest, then (for a name
    that is both a city and a state, like "rio de janeiro") the state only if the question says
    "state(s)", then categories."""
    text = f" {_norm(question)} "
    found = []

    def add(name: str, entity_type: str, value: str):
        start = 0
        while (i := text.find(f" {name} ", start)) != -1:
            found.append((i, i + len(name) + 1, entity_type, value))
            start = i + 1

    for cat in _distinct(con, "category"):
        add(_norm(cat), "category", cat)
    for name, code in STATE_NAMES.items():
        add(name, "state", code)
    # Two-letter codes only when written in capitals ("SP"), since many are also English words.
    for code in set(re.findall(r"\b[A-Z]{2}\b", question)) & set(STATE_NAMES.values()):
        add(code.lower(), "state", code)
    for city in _distinct(con, "customer_city"):
        if len(name := _norm(city)) >= 5 and name not in CITY_WORDS:
            add(name, "city", city)

    prefer = "state" if re.search(r" states? ", text) else "city"
    found.sort(key=lambda f: (f[1] - f[0], f[2] == prefer, f[2] == "category", -f[0]), reverse=True)
    chosen = []
    for f in found:
        if all(f[1] <= c[0] or f[0] >= c[1] for c in chosen):
            chosen.append(f)
    return chosen


def detect(con: duckdb.DuckDBPyConnection, question: str) -> tuple[str, str] | None:
    """The main category, state or city named in the question (the longest match)."""
    matches = _matches(con, question)
    return (matches[0][2], matches[0][3]) if matches else None


def detect_all(con: duckdb.DuckDBPyConnection, question: str) -> list[tuple[str, str]]:
    """Every category, state and city named in the question, in order of appearance."""
    return list(dict.fromkeys((t, v) for _, _, t, v in sorted(_matches(con, question))))


# --- SQL templates for plain ranking and counting questions --------------------------------------
# "most selling product", "top 5 categories by revenue", "which state has the most orders" are the
# most common simple questions and the easiest for a small model to get wrong, so their SQL is
# built here. Anything with conditions we don't parse goes to the model.

RANK_DIMENSIONS = [  # (pattern, column, label); first match wins
    (r"products?", "product_id", "product_id"),
    (r"categor(y|ies)", "category", "category"),
    (r"sellers?", "seller_id", "seller_id"),
    (r"cit(y|ies)", "customer_city", "city"),
    (r"states?", "customer_state", "state"),
    (r"months?", "order_month", "month"),
]
RANK_MEASURES = [  # (words, metric); first match wins. Averages and shares need MIN_ORDERS per group.
    (r"reviews?|ratings?|scores?|stars?|rated", "avg_review"),
    (r"late", "late_pct"),
    (r"deliver\w*|slowest|fastest|quickest", "avg_delivery_days"),
    (r"average|avg|mean|share|percent\w*|rate", None),  # an average or share we don't template
    (r"revenue|money|earn\w*|made|value", "revenue"),
    (r"customers?", "customers"),
]
COUNT_NOUNS = [(r"orders?", "orders"), (r"customers?|buyers?|clients?", "customers"),
               (r"sellers?|stores?|shops?", "sellers"), (r"products?", "products"), (r"items?|units?", "items")]
COUNT_SQL = {"orders": "COUNT(DISTINCT order_id)", "customers": "COUNT(DISTINCT customer_unique_id)",
             "sellers": "COUNT(DISTINCT seller_id)", "products": "COUNT(DISTINCT product_id)", "items": "COUNT(*)"}
HIGH_WORDS = r"\b(most|best|top|highest|biggest|largest|leading|popular|slowest|longest)\b"
LOW_WORDS = r"\b(least|lowest|worst|fewest|smallest|fastest|quickest|shortest)\b"
TEMPLATE_UNSUPPORTED = (r"\b(freight|shipping|payments?|paid|installments?|per|each|ratio|between|compare|"
                        r"vs|versus|at least|more than|less than|over|under|than|growth|trend|week|day)\b|%")
STATUSES = {"canceled": "canceled", "cancelled": "canceled", "shipped": "shipped",
            "unavailable": "unavailable", "invoiced": "invoiced", "processing": "processing"}


def template_sql(con: duckdb.DuckDBPyConnection, question: str) -> str | None:
    """SQL built in code for plain ranking and counting questions, else None (the model writes it)."""
    q = question.lower()
    if re.search(TEMPLATE_UNSUPPORTED, q):
        return None
    return _count_sql(con, q, question) if re.match(r"\s*how many\b", q) else _ranking_sql(con, q, question)


def _filters(con, q: str, question: str, skip_column: str | None = None) -> list[str]:
    """WHERE conditions for every year and city/state/category the question names
    ("in Sao Paulo and Rio de Janeiro" -> customer_city IN (both), "2017 and 2018" -> both years)."""
    filters = []
    if years := re.findall(r"\b(?:19|20)\d{2}\b", q):
        filters.append(f"YEAR(order_purchase_timestamp) IN ({', '.join(dict.fromkeys(years))})")
    by_column = {}
    for entity_type, value in detect_all(con, question):
        if ENTITIES[entity_type] != skip_column:
            by_column.setdefault(ENTITIES[entity_type], []).append(sql_str(value))
    return filters + [f"{column} IN ({', '.join(values)})" for column, values in by_column.items()]


def _ranking_sql(con, q: str, question: str) -> str | None:
    high, low = re.search(HIGH_WORDS, q), re.search(LOW_WORDS, q)
    dim = next(((col, label, m) for pat, col, label in RANK_DIMENSIONS if (m := re.search(rf"\b{pat}\b", q))), None)
    if bool(high) == bool(low) or dim is None:
        return None
    column, label, match = dim
    found = next(((words, metric) for words, metric in RANK_MEASURES if re.search(rf"\b({words})\b", q)), None)
    if found:
        metric = found[1]
        if metric is None:
            return None
    elif re.search(r"\borders?\b", q) and not re.search(r"\b(sell\w*|sold)\b", q):
        metric = "orders"
    else:  # "best selling", "most sold", "most popular", "top products"
        metric = "items_sold"

    number = re.search(r"\btop\s+(\d+)\b|\b(\d+)\s+(?:\w+\s+){0,2}?" + re.escape(match.group(0)), q)
    limit = int(number.group(1) or number.group(2)) if number else (10 if match.group(0).endswith("s") else 1)
    filters = ([ACTIVE] if metric == "revenue" else []) + _filters(con, q, question, skip_column=column)
    filters.append(f"{column} IS NOT NULL")
    average = METRICS[metric][3] is not None
    having = f" HAVING COUNT(DISTINCT order_id) >= {MIN_ORDERS}" if average else ""
    extra = ", ANY_VALUE(category) AS category" if column == "product_id" else ""
    return (f"SELECT {column} AS {label}{extra}, {METRICS[metric][0]} AS {metric} FROM sales "
            f"WHERE {' AND '.join(filters)} GROUP BY 1{having} "
            f"ORDER BY {metric} {'ASC' if low else 'DESC'} NULLS LAST, 1 LIMIT {limit}")


def _count_sql(con, q: str, question: str) -> str | None:
    words = re.findall(r"\w+", q)[2:5]  # the noun is within 3 words: "how many (unique) customers"
    alias = next((a for w in words for pat, a in COUNT_NOUNS if re.fullmatch(pat, w)), None)
    if alias is None:
        return None
    # Anything beyond a status, "late", a year or a named place/category is left to the model.
    if re.search(r"\b(average|avg|reviews?|ratings?|stars?|revenue|price|month|category|categories|"
                 r"cit(y|ies)|states?|in total per|each|most|top|best)\b", q) and not detect(con, question):
        return None
    measure, filters = COUNT_SQL[alias], _filters(con, q, question)
    status = next((s for w, s in STATUSES.items() if re.search(rf"\b{w}\b", q)), None)
    if re.search(r"\blate\b", q):
        filters.append("delivered_late")
        alias = f"late_{alias}"
    elif status or re.search(r"\bdelivered\b", q):
        status = status or "delivered"
        filters.append(f"order_status = '{status}'")
        alias = f"{status}_{alias}"
    where = f" WHERE {' AND '.join(filters)}" if filters else ""
    return f"SELECT {measure} AS {alias} FROM sales{where}"


# --- Reports ------------------------------------------------------------------------------------
# A report is shaped by the question: its focus (what it's about), an optional period, and extra
# filters ("health_beauty in SP"). Every figure is computed here; the model only interprets the
# facts, so the facts carry the comparisons that make interpretation possible: marketplace
# benchmark, rank among peers, previous period.

FOCUS_METRICS = {
    "sales": ["revenue", "orders", "customers", "avg_order_value", "avg_review", "late_pct"],
    "delivery": ["avg_delivery_days", "late_pct", "days_early", "avg_freight", "avg_review", "orders"],
    "reviews": ["avg_review", "five_star_pct", "one_star_pct", "late_pct", "avg_delivery_days", "orders"],
    "sellers": ["sellers", "revenue_per_seller", "revenue", "orders", "avg_review", "late_pct"],
    "payments": ["revenue", "orders", "avg_order_value", "customers", "avg_review", "late_pct"],
}
RANK_BY = {"sales": "revenue", "delivery": "late_pct", "reviews": "avg_review", "sellers": "revenue",
           "payments": "revenue"}
TREND = {"sales": ("revenue", "Monthly revenue (R$)"), "delivery": ("late_pct", "Late deliveries by month (%)"),
         "reviews": ("avg_review", "Average review by month"), "sellers": ("sellers", "Active sellers by month"),
         "payments": ("revenue", "Monthly revenue (R$)")}
# Where to look inside a subject: the geography one level down, then other dimensions.
BREAKDOWN_DIMS = {"none": ["customer_state", "category"], "category": ["customer_state", "customer_city"],
                  "seller": ["customer_state", "category"], "product": ["customer_state", "customer_city"],
                  "state": ["customer_city", "category"], "city": ["category", "seller_id"]}
DIM_LABEL = {"customer_state": "state", "customer_city": "city", "category": "category", "seller_id": "seller",
             "seller_state": "seller state"}


def where_clause(filters=(), period=None) -> str:
    """Rows of a subject: active orders, the given (entity_type, value) filters, and the period."""
    conds = [ACTIVE] + [f"{ENTITIES[t]} = {sql_str(v)}" for t, v in filters]
    if period:
        conds.append(f"order_purchase_timestamp >= DATE '{period['start']}' "
                     f"AND order_purchase_timestamp < DATE '{period['end']}'")
    return " AND ".join(conds)


def _runner(con):
    """run(purpose, sql) -> DataFrame, recording every query for the "SQL used" panel."""
    queries = []

    def run(purpose: str, sql: str) -> pd.DataFrame:
        df = db.run_query(con, sql)
        queries.append((purpose, " ".join(sql.split()), df))
        return df
    return run, queries


def _metrics(run, purpose: str, names: list[str], where: str) -> pd.Series:
    cols = ", ".join(f"{METRICS[n][0]} AS {n}" for n in names)
    return run(purpose, f"SELECT {cols} FROM sales WHERE {where}").iloc[0]


def build(con: duckdb.DuckDBPyConnection, entity_type: str, value: str | None, focus: str = "sales",
          period: dict | None = None, filters=()):
    """Run the report queries for one subject. Returns (kpis, charts, facts, queries)."""
    from analyst.agent import Chart  # avoid a circular import

    if focus not in FOCUS_METRICS or (focus == "sellers" and entity_type == "seller"):
        focus = "sales"
    subject = [] if entity_type == "none" else [(entity_type, value)]
    scope = subject + list(filters)
    where = where_clause(scope, period)
    run, queries = _runner(con)
    names = FOCUS_METRICS[focus]
    head = _metrics(run, "Headline figures", names, where)

    # Benchmark: the whole marketplace, or "all of SP" when the report is narrowed to SP.
    bench, bench_label = None, None
    if subject:
        bench_label = f"all of {' '.join(str(v) for _, v in filters)}" if filters else "marketplace"
        bench = _metrics(run, f"Same figures, {bench_label}", names, where_clause(filters, period))
    # Previous period: the same months a year earlier, if it has enough orders to compare.
    prev, prev_label, facts = None, None, []
    if period and period.get("prev_start"):
        prev_label = period["prev_label"]
        prev_period = {"start": period["prev_start"], "end": period["prev_end"]}
        prev = _metrics(run, f"Same figures, {prev_label}", list(dict.fromkeys(names + ["orders"])),
                        where_clause(scope, prev_period))
        if pd.isna(prev["orders"]) or prev["orders"] < MIN_ORDERS:
            facts.append(f"{prev_label} had too few orders to compare")
            prev, prev_label = None, None

    kpis = []
    for name in names:
        note = next((f"{label}: {_fmt_metric(name, row[name])}" for row, label in ((bench, bench_label), (prev, prev_label))
                     if row is not None and pd.notna(row[name])), None)
        kpis.append({"label": METRICS[name][1], "value": _fmt_metric(name, head[name]), "note": note})
    facts[:0] = [_metric_fact(name, head[name], bench, bench_label, prev, prev_label) for name in names]
    if subject and focus in ("sales", "payments"):
        total = con.execute(f"SELECT SUM(price) FROM sales WHERE {where_clause(filters, period)}").fetchone()[0]
        if total and pd.notna(head["revenue"]):
            facts.insert(1, f"Share of {bench_label} revenue: {head['revenue'] / total * 100:.1f}%")
    if entity_type in ("category", "city", "state", "seller"):
        facts.append(_rank_fact(run, entity_type, value, focus, filters, period))

    metric, title = TREND[focus]
    monthly = run(title, f"SELECT order_month, {METRICS[metric][0]} AS {metric}, COUNT(DISTINCT order_id) AS orders "
                         f"FROM sales WHERE {where} GROUP BY 1 ORDER BY 1")
    charts = [Chart("line", title, monthly, "order_month", metric, None)]
    facts += _trend_facts(monthly, metric, yoy=period is None)

    used = {ENTITIES[t] for t, _ in scope}
    if "customer_city" in used:
        used.add("customer_state")  # one city is in one state: a by-state chart would be one bar
    _breakdowns(run, focus, entity_type, where, head, used, charts, facts)
    return kpis, charts, facts, queries


def build_compare(con: duckdb.DuckDBPyConnection, subjects: list[dict], focus: str = "sales"):
    """Side-by-side report for 2+ subjects: [{"label", "filters", "period"}]. Returns (kpis, charts, facts, queries)."""
    from analyst.agent import Chart

    focus = focus if focus in FOCUS_METRICS else "sales"
    names = FOCUS_METRICS[focus]
    run, queries = _runner(con)
    for s in subjects:
        s["where"] = where_clause(s["filters"], s.get("period"))
    rows = [_metrics(run, f"Figures for {s['label']}", names, s["where"]) for s in subjects]
    labels = [s["label"] for s in subjects]

    table = pd.DataFrame({"metric": [METRICS[n][1] for n in names],
                          **{label: [_fmt_metric(n, r[n]) for n in names] for label, r in zip(labels, rows)}})
    per_subject = 3 if len(subjects) <= 2 else 1
    kpis = [{"label": f"{label}: {METRICS[n][1]}", "value": _fmt_metric(n, r[n]), "note": None}
            for label, r in zip(labels, rows) for n in names[:per_subject]]

    facts = []
    for n in names:
        values = [(label, r[n]) for label, r in zip(labels, rows) if pd.notna(r[n])]
        if len(values) < 2:
            continue
        text = ", ".join(f"{label} {_fmt_metric(n, v)}" for label, v in values)
        better = METRICS[n][3]
        hi, lo = max(values, key=lambda x: x[1]), min(values, key=lambda x: x[1])
        if better and lo[1] and abs(hi[1] / lo[1] - 1) >= 0.03:
            text += f" (best: {(hi if better == 'high' else lo)[0]})"
        elif better:
            text += " (about the same)"
        elif lo[1]:
            text += f" ({hi[0]} is {hi[1] / lo[1]:.1f}x {lo[0]})"
        facts.append(f"{METRICS[n][1]}: {text}")

    metric, title = TREND[focus]
    trend = pd.concat([run(f"{title}, {s['label']}", f"SELECT order_month, {METRICS[metric][0]} AS {metric} "
                                                    f"FROM sales WHERE {s['where']} GROUP BY 1 ORDER BY 1")
                       .assign(subject=s["label"]) for s in subjects])
    x = "order_month"
    # Comparing periods: put them on the same month axis so the lines overlap.
    if all(s.get("period") for s in subjects) and len({s["period"]["start"][5:] for s in subjects}) == 1:
        trend["month"], x = pd.to_datetime(trend.order_month).dt.strftime("%m (%b)"), "month"
    key = RANK_BY[focus]
    bars = pd.DataFrame({"subject": labels, key: [r[key] for r in rows]})
    charts = [Chart("table", "Side by side", table, None, None, None), Chart("line", title, trend, x, metric, "subject"),
              Chart("bar", f"{METRICS[key][1]} by subject", bars, "subject", key, None)]
    return kpis, charts, facts, queries


def _metric_fact(name, value, bench, bench_label, prev, prev_label) -> str:
    _, label, _, better = METRICS[name]
    # Volume metrics (revenue, orders...) vs the marketplace only restate size; the share fact covers that.
    pairs = [(bench, bench_label)] if better else []
    pairs.append((prev, prev_label))
    parts = []
    for other_row, other_label in pairs:
        other = None if other_row is None else other_row[name]
        if other is None or pd.isna(other) or pd.isna(value):
            continue
        part = f"{other_label} {_fmt_metric(name, other)}"
        if other:
            change = (value / other - 1) * 100
            part += f", {change:+.0f}%"
            if better and abs(change) >= 3:
                part += ", better" if (change > 0) == (better == "high") else ", worse"
        parts.append(part)
    return f"{label}: {_fmt_metric(name, value)}" + (f" (vs {'; vs '.join(parts)})" if parts else "")


def _rank_fact(run, entity_type, value, focus, filters, period) -> str:
    column, key = ENTITIES[entity_type], RANK_BY[focus]
    sql_metric, label, _, better = METRICS[key]
    noun = _plural(DIM_LABEL[column])
    df = run(f"Rank among {noun}",
             f"SELECT {column} AS name, {sql_metric} AS value FROM sales WHERE {where_clause(filters, period)} "
             f"AND {column} IS NOT NULL GROUP BY 1 HAVING COUNT(DISTINCT order_id) >= {MIN_ORDERS} "
             f"ORDER BY value {'ASC' if better == 'low' else 'DESC'} NULLS LAST, name")
    matches = df.index[df.name == value].tolist()
    if not matches:
        return f"Too few orders to rank among {noun} by {label.lower()}"
    return f"Rank by {label.lower()}: #{matches[0] + 1} of {len(df)} {noun} (#1 = {'best' if better else 'highest'})"


def _trend_facts(monthly: pd.DataFrame, metric: str, yoy: bool) -> list[str]:
    # Months with a handful of orders (the 2016 start, Sep-Oct 2018) aren't meaningful highs or lows.
    m = monthly[monthly.orders >= MIN_ORDERS].copy()
    if m.empty:
        m = monthly.copy()
    if m.empty or m[metric].isna().all():
        return []
    m["order_month"] = pd.to_datetime(m.order_month)
    _, label, _, better = METRICS[metric]
    hi, lo = m.loc[m[metric].idxmax()], m.loc[m[metric].idxmin()]
    if better == "low":
        hi, lo = lo, hi
    first, second = ("Best", "worst") if better else ("Peak", "lowest")
    facts = [f"{first} month for {label.lower()}: {hi.order_month:%B %Y} ({_fmt_metric(metric, hi[metric])}); "
             f"{second}: {lo.order_month:%B %Y} ({_fmt_metric(metric, lo[metric])})"]
    if yoy and metric == "revenue":
        # Jan-Aug: the last months with complete data in both 2017 and 2018.
        jan_aug = m[m.order_month.dt.month <= 8]
        by_year = jan_aug.groupby(jan_aug.order_month.dt.year)[metric].sum()
        if 2017 in by_year and 2018 in by_year and by_year[2017] > 0:
            change = (by_year[2018] / by_year[2017] - 1) * 100
            facts.append(f"Revenue Jan-Aug 2018 vs Jan-Aug 2017: {change:+.0f}% "
                         f"({_money(by_year[2017])} -> {_money(by_year[2018])})")
    return facts


def _breakdowns(run, focus: str, entity_type: str, where: str, head, used: set, charts: list, facts: list) -> None:
    """Add the focus's breakdown charts and facts."""
    from analyst.agent import Chart

    # Skip dimensions the report is already filtered on ("health_beauty in SP" by state is one bar).
    candidates = BREAKDOWN_DIMS[entity_type] + ["customer_state", "customer_city", "category", "seller_state", "seller_id"]
    geo, second = [c for c in dict.fromkeys(candidates) if c not in used][:2]

    def top(column: str, metric: str, title: str, kind: str, ascending=False, min_orders=1):
        label = DIM_LABEL[column]
        df = run(title, f"SELECT {column} AS \"{label}\", {METRICS[metric][0]} AS {metric}, "
                        f"COUNT(DISTINCT order_id) AS orders FROM sales WHERE {where} AND {column} IS NOT NULL "
                        f"GROUP BY 1 HAVING COUNT(DISTINCT order_id) >= {min_orders} "
                        f"ORDER BY {metric} {'ASC' if ascending else 'DESC'} NULLS LAST, 1 LIMIT 10")
        if not df.empty:
            charts.append(Chart(kind, title, df, label, metric, None))
            facts.append(f"{title}: " + ", ".join(f"{r[label]} ({_fmt_metric(metric, r[metric])})"
                                                  for _, r in df.head(3).iterrows()))
        return df

    def share_of_revenue(df, n: int, what: str):
        if not df.empty and head["revenue"]:
            facts.append(f"Top {n} {what} hold {df.head(n).revenue.sum() / head['revenue'] * 100:.0f}% of revenue")

    if focus in ("sales", "payments"):
        share_of_revenue(top(geo, "revenue", f"Top {_plural(DIM_LABEL[geo])} by revenue", "bar"),
                         3, _plural(DIM_LABEL[geo]))
    if focus == "sales":
        top(second, "revenue", f"Top {_plural(DIM_LABEL[second])} by revenue", "horizontal_bar")
    elif focus == "delivery":
        top(geo, "late_pct", f"Where deliveries are latest ({DIM_LABEL[geo]})", "horizontal_bar", min_orders=MIN_ORDERS)
        df = run("Delivery time distribution", f"""
            SELECT CASE WHEN delivery_days <= 7 THEN '0-7 days' WHEN delivery_days <= 14 THEN '8-14 days'
                        WHEN delivery_days <= 21 THEN '15-21 days' WHEN delivery_days <= 30 THEN '22-30 days'
                        ELSE 'over 30 days' END AS delivery_time,
                   COUNT(DISTINCT order_id) AS orders, MIN(delivery_days) AS sort_key
            FROM sales WHERE {where} AND delivery_days IS NOT NULL GROUP BY 1 ORDER BY sort_key""")
        if not df.empty:
            charts.append(Chart("bar", "Delivery time distribution", df.drop(columns="sort_key"),
                                "delivery_time", "orders", None))
            slow = df[df.delivery_time.isin(["22-30 days", "over 30 days"])].orders.sum() / df.orders.sum() * 100
            facts.append(f"Orders taking more than 3 weeks: {slow:.0f}%")
    elif focus == "reviews":
        df = run("Review score distribution", f"""
            SELECT CAST(ROUND(review_score) AS INTEGER) AS review_score, COUNT(DISTINCT order_id) AS orders
            FROM sales WHERE {where} AND review_score IS NOT NULL GROUP BY 1 ORDER BY 1""")
        if not df.empty:
            charts.append(Chart("bar", "Orders by review score", df, "review_score", "orders", None))
        dim = second if second != "seller_id" else "category"
        top(dim, "avg_review", f"Lowest-rated {_plural(DIM_LABEL[dim])}", "horizontal_bar",
            ascending=True, min_orders=MIN_ORDERS)
        late = run("Reviews: late vs on-time orders", f"""
            SELECT delivered_late, ROUND(AVG(review_score), 2) AS avg_review FROM sales
            WHERE {where} AND delivered_late IS NOT NULL GROUP BY 1""").set_index("delivered_late").avg_review
        if len(late) == 2:
            facts.append(f"Avg review when delivered on time: {late[False]} / 5; when late: {late[True]} / 5")
    elif focus == "sellers":
        share_of_revenue(top("seller_id", "revenue", "Top sellers by revenue", "horizontal_bar"), 10, "sellers")
        top("seller_state", "sellers", "Where the sellers are", "bar")
    if focus == "payments":
        pay_scope = f"order_id IN (SELECT order_id FROM sales WHERE {where})"
        df = run("Payment methods", f"""
            SELECT payment_type, COUNT(DISTINCT order_id) AS orders, ROUND(SUM(payment_value), 2) AS amount
            FROM payments WHERE {pay_scope} GROUP BY 1 ORDER BY orders DESC""")
        if not df.empty:
            charts.append(Chart("pie", "Payment methods (orders)", df, "payment_type", "orders", None))
            facts.append("Payment methods: " + ", ".join(
                f"{r.payment_type} {r.orders / df.orders.sum() * 100:.0f}%" for _, r in df.iterrows()))
        df = run("Credit card installments", f"""
            SELECT payment_installments AS installments, COUNT(DISTINCT order_id) AS orders FROM payments
            WHERE {pay_scope} AND payment_type = 'credit_card' GROUP BY 1 ORDER BY 1""")
        if not df.empty:
            charts.append(Chart("bar", "Credit card installments", df.head(12), "installments", "orders", None))
            avg = (df.installments * df.orders).sum() / df.orders.sum()
            one = df.loc[df.installments <= 1, "orders"].sum() / df.orders.sum() * 100
            facts.append(f"Credit card: average {avg:.1f} installments; {one:.0f}% pay in one installment")


def _plural(noun: str) -> str:
    return noun[:-1] + "ies" if noun.endswith("y") else noun + "s"


def _fmt_metric(name: str, value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    fmt = METRICS[name][2]
    if fmt == "money":
        return _money(value)
    return {"int": f"{int(value):,}", "pct": f"{value:.1f}%", "days": f"{value:.1f} days"}.get(fmt, f"{value:.2f} / 5")


def _money(x: float) -> str:
    if x >= 1e6:
        return f"R$ {x / 1e6:.2f}M"
    if x >= 1e3:
        return f"R$ {x / 1e3:.1f}K"
    return f"R$ {x:,.2f}"
