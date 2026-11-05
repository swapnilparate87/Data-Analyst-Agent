"""Prewritten report queries for an entity (category, city, state, seller, product or the
whole marketplace), plus matching user-typed names to real values in the data.

Small local models write unreliable SQL, so reports never depend on model-written queries:
the numbers, KPIs and key facts here are all computed in code.
"""

import difflib
import re
import unicodedata

import duckdb
import pandas as pd

from analyst import db

ACTIVE = "order_status NOT IN ('canceled', 'unavailable')"

# entity_type -> (column in `sales`, display name)
ENTITIES = {
    "category": ("category", "category"),
    "city": ("customer_city", "city"),
    "state": ("customer_state", "state"),
    "seller": ("seller_id", "seller"),
    "product": ("product_id", "product"),
}

STATE_NAMES = {
    "acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM", "bahia": "BA", "ceara": "CE",
    "distrito federal": "DF", "espirito santo": "ES", "goias": "GO", "maranhao": "MA",
    "mato grosso": "MT", "mato grosso do sul": "MS", "minas gerais": "MG", "para": "PA",
    "paraiba": "PB", "parana": "PR", "pernambuco": "PE", "piaui": "PI", "rio de janeiro": "RJ",
    "rio grande do norte": "RN", "rio grande do sul": "RS", "rondonia": "RO", "roraima": "RR",
    "santa catarina": "SC", "sao paulo": "SP", "sergipe": "SE", "tocantins": "TO",
}

# Per entity type: the two breakdowns shown under the trend chart.
# ("dim", column, label, chart title, chart type) or ("reviews",)
BREAKDOWNS = {
    "category": [("dim", "customer_state", "state", "Revenue by customer state", "bar"),
                 ("dim", "customer_city", "city", "Top customer cities", "horizontal_bar")],
    "city": [("dim", "category", "category", "Top categories", "horizontal_bar"), ("reviews",)],
    "state": [("dim", "customer_city", "city", "Top cities", "horizontal_bar"),
              ("dim", "category", "category", "Top categories", "horizontal_bar")],
    "seller": [("dim", "category", "category", "Revenue by category", "horizontal_bar"),
               ("dim", "customer_state", "state", "Where customers are", "bar")],
    "product": [("dim", "customer_state", "state", "Where customers are", "bar"), ("reviews",)],
    "none": [("dim", "category", "category", "Top categories", "horizontal_bar"),
             ("dim", "customer_state", "state", "Revenue by customer state", "bar")],
}


# Real city names that are also common words in questions ("SP sales" is not the city of Sales).
CITY_WORDS = {"sales", "central"}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def resolve(con: duckdb.DuckDBPyConnection, entity_type: str, value: str) -> tuple[str | None, list[str]]:
    """Match a user-typed name to a real value. Returns (match, suggestions)."""
    column, _ = ENTITIES[entity_type]
    target = _norm(value)
    if not target:
        return None, []
    if entity_type == "state":
        target = _norm(STATE_NAMES.get(target, target))
    values = [v for (v,) in con.execute(f"SELECT DISTINCT {column} FROM sales WHERE {column} IS NOT NULL").fetchall()]
    by_norm = {_norm(v): v for v in values}
    if target in by_norm:
        return by_norm[target], []
    close = difflib.get_close_matches(target, list(by_norm), n=3, cutoff=0.8)
    if close:
        return by_norm[close[0]], []
    containing = sorted((n for n in by_norm if target in n), key=len)
    if containing:
        return by_norm[containing[0]], []
    suggestions = difflib.get_close_matches(target, list(by_norm), n=3, cutoff=0.5)
    return None, [by_norm[s] for s in suggestions]


def fix_literal(sql: str, value: str) -> str:
    """Rewrite string literals that spell `value` differently ('Curitiba', 'São Paulo') to the stored value."""
    target = _norm(value)
    return re.sub(r"'([^']*)'", lambda m: f"'{value}'" if _norm(m.group(1)) == target else m.group(0), sql)


def detect(con: duckdb.DuckDBPyConnection, question: str) -> tuple[str, str] | None:
    """Find a category, state or city named in the question text. Longest match wins."""
    text = f" {_norm(question)} "
    found = []  # (match length, entity_type, value)
    for (cat,) in con.execute("SELECT DISTINCT category FROM sales WHERE category IS NOT NULL").fetchall():
        if f" {_norm(cat)} " in text:
            found.append((len(cat), "category", cat))
    for name, code in STATE_NAMES.items():
        if f" {name} " in text:
            found.append((len(name), "state", code))
    # Two-letter codes only when written in capitals ("SP"), since many are also English words.
    for code in set(re.findall(r"\b[A-Z]{2}\b", question)) & set(STATE_NAMES.values()):
        found.append((2, "state", code))
    for (city,) in con.execute("SELECT DISTINCT customer_city FROM sales").fetchall():
        name = _norm(city)
        if len(name) >= 5 and name not in CITY_WORDS and f" {name} " in text:
            found.append((len(name), "city", city))
    if not found:
        return None
    # "rio de janeiro" / "sao paulo" are both a city and a state: city unless "state" is said.
    prefer = "state" if " state " in text else "city"
    found.sort(key=lambda f: (f[0], f[1] == prefer, f[1] == "category"), reverse=True)
    return found[0][1], found[0][2]


# Ranking questions ("most selling product", "top 5 categories by revenue", "which state has the
# most orders") are the most common simple questions and the easiest for a small model to get
# wrong, so their SQL is built here. Anything with conditions we don't parse goes to the model.
RANK_DIMENSIONS = [  # (pattern, column, label); first match wins
    (r"products?", "product_id", "product_id"),
    (r"categor(y|ies)", "category", "category"),
    (r"sellers?", "seller_id", "seller_id"),
    (r"cit(y|ies)", "customer_city", "city"),
    (r"states?", "customer_state", "state"),
    (r"months?", "order_month", "month"),
]
HIGH_WORDS = r"\b(most|best|top|highest|biggest|largest|leading|popular|slowest|longest)\b"
LOW_WORDS = r"\b(least|lowest|worst|fewest|smallest|fastest|quickest|shortest)\b"
# Wording we don't translate (other measures, extra conditions): leave those to the model.
TEMPLATE_UNSUPPORTED = (r"\b(freight|shipping|payments?|paid|installments?|per|each|ratio|between|compare|"
                        r"vs|versus|at least|more than|less than|over|under|than|growth|trend|week|day)\b")
# Averages and shares over a handful of orders are noise ("2 orders, avg review 1.0").
MIN_ORDERS = 30
LATE_PCT = ("ROUND(100.0 * COUNT(DISTINCT order_id) FILTER (WHERE delivered_late) / NULLIF(COUNT(DISTINCT "
            "order_id) FILTER (WHERE order_delivered_customer_date IS NOT NULL), 0), 1)")
STATUSES = {"canceled": "canceled", "cancelled": "canceled", "shipped": "shipped",
            "unavailable": "unavailable", "invoiced": "invoiced", "processing": "processing"}


def template_sql(con: duckdb.DuckDBPyConnection, question: str) -> str | None:
    """SQL built in code for plain ranking and counting questions, else None (the model writes it)."""
    q = question.lower()
    if re.search(TEMPLATE_UNSUPPORTED, q) or "%" in q:
        return None
    return _count_sql(con, q, question) if re.match(r"\s*how many\b", q) else _ranking_sql(con, q, question)


def _filters(con, q: str, question: str, skip_column: str | None = None) -> list[str]:
    """WHERE conditions for the year and the city/state/category the question names."""
    filters = []
    if year := re.search(r"\b(19|20)\d{2}\b", q):
        filters.append(f"YEAR(order_purchase_timestamp) = {year.group(0)}")
    detected = detect(con, question)
    if detected and ENTITIES[detected[0]][0] != skip_column:
        filters.append(f"{ENTITIES[detected[0]][0]} = '{detected[1]}'")
    return filters


def _ranking_sql(con, q: str, question: str) -> str | None:
    high, low = re.search(HIGH_WORDS, q), re.search(LOW_WORDS, q)
    if not (high or low) or (high and low):
        return None
    dim = next(((col, label, m) for pat, col, label in RANK_DIMENSIONS
                if (m := re.search(rf"\b{pat}\b", q))), None)
    if dim is None:
        return None
    column, label, match = dim

    average = False
    if re.search(r"\b(reviews?|ratings?|scores?|stars?|rated)\b", q):
        measure, alias, average = "ROUND(AVG(review_score), 2)", "avg_review", True
    elif re.search(r"\blate\b", q):
        measure, alias, average = LATE_PCT, "late_delivery_pct", True
    elif re.search(r"\b(deliver\w*|slowest|fastest|quickest)\b", q):
        measure, alias, average = "ROUND(AVG(delivery_days), 1)", "avg_delivery_days", True
    elif re.search(r"\b(average|avg|mean|share|percent\w*|rate)\b", q):
        return None  # an average or share of something we don't template
    elif re.search(r"\b(revenue|money|earn\w*|made|value)\b", q):
        measure, alias = "ROUND(SUM(price), 2)", "revenue"
    elif re.search(r"\bcustomers?\b", q):
        measure, alias = "COUNT(DISTINCT customer_unique_id)", "customers"
    elif re.search(r"\borders?\b", q) and not re.search(r"\b(sell\w*|sold)\b", q):
        measure, alias = "COUNT(DISTINCT order_id)", "orders"
    else:  # "best selling", "most sold", "most popular", "top products"
        measure, alias = "COUNT(*)", "items_sold"

    number = re.search(r"\btop\s+(\d+)\b|\b(\d+)\s+(?:\w+\s+){0,2}?" + re.escape(match.group(0)), q)
    plural = match.group(0).endswith(("s", "ies"))
    limit = int(number.group(1) or number.group(2)) if number else (10 if plural else 1)

    filters = _filters(con, q, question, skip_column=column)
    if alias == "revenue":
        filters.insert(0, ACTIVE)
    filters.append(f"{column} IS NOT NULL")
    having = f" HAVING COUNT(DISTINCT order_id) >= {MIN_ORDERS}" if average else ""
    extra = ", ANY_VALUE(category) AS category" if column == "product_id" else ""
    return (f"SELECT {column} AS {label}{extra}, {measure} AS {alias} FROM sales "
            f"WHERE {' AND '.join(filters)} GROUP BY 1{having} "
            f"ORDER BY {alias} {'ASC' if low else 'DESC'} NULLS LAST LIMIT {limit}")


def _count_sql(con, q: str, question: str) -> str | None:
    counts = [(r"orders?", "COUNT(DISTINCT order_id)", "orders"),
              (r"customers?|buyers?|clients?", "COUNT(DISTINCT customer_unique_id)", "customers"),
              (r"sellers?|stores?|shops?", "COUNT(DISTINCT seller_id)", "sellers"),
              (r"products?", "COUNT(DISTINCT product_id)", "products"),
              (r"items?|units?", "COUNT(*)", "items")]
    words = re.findall(r"\w+", q)[2:5]  # the noun is within 3 words: "how many (unique) customers"
    found = next(((m, a) for w in words for pat, m, a in counts if re.fullmatch(pat, w)), None)
    if found is None:
        return None
    # Anything beyond a status, "late", a year or a named place/category is left to the model.
    if re.search(r"\b(average|avg|reviews?|ratings?|stars?|revenue|price|month|category|categories|"
                 r"cit(y|ies)|states?|in total per|each|most|top|best)\b", q) and not detect(con, question):
        return None
    measure, alias = found
    filters = _filters(con, q, question)
    if re.search(r"\blate\b", q):
        filters.append("delivered_late")
        alias = f"late_{alias}"
    elif status := next((s for w, s in STATUSES.items() if re.search(rf"\b{w}\b", q)), None):
        filters.append(f"order_status = '{status}'")
        alias = f"{status}_{alias}"
    elif re.search(r"\bdelivered\b", q):
        filters.append("order_status = 'delivered'")
        alias = f"delivered_{alias}"
    where = f" WHERE {' AND '.join(filters)}" if filters else ""
    return f"SELECT {measure} AS {alias} FROM sales{where}"


def build(con: duckdb.DuckDBPyConnection, entity_type: str, value: str | None):
    """Run the report queries. Returns (kpis, charts, facts, queries)."""
    from analyst.agent import Chart  # avoid a circular import

    if entity_type == "none":
        where, params = ACTIVE, []
    else:
        where, params = f"{ACTIVE} AND {ENTITIES[entity_type][0]} = ?", [value]
    queries = []

    def run(purpose: str, sql: str) -> pd.DataFrame:
        df = db.run_query(con, sql, params)
        queries.append((purpose, sql.replace("?", f"'{value}'") if params else sql, df))
        return df

    head = run("Headline totals", f"""
        SELECT ROUND(SUM(price), 2) AS revenue,
               COUNT(DISTINCT order_id) AS orders,
               COUNT(DISTINCT customer_unique_id) AS customers,
               ROUND(SUM(price) / COUNT(DISTINCT order_id), 2) AS avg_order_value,
               ROUND(AVG(review_score), 2) AS avg_review,
               ROUND(AVG(delivery_days), 1) AS avg_delivery_days,
               ROUND(100.0 * COUNT(DISTINCT order_id) FILTER (WHERE delivered_late)
                     / NULLIF(COUNT(DISTINCT order_id) FILTER (WHERE order_delivered_customer_date IS NOT NULL), 0), 1)
                   AS late_delivery_pct,
               ROUND(100.0 * SUM(price) / (SELECT SUM(price) FROM sales WHERE {ACTIVE}), 2) AS revenue_share_pct
        FROM sales WHERE {where}""").iloc[0]

    monthly = run("Monthly revenue and orders", f"""
        SELECT order_month, ROUND(SUM(price), 2) AS revenue, COUNT(DISTINCT order_id) AS orders
        FROM sales WHERE {where} GROUP BY 1 ORDER BY 1""")

    kpis = [
        {"label": "Revenue", "value": _money(head.revenue),
         "note": None if entity_type == "none" else f"{head.revenue_share_pct}% of marketplace"},
        {"label": "Orders", "value": f"{int(head.orders):,}", "note": f"{int(head.customers):,} customers"},
        {"label": "Avg order value", "value": _money(head.avg_order_value), "note": None},
        {"label": "Avg review", "value": f"{head.avg_review} / 5", "note": None},
        {"label": "Avg delivery time", "value": f"{head.avg_delivery_days} days", "note": None},
        {"label": "Late deliveries", "value": f"{head.late_delivery_pct}%", "note": "of delivered orders"},
    ]
    charts = [Chart("line", "Monthly revenue (R$)", monthly, "order_month", "revenue", None)]
    facts = [f"{k['label']}: {k['value']}" + (f" ({k['note']})" if k["note"] else "") for k in kpis]
    facts += _trend_facts(monthly)

    for spec in BREAKDOWNS[entity_type]:
        if spec[0] == "reviews":
            df = run("Review score distribution", f"""
                SELECT CAST(ROUND(review_score) AS INTEGER) AS review_score, COUNT(DISTINCT order_id) AS orders
                FROM sales WHERE {where} AND review_score IS NOT NULL GROUP BY 1 ORDER BY 1""")
            if not df.empty:
                charts.append(Chart("bar", "Orders by review score", df, "review_score", "orders", None))
                five, one = (df.loc[df.review_score == s, "orders"].sum() / df.orders.sum() * 100 for s in (5, 1))
                facts.append(f"{five:.0f}% of reviewed orders gave 5 stars, {one:.0f}% gave 1 star")
            continue
        _, column, label, title, kind = spec
        df = run(title, f"""
            SELECT {column} AS {label}, ROUND(SUM(price), 2) AS revenue, COUNT(DISTINCT order_id) AS orders
            FROM sales WHERE {where} AND {column} IS NOT NULL
            GROUP BY 1 ORDER BY revenue DESC LIMIT 10""")
        if df.empty:
            continue
        charts.append(Chart(kind, title, df, label, "revenue", None))
        top = df.head(3)
        share = lambda r: f"{r / head.revenue * 100:.0f}%"  # noqa: E731
        plural = "categories" if label == "category" else f"{label}s"
        facts.append(f"Top {plural} by revenue: " + ", ".join(
            f"{row[label]} ({_money(row.revenue)}, {share(row.revenue)})" for _, row in top.iterrows()))
    return kpis, charts, facts, queries


def _trend_facts(monthly: pd.DataFrame) -> list[str]:
    if monthly.empty:
        return []
    m = monthly.assign(order_month=pd.to_datetime(monthly.order_month))
    peak = m.loc[m.revenue.idxmax()]
    facts = [f"Peak month: {peak.order_month:%B %Y} ({_money(peak.revenue)}, {peak.orders:,} orders)"]
    # Year-over-year on Jan-Aug, the last months with complete data in both 2017 and 2018.
    jan_aug = m[m.order_month.dt.month <= 8]
    by_year = jan_aug.groupby(jan_aug.order_month.dt.year).revenue.sum()
    if 2017 in by_year and 2018 in by_year and by_year[2017] > 0:
        change = (by_year[2018] / by_year[2017] - 1) * 100
        facts.append(f"Revenue Jan-Aug 2018 vs Jan-Aug 2017: {change:+.0f}% "
                     f"({_money(by_year[2017])} -> {_money(by_year[2018])})")
    return facts


def _money(x: float) -> str:
    if x >= 1e6:
        return f"R$ {x / 1e6:.2f}M"
    if x >= 1e3:
        return f"R$ {x / 1e3:.1f}K"
    return f"R$ {x:,.2f}"
