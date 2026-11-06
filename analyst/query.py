"""Query analyser: works out what the user is asking before anything else runs.

For each message it:
  - splits it into separate questions ("How many orders were late, and how is RJ doing?")
  - finds every city, state and category named, the time period, and the topic (focus)
  - decides where each question goes: out_of_scope, not_in_data, compare, report, direct, or
    "unsure" (left to the model's router)
  - carries context: a split-off fragment borrows the place/period of its sibling ("How many orders
    and how much revenue came from Curitiba?"), and a follow-up borrows from the previous turn
    ("and in 2017?")

It is pure code (regular expressions and lookups), so it is fast and predictable, and can be
tested on its own (evals/datasets/query_analyzer.json).
"""

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import duckdb

from analyst import reports

# Patterns shared with the agent's scope check.
DATA_WORDS = re.compile(
    r"\b(orders?|ordered|revenue|sales|sold|sell(ers?|ing)?|buy(ers?|ing)?|bought|purchases?|customers?|"
    r"categor(y|ies)|products?|items?|deliver(y|ies|ed)|late|shipping|freight|reviews?|ratings?|"
    r"payments?|paid|installments?|boleto|credit card|vouchers?|prices?|olist|marketplace|"
    r"months?|monthly)\b",
    re.IGNORECASE,
)
MISSING_DATA = re.compile(
    r"\b(profits?|profitab\w*|margins?|net income|expenses?|marketing|advertis\w*|inventory|stock levels?|"
    r"employees?|staff|salar(y|ies)|competitors?|forecast\w*|predict\w*|next (month|year))\b"
    r"|\bnames?\b.*\bproducts?\b|\bproducts?\b.*\bnames?\b",
    re.IGNORECASE,
)
REPORT_WORDS = re.compile(
    r"\b(report|overview|summary|summari[sz]e|analy[sz]e|analysis|tell me about|perform\w*|doing|dashboard|"
    r"insights?|deep dive|breakdown of)\b",
    re.IGNORECASE,
)
DIRECT_START = re.compile(
    r"^\s*(how (many|much)|what('s| is| was| are| were)|which|who|when|list|count|top \d+)\b", re.IGNORECASE
)
COMPARE_WORDS = re.compile(r"\b(compar\w*|vs\.?|versus|against|difference between|differ)\b", re.IGNORECASE)
FOLLOW_UP = re.compile(r"^\s*(and|what about|how about|same|now|only|just|also|vs|versus)\b", re.IGNORECASE)

FOCUS_WORDS = {  # checked in this order; the first topic mentioned in the question wins
    "delivery": r"deliver\w*|shipping|shipped|late|lateness|delay\w*|on[- ]time|freight|logistics",
    "reviews": r"reviews?|ratings?|rated|stars?|satisf\w*|complain\w*|feedback|scores?",
    "sellers": r"sellers?|vendors?|merchants?|stores?|shops?",
    "payments": r"payments?|pay|paid|installments?|boleto|credit cards?|debit|vouchers?",
    "sales": r"sales|revenue|sold|orders?|growth|grow\w*",
}

# Questions inside one message are split at these joiners when a new question starts after them.
STARTERS = (r"what|which|how|who|when|where|why|is|are|was|were|do|does|did|can|could|give|show|tell|list|"
            r"compare|make|create|build|generate|prepare|analy[sz]e|summari[sz]e|find|get")

# The data: Sep 2016 - Oct 2018, but 2016 has almost no orders (Oct only) and Sep-Oct 2018 a handful.
DATA_START, DATA_END = date(2017, 1, 1), date(2018, 9, 1)
DATA_COVERAGE = "Sep 2016 to Aug 2018 (with very few orders before 2017)"
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]


@dataclass
class Part:
    text: str                      # the question as asked (one part of the message)
    intent: str                    # out_of_scope | not_in_data | compare | report | direct | unsure
    entities: list = field(default_factory=list)   # [(entity_type, value)] in order of mention
    focus: str = "sales"
    periods: list = field(default_factory=list)    # [period dict], see make_period
    query: str = ""                # text used for SQL and templates: the question plus borrowed context
    follow_up: bool = False
    sentence: int = 0              # parts split from the same sentence can share context
    sql: str = ""                  # draft SQL, when the model's router handled the question

    @property
    def period(self):
        return self.periods[0] if self.periods else None


def analyze(con: duckdb.DuckDBPyConnection, message: str, previous: list | None = None) -> list[Part]:
    """Split a message into questions and work out each one's intent, entities, focus and period."""
    pieces = split(message)
    parts = [_analyze_one(con, text) for text, _ in pieces]
    for i, part in enumerate(parts):
        part.sentence = pieces[i][1]
    _share_context(parts)
    if previous and len(parts) == 1:
        _follow_up(parts[0], previous[-1])
    for p in parts:
        if not p.query:
            p.query = p.text
    return parts


def split(message: str) -> list[tuple[str, int]]:
    """(question, sentence number) per question. Splits at '?', sentence ends, new lines, ';',
    numbered items, and at 'and' / 'also' / 'then' when a new question starts right after
    ("..., and how is RJ doing?")."""
    text = re.sub(r"(^|\s)\(?\d{1,2}[.)]\s+", "\n", message.strip())  # 1-2 digits: "in 2017. " is a year, not item 2017
    sentences = [p for p in re.split(r"(?<=[?!])\s+|(?<=[a-z0-9]\.)\s+(?=[A-Z])|\n+|;\s*", text) if p.strip()]
    out = []
    for n, sentence in enumerate(sentences):
        if n:  # "... Also, which ...": the joiner isn't part of the question (but "and in 2017?" alone is a follow-up)
            sentence = re.sub(r"^(also|and|then|plus)\b,?\s*", "", sentence.strip(), flags=re.IGNORECASE)
        for piece in re.split(rf"\s*,?\s*\b(?:and also|and then|and|also|then|plus)\b,?\s+(?=(?:{STARTERS})\b)",
                              sentence, flags=re.IGNORECASE):
            piece = piece.strip(" ,.")
            if len(piece.split()) >= 2:
                out.append((piece, n))
    return out or [(message.strip(), 0)]


def make_period(start: date, end: date, label: str) -> dict | None:
    """A period clipped to the data, with the same months a year earlier to compare with."""
    asked = (start, end)
    start, end = max(start, date(2016, 9, 1)), min(end, DATA_END)
    if start >= end:
        return {"label": label, "start": str(asked[0]), "end": str(asked[0]), "empty": True}
    if (start, end) != asked:
        last_month = end - timedelta(days=1)
        label = f"{label} ({start:%b}-{last_month:%b} only)"
    period = {"label": label, "start": str(start), "end": str(end)}
    prev_start, prev_end = start.replace(year=start.year - 1), end.replace(year=end.year - 1)
    if prev_start >= DATA_START:
        last = prev_end - timedelta(days=1)
        if (start.month, start.day, end.month, end.day) == (1, 1, 1, 1):
            prev_label = str(prev_start.year)  # a whole year
        elif (prev_start.year, prev_start.month) == (last.year, last.month):
            prev_label = f"{prev_start:%B %Y}"  # one month
        else:
            prev_label = f"{prev_start.year} ({prev_start:%b}-{last:%b})"
        period.update(prev_label=prev_label, prev_start=str(prev_start), prev_end=str(prev_end))
    return period


def find_periods(text: str) -> list[dict]:
    """Years, quarters and months named in the text, as periods."""
    q = text.lower()
    periods, used = [], set()
    for m in re.finditer(r"\bq([1-4])\s*(?:of\s*)?(20\d\d)\b|\b(20\d\d)\s*q([1-4])\b", q):
        quarter, year = int(m.group(1) or m.group(4)), int(m.group(2) or m.group(3))
        start = date(year, 3 * quarter - 2, 1)
        end = date(year + (quarter == 4), (3 * quarter) % 12 + 1, 1)
        periods.append(make_period(start, end, f"Q{quarter} {year}"))
        used.add(year)
    for m in re.finditer(rf"\b({'|'.join(MONTHS)}|{'|'.join(x[:3] for x in MONTHS)})\.?\s+(20\d\d)\b", q):
        month = next(i for i, name in enumerate(MONTHS, 1) if name.startswith(m.group(1)[:3]))
        year = int(m.group(2))
        end = date(year + (month == 12), month % 12 + 1, 1)
        periods.append(make_period(date(year, month, 1), end, f"{MONTHS[month - 1].title()} {year}"))
        used.add(year)
    for m in re.finditer(r"\b(20\d\d)\b", q):
        year = int(m.group(1))
        if year not in used:
            periods.append(make_period(date(year, 1, 1), date(year + 1, 1, 1), str(year)))
            used.add(year)
    return periods


def align_years(periods: list[dict]) -> list[dict]:
    """Compare years on the months both have: 2017 vs 2018 becomes Jan-Aug 2017 vs Jan-Aug 2018,
    because the data stops in August 2018."""
    starts = [date.fromisoformat(p["start"]) for p in periods]
    ends = [date.fromisoformat(p["end"]) for p in periods]
    if len(periods) < 2 or not all(s.month == 1 and s.day == 1 and e.year - s.year <= 1 for s, e in zip(starts, ends)):
        return periods
    first = max(s.month for s in starts)
    last = min((e.month - 1 if e.year > s.year or e.month > 1 else 12) or 12 for s, e in zip(starts, ends))
    if (first, last) == (1, 12):
        return periods
    aligned = []
    for p, s in zip(periods, starts):
        start, end = date(s.year, first, 1), date(s.year + (last == 12), last % 12 + 1, 1)
        if (str(start), str(end)) == (p["start"], p["end"]):
            aligned.append(p)  # already those months ("Q1 2017")
        else:
            aligned.append({"label": f"{s.year} ({start:%b}-{date(2000, last, 1):%b})", "start": str(start), "end": str(end)})
    return aligned


def find_focus(text: str) -> str:
    """The topic mentioned first: delivery, reviews, sellers, payments, or sales (the default)."""
    first = None
    for focus, pattern in FOCUS_WORDS.items():
        m = re.search(rf"\b({pattern})\b", text, re.IGNORECASE)
        if m and (first is None or m.start() < first[1]):
            first = (focus, m.start())
    return first[0] if first else "sales"


def _analyze_one(con, text: str) -> Part:
    entities = reports.detect_all(con, text)
    periods = find_periods(text)
    part = Part(text=text, intent="unsure", entities=entities, focus=find_focus(text), periods=periods)
    data_words = bool(DATA_WORDS.search(text))
    report_words = bool(REPORT_WORDS.search(text))
    compare_words = bool(COMPARE_WORDS.search(text))
    part.follow_up = bool(FOLLOW_UP.search(text))

    if not (data_words or entities or report_words or part.follow_up or (compare_words and periods)):
        part.intent = "out_of_scope"
    elif MISSING_DATA.search(text):
        part.intent = "not_in_data"
    elif compare_words and (_comparable(entities) or len(periods) >= 2):
        part.intent = "compare"
    elif reports.template_sql(con, text) or DIRECT_START.search(text):
        part.intent = "direct"
    elif report_words and not (data_words or entities or periods):
        part.intent = "unsure"  # "summarize the history of Rome": let the model judge the scope
    elif report_words or (entities and not re.search(r"\b(how many|how much|average|avg|total|count)\b", text, re.I)):
        part.intent = "compare" if _comparable(entities) else "report"  # "SP in RJ" can't be one report
    return part


def _comparable(entities) -> bool:
    types = [t for t, _ in entities]
    return any(types.count(t) >= 2 for t in set(types))


def _share_context(parts: list[Part]) -> None:
    """'How many orders and how much revenue came from Curitiba?' -> both parts are about Curitiba.
    Only a short fragment ("How many orders") borrows, and only from its own sentence."""
    for i, p in enumerate(parts):
        if p.intent in ("out_of_scope", "not_in_data") or len(p.text.split()) > 4:
            continue
        siblings = [s for s in parts[i + 1:] + parts[:i][::-1] if s.sentence == p.sentence]
        donor = next((s for s in siblings if s.entities or s.periods), None)
        if donor is None:
            continue
        extra = []
        if not p.entities and donor.entities:
            p.entities = list(donor.entities)
            extra += [f"{t} {v}" for t, v in donor.entities]
        if not p.periods and donor.periods:
            p.periods = list(donor.periods)
            extra += [donor.periods[0]["label"]]
        if extra:
            p.query = f"{p.text} ({', '.join(extra)})"
            if p.intent == "unsure" and p.entities:
                p.intent = "direct" if DIRECT_START.search(p.text) else p.intent


def _follow_up(part: Part, prev: Part) -> None:
    """'and in 2017?' / 'what about RJ?' / 'same for delivery': reuse the previous question."""
    short = len(part.text.split()) <= 6
    if not (part.follow_up or (short and part.intent == "unsure")) or prev.intent in ("out_of_scope", "not_in_data"):
        return
    new_focus = find_focus(part.text) if any(re.search(rf"\b({p})\b", part.text, re.I)
                                             for p in FOCUS_WORDS.values()) else None
    if prev.intent in ("report", "compare"):
        part.intent = prev.intent if part.intent in ("unsure", "report", "out_of_scope") else part.intent
        if part.intent == prev.intent:
            if not part.entities:
                part.entities = list(prev.entities)
            elif prev.intent == "report" and prev.entities and part.entities[0][0] != prev.entities[0][0]:
                # "what about in SP?" after a category report: keep the category, add SP as a filter
                part.entities = list(prev.entities[:1]) + part.entities
            part.periods = part.periods or list(prev.periods)
            part.focus = new_focus or prev.focus
    elif prev.intent == "direct" and part.intent in ("unsure", "out_of_scope", "direct", "report") \
            and not REPORT_WORDS.search(part.text):
        if part.entities:
            part.intent = "unsure"  # a new place for a direct question: the model rewrites it with history
        else:
            part.intent = "direct"
            part.query = f"{prev.query} ({part.text})"
            part.entities = list(prev.entities)
            part.periods = part.periods or list(prev.periods)
