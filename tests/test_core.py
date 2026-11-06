"""Fast checks of the code-only logic (no model). Run: python -m pytest tests"""

import pytest

from analyst import db, query, reports
from analyst.agent import Analyst


@pytest.fixture(scope="module")
def con():
    return db.connect()


def test_split_keeps_year_before_full_stop():
    parts = [text for text, _ in query.split("How many orders were there in 2017. How many orders in 2018?")]
    assert parts == ["How many orders were there in 2017", "How many orders in 2018?"]


def test_split_numbered_items():
    parts = [text for text, _ in query.split("1. Which is the most selling product? 2. Give me a report on Curitiba")]
    assert len(parts) == 2


def test_align_years_uses_months_both_have():
    p2017, p2018 = query.find_periods("2017 vs 2018")
    a, b = query.align_years([p2017, p2018])
    assert (a["start"], a["end"]) == ("2017-01-01", "2017-09-01")
    assert (b["start"], b["end"]) == ("2018-01-01", "2018-09-01")


def test_template_filters_on_every_place(con):
    sql = reports.template_sql(con, "How many late orders in Sao Paulo and Rio de Janeiro?")
    assert "'sao paulo'" in sql and "'rio de janeiro'" in sql
    both = db.run_query(con, sql).iat[0, 0]
    one = db.run_query(con, reports.template_sql(con, "How many late orders in Rio de Janeiro?")).iat[0, 0]
    assert both > one


def test_city_with_apostrophe(con):
    sql = reports.template_sql(con, "How many orders from santa barbara d'oeste?")
    assert db.run_query(con, sql).iat[0, 0] > 0


def test_years_all_filtered(con):
    sql = reports.template_sql(con, "How many orders in 2017 and 2018?")
    assert "IN (2017, 2018)" in sql


@pytest.fixture(scope="module")
def analyst(con):
    a = Analyst(con)
    a._chat = lambda *args, **kw: "(narrative)"  # no model: these tests check routing and the queries
    return a


@pytest.mark.parametrize("previous, message, subjects", [
    ("Compare SP and RJ", "what about MG?", 0),  # one place left: a report, not a crash
    (None, "How is SP doing compared to RJ?", 2),
    (None, "Compare SP and RJ in 2017 and 2018", 4),
])
def test_comparisons(analyst, previous, message, subjects):
    prev = query.analyze(analyst.con, previous) if previous else None
    report = analyst.run(query.analyze(analyst.con, message, prev)[0])
    assert report.charts and report.facts
    if subjects:
        assert report.text.startswith("### Comparison") and report.text.count(" vs ") == subjects - 1
    else:
        assert report.text.startswith("### Report on the state **MG**")


def test_category_in_city_has_no_state_breakdown(con):
    _, charts, _, _ = reports.build(con, "category", "health_beauty", filters=[("city", "curitiba")])
    assert "Top states by revenue" not in [c.title for c in charts]  # customer state: one bar (PR)
