"""Evaluate each component of the analyst on its own dataset, with deepeval.

    python -m evals.run_evals                    # every component
    python -m evals.run_evals router retriever   # just these
    EVAL_RUN=my_run python -m evals.run_evals    # name the run (default: <date>_<model>)

Each run is logged to evals/runs/<run>/: one <component>.json per component (every case with its
output, scores and the judge's reasons) and summary.json. Compare two runs with evals/compare.py.

Components and their datasets (evals/datasets/<component>.json):
  query_analyzer    splitting multi-question messages; intent, places, topic,
                    period of each question; follow-ups                        (code metric)
  scope_guard       in scope, out of scope, or asks for data we don't have   (code metric)
  router            report vs compare vs direct answer                        (code metric)
  entity_resolver   finding/matching the city, state or category in a question (code metric)
  retriever         the SQL the model writes returns the right data            (code metric)
  answer_generator  the written answer to a direct question, given the
                    correct data (so retriever mistakes don't leak in)        (LLM judge)
  report_generator  the narrative of a report or comparison, for a question,
                    given the computed facts                                  (LLM judge + code)

The judge for LLM metrics is NVIDIA Nemotron via NVIDIA's API when NVIDIA_API_KEY is set in .env,
else a local Ollama model (see evals/judges.py). A judge should be much stronger than the gemma2:2b
being judged.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
# A local judge on a small GPU can need minutes per test case; deepeval's default cap is 180s.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "1800")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "3600")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from deepeval import evaluate  # noqa: E402
from deepeval.dataset import EvaluationDataset, Golden  # noqa: E402
from deepeval.evaluate import AsyncConfig, DisplayConfig, ErrorConfig  # noqa: E402
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval  # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

from analyst import db, query, reports  # noqa: E402
from analyst.agent import MODEL, PROVIDER, Analyst, ModelOutputError  # noqa: E402
from evals.judges import make_judge  # noqa: E402
from evals.metrics import ExactMatch, NumbersGrounded, PlanMatch, ResultMatch  # noqa: E402

HERE = Path(__file__).parent
JUDGE, JUDGE_NAME = make_judge()
RUN_NAME = os.getenv("EVAL_RUN") or f"{time.strftime('%Y-%m-%d_%H%M')}_{re.sub(r'[^A-Za-z0-9.-]+', '-', MODEL)}"
RUN_DIR = HERE / "runs" / RUN_NAME


def load(component: str) -> EvaluationDataset:
    """Read evals/datasets/<component>.json into goldens; extra keys go to additional_metadata."""
    rows = json.loads((HERE / "datasets" / f"{component}.json").read_text(encoding="utf-8"))
    goldens = [Golden(input=r["input"], expected_output=_text(r.get("expected_output")),
                      additional_metadata={k: v for k, v in r.items() if k not in ("input", "expected_output")})
               for r in rows]
    return EvaluationDataset(goldens=goldens)


def _text(value):
    return value if value is None or isinstance(value, str) else json.dumps(value)


def route(analyst: Analyst, question: str):
    """The app's real routing for a fresh question: the query analyser, then the model's router for
    what the analyser leaves "unsure". Returns the first question's Part. A model error is a failed case."""
    analyst.history, analyst.last_parts = [], []
    part = analyst.plan(question)[0]
    if part.intent == "unsure":
        try:
            analyst.route_with_model(part)
        except ModelOutputError as e:
            part.intent = f"model error: {e}"
    return part


def rows_json(df) -> str:
    return df.head(20).to_json(orient="records", date_format="iso", double_precision=2)


# --- one builder per component: (test cases, metrics) -------------------------------------


def query_analyzer(analyst: Analyst, con):
    """The analyser alone (no model): its plan for each message, compared field by field."""
    cases = []
    for g in load("query_analyzer").goldens:
        previous = g.additional_metadata.get("previous")
        prev_parts = analyst.plan(previous) if previous else None
        parts = query.analyze(con, g.input, prev_parts)
        plan = [{"intent": p.intent, "entities": [f"{t}:{v}" for t, v in p.entities], "focus": p.focus,
                 "periods": [x["label"] for x in p.periods]} for p in parts]
        label = g.input if not previous else f"{g.input}  (after: {previous})"
        cases.append(LLMTestCase(input=label, actual_output=json.dumps(plan), expected_output=g.expected_output))
    return cases, [PlanMatch()]


def scope_guard(analyst: Analyst, con):
    cases = []
    for g in load("scope_guard").goldens:
        kind = route(analyst, g.input).intent
        label = "in_scope" if kind in ("report", "direct", "compare", "missing_entity") else kind
        cases.append(LLMTestCase(input=g.input, actual_output=label, expected_output=g.expected_output))
    return cases, [ExactMatch()]


def router(analyst: Analyst, con):
    cases = []
    for g in load("router").goldens:
        kind = route(analyst, g.input).intent
        cases.append(LLMTestCase(input=g.input, actual_output=kind, expected_output=g.expected_output))
    return cases, [ExactMatch()]


def entity_resolver(analyst: Analyst, con):
    cases = []
    for g in load("entity_resolver").goldens:
        meta = g.additional_metadata
        if meta["mode"] == "detect":
            found = reports.detect(con, g.input)
        else:
            value, _ = reports.resolve(con, meta["entity_type"], g.input)
            found = (meta["entity_type"], value) if value else None
        actual = f"{found[0]}: {found[1]}" if found else "none"
        cases.append(LLMTestCase(input=f"[{meta['mode']}] {g.input}", actual_output=actual,
                                 expected_output=g.expected_output))
    return cases, [ExactMatch()]


def retriever(analyst: Analyst, con):
    """Runs the real path for a direct question: the analyser (and model router if unsure), then
    retrieve() (SQL templates, entity hint, literal fixing, error-fix loop). The result must match
    the reference query's."""
    cases = []
    for g in load("retriever").goldens:
        expected = rows_json(db.run_query(con, g.additional_metadata["reference_sql"]))
        part = route(analyst, g.input)
        try:
            sql, df, _ = analyst.retrieve(part.query or g.input, part.sql if part.intent == "direct" else "")
            actual = rows_json(df) if df is not None else f"query failed: {sql}"
        except ModelOutputError as e:
            sql, actual = "", f"model error: {e}"
        cases.append(LLMTestCase(input=g.input, actual_output=actual, expected_output=expected,
                                 metadata={"sql": sql, "reference_sql": g.additional_metadata["reference_sql"]}))
    return cases, [ResultMatch()]


def answer_generator(analyst: Analyst, con):
    cases = []
    for g in load("answer_generator").goldens:
        sql = g.additional_metadata["reference_sql"]
        df = db.run_query(con, sql)
        rows = rows_json(df)
        answer = analyst.write_answer(g.input, sql, df)
        cases.append(LLMTestCase(input=g.input, actual_output=answer, expected_output=rows,
                                 retrieval_context=[f"Query result for '{g.input}': {rows}"]))
    model = JUDGE
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "The expected output is the query result (JSON rows) that the answer must be based on.",
            "Check that the actual output answers the question and that every figure or name it "
            "states matches the expected output (rounding and thousands separators are fine).",
            "If the result is empty or zero, the right answer says there is no matching data.",
            "Penalize wrong values, wrong rankings, invented numbers, and answers that only point "
            "to a table without saying anything.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=model, threshold=0.6,
    )
    return cases, [correctness, FaithfulnessMetric(model=model, threshold=0.7),
                   AnswerRelevancyMetric(model=model, threshold=0.7)]


def report_generator(analyst: Analyst, con):
    """Each question goes through the analyser and the report/comparison builder (all code), then
    the model writes the narrative, which is scored against the facts it was given."""
    cases = []
    for g in load("report_generator").goldens:
        analyst.history, analyst.last_parts = [], []
        part = analyst.plan(g.input)[0]
        try:
            report = analyst.run(part)
            narrative, facts = report.text, report.facts or ["(no facts: not routed to a report)"]
        except ModelOutputError as e:
            narrative, facts = f"model error: {e}", ["(model error)"]
        cases.append(LLMTestCase(input=g.input, actual_output=narrative, retrieval_context=facts))
    model = JUDGE
    quality = GEval(
        name="Report quality",
        evaluation_steps=[
            "The retrieval context lists the facts computed for the report; the actual output is "
            "the written analysis of them.",
            "Check that it starts with an 'Answer' that directly answers the question in the input.",
            "Check that the 'Key findings' bullets make comparisons supported by the facts (against "
            "the marketplace, rank among peers, previous period, or within a breakdown) and explain "
            "what they mean, instead of only repeating figures.",
            "Check that it ends with one concrete 'Recommendation' that follows from the weakest result.",
            "Penalize invented numbers, contradictions of the facts, claims not supported by the facts, "
            "emojis, and text over 180 words.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        model=model, threshold=0.6,
    )
    return cases, [NumbersGrounded(), FaithfulnessMetric(model=model, threshold=0.7), quality]


COMPONENTS = {f.__name__: f for f in
              (query_analyzer, scope_guard, router, entity_resolver, retriever, answer_generator, report_generator)}


def run(component: str, analyst: Analyst, con) -> dict:
    start = time.time()
    print(f"\n=== {component}: running the component ===", flush=True)
    cases, metrics = COMPONENTS[component](analyst, con)
    print(f"=== {component}: scoring {len(cases)} cases with {[m.__name__ for m in metrics]} ===", flush=True)
    result = evaluate(cases, metrics,
                      async_config=AsyncConfig(run_async=False),  # sequential: gentle on API rate limits and Ollama
                      display_config=DisplayConfig(print_results=False, show_indicator=False),
                      error_config=ErrorConfig(ignore_errors=True))  # a judge error fails one case, not the run

    rows, totals = [], {}
    for tr in result.test_results:
        # A metric whose judge call failed (rate limit, bad JSON) has an error and no real score: keep
        # it out of the averages instead of counting it as a zero.
        scores = {m.name: {"score": round(m.score or 0, 3), "passed": m.success, "reason": m.reason,
                           "error": m.error} for m in tr.metrics_data or []}
        for name, s in scores.items():
            if not s["error"]:
                totals.setdefault(name, []).append(s)
        errored = any(s["error"] for s in scores.values())
        rows.append({"input": tr.input, "actual_output": tr.actual_output, "expected_output": tr.expected_output,
                     "passed": tr.success and not errored, "errored": errored, "metrics": scores})
    summary = {
        "component": component, "model": MODEL, "judge": JUDGE_NAME, "cases": len(rows),
        "passed": sum(r["passed"] for r in rows), "errored": sum(r["errored"] for r in rows),
        "seconds": round(time.time() - start),
        "metrics": {name: {"pass_rate": round(sum(s["passed"] for s in ss) / len(ss), 3),
                           "avg_score": round(sum(s["score"] for s in ss) / len(ss), 3)}
                    for name, ss in totals.items()},
        "results": rows,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / f"{component}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"{component}: {summary['passed']}/{summary['cases']} cases passed in {summary['seconds']}s")
    if summary["errored"]:
        first = next(s["error"] for r in rows for s in r["metrics"].values() if s["error"])
        print(f"  WARNING: {summary['errored']} cases had judge errors and are not valid results "
              f"(first error: {str(first)[:200]}). Re-run this component.")
    for name, m in summary["metrics"].items():
        print(f"  {name:<20} pass rate {m['pass_rate']:.0%}   avg score {m['avg_score']:.2f}")
    for r in rows:
        if not r["passed"]:
            why = "; ".join(f"{n}: {s['reason']}" for n, s in r["metrics"].items() if not s["passed"])
            print(f"  FAIL {r['input'][:60]!r} -> {str(r['actual_output'])[:80]!r}\n       {str(why)[:300]}")
    return summary


def main(argv: list[str]) -> None:
    names = argv or list(COMPONENTS)
    unknown = [n for n in names if n not in COMPONENTS]
    if unknown:
        sys.exit(f"Unknown component(s) {unknown}. Choose from: {', '.join(COMPONENTS)}")
    keep_awake()
    con = db.connect()
    analyst = Analyst(con)
    print(f"Run {RUN_NAME}: model {MODEL} ({PROVIDER}), judge {JUDGE_NAME}")
    summaries = [run(name, analyst, con) for name in names]
    write_summary()
    print(f"\n=== Summary ({RUN_DIR}) ===")
    for s in summaries:
        print(f"{s['component']:<18} {s['passed']:>2}/{s['cases']:<2} passed  ({s['seconds']}s)")


def keep_awake() -> None:
    """Runs take an hour or more; if Windows goes to sleep the run freezes and its API calls break.
    Ask Windows not to sleep until this process exits (closing a laptop lid can still override it)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def write_summary() -> None:
    """summary.json: per-component totals for every component logged in this run so far."""
    components = {}
    for f in sorted(RUN_DIR.glob("*.json")):
        if f.name != "summary.json":
            d = json.loads(f.read_text(encoding="utf-8"))
            components[d["component"]] = {k: d.get(k, 0) for k in ("cases", "passed", "errored", "seconds", "metrics")}
    summary = {"run": RUN_NAME, "model": MODEL, "provider": PROVIDER, "judge": JUDGE_NAME,
               "created": time.strftime("%Y-%m-%d %H:%M"), "components": components}
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
