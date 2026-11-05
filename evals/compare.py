"""Compare two logged evaluation runs.

    python -m evals.compare <run_a> <run_b>     # names of folders in evals/runs/

Prints the comparison and writes it to evals/runs/compare_<run_a>_vs_<run_b>.md: totals and
metric scores per component, then every case whose pass/fail changed between the runs.
"""

import json
import sys
from pathlib import Path

RUNS = Path(__file__).parent / "runs"


def load(run: str) -> tuple[dict, dict]:
    """(summary.json, {component: per-case results})"""
    folder = RUNS / run
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    cases = {}
    for name in summary["components"]:
        data = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        cases[name] = {r["input"]: r for r in data["results"]}
    return summary, cases


def pct(passed: int, total: int) -> str:
    return f"{passed}/{total} ({passed / total:.0%})" if total else "-"


def compare(run_a: str, run_b: str) -> str:
    (sa, ca), (sb, cb) = load(run_a), load(run_b)
    a, b = sa["model"], sb["model"]
    lines = [f"# Evaluation: `{a}` vs `{b}`", "",
             f"- **A**: run `{run_a}`, model `{a}` ({sa['provider']}), judge `{sa['judge']}`",
             f"- **B**: run `{run_b}`, model `{b}` ({sb['provider']}), judge `{sb['judge']}`", "",
             "## Cases passed", "",
             "| Component | A | B | Change | Time A | Time B |", "|---|---|---|---|---|---|"]
    names = [n for n in sa["components"] if n in sb["components"]]
    tot = [0, 0, 0]
    for n in names:
        x, y = sa["components"][n], sb["components"][n]
        tot = [tot[0] + x["passed"], tot[1] + y["passed"], tot[2] + x["cases"]]
        lines.append(f"| {n} | {pct(x['passed'], x['cases'])} | {pct(y['passed'], y['cases'])} | "
                     f"{y['passed'] - x['passed']:+d} | {x['seconds']}s | {y['seconds']}s |")
    lines.append(f"| **total** | **{pct(tot[0], tot[2])}** | **{pct(tot[1], tot[2])}** | **{tot[1] - tot[0]:+d}** | | |")

    lines += ["", "## Metrics (pass rate / average score)", "",
              "| Component | Metric | A | B | Score change |", "|---|---|---|---|---|"]
    for n in names:
        ma, mb = sa["components"][n]["metrics"], sb["components"][n]["metrics"]
        for m in ma:
            if m in mb:
                lines.append(f"| {n} | {m} | {ma[m]['pass_rate']:.0%} / {ma[m]['avg_score']:.2f} | "
                             f"{mb[m]['pass_rate']:.0%} / {mb[m]['avg_score']:.2f} | "
                             f"{mb[m]['avg_score'] - ma[m]['avg_score']:+.2f} |")

    lines += ["", "## Cases that changed", ""]
    for n in names:
        fixed = [q for q in ca[n] if q in cb[n] and not ca[n][q]["passed"] and cb[n][q]["passed"]]
        broke = [q for q in ca[n] if q in cb[n] and ca[n][q]["passed"] and not cb[n][q]["passed"]]
        still = [q for q in ca[n] if q in cb[n] and not ca[n][q]["passed"] and not cb[n][q]["passed"]]
        if not (fixed or broke or still):
            continue
        lines.append(f"### {n}")
        for label, qs in (("Fixed in B", fixed), ("Broken in B", broke), ("Failing in both", still)):
            for q in qs:
                lines.append(f"- **{label}**: {q}")
                lines.append(f"  - A: {_short(ca[n][q]['actual_output'])}")
                lines.append(f"  - B: {_short(cb[n][q]['actual_output'])}")
        lines.append("")
    return "\n".join(lines)


def _short(text, n: int = 160) -> str:
    text = " ".join(str(text).split())
    return f"`{text[:n]}{'…' if len(text) > n else ''}`"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    report = compare(sys.argv[1], sys.argv[2])
    out = RUNS / f"compare_{sys.argv[1]}_vs_{sys.argv[2]}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {out}")
