"""Code-based deepeval metrics for the components whose output is either right or wrong.

The LLM-judged metrics (faithfulness, answer relevancy, GEval) are built in run_evals.py.
"""

import json
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from analyst.reports import _norm

NUMBER = re.compile(r"-?(?<![\w.])\d[\d,]*(?:\.\d+)?(?![A-Za-z]*\d)")  # skips digits inside IDs


class _CodeMetric(BaseMetric):
    """Shared plumbing: subclasses implement score(test_case) -> (score, reason)."""

    name = "Code metric"

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def score_case(self, test_case: LLMTestCase) -> tuple[float, str]:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score, self.reason = self.score_case(test_case)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return self.name


class ExactMatch(_CodeMetric):
    """actual_output must equal expected_output (case-insensitive). For labels like 'report'."""

    name = "Exact match"

    def score_case(self, test_case):
        ok = test_case.actual_output.strip().lower() == test_case.expected_output.strip().lower()
        return float(ok), f"expected {test_case.expected_output!r}, got {test_case.actual_output!r}"


class ResultMatch(_CodeMetric):
    """The agent's query result must contain every value of the reference query's result.

    Both outputs are JSON row lists. Numbers match within 1% (so 3.9 vs 3.91 or an unrounded
    total still pass); text matches ignoring case and accents. Extra columns are fine, since a
    question rarely pins down exactly which columns to return. Score = share of values found.
    """

    name = "Result match"

    def score_case(self, test_case):
        expected = _values(json.loads(test_case.expected_output))
        try:
            actual = _values(json.loads(test_case.actual_output))
        except json.JSONDecodeError:
            return 0.0, f"no result: {test_case.actual_output[:200]}"
        missing = [v for v in expected if not any(_same(v, a) for a in actual)]
        score = 1 - len(missing) / len(expected) if expected else float(not actual)
        reason = "all reference values found" if not missing else f"missing {missing[:5]}"
        return score, reason


class NumbersGrounded(_CodeMetric):
    """Every number in the output must appear in the retrieval context (within 1%, or written
    in thousands/millions). Small numbers (<= 10) are skipped: "top 3", "5 stars"."""

    name = "Numbers grounded"

    def score_case(self, test_case):
        source = [_num(n) for c in test_case.retrieval_context for n in NUMBER.findall(c)]
        source += [_num(n) for n in NUMBER.findall(test_case.input)]
        numbers = [_num(n) for n in NUMBER.findall(test_case.actual_output)]
        numbers = [n for n in numbers if not (abs(n) <= 10 and n == int(n))]
        made_up = [n for n in numbers
                   if not any(abs(s / scale - n) <= 0.01 * max(abs(n), 1) for s in source for scale in (1, 1e3, 1e6))]
        if not numbers:
            return 1.0, "no numbers to check"
        score = 1 - len(made_up) / len(numbers)
        return score, "all numbers come from the data" if not made_up else f"not in the data: {made_up[:5]}"


def _num(token: str) -> float:
    return float(token.replace(",", ""))


def _values(rows: list[dict]) -> list:
    return [v for row in rows for v in row.values() if v is not None]


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= 0.01 * max(abs(a), 1e-9)
    return _norm(str(a))[:10] == _norm(str(b))[:10] if _is_date(a) else _norm(str(a)) == _norm(str(b))


def _is_date(v) -> bool:
    return isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}", v) is not None
