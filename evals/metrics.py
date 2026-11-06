"""Code-based deepeval metrics for the components whose output is either right or wrong.

The LLM-judged metrics (faithfulness, answer relevancy, GEval) are built in run_evals.py.
"""

import json
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from analyst.agent import numbers_in, unsupported_numbers
from analyst.reports import _norm


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


class PlanMatch(_CodeMetric):
    """The query analyser's plan must match the expected one: the same number of questions, and for
    each question the fields the dataset specifies (intent, entities, focus, periods).
    Score = share of expected fields that match; a missing or extra question counts as all wrong."""

    name = "Plan match"

    def score_case(self, test_case):
        expected = json.loads(test_case.expected_output)
        actual = json.loads(test_case.actual_output)
        total = sum(len(p) for p in expected) + sum(len(expected[0]) for _ in actual[len(expected):])
        matched, wrong = 0, []
        for i, exp in enumerate(expected):
            act = actual[i] if i < len(actual) else {}
            for key, value in exp.items():
                if act.get(key) == value:
                    matched += 1
                else:
                    wrong.append(f"q{i + 1} {key}: expected {value}, got {act.get(key)}")
        if len(actual) != len(expected):
            wrong.insert(0, f"expected {len(expected)} questions, got {len(actual)}")
        return matched / total, "plan matches" if not wrong else "; ".join(wrong)


class NumbersGrounded(_CodeMetric):
    """Every number in the output must appear in the retrieval context or the question (the same
    check the app applies to direct answers). Score = share of numbers that do."""

    name = "Numbers grounded"

    def score_case(self, test_case):
        numbers = [n for n in numbers_in(test_case.actual_output) if not (n <= 10 and n == int(n))]
        if not numbers:
            return 1.0, "no numbers to check"
        source = numbers_in(" ".join(test_case.retrieval_context) + " " + test_case.input)
        made_up = unsupported_numbers(test_case.actual_output, source)
        reason = f"not in the data: {made_up[:5]}" if made_up else "all numbers come from the data"
        return 1 - len(made_up) / len(numbers), reason


def _values(rows: list[dict]) -> list:
    return [v for row in rows for v in row.values() if v is not None]


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= 0.01 * max(abs(a), 1e-9)
    return _norm(str(a))[:10] == _norm(str(b))[:10] if _is_date(a) else _norm(str(a)) == _norm(str(b))


def _is_date(v) -> bool:
    return isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}", v) is not None
