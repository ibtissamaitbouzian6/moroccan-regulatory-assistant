# src/agents/orchestration_support/orchestration_limits.py
"""The limits of one request, as one object, plus a deadline and a check of the `agents` config.

`graph.py` and `real.py` read `cfg.agents.*` directly and are not changed. This module gives the
same numbers a name and derives the worst cases from them (P3-16), so tuning (schedule day 10) can
see them before a run, and a later change can use one `RequestLimits` instead of repeating the
formulas. A test keeps `recommended_recursion_limit` equal to `real._recursion_limit`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.core.config import cfg

# Codes returned by `check_limits`.
NOT_POSITIVE = "not_positive"
MIN_RELEVANT_ABOVE_GRADED = "min_relevant_above_graded"
GRADED_ABOVE_RETRIEVED = "graded_above_retrieved"
WRITER_BELOW_RELEVANT = "writer_below_min_relevant"


@dataclass(frozen=True)
class RequestLimits:
    """Limits of one `/ask` request (`config.yaml`, section `agents`)."""

    max_iterations: int
    max_sub_questions: int
    max_tokens: int
    grade_top_n: int
    min_relevant: int
    max_passages_to_writer: int
    request_timeout_s: float

    @classmethod
    def from_config(cls, agents: Any = None) -> RequestLimits:
        """Read the limits from `cfg.agents`, or from any object with the same attributes."""
        a = agents if agents is not None else cfg.agents
        return cls(
            max_iterations=int(a.max_iterations),
            max_sub_questions=int(a.max_sub_questions),
            max_tokens=int(a.max_tokens_per_request),
            grade_top_n=int(a.grade_top_n),
            min_relevant=int(a.min_relevant),
            max_passages_to_writer=int(a.max_passages_to_writer),
            request_timeout_s=float(a.request_timeout_s),
        )

    @property
    def worst_case_searches(self) -> int:
        """P3-16: max_sub_questions x max_iterations."""
        return self.max_sub_questions * self.max_iterations

    @property
    def worst_case_llm_calls(self) -> int:
        """1 plan + a grade per search + a reformulation between two searches of a sub-question."""
        reformulations = self.max_sub_questions * max(self.max_iterations - 1, 0)
        return 1 + self.worst_case_searches + reformulations

    @property
    def recommended_recursion_limit(self) -> int:
        """LangGraph steps in the worst case: (search + grade + reformulate) per iteration."""
        return self.max_sub_questions * (3 * self.max_iterations + 1) + 10

    def tokens_left(self, used: int) -> int:
        return max(self.max_tokens - used, 0)

    def budget_reached(self, used: int) -> bool:
        """Same rule as `graph.over_budget`: stop once the total reaches the budget."""
        return used >= self.max_tokens


class Deadline:
    """A point in time on a monotonic clock. `clock` can be replaced in a test."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._end = clock() + max(float(seconds), 0.0)

    def remaining(self) -> float:
        """Seconds left, never below 0."""
        return max(self._end - self._clock(), 0.0)

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def check_limits(limits: RequestLimits, top_k_final: int | None = None) -> list[str]:
    """Problems of a configuration, as `code:name` strings; [] means consistent.

    `top_k_final` (`retrieval.top_k_final`) enables the check that grading does not ask for more
    passages than one search returns.
    """
    problems: list[str] = []
    values = {
        "max_iterations": limits.max_iterations,
        "max_sub_questions": limits.max_sub_questions,
        "max_tokens": limits.max_tokens,
        "grade_top_n": limits.grade_top_n,
        "min_relevant": limits.min_relevant,
        "max_passages_to_writer": limits.max_passages_to_writer,
        "request_timeout_s": limits.request_timeout_s,
    }
    problems.extend(f"{NOT_POSITIVE}:{name}" for name, v in values.items() if v <= 0)
    if limits.min_relevant > limits.grade_top_n:
        # a sub-question could never be answered
        problems.append(f"{MIN_RELEVANT_ABOVE_GRADED}:min_relevant")
    if top_k_final is not None and limits.grade_top_n > top_k_final:
        problems.append(f"{GRADED_ABOVE_RETRIEVED}:grade_top_n")
    if limits.max_passages_to_writer < limits.min_relevant:
        problems.append(f"{WRITER_BELOW_RELEVANT}:max_passages_to_writer")
    return problems
