# src/agents/orchestration_support/plan_validator.py
"""Diagnosis of a planner reply: what the orchestrator will use, and why.

`diagnose_plan` reproduces EXACTLY what `plan_node` (src/agents/orchestrator/graph.py) keeps from
a reply, for the same input: same JSON parsing (`graph.parse_json_or_none`), same filtering and
cap of `sub_questions`, same reading of `needs_web`. It does not clean "better" than `plan_node`:
a diagnosis that disagreed with the graph would describe a run that never happens.

What `plan_node` does today (and so what this module returns):

- `sub_questions`: keeps the items that are strings and non-empty after `normalize_ar`, as
  written (no `strip`), duplicates included, then cuts to `max_sub_questions`; none left -> the
  question itself;
- `needs_web`: `bool(value)` of the reply's value. So the string "false" and the string "maybe"
  both give True (any non-empty string is truthy).

The strict readers in `src.agents.orchestrator.validation` (`clean_sub_questions`, `coerce_bool`)
are NOT used by `plan_node`. This module uses `coerce_bool` only to say what the planner most
likely MEANT, so that a gap between the planner's intent and the value the graph really uses is
reported (`needs_web_misread`) instead of hidden. The same goes for duplicates
(`sub_questions_duplicated`): they are kept, and reported.

`PlanDiagnosis` never contains the reply, a sub-question or the question: only the values that go
to the graph, counts and codes (P3-33).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.agents.orchestrator.graph import parse_json_or_none
from src.agents.orchestrator.validation import coerce_bool
from src.ingestion import normalize_ar

# Issue codes, in the order they are looked for.
EMPTY_REPLY = "empty_reply"
INVALID_JSON = "invalid_json"
NOT_AN_OBJECT = "not_an_object"
SUB_QUESTIONS_MISSING = "sub_questions_missing"
SUB_QUESTIONS_NOT_A_LIST = "sub_questions_not_a_list"
SUB_QUESTIONS_NONE_USABLE = "sub_questions_none_usable"
SUB_QUESTIONS_DROPPED = "sub_questions_dropped"
SUB_QUESTIONS_TRUNCATED = "sub_questions_truncated"
SUB_QUESTIONS_DUPLICATED = "sub_questions_duplicated"
NEEDS_WEB_MISSING = "needs_web_missing"
NEEDS_WEB_NOT_BOOLEAN = "needs_web_not_boolean"
NEEDS_WEB_UNREADABLE = "needs_web_unreadable"
NEEDS_WEB_MISREAD = "needs_web_misread"


@dataclass(frozen=True)
class PlanDiagnosis:
    """What `plan_node` keeps from a reply (`sub_questions`, `needs_web`) and the reasons."""

    sub_questions: tuple[str, ...]
    needs_web: bool
    issues: tuple[str, ...]
    used_fallback: bool  # True: no sub-question was usable, so the question itself is used

    @property
    def clean(self) -> bool:
        """The reply was used as it came, with nothing to report."""
        return not self.issues


def _classify_unparsed(reply: Any) -> str:
    if not isinstance(reply, str) or not reply.strip():
        return EMPTY_REPLY
    text = reply.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        json.loads(text.strip())
    except (ValueError, RecursionError):
        return INVALID_JSON
    return NOT_AN_OBJECT


def diagnose_plan(reply: Any, question: str, max_sub_questions: int) -> PlanDiagnosis:
    """The sub-questions and `needs_web` the graph gets from `reply`, with the reasons.

    Never raises, whatever `reply` is (None, a number, a broken JSON...). With an unusable reply
    the result is the spec's fallback (P3 §7): `sub_questions = (question,)`, `needs_web = False`.

    `max_sub_questions` must be the value `plan_node` reads (`cfg.agents.max_sub_questions`).
    """
    issues: list[str] = []
    data = parse_json_or_none(reply)
    if data is None:
        issues.append(_classify_unparsed(reply))
        return PlanDiagnosis((question,), False, tuple(issues), used_fallback=True)

    # --- sub_questions: the same filter and cut as `plan_node` ---
    raw = data.get("sub_questions")
    subs: list[str] = []
    if "sub_questions" not in data:
        issues.append(SUB_QUESTIONS_MISSING)
    elif not isinstance(raw, list):
        issues.append(SUB_QUESTIONS_NOT_A_LIST)
    else:
        usable = [q for q in raw if isinstance(q, str) and normalize_ar(q)]
        if len(usable) < len(raw):
            issues.append(SUB_QUESTIONS_DROPPED)
        if len(usable) > max_sub_questions:
            issues.append(SUB_QUESTIONS_TRUNCATED)
        subs = usable[:max_sub_questions]
        if not subs:
            issues.append(SUB_QUESTIONS_NONE_USABLE)
        elif len(set(subs)) < len(subs):
            # `plan_node` keeps duplicates: the same sub-question would be searched twice.
            issues.append(SUB_QUESTIONS_DUPLICATED)
    used_fallback = not subs
    if used_fallback:
        subs = [question]

    # --- needs_web: the same `bool(...)` as `plan_node` ---
    needs_web = bool(data.get("needs_web", False)) if data else False
    if "needs_web" not in data:
        issues.append(NEEDS_WEB_MISSING)
    else:
        value = data["needs_web"]
        if not isinstance(value, bool):
            # Was the value a recognisable yes/no (a string "false", a 0/1), or garbage?
            understood = coerce_bool(value, default=True) == coerce_bool(value, default=False)
            issues.append(NEEDS_WEB_NOT_BOOLEAN if understood else NEEDS_WEB_UNREADABLE)
            if coerce_bool(value) is not needs_web:
                # The graph uses `bool(value)`; a strict reading gives the opposite.
                issues.append(NEEDS_WEB_MISREAD)

    return PlanDiagnosis(tuple(subs), needs_web, tuple(issues), used_fallback)


def count_issues(diagnoses: Iterable[PlanDiagnosis]) -> Counter[str]:
    """How many times each issue code appears over many replies (an evaluation run)."""
    counter: Counter[str] = Counter()
    for diagnosis in diagnoses:
        counter.update(diagnosis.issues)
    return counter
