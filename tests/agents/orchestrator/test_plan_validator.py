# tests/agents/orchestrator/test_plan_validator.py
"""Tests of src/agents/orchestration_support/plan_validator.py.

The diagnosis must give exactly what the existing `plan_node` keeps, plus the reasons.

Two things `plan_node` does today are reproduced on purpose (and reported as issues, not hidden):
it keeps duplicate sub-questions, and it reads `needs_web` with `bool(value)`, so the strings
"false" and "maybe" both give True.
"""

from uuid import uuid4

import pytest
from test_orchestrator_graph import ScriptedLLM

from src.agents.orchestration_support import plan_validator as pv
from src.agents.orchestration_support.plan_validator import (
    PlanDiagnosis,
    count_issues,
    diagnose_plan,
)
from src.agents.orchestrator.graph import plan_node
from src.core.config import cfg
from src.core.contracts import registry
from src.core.contracts.models import UserContext

Q = "ما هو أجل الطعن؟"


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    registry.reset()


CASES = [
    # reply, expected sub_questions, needs_web, issues, used_fallback
    ('{"sub_questions": ["a", "b"], "needs_web": false}', ("a", "b"), False, (), False),
    ('{"sub_questions": ["a"], "needs_web": true}', ("a",), True, (), False),
    ('```json\n{"sub_questions": ["a"], "needs_web": true}\n```', ("a",), True, (), False),
    ("", (Q,), False, (pv.EMPTY_REPLY,), True),
    ("   ", (Q,), False, (pv.EMPTY_REPLY,), True),
    (None, (Q,), False, (pv.EMPTY_REPLY,), True),
    ("garbage", (Q,), False, (pv.INVALID_JSON,), True),
    ("{broken", (Q,), False, (pv.INVALID_JSON,), True),
    ('["a"]', (Q,), False, (pv.NOT_AN_OBJECT,), True),
    ("42", (Q,), False, (pv.NOT_AN_OBJECT,), True),
    (
        '{"needs_web": true}',
        (Q,),
        True,
        (pv.SUB_QUESTIONS_MISSING,),
        True,
    ),
    (
        '{"sub_questions": "a", "needs_web": false}',
        (Q,),
        False,
        (pv.SUB_QUESTIONS_NOT_A_LIST,),
        True,
    ),
    (
        '{"sub_questions": [1, null, ""], "needs_web": false}',
        (Q,),
        False,
        (pv.SUB_QUESTIONS_DROPPED, pv.SUB_QUESTIONS_NONE_USABLE),
        True,
    ),
    (
        '{"sub_questions": ["a", 1, "a", "b"], "needs_web": false}',
        ("a", "a", "b"),  # plan_node keeps the duplicate
        False,
        (pv.SUB_QUESTIONS_DROPPED, pv.SUB_QUESTIONS_DUPLICATED),
        False,
    ),
    (
        '{"sub_questions": ["a", "b", "c", "d", "e"], "needs_web": false}',
        ("a", "b", "c"),
        False,
        (pv.SUB_QUESTIONS_TRUNCATED,),
        False,
    ),
    ('{"sub_questions": ["a"]}', ("a",), False, (pv.NEEDS_WEB_MISSING,), False),
    (
        '{"sub_questions": ["a"], "needs_web": "false"}',
        ("a",),
        True,  # plan_node: bool("false") is True
        (pv.NEEDS_WEB_NOT_BOOLEAN, pv.NEEDS_WEB_MISREAD),
        False,
    ),
    (
        '{"sub_questions": ["a"], "needs_web": "true"}',
        ("a",),
        True,
        (pv.NEEDS_WEB_NOT_BOOLEAN,),
        False,
    ),
    (
        '{"sub_questions": ["a"], "needs_web": 1}',
        ("a",),
        True,
        (pv.NEEDS_WEB_NOT_BOOLEAN,),
        False,
    ),
    (
        '{"sub_questions": ["a"], "needs_web": "maybe"}',
        ("a",),
        True,  # plan_node: bool("maybe") is True
        (pv.NEEDS_WEB_UNREADABLE, pv.NEEDS_WEB_MISREAD),
        False,
    ),
    (
        '{"sub_questions": ["a"], "needs_web": null}',
        ("a",),
        False,
        (pv.NEEDS_WEB_UNREADABLE,),
        False,
    ),
]


@pytest.mark.parametrize("reply,subs,needs_web,issues,fallback", CASES)
def test_diagnosis(reply, subs, needs_web, issues, fallback):
    d = diagnose_plan(reply, Q, 3)
    assert d.sub_questions == subs
    assert d.needs_web is needs_web
    assert d.issues == issues
    assert d.used_fallback is fallback
    assert d.clean is (issues == ())


def _run_plan_node(reply: str) -> dict:
    registry.override("llm", ScriptedLLM([reply]))
    ctx = UserContext(
        user_id=uuid4(), username="meryem", role="lecteur", web_enabled=True, request_id=uuid4()
    )
    state = {
        "question": Q,
        "ctx": ctx,
        "tokens": 0,
        "emit": lambda event: None,
    }
    return plan_node(state)  # type: ignore[arg-type]


# Replies that are not in CASES: whitespace, wrong types, truthy and falsy values of every kind.
EQUIVALENCE_EXTRA = [
    '{"sub_questions": ["  a  ", "b"], "needs_web": false}',
    '{"sub_questions": ["a", " ", "\u200b", "b"], "needs_web": false}',
    '{"sub_questions": ["a", "a", "a"], "needs_web": true}',
    '{"sub_questions": [["a"], {"q": "a"}, "b"], "needs_web": false}',
    '{"sub_questions": [], "needs_web": false}',
    '{"sub_questions": null, "needs_web": true}',
    '{"sub_questions": ["a"], "needs_web": "no"}',
    '{"sub_questions": ["a"], "needs_web": "0"}',
    '{"sub_questions": ["a"], "needs_web": ""}',
    '{"sub_questions": ["a"], "needs_web": 0}',
    '{"sub_questions": ["a"], "needs_web": 2}',
    '{"sub_questions": ["a"], "needs_web": []}',
    '{"sub_questions": ["a"], "needs_web": ["x"]}',
    '{"sub_questions": ["a"], "needs_web": {}}',
    '{"sub_questions": ["a"], "needs_web": 0.0}',
    "{}",
]


@pytest.mark.parametrize("reply", [c[0] for c in CASES if c[0] is not None] + EQUIVALENCE_EXTRA)
def test_diagnosis_equals_what_the_existing_plan_node_keeps(reply):
    """Same input, same output as the frozen `plan_node`: no second way of cleaning."""
    update = _run_plan_node(reply)
    d = diagnose_plan(reply, Q, int(cfg.agents.max_sub_questions))
    assert tuple(update["sub_questions"]) == d.sub_questions
    assert update["needs_web"] is d.needs_web


def test_none_reply_equals_plan_node_with_an_empty_reply():
    update = _run_plan_node("")
    d = diagnose_plan(None, Q, int(cfg.agents.max_sub_questions))
    assert tuple(update["sub_questions"]) == d.sub_questions
    assert update["needs_web"] is d.needs_web


def test_plan_node_gaps_are_reported_not_hidden():
    """Where plan_node differs from a strict reading, the diagnosis keeps plan_node's value."""
    d = diagnose_plan('{"sub_questions": ["a", "a"], "needs_web": "false"}', Q, 3)
    assert d.sub_questions == ("a", "a")
    assert d.needs_web is True
    assert pv.SUB_QUESTIONS_DUPLICATED in d.issues
    assert pv.NEEDS_WEB_MISREAD in d.issues
    assert d.clean is False


def test_a_real_boolean_is_never_misread():
    for word in ("true", "false"):
        d = diagnose_plan(f'{{"sub_questions": ["a"], "needs_web": {word}}}', Q, 3)
        assert pv.NEEDS_WEB_MISREAD not in d.issues
        assert d.clean is True


def test_never_raises_on_unexpected_input():
    for reply in (object(), 3.5, b"bytes", ["a"], {"sub_questions": ["a"]}, True):
        d = diagnose_plan(reply, Q, 3)
        assert isinstance(d, PlanDiagnosis)
        assert d.sub_questions == (Q,)
        assert d.used_fallback is True


def test_diagnosis_keeps_no_text_beyond_the_cleaned_values():
    reply = '{"sub_questions": ["SECRET_SUB"], "needs_web": "SECRET_FLAG"}'
    d = diagnose_plan(reply, "SECRET_QUESTION", 3)
    assert "SECRET_FLAG" not in repr(d.issues)
    assert all(code.isidentifier() for code in d.issues)


def test_count_issues_over_many_replies():
    diagnoses = [diagnose_plan(c[0], Q, 3) for c in CASES]
    counts = count_issues(diagnoses)
    assert counts[pv.EMPTY_REPLY] == 3
    assert counts[pv.INVALID_JSON] == 2
    assert counts[pv.NEEDS_WEB_NOT_BOOLEAN] == 3
    assert counts[pv.NEEDS_WEB_MISREAD] == 2
    assert counts[pv.SUB_QUESTIONS_DUPLICATED] == 1
    assert sum(counts.values()) == sum(len(d.issues) for d in diagnoses)
