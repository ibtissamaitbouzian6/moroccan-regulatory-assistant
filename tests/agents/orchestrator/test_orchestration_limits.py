# tests/agents/orchestrator/test_orchestration_limits.py
"""Tests of src/agents/orchestration_support/orchestration_limits.py."""

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from test_orchestrator_graph import (
    ScriptedLLM,
    StubRetrieval,
    StubWebAgent,
    StubWriter,
    _make_chunk,
)

from src.agents.orchestration_support.orchestration_limits import (
    GRADED_ABOVE_RETRIEVED,
    MIN_RELEVANT_ABOVE_GRADED,
    NOT_POSITIVE,
    WRITER_BELOW_RELEVANT,
    Deadline,
    RequestLimits,
    check_limits,
)
from src.agents.orchestrator.graph import GraphState, over_budget
from src.agents.orchestrator.real import RealOrchestrator, _recursion_limit
from src.core.config import cfg
from src.core.contracts import registry
from src.core.contracts.models import Passage, UserContext


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    registry.reset()


def agents(**changes):
    base = dict(
        max_iterations=3,
        max_sub_questions=3,
        max_tokens_per_request=8000,
        grade_top_n=5,
        min_relevant=1,
        max_passages_to_writer=8,
        request_timeout_s=60,
    )
    base.update(changes)
    return SimpleNamespace(**base)


# --- RequestLimits -----------------------------------------------------------------------------
def test_from_config_reads_the_real_config():
    limits = RequestLimits.from_config()
    assert limits.max_iterations == cfg.agents.max_iterations
    assert limits.max_sub_questions == cfg.agents.max_sub_questions
    assert limits.max_tokens == cfg.agents.max_tokens_per_request
    assert limits.request_timeout_s == float(cfg.agents.request_timeout_s)


def test_worst_cases_with_the_default_values():
    limits = RequestLimits.from_config(agents())
    assert limits.worst_case_searches == 9
    assert limits.worst_case_llm_calls == 1 + 9 + 6
    assert limits.recommended_recursion_limit == 3 * 10 + 10


@pytest.mark.parametrize("subs,iters", [(1, 1), (1, 3), (3, 3), (5, 2), (2, 6)])
def test_recursion_limit_equals_the_one_of_real_orchestrator(subs, iters):
    limits = RequestLimits.from_config(agents(max_sub_questions=subs, max_iterations=iters))
    assert limits.recommended_recursion_limit == _recursion_limit(subs, iters)


@pytest.mark.parametrize("used", [0, 1, 7999, 8000, 8001, 100000])
def test_budget_rule_equals_over_budget_of_the_graph(used):
    limits = RequestLimits.from_config()
    state: GraphState = {"tokens": used}
    assert limits.budget_reached(used) is over_budget(state)
    assert limits.tokens_left(used) == max(cfg.agents.max_tokens_per_request - used, 0)


def test_worst_case_searches_is_what_a_real_run_does():
    """Grader always empty: the existing graph makes exactly max_sub x max_iterations searches."""
    p = Passage(chunk=_make_chunk(), score=0.5, rank=1)
    subs = cfg.agents.max_sub_questions
    iters = cfg.agents.max_iterations
    plan = {"sub_questions": [f"q{i}" for i in range(subs)], "needs_web": False}
    replies = [json.dumps(plan)]
    for s in range(subs):
        for it in range(iters):
            replies.append('{"relevant": []}')
            if it < iters - 1:  # no reformulation after the last search of a sub-question
                replies.append(f"q{s}_v{it + 2}")
    llm = ScriptedLLM(replies)
    ret = StubRetrieval(default=[p])
    registry.override("llm", llm)
    registry.override("retrieval", ret)
    registry.override("web_agent", StubWebAgent())
    registry.override("writer", StubWriter())
    ctx = UserContext(
        user_id=uuid4(), username="meryem", role="lecteur", web_enabled=True, request_id=uuid4()
    )

    async def run():
        return [e async for e in RealOrchestrator().run("سؤال", ctx, use_web=True)]

    asyncio.run(run())
    limits = RequestLimits.from_config()
    assert len(ret.queries) == limits.worst_case_searches
    assert llm.calls <= limits.worst_case_llm_calls


# --- Deadline ----------------------------------------------------------------------------------
class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_deadline_counts_down_and_expires():
    clock = FakeClock()
    deadline = Deadline(10, clock)
    assert deadline.remaining() == 10
    assert not deadline.expired()
    clock.now += 4
    assert deadline.remaining() == 6
    clock.now += 6
    assert deadline.expired()
    clock.now += 50
    assert deadline.remaining() == 0.0  # never negative


def test_deadline_with_zero_or_negative_seconds_is_already_expired():
    assert Deadline(0, FakeClock()).expired()
    assert Deadline(-5, FakeClock()).expired()


def test_deadline_uses_a_real_monotonic_clock_by_default():
    deadline = Deadline(60)
    assert 0 < deadline.remaining() <= 60


# --- check_limits ------------------------------------------------------------------------------
def test_the_current_configuration_is_consistent():
    assert check_limits(RequestLimits.from_config(), cfg.retrieval.top_k_final) == []


@pytest.mark.parametrize(
    "name",
    [
        "max_iterations",
        "max_sub_questions",
        "max_tokens_per_request",
        "grade_top_n",
        "min_relevant",
        "max_passages_to_writer",
        "request_timeout_s",
    ],
)
def test_zero_values_are_reported(name):
    problems = check_limits(RequestLimits.from_config(agents(**{name: 0})))
    assert any(p.startswith(NOT_POSITIVE) for p in problems)


def test_relations_between_values():
    limits = RequestLimits.from_config(agents(min_relevant=6, grade_top_n=5))
    assert f"{MIN_RELEVANT_ABOVE_GRADED}:min_relevant" in check_limits(limits)

    limits = RequestLimits.from_config(agents(grade_top_n=12))
    assert f"{GRADED_ABOVE_RETRIEVED}:grade_top_n" in check_limits(limits, top_k_final=10)
    assert check_limits(limits) == []  # not checked without top_k_final

    limits = RequestLimits.from_config(agents(min_relevant=3, max_passages_to_writer=2))
    assert f"{WRITER_BELOW_RELEVANT}:max_passages_to_writer" in check_limits(limits)
