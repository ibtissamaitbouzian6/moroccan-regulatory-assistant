# tests/agents/orchestrator/test_validation.py
"""Tests of src/agents/orchestrator/validation.py (strict readers of LLM replies).

These helpers are NOT called by `graph.py` today (`plan_node` and `grade_node` have their own
inline logic); they define the strict behaviour a later change of the graph can adopt.
"""

import pytest

from src.agents.orchestrator.validation import (
    clean_relevant,
    clean_sub_questions,
    coerce_bool,
    parse_json_or_none,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('  {"a": 1}  ', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ("{}", {}),
        ("", None),
        ("   ", None),
        ("garbage", None),
        ("{broken", None),
        ('["a"]', None),
        ("42", None),
        ('"text"', None),
        ("null", None),
        (None, None),
        (42, None),
        (b'{"a": 1}', None),
    ],
)
def test_parse_json_or_none(text, expected):
    assert parse_json_or_none(text) == expected


def test_parse_json_or_none_deeply_nested_does_not_raise():
    assert parse_json_or_none("[" * 100000) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        (" yes ", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("no", False),
        ("0", False),
        ("", False),
        (1, True),
        (0, False),
        (2, False),
        (-1, False),
        ("maybe", False),
        (None, False),
        ([], False),
        (1.0, False),
    ],
)
def test_coerce_bool_default_false(value, expected):
    assert coerce_bool(value) is expected


def test_coerce_bool_string_false_is_not_true():
    """The reason this function exists: bool("false") is True."""
    assert bool("false") is True
    assert coerce_bool("false") is False


def test_coerce_bool_default_is_used_only_for_unreadable_strings_and_other_types():
    assert coerce_bool("maybe", default=True) is True
    assert coerce_bool(None, default=True) is True
    assert coerce_bool("false", default=True) is False
    assert coerce_bool("true", default=False) is True


def test_clean_sub_questions_filters_strips_dedupes_and_caps():
    raw = ["  a ", "a", "", " ", "\u200b", 1, None, ["x"], "b", "c", "d"]
    assert clean_sub_questions(raw, 3) == ["a", "b", "c"]


def test_clean_sub_questions_keeps_the_planner_order():
    assert clean_sub_questions(["c", "a", "b"], 5) == ["c", "a", "b"]


@pytest.mark.parametrize("value", [None, "a", 1, {"a": 1}, ("a", "b")])
def test_clean_sub_questions_not_a_list_gives_empty(value):
    assert clean_sub_questions(value, 3) == []


@pytest.mark.parametrize("limit", [0, -1])
def test_clean_sub_questions_non_positive_limit_gives_empty(limit):
    assert clean_sub_questions(["a"], limit) == []


def test_clean_sub_questions_arabic_punctuation_only_is_dropped():
    assert clean_sub_questions(["؟", "سؤال صحيح"], 3) == ["سؤال صحيح"]


def test_clean_relevant_distinct_in_range_ints():
    assert clean_relevant([1, 2, 2, 3, 9, 0, -1], 3) == [1, 2, 3]


def test_clean_relevant_accepts_digit_strings_but_not_booleans_or_floats():
    assert clean_relevant(["2", " 3 ", True, False, 1.0, "x", None], 3) == [2, 3]


@pytest.mark.parametrize("value", [None, 1, "1", {"relevant": [1]}])
def test_clean_relevant_not_a_list_gives_empty(value):
    assert clean_relevant(value, 3) == []


def test_clean_relevant_zero_total_gives_empty():
    assert clean_relevant([1], 0) == []
