# src/agents/orchestrator/validation.py
"""Strict validation of what the LLM returns to the orchestrator (P3, spec section 7).

`json.loads` only proves that a text is JSON. These helpers check the structure and the types,
and always fall back to the safe value of the spec's error table instead of raising:

- planner: `sub_questions` -> the list of non-empty strings, else [question]; `needs_web` -> bool;
- grader: `relevant` -> the distinct, in-range document numbers, else "none relevant".

They never see a user, a passage or a role: access control stays in P2 (Qdrant and BM25 filters).
"""

from __future__ import annotations

import json
from typing import Any

from src.ingestion import normalize_ar

_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0", ""}


def parse_json_or_none(text: Any) -> dict | None:
    """The JSON object in an LLM reply (code fences tolerated), or None.

    None for an empty reply, a non-string, invalid JSON, or valid JSON that is not an object
    (a list, a number, a string, null).
    """
    if not isinstance(text, str):
        return None
    clean = text.strip()
    if not clean:
        return None
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        data = json.loads(clean.strip())
    except (ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def coerce_bool(value: Any, default: bool = False) -> bool:
    """A real boolean from what an LLM wrote.

    `bool("false")` is True in Python, which is the bug this function exists to prevent: the
    strings "true"/"false" (any case) and the integers 0/1 are understood, anything else gives
    `default`.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value in (0, 1) and bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE:
            return True
        if word in _FALSE:
            return False
    return default


def clean_sub_questions(value: Any, limit: int) -> list[str]:
    """The usable sub-questions: strings, stripped, non-empty after `normalize_ar`, distinct.

    Anything that is not a list gives []. At most `limit` are kept, in the planner's order.
    """
    if not isinstance(value, list):
        return []
    kept: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and normalize_ar(text) and text not in kept:
            kept.append(text)
    return kept[: max(limit, 0)]


def clean_relevant(value: Any, total: int) -> list[int]:
    """The distinct document numbers (1-based) in `1..total` the grader found helpful.

    Accepts integers and integer-like strings ("2"), which small models often write. Booleans,
    floats, out-of-range numbers and anything that is not a list are ignored.
    """
    if not isinstance(value, list):
        return []
    numbers: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, str) and item.strip().isdigit():
            item = int(item.strip())
        if isinstance(item, int) and 1 <= item <= total and item not in numbers:
            numbers.append(item)
    return numbers
