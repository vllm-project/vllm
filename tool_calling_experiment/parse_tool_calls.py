#!/usr/bin/env python3
"""Parse ``<tool_call>`` XML blocks and final predictions.

Handles malformed JSON, missing closing tags, and multiple tool
calls in a single model response.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ------------------------------------------------------------------
# Tool-call parsing
# ------------------------------------------------------------------

# Matches <tool_call>...</tool_call> (greedy inside)
_TC_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# Fallback: unclosed <tool_call> at end of string
_TC_UNCLOSED = re.compile(
    r"<tool_call>\s*(.*)",
    re.DOTALL,
)

# JSON-like object pattern (very permissive)
_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse *text* as JSON dict."""
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting the first JSON object from the text
    m = _JSON_OBJ.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def parse_tool_calls(
    text: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Extract tool calls from model-generated text.

    Returns a list of ``(tool_name, arguments_dict)`` tuples.
    Gracefully handles:
    - Malformed JSON
    - Multiple tool calls in one response
    - Missing closing ``</tool_call>`` tags
    - Extra whitespace or newlines
    """
    results: list[tuple[str, dict[str, Any]]] = []

    # First try properly closed tags
    matches = _TC_PATTERN.findall(text)

    # If none found, try unclosed tag at end
    if not matches:
        m = _TC_UNCLOSED.search(text)
        if m:
            matches = [m.group(1)]

    for raw in matches:
        obj = _try_parse_json(raw)
        if obj is None:
            continue
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            # Sometimes model puts args as a string
            parsed_args = _try_parse_json(str(args))
            args = parsed_args if parsed_args else {}
        if name:
            results.append((name, args))

    return results


# ------------------------------------------------------------------
# Final-prediction parsing
# ------------------------------------------------------------------

_SCENE_PATTERN = re.compile(
    r"FINAL_SCENE:\s*(\S+)", re.IGNORECASE
)
_LONG_ACTION_PATTERN = re.compile(
    r"FINAL_LONG_ACTION:\s*(\S+)", re.IGNORECASE
)
_LAT_ACTION_PATTERN = re.compile(
    r"FINAL_LAT_ACTION:\s*(\S+)", re.IGNORECASE
)
_REVISED_PATTERN = re.compile(
    r"REVISED:\s*(\S+)", re.IGNORECASE
)
_REASON_PATTERN = re.compile(
    r"REASON:\s*(.+?)(?:\n|$)", re.IGNORECASE
)

VALID_SCENES = frozenset(
    [
        "nominal",
        "flooded",
        "incident_zone",
        "mounted_police",
        "flagger",
    ]
)
VALID_LONG_ACTIONS = frozenset(
    ["stop", "slowdown", "proceed", "null"]
)
VALID_LAT_ACTIONS = frozenset(
    ["lc_left", "lc_right", "null"]
)


def _clean_value(value: str) -> str:
    """Strip punctuation and whitespace from parsed value."""
    return value.strip().strip(".,;:\"'`").lower()


def _try_extract_scene_freeform(
    text: str,
) -> str | None:
    """Attempt to find a scene type from freeform text."""
    lower = text.lower()
    found = []
    for scene in VALID_SCENES:
        if scene in lower:
            found.append(scene)
    # If exactly one scene mentioned, use it
    if len(found) == 1:
        return found[0]
    # If multiple, prefer non-nominal (rarer = more specific)
    if len(found) > 1:
        non_nominal = [s for s in found if s != "nominal"]
        if len(non_nominal) == 1:
            return non_nominal[0]
    return None


def parse_final_predictions(
    text: str,
) -> dict[str, str | None]:
    """Parse structured final predictions from model output.

    Returns a dict with keys:
    - ``final_scene``
    - ``final_long_action``
    - ``final_lat_action``
    - ``revised``  (``"yes"`` or ``"no"`` or ``None``)
    - ``reason``

    Values are ``None`` when not parseable.
    """
    result: dict[str, str | None] = {
        "final_scene": None,
        "final_long_action": None,
        "final_lat_action": None,
        "revised": None,
        "reason": None,
    }

    # Scene
    m = _SCENE_PATTERN.search(text)
    if m:
        val = _clean_value(m.group(1))
        if val in VALID_SCENES:
            result["final_scene"] = val

    # Long action
    m = _LONG_ACTION_PATTERN.search(text)
    if m:
        val = _clean_value(m.group(1))
        if val in VALID_LONG_ACTIONS:
            result["final_long_action"] = val

    # Lat action
    m = _LAT_ACTION_PATTERN.search(text)
    if m:
        val = _clean_value(m.group(1))
        if val in VALID_LAT_ACTIONS:
            result["final_lat_action"] = val

    # Revised
    m = _REVISED_PATTERN.search(text)
    if m:
        val = _clean_value(m.group(1))
        if val in ("yes", "no"):
            result["revised"] = val

    # Reason
    m = _REASON_PATTERN.search(text)
    if m:
        result["reason"] = m.group(1).strip()

    # Fallback: try freeform scene extraction
    if result["final_scene"] is None:
        result["final_scene"] = _try_extract_scene_freeform(
            text
        )

    return result
