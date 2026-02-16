#!/usr/bin/env python3
"""Lesson 5: Testing Sequential Tool Chain Coherence.

Tests a 3-step tool chain where each step depends on the previous:
  Step 1: Classify scene -> check_scene_prior + check_confusion_risk
  Step 2: Based on scene, predict action -> check_scene_action_compatibility
  Step 3: Based on action, assess waypoint -> check_waypoint_feasibility

This requires 4 tool calls in sequence across multiple conversation
turns, with the model maintaining context across all of them.

Also tests:
  - Order variation: reverse the chain for 3 samples
  - Contradiction injection: force step 2 tool to say invalid

Usage:
    python tool_calling_experiment/lesson5_tool_chain.py
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any

_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
if os.path.isdir(os.path.join(_PARENT, "vllm")):
    sys.path[:] = [
        p
        for p in sys.path
        if os.path.abspath(p or os.getcwd()) != _PARENT
    ]

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_PATH = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_ID = 4
PORT = 8304
SC_DB_PATH = os.path.join(
    _PARENT,
    "self_consistency_experiment",
    "self_consistency.db",
)
RESULTS_PATH = os.path.join(_DIR, "lesson5_results.json")

VALID_SCENES = {
    "nominal",
    "flooded",
    "incident_zone",
    "mounted_police",
    "flagger",
}
LONG_VALID = {"stop", "slowdown", "proceed", "null"}
LAT_VALID = {"lc_left", "lc_right", "null"}

# ------------------------------------------------------------------
# Scene description templates
# ------------------------------------------------------------------
SCENE_DESCRIPTIONS: dict[str, str] = {
    "nominal_clean": (
        "A clear highway with multiple lanes, no unusual "
        "objects or hazards visible. Traffic is flowing "
        "normally. The road surface is dry and "
        "well-maintained. No construction zones, emergency "
        "vehicles, or obstacles in sight."
    ),
    "nominal_triggers": (
        "A highway scene with some visual triggers present "
        "-- there appear to be traffic cones or barriers on "
        "the shoulder of the road, but traffic continues to "
        "flow. No actual road closures or emergency "
        "situations are visible. The road surface is dry."
    ),
    "incident_zone_1": (
        "A highway scene showing what appears to be an "
        "active incident zone ahead. There are emergency "
        "vehicles with flashing lights visible, and traffic "
        "is being directed around the area. Some lanes "
        "appear blocked. The vehicle ahead is slowing down "
        "and moving to the left."
    ),
    "incident_zone_2": (
        "A road scene with clear signs of an accident or "
        "road closure. Emergency vehicles are present, and "
        "there are multiple cones and barriers blocking the "
        "right lanes. A police car is directing traffic. "
        "The vehicle needs to stop or slow down."
    ),
    "flooded_1": (
        "A road scene where standing water is visible on "
        "the road surface. The water appears to be several "
        "inches deep across the driving lanes. Some "
        "vehicles ahead are driving slowly through the "
        "water. The sky looks overcast."
    ),
    "flooded_2": (
        "A highway with significant water accumulation on "
        "the road. The driving surface is partially "
        "submerged, and spray from vehicles ahead is "
        "visible. The situation suggests recent heavy "
        "rainfall. Visibility is somewhat reduced."
    ),
    "flagger_1": (
        "A road scene with a construction zone ahead. A "
        "human flagger is standing at the edge of the lane "
        "holding a stop/slow sign, directing traffic. "
        "Construction equipment is visible behind the "
        "flagger. The vehicle needs to respond to the "
        "flagger's instructions."
    ),
    "flagger_2": (
        "A work zone where a person in a high-visibility "
        "vest is actively flagging traffic. They are "
        "holding a paddle sign and signaling vehicles to "
        "slow down. Construction activity is happening on "
        "the right side of the road."
    ),
    "mounted_police_1": (
        "A road scene where mounted police officers on "
        "horseback are visible near the roadside. The "
        "horses are walking along the shoulder of the "
        "road. Traffic appears to be flowing normally "
        "around them. No emergency situation is apparent."
    ),
}

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are classifying a driving scene step by step. "
    "At each step, use the available tools to verify your "
    "reasoning before proceeding."
)


# ------------------------------------------------------------------
# Sample selection
# ------------------------------------------------------------------


def select_samples() -> list[dict[str, Any]]:
    """Select 10 diverse samples from the self-consistency DB."""
    conn = sqlite3.connect(SC_DB_PATH)
    conn.row_factory = sqlite3.Row

    samples: list[dict[str, Any]] = []

    def _pick(
        query: str,
        desc_key: str,
        label: str,
    ) -> dict[str, Any]:
        row = conn.execute(query).fetchone()
        if row is None:
            msg = f"No sample found for: {label}"
            raise RuntimeError(msg)
        return {
            "sample_id": row["sample_id"],
            "scene_type_gt": row["scene_type_gt"],
            "fine_class": row["fine_class"],
            "long_action_gt": row["long_action_gt"],
            "lat_action_gt": row["lat_action_gt"],
            "location": row["location"],
            "description": SCENE_DESCRIPTIONS[desc_key],
            "label": label,
        }

    # 1 easy nominal (nominal_clean, correctly predicted)
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='nominal_clean' "
            "AND predicted_scene='nominal' LIMIT 1",
            "nominal_clean",
            "nominal_easy",
        )
    )

    # 2 nominal_triggers (tricky -- some misclassified)
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='nominal_triggers' "
            "AND predicted_scene='incident_zone' LIMIT 1",
            "nominal_triggers",
            "nominal_trigger_1",
        )
    )
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='nominal_triggers' "
            "AND predicted_scene='nominal' "
            "AND sample_id > 0 LIMIT 1",
            "nominal_triggers",
            "nominal_trigger_2",
        )
    )

    # 2 incident_zone (real ones)
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='incident_zone' "
            "AND long_action_gt='slowdown' LIMIT 1",
            "incident_zone_1",
            "incident_zone_1",
        )
    )
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='incident_zone' "
            "AND long_action_gt='stop' LIMIT 1",
            "incident_zone_2",
            "incident_zone_2",
        )
    )

    # 2 flooded
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='flooded' "
            "AND long_action_gt='slowdown' LIMIT 1",
            "flooded_1",
            "flooded_1",
        )
    )
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='flooded' "
            "AND long_action_gt='stop' LIMIT 1",
            "flooded_2",
            "flooded_2",
        )
    )

    # 2 flagger
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='flagger' "
            "AND long_action_gt='stop' LIMIT 1",
            "flagger_1",
            "flagger_1",
        )
    )
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='flagger' "
            "AND long_action_gt='slowdown' LIMIT 1",
            "flagger_2",
            "flagger_2",
        )
    )

    # 1 mounted_police
    samples.append(
        _pick(
            "SELECT * FROM predictions "
            "WHERE fine_class='mounted_police' LIMIT 1",
            "mounted_police_1",
            "mounted_police_1",
        )
    )

    conn.close()
    return samples


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

_SCENE_PATTERNS = [
    re.compile(
        r"(?:SCENE|scene[_ ]type|classification)"
        r"\s*[:=]\s*(\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\*\*Scene[^*]*\*\*\s*[:=]?\s*(\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"scene\s+(?:is|type|classification)"
        r"\s+(?:is\s+)?['\"]?(\w+)",
        re.IGNORECASE,
    ),
]

_LONG_PATTERNS = [
    re.compile(
        r"(?:LONG_ACTION|longitudinal[_ ]action)"
        r"\s*[:=]\s*(\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\*\*(?:Longitudinal|Long)[^*]*\*\*"
        r"\s*[:=]?\s*(\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"longitudinal\s+action"
        r"\s+(?:is|should be)\s+['\"]?(\w+)",
        re.IGNORECASE,
    ),
]

_LAT_PATTERNS = [
    re.compile(
        r"(?:LAT_ACTION|lateral[_ ]action)"
        r"\s*[:=]\s*(\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\*\*(?:Lateral|Lat)[^*]*\*\*"
        r"\s*[:=]?\s*(\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"lateral\s+action"
        r"\s+(?:is|should be)\s+['\"]?(\w+)",
        re.IGNORECASE,
    ),
]

_WP_X_PATTERN = re.compile(
    r"(?:WAYPOINT_X|waypoint.*x|first_waypoint_x)"
    r"\s*[:=]\s*([-\d.]+)",
    re.IGNORECASE,
)
_WP_Y_PATTERN = re.compile(
    r"(?:WAYPOINT_Y|waypoint.*y|first_waypoint_y)"
    r"\s*[:=]\s*([-\d.]+)",
    re.IGNORECASE,
)
_WP_PAIR_PATTERN = re.compile(
    r"\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?"
)


def parse_scene(text: str) -> str | None:
    """Extract scene classification from model text."""
    if not text:
        return None
    strip_chars = ".,;:\"'*`"
    for pat in _SCENE_PATTERNS:
        m = pat.search(text)
        if m:
            val = m.group(1).strip().lower().strip(strip_chars)
            if val in VALID_SCENES:
                return val
    lower = text.lower()
    found = [s for s in VALID_SCENES if s in lower]
    if len(found) == 1:
        return found[0]
    if found:
        non_nominal = [s for s in found if s != "nominal"]
        if len(non_nominal) == 1:
            return non_nominal[0]
    return found[0] if found else None


def parse_actions(
    text: str,
) -> tuple[str | None, str | None]:
    """Extract longitudinal and lateral actions."""
    if not text:
        return None, None
    long_action = None
    lat_action = None
    strip_chars = ".,;:\"'*`"

    for pat in _LONG_PATTERNS:
        m = pat.search(text)
        if m:
            val = m.group(1).strip().lower().strip(strip_chars)
            if val in LONG_VALID:
                long_action = val
                break
    if long_action is None:
        lower = text.lower()
        for act in ["slowdown", "stop", "proceed"]:
            if act in lower:
                long_action = act
                break

    for pat in _LAT_PATTERNS:
        m = pat.search(text)
        if m:
            val = m.group(1).strip().lower().strip(strip_chars)
            if val in LAT_VALID:
                lat_action = val
                break
    if lat_action is None:
        lower = text.lower()
        if "lc_left" in lower:
            lat_action = "lc_left"
        elif "lc_right" in lower:
            lat_action = "lc_right"
        elif "null" in lower:
            lat_action = "null"

    return long_action, lat_action


def parse_waypoint(
    text: str,
) -> tuple[float | None, float | None]:
    """Extract waypoint x, y from model text."""
    if not text:
        return None, None
    wx: float | None = None
    wy: float | None = None

    m = _WP_X_PATTERN.search(text)
    if m:
        with contextlib.suppress(ValueError):
            wx = float(m.group(1))

    m = _WP_Y_PATTERN.search(text)
    if m:
        with contextlib.suppress(ValueError):
            wy = float(m.group(1))

    if wx is None or wy is None:
        m = _WP_PAIR_PATTERN.search(text)
        if m:
            with contextlib.suppress(ValueError):
                wx = wx or float(m.group(1))
                wy = wy or float(m.group(2))

    return wx, wy


# ------------------------------------------------------------------
# Tool call helpers
# ------------------------------------------------------------------


def _get_tools() -> tuple[list[dict[str, Any]], Any]:
    """Import and return (ALL_TOOLS, execute_tool)."""
    from tools import ALL_TOOLS as _all  # type: ignore[import-not-found]
    from tools import execute_tool as _exec  # type: ignore[import-not-found]

    return _all, _exec


def extract_tool_calls(
    msg: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Extract (tool_name, arguments) from a response."""
    tool_calls = msg.get("tool_calls") or []
    results: list[tuple[str, dict[str, Any]]] = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        else:
            args = (
                raw_args
                if isinstance(raw_args, dict)
                else {}
            )
        if name:
            results.append((name, args))
    return results


def execute_and_record_tools(
    msg: dict[str, Any],
    override_results: (
        dict[str, dict[str, Any]] | None
    ) = None,
) -> list[dict[str, Any]]:
    """Execute tool calls and return results with metadata.

    Parameters
    ----------
    msg : dict
        The assistant message containing tool_calls.
    override_results : dict, optional
        Maps tool_name -> forced result dict for
        contradiction injection experiments.
    """
    _, exec_tool = _get_tools()
    calls = extract_tool_calls(msg)
    results: list[dict[str, Any]] = []
    for tname, targs in calls:
        if override_results and tname in override_results:
            result = override_results[tname]
        else:
            result = exec_tool(tname, targs)
        results.append(
            {
                "tool_name": tname,
                "arguments": targs,
                "result": result,
            }
        )
    return results


def build_tool_response_messages(
    conversation: list[dict[str, Any]],
    assistant_msg: dict[str, Any],
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build conversation with tool results appended."""
    msgs = list(conversation)
    msgs.append(assistant_msg)

    atc = assistant_msg.get("tool_calls") or []
    for idx, tr in enumerate(tool_results):
        call_id = ""
        if idx < len(atc):
            call_id = atc[idx].get("id", "")
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tr["result"]),
            }
        )
    return msgs


# ------------------------------------------------------------------
# Turn execution helper
# ------------------------------------------------------------------


def _do_turn(
    server: Any,
    conversation: list[dict[str, Any]],
    all_tools: list[dict[str, Any]],
    override_results: (
        dict[str, dict[str, Any]] | None
    ) = None,
    max_tool_rounds: int = 3,
) -> tuple[
    str,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """Execute one logical turn (with tool-call follow-ups).

    Returns (final_text, tool_records, updated_conversation, errors).
    """
    errors: list[str] = []
    all_tool_records: list[dict[str, Any]] = []
    conv = list(conversation)

    for _round in range(max_tool_rounds):
        try:
            msg = server.chat(
                conv,
                tools=all_tools,
                temperature=0,
                max_tokens=1024,
            )
        except Exception as exc:
            errors.append(str(exc))
            return "", all_tool_records, conv, errors

        tool_results = execute_and_record_tools(
            msg, override_results=override_results
        )
        # Only override on the first round
        override_results = None

        if tool_results:
            all_tool_records.extend(tool_results)
            conv = build_tool_response_messages(
                conv, msg, tool_results
            )
        else:
            conv.append(msg)
            return (
                msg.get("content", ""),
                all_tool_records,
                conv,
                errors,
            )

    # Exhausted tool rounds -- get final text
    try:
        msg = server.chat(
            conv,
            tools=all_tools,
            temperature=0,
            max_tokens=1024,
        )
    except Exception as exc:
        errors.append(str(exc))
        return "", all_tool_records, conv, errors

    conv.append(msg)
    return (
        msg.get("content", ""),
        all_tool_records,
        conv,
        errors,
    )


def _do_final_turn(
    server: Any,
    conversation: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Execute a final turn with no tools."""
    errors: list[str] = []
    try:
        msg = server.chat(
            conversation,
            tools=None,
            temperature=0,
            max_tokens=1024,
            tool_choice="none",
        )
    except Exception as exc:
        errors.append(str(exc))
        return "", errors
    return msg.get("content", ""), errors


# ------------------------------------------------------------------
# Coherence analysis
# ------------------------------------------------------------------


def analyze_coherence(
    trace: dict[str, Any],
) -> dict[str, Any]:
    """Analyze a conversation trace for coherence."""
    analysis: dict[str, Any] = {
        "step1_scene": None,
        "step2_scene_referenced": None,
        "step2_action_long": None,
        "step2_action_lat": None,
        "step3_action_referenced": None,
        "final_scene": None,
        "final_long": None,
        "final_lat": None,
        "coherence_scene_maintained": False,
        "coherence_action_maintained": False,
        "tool_integration_scores": [],
        "backtracking_events": [],
        "chain_break_step": None,
    }

    turns = trace.get("turns", [])
    if not turns:
        return analysis

    # Step 1: What scene was decided
    for turn in turns:
        if turn.get("step") == 1:
            analysis["step1_scene"] = turn.get(
                "parsed_scene"
            )
            break

    # Step 2: scene reference + action
    for turn in turns:
        if turn.get("step") != 2:
            continue
        content = turn.get("assistant_content", "")
        s1 = analysis["step1_scene"]
        if s1 and s1 in (content or "").lower():
            analysis["step2_scene_referenced"] = s1
        analysis["step2_action_long"] = turn.get(
            "parsed_long_action"
        )
        analysis["step2_action_lat"] = turn.get(
            "parsed_lat_action"
        )
        for tc in turn.get("tool_calls_made", []):
            tn = tc["tool_name"]
            if tn == "check_scene_action_compatibility":
                ts = tc["arguments"].get("scene")
                passed = ts == s1
                analysis["tool_integration_scores"].append(
                    {
                        "step": 2,
                        "metric": "scene_matches_step1",
                        "passed": passed,
                        "expected": s1,
                        "got": ts,
                    }
                )
                if not passed:
                    analysis["backtracking_events"].append(
                        {
                            "step": 2,
                            "type": "scene_changed",
                            "from": s1,
                            "to": ts,
                        }
                    )
        break

    # Step 3: action reference
    for turn in turns:
        if turn.get("step") != 3:
            continue
        content = turn.get("assistant_content", "")
        la = analysis["step2_action_long"]
        if la and la in (content or "").lower():
            analysis["step3_action_referenced"] = la
        for tc in turn.get("tool_calls_made", []):
            tn = tc["tool_name"]
            if tn == "check_waypoint_feasibility":
                ts = tc["arguments"].get("scene")
                tl = tc["arguments"].get("long_action")
                s1 = analysis["step1_scene"]
                analysis["tool_integration_scores"].append(
                    {
                        "step": 3,
                        "metric": "scene_in_wp_matches",
                        "passed": ts == s1,
                        "expected": s1,
                        "got": ts,
                    }
                )
                analysis["tool_integration_scores"].append(
                    {
                        "step": 3,
                        "metric": "action_in_wp_matches",
                        "passed": tl == la,
                        "expected": la,
                        "got": tl,
                    }
                )
        break

    # Final answer
    for turn in turns:
        if turn.get("step") == 4:
            analysis["final_scene"] = turn.get(
                "parsed_scene"
            )
            analysis["final_long"] = turn.get(
                "parsed_long_action"
            )
            analysis["final_lat"] = turn.get(
                "parsed_lat_action"
            )
            break

    # Overall coherence
    analysis["coherence_scene_maintained"] = (
        analysis["step1_scene"] is not None
        and analysis["final_scene"]
        == analysis["step1_scene"]
    )
    analysis["coherence_action_maintained"] = (
        analysis["step2_action_long"] is not None
        and analysis["final_long"]
        == analysis["step2_action_long"]
    )

    # Chain break detection
    if analysis["step1_scene"] is None:
        analysis["chain_break_step"] = 1
    elif analysis["step2_action_long"] is None:
        analysis["chain_break_step"] = 2
    elif analysis["final_scene"] is None:
        analysis["chain_break_step"] = 4

    return analysis


# ------------------------------------------------------------------
# Multi-turn chain execution
# ------------------------------------------------------------------


def run_normal_chain(
    server: Any,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Run the normal 4-turn tool chain for a sample."""
    all_tools, _ = _get_tools()

    trace: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "scene_type_gt": sample["scene_type_gt"],
        "mode": "normal",
        "turns": [],
        "errors": [],
    }

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # -- TURN 1: Scene classification --
    turn1_user = (
        f"A dashcam image shows: {sample['description']}"
        "\n\nStep 1: What scene type is this? Before "
        "answering, check the prior probability and "
        "confusion risk for your predicted scene. "
        "Available scene types: nominal, flooded, "
        "incident_zone, mounted_police, flagger.\n\n"
        "After using the tools, respond with:\n"
        "SCENE: <scene_type>"
    )
    conversation.append(
        {"role": "user", "content": turn1_user}
    )

    text1, tools1, conversation, errs1 = _do_turn(
        server, conversation, all_tools
    )

    turn1_data: dict[str, Any] = {
        "step": 1,
        "user_prompt": turn1_user,
        "assistant_content": text1,
        "tool_calls_made": tools1,
        "parsed_scene": parse_scene(text1),
    }
    trace["turns"].append(turn1_data)
    trace["errors"].extend(
        {"step": 1, "error": e} for e in errs1
    )

    step1_scene = turn1_data["parsed_scene"]
    if step1_scene is None:
        trace["errors"].append(
            {
                "step": 1,
                "error": "Could not parse scene",
                "text": text1[:300],
            }
        )
        step1_scene = "nominal"

    # -- TURN 2: Action prediction --
    turn2_user = (
        f"Good. Now based on your scene classification "
        f"of {step1_scene}, what longitudinal and "
        f"lateral action should the vehicle take? Check "
        f"whether your scene-action combination is "
        f"compatible.\n\n"
        f"Longitudinal actions: stop, slowdown, proceed, "
        f"null.\n"
        f"Lateral actions: lc_left, lc_right, null.\n\n"
        f"After using the tools, respond with:\n"
        f"LONG_ACTION: <action>\nLAT_ACTION: <action>"
    )
    conversation.append(
        {"role": "user", "content": turn2_user}
    )

    text2, tools2, conversation, errs2 = _do_turn(
        server, conversation, all_tools
    )

    step2_long, step2_lat = parse_actions(text2)
    turn2_data: dict[str, Any] = {
        "step": 2,
        "user_prompt": turn2_user,
        "assistant_content": text2,
        "tool_calls_made": tools2,
        "parsed_long_action": step2_long,
        "parsed_lat_action": step2_lat,
    }
    trace["turns"].append(turn2_data)
    trace["errors"].extend(
        {"step": 2, "error": e} for e in errs2
    )

    if step2_long is None:
        trace["errors"].append(
            {
                "step": 2,
                "error": "Could not parse long action",
                "text": text2[:300],
            }
        )
        step2_long = "null"
    if step2_lat is None:
        step2_lat = "null"

    # -- TURN 3: Waypoint assessment --
    turn3_user = (
        f"Now assess the waypoint. If the action is "
        f"{step2_long}, check whether a typical first "
        f"waypoint for this scene-action pair is "
        f"feasible.\n\n"
        f"Use the check_waypoint_feasibility tool with "
        f"reasonable waypoint deltas for a {step1_scene} "
        f"scene with {step2_long} action, then report "
        f"your assessment.\n\n"
        f"Respond with:\n"
        f"WAYPOINT_X: <number>\nWAYPOINT_Y: <number>\n"
        f"FEASIBLE: <yes|no>"
    )
    conversation.append(
        {"role": "user", "content": turn3_user}
    )

    text3, tools3, conversation, errs3 = _do_turn(
        server, conversation, all_tools
    )

    wp_x, wp_y = parse_waypoint(text3)
    turn3_data: dict[str, Any] = {
        "step": 3,
        "user_prompt": turn3_user,
        "assistant_content": text3,
        "tool_calls_made": tools3,
        "parsed_waypoint_x": wp_x,
        "parsed_waypoint_y": wp_y,
    }
    trace["turns"].append(turn3_data)
    trace["errors"].extend(
        {"step": 3, "error": e} for e in errs3
    )

    # -- TURN 4: Final complete prediction --
    turn4_user = (
        "Now provide your final complete prediction for "
        "this driving scene, incorporating all the tool "
        "feedback you received across the previous "
        "steps.\n\n"
        "Respond with:\n"
        "SCENE: <scene_type>\n"
        "LONG_ACTION: <longitudinal_action>\n"
        "LAT_ACTION: <lateral_action>\n"
        "WAYPOINT_X: <number>\n"
        "WAYPOINT_Y: <number>\n"
        "REASONING: <brief explanation>"
    )
    conversation.append(
        {"role": "user", "content": turn4_user}
    )

    text4, errs4 = _do_final_turn(server, conversation)

    final_long, final_lat = parse_actions(text4)
    final_wx, final_wy = parse_waypoint(text4)
    turn4_data: dict[str, Any] = {
        "step": 4,
        "user_prompt": turn4_user,
        "assistant_content": text4,
        "parsed_scene": parse_scene(text4),
        "parsed_long_action": final_long,
        "parsed_lat_action": final_lat,
        "parsed_waypoint_x": final_wx,
        "parsed_waypoint_y": final_wy,
    }
    trace["turns"].append(turn4_data)
    trace["errors"].extend(
        {"step": 4, "error": e} for e in errs4
    )

    return trace


def run_reversed_chain(
    server: Any,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Run the chain in reverse: waypoint -> action -> scene."""
    all_tools, _ = _get_tools()

    trace: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "scene_type_gt": sample["scene_type_gt"],
        "mode": "reversed",
        "turns": [],
        "errors": [],
    }

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    desc = sample["description"]

    # -- TURN 1 (reversed): Waypoint first --
    t1_user = (
        f"A dashcam image shows: {desc}\n\n"
        "Step 1: What would be an appropriate first "
        "waypoint (x-delta, y-delta) for this scene? "
        "Use the check_waypoint_feasibility tool with "
        "your estimated scene type and action to "
        "verify.\n\n"
        "Respond with:\n"
        "SCENE_GUESS: <scene_type>\n"
        "ACTION_GUESS: <longitudinal_action>\n"
        "WAYPOINT_X: <number>\nWAYPOINT_Y: <number>"
    )
    conversation.append(
        {"role": "user", "content": t1_user}
    )

    text1, tools1, conversation, errs1 = _do_turn(
        server, conversation, all_tools
    )

    wp_x, wp_y = parse_waypoint(text1)
    long1, _ = parse_actions(text1)
    turn1_data: dict[str, Any] = {
        "step": 1,
        "reversed_step": "waypoint",
        "user_prompt": t1_user,
        "assistant_content": text1,
        "tool_calls_made": tools1,
        "parsed_waypoint_x": wp_x,
        "parsed_waypoint_y": wp_y,
        "parsed_scene": parse_scene(text1),
        "parsed_long_action": long1,
    }
    trace["turns"].append(turn1_data)
    trace["errors"].extend(
        {"step": 1, "error": e} for e in errs1
    )

    g_scene = turn1_data["parsed_scene"] or "nominal"
    g_long = long1 or "null"

    # -- TURN 2 (reversed): Action --
    t2_user = (
        f"Now verify the action. Check whether the "
        f"action {g_long} is compatible with the scene "
        f"{g_scene} using the "
        f"check_scene_action_compatibility tool.\n\n"
        f"Respond with:\n"
        f"LONG_ACTION: <longitudinal_action>\n"
        f"LAT_ACTION: <lateral_action>"
    )
    conversation.append(
        {"role": "user", "content": t2_user}
    )

    text2, tools2, conversation, errs2 = _do_turn(
        server, conversation, all_tools
    )

    long2, lat2 = parse_actions(text2)
    turn2_data: dict[str, Any] = {
        "step": 2,
        "reversed_step": "action",
        "user_prompt": t2_user,
        "assistant_content": text2,
        "tool_calls_made": tools2,
        "parsed_long_action": long2,
        "parsed_lat_action": lat2,
    }
    trace["turns"].append(turn2_data)
    trace["errors"].extend(
        {"step": 2, "error": e} for e in errs2
    )

    # -- TURN 3 (reversed): Scene classification --
    t3_user = (
        "Finally, confirm the scene classification. "
        "Check the prior probability and confusion risk "
        "for your predicted scene using the tools.\n\n"
        "Available scene types: nominal, flooded, "
        "incident_zone, mounted_police, flagger.\n\n"
        "Respond with:\nSCENE: <scene_type>"
    )
    conversation.append(
        {"role": "user", "content": t3_user}
    )

    text3, tools3, conversation, errs3 = _do_turn(
        server, conversation, all_tools
    )

    turn3_data: dict[str, Any] = {
        "step": 3,
        "reversed_step": "scene",
        "user_prompt": t3_user,
        "assistant_content": text3,
        "tool_calls_made": tools3,
        "parsed_scene": parse_scene(text3),
    }
    trace["turns"].append(turn3_data)
    trace["errors"].extend(
        {"step": 3, "error": e} for e in errs3
    )

    # -- TURN 4: Final prediction --
    t4_user = (
        "Now provide your final complete prediction for "
        "this driving scene, incorporating all the tool "
        "feedback.\n\n"
        "Respond with:\n"
        "SCENE: <scene_type>\n"
        "LONG_ACTION: <longitudinal_action>\n"
        "LAT_ACTION: <lateral_action>\n"
        "WAYPOINT_X: <number>\nWAYPOINT_Y: <number>\n"
        "REASONING: <brief explanation>"
    )
    conversation.append(
        {"role": "user", "content": t4_user}
    )

    text4, errs4 = _do_final_turn(server, conversation)
    final_long, final_lat = parse_actions(text4)
    turn4_data: dict[str, Any] = {
        "step": 4,
        "user_prompt": t4_user,
        "assistant_content": text4,
        "parsed_scene": parse_scene(text4),
        "parsed_long_action": final_long,
        "parsed_lat_action": final_lat,
    }
    trace["turns"].append(turn4_data)
    trace["errors"].extend(
        {"step": 4, "error": e} for e in errs4
    )

    return trace


def run_contradiction_chain(
    server: Any,
    sample: dict[str, Any],
) -> dict[str, Any]:
    """Run chain with contradiction injected at step 2.

    Step 2's tool says the scene-action pair is invalid.
    We observe whether the model revises the scene or
    just changes the action.
    """
    all_tools, _ = _get_tools()

    trace: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "scene_type_gt": sample["scene_type_gt"],
        "mode": "contradiction",
        "turns": [],
        "errors": [],
    }

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # -- TURN 1: Same as normal --
    t1_user = (
        f"A dashcam image shows: {sample['description']}"
        "\n\nStep 1: What scene type is this? Before "
        "answering, check the prior probability and "
        "confusion risk for your predicted scene. "
        "Available scene types: nominal, flooded, "
        "incident_zone, mounted_police, flagger.\n\n"
        "After using the tools, respond with:\n"
        "SCENE: <scene_type>"
    )
    conversation.append(
        {"role": "user", "content": t1_user}
    )

    text1, tools1, conversation, errs1 = _do_turn(
        server, conversation, all_tools
    )

    step1_scene = parse_scene(text1)
    turn1_data: dict[str, Any] = {
        "step": 1,
        "user_prompt": t1_user,
        "assistant_content": text1,
        "tool_calls_made": tools1,
        "parsed_scene": step1_scene,
    }
    trace["turns"].append(turn1_data)
    trace["errors"].extend(
        {"step": 1, "error": e} for e in errs1
    )

    if step1_scene is None:
        step1_scene = "nominal"

    # -- TURN 2: With contradiction injection --
    t2_user = (
        f"Good. Now based on your scene classification "
        f"of {step1_scene}, what longitudinal and "
        f"lateral action should the vehicle take? Check "
        f"whether your scene-action combination is "
        f"compatible.\n\n"
        f"Longitudinal actions: stop, slowdown, proceed, "
        f"null.\n"
        f"Lateral actions: lc_left, lc_right, null.\n\n"
        f"After using the tools, respond with:\n"
        f"LONG_ACTION: <action>\nLAT_ACTION: <action>"
    )
    conversation.append(
        {"role": "user", "content": t2_user}
    )

    contradiction_override = {
        "check_scene_action_compatibility": {
            "scene": step1_scene,
            "long_action": "unknown",
            "lat_action": "unknown",
            "observed": False,
            "count": 0,
            "rate": 0.0,
            "typical_actions": [],
            "warning": (
                "WARNING: No valid action combinations "
                f"have ever been observed for "
                f"{step1_scene}. This scene "
                "classification may be incorrect. "
                "Consider re-evaluating whether this "
                f"is truly a {step1_scene} scene."
            ),
        }
    }

    text2, tools2, conversation, errs2 = _do_turn(
        server,
        conversation,
        all_tools,
        override_results=contradiction_override,
    )

    step2_long, step2_lat = parse_actions(text2)
    revised_scene = parse_scene(text2)
    turn2_data: dict[str, Any] = {
        "step": 2,
        "user_prompt": t2_user,
        "assistant_content": text2,
        "tool_calls_made": tools2,
        "parsed_long_action": step2_long,
        "parsed_lat_action": step2_lat,
        "contradiction_injected": True,
        "revised_scene_after_contradiction": revised_scene,
        "scene_revision_detected": (
            revised_scene is not None
            and revised_scene != step1_scene
        ),
    }
    trace["turns"].append(turn2_data)
    trace["errors"].extend(
        {"step": 2, "error": e} for e in errs2
    )

    # -- TURN 3: Final prediction after contradiction --
    t3_user = (
        "The tool indicated a problem. Please provide "
        "your final complete prediction, revising as "
        "necessary.\n\n"
        "Respond with:\n"
        "SCENE: <scene_type>\n"
        "LONG_ACTION: <longitudinal_action>\n"
        "LAT_ACTION: <lateral_action>\n"
        "REASONING: <explanation of what you revised>"
    )
    conversation.append(
        {"role": "user", "content": t3_user}
    )

    text3, errs3 = _do_final_turn(server, conversation)

    final_scene = parse_scene(text3)
    final_long, final_lat = parse_actions(text3)
    turn3_data: dict[str, Any] = {
        "step": 3,
        "user_prompt": t3_user,
        "assistant_content": text3,
        "parsed_scene": final_scene,
        "parsed_long_action": final_long,
        "parsed_lat_action": final_lat,
    }
    trace["turns"].append(turn3_data)
    trace["errors"].extend(
        {"step": 3, "error": e} for e in errs3
    )

    # Determine revision strategy
    scene_changed = (
        final_scene != step1_scene
        if final_scene
        else None
    )
    action_changed = (
        final_long != step2_long
        if final_long and step2_long
        else None
    )

    if final_scene and final_scene != step1_scene:
        strategy = "revised_scene"
    elif final_long and final_long != step2_long:
        strategy = "revised_action"
    else:
        strategy = "no_revision"

    trace["contradiction_response"] = {
        "original_scene": step1_scene,
        "final_scene": final_scene,
        "scene_changed": scene_changed,
        "action_changed": action_changed,
        "strategy": strategy,
    }

    return trace


# ------------------------------------------------------------------
# Printing / summary
# ------------------------------------------------------------------


def print_trace(
    trace: dict[str, Any],
    verbose: bool = True,
) -> None:
    """Print a readable conversation trace."""
    mode = trace.get("mode", "normal")
    label = trace.get("label", "unknown")
    gt = trace.get("scene_type_gt", "unknown")
    sid = trace.get("sample_id", -1)

    print(f"\n{'=' * 70}")
    hdr = f"Sample {sid} | {label} | GT: {gt} | Mode: {mode}"
    print(hdr)
    print(f"{'=' * 70}")

    for turn in trace.get("turns", []):
        step = turn.get("step", "?")
        rev = turn.get("reversed_step", "")
        step_label = f"Step {step}"
        if rev:
            step_label += f" ({rev})"
        print(f"\n--- {step_label} ---")

        tool_calls = turn.get("tool_calls_made", [])
        if tool_calls:
            print(f"  Tool calls ({len(tool_calls)}):")
            for tc in tool_calls:
                tn = tc.get("tool_name", "?")
                ta = tc.get("arguments", {})
                tr = tc.get("result", {})
                args_str = json.dumps(ta)
                print(f"    -> {tn}({args_str})")
                warn = tr.get("warning")
                if warn:
                    print(f"       WARNING: {warn}")
                obs = tr.get("observed")
                if obs is not None:
                    print(f"       observed: {obs}")
                feas = tr.get("feasible")
                if feas is not None:
                    print(f"       feasible: {feas}")

        parsed = []
        if turn.get("parsed_scene"):
            parsed.append(f"scene={turn['parsed_scene']}")
        if turn.get("parsed_long_action"):
            val = turn["parsed_long_action"]
            parsed.append(f"long={val}")
        if turn.get("parsed_lat_action"):
            val = turn["parsed_lat_action"]
            parsed.append(f"lat={val}")
        if turn.get("parsed_waypoint_x") is not None:
            val = turn["parsed_waypoint_x"]
            parsed.append(f"wp_x={val}")
        if turn.get("parsed_waypoint_y") is not None:
            val = turn["parsed_waypoint_y"]
            parsed.append(f"wp_y={val}")
        if parsed:
            print(f"  Parsed: {', '.join(parsed)}")

        if turn.get("contradiction_injected"):
            print("  ** CONTRADICTION INJECTED **")
            if turn.get("scene_revision_detected"):
                rev_s = turn.get(
                    "revised_scene_after_contradiction"
                )
                print(f"  Scene revised to: {rev_s}")

        if verbose:
            content = turn.get("assistant_content", "")
            if content:
                lines = content.strip().split("\n")
                disp = "\n    ".join(lines[:10])
                if len(lines) > 10:
                    extra = len(lines) - 10
                    disp += f"\n    ... ({extra} more lines)"
                print(f"  Response:\n    {disp}")

    errors = trace.get("errors", [])
    if errors:
        print("\n  ERRORS:")
        for e in errors:
            s = e.get("step")
            msg = e.get("error", "")[:200]
            print(f"    Step {s}: {msg}")

    if "contradiction_response" in trace:
        cr = trace["contradiction_response"]
        print("\n  Contradiction Response:")
        print(f"    Original scene: {cr.get('original_scene')}")
        print(f"    Final scene: {cr.get('final_scene')}")
        print(f"    Strategy: {cr.get('strategy')}")


def print_summary(results: dict[str, Any]) -> None:
    """Print overall experiment summary."""
    print("\n" + "=" * 70)
    print("LESSON 5 SUMMARY: Sequential Tool Chain Coherence")
    print("=" * 70)

    normal = results.get("normal_chains", [])
    if normal:
        n = len(normal)
        print(f"\n--- Normal Chain ({n} samples) ---")
        n_sc = 0
        n_ac = 0
        n_ti_pass = 0
        n_ti_total = 0
        n_bt = 0
        chain_breaks: list[Any] = []

        for t in normal:
            a = t.get("analysis", {})
            if a.get("coherence_scene_maintained"):
                n_sc += 1
            if a.get("coherence_action_maintained"):
                n_ac += 1
            for ti in a.get("tool_integration_scores", []):
                n_ti_total += 1
                if ti.get("passed"):
                    n_ti_pass += 1
            n_bt += len(a.get("backtracking_events", []))
            cb = a.get("chain_break_step")
            if cb is not None:
                chain_breaks.append(cb)

        print(f"  Scene coherence (step1==final): {n_sc}/{n}")
        print(f"  Action coherence (step2==final): {n_ac}/{n}")
        ti_r = (
            n_ti_pass / n_ti_total if n_ti_total > 0 else 0
        )
        print(
            f"  Tool integration: "
            f"{n_ti_pass}/{n_ti_total} ({ti_r:.0%})"
        )
        print(f"  Backtracking events: {n_bt}")
        if chain_breaks:
            print(f"  Chain breaks at steps: {chain_breaks}")
        else:
            print("  Chain breaks: none (all completed)")

        print("\n  Per-sample results:")
        for t in normal:
            a = t.get("analysis", {})
            lbl = t.get("label", "?")
            gt = t.get("scene_type_gt", "?")
            s1 = a.get("step1_scene", "?")
            sf = a.get("final_scene", "?")
            lf = a.get("final_long", "?")
            s_ok = (
                "Y"
                if a.get("coherence_scene_maintained")
                else "N"
            )
            a_ok = (
                "Y"
                if a.get("coherence_action_maintained")
                else "N"
            )
            cor = "Y" if sf == gt else "N"
            print(
                f"    {lbl:25s} | GT={gt:16s} "
                f"| s1={s1!s:16s} | final={sf!s:16s} "
                f"| long={lf!s:10s} | coh={s_ok}/{a_ok}"
                f" | ok={cor}"
            )

    rev = results.get("reversed_chains", [])
    if rev:
        print(f"\n--- Reversed Chain ({len(rev)} samples) ---")
        for t in rev:
            lbl = t.get("label", "?")
            turns = t.get("turns", [])
            ft = turns[-1] if turns else {}
            fs = ft.get("parsed_scene", "?")
            gt = t.get("scene_type_gt", "?")
            nt = sum(
                len(tn.get("tool_calls_made", []))
                for tn in turns
            )
            cor = "Y" if fs == gt else "N"
            print(
                f"    {lbl:25s} | GT={gt:16s} "
                f"| final={fs!s:16s} | tools={nt} "
                f"| ok={cor}"
            )

    contra = results.get("contradiction_chains", [])
    if contra:
        n_c = len(contra)
        print(f"\n--- Contradiction ({n_c} samples) ---")
        strats: dict[str, int] = {
            "revised_scene": 0,
            "revised_action": 0,
            "no_revision": 0,
        }
        for t in contra:
            cr = t.get("contradiction_response", {})
            s = cr.get("strategy", "no_revision")
            strats[s] = strats.get(s, 0) + 1
            lbl = t.get("label", "?")
            orig = cr.get("original_scene", "?")
            final = cr.get("final_scene", "?")
            print(
                f"    {lbl:25s} | orig={orig!s:16s} "
                f"| final={final!s:16s} | {s}"
            )
        print(f"\n  Strategies: {json.dumps(strats)}")

    if rev and normal:
        print("\n--- Order Effect Analysis ---")
        rev_labels = {t["label"] for t in rev}
        for nt in normal:
            if nt["label"] not in rev_labels:
                continue
            rt = next(
                t for t in rev if t["label"] == nt["label"]
            )
            n_final = nt.get("analysis", {}).get(
                "final_scene", "?"
            )
            r_turns = rt.get("turns", [])
            r_final = (
                r_turns[-1].get("parsed_scene", "?")
                if r_turns
                else "?"
            )
            gt = nt["scene_type_gt"]
            same = (
                "SAME" if n_final == r_final else "DIFFERENT"
            )
            print(
                f"    {nt['label']:25s} | "
                f"normal={n_final!s:16s} | "
                f"reversed={r_final!s:16s} | "
                f"GT={gt:16s} | {same}"
            )


# ------------------------------------------------------------------
# JSON cleaning
# ------------------------------------------------------------------


def _clean_for_json(obj: Any) -> Any:
    """Strip non-serializable objects for JSON output."""
    if isinstance(obj, dict):
        return {
            k: _clean_for_json(v)
            for k, v in obj.items()
            if k != "assistant_raw"
        }
    if isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    if isinstance(
        obj, (str, int, float, bool, type(None))
    ):
        return obj
    return str(obj)


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------


def _ensure_healthy(server: Any, label: str) -> None:
    """Check server health; restart if needed."""
    import requests as _req  # type: ignore[import-not-found]

    for attempt in range(3):
        try:
            r = _req.get(
                f"{server.base_url}/health", timeout=5
            )
            if r.status_code == 200:
                return
        except Exception:
            pass
        print(
            f"  Server unhealthy before {label} "
            f"(attempt {attempt + 1}/3), restarting..."
        )
        server.restart(timeout=420)

    print("  WARNING: server may still be unhealthy")


def main() -> None:
    """Run the full lesson 5 experiment."""
    from server_utils import VLLMServer  # type: ignore[import-not-found]

    print("=" * 70)
    print("Lesson 5: Sequential Tool Chain Coherence")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {GPU_ID}, Port: {PORT}")
    ts = datetime.now(tz=timezone.utc).isoformat()
    print(f"Time: {ts}")

    # Select samples
    print("\nSelecting samples...")
    samples = select_samples()
    print(f"Selected {len(samples)} samples:")
    for s in samples:
        lbl = s["label"]
        gt = s["scene_type_gt"]
        fc = s["fine_class"]
        print(f"  {lbl:25s} | GT={gt:16s} | fine={fc}")

    # Start server (force V0 engine for Qwen3-VL stability)
    os.environ["VLLM_USE_V1"] = "0"
    print(
        f"\nStarting vLLM server on GPU {GPU_ID}, "
        f"port {PORT} (V0 engine)..."
    )
    server = VLLMServer(
        model_path=MODEL_PATH,
        port=PORT,
        gpu_id=GPU_ID,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_tools=True,
    )

    results: dict[str, Any] = {
        "experiment": "lesson5_tool_chain_coherence",
        "model": MODEL_PATH,
        "timestamp": ts,
        "samples": list(samples),
        "normal_chains": [],
        "reversed_chains": [],
        "contradiction_chains": [],
    }

    try:
        server.start()

        # -- Phase 1: Normal chains for all 10 --
        print("\n" + "=" * 70)
        print("PHASE 1: Normal Chain (all 10 samples)")
        print("=" * 70)

        for i, sample in enumerate(samples):
            lbl = sample["label"]
            _ensure_healthy(server, f"normal/{lbl}")
            print(f"\n[{i + 1}/10] Normal chain: {lbl}...")
            t0 = time.time()
            trace = run_normal_chain(server, sample)
            elapsed = time.time() - t0
            trace["wall_time_s"] = round(elapsed, 1)
            trace["analysis"] = analyze_coherence(trace)
            results["normal_chains"].append(trace)
            print_trace(trace, verbose=False)
            print(f"  (completed in {elapsed:.1f}s)")

        # -- Phase 2: Reversed chains for 3 --
        print("\n" + "=" * 70)
        print("PHASE 2: Reversed Chain (3 samples)")
        print("=" * 70)

        # nominal_trigger_1, incident_zone_1, flooded_1
        rev_indices = [1, 3, 5]
        for idx in rev_indices:
            sample = samples[idx]
            lbl = sample["label"]
            _ensure_healthy(server, f"reversed/{lbl}")
            print(f"\nReversed chain: {lbl}...")
            t0 = time.time()
            trace = run_reversed_chain(server, sample)
            elapsed = time.time() - t0
            trace["wall_time_s"] = round(elapsed, 1)
            results["reversed_chains"].append(trace)
            print_trace(trace, verbose=False)
            print(f"  (completed in {elapsed:.1f}s)")

        # -- Phase 3: Contradiction injection for 3 --
        print("\n" + "=" * 70)
        print("PHASE 3: Contradiction Injection (3 samples)")
        print("=" * 70)

        # nominal_trigger_2, flagger_1, mounted_police_1
        contra_indices = [2, 7, 9]
        for idx in contra_indices:
            sample = samples[idx]
            lbl = sample["label"]
            _ensure_healthy(
                server, f"contradiction/{lbl}"
            )
            print(f"\nContradiction chain: {lbl}...")
            t0 = time.time()
            trace = run_contradiction_chain(
                server, sample
            )
            elapsed = time.time() - t0
            trace["wall_time_s"] = round(elapsed, 1)
            results["contradiction_chains"].append(trace)
            print_trace(trace, verbose=False)
            print(f"  (completed in {elapsed:.1f}s)")

    finally:
        print("\nStopping server...")
        server.stop()

    # Save results
    clean = _clean_for_json(results)
    print(f"\nSaving results to {RESULTS_PATH}")
    with open(RESULTS_PATH, "w") as f:
        json.dump(clean, f, indent=2)

    # Print summary
    print_summary(results)

    # Print 2 full example traces (verbose)
    print("\n" + "=" * 70)
    print("EXAMPLE TRACES (verbose)")
    print("=" * 70)

    if results["normal_chains"]:
        print("\n--- Example 1: Normal chain ---")
        print_trace(results["normal_chains"][0], verbose=True)

    if results["contradiction_chains"]:
        print("\n--- Example 2: Contradiction chain ---")
        print_trace(
            results["contradiction_chains"][0], verbose=True
        )


if __name__ == "__main__":
    main()
