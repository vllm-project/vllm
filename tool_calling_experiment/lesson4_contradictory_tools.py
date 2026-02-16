#!/usr/bin/env python3
"""Lesson 4: Contradictory tool results experiment.

Multi-turn experiment testing model behavior when
tool results contradict the initial assessment.

For each sample:
  Turn 1: Verification prompt (model calls tools)
  Turn 2: Feed back tool results (contradictory info)
  Turn 3: Model gives final answer, categorized as
    IGNORES / BLINDLY_DEFERS / ACTUALLY_REASONS

Sample categories:
  A) pred=incident_zone, GT=nominal  (15)
  B) pred=incident_zone, GT=incident_zone (5)
  C) pred=nominal, GT=incident_zone (5)
  D) pred=correct (5)
"""

from __future__ import annotations

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

# Ensure sibling modules importable without polluting
# sys.path with the vllm source tree.
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
if os.path.isdir(os.path.join(_PARENT, "vllm")):
    sys.path[:] = [
        p
        for p in sys.path
        if os.path.abspath(p or os.getcwd()) != _PARENT
    ]

from server_utils import VLLMServer  # type: ignore[import-not-found] # noqa: E402

from tools import ALL_TOOLS, execute_tool  # type: ignore[import-not-found] # noqa: E402

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
MODEL_PATH = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_ID = 7
PORT = 8307
SELF_CONSISTENCY_DB = os.path.join(
    _PARENT,
    "self_consistency_experiment",
    "self_consistency.db",
)
RESULTS_PATH = os.path.join(
    _DIR, "lesson4_results.json"
)

# Sample counts per category
N_FALSE_POS = 15  # pred=incident_zone, GT=nominal
N_TRUE_POS = 5  # pred=incident_zone, GT=iz
N_FALSE_NEG = 5  # pred=nominal, GT=incident_zone
N_CORRECT = 5  # pred matches GT (non-incident)

# ---------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a driving scene verification system. "
    "You have access to statistical tools that provide "
    "information about how common different scene types "
    "are, confusion risks, and action compatibility. "
    "Use these tools to verify predictions, then "
    "provide your final assessment.\n\n"
    "Always respond with your final assessment in "
    "this exact format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>\n"
    "REVISED: <yes|no>\n"
    "REASON: <brief explanation of your reasoning>"
)

VERIFICATION_PROMPT = (
    "A driving scene classification model analyzed "
    "a dashcam image and predicted:\n"
    "- Scene: {predicted_scene}\n"
    "- Longitudinal action: {predicted_long_action}\n"
    "- Lateral action: {predicted_lat_action}\n\n"
    "Please verify this prediction using the available "
    "tools before giving your final answer."
)

# ---------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------


def select_samples(
    db_path: str,
) -> list[dict[str, Any]]:
    """Select 30 diverse samples from the DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    samples: list[dict[str, Any]] = []

    # Category A: pred=incident_zone, GT=nominal
    rows = conn.execute(
        "SELECT * FROM predictions "
        "WHERE predicted_scene='incident_zone' "
        "AND scene_type_gt='nominal' "
        "ORDER BY sample_id LIMIT ?",
        (N_FALSE_POS,),
    ).fetchall()
    for r in rows:
        d = dict(r)
        d["category"] = "A_false_positive"
        d["category_desc"] = (
            "pred=incident_zone, GT=nominal "
            "(should revise)"
        )
        samples.append(d)
    print(
        f"Category A (false positive): "
        f"{len(rows)} samples"
    )

    # Category B: pred=incident_zone, GT=incident_zone
    rows = conn.execute(
        "SELECT * FROM predictions "
        "WHERE predicted_scene='incident_zone' "
        "AND scene_type_gt='incident_zone' "
        "ORDER BY sample_id LIMIT ?",
        (N_TRUE_POS,),
    ).fetchall()
    for r in rows:
        d = dict(r)
        d["category"] = "B_true_positive"
        d["category_desc"] = (
            "pred=incident_zone, GT=incident_zone "
            "(should NOT revise)"
        )
        samples.append(d)
    print(
        f"Category B (true positive): "
        f"{len(rows)} samples"
    )

    # Category C: pred=nominal, GT=incident_zone
    rows = conn.execute(
        "SELECT * FROM predictions "
        "WHERE predicted_scene='nominal' "
        "AND scene_type_gt='incident_zone' "
        "ORDER BY sample_id LIMIT ?",
        (N_FALSE_NEG,),
    ).fetchall()
    for r in rows:
        d = dict(r)
        d["category"] = "C_false_negative"
        d["category_desc"] = (
            "pred=nominal, GT=incident_zone "
            "(under-predicted)"
        )
        samples.append(d)
    print(
        f"Category C (false negative): "
        f"{len(rows)} samples"
    )

    # Category D: pred matches GT, non-incident
    rows = conn.execute(
        "SELECT * FROM predictions "
        "WHERE predicted_scene=scene_type_gt "
        "AND predicted_scene != 'incident_zone' "
        "ORDER BY sample_id LIMIT ?",
        (N_CORRECT,),
    ).fetchall()
    for r in rows:
        d = dict(r)
        d["category"] = "D_correct"
        d["category_desc"] = (
            "pred matches GT (tools should confirm)"
        )
        samples.append(d)
    print(
        f"Category D (correct): {len(rows)} samples"
    )

    conn.close()
    print(f"\nTotal samples selected: {len(samples)}")
    return samples


# ---------------------------------------------------------------
# Multi-turn conversation flow
# ---------------------------------------------------------------


def build_turn1_messages(
    sample: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build Turn 1 messages: system + user prompt."""
    user_prompt = VERIFICATION_PROMPT.format(
        predicted_scene=sample["predicted_scene"],
        predicted_long_action=sample[
            "predicted_long_action"
        ],
        predicted_lat_action=sample[
            "predicted_lat_action"
        ],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_tool_calls(
    msg: dict[str, Any],
) -> list[tuple[str, dict[str, Any], str]]:
    """Extract tool calls from assistant message.

    Returns list of (tool_name, arguments_dict, call_id).
    """
    tool_calls = msg.get("tool_calls") or []
    results: list[tuple[str, dict[str, Any], str]] = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        call_id = tc.get("id", "")
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
            results.append((name, args, call_id))
    return results


ToolResultTuple = tuple[
    str, dict[str, Any], str, dict[str, Any]
]


def build_turn2_messages(
    turn1_msgs: list[dict[str, Any]],
    assistant_msg: dict[str, Any],
    tool_results: list[ToolResultTuple],
) -> list[dict[str, Any]]:
    """Build Turn 2: original + assistant + tool results.

    tool_results: list of (name, args, call_id, result).
    """
    msgs = list(turn1_msgs)
    msgs.append(assistant_msg)

    for _tname, _targs, call_id, tresult in tool_results:
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tresult),
            }
        )

    return msgs


def _execute_tools(
    calls: list[tuple[str, dict[str, Any], str]],
) -> list[ToolResultTuple]:
    """Execute tool calls locally and return results."""
    results: list[ToolResultTuple] = []
    for name, args, call_id in calls:
        try:
            res = execute_tool(
                name, args, ground_truth=None
            )
        except Exception as exc:
            res = {"error": str(exc)}
        results.append((name, args, call_id, res))
    return results


def _tool_calls_to_dicts(
    calls: list[tuple[str, dict[str, Any], str]],
) -> list[dict[str, Any]]:
    """Convert tool call tuples to JSON-friendly dicts."""
    return [
        {"name": n, "args": a, "call_id": c}
        for n, a, c in calls
    ]


def _tool_results_to_dicts(
    results: list[ToolResultTuple],
) -> list[dict[str, Any]]:
    """Convert tool result tuples to JSON-friendly dicts."""
    return [
        {"name": n, "args": a, "result": r}
        for n, a, _c, r in results
    ]


def run_multi_turn(
    server: VLLMServer,
    sample: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the full multi-turn flow for one sample.

    Returns a result dict with all turn details.
    """
    result: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "category": sample["category"],
        "category_desc": sample["category_desc"],
        "predicted_scene": sample["predicted_scene"],
        "predicted_long_action": sample[
            "predicted_long_action"
        ],
        "predicted_lat_action": sample[
            "predicted_lat_action"
        ],
        "scene_type_gt": sample["scene_type_gt"],
        "long_action_gt": sample["long_action_gt"],
        "lat_action_gt": sample["lat_action_gt"],
    }

    # -- TURN 1: Ask model to verify with tools --
    turn1_msgs = build_turn1_messages(sample)
    try:
        turn1_resp = server.chat(
            turn1_msgs,
            tools=tools,
            temperature=0,
            max_tokens=1024,
            tool_choice="auto",
        )
    except Exception as exc:
        result["error"] = f"Turn 1 failed: {exc}"
        result["turn1_response"] = None
        result["tool_calls"] = []
        result["tool_results"] = []
        result["final_response"] = None
        result["category_result"] = "ERROR"
        return result

    result["turn1_response"] = turn1_resp

    # Extract tool calls
    calls = extract_tool_calls(turn1_resp)
    result["tool_calls"] = _tool_calls_to_dicts(calls)

    if not calls:
        # Model answered directly without tools
        final_text = turn1_resp.get("content") or ""
        result["tool_results"] = []
        result["final_response"] = final_text
        result["num_turns"] = 1
        result["category_result"] = categorize_response(
            final_text,
            sample,
            tool_results_received=False,
        )
        return result

    # -- Execute tools locally (statistical, not oracle) --
    tool_res = _execute_tools(calls)
    result["tool_results"] = _tool_results_to_dicts(
        tool_res
    )

    # -- TURN 2: Feed tool results back --
    turn2_msgs = build_turn2_messages(
        turn1_msgs, turn1_resp, tool_res
    )

    try:
        turn2_resp = server.chat(
            turn2_msgs,
            tools=tools,
            temperature=0,
            max_tokens=1024,
            tool_choice="auto",
        )
    except Exception as exc:
        result["error"] = f"Turn 2 failed: {exc}"
        result["final_response"] = None
        result["category_result"] = "ERROR"
        return result

    # Check if Turn 2 also has tool calls
    turn2_calls = extract_tool_calls(turn2_resp)
    if turn2_calls:
        # Execute and do Turn 3
        tool_res_2 = _execute_tools(turn2_calls)
        result["tool_calls_turn2"] = (
            _tool_calls_to_dicts(turn2_calls)
        )
        result["tool_results_turn2"] = (
            _tool_results_to_dicts(tool_res_2)
        )

        turn3_msgs = build_turn2_messages(
            turn2_msgs, turn2_resp, tool_res_2
        )

        try:
            turn3_resp = server.chat(
                turn3_msgs,
                tools=tools,
                temperature=0,
                max_tokens=1024,
                tool_choice="none",
            )
            final_text = (
                turn3_resp.get("content") or ""
            )
            result["num_turns"] = 3
        except Exception as exc:
            result["error"] = (
                f"Turn 3 failed: {exc}"
            )
            final_text = ""
            result["num_turns"] = 2
    else:
        final_text = turn2_resp.get("content") or ""
        result["num_turns"] = 2

    result["final_response"] = final_text
    result["category_result"] = categorize_response(
        final_text,
        sample,
        tool_results_received=True,
    )

    return result


# ---------------------------------------------------------------
# Response categorization
# ---------------------------------------------------------------

# Indicators that model engaged with tool output
_REASONING_INDICATORS = [
    "base rate",
    "base_rate",
    "rare",
    "3.7%",
    "3.7 %",
    "false positive",
    "error rate",
    "error_rate",
    "confusion",
    "confused",
    "commonly confused",
    "50.6%",
    "50.6 %",
    "506",
    "however",
    "but",
    "although",
    "despite",
    "looking more carefully",
    "re-evaluat",
    "reconsider",
    "on closer inspection",
    "upon review",
    "the tool",
    "tools indicate",
    "tools suggest",
    "according to",
    "data shows",
    "data suggests",
    "statistical",
    "historically",
    "no evidence",
    "no actual",
    "no emergency",
    "visual triggers",
    "likely nominal",
    "likely actually",
    "could actually be",
    "probably",
    "the low",
    "high rate",
    "high confusion",
    "weigh",
    "balance",
    "consider",
    "on one hand",
    "on the other",
]

_DISMISSAL_INDICATORS = [
    "regardless",
    "nevertheless",
    "still believe",
    "maintain my",
    "stand by",
    "confident in",
    "override",
    "original prediction stands",
]


def parse_final_scene(text: str) -> str | None:
    """Extract FINAL_SCENE from model response."""
    m = re.search(
        r"FINAL_SCENE:\s*(\S+)", text, re.IGNORECASE
    )
    if m:
        return m.group(1).strip().lower()
    scenes = [
        "nominal",
        "incident_zone",
        "flooded",
        "flagger",
        "mounted_police",
    ]
    for scene in scenes:
        if scene in text.lower():
            return scene
    return None


def parse_revised(text: str) -> str | None:
    """Extract REVISED field from model response."""
    m = re.search(
        r"REVISED:\s*(yes|no)", text, re.IGNORECASE
    )
    if m:
        return m.group(1).lower()
    return None


def categorize_response(
    text: str,
    sample: dict[str, Any],
    tool_results_received: bool,
) -> str:
    """Categorize the model response.

    Categories:
      IGNORES - sticks with original despite warnings
      BLINDLY_DEFERS - changes without re-reasoning
      ACTUALLY_REASONS - engages with contradiction
      ERROR - could not parse
    """
    if not text:
        return "ERROR"

    predicted_scene = sample["predicted_scene"]
    final_scene = parse_final_scene(text)

    text_lower = text.lower()

    # No tool results => cannot reason about them
    if not tool_results_received:
        if final_scene == predicted_scene:
            return "IGNORES"
        return "BLINDLY_DEFERS"

    # Did the model change its prediction?
    scene_changed = (
        final_scene is not None
        and final_scene != predicted_scene
    )

    has_reasoning = sum(
        1
        for ind in _REASONING_INDICATORS
        if ind in text_lower
    )
    has_dismissal = sum(
        1
        for ind in _DISMISSAL_INDICATORS
        if ind in text_lower
    )

    # Extract REASON field
    reason_field = ""
    m = re.search(
        r"REASON:\s*(.+?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        reason_field = m.group(1).strip().lower()

    if not scene_changed:
        if has_reasoning >= 2:
            return "ACTUALLY_REASONS"
        if has_dismissal > 0:
            return "IGNORES"
        if has_reasoning == 1 and len(reason_field) > 30:
            return "ACTUALLY_REASONS"
        return "IGNORES"

    # Model changed prediction
    if has_reasoning >= 2:
        return "ACTUALLY_REASONS"
    if len(reason_field) > 40 and has_reasoning >= 1:
        return "ACTUALLY_REASONS"
    return "BLINDLY_DEFERS"


# ---------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------

_SAMPLE_CATS = [
    "A_false_positive",
    "B_true_positive",
    "C_false_negative",
    "D_correct",
]
_CAT_LABELS = {
    "A_false_positive": (
        "A: pred=IZ, GT=nominal (should revise)"
    ),
    "B_true_positive": (
        "B: pred=IZ, GT=IZ (should NOT revise)"
    ),
    "C_false_negative": (
        "C: pred=nom, GT=IZ (under-predicted)"
    ),
    "D_correct": "D: pred=GT (confirm)",
}
_RESP_CATS = [
    "IGNORES",
    "BLINDLY_DEFERS",
    "ACTUALLY_REASONS",
    "ERROR",
]


def _count_by(
    results: list[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    """Count results by a given key."""
    counts: dict[str, int] = {}
    for r in results:
        val = r.get(key, "ERROR")
        counts[val] = counts.get(val, 0) + 1
    return counts


def _scene_changed(r: dict[str, Any]) -> bool:
    """Check if the final scene differs from predicted."""
    final = parse_final_scene(
        r.get("final_response") or ""
    )
    return (
        final is not None
        and final != r["predicted_scene"]
    )


def _scene_correct_change(r: dict[str, Any]) -> bool:
    """Check if the scene change was correct."""
    final = parse_final_scene(
        r.get("final_response") or ""
    )
    return (
        final is not None
        and final != r["predicted_scene"]
        and final == r["scene_type_gt"]
    )


def print_report(
    results: list[dict[str, Any]],
) -> None:
    """Print a detailed report of experiment results."""
    print("\n" + "=" * 70)
    print(
        "LESSON 4: CONTRADICTORY TOOL RESULTS EXPERIMENT"
    )
    print("=" * 70)

    total = len(results)
    cats = _count_by(results, "category_result")

    print(
        f"\n--- Overall Response Categories "
        f"(n={total}) ---"
    )
    for cat in _RESP_CATS:
        count = cats.get(cat, 0)
        pct = count / total * 100 if total else 0
        print(f"  {cat:20s}: {count:3d} ({pct:5.1f}%)")

    # Per sample-category breakdown
    for scat in _SAMPLE_CATS:
        subset = [
            r for r in results if r["category"] == scat
        ]
        if not subset:
            continue
        label = _CAT_LABELS[scat]
        n = len(subset)
        print(f"\n--- {label} (n={n}) ---")

        sub_cats = _count_by(subset, "category_result")
        for cat in _RESP_CATS:
            count = sub_cats.get(cat, 0)
            pct = count / n * 100 if n else 0
            print(
                f"  {cat:20s}: {count:3d} "
                f"({pct:5.1f}%)"
            )

        changed = sum(1 for r in subset if _scene_changed(r))
        print(f"  Scene changed: {changed}/{n}")
        print(f"  Scene kept:    {n - changed}/{n}")

        correct_ch = sum(
            1 for r in subset if _scene_correct_change(r)
        )
        incorrect_ch = changed - correct_ch
        if changed > 0:
            print(
                f"  Correct changes:   "
                f"{correct_ch}/{changed}"
            )
            print(
                f"  Incorrect changes: "
                f"{incorrect_ch}/{changed}"
            )

    # Key questions
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Q1: False positive warning effectiveness
    cat_a = [
        r
        for r in results
        if r["category"] == "A_false_positive"
    ]
    if cat_a:
        changed_a = sum(
            1 for r in cat_a if _scene_changed(r)
        )
        n_a = len(cat_a)
        pct_a = changed_a / n_a * 100
        print(
            "\nQ1: When tools warn about false "
            "positives (pred=IZ, GT=nominal):"
        )
        print(
            f"  Model changed scene: "
            f"{changed_a}/{n_a} ({pct_a:.0f}%)"
        )
        to_nom = sum(
            1
            for r in cat_a
            if parse_final_scene(
                r.get("final_response") or ""
            )
            == "nominal"
        )
        print(
            f"  Changed correctly to nominal: "
            f"{to_nom}/{n_a}"
        )

    # Q2: True positive resilience
    cat_b = [
        r
        for r in results
        if r["category"] == "B_true_positive"
    ]
    if cat_b:
        wrong_b = sum(
            1 for r in cat_b if _scene_changed(r)
        )
        n_b = len(cat_b)
        pct_b = wrong_b / n_b * 100
        print(
            "\nQ2: When model IS correct "
            "(pred=IZ, GT=IZ) - wrongly deferred?"
        )
        print(
            f"  Wrongly changed: "
            f"{wrong_b}/{n_b} ({pct_b:.0f}%)"
        )

    # Q3: Correct prediction handling
    cat_d = [
        r
        for r in results
        if r["category"] == "D_correct"
    ]
    if cat_d:
        d_cats = _count_by(cat_d, "category_result")
        print(
            "\nQ3: When prediction is already "
            "correct (Category D):"
        )
        for cat, count in sorted(d_cats.items()):
            print(f"  {cat}: {count}/{len(cat_d)}")

    # Q4: Overall reasoning quality
    n_reasons = cats.get("ACTUALLY_REASONS", 0)
    pct_r = n_reasons / total * 100 if total else 0
    print("\nQ4: Overall reasoning quality:")
    print(
        f"  Actually reasons: "
        f"{n_reasons}/{total} ({pct_r:.0f}%)"
    )

    # Example responses
    print("\n" + "=" * 70)
    print("EXAMPLE RESPONSES")
    print("=" * 70)

    for resp_cat in _RESP_CATS[:3]:
        examples = [
            r
            for r in results
            if r.get("category_result") == resp_cat
        ]
        if not examples:
            continue
        ex = examples[0]
        pred_str = (
            f"{ex['predicted_scene']}/"
            f"{ex['predicted_long_action']}/"
            f"{ex['predicted_lat_action']}"
        )
        gt_str = (
            f"{ex['scene_type_gt']}/"
            f"{ex['long_action_gt']}/"
            f"{ex['lat_action_gt']}"
        )
        tc_names = [
            tc["name"]
            for tc in ex.get("tool_calls", [])
        ]
        final_text = (
            ex.get("final_response") or "(no response)"
        )
        if len(final_text) > 500:
            final_text = final_text[:500] + "..."

        print(f"\n--- Example: {resp_cat} ---")
        print(
            f"  Sample: {ex['sample_id']} "
            f"({ex['category']})"
        )
        print(f"  Predicted: {pred_str}")
        print(f"  GT: {gt_str}")
        print(f"  Tools called: {tc_names}")
        indented = final_text.replace(
            chr(10), chr(10) + "    "
        )
        print(f"  Final response:\n    {indented}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def _server_is_healthy(port: int) -> bool:
    """Check if a server is already running."""
    import requests  # type: ignore[import-not-found]

    try:
        r = requests.get(
            f"http://localhost:{port}/health",
            timeout=3,
        )
        return r.status_code == 200
    except Exception:
        return False


def main() -> None:
    """Run the Lesson 4 experiment."""
    start_time = time.time()
    owns_server = False

    print("=" * 60)
    print(
        "Lesson 4: Contradictory Tool Results Experiment"
    )
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {GPU_ID}, Port: {PORT}")
    print(f"DB: {SELF_CONSISTENCY_DB}")
    print()

    # 1. Select samples
    print("--- Selecting samples ---")
    samples = select_samples(SELF_CONSISTENCY_DB)
    if not samples:
        print("ERROR: No samples found.")
        return

    # 2. Connect to or start server
    server = VLLMServer(
        model_path=MODEL_PATH,
        port=PORT,
        gpu_id=GPU_ID,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enable_tools=True,
    )

    if _server_is_healthy(PORT):
        print(
            f"\nServer already running on port "
            f"{PORT}, reusing it."
        )
    else:
        print(
            f"\n--- Starting vLLM server on "
            f"GPU {GPU_ID}, port {PORT} ---"
        )
        server.start(timeout=420)
        owns_server = True
        print("Server ready.")

    # 3. Run multi-turn flow for each sample
    n_samples = len(samples)
    print(
        f"\n--- Running multi-turn experiment "
        f"({n_samples} samples) ---"
    )
    all_results: list[dict[str, Any]] = []

    try:
        for i, sample in enumerate(samples):
            sid = sample["sample_id"]
            scat = sample["category"]
            pred = sample["predicted_scene"]
            gt = sample["scene_type_gt"]
            print(
                f"\n[{i + 1}/{n_samples}] "
                f"Sample {sid} ({scat}) "
                f"pred={pred}, GT={gt}"
            )

            result = run_multi_turn(
                server, sample, ALL_TOOLS
            )

            cat_result = result.get(
                "category_result", "ERROR"
            )
            final_scene = parse_final_scene(
                result.get("final_response") or ""
            )
            n_tools = len(
                result.get("tool_calls", [])
            )
            print(
                f"  -> Tools: {n_tools}, "
                f"Final: {final_scene}, "
                f"Cat: {cat_result}"
            )

            all_results.append(result)
    finally:
        # 4. Stop server only if we started it
        if owns_server:
            print("\n--- Stopping server ---")
            server.stop()

    # 5. Save results
    print(
        f"\n--- Saving results to "
        f"{RESULTS_PATH} ---"
    )
    output = {
        "experiment": (
            "lesson4_contradictory_tools"
        ),
        "model": MODEL_PATH,
        "timestamp": datetime.now(
            tz=timezone.utc
        ).isoformat(),
        "num_samples": len(all_results),
        "results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(
            output, f, indent=2, default=str
        )
    print(f"Saved {len(all_results)} results.")

    # 6. Print report
    print_report(all_results)

    elapsed = time.time() - start_time
    print(
        f"\nTotal experiment time: {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
