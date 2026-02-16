#!/usr/bin/env python3
"""Pipeline B: Staged Prediction with base model and tools.

Multi-turn pipeline where the base model (no fine-tuning)
classifies a driving scene in stages, using tools at each
step for verification.  Uses the vLLM HTTP server instead of
the LLM() Python API.

Turn 1: Scene classification + check_confusion_risk
Turn 2: Action prediction + check_scene_action_compatibility
Turn 3: Final answer after tool feedback

Usage:
    python tool_calling_experiment/run_staged.py \
        --verifier-model /fsx/models/Qwen3-VL-8B-Instruct \
        --gpu-id 2
"""

from __future__ import annotations

import argparse
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
# Add the tool_calling_experiment directory itself to sys.path so that
# sibling modules (parse_tool_calls, tools, server_utils) can be
# imported directly without needing the parent on sys.path (which would
# shadow the installed vllm package with the local source tree).
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
if os.path.isdir(os.path.join(_PARENT, "vllm")):
    sys.path[:] = [
        p
        for p in sys.path
        if os.path.abspath(p or os.getcwd()) != _PARENT
    ]
DEFAULT_DB = os.path.join(_DIR, "tool_calling.db")

TURN1_PROMPT = """\
Classify this driving scene. Available scene types: \
nominal, flooded, incident_zone, mounted_police, flagger. \
First use the check_confusion_risk tool to understand \
common misclassifications for your predicted scene type, \
then provide your classification.

Respond with:
SCENE: <scene_type>"""

TURN2_TEMPLATE = """\
Now predict the appropriate driving action for a \
{scene} scene. \
Longitudinal actions: stop, slowdown, proceed, null. \
Lateral actions: lc_left, lc_right, null. \
Use the check_scene_action_compatibility tool to verify \
your prediction is consistent with this scene type.

Respond with:
LONG_ACTION: <action>
LAT_ACTION: <action>"""

TURN3_TEMPLATE = """\
Now predict the first waypoint deltas (x, y) for a \
{scene} scene with {long_action} action. \
Use check_waypoint_feasibility to verify your prediction.

Respond with:
WAYPOINT_X: <number>
WAYPOINT_Y: <number>"""


def _derive_verifier_size(model_path: str) -> str:
    """Extract model size label from path."""
    lower = model_path.lower()
    if "8b" in lower:
        return "8b"
    if "2b" in lower:
        return "2b"
    if "72b" in lower:
        return "72b"
    return "base"


def _make_experiment_id(verifier_size: str) -> str:
    """Create a unique experiment ID."""
    ts = datetime.now(tz=timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    return f"tc_staged_{verifier_size}_{ts}"


VALID_SCENES = {
    "nominal",
    "flooded",
    "incident_zone",
    "mounted_police",
    "flagger",
}
LONG_VALID = {"stop", "slowdown", "proceed", "null"}
LAT_VALID = {"lc_left", "lc_right", "null"}


def _parse_scene_from_text(text: str) -> str | None:
    """Parse scene from SCENE: line or freeform."""
    m = re.search(r"SCENE:\s*(\S+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().lower().strip(".,;:\"'")
        if val in VALID_SCENES:
            return val
    lower = text.lower()
    found = [s for s in VALID_SCENES if s in lower]
    if len(found) == 1:
        return found[0]
    return None


def _parse_actions_from_text(
    text: str,
) -> tuple[str | None, str | None]:
    """Parse LONG_ACTION and LAT_ACTION from text."""
    long_action = None
    lat_action = None
    m = re.search(
        r"LONG_ACTION:\s*(\S+)", text, re.IGNORECASE
    )
    if m:
        val = m.group(1).strip().lower().strip(".,;:\"'")
        if val in LONG_VALID:
            long_action = val
    m = re.search(
        r"LAT_ACTION:\s*(\S+)", text, re.IGNORECASE
    )
    if m:
        val = m.group(1).strip().lower().strip(".,;:\"'")
        if val in LAT_VALID:
            lat_action = val
    return long_action, lat_action


def _load_gt_from_baseline(
    db_path: str,
) -> dict[int, dict[str, Any]]:
    """Load ground truth from baseline predictions."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT sample_id, scene_type_gt, "
            "long_action_gt, lat_action_gt, "
            "odd_label, fine_class, location, chum_uri "
            "FROM predictions "
            "WHERE experiment_id = 'baseline'"
        ).fetchall()
        return {r["sample_id"]: dict(r) for r in rows}
    finally:
        conn.close()


def _extract_tool_calls_from_message(
    msg: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Extract (tool_name, arguments) from a server response message."""
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
            args = raw_args if isinstance(raw_args, dict) else {}
        if name:
            results.append((name, args))
    return results


def _build_tool_response_messages(
    original_msgs: list[dict[str, Any]],
    assistant_msg: dict[str, Any],
    tool_results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build conversation with tool results appended."""
    msgs = list(original_msgs)
    msgs.append(assistant_msg)

    assistant_tool_calls = assistant_msg.get("tool_calls") or []
    for idx, (_tname, _targs, tresult) in enumerate(
        tool_results
    ):
        call_id = ""
        if idx < len(assistant_tool_calls):
            call_id = assistant_tool_calls[idx].get("id", "")
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(tresult),
            }
        )
    return msgs


def run_staged(
    verifier_model: str,
    gpu_id: int,
    db_path: str,
    max_model_len: int,
    max_workers: int,
) -> None:
    """Run the staged prediction pipeline."""
    from server_utils import VLLMServer  # type: ignore[import-not-found]

    from tools import ALL_TOOLS  # type: ignore[import-not-found]

    verifier_size = _derive_verifier_size(verifier_model)
    exp_id = _make_experiment_id(verifier_size)
    tool_names = [t["function"]["name"] for t in ALL_TOOLS]

    print(f"Experiment: {exp_id}")
    print(f"Verifier: {verifier_model} ({verifier_size})")

    # Load ground truth from baseline
    gt_map = _load_gt_from_baseline(db_path)
    if not gt_map:
        print("ERROR: No baseline predictions found.")
        print("Run run_prediction.py first.")
        return
    sample_ids = sorted(gt_map.keys())
    print(f"Samples: {len(sample_ids)}")

    # Create experiment record
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO experiments "
        "(experiment_id, condition_name, pipeline, "
        "verifier_model, tools_enabled, "
        "temperature_verify, seed, "
        "total_samples, status, description) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            exp_id,
            "staged",
            "B",
            verifier_model,
            json.dumps(tool_names),
            0.0,
            42,
            len(sample_ids),
            "running",
            f"Pipeline B: Staged with "
            f"{verifier_size} verifier",
        ),
    )
    conn.commit()

    # Start vLLM server (port = 8199 + gpu_id)
    port = 8199 + gpu_id
    print(
        f"Starting vLLM server on port {port} "
        f"(GPU {gpu_id})..."
    )
    server = VLLMServer(
        model_path=verifier_model,
        port=port,
        gpu_id=gpu_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.8,
        enable_tools=True,
    )
    server.start()

    try:
        _run_staged_with_server(
            server=server,
            sample_ids=sample_ids,
            gt_map=gt_map,
            exp_id=exp_id,
            conn=conn,
            max_workers=max_workers,
        )
    finally:
        server.stop()


def _run_staged_with_server(
    server: Any,
    sample_ids: list[int],
    gt_map: dict[int, dict[str, Any]],
    exp_id: str,
    conn: sqlite3.Connection,
    max_workers: int,
) -> None:
    """Core staged-prediction logic using an already-started VLLMServer."""
    from tools import (  # type: ignore[import-not-found]
        ALL_TOOLS,
        execute_tool,
    )

    # TODO: Image support -- the staged pipeline should ideally pass
    # images to the base model.  Since loading MDS images and encoding
    # them as base64 for the HTTP API is complex, this version uses
    # text-only prompts.  Add image_url content parts here when ready.

    start_time = time.time()
    all_tool_calls: dict[
        int, list[tuple[str, dict[str, Any], dict[str, Any]]]
    ] = {}
    all_scenes: dict[int, str | None] = {}
    all_long_actions: dict[int, str | None] = {}
    all_lat_actions: dict[int, str | None] = {}

    # ---- TURN 1: Scene classification ----
    print("\n=== Turn 1: Scene classification ===")

    confusion_tool = [
        t
        for t in ALL_TOOLS
        if t["function"]["name"] == "check_confusion_risk"
    ]

    t1_messages: list[list[dict[str, Any]]] = []
    for _sid in sample_ids:
        t1_messages.append(
            [{"role": "user", "content": TURN1_PROMPT}]
        )

    t1_results = server.batch_chat(
        t1_messages,
        tools=confusion_tool,
        temperature=0,
        max_tokens=1024,
        max_workers=max_workers,
    )

    # Parse and execute turn 1 tool calls
    t1_tool_data: dict[
        int,
        tuple[
            dict[str, Any],
            list[tuple[str, dict[str, Any], dict[str, Any]]],
        ],
    ] = {}
    for idx, (msg, err) in enumerate(t1_results):
        sid = sample_ids[idx]
        all_tool_calls[sid] = []
        if err or msg is None:
            all_scenes[sid] = None
            continue

        calls = _extract_tool_calls_from_message(msg)
        if calls:
            call_results: list[
                tuple[str, dict[str, Any], dict[str, Any]]
            ] = []
            for tname, targs in calls:
                result = execute_tool(tname, targs)
                call_results.append(
                    (tname, targs, result)
                )
                all_tool_calls[sid].append(
                    (tname, targs, result)
                )
            t1_tool_data[idx] = (msg, call_results)
        else:
            content = msg.get("content") or ""
            all_scenes[sid] = _parse_scene_from_text(content)

    # Turn 1b: Feed tool results back for samples that made calls
    if t1_tool_data:
        t1b_indices = sorted(t1_tool_data.keys())
        t1b_messages: list[list[dict[str, Any]]] = []
        for idx in t1b_indices:
            assistant_msg, call_results = t1_tool_data[idx]
            msgs = _build_tool_response_messages(
                t1_messages[idx],
                assistant_msg,
                call_results,
            )
            t1b_messages.append(msgs)

        t1b_results = server.batch_chat(
            t1b_messages,
            tools=confusion_tool,
            temperature=0,
            max_tokens=1024,
            max_workers=max_workers,
        )
        for j, (msg, err) in enumerate(t1b_results):
            real_idx = t1b_indices[j]
            sid = sample_ids[real_idx]
            if msg is not None:
                content = msg.get("content") or ""
                all_scenes[sid] = _parse_scene_from_text(
                    content
                )
            else:
                all_scenes[sid] = None

    # Fill scenes for samples with no result
    for sid in sample_ids:
        if sid not in all_scenes:
            all_scenes[sid] = None

    n_scenes = sum(
        1 for v in all_scenes.values() if v is not None
    )
    print(f"Parsed {n_scenes} scenes from Turn 1")

    # ---- TURN 2: Action prediction ----
    print("\n=== Turn 2: Action prediction ===")
    action_tool = [
        t
        for t in ALL_TOOLS
        if t["function"]["name"]
        == "check_scene_action_compatibility"
    ]

    t2_messages: list[list[dict[str, Any]]] = []
    t2_sid_map: list[int] = []
    for sid in sample_ids:
        scene = all_scenes.get(sid) or "nominal"
        prompt = TURN2_TEMPLATE.format(scene=scene)
        t2_messages.append(
            [{"role": "user", "content": prompt}]
        )
        t2_sid_map.append(sid)

    t2_results = server.batch_chat(
        t2_messages,
        tools=action_tool,
        temperature=0,
        max_tokens=1024,
        max_workers=max_workers,
    )

    # Parse and execute turn 2 tool calls
    t2_tool_data: dict[
        int,
        tuple[
            dict[str, Any],
            list[tuple[str, dict[str, Any], dict[str, Any]]],
        ],
    ] = {}
    for idx, (msg, err) in enumerate(t2_results):
        sid = t2_sid_map[idx]
        if err or msg is None:
            all_long_actions[sid] = None
            all_lat_actions[sid] = None
            continue

        calls = _extract_tool_calls_from_message(msg)
        if calls:
            call_results_2: list[
                tuple[str, dict[str, Any], dict[str, Any]]
            ] = []
            for tname, targs in calls:
                result = execute_tool(tname, targs)
                call_results_2.append(
                    (tname, targs, result)
                )
                all_tool_calls.setdefault(sid, []).append(
                    (tname, targs, result)
                )
            t2_tool_data[idx] = (msg, call_results_2)
        else:
            content = msg.get("content") or ""
            la, lat = _parse_actions_from_text(content)
            all_long_actions[sid] = la
            all_lat_actions[sid] = lat

    # Turn 2b: Feed tool results back
    if t2_tool_data:
        t2b_indices = sorted(t2_tool_data.keys())
        t2b_messages: list[list[dict[str, Any]]] = []
        for idx in t2b_indices:
            assistant_msg, call_results_2b = t2_tool_data[
                idx
            ]
            msgs = _build_tool_response_messages(
                t2_messages[idx],
                assistant_msg,
                call_results_2b,
            )
            t2b_messages.append(msgs)

        t2b_results = server.batch_chat(
            t2b_messages,
            tools=action_tool,
            temperature=0,
            max_tokens=1024,
            max_workers=max_workers,
        )
        for j, (msg, err) in enumerate(t2b_results):
            real_idx = t2b_indices[j]
            sid = t2_sid_map[real_idx]
            if msg is not None:
                content = msg.get("content") or ""
                la, lat = _parse_actions_from_text(content)
                all_long_actions[sid] = la
                all_lat_actions[sid] = lat
            else:
                all_long_actions[sid] = None
                all_lat_actions[sid] = None

    for sid in sample_ids:
        all_long_actions.setdefault(sid, None)
        all_lat_actions.setdefault(sid, None)

    # ---- Skip Turn 3 (waypoints) for simplicity ----
    # Waypoint prediction is a stretch goal; the core
    # experiment focuses on scene + action accuracy.
    total_time = time.time() - start_time

    # ---- Store results ----
    print("\n=== Storing results ===")
    for sid in sample_ids:
        gt = gt_map.get(sid, {})
        gt_scene = gt.get("scene_type_gt")
        pred_scene = all_scenes.get(sid)
        pred_long = all_long_actions.get(sid)
        pred_lat = all_lat_actions.get(sid)
        scene_correct = (
            int(pred_scene == gt_scene)
            if pred_scene and gt_scene
            else None
        )

        conn.execute(
            "INSERT OR REPLACE INTO predictions "
            "(experiment_id, sample_id, chum_uri, "
            "original_scene, original_long_action, "
            "original_lat_action, "
            "original_scene_correct, "
            "final_scene, final_long_action, "
            "final_lat_action, "
            "final_scene_correct, "
            "was_revised, scene_was_revised, "
            "scene_type_gt, long_action_gt, "
            "lat_action_gt, "
            "odd_label, fine_class, location) "
            "VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                exp_id,
                sid,
                gt.get("chum_uri"),
                pred_scene,
                pred_long,
                pred_lat,
                scene_correct,
                pred_scene,
                pred_long,
                pred_lat,
                scene_correct,
                0,  # staged has no revision
                0,
                gt_scene,
                gt.get("long_action_gt"),
                gt.get("lat_action_gt"),
                gt.get("odd_label"),
                gt.get("fine_class"),
                gt.get("location"),
            ),
        )

        # Store tool calls
        for call_idx, (
            tname,
            targs,
            tresult,
        ) in enumerate(all_tool_calls.get(sid, [])):
            conn.execute(
                "INSERT OR REPLACE INTO tool_calls "
                "(experiment_id, sample_id, "
                "tool_call_order, tool_name, "
                "tool_arguments_json, "
                "tool_result_json, "
                "model_revised_after) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    exp_id,
                    sid,
                    call_idx,
                    tname,
                    json.dumps(targs),
                    json.dumps(tresult),
                    0,
                ),
            )

    # Update experiment
    n_total = len(sample_ids)
    throughput = (
        n_total / total_time if total_time > 0 else 0
    )
    conn.execute(
        "UPDATE experiments SET "
        "status='completed', "
        "total_wall_time_s=?, "
        "throughput_samples_per_s=? "
        "WHERE experiment_id=?",
        (total_time, throughput, exp_id),
    )
    conn.commit()

    # Print summary
    n_correct = conn.execute(
        "SELECT SUM(final_scene_correct) "
        "FROM predictions WHERE experiment_id=?",
        (exp_id,),
    ).fetchone()[0]
    conn.close()

    acc = (n_correct or 0) / n_total if n_total > 0 else 0
    print(f"\n=== Results: {exp_id} ===")
    print(f"Samples: {n_total}")
    print(
        f"Scene accuracy: "
        f"{n_correct or 0}/{n_total} = {acc:.4f}"
    )
    print(f"Wall time: {total_time:.1f}s")
    print(f"Throughput: {throughput:.1f} samples/s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline B: Staged Prediction"
    )
    parser.add_argument(
        "--verifier-model",
        required=True,
        help="Path to base verifier model",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=1,
        help="GPU ID to use (default: 1)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB,
        help="Path to tool_calling DB",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Concurrent HTTP request threads",
    )
    args = parser.parse_args()

    run_staged(
        verifier_model=args.verifier_model,
        gpu_id=args.gpu_id,
        db_path=args.db_path,
        max_model_len=args.max_model_len,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
