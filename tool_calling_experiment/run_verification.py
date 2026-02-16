#!/usr/bin/env python3
"""Pipeline A: Predict-then-Verify with tool calling.

Reads baseline predictions from the tool_calling DB, constructs
verification prompts, runs them through a base Qwen3-VL model
with tool definitions via the vLLM HTTP server, executes tool calls
locally, feeds results back for a second turn, and stores final
predictions.

Usage:
    python tool_calling_experiment/run_verification.py \
        --condition oracle \
        --verifier-model /fsx/models/Qwen3-VL-8B-Instruct \
        --gpu-id 2
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any

_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
# Add the tool_calling_experiment directory itself to sys.path so that
# sibling modules (parse_tool_calls, tools) can be imported directly
# without needing the parent on sys.path (which would shadow the
# installed vllm package with the local source tree).
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
# Remove the parent directory (and CWD alias '') from sys.path if it
# contains a local ``vllm/`` source tree that would shadow the installed
# vllm package (which has compiled CUDA extensions).
if os.path.isdir(os.path.join(_PARENT, "vllm")):
    sys.path[:] = [
        p
        for p in sys.path
        if os.path.abspath(p or os.getcwd()) != _PARENT
    ]
DEFAULT_DB = os.path.join(_DIR, "tool_calling.db")

VERIFICATION_PROMPT = """\
You are a driving scene verification system. A prediction \
model has analyzed dashcam images and made predictions about \
the scene. Your job is to verify these predictions using the \
provided tools, then output your corrected predictions.

The model predicted:
- Scene type: {predicted_scene}
- Longitudinal action: {predicted_long_action}
- Lateral action: {predicted_lat_action}

Use the available tools to check whether these predictions \
are reasonable. After checking, provide your final corrected \
predictions in this exact format:

FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REVISED: <yes|no>
REASON: <brief explanation>"""

CONDITIONS = [
    "oracle",
    "prior_only",
    "confusion_only",
    "all_tools",
]


def _derive_verifier_size(model_path: str) -> str:
    """Extract model size label from path (e.g. '8b')."""
    lower = model_path.lower()
    if "8b" in lower:
        return "8b"
    if "2b" in lower:
        return "2b"
    if "72b" in lower:
        return "72b"
    return "base"


def _make_experiment_id(
    condition: str, verifier_size: str
) -> str:
    """Create a unique experiment ID."""
    ts = datetime.now(tz=timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    return f"tc_{condition}_{verifier_size}_{ts}"


def _load_baseline_predictions(
    db_path: str,
) -> list[dict[str, Any]]:
    """Load all baseline predictions from the DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM predictions "
            "WHERE experiment_id = 'baseline' "
            "ORDER BY sample_id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _build_verification_message(
    pred: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the user message for verification."""
    prompt = VERIFICATION_PROMPT.format(
        predicted_scene=pred.get(
            "original_scene", "unknown"
        ),
        predicted_long_action=pred.get(
            "original_long_action", "unknown"
        ),
        predicted_lat_action=pred.get(
            "original_lat_action", "unknown"
        ),
    )
    return [{"role": "user", "content": prompt}]


def _build_ground_truth(
    pred: dict[str, Any],
) -> dict[str, Any]:
    """Extract ground truth dict for oracle tools."""
    return {
        "scene_type_gt": pred.get("scene_type_gt"),
        "long_action_gt": pred.get("long_action_gt"),
        "lat_action_gt": pred.get("lat_action_gt"),
    }


def _extract_tool_calls_from_message(
    msg: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Extract (tool_name, arguments) pairs from a server response message.

    The vLLM HTTP API with ``--enable-auto-tool-choice --tool-call-parser hermes``
    returns structured ``tool_calls`` in the response message when the model
    invokes tools.  This helper normalises that into a simple list.
    """
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


def _build_turn2_messages(
    original_msgs: list[dict[str, Any]],
    assistant_msg: dict[str, Any],
    tool_calls_with_results: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build the multi-turn message list for Turn 2.

    Follows the OpenAI convention: user -> assistant (with tool_calls) ->
    tool (one per call) -> ... so the model can reason about the results.
    """
    msgs = list(original_msgs)

    # Append the assistant message as-is (it contains tool_calls metadata)
    msgs.append(assistant_msg)

    # Append one tool-role message per tool call result
    assistant_tool_calls = assistant_msg.get("tool_calls") or []
    for idx, (_tname, _targs, tresult) in enumerate(
        tool_calls_with_results
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


def run_verification(
    condition: str,
    verifier_model: str,
    gpu_id: int,
    db_path: str,
    max_model_len: int,
    max_workers: int,
) -> None:
    """Run the full verification pipeline."""
    from server_utils import VLLMServer  # type: ignore[import-not-found]

    from tools import get_tools_for_condition  # type: ignore[import-not-found]

    # Load baseline predictions
    predictions = _load_baseline_predictions(db_path)
    if not predictions:
        print("ERROR: No baseline predictions found.")
        print("Run run_prediction.py first.")
        return

    print(f"Loaded {len(predictions)} baseline predictions")

    # Setup
    verifier_size = _derive_verifier_size(verifier_model)
    exp_id = _make_experiment_id(condition, verifier_size)
    tools = get_tools_for_condition(condition)
    tool_names = [
        t["function"]["name"] for t in tools
    ]
    is_oracle = condition == "oracle"

    print(f"Experiment: {exp_id}")
    print(f"Condition: {condition}")
    print(f"Verifier: {verifier_model} ({verifier_size})")
    print(f"Tools: {tool_names}")
    print(f"Oracle mode: {is_oracle}")

    # Create experiment record
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO experiments "
        "(experiment_id, condition_name, pipeline, "
        "predictor_model, verifier_model, "
        "tools_enabled, temperature_verify, seed, "
        "total_samples, status, description) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            exp_id,
            condition,
            "A",
            "baseline",
            verifier_model,
            json.dumps(tool_names),
            0.0,
            42,
            len(predictions),
            "running",
            f"Pipeline A: {condition} with "
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
        _run_verification_with_server(
            server=server,
            predictions=predictions,
            tools=tools,
            is_oracle=is_oracle,
            exp_id=exp_id,
            conn=conn,
            max_workers=max_workers,
        )
    finally:
        server.stop()


def _run_verification_with_server(
    server: Any,
    predictions: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    is_oracle: bool,
    exp_id: str,
    conn: sqlite3.Connection,
    max_workers: int,
) -> None:
    """Core verification logic using an already-started VLLMServer."""
    from parse_tool_calls import (  # type: ignore[import-not-found]
        parse_final_predictions,
    )

    from tools import execute_tool  # type: ignore[import-not-found]

    # ---- TURN 1: Initial verification ----
    print("\n=== Turn 1: Initial verification ===")
    t1_start = time.time()

    # Build messages for all samples
    all_messages: list[list[dict[str, Any]]] = []
    for pred in predictions:
        msgs = _build_verification_message(pred)
        all_messages.append(msgs)

    # Batch request all samples
    turn1_results = server.batch_chat(
        all_messages,
        tools=tools,
        temperature=0,
        max_tokens=1024,
        max_workers=max_workers,
    )

    t1_time = time.time() - t1_start
    n_errors = sum(1 for _, err in turn1_results if err)
    print(
        f"Turn 1 completed in {t1_time:.1f}s "
        f"({n_errors} errors)"
    )

    # ---- Parse tool calls and execute locally ----
    print("\n=== Executing tool calls ===")
    # For each sample: list of (tool_name, args, result)
    sample_tool_calls: list[
        list[tuple[str, dict[str, Any], dict[str, Any]]]
    ] = []
    needs_turn2: list[bool] = []
    turn1_messages: list[dict[str, Any]] = []

    for idx, (msg, err) in enumerate(turn1_results):
        if err or msg is None:
            sample_tool_calls.append([])
            needs_turn2.append(False)
            turn1_messages.append(
                {"role": "assistant", "content": ""}
            )
            continue

        turn1_messages.append(msg)
        calls = _extract_tool_calls_from_message(msg)
        sample_calls: list[
            tuple[str, dict[str, Any], dict[str, Any]]
        ] = []
        if calls:
            gt = (
                _build_ground_truth(predictions[idx])
                if is_oracle
                else None
            )
            for tool_name, args in calls:
                result = execute_tool(
                    tool_name, args, ground_truth=gt
                )
                sample_calls.append(
                    (tool_name, args, result)
                )
        sample_tool_calls.append(sample_calls)
        needs_turn2.append(len(sample_calls) > 0)

    n_with_calls = sum(needs_turn2)
    print(
        f"Samples with tool calls: {n_with_calls}/"
        f"{len(predictions)}"
    )

    # ---- TURN 2: Feed tool results back ----
    print("\n=== Turn 2: Process tool results ===")
    t2_start = time.time()

    turn2_indices: list[int] = []
    turn2_messages: list[list[dict[str, Any]]] = []

    for idx in range(len(predictions)):
        if not needs_turn2[idx]:
            continue
        msgs = _build_turn2_messages(
            all_messages[idx],
            turn1_messages[idx],
            sample_tool_calls[idx],
        )
        turn2_indices.append(idx)
        turn2_messages.append(msgs)

    # Run turn 2 batch
    turn2_outputs: dict[int, str] = {}
    if turn2_messages:
        t2_results = server.batch_chat(
            turn2_messages,
            tools=tools,
            temperature=0,
            max_tokens=1024,
            max_workers=max_workers,
        )
        for j, (msg, err) in enumerate(t2_results):
            real_idx = turn2_indices[j]
            if msg is not None:
                turn2_outputs[real_idx] = (
                    msg.get("content") or ""
                )
            else:
                turn2_outputs[real_idx] = ""

    t2_time = time.time() - t2_start
    print(f"Turn 2 completed in {t2_time:.1f}s")

    # ---- Parse final predictions and store ----
    print("\n=== Storing results ===")
    total_time = t1_time + t2_time

    for idx, pred in enumerate(predictions):
        # Determine which output to parse
        if idx in turn2_outputs:
            final_text = turn2_outputs[idx]
        else:
            msg = turn1_messages[idx]
            final_text = msg.get("content") or ""

        parsed = parse_final_predictions(final_text)
        final_scene = (
            parsed["final_scene"]
            or pred.get("original_scene")
        )
        final_long = (
            parsed["final_long_action"]
            or pred.get("original_long_action")
        )
        final_lat = (
            parsed["final_lat_action"]
            or pred.get("original_lat_action")
        )

        gt_scene = pred.get("scene_type_gt")
        orig_scene = pred.get("original_scene")

        orig_correct = (
            int(orig_scene == gt_scene)
            if orig_scene and gt_scene
            else None
        )
        final_correct = (
            int(final_scene == gt_scene)
            if final_scene and gt_scene
            else None
        )
        was_revised = int(
            final_scene != orig_scene
            or final_long
            != pred.get("original_long_action")
            or final_lat
            != pred.get("original_lat_action")
        )
        scene_revised = int(final_scene != orig_scene)

        # Flip analysis
        flip_correct = (
            int(
                orig_correct == 0 and final_correct == 1
            )
            if orig_correct is not None
            and final_correct is not None
            else 0
        )
        flip_incorrect = (
            int(
                orig_correct == 1 and final_correct == 0
            )
            if orig_correct is not None
            and final_correct is not None
            else 0
        )

        conn.execute(
            "INSERT OR REPLACE INTO predictions "
            "(experiment_id, sample_id, chum_uri, "
            "original_scene, original_long_action, "
            "original_lat_action, "
            "original_generated_text, "
            "original_scene_correct, "
            "final_scene, final_long_action, "
            "final_lat_action, "
            "final_generated_text, "
            "final_scene_correct, "
            "was_revised, scene_was_revised, "
            "revision_reason, "
            "scene_type_gt, long_action_gt, "
            "lat_action_gt, "
            "odd_label, fine_class, location, "
            "original_flipped_correct, "
            "original_flipped_incorrect, "
            "verify_time_ms, total_time_ms) "
            "VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
            "?,?,?,?,?,?,?)",
            (
                exp_id,
                pred["sample_id"],
                pred.get("chum_uri"),
                orig_scene,
                pred.get("original_long_action"),
                pred.get("original_lat_action"),
                pred.get("original_generated_text"),
                orig_correct,
                final_scene,
                final_long,
                final_lat,
                final_text,
                final_correct,
                was_revised,
                scene_revised,
                parsed.get("reason"),
                gt_scene,
                pred.get("long_action_gt"),
                pred.get("lat_action_gt"),
                pred.get("odd_label"),
                pred.get("fine_class"),
                pred.get("location"),
                flip_correct,
                flip_incorrect,
                (t2_time / len(predictions) * 1000)
                if predictions
                else 0,
                (total_time / len(predictions) * 1000)
                if predictions
                else 0,
            ),
        )

        # Store tool calls
        for call_idx, (
            tname,
            targs,
            tresult,
        ) in enumerate(sample_tool_calls[idx]):
            # Determine if model revised after this call
            revised_after = (
                int(scene_revised)
                if call_idx
                == len(sample_tool_calls[idx]) - 1
                else 0
            )
            revised_field = None
            old_val = None
            new_val = None
            if revised_after and scene_revised:
                revised_field = "scene"
                old_val = orig_scene
                new_val = final_scene

            conn.execute(
                "INSERT OR REPLACE INTO tool_calls "
                "(experiment_id, sample_id, "
                "tool_call_order, tool_name, "
                "tool_arguments_json, "
                "tool_result_json, "
                "model_revised_after, "
                "revised_field, old_value, "
                "new_value) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    exp_id,
                    pred["sample_id"],
                    call_idx,
                    tname,
                    json.dumps(targs),
                    json.dumps(tresult),
                    revised_after,
                    revised_field,
                    old_val,
                    new_val,
                ),
            )

    # Update experiment status
    n_total = len(predictions)
    throughput = (
        n_total / total_time if total_time > 0 else 0
    )
    conn.execute(
        "UPDATE experiments SET "
        "status='completed', "
        "total_wall_time_s=?, "
        "verify_wall_time_s=?, "
        "throughput_samples_per_s=? "
        "WHERE experiment_id=?",
        (total_time, t2_time, throughput, exp_id),
    )
    conn.commit()

    # Print summary
    n_correct = conn.execute(
        "SELECT SUM(final_scene_correct) "
        "FROM predictions WHERE experiment_id=?",
        (exp_id,),
    ).fetchone()[0]
    n_revised = conn.execute(
        "SELECT SUM(was_revised) "
        "FROM predictions WHERE experiment_id=?",
        (exp_id,),
    ).fetchone()[0]
    n_saves = conn.execute(
        "SELECT SUM(original_flipped_correct) "
        "FROM predictions WHERE experiment_id=?",
        (exp_id,),
    ).fetchone()[0]
    n_breaks = conn.execute(
        "SELECT SUM(original_flipped_incorrect) "
        "FROM predictions WHERE experiment_id=?",
        (exp_id,),
    ).fetchone()[0]
    conn.close()

    acc = (
        (n_correct or 0) / n_total if n_total > 0 else 0
    )
    print(f"\n=== Results: {exp_id} ===")
    print(f"Samples: {n_total}")
    print(f"Scene accuracy: {acc:.4f}")
    print(
        f"Revised: {n_revised} "
        f"({(n_revised or 0) / n_total:.1%})"
    )
    print(f"Saves (wrong->right): {n_saves or 0}")
    print(f"Breaks (right->wrong): {n_breaks or 0}")
    print(
        f"Net improvement: "
        f"{(n_saves or 0) - (n_breaks or 0)}"
    )
    print(f"Wall time: {total_time:.1f}s")
    print(
        f"Throughput: {throughput:.1f} samples/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline A: Predict-then-Verify"
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=CONDITIONS,
        help="Experimental condition",
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

    run_verification(
        condition=args.condition,
        verifier_model=args.verifier_model,
        gpu_id=args.gpu_id,
        db_path=args.db_path,
        max_model_len=args.max_model_len,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
