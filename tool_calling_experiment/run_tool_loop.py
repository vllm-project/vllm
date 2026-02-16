#!/usr/bin/env python3
"""End-to-end tool-calling loop: 10-sample mechanical proof.

Starts a vLLM server with tool-calling enabled, runs 10 curated samples
through a verify-after-predict loop where the model drives tool usage,
and logs every detail.

Usage:
    python run_tool_loop.py
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sqlite3
import sys

# ---- sys.path fix: remove /workspace/vllm to avoid vllm shadow ----
sys.path = [p for p in sys.path if p != "/workspace/vllm"]

# Now safe to import from tool_calling_experiment
sys.path.insert(0, "/workspace/vllm/tool_calling_experiment")
_server_utils = importlib.import_module("server_utils")
_tools_v2 = importlib.import_module("tools_v2")
VLLMServer = _server_utils.VLLMServer
execute_tool_v2 = _tools_v2.execute_tool_v2
get_tools_for_level = _tools_v2.get_tools_for_level

# ====================================================================
# Configuration
# ====================================================================
MODEL_PATH = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_ID = 3
PORT = 8313
TOOL_LEVEL = 3
MAX_TOOL_ROUNDS = 5
TEMPERATURE = 0
MAX_TOKENS = 1024
DB_PATH = "/workspace/vllm/self_consistency_experiment/self_consistency.db"
OUTPUT_PATH = "/workspace/vllm/tool_calling_experiment/tool_loop_results.json"

# ====================================================================
# Sample selection
# ====================================================================
# 5 where model predicted incident_zone but GT is nominal
# 2 where model predicted incident_zone and GT IS incident_zone
# 2 where model predicted correctly
# 1 wildcard (other error type)

SELECTED_SAMPLES = [
    # Category 1: predicted=incident_zone, GT=nominal (5 samples)
    {"sample_id": 2, "category": "false_incident_zone"},
    {"sample_id": 6, "category": "false_incident_zone"},
    {"sample_id": 7, "category": "false_incident_zone"},
    {"sample_id": 8, "category": "false_incident_zone"},
    {"sample_id": 13, "category": "false_incident_zone"},
    # Category 2: predicted=incident_zone, GT=incident_zone (2 samples)
    {"sample_id": 8020, "category": "true_incident_zone"},
    {"sample_id": 8023, "category": "true_incident_zone"},
    # Category 3: correct predictions (2 samples)
    {"sample_id": 0, "category": "correct_prediction"},
    {"sample_id": 1, "category": "correct_prediction"},
    # Category 4: wildcard (1 sample)
    {"sample_id": 25, "category": "wildcard"},
]


def load_samples() -> list[dict]:
    """Load the selected samples from the self-consistency DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    samples = []
    for sel in SELECTED_SAMPLES:
        sid = sel["sample_id"]
        cursor.execute(
            """
            SELECT sample_id, predicted_scene, predicted_long_action,
                   predicted_lat_action, scene_type_gt, long_action_gt,
                   lat_action_gt, fine_class
            FROM predictions
            WHERE sample_id = ?
            LIMIT 1
            """,
            (sid,),
        )
        row = cursor.fetchone()
        if row is None:
            print(f"WARNING: sample_id={sid} not found in DB, skipping")
            continue
        samples.append(
            {
                "sample_id": row["sample_id"],
                "predicted_scene": row["predicted_scene"],
                "predicted_long_action": row["predicted_long_action"],
                "predicted_lat_action": row["predicted_lat_action"],
                "scene_type_gt": row["scene_type_gt"],
                "long_action_gt": row["long_action_gt"],
                "lat_action_gt": row["lat_action_gt"],
                "fine_class": row["fine_class"],
                "category": sel["category"],
            }
        )
    conn.close()
    return samples


def build_system_prompt() -> str:
    """Return the system prompt for the verification assistant."""
    return (
        "You are a driving scene analysis assistant. You have access to "
        "statistical tools that can help you verify predictions about "
        "driving scenes. Use the tools to check predictions before giving "
        "your final answer. You MUST call at least one tool before giving "
        "your final answer."
    )


def build_user_prompt(sample: dict) -> str:
    """Return the user prompt for a given sample."""
    return (
        f"You are verifying a driving scene prediction. A model analyzed "
        f"a dashcam image and predicted:\n"
        f"- Scene: {sample['predicted_scene']}\n"
        f"- Longitudinal action: {sample['predicted_long_action']}\n"
        f"- Lateral action: {sample['predicted_lat_action']}\n\n"
        f"Before accepting or rejecting this prediction, use the available "
        f"tools to check whether it's reasonable. Call whichever tools you "
        f"think are relevant. After checking, give your final prediction.\n\n"
        f"Output format:\n"
        f"FINAL_SCENE: <scene_type>\n"
        f"FINAL_LONG_ACTION: <action>\n"
        f"FINAL_LAT_ACTION: <action>"
    )


def parse_final_prediction(text: str) -> dict:
    """Extract FINAL_SCENE, FINAL_LONG_ACTION, FINAL_LAT_ACTION from text."""
    result: dict[str, str | None] = {
        "final_scene": None,
        "final_long_action": None,
        "final_lat_action": None,
    }
    if not text:
        return result

    scene_match = re.search(r"FINAL_SCENE:\s*(\S+)", text, re.IGNORECASE)
    long_match = re.search(r"FINAL_LONG_ACTION:\s*(\S+)", text, re.IGNORECASE)
    lat_match = re.search(r"FINAL_LAT_ACTION:\s*(\S+)", text, re.IGNORECASE)

    if scene_match:
        result["final_scene"] = scene_match.group(1).strip().lower()
    if long_match:
        result["final_long_action"] = long_match.group(1).strip().lower()
    if lat_match:
        result["final_lat_action"] = lat_match.group(1).strip().lower()

    return result


def run_tool_loop(
    server: object,
    sample: dict,
    tools: list[dict],
    level: int,
) -> dict:
    """Run the tool-calling loop for a single sample.

    The model drives. The code is a dumb librarian.
    """
    conversation: list[dict] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(sample)},
    ]

    tool_call_log: list[dict] = []
    round_num = 0
    final_text = ""

    while round_num < MAX_TOOL_ROUNDS:
        round_num += 1
        print(f"    Round {round_num}...", end=" ", flush=True)

        try:
            output = server.chat(  # type: ignore[union-attr]
                conversation,
                tools=tools,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            return {
                "error": str(e),
                "round_failed": round_num,
                "tool_calls": tool_call_log,
                "conversation": conversation,
            }

        # Check for tool calls
        tool_calls = output.get("tool_calls")

        if tool_calls:
            # Model wants to call tools
            assistant_msg = {
                "role": "assistant",
                "content": output.get("content") or "",
                "tool_calls": tool_calls,
            }
            conversation.append(assistant_msg)

            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {}

                print(f"{tool_name}({args})", end=" ", flush=True)

                # Execute the tool
                result = execute_tool_v2(tool_name, args, level=level)

                tool_call_log.append(
                    {
                        "round": round_num,
                        "tool": tool_name,
                        "args": args,
                        "result": result,
                    }
                )

                # Append tool result to conversation
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    }
                )

            print()  # newline after tool calls
            continue

        # No tool calls -- model is done
        final_text = output.get("content") or ""
        print(f"DONE (final response, {len(final_text)} chars)")
        break
    else:
        # Hit MAX_TOOL_ROUNDS -- force a final answer
        print(
            f"    Hit max rounds ({MAX_TOOL_ROUNDS}), "
            f"requesting final answer..."
        )
        conversation.append(
            {
                "role": "user",
                "content": (
                    "You have used all available tool call rounds. "
                    "Please give your final prediction now.\n\n"
                    "Output format:\n"
                    "FINAL_SCENE: <scene_type>\n"
                    "FINAL_LONG_ACTION: <action>\n"
                    "FINAL_LAT_ACTION: <action>"
                ),
            }
        )
        try:
            output = server.chat(  # type: ignore[union-attr]
                conversation,
                tools=None,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            final_text = output.get("content") or ""
        except Exception as e:
            final_text = f"ERROR: {e}"

    # Parse the final prediction
    parsed = parse_final_prediction(final_text)

    return {
        "final_text": final_text,
        "parsed_prediction": parsed,
        "tool_calls": tool_call_log,
        "num_tool_calls": len(tool_call_log),
        "num_rounds": round_num,
        "conversation": conversation,
    }


def main() -> None:  # noqa: C901
    """Run the 10-sample tool-calling proof."""
    print("=" * 70)
    print("TOOL-CALLING LOOP: 10-SAMPLE MECHANICAL PROOF")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU: {GPU_ID}, Port: {PORT}")
    print(f"Tool level: {TOOL_LEVEL}")
    print(f"Max tool rounds: {MAX_TOOL_ROUNDS}")
    print(f"Temperature: {TEMPERATURE}")
    print()

    # Load samples
    print("Loading samples from DB...")
    samples = load_samples()
    print(f"Loaded {len(samples)} samples")
    for s in samples:
        print(
            f"  sample_id={s['sample_id']:5d}  "
            f"pred={s['predicted_scene']:15s}  "
            f"GT={s['scene_type_gt']:15s}  "
            f"cat={s['category']}"
        )
    print()

    # Get tool definitions
    tools = get_tools_for_level(level=TOOL_LEVEL)
    print(f"Tool definitions loaded: {len(tools)} tools")
    for t in tools:
        print(f"  - {t['function']['name']}")
    print()

    # Start server
    print("Starting vLLM server...")
    server = VLLMServer(
        model_path=MODEL_PATH,
        port=PORT,
        gpu_id=GPU_ID,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_tools=True,
    )
    try:
        server.start(timeout=360)
    except RuntimeError as e:
        print(f"FATAL: Server failed to start: {e}")
        sys.exit(1)

    print()

    # Run the loop for each sample
    all_results: list[dict] = []
    mechanics_ok = True
    failure_modes: list[str] = []

    try:
        for i, sample in enumerate(samples, 1):
            print("-" * 60)
            print(
                f"Sample {i}/{len(samples)}: "
                f"id={sample['sample_id']}, "
                f"predicted={sample['predicted_scene']}, "
                f"GT={sample['scene_type_gt']} "
                f"[{sample['category']}]"
            )

            result = run_tool_loop(server, sample, tools, TOOL_LEVEL)

            # Analyze result
            parsed = result.get("parsed_prediction", {})
            final_scene = parsed.get("final_scene")
            final_long = parsed.get("final_long_action")
            final_lat = parsed.get("final_lat_action")

            was_revised = final_scene != sample["predicted_scene"]
            scene_correct = final_scene == sample["scene_type_gt"]
            revision_correct = scene_correct if was_revised else None

            # Check mechanics
            if result.get("error"):
                mechanics_ok = False
                failure_modes.append(
                    f"Sample {sample['sample_id']}: "
                    f"HTTP error: {result['error']}"
                )
            elif result["num_tool_calls"] == 0:
                mechanics_ok = False
                failure_modes.append(
                    f"Sample {sample['sample_id']}: "
                    f"Model did not call any tools"
                )
            elif final_scene is None:
                mechanics_ok = False
                failure_modes.append(
                    f"Sample {sample['sample_id']}: "
                    f"Could not parse FINAL_SCENE from output"
                )

            sample_result = {
                "sample_id": sample["sample_id"],
                "category": sample["category"],
                "gt_scene": sample["scene_type_gt"],
                "gt_long_action": sample["long_action_gt"],
                "gt_lat_action": sample["lat_action_gt"],
                "original_prediction": sample["predicted_scene"],
                "original_long_action": sample["predicted_long_action"],
                "original_lat_action": sample["predicted_lat_action"],
                "tool_calls": result.get("tool_calls", []),
                "num_tool_calls": result.get("num_tool_calls", 0),
                "num_rounds": result.get("num_rounds", 0),
                "final_scene": final_scene,
                "final_long_action": final_long,
                "final_lat_action": final_lat,
                "was_revised": was_revised,
                "scene_correct": scene_correct,
                "revision_correct": revision_correct,
                "final_text": result.get("final_text", ""),
                "full_conversation": result.get("conversation", []),
            }
            all_results.append(sample_result)

            # Print per-sample summary
            tools_used = [
                tc["tool"] for tc in result.get("tool_calls", [])
            ]
            tools_str = (
                ", ".join(tools_used) if tools_used else "(none)"
            )

            # Extract a short reasoning snippet
            final_text_str = result.get("final_text", "")
            reasoning_lines = [
                ln.strip()
                for ln in final_text_str.split("\n")
                if ln.strip()
                and not ln.strip().startswith("FINAL_")
            ]
            reasoning_snippet = (
                reasoning_lines[0][:100]
                if reasoning_lines
                else "(no reasoning)"
            )

            status_parts = []
            status_parts.append("REVISED" if was_revised else "KEPT")
            status_parts.append(
                "CORRECT" if scene_correct else "WRONG"
            )

            print(f"  Tools called: {tools_str}")
            print(f"  Model reasoning: \"{reasoning_snippet}\"")
            print(
                f"  Final: {final_scene} "
                f"({', '.join(status_parts)})"
            )
            print()

    finally:
        # Always stop the server
        print("Stopping vLLM server...")
        server.stop()
        # Also pkill just in case
        os.system(f"pkill -f 'vllm serve.*{PORT}' 2>/dev/null")

    # ================================================================
    # Summary statistics
    # ================================================================
    _print_summary(all_results, mechanics_ok, failure_modes)


def _print_summary(
    all_results: list[dict],
    mechanics_ok: bool,
    failure_modes: list[str],
) -> None:
    """Print summary statistics and save results to JSON."""
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(all_results)
    completed = sum(
        1 for r in all_results if r["final_scene"] is not None
    )
    total_tool_calls = [r["num_tool_calls"] for r in all_results]
    revised = [r for r in all_results if r["was_revised"]]
    kept = [r for r in all_results if not r["was_revised"]]

    # Revision accuracy
    revised_correct = sum(1 for r in revised if r["scene_correct"])
    # Kept that were correct
    _kept_correct = sum(1 for r in kept if r["scene_correct"])
    # Original correctness (scene only)
    orig_correct = sum(
        1
        for r in all_results
        if r["original_prediction"] == r["gt_scene"]
    )

    # Saves: revised AND now correct (was wrong before)
    saves = sum(
        1
        for r in revised
        if r["scene_correct"]
        and r["original_prediction"] != r["gt_scene"]
    )
    # Breaks: revised AND now wrong (was correct before)
    breaks = sum(
        1
        for r in revised
        if not r["scene_correct"]
        and r["original_prediction"] == r["gt_scene"]
    )

    avg_tools = (
        sum(total_tool_calls) / len(total_tool_calls)
        if total_tool_calls
        else 0
    )
    min_tools = min(total_tool_calls) if total_tool_calls else 0
    max_tools = max(total_tool_calls) if total_tool_calls else 0

    print(
        f"Mechanics: {completed}/{total} samples completed "
        f"(no crashes)"
    )
    print(
        f"Tool calls: avg {avg_tools:.1f} per sample, "
        f"range {min_tools}-{max_tools}"
    )
    pct_revised = len(revised) / total * 100 if total else 0
    print(f"Revised: {len(revised)}/{total} ({pct_revised:.0f}%)")
    if revised:
        pct_rev_correct = revised_correct / len(revised) * 100
        print(
            f"Revision accuracy: {revised_correct}/{len(revised)} "
            f"correct ({pct_rev_correct:.0f}%)"
        )
    print(
        f"Net improvement: +{saves} saves, -{breaks} breaks = "
        f"+{saves - breaks}"
    )
    final_correct = sum(
        1 for r in all_results if r["scene_correct"]
    )
    print(
        f"Scene accuracy: {orig_correct}/{total} original -> "
        f"{final_correct}/{total} after tools"
    )
    print()

    # Per-sample detail
    for i, r in enumerate(all_results, 1):
        tools_used = ", ".join(tc["tool"] for tc in r["tool_calls"])
        status = []
        status.append("REVISED" if r["was_revised"] else "KEPT")
        status.append("CORRECT" if r["scene_correct"] else "WRONG")
        print(
            f"Sample {i}: predicted={r['original_prediction']}, "
            f"GT={r['gt_scene']}"
        )
        print(f"  Tools: {tools_used}")
        print(
            f"  Final: {r['final_scene']} ({', '.join(status)})"
        )
    print()

    # Verdict
    if mechanics_ok and completed == total:
        print("MECHANICS VERIFIED -- ready to scale to 100 samples")
    else:
        print(
            f"MECHANICS BROKEN -- {'; '.join(failure_modes)}"
        )

    # Save results
    print(f"\nSaving results to {OUTPUT_PATH}...")
    output_data = {
        "config": {
            "model": MODEL_PATH,
            "gpu_id": GPU_ID,
            "port": PORT,
            "tool_level": TOOL_LEVEL,
            "max_tool_rounds": MAX_TOOL_ROUNDS,
            "temperature": TEMPERATURE,
            "num_samples": total,
        },
        "summary": {
            "completed": completed,
            "total": total,
            "avg_tool_calls": round(avg_tools, 1),
            "min_tool_calls": min_tools,
            "max_tool_calls": max_tools,
            "num_revised": len(revised),
            "revision_accuracy": (
                round(revised_correct / len(revised), 3)
                if revised
                else None
            ),
            "saves": saves,
            "breaks": breaks,
            "net_improvement": saves - breaks,
            "original_scene_accuracy": orig_correct,
            "final_scene_accuracy": final_correct,
            "mechanics_ok": mechanics_ok,
            "failure_modes": failure_modes,
        },
        "samples": all_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print("Done.")


if __name__ == "__main__":
    main()
