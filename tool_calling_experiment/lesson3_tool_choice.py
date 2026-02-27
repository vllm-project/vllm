#!/usr/bin/env python3
"""Lesson 3: Tool choice behavior under different conditions.

Tests 3 conditions x 2 models x 20 samples = 120 requests:
  A) tool_choice="auto"  -- model decides whether to use tools
  B) tool_choice="required" -- forced to call at least one tool
  C) Metacognitive prompt with tool_choice="auto"

Uses text-only mode (scene descriptions from DB) since MDS image
loading is not straightforward in this environment.

Usage:
    python tool_calling_experiment/lesson3_tool_choice.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

# ------------------------------------------------------------------
# Path setup -- keep tool_calling_experiment importable without
# shadowing the installed vllm package.
# ------------------------------------------------------------------
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

from server_utils import VLLMServer  # noqa: E402, I001  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
from tools import ALL_TOOLS, execute_tool  # noqa: E402, I001  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_2B = 0
GPU_8B = 1
PORT_2B = 8310
PORT_8B = 8311

SC_DB_PATH = os.path.join(
    _PARENT,
    "self_consistency_experiment",
    "self_consistency.db",
)
RESULTS_PATH = os.path.join(_DIR, "lesson3_results.json")

VALID_SCENES = {
    "nominal",
    "flooded",
    "incident_zone",
    "mounted_police",
    "flagger",
}

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
PROMPT_A_B = """\
Analyze this driving scene description from a dashcam image. \
Classify the driving scene as one of: nominal, flooded, \
incident_zone, mounted_police, flagger. Then predict the \
driving action (longitudinal: stop/slowdown/proceed/null; \
lateral: lc_left/lc_right/null).

Scene description from the prediction model:
{scene_description}

Provide your answer in this format:
FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REASON: <brief explanation>"""

PROMPT_C = """\
Analyze this driving scene description from a dashcam image. \
Before committing to a prediction, think about what you are \
uncertain about. Use the available tools to check your \
reasoning before giving a final answer.

Scene description from the prediction model:
{scene_description}

Classify the scene as: nominal, flooded, incident_zone, \
mounted_police, flagger. Predict the action \
(longitudinal: stop/slowdown/proceed/null; \
lateral: lc_left/lc_right/null).

Provide your answer in this format:
FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REASON: <brief explanation>"""

# ------------------------------------------------------------------
# Parsing helpers (compiled regexes)
# ------------------------------------------------------------------
_SCENE_RE = re.compile(
    r"FINAL_SCENE:\s*(\S+)", re.IGNORECASE
)
_LONG_RE = re.compile(
    r"FINAL_LONG_ACTION:\s*(\S+)", re.IGNORECASE
)
_LAT_RE = re.compile(
    r"FINAL_LAT_ACTION:\s*(\S+)", re.IGNORECASE
)


# ------------------------------------------------------------------
# Sample selection
# ------------------------------------------------------------------


def select_samples(
    db_path: str, n: int = 20
) -> list[dict[str, Any]]:
    """Select *n* diverse samples from the self-consistency DB.

    Picks 4 samples per scene type (nominal, flooded,
    incident_zone, mounted_police, flagger), including some
    nominal_triggers.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    samples: list[dict[str, Any]] = []

    scene_types = [
        "nominal",
        "flooded",
        "incident_zone",
        "mounted_police",
        "flagger",
    ]

    for scene in scene_types:
        if scene == "nominal":
            # 2 nominal_triggers + 2 nominal_clean
            for fc in ("nominal_triggers", "nominal_clean"):
                rows = conn.execute(
                    "SELECT * FROM predictions "
                    "WHERE scene_type_gt = ? "
                    "AND fine_class = ? "
                    "AND temperature = 0 "
                    "ORDER BY sample_id LIMIT 2",
                    (scene, fc),
                ).fetchall()
                for r in rows:
                    samples.append(dict(r))
        else:
            # 2 correct + 2 incorrect
            rows_ok = conn.execute(
                "SELECT * FROM predictions "
                "WHERE scene_type_gt = ? "
                "AND predicted_scene = ? "
                "AND temperature = 0 "
                "ORDER BY sample_id LIMIT 2",
                (scene, scene),
            ).fetchall()
            for r in rows_ok:
                samples.append(dict(r))

            rows_bad = conn.execute(
                "SELECT * FROM predictions "
                "WHERE scene_type_gt = ? "
                "AND predicted_scene != ? "
                "AND temperature = 0 "
                "ORDER BY sample_id LIMIT 2",
                (scene, scene),
            ).fetchall()
            for r in rows_bad:
                samples.append(dict(r))

    conn.close()
    print(f"Selected {len(samples)} samples:")
    scene_counts = Counter(
        s["scene_type_gt"] for s in samples
    )
    for sc, cnt in sorted(scene_counts.items()):
        fine_counts = Counter(
            s.get("fine_class", "?")
            for s in samples
            if s["scene_type_gt"] == sc
        )
        print(f"  {sc}: {cnt} ({dict(fine_counts)})")
    return samples


# ------------------------------------------------------------------
# Tool call extraction and execution
# ------------------------------------------------------------------


def extract_tool_calls(
    msg: dict[str, Any],
) -> list[tuple[str, dict[str, Any], str]]:
    """Extract (name, args, call_id) from a response."""
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


def run_with_tool_loop(
    server: VLLMServer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str,
    max_rounds: int = 3,
) -> dict[str, Any]:
    """Run a request, executing tool calls in a loop.

    Returns a dict with response_text, tool_calls_made,
    num_tool_calls, error, and rounds.
    """
    result: dict[str, Any] = {
        "response_text": "",
        "tool_calls_made": [],
        "num_tool_calls": 0,
        "error": None,
        "rounds": 0,
    }

    conv = list(messages)

    for round_num in range(max_rounds):
        result["rounds"] = round_num + 1
        try:
            # Turn 2+ never forces tool use
            choice = (
                tool_choice if round_num == 0 else "auto"
            )
            msg = server.chat(
                conv,
                tools=tools,
                temperature=0,
                max_tokens=1024,
                tool_choice=choice,
            )
        except Exception as exc:
            result["error"] = str(exc)
            return result

        calls = extract_tool_calls(msg)

        if not calls:
            # Model responded with text only
            result["response_text"] = (
                msg.get("content") or ""
            )
            return result

        # Execute tool calls
        conv.append(msg)

        for name, args, call_id in calls:
            try:
                tool_result = execute_tool(name, args)
            except Exception as exc:
                tool_result = {"error": str(exc)}

            result["tool_calls_made"].append(
                {
                    "name": name,
                    "args": args,
                    "result": tool_result,
                }
            )
            result["num_tool_calls"] += 1

            conv.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(tool_result),
                }
            )

    # Exhausted rounds -- get final text
    try:
        final_choice = (
            "none" if tool_choice == "required" else "auto"
        )
        final_msg = server.chat(
            conv,
            tools=tools,
            temperature=0,
            max_tokens=1024,
            tool_choice=final_choice,
        )
        result["response_text"] = (
            final_msg.get("content") or ""
        )
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------


def parse_response(text: str) -> dict[str, str | None]:
    """Parse scene/action predictions from response text."""
    result: dict[str, str | None] = {
        "predicted_scene": None,
        "predicted_long_action": None,
        "predicted_lat_action": None,
    }
    if not text:
        return result

    m = _SCENE_RE.search(text)
    if m:
        val = m.group(1).strip().strip(".,;:\"'`").lower()
        if val in VALID_SCENES:
            result["predicted_scene"] = val

    m = _LONG_RE.search(text)
    if m:
        val = m.group(1).strip().strip(".,;:\"'`").lower()
        if val in {"stop", "slowdown", "proceed", "null"}:
            result["predicted_long_action"] = val

    m = _LAT_RE.search(text)
    if m:
        val = m.group(1).strip().strip(".,;:\"'`").lower()
        if val in {"lc_left", "lc_right", "null"}:
            result["predicted_lat_action"] = val

    # Fallback: look for scene type in freeform text
    if result["predicted_scene"] is None:
        lower = text.lower()
        found = [s for s in VALID_SCENES if s in lower]
        if len(found) == 1:
            result["predicted_scene"] = found[0]
        elif len(found) > 1:
            non_nom = [
                s for s in found if s != "nominal"
            ]
            if len(non_nom) == 1:
                result["predicted_scene"] = non_nom[0]

    return result


def tool_result_referenced(
    response_text: str, tool_calls: list[dict[str, Any]]
) -> bool:
    """Heuristic: did the model reference tool results?"""
    if not tool_calls or not response_text:
        return False
    lower = response_text.lower()
    keywords = [
        "base rate",
        "prior",
        "rare",
        "confusion",
        "confused",
        "compatible",
        "typical",
        "co-occurrence",
        "cooccurrence",
        "feasible",
        "waypoint",
        "error rate",
        "commonly confused",
        "warning",
        "check",
        "tool",
        "according to",
    ]
    return any(kw in lower for kw in keywords)


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------


def run_condition(
    server: VLLMServer,
    model_name: str,
    samples: list[dict[str, Any]],
    condition: str,
    tool_choice: str,
    prompt_template: str,
) -> list[dict[str, Any]]:
    """Run one condition across all samples."""
    print(
        f"\n--- {model_name} | Condition {condition} "
        f"| tool_choice={tool_choice} ---"
    )
    results: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        gt_scene = sample["scene_type_gt"]
        fine_class = sample.get("fine_class", "?")
        scene_desc = sample.get("generated_text", "")

        prompt = prompt_template.format(
            scene_description=scene_desc
        )
        messages = [{"role": "user", "content": prompt}]

        t0 = time.time()
        outcome = run_with_tool_loop(
            server,
            messages,
            ALL_TOOLS,
            tool_choice,
            max_rounds=3,
        )
        elapsed = time.time() - t0

        parsed = parse_response(outcome["response_text"])
        pred_scene = parsed["predicted_scene"]
        scene_correct = (
            (pred_scene == gt_scene) if pred_scene else None
        )

        used_meaningfully = tool_result_referenced(
            outcome["response_text"],
            outcome["tool_calls_made"],
        )

        rec = {
            "sample_id": sid,
            "gt_scene": gt_scene,
            "fine_class": fine_class,
            "original_prediction": sample.get(
                "predicted_scene", ""
            ),
            "model": model_name,
            "condition": condition,
            "tool_choice": tool_choice,
            "predicted_scene": pred_scene,
            "predicted_long_action": parsed[
                "predicted_long_action"
            ],
            "predicted_lat_action": parsed[
                "predicted_lat_action"
            ],
            "scene_correct": scene_correct,
            "num_tool_calls": outcome["num_tool_calls"],
            "tools_called": [
                tc["name"]
                for tc in outcome["tool_calls_made"]
            ],
            "tool_results_referenced": used_meaningfully,
            "response_text": outcome["response_text"][
                :500
            ],
            "rounds": outcome["rounds"],
            "error": outcome["error"],
            "latency_s": round(elapsed, 2),
        }
        results.append(rec)

        if scene_correct:
            status = "OK"
        elif scene_correct is False:
            status = "WRONG"
        else:
            status = "UNPARSE"

        tools_str = (
            ", ".join(rec["tools_called"])
            if rec["tools_called"]
            else "none"
        )
        print(
            f"  [{i + 1:2d}/{len(samples)}] "
            f"sid={sid} gt={gt_scene:16s} "
            f"pred={str(pred_scene):16s} "
            f"{status:7s} "
            f"tc={rec['num_tool_calls']} "
            f"({tools_str}) {elapsed:.1f}s"
        )

    return results


def compute_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate metrics for a set of results."""
    n = len(results)
    if n == 0:
        return {}

    # Tool call rate
    with_tools = sum(
        1 for r in results if r["num_tool_calls"] > 0
    )
    tool_call_rate = with_tools / n

    # Tool selection counts
    tool_counter: Counter = Counter()
    for r in results:
        for t in r["tools_called"]:
            tool_counter[t] += 1

    # Avg tool calls when tools are used
    tool_counts = [
        r["num_tool_calls"]
        for r in results
        if r["num_tool_calls"] > 0
    ]
    avg_tools_when_used = (
        sum(tool_counts) / len(tool_counts)
        if tool_counts
        else 0
    )

    # Tool usage quality
    with_meaningful = sum(
        1
        for r in results
        if r["num_tool_calls"] > 0
        and r["tool_results_referenced"]
    )
    quality_rate = (
        with_meaningful / with_tools
        if with_tools > 0
        else 0
    )

    # Scene accuracy
    parseable = [
        r
        for r in results
        if r["scene_correct"] is not None
    ]
    correct = sum(
        1 for r in parseable if r["scene_correct"]
    )
    accuracy = (
        correct / len(parseable) if parseable else 0
    )
    unparsed = n - len(parseable)

    # Accuracy by scene type
    scene_acc: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    for r in results:
        if r["scene_correct"] is not None:
            scene_acc[r["gt_scene"]]["total"] += 1
            if r["scene_correct"]:
                scene_acc[r["gt_scene"]]["correct"] += 1

    # Latency
    avg_latency = (
        sum(r["latency_s"] for r in results) / n
    )

    # Errors
    errors = sum(1 for r in results if r["error"])

    per_scene = {}
    for sc, v in sorted(scene_acc.items()):
        acc_val = (
            round(v["correct"] / v["total"], 3)
            if v["total"] > 0
            else 0
        )
        per_scene[sc] = {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": acc_val,
        }

    return {
        "n_samples": n,
        "tool_call_rate": round(tool_call_rate, 3),
        "n_with_tools": with_tools,
        "tool_selection": dict(
            tool_counter.most_common()
        ),
        "avg_tools_when_used": round(
            avg_tools_when_used, 2
        ),
        "tool_quality_rate": round(quality_rate, 3),
        "scene_accuracy": round(accuracy, 3),
        "n_correct": correct,
        "n_parseable": len(parseable),
        "n_unparsed": unparsed,
        "accuracy_by_scene": per_scene,
        "avg_latency_s": round(avg_latency, 2),
        "n_errors": errors,
    }


def print_comparison_table(
    all_metrics: dict[str, dict[str, Any]],
) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("LESSON 3: TOOL CHOICE COMPARISON TABLE")
    print("=" * 80)

    cols = (
        f"{'Model':>5s} | {'Condition':>14s} | "
        f"{'TRate':>5s} | {'AvgTC':>5s} | "
        f"{'Qual':>5s} | {'Acc':>10s} | "
        f"{'Unp':>3s} | {'Lat':>5s}"
    )
    print(cols)
    print("-" * len(cols))

    for key in sorted(all_metrics.keys()):
        m = all_metrics[key]
        parts = key.split("|")
        model = parts[0] if parts else "?"
        cond = parts[1] if len(parts) > 1 else "?"
        n_c = m["n_correct"]
        n_p = m["n_parseable"]
        acc_s = f"{n_c}/{n_p} {m['scene_accuracy']:.0%}"
        print(
            f"{model:>5s} | {cond:>14s} | "
            f"{m['tool_call_rate']:>4.0%} | "
            f"{m['avg_tools_when_used']:>5.1f} | "
            f"{m['tool_quality_rate']:>4.0%} | "
            f"{acc_s:>10s} | "
            f"{m['n_unparsed']:>3d} | "
            f"{m['avg_latency_s']:>4.1f}s"
        )

    # Tool selection breakdown
    print("\n" + "-" * 80)
    print("TOOL SELECTION BREAKDOWN")
    print("-" * 80)
    col2 = (
        f"{'Model':>5s} | {'Condition':>14s} | "
        f"{'prior':>6s} | {'confuse':>7s} | "
        f"{'action':>6s} | {'waypt':>5s}"
    )
    print(col2)
    print("-" * len(col2))

    for key in sorted(all_metrics.keys()):
        m = all_metrics[key]
        parts = key.split("|")
        model = parts[0] if parts else "?"
        cond = parts[1] if len(parts) > 1 else "?"
        ts = m.get("tool_selection", {})
        print(
            f"{model:>5s} | {cond:>14s} | "
            f"{ts.get('check_scene_prior', 0):>6d} | "
            f"{ts.get('check_confusion_risk', 0):>7d} | "
            f"{ts.get('check_scene_action_compatibility', 0):>6d} | "  # noqa: E501
            f"{ts.get('check_waypoint_feasibility', 0):>5d}"
        )

    # Metacognitive effect
    print("\n" + "-" * 80)
    print("METACOGNITIVE EFFECT (Condition C vs A)")
    print("-" * 80)
    for model in ["2B", "8B"]:
        key_a = f"{model}|A_auto"
        key_c = f"{model}|C_metacognitive"
        if key_a in all_metrics and key_c in all_metrics:
            ma = all_metrics[key_a]
            mc = all_metrics[key_c]
            dr = mc["tool_call_rate"] - ma["tool_call_rate"]
            da = mc["scene_accuracy"] - ma["scene_accuracy"]
            dt = (
                mc["avg_tools_when_used"]
                - ma["avg_tools_when_used"]
            )
            print(
                f"  {model}: tool_rate {dr:+.1%} | "
                f"accuracy {da:+.1%} | "
                f"avg_tools {dt:+.2f}"
            )

    print("=" * 80)


def main() -> None:
    """Entry point for Lesson 3 experiment."""
    ts = datetime.now(tz=timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    print(f"Lesson 3: Tool Choice Experiment -- {ts}")
    print(
        "NOTE: Text-only mode (scene descriptions from DB)."
    )
    print("      Images are NOT loaded from MDS dataset.")
    print()

    # ----------------------------------------------------------
    # 1. Load samples
    # ----------------------------------------------------------
    print("Loading samples from self-consistency DB...")
    samples = select_samples(SC_DB_PATH, n=20)
    if len(samples) < 20:
        print(
            f"WARNING: Only {len(samples)} samples "
            f"(wanted 20)"
        )

    # ----------------------------------------------------------
    # 2. Start servers
    # ----------------------------------------------------------
    print(
        f"\nStarting 2B server on GPU {GPU_2B}, "
        f"port {PORT_2B}..."
    )
    server_2b = VLLMServer(
        model_path=MODEL_2B,
        port=PORT_2B,
        gpu_id=GPU_2B,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enable_tools=True,
    )

    print(
        f"Starting 8B server on GPU {GPU_8B}, "
        f"port {PORT_8B}..."
    )
    server_8b = VLLMServer(
        model_path=MODEL_8B,
        port=PORT_8B,
        gpu_id=GPU_8B,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        enable_tools=True,
    )

    all_results: list[dict[str, Any]] = []
    all_metrics: dict[str, dict[str, Any]] = {}

    try:
        server_2b.start(timeout=420)
        server_8b.start(timeout=420)

        servers = {"2B": server_2b, "8B": server_8b}

        conditions = [
            ("A_auto", "auto", PROMPT_A_B),
            ("B_required", "required", PROMPT_A_B),
            ("C_metacognitive", "auto", PROMPT_C),
        ]

        for model_name, server in servers.items():
            for cond_name, tc, tpl in conditions:
                cond_results = run_condition(
                    server=server,
                    model_name=model_name,
                    samples=samples,
                    condition=cond_name,
                    tool_choice=tc,
                    prompt_template=tpl,
                )
                all_results.extend(cond_results)

                key = f"{model_name}|{cond_name}"
                metrics = compute_metrics(cond_results)
                all_metrics[key] = metrics
                print(
                    f"  >> {key}: "
                    f"acc={metrics['scene_accuracy']:.0%}"
                    f", tool_rate="
                    f"{metrics['tool_call_rate']:.0%}"
                )

    finally:
        print("\nStopping servers...")
        server_2b.stop()
        server_8b.stop()

    # ----------------------------------------------------------
    # 4. Print comparison table
    # ----------------------------------------------------------
    print_comparison_table(all_metrics)

    # ----------------------------------------------------------
    # 5. Save results
    # ----------------------------------------------------------
    output = {
        "experiment": "lesson3_tool_choice",
        "timestamp": ts,
        "note": (
            "Text-only mode: images not loaded from MDS "
            "dataset. Scene descriptions from "
            "self-consistency DB are used instead."
        ),
        "n_samples": len(samples),
        "n_requests": len(all_results),
        "samples": [
            {
                "sample_id": s["sample_id"],
                "gt_scene": s["scene_type_gt"],
                "fine_class": s.get("fine_class"),
                "original_prediction": s.get(
                    "predicted_scene"
                ),
                "generated_text": s.get(
                    "generated_text"
                ),
            }
            for s in samples
        ],
        "metrics": all_metrics,
        "detailed_results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print(f"  Total requests: {len(all_results)}")
    print(f"  Results file: {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
