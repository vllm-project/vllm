#!/usr/bin/env python3
# ruff: noqa: E501,E402,I001
# type: ignore[import-not-found]
"""Phase 2: Zoom Tool Deep Dive. Run independently with: nohup python3 -u run_phase2_zoom.py &

Tasks:
    16: Zoom on false IZ predictions (2B, 100 samples)
    17: Zoom on false IZ predictions (8B, 100 samples)
    19: Multiple zooms up to 3 rounds (2B, 100 samples)
    20: Zoom on correct predictions -- harm rate (2B, 100 samples)
    21: Directed vs free zoom (2B, 100 false IZ, 2 conditions)
    25: Token cost analysis (aggregated from all prior tasks)

Servers: GPU 0 (2B port 8400), GPU 1 (8B port 8401)
Saves to: phase2_zoom_final.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
_PARENT = os.path.dirname(_DIR)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _PARENT]

from robust_runner import (  # type: ignore[import-not-found]
    SYSTEM_PROMPT,
    RobustServer,
    load_samples,
    run_experiment,
)
from visual_tools import TOOL_ZOOM, zoom_region  # type: ignore[import-not-found]

DB_PATH = "/workspace/vllm/self_consistency_experiment/self_consistency.db"
RESULTS_PATH = os.path.join(_DIR, "phase2_zoom_final.json")

# Server config
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_2B = 0
GPU_8B = 1
PORT_2B = 8400
PORT_8B = 8401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_all(all_results: dict[str, Any]) -> None:
    """Save all results to disk."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  [saved to {RESULTS_PATH}]")


def _compute_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy metrics from results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "accuracy": 0}

    correct = 0
    baseline_correct = 0
    used_zoom = 0

    for r in results:
        pred = r.get("final_prediction", {})
        predicted_scene = pred.get("scene") if isinstance(pred, dict) else None
        gt_scene = r.get("gt_scene")
        original = r.get("original_pred")

        if predicted_scene and predicted_scene == gt_scene:
            correct += 1
        if original and original == gt_scene:
            baseline_correct += 1
        if r.get("num_tool_calls", 0) > 0:
            used_zoom += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": round(baseline_correct / total, 4),
        "delta": round((correct - baseline_correct) / total, 4),
        "used_zoom": used_zoom,
        "zoom_rate": round(used_zoom / total, 4),
    }


def _compute_harm_rate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute harm rate: how often zoom flips correct -> wrong."""
    baseline_correct = 0
    still_correct = 0
    flipped_wrong = 0

    for r in results:
        pred = r.get("final_prediction", {})
        predicted_scene = pred.get("scene") if isinstance(pred, dict) else None
        gt_scene = r.get("gt_scene")
        original = r.get("original_pred")

        was_correct = original and original == gt_scene
        if was_correct:
            baseline_correct += 1
            if predicted_scene and predicted_scene == gt_scene:
                still_correct += 1
            elif predicted_scene:
                flipped_wrong += 1

    return {
        "baseline_correct": baseline_correct,
        "still_correct": still_correct,
        "flipped_wrong": flipped_wrong,
        "harm_rate": round(flipped_wrong / baseline_correct, 4) if baseline_correct > 0 else 0,
        "retention_rate": round(still_correct / baseline_correct, 4) if baseline_correct > 0 else 0,
    }


def _compute_token_stats(task_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate token cost analysis across tasks."""
    stats: dict[str, Any] = {}
    for task_name, results in task_results.items():
        total_latency = sum(r.get("latency_ms", 0) for r in results if "error" not in r)
        total_gen = sum(r.get("generation_ms", 0) for r in results if "error" not in r)
        total_tool = sum(r.get("tool_execution_ms", 0) for r in results if "error" not in r)
        total_zooms = sum(r.get("num_tool_calls", 0) for r in results if "error" not in r)
        n = len(results)
        acc = _compute_accuracy(results)
        stats[task_name] = {
            "n_samples": n,
            "total_zooms": total_zooms,
            "avg_latency_ms": round(total_latency / n, 1) if n > 0 else 0,
            "avg_generation_ms": round(total_gen / n, 1) if n > 0 else 0,
            "avg_tool_ms": round(total_tool / n, 1) if n > 0 else 0,
            "accuracy": acc["accuracy"],
            "delta": acc["delta"],
        }
    return stats


# ---------------------------------------------------------------------------
# Prompt functions
# ---------------------------------------------------------------------------
ZOOM_CLASSIFY_PROMPT = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "If you think it might be an incident or unusual situation, "
    "zoom into the area of concern to verify before making your "
    "final classification.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

MULTI_ZOOM_PROMPT = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "You can zoom into up to 3 different regions. Investigate "
    "the scene thoroughly before classifying.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

CORRECT_ZOOM_PROMPT = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "If you want to examine any part of the scene more closely, "
    "you can zoom into a region.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

FREE_ZOOM_PROMPT = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "Zoom into any region you think is relevant before classifying.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

DIRECTED_ZOOM_PROMPT = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "Zoom into the center of the road ahead (approximately x=252, y=200) "
    "before classifying.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)


def _prompt_classify(sample: dict[str, Any]) -> tuple[str, str]:
    return SYSTEM_PROMPT, ZOOM_CLASSIFY_PROMPT


def _prompt_multi_zoom(sample: dict[str, Any]) -> tuple[str, str]:
    return SYSTEM_PROMPT, MULTI_ZOOM_PROMPT


def _prompt_correct(sample: dict[str, Any]) -> tuple[str, str]:
    return SYSTEM_PROMPT, CORRECT_ZOOM_PROMPT


def _prompt_free(sample: dict[str, Any]) -> tuple[str, str]:
    return SYSTEM_PROMPT, FREE_ZOOM_PROMPT


def _prompt_directed(sample: dict[str, Any]) -> tuple[str, str]:
    return SYSTEM_PROMPT, DIRECTED_ZOOM_PROMPT


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: Zoom Tool Deep Dive (Tasks 16, 17, 19, 20, 21, 25)")
    print("=" * 70)

    all_results: dict[str, Any] = {
        "experiment": "phase2_zoom_final",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    task_result_lists: dict[str, list[dict[str, Any]]] = {}

    zoom_tools = {"zoom_region": zoom_region}
    zoom_defs = [TOOL_ZOOM]

    # --- Load samples ---
    print("\nLoading samples...")
    false_iz = load_samples(DB_PATH, "false_iz", 100)
    correct_samples = load_samples(DB_PATH, "correct", 100)
    print(f"Loaded {len(false_iz)} false IZ, {len(correct_samples)} correct")

    # === 2B Tasks — SKIPPED (already completed) ===
    SKIP_2B = True
    if not SKIP_2B:
        print("\n--- Starting 2B server ---")
        server_2b = RobustServer(MODEL_2B, gpu_id=GPU_2B, port=PORT_2B)
        server_2b.start(timeout=1800)

        try:
            # Task 16: Zoom on false IZ (2B)
            print("\n=== Task 16: Zoom on false IZ (2B, 100 samples) ===")
            t16 = run_experiment(
                "task16", server_2b, false_iz, _prompt_classify,
                zoom_tools, zoom_defs, max_rounds=1, save_path=RESULTS_PATH,
            )
            acc16 = _compute_accuracy(t16)
            all_results["task16"] = {"description": "Zoom false IZ 2B", "accuracy": acc16}
            task_result_lists["task16"] = t16
            print(f"  T16: accuracy={acc16['accuracy']:.4f} (baseline={acc16['baseline_accuracy']:.4f})")
            _save_all(all_results)

            # Task 19: Multiple zooms (2B)
            print("\n=== Task 19: Multiple zooms (2B, 100 samples) ===")
            t19 = run_experiment(
                "task19", server_2b, false_iz, _prompt_multi_zoom,
                zoom_tools, zoom_defs, max_rounds=3, save_path=RESULTS_PATH,
            )
            acc19 = _compute_accuracy(t19)
            all_results["task19"] = {"description": "Multi-zoom 2B", "accuracy": acc19}
            task_result_lists["task19"] = t19
            print(f"  T19: accuracy={acc19['accuracy']:.4f}")
            _save_all(all_results)

            # Task 20: Harm rate on correct (2B)
            print("\n=== Task 20: Zoom harm rate (2B, 100 correct samples) ===")
            t20 = run_experiment(
                "task20", server_2b, correct_samples, _prompt_correct,
                zoom_tools, zoom_defs, max_rounds=1, save_path=RESULTS_PATH,
            )
            acc20 = _compute_accuracy(t20)
            harm20 = _compute_harm_rate(t20)
            all_results["task20"] = {
                "description": "Zoom harm rate 2B",
                "accuracy": acc20,
                "harm_rate": harm20,
            }
            task_result_lists["task20"] = t20
            print(f"  T20: harm_rate={harm20['harm_rate']:.4f}, retention={harm20['retention_rate']:.4f}")
            _save_all(all_results)

            # Task 21a: Free zoom (2B)
            print("\n=== Task 21a: Free zoom (2B, 100 false IZ) ===")
            t21a = run_experiment(
                "task21a_free", server_2b, false_iz, _prompt_free,
                zoom_tools, zoom_defs, max_rounds=1, save_path=RESULTS_PATH,
            )
            acc21a = _compute_accuracy(t21a)
            task_result_lists["task21a"] = t21a

            # Task 21b: Directed zoom (2B)
            print("\n=== Task 21b: Directed zoom (2B, 100 false IZ) ===")
            t21b = run_experiment(
                "task21b_directed", server_2b, false_iz, _prompt_directed,
                zoom_tools, zoom_defs, max_rounds=1, save_path=RESULTS_PATH,
            )
            acc21b = _compute_accuracy(t21b)
            task_result_lists["task21b"] = t21b

            all_results["task21"] = {
                "description": "Directed vs free zoom 2B",
                "free_accuracy": acc21a,
                "directed_accuracy": acc21b,
                "delta": round(acc21b["accuracy"] - acc21a["accuracy"], 4),
            }
            print(f"  T21: free={acc21a['accuracy']:.4f}, directed={acc21b['accuracy']:.4f}")
            _save_all(all_results)

        finally:
            print("\n--- Stopping 2B server ---")
            server_2b.stop()
            time.sleep(5)

    # === 8B Tasks (GPU 1) ===
    print("\n--- Starting 8B server ---")
    server_8b = RobustServer(MODEL_8B, gpu_id=GPU_8B, port=PORT_8B)
    server_8b.start(timeout=1800)

    try:
        # Task 17: Zoom on false IZ (8B)
        print("\n=== Task 17: Zoom on false IZ (8B, 100 samples) ===")
        t17 = run_experiment(
            "task17", server_8b, false_iz, _prompt_classify,
            zoom_tools, zoom_defs, max_rounds=1, save_path=RESULTS_PATH,
        )
        acc17 = _compute_accuracy(t17)
        all_results["task17"] = {"description": "Zoom false IZ 8B", "accuracy": acc17}
        task_result_lists["task17"] = t17
        print(f"  T17: accuracy={acc17['accuracy']:.4f} (baseline={acc17['baseline_accuracy']:.4f})")
        _save_all(all_results)

    finally:
        print("\n--- Stopping 8B server ---")
        server_8b.stop()

    # === Task 25: Token cost analysis ===
    print("\n=== Task 25: Token cost analysis ===")
    token_stats = _compute_token_stats(task_result_lists)
    all_results["task25"] = {
        "description": "Token cost analysis",
        "per_task": token_stats,
    }
    _save_all(all_results)

    elapsed = time.time() - start_time
    print(f"\n=== ALL TASKS COMPLETE ({elapsed / 60:.1f} min) ===")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
