#!/usr/bin/env python3
"""Phase 3: Multi-Tool Interaction Experiments (Tasks 51-70).

Runs experiments that test how well models use multiple visual tools
together for driving scene analysis.

Tasks:
    51: All tools available, no guidance (2B, 100 mixed samples)
    54: Tool subset ablations (2B, 100 mixed samples x 4 conditions)
    57: Tools on correct predictions -- harm rate (2B, 100 correct samples)
    60: 2B vs 8B with all tools (100 mixed samples, both models)
    66: Pipeline A -- FT predict, base verify (2B, 100 mixed samples)
    67: Pipeline B -- base + tools from scratch (2B, 100 mixed samples)
    68: Pipeline A vs B head-to-head (same 100 samples)
    69: Error taxonomy (from all Phase 3 data)
    70: Cost-benefit summary

Usage:
    python3 run_phase3.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Path setup -- must happen before local imports
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
# Remove /workspace/vllm from sys.path to avoid shadowing installed vllm
if os.path.isdir(os.path.join(_PARENT, "vllm")):
    sys.path[:] = [
        p for p in sys.path
        if os.path.abspath(p or os.getcwd()) != _PARENT
    ]

from orchestrator import (  # noqa: E402  # type: ignore[import-not-found]
    ToolCallingOrchestrator,
)
from robust_runner import (  # noqa: E402  # type: ignore[import-not-found]
    SYSTEM_PROMPT,
    RobustServer,
    load_samples,
)
from visual_tools import (  # noqa: E402  # type: ignore[import-not-found]
    ALL_VISUAL_TOOLS,
    TOOL_ROAD_GEOMETRY,
    TOOL_WAYPOINT_VIZ,
    TOOL_ZOOM,
    analyze_road_geometry,
    find_similar_scenes,
    visualize_waypoint,
    zoom_region,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
DB_PATH = os.path.join(
    os.path.dirname(_DIR),
    "self_consistency_experiment",
    "self_consistency.db",
)
RESULTS_PATH = os.path.join(_DIR, "phase3_results.json")

PORT_2B = 8450
PORT_8B = 8451
GPU_2B = 0
GPU_8B = 1

# All 4 visual tools as callables
ALL_TOOL_CALLABLES: dict[str, Any] = {
    "zoom_region": zoom_region,
    "visualize_waypoint": visualize_waypoint,
    "analyze_road_geometry": analyze_road_geometry,
    "find_similar_scenes": find_similar_scenes,
}

# Prompt templates
PROMPT_NO_GUIDANCE = (
    "Analyze this driving scene and predict the scene type, action, and waypoint. "
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger.\n\n"
    "Output your final answer in this format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <stop|slowdown|proceed|null>\n"
    "FINAL_LAT_ACTION: <lc_left|lc_right|null>"
)

PROMPT_VERIFY = (
    "A model predicted this scene as: {scene}. "
    "Use the available tools to verify whether this prediction is correct. "
    "Look for visual evidence that supports or contradicts the prediction.\n\n"
    "After your analysis, output your final answer in this format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <stop|slowdown|proceed|null>\n"
    "FINAL_LAT_ACTION: <lc_left|lc_right|null>"
)

PROMPT_FROM_SCRATCH = (
    "Classify this driving scene and predict the appropriate action using "
    "the available tools. Scene types: nominal, flooded, incident_zone, "
    "mounted_police, flagger.\n\n"
    "Use the tools to gather evidence before making your prediction.\n\n"
    "Output your final answer in this format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <stop|slowdown|proceed|null>\n"
    "FINAL_LAT_ACTION: <lc_left|lc_right|null>"
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def save_results(all_results: dict[str, Any]) -> None:
    """Save results to disk, merging with existing."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {RESULTS_PATH}")


def compute_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy metrics from a list of result dicts."""
    total = len(results)
    if total == 0:
        return {"total": 0, "accuracy": 0.0}

    correct = 0
    parsed = 0
    scene_correct: dict[str | None, list[bool]] = {}

    for r in results:
        pred = r.get("final_prediction", {})
        gt: str | None = r.get("gt_scene")
        pred_scene = pred.get("scene") if isinstance(pred, dict) else None

        if pred_scene is not None:
            parsed += 1
            is_correct = pred_scene == gt
            correct += int(is_correct)
            scene_correct.setdefault(gt, []).append(is_correct)

    per_scene: dict[str | None, dict[str, Any]] = {}
    for scene, vals in scene_correct.items():
        per_scene[scene] = {
            "total": len(vals),
            "correct": sum(vals),
            "accuracy": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }

    return {
        "total": total,
        "parsed": parsed,
        "parse_rate": round(parsed / total, 4) if total else 0.0,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "accuracy_of_parsed": round(correct / parsed, 4) if parsed else 0.0,
        "per_scene": per_scene,
    }


def compute_tool_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute tool usage statistics from results."""
    tool_counts: Counter[str] = Counter()
    tool_per_sample: list[int] = []

    for r in results:
        calls = r.get("tool_calls", [])
        n = len(calls)
        tool_per_sample.append(n)
        names = [c.get("tool_name", "unknown") for c in calls]
        for name in names:
            tool_counts[name] += 1

    avg_calls = (
        round(sum(tool_per_sample) / len(tool_per_sample), 2)
        if tool_per_sample
        else 0.0
    )
    samples_using_tools = sum(1 for n in tool_per_sample if n > 0)

    return {
        "tool_usage_counts": dict(tool_counts),
        "avg_tool_calls_per_sample": avg_calls,
        "samples_using_tools": samples_using_tools,
        "samples_total": len(results),
        "tool_adoption_rate": (
            round(samples_using_tools / len(results), 4) if results else 0.0
        ),
    }


def run_orchestrator_on_samples(
    server_url: str,
    samples: list[dict[str, Any]],
    prompt: str,
    tool_callables: dict[str, Any],
    tool_defs: list[dict[str, Any]],
    task_id: str,
    tool_choice: str = "auto",
    max_rounds: int = 5,
) -> list[dict[str, Any]]:
    """Run the ToolCallingOrchestrator on a list of samples.

    Returns a list of result dicts with sample metadata attached.
    """
    results: list[dict[str, Any]] = []
    total = len(samples)

    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"  [{task_id}] {i}/{total}...")

        orch = ToolCallingOrchestrator(
            server_url=server_url,
            tools=tool_callables,
            tool_definitions=tool_defs,
            max_tool_rounds=max_rounds,
            temperature=0,
            max_tokens=1024,
        )

        image_path = sample.get("image_path")
        # For verify prompts, fill in the prediction
        user_prompt = prompt.format(
            scene=sample.get("predicted_scene", "unknown"),
            long_action=sample.get("predicted_long_action", "null"),
            lat_action=sample.get("predicted_lat_action", "null"),
        )

        try:
            result = orch.run(
                image_path=image_path,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tool_choice=tool_choice,
            )
        except Exception as e:
            print(f"  [{task_id}] Sample {i} error: {e}")
            result = {"error": str(e)}

        # Attach sample metadata
        result["sample_id"] = sample.get("sample_id")
        result["gt_scene"] = sample.get("scene_type_gt")
        result["gt_long_action"] = sample.get("long_action_gt")
        result["gt_lat_action"] = sample.get("lat_action_gt")
        result["original_pred"] = sample.get("predicted_scene")
        result["original_correct"] = (
            sample.get("predicted_scene") == sample.get("scene_type_gt")
        )
        results.append(result)

        # Incremental save every 25 samples
        if (i + 1) % 25 == 0:
            _incremental_save(task_id, results)

    print(f"  [{task_id}] COMPLETE: {total} samples")
    return results


def _incremental_save(task_id: str, results: list[dict[str, Any]]) -> None:
    """Save partial results incrementally."""
    existing: dict[str, Any] = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing[task_id] = {
        "n_samples": len(results),
        "n_errors": sum(1 for r in results if "error" in r and r["error"]),
        "results": results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------


def run_task51(
    server_url: str, samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task 51: All tools available, no guidance (2B, 100 mixed samples)."""
    print("\n" + "=" * 70)
    print("TASK 51: All tools, no guidance (2B, 100 mixed)")
    print("=" * 70)

    results = run_orchestrator_on_samples(
        server_url=server_url,
        samples=samples,
        prompt=PROMPT_NO_GUIDANCE,
        tool_callables=ALL_TOOL_CALLABLES,
        tool_defs=ALL_VISUAL_TOOLS,
        task_id="task51",
        tool_choice="auto",
        max_rounds=5,
    )

    acc = compute_accuracy(results)
    tool_stats = compute_tool_stats(results)

    summary: dict[str, Any] = {
        "description": "All 4 visual tools, no guidance, 2B model",
        "n_samples": len(results),
        "accuracy": acc,
        "tool_stats": tool_stats,
        "results": results,
    }

    print("\n  Task 51 Summary:")
    print(f"    Accuracy: {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})")
    print(f"    Parse rate: {acc['parse_rate']:.1%}")
    print(f"    Tool adoption: {tool_stats['tool_adoption_rate']:.1%}")
    print(f"    Avg tool calls: {tool_stats['avg_tool_calls_per_sample']}")
    print(f"    Tool counts: {tool_stats['tool_usage_counts']}")

    return summary


def run_task54(
    server_url: str, samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task 54: Tool subset ablations (2B, 100 samples x 4 conditions)."""
    print("\n" + "=" * 70)
    print("TASK 54: Tool subset ablations (2B, 4 conditions)")
    print("=" * 70)

    conditions: dict[str, dict[str, Any]] = {
        "zoom_only": {
            "callables": {"zoom_region": zoom_region},
            "defs": [TOOL_ZOOM],
        },
        "waypoint_viz_only": {
            "callables": {"visualize_waypoint": visualize_waypoint},
            "defs": [TOOL_WAYPOINT_VIZ],
        },
        "geometry_only": {
            "callables": {"analyze_road_geometry": analyze_road_geometry},
            "defs": [TOOL_ROAD_GEOMETRY],
        },
        "all_tools": {
            "callables": ALL_TOOL_CALLABLES,
            "defs": ALL_VISUAL_TOOLS,
        },
    }

    condition_results: dict[str, Any] = {}

    for cond_name, cond_config in conditions.items():
        print(f"\n  --- Condition: {cond_name} ---")
        results = run_orchestrator_on_samples(
            server_url=server_url,
            samples=samples,
            prompt=PROMPT_NO_GUIDANCE,
            tool_callables=cond_config["callables"],
            tool_defs=cond_config["defs"],
            task_id=f"task54_{cond_name}",
            tool_choice="auto",
            max_rounds=5,
        )

        acc = compute_accuracy(results)
        tool_stats = compute_tool_stats(results)

        condition_results[cond_name] = {
            "accuracy": acc,
            "tool_stats": tool_stats,
            "results": results,
        }

        print(
            f"    {cond_name}: accuracy={acc['accuracy']:.1%}, "
            f"tool_adoption={tool_stats['tool_adoption_rate']:.1%}"
        )

    # Comparison
    comparison: dict[str, Any] = {}
    for cond_name, cond_data in condition_results.items():
        comparison[cond_name] = {
            "accuracy": cond_data["accuracy"]["accuracy"],
            "parse_rate": cond_data["accuracy"]["parse_rate"],
            "tool_adoption": cond_data["tool_stats"]["tool_adoption_rate"],
            "avg_tool_calls": cond_data["tool_stats"]["avg_tool_calls_per_sample"],
        }

    summary: dict[str, Any] = {
        "description": "Tool subset ablation study: zoom, waypoint, geometry, all",
        "n_samples_per_condition": len(samples),
        "conditions": condition_results,
        "comparison": comparison,
    }

    print("\n  Task 54 Comparison:")
    for name, stats in comparison.items():
        print(
            f"    {name:20s}: acc={stats['accuracy']:.1%}, "
            f"tools={stats['avg_tool_calls']:.1f}"
        )

    return summary


def run_task57(
    server_url: str, correct_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task 57: Tools on correct predictions -- measure harm rate."""
    print("\n" + "=" * 70)
    print("TASK 57: Tools on correct predictions -- harm rate (2B)")
    print("=" * 70)

    results = run_orchestrator_on_samples(
        server_url=server_url,
        samples=correct_samples,
        prompt=PROMPT_NO_GUIDANCE,
        tool_callables=ALL_TOOL_CALLABLES,
        tool_defs=ALL_VISUAL_TOOLS,
        task_id="task57",
        tool_choice="auto",
        max_rounds=5,
    )

    # Measure harm: how often does the model change a correct prediction to wrong
    total = len(results)
    still_correct = 0
    now_wrong = 0
    unparsed = 0

    for r in results:
        pred = r.get("final_prediction", {})
        gt = r.get("gt_scene")
        pred_scene = pred.get("scene") if isinstance(pred, dict) else None

        if pred_scene is None:
            unparsed += 1
        elif pred_scene == gt:
            still_correct += 1
        else:
            now_wrong += 1

    harm_rate = round(now_wrong / total, 4) if total else 0.0
    retention_rate = round(still_correct / total, 4) if total else 0.0

    acc = compute_accuracy(results)
    tool_stats = compute_tool_stats(results)

    summary: dict[str, Any] = {
        "description": "Tools on already-correct predictions: does it cause harm?",
        "n_samples": total,
        "still_correct": still_correct,
        "now_wrong": now_wrong,
        "unparsed": unparsed,
        "harm_rate": harm_rate,
        "retention_rate": retention_rate,
        "accuracy": acc,
        "tool_stats": tool_stats,
        "results": results,
    }

    print("\n  Task 57 Summary:")
    print(f"    Total correct samples: {total}")
    print(f"    Still correct after tools: {still_correct} ({retention_rate:.1%})")
    print(f"    Now wrong (harmed): {now_wrong} ({harm_rate:.1%})")
    print(f"    Unparsed: {unparsed}")

    return summary


def run_task60(
    server_url_2b: str,
    server_url_8b: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task 60: 2B vs 8B with all tools (100 mixed samples)."""
    print("\n" + "=" * 70)
    print("TASK 60: 2B vs 8B with all tools (100 mixed)")
    print("=" * 70)

    # Run 2B
    print("\n  --- 2B model ---")
    results_2b = run_orchestrator_on_samples(
        server_url=server_url_2b,
        samples=samples,
        prompt=PROMPT_NO_GUIDANCE,
        tool_callables=ALL_TOOL_CALLABLES,
        tool_defs=ALL_VISUAL_TOOLS,
        task_id="task60_2b",
        tool_choice="auto",
        max_rounds=5,
    )

    # Run 8B
    print("\n  --- 8B model ---")
    results_8b = run_orchestrator_on_samples(
        server_url=server_url_8b,
        samples=samples,
        prompt=PROMPT_NO_GUIDANCE,
        tool_callables=ALL_TOOL_CALLABLES,
        tool_defs=ALL_VISUAL_TOOLS,
        task_id="task60_8b",
        tool_choice="auto",
        max_rounds=5,
    )

    acc_2b = compute_accuracy(results_2b)
    acc_8b = compute_accuracy(results_8b)
    tools_2b = compute_tool_stats(results_2b)
    tools_8b = compute_tool_stats(results_8b)

    summary: dict[str, Any] = {
        "description": "2B vs 8B with all visual tools",
        "n_samples": len(samples),
        "model_2b": {
            "accuracy": acc_2b,
            "tool_stats": tools_2b,
            "results": results_2b,
        },
        "model_8b": {
            "accuracy": acc_8b,
            "tool_stats": tools_8b,
            "results": results_8b,
        },
        "comparison": {
            "accuracy_2b": acc_2b["accuracy"],
            "accuracy_8b": acc_8b["accuracy"],
            "accuracy_delta": round(acc_8b["accuracy"] - acc_2b["accuracy"], 4),
            "avg_tools_2b": tools_2b["avg_tool_calls_per_sample"],
            "avg_tools_8b": tools_8b["avg_tool_calls_per_sample"],
            "tool_adoption_2b": tools_2b["tool_adoption_rate"],
            "tool_adoption_8b": tools_8b["tool_adoption_rate"],
        },
    }

    print("\n  Task 60 Comparison:")
    c = summary["comparison"]
    print(
        f"    2B accuracy: {c['accuracy_2b']:.1%}, "
        f"avg tools: {c['avg_tools_2b']:.1f}"
    )
    print(
        f"    8B accuracy: {c['accuracy_8b']:.1%}, "
        f"avg tools: {c['avg_tools_8b']:.1f}"
    )
    print(f"    Delta (8B - 2B): {c['accuracy_delta']:+.1%}")

    return summary


def run_task66(
    server_url: str, samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task 66: Pipeline A -- FT predict, base verify."""
    print("\n" + "=" * 70)
    print("TASK 66: Pipeline A -- FT predict, base + tools verify (2B)")
    print("=" * 70)

    results = run_orchestrator_on_samples(
        server_url=server_url,
        samples=samples,
        prompt=PROMPT_VERIFY,
        tool_callables=ALL_TOOL_CALLABLES,
        tool_defs=ALL_VISUAL_TOOLS,
        task_id="task66",
        tool_choice="auto",
        max_rounds=5,
    )

    acc = compute_accuracy(results)
    tool_stats = compute_tool_stats(results)

    # Track whether verifier agreed with or overrode the FT prediction
    agreed = 0
    overrode = 0
    override_correct = 0
    override_wrong = 0

    for r in results:
        pred = r.get("final_prediction", {})
        original = r.get("original_pred")
        gt = r.get("gt_scene")
        pred_scene = pred.get("scene") if isinstance(pred, dict) else None

        if pred_scene is None:
            continue
        if pred_scene == original:
            agreed += 1
        else:
            overrode += 1
            if pred_scene == gt:
                override_correct += 1
            else:
                override_wrong += 1

    summary: dict[str, Any] = {
        "description": "Pipeline A: FT model predicts, base + tools verifies",
        "n_samples": len(results),
        "accuracy": acc,
        "tool_stats": tool_stats,
        "verification_stats": {
            "agreed_with_ft": agreed,
            "overrode_ft": overrode,
            "override_correct": override_correct,
            "override_wrong": override_wrong,
            "override_precision": (
                round(override_correct / overrode, 4) if overrode else 0.0
            ),
        },
        "results": results,
    }

    print("\n  Task 66 Summary:")
    print(f"    Accuracy: {acc['accuracy']:.1%}")
    print(f"    Agreed with FT: {agreed}, Overrode: {overrode}")
    print(
        f"    Override precision: "
        f"{override_correct}/{overrode} correct overrides"
    )

    return summary


def run_task67(
    server_url: str, samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task 67: Pipeline B -- base + tools from scratch."""
    print("\n" + "=" * 70)
    print("TASK 67: Pipeline B -- base + tools from scratch (2B)")
    print("=" * 70)

    results = run_orchestrator_on_samples(
        server_url=server_url,
        samples=samples,
        prompt=PROMPT_FROM_SCRATCH,
        tool_callables=ALL_TOOL_CALLABLES,
        tool_defs=ALL_VISUAL_TOOLS,
        task_id="task67",
        tool_choice="auto",
        max_rounds=5,
    )

    acc = compute_accuracy(results)
    tool_stats = compute_tool_stats(results)

    # Compare with original FT prediction
    ft_correct = sum(
        1 for r in results
        if r.get("original_pred") == r.get("gt_scene")
    )
    ft_accuracy = round(ft_correct / len(results), 4) if results else 0.0

    summary: dict[str, Any] = {
        "description": "Pipeline B: Base model + tools from scratch (no FT)",
        "n_samples": len(results),
        "accuracy": acc,
        "tool_stats": tool_stats,
        "ft_baseline_accuracy": ft_accuracy,
        "ft_baseline_correct": ft_correct,
        "delta_vs_ft": round(acc["accuracy"] - ft_accuracy, 4),
        "results": results,
    }

    print("\n  Task 67 Summary:")
    print(f"    Base + tools accuracy: {acc['accuracy']:.1%}")
    print(f"    FT baseline accuracy: {ft_accuracy:.1%}")
    print(f"    Delta: {acc['accuracy'] - ft_accuracy:+.1%}")

    return summary


def run_task68(
    task66_results: dict[str, Any],
    task67_results: dict[str, Any],
) -> dict[str, Any]:
    """Task 68: Pipeline A vs B head-to-head."""
    print("\n" + "=" * 70)
    print("TASK 68: Pipeline A vs B head-to-head comparison")
    print("=" * 70)

    results_a = task66_results.get("results", [])
    results_b = task67_results.get("results", [])

    # Build sample_id -> result maps
    map_a = {r.get("sample_id"): r for r in results_a}
    map_b = {r.get("sample_id"): r for r in results_b}

    common_ids = set(map_a.keys()) & set(map_b.keys())
    print(f"  Common samples: {len(common_ids)}")

    both_correct = 0
    a_only_correct = 0
    b_only_correct = 0
    both_wrong = 0
    a_better_cases: list[dict[str, Any]] = []
    b_better_cases: list[dict[str, Any]] = []

    for sid in sorted(common_ids):
        ra = map_a[sid]
        rb = map_b[sid]
        gt = ra.get("gt_scene")

        pred_a = ra.get("final_prediction", {})
        pred_b = rb.get("final_prediction", {})
        scene_a = pred_a.get("scene") if isinstance(pred_a, dict) else None
        scene_b = pred_b.get("scene") if isinstance(pred_b, dict) else None

        a_ok = scene_a == gt
        b_ok = scene_b == gt

        if a_ok and b_ok:
            both_correct += 1
        elif a_ok and not b_ok:
            a_only_correct += 1
            a_better_cases.append({
                "sample_id": sid, "gt": gt,
                "a_pred": scene_a, "b_pred": scene_b,
            })
        elif not a_ok and b_ok:
            b_only_correct += 1
            b_better_cases.append({
                "sample_id": sid, "gt": gt,
                "a_pred": scene_a, "b_pred": scene_b,
            })
        else:
            both_wrong += 1

    total = len(common_ids)
    acc_a = task66_results.get("accuracy", {}).get("accuracy", 0.0)
    acc_b = task67_results.get("accuracy", {}).get("accuracy", 0.0)

    summary: dict[str, Any] = {
        "description": (
            "Pipeline A (FT+verify) vs Pipeline B (base+tools) head-to-head"
        ),
        "n_common_samples": total,
        "both_correct": both_correct,
        "a_only_correct": a_only_correct,
        "b_only_correct": b_only_correct,
        "both_wrong": both_wrong,
        "pipeline_a_accuracy": acc_a,
        "pipeline_b_accuracy": acc_b,
        "a_advantage": a_only_correct - b_only_correct,
        "a_better_examples": a_better_cases[:5],
        "b_better_examples": b_better_cases[:5],
    }

    print("\n  Task 68 Summary:")
    print(f"    Common samples: {total}")
    print(f"    Both correct: {both_correct}")
    print(f"    A only correct: {a_only_correct}")
    print(f"    B only correct: {b_only_correct}")
    print(f"    Both wrong: {both_wrong}")
    print(f"    Pipeline A accuracy: {acc_a:.1%}")
    print(f"    Pipeline B accuracy: {acc_b:.1%}")

    return summary


def _categorize_error(
    tool_calls: list[dict[str, Any]],
    reasoning: str,
    gt: str,
    pred_scene: str,
    error_categories: dict[str, int],
    error_examples: dict[str, list[dict[str, Any]]],
    example: dict[str, Any],
) -> None:
    """Categorize a single error into one of the taxonomy buckets."""
    n_tools = len(tool_calls)

    if n_tools == 0:
        # Category 1: No tool called
        error_categories["no_tool_called"] += 1
        if len(error_examples["no_tool_called"]) < 3:
            error_examples["no_tool_called"].append(example)
        return

    tool_names = [c.get("tool_name", "") for c in tool_calls]
    tool_results = [c.get("result_metadata", {}) for c in tool_calls]

    # Check if tool had error
    has_tool_error = any(
        "error" in str(tr).lower() for tr in tool_results
    )

    if has_tool_error:
        # Category 4: Tool returned bad info
        error_categories["tool_returned_bad_info"] += 1
        if len(error_examples["tool_returned_bad_info"]) < 3:
            error_examples["tool_returned_bad_info"].append(example)
    elif "find_similar_scenes" in tool_names:
        # Check if similar scenes suggested the correct answer
        ignored = False
        for tr in tool_results:
            consensus = tr.get("consensus", {})
            if isinstance(consensus, dict):
                suggested = consensus.get("scene_label")
                if suggested == gt and pred_scene != gt:
                    error_categories["called_but_ignored"] += 1
                    if len(error_examples["called_but_ignored"]) < 3:
                        error_examples["called_but_ignored"].append(example)
                    ignored = True
                    break
        if not ignored:
            error_categories["used_correctly_still_wrong"] += 1
            if len(error_examples["used_correctly_still_wrong"]) < 3:
                error_examples["used_correctly_still_wrong"].append(example)
    elif reasoning and gt and gt in reasoning.lower():
        # Model saw the right answer somewhere but chose wrong
        error_categories["called_but_misinterpreted"] += 1
        if len(error_examples["called_but_misinterpreted"]) < 3:
            error_examples["called_but_misinterpreted"].append(example)
    else:
        error_categories["used_correctly_still_wrong"] += 1
        if len(error_examples["used_correctly_still_wrong"]) < 3:
            error_examples["used_correctly_still_wrong"].append(example)


def run_task69(all_results: dict[str, Any]) -> dict[str, Any]:
    """Task 69: Error taxonomy across all Phase 3 data."""
    print("\n" + "=" * 70)
    print("TASK 69: Error taxonomy")
    print("=" * 70)

    error_categories: dict[str, int] = {
        "no_tool_called": 0,
        "called_but_ignored": 0,
        "called_but_misinterpreted": 0,
        "tool_returned_bad_info": 0,
        "used_correctly_still_wrong": 0,
    }

    error_examples: dict[str, list[dict[str, Any]]] = {
        k: [] for k in error_categories
    }

    tasks_to_analyze = ["task51", "task57", "task66", "task67"]
    total_errors = 0

    for task_key in tasks_to_analyze:
        task_data = all_results.get(task_key, {})
        if not isinstance(task_data, dict):
            continue
        results = task_data.get("results", [])

        for r in results:
            if r.get("error"):
                continue

            pred = r.get("final_prediction", {})
            gt = r.get("gt_scene")
            pred_scene = (
                pred.get("scene") if isinstance(pred, dict) else None
            )

            # Only analyze wrong predictions
            if pred_scene is None or pred_scene == gt:
                continue

            total_errors += 1
            tool_calls = r.get("tool_calls", [])
            reasoning = r.get("reasoning_text", "")

            example: dict[str, Any] = {
                "task": task_key,
                "sample_id": r.get("sample_id"),
                "gt": gt,
                "predicted": pred_scene,
                "n_tools": len(tool_calls),
            }

            _categorize_error(
                tool_calls, reasoning, gt or "",
                pred_scene, error_categories,
                error_examples, example,
            )

    # Compute percentages
    error_pcts: dict[str, dict[str, Any]] = {}
    for cat, count in error_categories.items():
        error_pcts[cat] = {
            "count": count,
            "pct": (
                round(count / total_errors * 100, 1)
                if total_errors
                else 0.0
            ),
        }

    summary: dict[str, Any] = {
        "description": "Error taxonomy across all Phase 3 wrong predictions",
        "total_errors_analyzed": total_errors,
        "tasks_analyzed": tasks_to_analyze,
        "categories": error_pcts,
        "examples": {k: v for k, v in error_examples.items() if v},
    }

    print(f"\n  Task 69 Summary ({total_errors} total errors):")
    for cat, data in error_pcts.items():
        print(f"    {cat:35s}: {data['count']:4d} ({data['pct']:.1f}%)")

    return summary


def run_task70(all_results: dict[str, Any]) -> dict[str, Any]:
    """Task 70: Cost-benefit summary."""
    print("\n" + "=" * 70)
    print("TASK 70: Cost-benefit summary")
    print("=" * 70)

    tasks_to_analyze = {
        "task51": "All tools, no guidance (2B)",
        "task54_zoom_only": "Zoom only (2B)",
        "task54_waypoint_viz_only": "Waypoint viz only (2B)",
        "task54_geometry_only": "Geometry only (2B)",
        "task54_all_tools": "All tools (ablation) (2B)",
        "task57": "Tools on correct preds (2B)",
        "task60_2b": "2B with all tools",
        "task60_8b": "8B with all tools",
        "task66": "Pipeline A: FT+verify (2B)",
        "task67": "Pipeline B: base+tools (2B)",
    }

    # Extract task54 conditions from nested structure
    task54_data = all_results.get("task54", {})
    conditions = (
        task54_data.get("conditions", {})
        if isinstance(task54_data, dict) else {}
    )

    # Build a flat map for analysis
    flat_results: dict[str, dict[str, Any]] = {}

    for task_key, label in tasks_to_analyze.items():
        results_list: list[dict[str, Any]] = []
        if task_key.startswith("task54_"):
            cond = task_key.replace("task54_", "")
            cond_data = conditions.get(cond, {})
            results_list = cond_data.get("results", [])
        elif task_key.startswith("task60_"):
            model_key = task_key.replace("task60_", "model_")
            task60_data = all_results.get("task60", {})
            if isinstance(task60_data, dict):
                model_data = task60_data.get(model_key, {})
                results_list = model_data.get("results", [])
        else:
            task_data = all_results.get(task_key, {})
            if isinstance(task_data, dict):
                results_list = task_data.get("results", [])

        if not results_list:
            continue

        # Compute metrics
        total_latency = 0.0
        total_gen_ms = 0.0
        total_tool_ms = 0.0
        total_tool_calls = 0
        total_rounds = 0
        n_valid = 0

        for r in results_list:
            if r.get("error"):
                continue
            n_valid += 1
            total_latency += r.get("latency_ms", 0)
            total_gen_ms += r.get("generation_ms", 0)
            total_tool_ms += r.get("tool_execution_ms", 0)
            total_tool_calls += r.get("num_tool_calls", 0)
            total_rounds += r.get("num_rounds", 0)

        acc = compute_accuracy(results_list)

        flat_results[task_key] = {
            "label": label,
            "n_samples": len(results_list),
            "accuracy": acc["accuracy"],
            "parse_rate": acc["parse_rate"],
            "avg_latency_ms": (
                round(total_latency / n_valid, 1) if n_valid else 0.0
            ),
            "avg_generation_ms": (
                round(total_gen_ms / n_valid, 1) if n_valid else 0.0
            ),
            "avg_tool_ms": (
                round(total_tool_ms / n_valid, 1) if n_valid else 0.0
            ),
            "avg_tool_calls": (
                round(total_tool_calls / n_valid, 2) if n_valid else 0.0
            ),
            "avg_rounds": (
                round(total_rounds / n_valid, 2) if n_valid else 0.0
            ),
            "total_latency_sec": round(total_latency / 1000, 1),
        }

    # FT baseline for reference
    task67_data = all_results.get("task67", {})
    ft_baseline_acc = (
        task67_data.get("ft_baseline_accuracy", 0.0)
        if isinstance(task67_data, dict) else 0.0
    )

    summary: dict[str, Any] = {
        "description": (
            "Cost-benefit analysis: accuracy vs token/time cost per condition"
        ),
        "conditions": flat_results,
        "ft_baseline_accuracy": ft_baseline_acc,
    }

    header = (
        f"  {'Condition':<35s} {'Acc':>6s} {'Parse':>6s} "
        f"{'AvgMs':>8s} {'Tools':>6s} {'Rounds':>6s}"
    )
    print("\n  Task 70 Cost-Benefit Table:")
    print(header)
    print(f"  {'-' * 35} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 6}")
    for _key, data in flat_results.items():
        print(
            f"  {data['label']:<35s} "
            f"{data['accuracy']:>5.1%} "
            f"{data['parse_rate']:>5.1%} "
            f"{data['avg_latency_ms']:>7.0f} "
            f"{data['avg_tool_calls']:>5.1f} "
            f"{data['avg_rounds']:>5.1f}"
        )
    if ft_baseline_acc:
        print(f"\n  FT baseline accuracy (no tools): {ft_baseline_acc:.1%}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run all Phase 3 experiments sequentially."""
    start_time = time.time()
    all_results: dict[str, Any] = {}

    # Load existing results if any
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                all_results = json.load(f)
            print(f"Loaded existing results from {RESULTS_PATH}")
        except (json.JSONDecodeError, OSError):
            all_results = {}

    print("=" * 70)
    print("PHASE 3: Multi-Tool Interaction Experiments (Tasks 51-70)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load samples
    # ------------------------------------------------------------------
    print("\nLoading samples from DB...")
    mixed_samples = load_samples(DB_PATH, "mixed", 100)
    correct_samples = load_samples(DB_PATH, "correct", 100)
    print(
        f"  Mixed samples: {len(mixed_samples)} "
        f"(target 100: 50 wrong + 50 correct)"
    )
    print(f"  Correct samples: {len(correct_samples)} (target 100)")

    # ------------------------------------------------------------------
    # Start 2B server
    # ------------------------------------------------------------------
    print(f"\nStarting 2B server on GPU {GPU_2B}, port {PORT_2B}...")
    server_2b = RobustServer(MODEL_2B, gpu_id=GPU_2B, port=PORT_2B)
    server_2b.start(timeout=600)
    url_2b = server_2b.url

    # ------------------------------------------------------------------
    # Task 51: All tools, no guidance
    # ------------------------------------------------------------------
    try:
        t51 = run_task51(url_2b, mixed_samples)
        all_results["task51"] = t51
        save_results(all_results)
    except Exception as e:
        print(f"  Task 51 FAILED: {e}")
        traceback.print_exc()
        all_results["task51"] = {"error": str(e)}
        save_results(all_results)

    # ------------------------------------------------------------------
    # Task 54: Tool subset ablations
    # ------------------------------------------------------------------
    try:
        t54 = run_task54(url_2b, mixed_samples)
        all_results["task54"] = t54
        save_results(all_results)
    except Exception as e:
        print(f"  Task 54 FAILED: {e}")
        traceback.print_exc()
        all_results["task54"] = {"error": str(e)}
        save_results(all_results)

    # ------------------------------------------------------------------
    # Task 57: Tools on correct predictions (harm rate)
    # ------------------------------------------------------------------
    try:
        t57 = run_task57(url_2b, correct_samples)
        all_results["task57"] = t57
        save_results(all_results)
    except Exception as e:
        print(f"  Task 57 FAILED: {e}")
        traceback.print_exc()
        all_results["task57"] = {"error": str(e)}
        save_results(all_results)

    # ------------------------------------------------------------------
    # Task 66: Pipeline A -- FT predict, base verify
    # ------------------------------------------------------------------
    try:
        t66 = run_task66(url_2b, mixed_samples)
        all_results["task66"] = t66
        save_results(all_results)
    except Exception as e:
        print(f"  Task 66 FAILED: {e}")
        traceback.print_exc()
        all_results["task66"] = {"error": str(e)}
        save_results(all_results)

    # ------------------------------------------------------------------
    # Task 67: Pipeline B -- base + tools from scratch
    # ------------------------------------------------------------------
    try:
        t67 = run_task67(url_2b, mixed_samples)
        all_results["task67"] = t67
        save_results(all_results)
    except Exception as e:
        print(f"  Task 67 FAILED: {e}")
        traceback.print_exc()
        all_results["task67"] = {"error": str(e)}
        save_results(all_results)

    # ------------------------------------------------------------------
    # Stop 2B server
    # ------------------------------------------------------------------
    print("\nStopping 2B server...")
    server_2b.stop()
    time.sleep(5)

    # ------------------------------------------------------------------
    # Task 60: 2B vs 8B -- need both servers
    # ------------------------------------------------------------------
    print(f"\nStarting 8B server on GPU {GPU_8B}, port {PORT_8B}...")
    server_8b = RobustServer(MODEL_8B, gpu_id=GPU_8B, port=PORT_8B)
    server_8b.start(timeout=600)
    url_8b = server_8b.url

    # Also restart 2B for the comparison
    print(f"Starting 2B server on GPU {GPU_2B}, port {PORT_2B}...")
    server_2b = RobustServer(MODEL_2B, gpu_id=GPU_2B, port=PORT_2B)
    server_2b.start(timeout=600)
    url_2b = server_2b.url

    try:
        t60 = run_task60(url_2b, url_8b, mixed_samples)
        all_results["task60"] = t60
        save_results(all_results)
    except Exception as e:
        print(f"  Task 60 FAILED: {e}")
        traceback.print_exc()
        all_results["task60"] = {"error": str(e)}
        save_results(all_results)

    # Stop both servers
    print("\nStopping servers...")
    server_2b.stop()
    server_8b.stop()
    time.sleep(5)

    # ------------------------------------------------------------------
    # Task 68: Pipeline A vs B head-to-head (no GPU needed)
    # ------------------------------------------------------------------
    try:
        t66_data = all_results.get("task66", {})
        t67_data = all_results.get("task67", {})
        if isinstance(t66_data, dict) and "error" not in t66_data:
            if isinstance(t67_data, dict) and "error" not in t67_data:
                t68 = run_task68(t66_data, t67_data)
                all_results["task68"] = t68
                save_results(all_results)
            else:
                print("  Task 68 SKIPPED: task67 has errors")
        else:
            print("  Task 68 SKIPPED: task66 has errors")
    except Exception as e:
        print(f"  Task 68 FAILED: {e}")
        traceback.print_exc()
        all_results["task68"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Task 69: Error taxonomy (no GPU needed)
    # ------------------------------------------------------------------
    try:
        t69 = run_task69(all_results)
        all_results["task69"] = t69
        save_results(all_results)
    except Exception as e:
        print(f"  Task 69 FAILED: {e}")
        traceback.print_exc()
        all_results["task69"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Task 70: Cost-benefit summary (no GPU needed)
    # ------------------------------------------------------------------
    try:
        t70 = run_task70(all_results)
        all_results["task70"] = t70
        save_results(all_results)
    except Exception as e:
        print(f"  Task 70 FAILED: {e}")
        traceback.print_exc()
        all_results["task70"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Final metadata
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    all_results["metadata"] = {
        "phase": "Phase 3: Multi-Tool Interaction",
        "tasks": "51, 54, 57, 60, 66, 67, 68, 69, 70",
        "total_elapsed_seconds": round(elapsed, 1),
        "total_elapsed_minutes": round(elapsed / 60, 1),
        "model_2b": MODEL_2B,
        "model_8b": MODEL_8B,
        "n_mixed_samples": len(mixed_samples),
        "n_correct_samples": len(correct_samples),
    }
    save_results(all_results)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Results: {RESULTS_PATH}")

    # Print summary table
    print(
        f"\n  {'Task':<10s} {'Description':<45s} {'Status':<10s}"
    )
    print(f"  {'-' * 10} {'-' * 45} {'-' * 10}")
    task_descs = {
        "task51": "All tools, no guidance (2B)",
        "task54": "Tool subset ablations (2B, 4 conditions)",
        "task57": "Tools on correct preds -- harm rate",
        "task60": "2B vs 8B with all tools",
        "task66": "Pipeline A: FT predict, base verify",
        "task67": "Pipeline B: base + tools from scratch",
        "task68": "Pipeline A vs B head-to-head",
        "task69": "Error taxonomy",
        "task70": "Cost-benefit summary",
    }
    for task_key, desc in task_descs.items():
        data = all_results.get(task_key, {})
        if isinstance(data, dict) and "error" in data:
            status = "FAILED"
        elif task_key in all_results:
            status = "DONE"
        else:
            status = "SKIPPED"
        print(f"  {task_key:<10s} {desc:<45s} {status:<10s}")

    print()


if __name__ == "__main__":
    main()
