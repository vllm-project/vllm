#!/usr/bin/env python3
# ruff: noqa: E501,E402
# type: ignore[import-not-found]
"""Phase 3 Multi-Tool Interaction Experiments: Tasks 51-70.

Investigates how multiple tools interact, which combinations help most,
and builds end-to-end pipelines for tool-augmented driving scene analysis.

Priority tasks: 51, 54, 57, 60, 66-68.

Usage:
    python tool_calling_experiment/phase3_multitool.py
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Ensure sibling modules are importable
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from orchestrator import ToolCallingOrchestrator
from tools_v2 import (
    ALL_TOOLS as STAT_TOOLS,
)
from tools_v2 import (
    execute_tool_v2,
)
from visual_tools import (
    FAISS_INDEX_PATH,
    TOOL_ROAD_GEOMETRY,
    TOOL_SIMILAR_SCENES,
    TOOL_WAYPOINT_VIZ,
    TOOL_ZOOM,
    analyze_road_geometry,
    find_similar_scenes,
    load_sample_image,
    visualize_waypoint,
    zoom_region,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(_DIR, "phase3_results.json")
DB_PATH = os.path.join(_DIR, "tool_calling.db")
SC_DB_PATH = os.path.join(
    os.path.dirname(_DIR), "self_consistency_experiment", "self_consistency.db"
)

BASE_2B_PATH = "/fsx/models/Qwen3-VL-2B-Instruct"
BASE_8B_PATH = "/fsx/models/Qwen3-VL-8B-Instruct"
FT_MODEL_PATH = "/workspace/vllm/models/checkpoint"
DATASET_PATH = "/workspace/vllm/models/dataset"

BASE_2B_PORT = 8338
BASE_8B_PORT = 8339
BASE_2B_GPU = 6
BASE_8B_GPU = 7

IMG_W, IMG_H = 504, 336
GRID_SIZE = 63

SYSTEM_PROMPT = (
    "The image is 504x336 pixels. The waypoint grid is 63x63. "
    "When specifying pixel coordinates, keep x in 0-503 and y in 0-335."
)

# Check if FAISS index exists
HAS_FAISS = os.path.exists(FAISS_INDEX_PATH)

# Build available visual tools list (skip retrieval if no FAISS)
AVAILABLE_VISUAL_TOOLS_DEFS = [TOOL_ZOOM, TOOL_WAYPOINT_VIZ, TOOL_ROAD_GEOMETRY]
AVAILABLE_VISUAL_TOOL_EXECUTORS: dict[str, Any] = {
    "zoom_region": zoom_region,
    "visualize_waypoint": visualize_waypoint,
    "analyze_road_geometry": analyze_road_geometry,
}

if HAS_FAISS:
    AVAILABLE_VISUAL_TOOLS_DEFS.append(TOOL_SIMILAR_SCENES)
    AVAILABLE_VISUAL_TOOL_EXECUTORS["find_similar_scenes"] = find_similar_scenes

# All tools = visual + statistical
ALL_TOOL_DEFS = AVAILABLE_VISUAL_TOOLS_DEFS[:]
ALL_TOOL_EXECUTORS = dict(AVAILABLE_VISUAL_TOOL_EXECUTORS)

# We also include statistical tools via a wrapper
def _stat_tool_wrapper(tool_name: str):
    """Create a callable wrapper for statistical tools."""
    def wrapper(**kwargs):
        return execute_tool_v2(tool_name, kwargs, level=3)
    return wrapper

for tdef in STAT_TOOLS:
    name = tdef["function"]["name"]
    ALL_TOOL_DEFS.append(tdef)
    ALL_TOOL_EXECUTORS[name] = _stat_tool_wrapper(name)


# ---------------------------------------------------------------------------
# Server Management
# ---------------------------------------------------------------------------
def start_vllm_server(model_path: str, port: int, gpu_id: int) -> subprocess.Popen:
    """Start a vLLM server process with tool-calling support."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Remove /workspace/vllm from PYTHONPATH so installed vllm is used
    pp = env.get("PYTHONPATH", "")
    parts = [p for p in pp.split(":") if p and "/workspace/vllm" not in p]
    env["PYTHONPATH"] = ":".join(parts)

    vllm_bin = "/home/mketkar/.local/bin/vllm"
    cmd = [
        vllm_bin, "serve", model_path,
        "--trust-remote-code",
        "--max-model-len", "8192",
        "--enforce-eager",
        "--port", str(port),
        "--gpu-memory-utilization", "0.8",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]

    log_path = f"/tmp/vllm_phase3_{port}.log"
    print(f"  Starting server: model={os.path.basename(model_path)}, GPU={gpu_id}, port={port}")
    print(f"  Log: {log_path}")
    log_fh = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd, env=env, cwd="/tmp",
        stdout=log_fh, stderr=subprocess.STDOUT,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


def wait_for_server(port: int, timeout: int = 600) -> bool:
    """Wait for vLLM server to become healthy."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                elapsed = int(time.time() - start)
                print(f"  Server on port {port} ready in {elapsed}s")
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_servers():
    """Kill phase3 vllm servers."""
    subprocess.run(["pkill", "-f", f"vllm serve.*{BASE_2B_PORT}"], capture_output=True)
    subprocess.run(["pkill", "-f", f"vllm serve.*{BASE_8B_PORT}"], capture_output=True)
    time.sleep(2)


# ---------------------------------------------------------------------------
# Sample Selection
# ---------------------------------------------------------------------------
def load_baseline_predictions() -> dict[int, dict]:
    """Load baseline (fine-tuned) predictions from tool_calling.db."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("""
        SELECT sample_id, original_scene, original_long_action, original_lat_action,
               scene_type_gt, long_action_gt, lat_action_gt, original_scene_correct,
               fine_class, odd_label
        FROM predictions
        WHERE experiment_id='baseline'
    """)
    preds = {}
    for row in cur.fetchall():
        preds[row[0]] = {
            "sample_id": row[0],
            "ft_scene": row[1],
            "ft_long_action": row[2],
            "ft_lat_action": row[3],
            "scene_type_gt": row[4],
            "long_action_gt": row[5],
            "lat_action_gt": row[6],
            "ft_scene_correct": bool(row[7]),
            "fine_class": row[8],
            "odd_label": row[9],
        }
    conn.close()
    return preds


def select_mixed_samples(preds: dict[int, dict], n_wrong: int = 50, n_correct: int = 50, seed: int = 42) -> list[int]:
    """Select a mix of samples where FT model is wrong/correct."""
    rng = random.Random(seed)
    wrong_ids = [sid for sid, p in preds.items() if not p["ft_scene_correct"]]
    correct_ids = [sid for sid, p in preds.items() if p["ft_scene_correct"]]
    rng.shuffle(wrong_ids)
    rng.shuffle(correct_ids)
    selected = wrong_ids[:n_wrong] + correct_ids[:n_correct]
    rng.shuffle(selected)
    return selected


def select_correct_samples(preds: dict[int, dict], n: int = 100, seed: int = 42) -> list[int]:
    """Select samples where FT model is correct."""
    rng = random.Random(seed)
    correct_ids = [sid for sid, p in preds.items() if p["ft_scene_correct"]]
    rng.shuffle(correct_ids)
    return correct_ids[:n]


# ---------------------------------------------------------------------------
# Orchestrator Helpers
# ---------------------------------------------------------------------------
def make_orchestrator(
    port: int,
    tool_defs: list[dict],
    tool_executors: dict[str, Any],
    max_rounds: int = 5,
    max_tokens: int = 1024,
) -> ToolCallingOrchestrator:
    """Create an orchestrator for a specific server/tool configuration."""
    return ToolCallingOrchestrator(
        server_url=f"http://localhost:{port}",
        tools=tool_executors,
        tool_definitions=tool_defs,
        max_tool_rounds=max_rounds,
        temperature=0,
        max_tokens=max_tokens,
    )


def run_single_sample(
    orch: ToolCallingOrchestrator,
    sample_id: int,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    use_image: bool = True,
) -> dict[str, Any]:
    """Run the orchestrator on a single sample. Returns full result dict."""
    try:
        img_path = load_sample_image(sample_id) if use_image else None
    except Exception as e:
        return {"error": f"Failed to load image for sample {sample_id}: {e}", "sample_id": sample_id}

    try:
        result = orch.run(
            image_path=img_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tool_choice="auto",
        )
        result["sample_id"] = sample_id
        # Strip full_conversation to save space (can be very large due to base64 images)
        if "full_conversation" in result:
            result["conversation_length"] = len(result["full_conversation"])
            del result["full_conversation"]
        return result
    except Exception as e:
        return {"error": str(e), "sample_id": sample_id, "traceback": traceback.format_exc()}


def compute_accuracy(results: list[dict], preds: dict[int, dict]) -> dict[str, Any]:
    """Compute scene/action accuracy for a list of results."""
    n_total = 0
    scene_correct = 0
    long_correct = 0
    lat_correct = 0
    all_correct = 0
    parse_failures = 0

    for r in results:
        if "error" in r and r.get("final_prediction") is None:
            parse_failures += 1
            continue
        sid = r.get("sample_id")
        if sid is None or sid not in preds:
            parse_failures += 1
            continue

        gt = preds[sid]
        fp = r.get("final_prediction", {})
        if fp is None:
            parse_failures += 1
            continue

        n_total += 1
        s_ok = fp.get("scene") == gt["scene_type_gt"]
        l_ok = fp.get("long_action") == gt["long_action_gt"]
        la_ok = fp.get("lat_action") == gt["lat_action_gt"]
        if s_ok:
            scene_correct += 1
        if l_ok:
            long_correct += 1
        if la_ok:
            lat_correct += 1
        if s_ok and l_ok and la_ok:
            all_correct += 1

    return {
        "n_total": n_total,
        "parse_failures": parse_failures,
        "scene_accuracy": round(scene_correct / n_total, 4) if n_total else 0,
        "long_action_accuracy": round(long_correct / n_total, 4) if n_total else 0,
        "lat_action_accuracy": round(lat_correct / n_total, 4) if n_total else 0,
        "all_correct_accuracy": round(all_correct / n_total, 4) if n_total else 0,
        "scene_correct": scene_correct,
        "long_correct": long_correct,
        "lat_correct": lat_correct,
        "all_correct": all_correct,
    }


def compute_tool_usage_stats(results: list[dict]) -> dict[str, Any]:
    """Compute tool usage statistics from results."""
    total_calls = 0
    tool_counts: dict[str, int] = {}
    tool_sequences: list[list[str]] = []
    rounds_list = []
    calls_list = []

    for r in results:
        tc_list = r.get("tool_calls", [])
        n_calls = len(tc_list)
        total_calls += n_calls
        calls_list.append(n_calls)
        rounds_list.append(r.get("num_rounds", 0))

        seq = []
        for tc in tc_list:
            name = tc.get("tool_name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1
            seq.append(name)
        tool_sequences.append(seq)

    # Common sequences
    seq_counter = collections.Counter(tuple(s) for s in tool_sequences)
    top_sequences = seq_counter.most_common(10)

    return {
        "total_calls": total_calls,
        "avg_calls_per_sample": round(total_calls / len(results), 2) if results else 0,
        "avg_rounds": round(sum(rounds_list) / len(rounds_list), 2) if rounds_list else 0,
        "tool_counts": dict(sorted(tool_counts.items(), key=lambda x: -x[1])),
        "top_sequences": [{"seq": list(s), "count": c} for s, c in top_sequences],
        "samples_with_no_tools": sum(1 for c in calls_list if c == 0),
        "max_calls": max(calls_list) if calls_list else 0,
    }


def print_accuracy_table(label: str, acc: dict):
    """Print a formatted accuracy summary."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {acc['n_total']} (parse failures: {acc['parse_failures']})")
    print(f"  Scene accuracy:      {acc['scene_accuracy']:.1%} ({acc['scene_correct']}/{acc['n_total']})")
    print(f"  Long action acc:     {acc['long_action_accuracy']:.1%} ({acc['long_correct']}/{acc['n_total']})")
    print(f"  Lat action acc:      {acc['lat_action_accuracy']:.1%} ({acc['lat_correct']}/{acc['n_total']})")
    print(f"  All correct:         {acc['all_correct_accuracy']:.1%} ({acc['all_correct']}/{acc['n_total']})")
    print(f"{'='*60}")


def print_tool_stats(label: str, stats: dict):
    """Print formatted tool usage statistics."""
    print(f"\n  Tool Usage Stats ({label}):")
    print(f"    Total calls: {stats['total_calls']}, avg/sample: {stats['avg_calls_per_sample']}")
    print(f"    Avg rounds: {stats['avg_rounds']}, max calls: {stats['max_calls']}")
    print(f"    Samples with no tools: {stats['samples_with_no_tools']}")
    print(f"    Tool counts: {stats['tool_counts']}")
    if stats['top_sequences']:
        print("    Top sequences:")
        for s in stats['top_sequences'][:5]:
            print(f"      {s['seq']} (count={s['count']})")


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
PROMPT_BASIC = (
    "Analyze this driving scene and predict scene type, action, and waypoint.\n\n"
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger\n"
    "Longitudinal actions: stop, slowdown, proceed, null\n"
    "Lateral actions: lc_left, lc_right, null\n\n"
    "Output format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>"
)

PROMPT_REASONING = (
    "Analyze this driving scene and predict scene type, action, and waypoint.\n"
    "Think step by step. Use tools to verify your reasoning at each step.\n\n"
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger\n"
    "Longitudinal actions: stop, slowdown, proceed, null\n"
    "Lateral actions: lc_left, lc_right, null\n\n"
    "Output format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>"
)

PROMPT_PRESCRIBED = (
    "Analyze this driving scene step by step:\n"
    "1. First analyze road geometry using the analyze_road_geometry tool.\n"
    "2. Then classify the scene type.\n"
    "3. Then predict the action and waypoint.\n"
    "4. Visualize the waypoint to verify it looks correct.\n\n"
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger\n"
    "Longitudinal actions: stop, slowdown, proceed, null\n"
    "Lateral actions: lc_left, lc_right, null\n\n"
    "Output format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>"
)

PROMPT_COT_WITH_TOOLS = (
    "Analyze this driving scene. Write your reasoning step by step. "
    "Use tools to verify each step of your reasoning.\n\n"
    "For each step:\n"
    "1. State your observation\n"
    "2. Use a relevant tool to verify\n"
    "3. Adjust if the tool result contradicts your observation\n\n"
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger\n"
    "Longitudinal actions: stop, slowdown, proceed, null\n"
    "Lateral actions: lc_left, lc_right, null\n\n"
    "Output format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>"
)

PROMPT_CONFIDENCE = (
    "Analyze this driving scene and predict scene type, action, and waypoint.\n\n"
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger\n"
    "Longitudinal actions: stop, slowdown, proceed, null\n"
    "Lateral actions: lc_left, lc_right, null\n\n"
    "Output format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>\n"
    "CONFIDENCE: <1-10>"
)

PROMPT_NO_IMAGE = (
    "You are analyzing a driving scene but do NOT have an image. "
    "Use the available tools to reason about scenes statistically.\n"
    "The scene has been described as: {description}\n\n"
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger\n"
    "Longitudinal actions: stop, slowdown, proceed, null\n"
    "Lateral actions: lc_left, lc_right, null\n\n"
    "Output format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>"
)

PROMPT_VERIFY_FT = (
    "A fine-tuned model predicted the following for this driving scene:\n"
    "  Scene: {ft_scene}\n"
    "  Longitudinal action: {ft_long}\n"
    "  Lateral action: {ft_lat}\n\n"
    "Your job is to VERIFY this prediction. Use tools to check if it is correct.\n"
    "If you find evidence the prediction is wrong, provide the corrected prediction.\n"
    "If the prediction looks correct, confirm it.\n\n"
    "Scene types: nominal, flooded, incident_zone, mounted_police, flagger\n"
    "Longitudinal actions: stop, slowdown, proceed, null\n"
    "Lateral actions: lc_left, lc_right, null\n\n"
    "Output format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "FINAL_LONG_ACTION: <action>\n"
    "FINAL_LAT_ACTION: <action>"
)


# ---------------------------------------------------------------------------
# Tool subset configurations for Task 54
# ---------------------------------------------------------------------------
def get_tool_subset(condition: str) -> tuple[list[dict], dict[str, Any]]:
    """Return (tool_definitions, tool_executors) for a given condition."""
    if condition == "zoom_only":
        return [TOOL_ZOOM], {"zoom_region": zoom_region}
    elif condition == "waypoint_only":
        return [TOOL_WAYPOINT_VIZ], {"visualize_waypoint": visualize_waypoint}
    elif condition == "geometry_only":
        return [TOOL_ROAD_GEOMETRY], {"analyze_road_geometry": analyze_road_geometry}
    elif condition == "retrieval_only":
        if HAS_FAISS:
            return [TOOL_SIMILAR_SCENES], {"find_similar_scenes": find_similar_scenes}
        else:
            return [], {}
    elif condition == "zoom_waypoint":
        return (
            [TOOL_ZOOM, TOOL_WAYPOINT_VIZ],
            {"zoom_region": zoom_region, "visualize_waypoint": visualize_waypoint},
        )
    elif condition == "zoom_geometry":
        return (
            [TOOL_ZOOM, TOOL_ROAD_GEOMETRY],
            {"zoom_region": zoom_region, "analyze_road_geometry": analyze_road_geometry},
        )
    elif condition == "all_tools":
        return ALL_TOOL_DEFS[:], dict(ALL_TOOL_EXECUTORS)
    elif condition == "no_tools":
        return [], {}
    else:
        return ALL_TOOL_DEFS[:], dict(ALL_TOOL_EXECUTORS)


# ---------------------------------------------------------------------------
# Task Implementations
# ---------------------------------------------------------------------------

def run_task_51(preds: dict, sample_ids: list[int]) -> dict:
    """Task 51: All tools available, no guidance (2B, 100 samples)."""
    print("\n" + "=" * 70)
    print("TASK 51: All tools available, no guidance (2B)")
    print("=" * 70)

    orch = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS)
    results = []
    for i, sid in enumerate(sample_ids):
        print(f"  [{i+1}/{len(sample_ids)}] Sample {sid}...", end=" ", flush=True)
        r = run_single_sample(orch, sid, PROMPT_BASIC)
        results.append(r)
        n_tc = r.get("num_tool_calls", 0)
        fp = r.get("final_prediction", {})
        scene = fp.get("scene", "?") if fp else "?"
        print(f"tools={n_tc}, scene={scene}")

    acc = compute_accuracy(results, preds)
    tool_stats = compute_tool_usage_stats(results)
    print_accuracy_table("Task 51: All tools, no guidance", acc)
    print_tool_stats("Task 51", tool_stats)

    return {"task": 51, "accuracy": acc, "tool_stats": tool_stats, "results": results}


def run_task_52(preds: dict, sample_ids: list[int]) -> dict:
    """Task 52: All tools + reasoning instruction (2B)."""
    print("\n" + "=" * 70)
    print("TASK 52: All tools + reasoning instruction (2B)")
    print("=" * 70)

    orch = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS)
    results = []
    for i, sid in enumerate(sample_ids):
        print(f"  [{i+1}/{len(sample_ids)}] Sample {sid}...", end=" ", flush=True)
        r = run_single_sample(orch, sid, PROMPT_REASONING)
        results.append(r)
        n_tc = r.get("num_tool_calls", 0)
        print(f"tools={n_tc}")

    acc = compute_accuracy(results, preds)
    tool_stats = compute_tool_usage_stats(results)
    print_accuracy_table("Task 52: All tools + reasoning", acc)
    print_tool_stats("Task 52", tool_stats)

    return {"task": 52, "accuracy": acc, "tool_stats": tool_stats, "results": results}


def run_task_53(preds: dict, sample_ids: list[int]) -> dict:
    """Task 53: Prescribed tool order (2B)."""
    print("\n" + "=" * 70)
    print("TASK 53: Prescribed tool order (2B)")
    print("=" * 70)

    orch = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS, max_rounds=6)
    results = []
    for i, sid in enumerate(sample_ids):
        print(f"  [{i+1}/{len(sample_ids)}] Sample {sid}...", end=" ", flush=True)
        r = run_single_sample(orch, sid, PROMPT_PRESCRIBED)
        results.append(r)
        n_tc = r.get("num_tool_calls", 0)
        print(f"tools={n_tc}")

    acc = compute_accuracy(results, preds)
    tool_stats = compute_tool_usage_stats(results)
    print_accuracy_table("Task 53: Prescribed order", acc)
    print_tool_stats("Task 53", tool_stats)

    return {"task": 53, "accuracy": acc, "tool_stats": tool_stats, "results": results}


def run_task_54(preds: dict, sample_ids: list[int]) -> dict:
    """Task 54: Tool subset ablations (2B, 100 samples x 6 conditions)."""
    print("\n" + "=" * 70)
    print("TASK 54: Tool subset ablations (2B)")
    print("=" * 70)

    conditions = ["no_tools", "zoom_only", "waypoint_only", "geometry_only", "zoom_geometry", "all_tools"]
    if HAS_FAISS:
        conditions.insert(4, "retrieval_only")

    condition_results = {}
    for cond in conditions:
        print(f"\n  --- Condition: {cond} ---")
        t_defs, t_execs = get_tool_subset(cond)
        if cond == "no_tools":
            orch = make_orchestrator(BASE_2B_PORT, [], {})
        else:
            orch = make_orchestrator(BASE_2B_PORT, t_defs, t_execs)

        results = []
        for i, sid in enumerate(sample_ids):
            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(sample_ids)}]", flush=True)
            r = run_single_sample(orch, sid, PROMPT_BASIC)
            results.append(r)

        acc = compute_accuracy(results, preds)
        tool_stats = compute_tool_usage_stats(results)
        print_accuracy_table(f"Task 54 ({cond})", acc)

        condition_results[cond] = {
            "accuracy": acc,
            "tool_stats": tool_stats,
            "results": results,
        }

    # Summary comparison
    print("\n  Task 54 Summary Comparison:")
    print(f"  {'Condition':<20} {'Scene Acc':>10} {'All Correct':>12} {'Avg Tools':>10}")
    print(f"  {'-'*52}")
    for cond in conditions:
        cr = condition_results[cond]
        print(f"  {cond:<20} {cr['accuracy']['scene_accuracy']:>10.1%} "
              f"{cr['accuracy']['all_correct_accuracy']:>12.1%} "
              f"{cr['tool_stats']['avg_calls_per_sample']:>10.1f}")

    return {"task": 54, "conditions": {c: {"accuracy": v["accuracy"], "tool_stats": v["tool_stats"]} for c, v in condition_results.items()}, "condition_results": condition_results}


def run_task_55(task54_data: dict) -> dict:
    """Task 55: Interaction effects from Task 54 ablation data."""
    print("\n" + "=" * 70)
    print("TASK 55: Interaction effects")
    print("=" * 70)

    conditions = task54_data.get("conditions", {})
    no_tools_acc = conditions.get("no_tools", {}).get("accuracy", {}).get("scene_accuracy", 0)
    zoom_acc = conditions.get("zoom_only", {}).get("accuracy", {}).get("scene_accuracy", 0)
    geom_acc = conditions.get("geometry_only", {}).get("accuracy", {}).get("scene_accuracy", 0)
    zoom_geom_acc = conditions.get("zoom_geometry", {}).get("accuracy", {}).get("scene_accuracy", 0)
    all_acc = conditions.get("all_tools", {}).get("accuracy", {}).get("scene_accuracy", 0)

    zoom_gain = zoom_acc - no_tools_acc
    geom_gain = geom_acc - no_tools_acc
    individual_sum = zoom_gain + geom_gain
    combined_gain = zoom_geom_acc - no_tools_acc

    interaction = combined_gain - individual_sum
    is_superadditive = interaction > 0.01
    is_subadditive = interaction < -0.01

    analysis = {
        "no_tools_baseline": no_tools_acc,
        "zoom_gain_over_baseline": round(zoom_gain, 4),
        "geometry_gain_over_baseline": round(geom_gain, 4),
        "sum_of_individual_gains": round(individual_sum, 4),
        "combined_gain_over_baseline": round(combined_gain, 4),
        "interaction_effect": round(interaction, 4),
        "all_tools_gain": round(all_acc - no_tools_acc, 4),
        "is_superadditive": is_superadditive,
        "is_subadditive": is_subadditive,
        "interpretation": (
            "Superadditive: zoom+geometry together exceeds the sum of individual gains"
            if is_superadditive else
            "Subadditive: zoom+geometry together is less than the sum of individual gains"
            if is_subadditive else
            "Approximately additive: combined gain is roughly sum of individual gains"
        ),
    }

    print(f"  No-tools baseline: {no_tools_acc:.1%}")
    print(f"  Zoom gain: {zoom_gain:+.1%}")
    print(f"  Geometry gain: {geom_gain:+.1%}")
    print(f"  Sum of individual gains: {individual_sum:+.1%}")
    print(f"  Combined (zoom+geometry) gain: {combined_gain:+.1%}")
    print(f"  Interaction effect: {interaction:+.1%}")
    print(f"  All-tools gain: {all_acc - no_tools_acc:+.1%}")
    print(f"  Interpretation: {analysis['interpretation']}")

    return {"task": 55, "analysis": analysis}


def run_task_56(task51_data: dict, task52_data: dict, task53_data: dict) -> dict:
    """Task 56: Tool sequence analysis from Tasks 51-53."""
    print("\n" + "=" * 70)
    print("TASK 56: Tool sequence analysis")
    print("=" * 70)

    analysis = {}
    for label, data in [("task51_basic", task51_data), ("task52_reasoning", task52_data), ("task53_prescribed", task53_data)]:
        results = data.get("results", [])
        sequences = []
        for r in results:
            seq = [tc["tool_name"] for tc in r.get("tool_calls", [])]
            sequences.append(seq)

        seq_counter = collections.Counter(tuple(s) for s in sequences)
        top = seq_counter.most_common(10)

        # First tool called
        first_tool_counter = collections.Counter(s[0] for s in sequences if s)
        # Tool ordering patterns
        analysis[label] = {
            "total_samples": len(results),
            "top_sequences": [{"seq": list(s), "count": c} for s, c in top],
            "first_tool_used": dict(first_tool_counter.most_common()),
            "avg_sequence_length": round(sum(len(s) for s in sequences) / max(len(sequences), 1), 2),
        }

        print(f"\n  {label}:")
        print(f"    Avg sequence length: {analysis[label]['avg_sequence_length']}")
        print(f"    First tool distribution: {dict(first_tool_counter.most_common(5))}")
        if top:
            print("    Top 3 sequences:")
            for s, c in top[:3]:
                print(f"      {list(s)} ({c} times)")

    return {"task": 56, "analysis": analysis}


def run_task_57(preds: dict, correct_ids: list[int]) -> dict:
    """Task 57: Tools on correct predictions -- measure harm rate."""
    print("\n" + "=" * 70)
    print("TASK 57: Tools on correct predictions (harm rate)")
    print("=" * 70)

    orch = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS)
    results = []
    for i, sid in enumerate(correct_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(correct_ids)}] Sample {sid}...", flush=True)
        r = run_single_sample(orch, sid, PROMPT_BASIC)
        results.append(r)

    # For these samples, FT model was correct.
    # Check how many the tool-augmented base model gets wrong.
    n_total = 0
    n_still_correct = 0
    n_broken = 0
    n_parse_fail = 0

    for r in results:
        sid = r.get("sample_id")
        if sid is None or sid not in preds:
            n_parse_fail += 1
            continue
        fp = r.get("final_prediction")
        if not fp:
            n_parse_fail += 1
            continue
        gt = preds[sid]
        n_total += 1
        if fp.get("scene") == gt["scene_type_gt"]:
            n_still_correct += 1
        else:
            n_broken += 1

    harm_rate = n_broken / n_total if n_total else 0

    acc = compute_accuracy(results, preds)
    print_accuracy_table("Task 57: Tools on correct predictions", acc)
    print("  Harm analysis:")
    print(f"    FT-correct samples tested: {n_total}")
    print(f"    Still correct with tools: {n_still_correct} ({n_still_correct/max(n_total,1):.1%})")
    print(f"    Broken by tools: {n_broken} ({harm_rate:.1%})")

    return {
        "task": 57,
        "accuracy": acc,
        "harm_analysis": {
            "n_tested": n_total,
            "n_still_correct": n_still_correct,
            "n_broken": n_broken,
            "harm_rate": round(harm_rate, 4),
        },
        "results": results,
    }


def run_task_60(preds: dict, sample_ids: list[int]) -> dict:
    """Task 60: 2B vs 8B with all tools."""
    print("\n" + "=" * 70)
    print("TASK 60: 2B vs 8B with all tools")
    print("=" * 70)

    results_by_model = {}
    for label, port in [("2B", BASE_2B_PORT), ("8B", BASE_8B_PORT)]:
        print(f"\n  --- Model: {label} ---")
        orch = make_orchestrator(port, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS)
        results = []
        for i, sid in enumerate(sample_ids):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    [{i+1}/{len(sample_ids)}]", flush=True)
            r = run_single_sample(orch, sid, PROMPT_BASIC)
            results.append(r)

        acc = compute_accuracy(results, preds)
        tool_stats = compute_tool_usage_stats(results)
        print_accuracy_table(f"Task 60 ({label})", acc)
        print_tool_stats(label, tool_stats)
        results_by_model[label] = {"accuracy": acc, "tool_stats": tool_stats, "results": results}

    # Head-to-head comparison
    print("\n  Head-to-head comparison:")
    for metric in ["scene_accuracy", "long_action_accuracy", "lat_action_accuracy", "all_correct_accuracy"]:
        v2b = results_by_model["2B"]["accuracy"][metric]
        v8b = results_by_model["8B"]["accuracy"][metric]
        diff = v8b - v2b
        print(f"    {metric}: 2B={v2b:.1%}, 8B={v8b:.1%}, diff={diff:+.1%}")

    return {
        "task": 60,
        "comparison": {
            model: {"accuracy": d["accuracy"], "tool_stats": d["tool_stats"]}
            for model, d in results_by_model.items()
        },
        "results_by_model": results_by_model,
    }


def run_task_61(preds: dict, sample_ids: list[int]) -> dict:
    """Task 61: Tool depth analysis (accuracy vs number of tool calls)."""
    print("\n" + "=" * 70)
    print("TASK 61: Tool depth analysis (max 8 rounds)")
    print("=" * 70)

    orch = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS, max_rounds=8)
    results = []
    for i, sid in enumerate(sample_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(sample_ids)}]", flush=True)
        r = run_single_sample(orch, sid, PROMPT_BASIC)
        results.append(r)

    # Bin by number of tool calls
    bins: dict[int, list[dict]] = {}
    for r in results:
        n = r.get("num_tool_calls", 0)
        bins.setdefault(n, []).append(r)

    depth_analysis = {}
    print("\n  Accuracy by tool call depth:")
    print(f"  {'Calls':>6} {'Count':>6} {'Scene Acc':>10}")
    for n_calls in sorted(bins.keys()):
        bin_results = bins[n_calls]
        acc = compute_accuracy(bin_results, preds)
        depth_analysis[n_calls] = {"count": len(bin_results), "accuracy": acc}
        print(f"  {n_calls:>6} {len(bin_results):>6} {acc['scene_accuracy']:>10.1%}")

    return {"task": 61, "depth_analysis": depth_analysis, "results": results}


def run_task_65(preds: dict, sample_ids: list[int]) -> dict:
    """Task 65: Tool-augmented CoT vs CoT alone vs tools alone."""
    print("\n" + "=" * 70)
    print("TASK 65: Tool-augmented CoT comparison")
    print("=" * 70)

    conditions = {
        "tools_only": (ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS, PROMPT_BASIC),
        "cot_only": ([], {}, PROMPT_REASONING),
        "tools_plus_cot": (ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS, PROMPT_COT_WITH_TOOLS),
    }

    condition_results = {}
    for cond_name, (t_defs, t_execs, prompt) in conditions.items():
        print(f"\n  --- Condition: {cond_name} ---")
        orch = make_orchestrator(BASE_2B_PORT, t_defs, t_execs)
        results = []
        for i, sid in enumerate(sample_ids):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    [{i+1}/{len(sample_ids)}]", flush=True)
            r = run_single_sample(orch, sid, prompt)
            results.append(r)

        acc = compute_accuracy(results, preds)
        print_accuracy_table(f"Task 65 ({cond_name})", acc)
        condition_results[cond_name] = {"accuracy": acc, "results": results}

    print("\n  Task 65 Summary:")
    for cond_name, cr in condition_results.items():
        print(f"    {cond_name}: scene={cr['accuracy']['scene_accuracy']:.1%}, "
              f"all={cr['accuracy']['all_correct_accuracy']:.1%}")

    return {"task": 65, "conditions": {c: v["accuracy"] for c, v in condition_results.items()}, "condition_results": condition_results}


def run_task_66(preds: dict, sample_ids: list[int]) -> dict:
    """Task 66: Pipeline A -- FT predict, base verify."""
    print("\n" + "=" * 70)
    print("TASK 66: Pipeline A -- FT predict, base 2B verify")
    print("=" * 70)

    orch = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS, max_rounds=5)
    results = []
    for i, sid in enumerate(sample_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(sample_ids)}]", flush=True)

        gt = preds[sid]
        prompt = PROMPT_VERIFY_FT.format(
            ft_scene=gt["ft_scene"],
            ft_long=gt["ft_long_action"],
            ft_lat=gt["ft_lat_action"],
        )
        r = run_single_sample(orch, sid, prompt)
        results.append(r)

    acc = compute_accuracy(results, preds)
    tool_stats = compute_tool_usage_stats(results)
    print_accuracy_table("Task 66: Pipeline A (FT predict, base verify)", acc)
    print_tool_stats("Task 66", tool_stats)

    # Compare to FT baseline accuracy on same samples
    ft_correct = sum(1 for sid in sample_ids if preds[sid]["ft_scene_correct"])
    ft_acc = ft_correct / len(sample_ids) if sample_ids else 0
    improvement = acc["scene_accuracy"] - ft_acc
    print(f"\n  FT baseline scene accuracy on these samples: {ft_acc:.1%}")
    print(f"  Pipeline A scene accuracy: {acc['scene_accuracy']:.1%}")
    print(f"  Improvement: {improvement:+.1%}")

    # Detailed: saves vs breaks
    n_saves = 0
    n_breaks = 0
    n_both_correct = 0
    n_both_wrong = 0
    for r in results:
        sid = r.get("sample_id")
        if sid is None or sid not in preds:
            continue
        gt = preds[sid]
        fp = r.get("final_prediction", {})
        if not fp:
            continue
        ft_ok = gt["ft_scene_correct"]
        pipeline_ok = fp.get("scene") == gt["scene_type_gt"]
        if ft_ok and pipeline_ok:
            n_both_correct += 1
        elif ft_ok and not pipeline_ok:
            n_breaks += 1
        elif not ft_ok and pipeline_ok:
            n_saves += 1
        else:
            n_both_wrong += 1

    print(f"  Saves (FT wrong -> pipeline correct): {n_saves}")
    print(f"  Breaks (FT correct -> pipeline wrong): {n_breaks}")
    print(f"  Both correct: {n_both_correct}")
    print(f"  Both wrong: {n_both_wrong}")

    return {
        "task": 66,
        "accuracy": acc,
        "tool_stats": tool_stats,
        "ft_baseline_accuracy": ft_acc,
        "improvement": round(improvement, 4),
        "saves": n_saves,
        "breaks": n_breaks,
        "both_correct": n_both_correct,
        "both_wrong": n_both_wrong,
        "results": results,
    }


def run_task_67(preds: dict, sample_ids: list[int]) -> dict:
    """Task 67: Pipeline B -- base + tools from scratch."""
    print("\n" + "=" * 70)
    print("TASK 67: Pipeline B -- base 2B + all tools from scratch")
    print("=" * 70)

    orch = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS, max_rounds=6)
    results = []
    for i, sid in enumerate(sample_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(sample_ids)}]", flush=True)
        r = run_single_sample(orch, sid, PROMPT_BASIC)
        results.append(r)

    acc = compute_accuracy(results, preds)
    tool_stats = compute_tool_usage_stats(results)
    print_accuracy_table("Task 67: Pipeline B (base + tools from scratch)", acc)
    print_tool_stats("Task 67", tool_stats)

    ft_correct = sum(1 for sid in sample_ids if preds[sid]["ft_scene_correct"])
    ft_acc = ft_correct / len(sample_ids) if sample_ids else 0
    print(f"\n  FT baseline scene accuracy on these samples: {ft_acc:.1%}")
    print(f"  Pipeline B scene accuracy: {acc['scene_accuracy']:.1%}")
    print(f"  Can tools+base match FT? {'YES' if acc['scene_accuracy'] >= ft_acc else 'NO'} (diff={acc['scene_accuracy'] - ft_acc:+.1%})")

    return {
        "task": 67,
        "accuracy": acc,
        "tool_stats": tool_stats,
        "ft_baseline_accuracy": ft_acc,
        "results": results,
    }


def run_task_68(preds: dict, sample_ids: list[int]) -> dict:
    """Task 68: Pipeline A vs B head-to-head."""
    print("\n" + "=" * 70)
    print("TASK 68: Pipeline A vs B head-to-head")
    print("=" * 70)

    # Pipeline A: FT predict, base verify
    orch_a = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS, max_rounds=5)
    # Pipeline B: base from scratch
    orch_b = make_orchestrator(BASE_2B_PORT, ALL_TOOL_DEFS, ALL_TOOL_EXECUTORS, max_rounds=6)

    results_a = []
    results_b = []

    for i, sid in enumerate(sample_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(sample_ids)}]", flush=True)

        gt = preds[sid]

        # Pipeline A
        prompt_a = PROMPT_VERIFY_FT.format(
            ft_scene=gt["ft_scene"],
            ft_long=gt["ft_long_action"],
            ft_lat=gt["ft_lat_action"],
        )
        ra = run_single_sample(orch_a, sid, prompt_a)
        results_a.append(ra)

        # Pipeline B
        rb = run_single_sample(orch_b, sid, PROMPT_BASIC)
        results_b.append(rb)

    acc_a = compute_accuracy(results_a, preds)
    acc_b = compute_accuracy(results_b, preds)
    ts_a = compute_tool_usage_stats(results_a)
    ts_b = compute_tool_usage_stats(results_b)

    print_accuracy_table("Task 68: Pipeline A (FT+verify)", acc_a)
    print_accuracy_table("Task 68: Pipeline B (base+tools)", acc_b)

    ft_correct = sum(1 for sid in sample_ids if preds[sid]["ft_scene_correct"])
    ft_acc = ft_correct / len(sample_ids) if sample_ids else 0

    print(f"\n  Head-to-head comparison (n={len(sample_ids)}):")
    print(f"  {'Pipeline':<20} {'Scene Acc':>10} {'All Correct':>12} {'Avg Tools':>10}")
    print(f"  {'-'*52}")
    print(f"  {'FT baseline':<20} {ft_acc:>10.1%}")
    print(f"  {'A (FT+verify)':<20} {acc_a['scene_accuracy']:>10.1%} "
          f"{acc_a['all_correct_accuracy']:>12.1%} {ts_a['avg_calls_per_sample']:>10.1f}")
    print(f"  {'B (base+tools)':<20} {acc_b['scene_accuracy']:>10.1%} "
          f"{acc_b['all_correct_accuracy']:>12.1%} {ts_b['avg_calls_per_sample']:>10.1f}")

    # Per-sample comparison
    a_wins = 0
    b_wins = 0
    ties = 0
    for ra, rb in zip(results_a, results_b):
        sid = ra.get("sample_id")
        if sid is None or sid not in preds:
            continue
        gt = preds[sid]
        fp_a = ra.get("final_prediction", {})
        fp_b = rb.get("final_prediction", {})
        if not fp_a or not fp_b:
            continue
        a_ok = fp_a.get("scene") == gt["scene_type_gt"]
        b_ok = fp_b.get("scene") == gt["scene_type_gt"]
        if a_ok and not b_ok:
            a_wins += 1
        elif b_ok and not a_ok:
            b_wins += 1
        else:
            ties += 1

    print(f"\n  Per-sample: A wins={a_wins}, B wins={b_wins}, ties={ties}")

    return {
        "task": 68,
        "pipeline_a": {"accuracy": acc_a, "tool_stats": ts_a},
        "pipeline_b": {"accuracy": acc_b, "tool_stats": ts_b},
        "ft_baseline_accuracy": ft_acc,
        "per_sample": {"a_wins": a_wins, "b_wins": b_wins, "ties": ties},
        "results_a": results_a,
        "results_b": results_b,
    }


def run_task_69(all_results: dict) -> dict:
    """Task 69: Error taxonomy from all Phase 3 data."""
    print("\n" + "=" * 70)
    print("TASK 69: Error taxonomy")
    print("=" * 70)

    # Collect all results that have tool calls
    all_items: list[dict] = []
    for task_key, task_data in all_results.items():
        if isinstance(task_data, dict):
            results = task_data.get("results", [])
            if not results and "condition_results" in task_data:
                for cond_name, cond_data in task_data.get("condition_results", {}).items():
                    results.extend(cond_data.get("results", []))
            if not results and "results_a" in task_data:
                results.extend(task_data.get("results_a", []))
                results.extend(task_data.get("results_b", []))
            if not results and "results_by_model" in task_data:
                for model_data in task_data.get("results_by_model", {}).values():
                    results.extend(model_data.get("results", []))
            for r in results:
                if r.get("sample_id") is not None:
                    all_items.append(r)

    # Load predictions for GT
    preds = load_baseline_predictions()

    # Categorize errors
    categories = {
        "no_relevant_tool": 0,       # 1. Model didn't call relevant tool
        "called_but_ignored": 0,      # 2. Called tool but ignored result
        "called_but_misinterpreted": 0,  # 3. Called tool but misinterpreted
        "tool_bad_info": 0,           # 4. Tool returned bad info
        "tool_correct_still_wrong": 0,  # 5. Used tool correctly, still wrong
        "correct_no_tools": 0,        # Correct without tools
        "correct_with_tools": 0,      # Correct with tools
    }

    total_analyzed = 0
    total_wrong = 0

    for r in all_items:
        sid = r.get("sample_id")
        if sid is None or sid not in preds:
            continue
        gt = preds[sid]
        fp = r.get("final_prediction", {})
        if not fp:
            continue

        total_analyzed += 1
        is_correct = fp.get("scene") == gt["scene_type_gt"]
        has_tools = len(r.get("tool_calls", [])) > 0

        if is_correct:
            if has_tools:
                categories["correct_with_tools"] += 1
            else:
                categories["correct_no_tools"] += 1
            continue

        total_wrong += 1
        tc_list = r.get("tool_calls", [])

        if not tc_list:
            categories["no_relevant_tool"] += 1
        else:
            # Heuristic classification:
            # Check if model changed mind after tool calls
            changed = r.get("changed_mind", False)

            # If tools were called but model didn't change from its initial assessment
            # and initial was wrong, it likely ignored the tool
            if not changed:
                categories["called_but_ignored"] += 1
            else:
                # Model changed but still wrong -- misinterpretation or bad tool info
                # Check if any tool had errors
                has_tool_error = any(
                    tc.get("result_metadata", {}).get("error") is not None
                    for tc in tc_list
                )
                if has_tool_error:
                    categories["tool_bad_info"] += 1
                else:
                    categories["called_but_misinterpreted"] += 1

    print(f"  Total predictions analyzed: {total_analyzed}")
    print(f"  Total wrong: {total_wrong}")
    print("\n  Error Taxonomy:")
    for cat, count in categories.items():
        pct = count / max(total_wrong, 1) * 100 if "correct" not in cat else count / max(total_analyzed, 1) * 100
        label_type = "of wrong" if "correct" not in cat else "of all"
        print(f"    {cat}: {count} ({pct:.1f}% {label_type})")

    return {
        "task": 69,
        "total_analyzed": total_analyzed,
        "total_wrong": total_wrong,
        "categories": categories,
    }


def run_task_70(all_results: dict) -> dict:
    """Task 70: Cost-benefit summary."""
    print("\n" + "=" * 70)
    print("TASK 70: Cost-benefit summary")
    print("=" * 70)

    # Gather latency and accuracy data from tasks
    configs: list[dict] = []

    # From Task 54 ablation data
    task54 = all_results.get("task_54", {})
    conditions = task54.get("conditions", {})
    for cond_name, cond_data in conditions.items():
        acc = cond_data.get("accuracy", {})
        ts = cond_data.get("tool_stats", {})
        configs.append({
            "config": f"ablation_{cond_name}",
            "scene_accuracy": acc.get("scene_accuracy", 0),
            "all_correct": acc.get("all_correct_accuracy", 0),
            "avg_tools": ts.get("avg_calls_per_sample", 0),
            "total_calls": ts.get("total_calls", 0),
        })

    # From Task 60 (2B vs 8B)
    task60 = all_results.get("task_60", {})
    for model, data in task60.get("comparison", {}).items():
        acc = data.get("accuracy", {})
        ts = data.get("tool_stats", {})
        configs.append({
            "config": f"model_{model}",
            "scene_accuracy": acc.get("scene_accuracy", 0),
            "all_correct": acc.get("all_correct_accuracy", 0),
            "avg_tools": ts.get("avg_calls_per_sample", 0),
            "total_calls": ts.get("total_calls", 0),
        })

    # From Task 68 (Pipeline A vs B)
    task68 = all_results.get("task_68", {})
    for pipe_name in ["pipeline_a", "pipeline_b"]:
        data = task68.get(pipe_name, {})
        acc = data.get("accuracy", {})
        ts = data.get("tool_stats", {})
        configs.append({
            "config": pipe_name,
            "scene_accuracy": acc.get("scene_accuracy", 0),
            "all_correct": acc.get("all_correct_accuracy", 0),
            "avg_tools": ts.get("avg_calls_per_sample", 0),
            "total_calls": ts.get("total_calls", 0),
        })

    # Sort by scene accuracy descending
    configs.sort(key=lambda x: -x["scene_accuracy"])

    print(f"\n  {'Config':<25} {'Scene Acc':>10} {'All Correct':>12} {'Avg Tools':>10}")
    print(f"  {'-'*57}")
    for c in configs:
        print(f"  {c['config']:<25} {c['scene_accuracy']:>10.1%} "
              f"{c['all_correct']:>12.1%} {c['avg_tools']:>10.1f}")

    # Compute Pareto-optimal configs (maximizing accuracy, minimizing tool calls)
    pareto = []
    for c in configs:
        dominated = False
        for other in configs:
            if (other["scene_accuracy"] > c["scene_accuracy"] and
                    other["avg_tools"] <= c["avg_tools"]):
                dominated = True
                break
        if not dominated:
            pareto.append(c["config"])

    print("\n  Pareto-optimal configs (max accuracy, min tool calls):")
    for p in pareto:
        matching = [c for c in configs if c["config"] == p][0]
        print(f"    {p}: scene_acc={matching['scene_accuracy']:.1%}, avg_tools={matching['avg_tools']:.1f}")

    return {
        "task": 70,
        "configs": configs,
        "pareto_optimal": pareto,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PHASE 3: Multi-Tool Interaction Experiments (Tasks 51-70)")
    print("=" * 70)
    print(f"  FAISS index available: {HAS_FAISS}")
    print(f"  Base 2B model: {BASE_2B_PATH} (GPU {BASE_2B_GPU}, port {BASE_2B_PORT})")
    print(f"  Base 8B model: {BASE_8B_PATH} (GPU {BASE_8B_GPU}, port {BASE_8B_PORT})")
    print(f"  System prompt: {SYSTEM_PROMPT}")
    print(f"  Results will be saved to: {RESULTS_PATH}")

    if not HAS_FAISS:
        print("  NOTE: FAISS index not found. Skipping find_similar_scenes and retrieval_only condition.")

    # Load predictions
    print("\nLoading baseline predictions...")
    preds = load_baseline_predictions()
    print(f"  Loaded {len(preds)} baseline predictions")

    # Select samples
    mixed_100 = select_mixed_samples(preds, n_wrong=50, n_correct=50, seed=42)
    correct_100 = select_correct_samples(preds, n=100, seed=42)
    # Extended set for Task 68 (200 samples)
    mixed_200 = select_mixed_samples(preds, n_wrong=100, n_correct=100, seed=43)

    print(f"  Mixed 100 samples: {len(mixed_100)} (50 wrong + 50 correct)")
    print(f"  Correct 100 samples: {len(correct_100)}")
    print(f"  Mixed 200 samples: {len(mixed_200)}")

    # Start servers
    print("\nStarting vLLM servers...")
    kill_servers()
    time.sleep(2)

    procs = []
    proc_2b = start_vllm_server(BASE_2B_PATH, BASE_2B_PORT, BASE_2B_GPU)
    procs.append(proc_2b)
    proc_8b = start_vllm_server(BASE_8B_PATH, BASE_8B_PORT, BASE_8B_GPU)
    procs.append(proc_8b)

    if not wait_for_server(BASE_2B_PORT, timeout=600):
        print("FATAL: 2B server failed to start!")
        kill_servers()
        return
    if not wait_for_server(BASE_8B_PORT, timeout=600):
        print("FATAL: 8B server failed to start!")
        kill_servers()
        return

    all_results: dict[str, Any] = {}
    start_time = time.time()

    try:
        # ===== PRIORITY TASKS =====

        # Task 51: All tools, no guidance (2B)
        t51 = run_task_51(preds, mixed_100)
        all_results["task_51"] = {k: v for k, v in t51.items() if k != "results"}
        all_results["task_51"]["sample_results"] = [
            {
                "sample_id": r.get("sample_id"),
                "final_prediction": r.get("final_prediction"),
                "num_tool_calls": r.get("num_tool_calls", 0),
                "tool_names": [tc["tool_name"] for tc in r.get("tool_calls", [])],
                "error": r.get("error"),
            }
            for r in t51.get("results", [])
        ]
        _save_results(all_results)

        # Task 54: Tool subset ablations (2B)
        t54 = run_task_54(preds, mixed_100)
        all_results["task_54"] = {k: v for k, v in t54.items() if k != "condition_results"}
        _save_results(all_results)

        # Task 55: Interaction effects (from Task 54)
        t55 = run_task_55(t54)
        all_results["task_55"] = t55
        _save_results(all_results)

        # Task 57: Tools on correct predictions
        t57 = run_task_57(preds, correct_100)
        all_results["task_57"] = {k: v for k, v in t57.items() if k != "results"}
        _save_results(all_results)

        # Task 60: 2B vs 8B
        t60 = run_task_60(preds, mixed_100)
        all_results["task_60"] = {k: v for k, v in t60.items() if k != "results_by_model"}
        _save_results(all_results)

        # Task 66: Pipeline A (FT predict, base verify)
        t66 = run_task_66(preds, mixed_100)
        all_results["task_66"] = {k: v for k, v in t66.items() if k != "results"}
        _save_results(all_results)

        # Task 67: Pipeline B (base + tools from scratch)
        t67 = run_task_67(preds, mixed_100)
        all_results["task_67"] = {k: v for k, v in t67.items() if k != "results"}
        _save_results(all_results)

        # Task 68: Pipeline A vs B head-to-head (200 samples)
        t68 = run_task_68(preds, mixed_200)
        all_results["task_68"] = {k: v for k, v in t68.items() if k not in ("results_a", "results_b")}
        _save_results(all_results)

        # ===== SECONDARY TASKS =====

        # Task 52: All tools + reasoning instruction
        t52 = run_task_52(preds, mixed_100)
        all_results["task_52"] = {k: v for k, v in t52.items() if k != "results"}
        _save_results(all_results)

        # Task 53: Prescribed tool order
        t53 = run_task_53(preds, mixed_100)
        all_results["task_53"] = {k: v for k, v in t53.items() if k != "results"}
        _save_results(all_results)

        # Task 56: Tool sequence analysis
        t56 = run_task_56(t51, t52, t53)
        all_results["task_56"] = t56
        _save_results(all_results)

        # Task 61: Tool depth analysis
        t61 = run_task_61(preds, mixed_100)
        all_results["task_61"] = {k: v for k, v in t61.items() if k != "results"}
        _save_results(all_results)

        # Task 65: CoT comparison
        t65 = run_task_65(preds, mixed_100)
        all_results["task_65"] = {k: v for k, v in t65.items() if k != "condition_results"}
        _save_results(all_results)

        # Task 69: Error taxonomy (from all data so far)
        t69 = run_task_69({
            "t51": t51, "t54": t54, "t57": t57, "t60": t60,
            "t66": t66, "t67": t67, "t68": t68,
        })
        all_results["task_69"] = t69
        _save_results(all_results)

        # Task 70: Cost-benefit summary
        t70 = run_task_70(all_results)
        all_results["task_70"] = t70
        _save_results(all_results)

    except Exception as e:
        print(f"\nERROR during experiment: {e}")
        traceback.print_exc()
        all_results["error"] = str(e)
        all_results["error_traceback"] = traceback.format_exc()
    finally:
        elapsed = time.time() - start_time
        all_results["total_wall_time_s"] = round(elapsed, 1)
        _save_results(all_results)

        print(f"\nTotal wall time: {elapsed/60:.1f} minutes")
        print(f"Results saved to: {RESULTS_PATH}")

        # Kill servers
        print("\nKilling servers...")
        kill_servers()
        for proc in procs:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                with contextlib.suppress(Exception):
                    proc.kill()
            if hasattr(proc, '_log_fh'):
                with contextlib.suppress(Exception):
                    proc._log_fh.close()

    # Print final summary
    print_final_summary(all_results)


def _save_results(results: dict):
    """Save results incrementally."""
    try:
        # Make a serializable copy (strip non-serializable items)
        serializable = _make_serializable(results)
        with open(RESULTS_PATH, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
    except Exception as e:
        print(f"  Warning: failed to save results: {e}")


def _make_serializable(obj):
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def print_final_summary(all_results: dict):
    """Print a comprehensive final summary."""
    print("\n" + "=" * 70)
    print("PHASE 3 FINAL SUMMARY")
    print("=" * 70)

    # Task 51 summary
    t51 = all_results.get("task_51", {})
    if t51:
        acc = t51.get("accuracy", {})
        print("\n  Task 51 (All tools, no guidance):")
        print(f"    Scene: {acc.get('scene_accuracy', 0):.1%}, All correct: {acc.get('all_correct_accuracy', 0):.1%}")

    # Task 54 summary
    t54 = all_results.get("task_54", {})
    if t54:
        print("\n  Task 54 (Tool ablations):")
        conds = t54.get("conditions", {})
        for c, data in sorted(conds.items()):
            acc_val = data.get("accuracy", {}).get("scene_accuracy", 0)
            print(f"    {c}: {acc_val:.1%}")

    # Task 57 summary
    t57 = all_results.get("task_57", {})
    if t57:
        harm = t57.get("harm_analysis", {})
        print("\n  Task 57 (Harm rate on correct predictions):")
        print(f"    Harm rate: {harm.get('harm_rate', 0):.1%} ({harm.get('n_broken', 0)}/{harm.get('n_tested', 0)})")

    # Task 60 summary
    t60 = all_results.get("task_60", {})
    if t60:
        print("\n  Task 60 (2B vs 8B):")
        comp = t60.get("comparison", {})
        for model, data in comp.items():
            acc = data.get("accuracy", {})
            print(f"    {model}: scene={acc.get('scene_accuracy', 0):.1%}")

    # Task 66 summary
    t66 = all_results.get("task_66", {})
    if t66:
        print("\n  Task 66 (Pipeline A: FT+verify):")
        print(f"    Scene: {t66.get('accuracy', {}).get('scene_accuracy', 0):.1%}, "
              f"FT baseline: {t66.get('ft_baseline_accuracy', 0):.1%}, "
              f"Improvement: {t66.get('improvement', 0):+.1%}")
        print(f"    Saves: {t66.get('saves', 0)}, Breaks: {t66.get('breaks', 0)}")

    # Task 67 summary
    t67 = all_results.get("task_67", {})
    if t67:
        print("\n  Task 67 (Pipeline B: base+tools):")
        print(f"    Scene: {t67.get('accuracy', {}).get('scene_accuracy', 0):.1%}, "
              f"FT baseline: {t67.get('ft_baseline_accuracy', 0):.1%}")

    # Task 68 summary
    t68 = all_results.get("task_68", {})
    if t68:
        print("\n  Task 68 (Pipeline A vs B head-to-head):")
        pa = t68.get("pipeline_a", {}).get("accuracy", {})
        pb = t68.get("pipeline_b", {}).get("accuracy", {})
        ps = t68.get("per_sample", {})
        print(f"    Pipeline A: {pa.get('scene_accuracy', 0):.1%}")
        print(f"    Pipeline B: {pb.get('scene_accuracy', 0):.1%}")
        print(f"    Per-sample: A wins={ps.get('a_wins', 0)}, B wins={ps.get('b_wins', 0)}")

    # Task 69 error taxonomy
    t69 = all_results.get("task_69", {})
    if t69:
        print("\n  Task 69 (Error taxonomy):")
        cats = t69.get("categories", {})
        for cat, count in cats.items():
            print(f"    {cat}: {count}")

    # Task 70 Pareto
    t70 = all_results.get("task_70", {})
    if t70:
        print("\n  Task 70 (Pareto-optimal configs):")
        for p in t70.get("pareto_optimal", []):
            print(f"    {p}")

    wall = all_results.get("total_wall_time_s", 0)
    print(f"\n  Total wall time: {wall/60:.1f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    # Ensure unbuffered output for background execution
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
