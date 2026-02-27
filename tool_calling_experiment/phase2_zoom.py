#!/usr/bin/env python3
# ruff: noqa: E501,E402
# type: ignore[import-not-found]
"""Phase 2 Zoom Tool Deep Dive: Tasks 16-25 for visual tool-calling experiment.

Systematically evaluates how the zoom_region tool affects scene classification
accuracy, using 100-sample batches across multiple experimental conditions.

Tasks:
    16: Zoom on incident_zone predictions (2B, 100 samples)
    17: Same as 16 with 8B model
    18: Zoom location quality analysis (from Tasks 16-17)
    19: Multiple zooms up to 3 rounds (2B, 100 samples)
    20: Zoom on correct predictions -- harm rate (2B, 100 samples)
    21: Directed vs free zoom (2B, 100 false IZ, 2 conditions)
    22: Crop size comparison (2B, 100 false IZ, 3 conditions)
    23: Zoom as confidence signal (2B, 100 mixed)
    24: Zoom on non-incident errors (2B, 100 non-IZ errors)
    25: Token cost analysis (aggregated from all prior tasks)

Usage:
    python tool_calling_experiment/phase2_zoom.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any

import requests

# Ensure sibling modules are importable
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from orchestrator import ToolCallingOrchestrator
from visual_tools import (
    TOOL_ZOOM,
    load_sample_image,
    zoom_region,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(_DIR, "phase2_zoom_results.json")
SC_DB_PATH = os.path.join(
    os.path.dirname(_DIR), "self_consistency_experiment", "self_consistency.db"
)

BASE_2B_PATH = "/fsx/models/Qwen3-VL-2B-Instruct"
BASE_8B_PATH = "/fsx/models/Qwen3-VL-8B-Instruct"

PORT_2B = 8330
PORT_8B = 8331
GPU_2B = 6
GPU_8B = 7

IMG_W = 504
IMG_H = 336

VALID_SCENES = frozenset(
    ["nominal", "flooded", "incident_zone", "mounted_police", "flagger"]
)


# ---------------------------------------------------------------------------
# Server management
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

    log_path = f"/tmp/vllm_phase2_{port}.log"
    print(f"  Starting server: model={os.path.basename(model_path)}, GPU={gpu_id}, port={port}")
    print(f"  Log: {log_path}")
    log_file = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd, env=env, cwd="/tmp",
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_server(
    port: int,
    timeout: int = 600,
    proc: subprocess.Popen | None = None,
) -> bool:
    """Wait for vLLM server to become healthy."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        # Check if the process died
        if proc is not None and proc.poll() is not None:
            print(f"  Server process on port {port} died (exit code {proc.returncode})")
            log_path = f"/tmp/vllm_phase2_{port}.log"
            if os.path.exists(log_path):
                with open(log_path) as lf:
                    content = lf.read()
                    if content:
                        print(f"  Last log lines:\n{content[-500:]}")
            return False
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


def is_server_healthy(port: int) -> bool:
    """Check if a vLLM server is already running and healthy."""
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def ensure_2b_server(model_path: str = BASE_2B_PATH) -> bool:
    """Ensure the 2B server on PORT_2B is healthy; restart if not."""
    if is_server_healthy(PORT_2B):
        return True
    print(f"  2B server on port {PORT_2B} is not healthy. Restarting...")
    # Kill any existing process on the port
    subprocess.run(
        ["pkill", "-9", "-f", f"vllm serve.*{PORT_2B}"], capture_output=True,
    )
    time.sleep(5)
    proc = start_vllm_server(model_path, PORT_2B, GPU_2B)
    return wait_for_server(PORT_2B, timeout=600, proc=proc)


def kill_servers() -> None:
    """Kill all phase2 vllm servers."""
    subprocess.run(["pkill", "-f", "vllm serve.*833[01]"], capture_output=True)
    time.sleep(2)


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def load_sample_sets() -> dict[str, list[int]]:
    """Load sample ID sets from the self-consistency database.

    Returns dict with keys:
        false_iz_50: 50 false incident_zone samples (pred=IZ, gt=nominal)
        correct_25:  25 correctly predicted samples
        other_err_25: 25 other error type samples
        mixed_100:   all 100 combined (for tasks 16-17, 19, 23)
        correct_100: 100 correctly predicted samples (for task 20)
        false_iz_100: 100 false IZ samples (for tasks 21, 22)
        other_err_100: 100 non-IZ error samples (for task 24)
    """
    conn = sqlite3.connect(f"file:{SC_DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    # False IZ: predicted=incident_zone AND gt=nominal
    cursor.execute(
        "SELECT sample_id FROM predictions "
        "WHERE predicted_scene='incident_zone' AND scene_type_gt='nominal' "
        "ORDER BY sample_id"
    )
    all_false_iz = [r[0] for r in cursor.fetchall()]

    # Correct predictions
    cursor.execute(
        "SELECT sample_id FROM predictions "
        "WHERE predicted_scene=scene_type_gt "
        "ORDER BY sample_id"
    )
    all_correct = [r[0] for r in cursor.fetchall()]

    # Other errors (not false IZ, not correct)
    cursor.execute(
        "SELECT sample_id FROM predictions "
        "WHERE predicted_scene!=scene_type_gt "
        "AND NOT (predicted_scene='incident_zone' AND scene_type_gt='nominal') "
        "ORDER BY sample_id"
    )
    all_other_err = [r[0] for r in cursor.fetchall()]

    conn.close()

    # Deterministic selection: take first N for reproducibility
    false_iz_50 = all_false_iz[:50]
    correct_25 = all_correct[:25]
    other_err_25 = all_other_err[:25]
    mixed_100 = false_iz_50 + correct_25 + other_err_25

    correct_100 = all_correct[:100]
    false_iz_100 = all_false_iz[:100]
    other_err_100 = all_other_err[:100]

    print("  Sample sets loaded:")
    print(f"    false_iz_50:   {len(false_iz_50)} samples (IDs: {false_iz_50[:5]}...)")
    print(f"    correct_25:    {len(correct_25)} samples (IDs: {correct_25[:5]}...)")
    print(f"    other_err_25:  {len(other_err_25)} samples (IDs: {other_err_25[:5]}...)")
    print(f"    mixed_100:     {len(mixed_100)} samples")
    print(f"    correct_100:   {len(correct_100)} samples")
    print(f"    false_iz_100:  {len(false_iz_100)} samples")
    print(f"    other_err_100: {len(other_err_100)} samples")

    return {
        "false_iz_50": false_iz_50,
        "correct_25": correct_25,
        "other_err_25": other_err_25,
        "mixed_100": mixed_100,
        "correct_100": correct_100,
        "false_iz_100": false_iz_100,
        "other_err_100": other_err_100,
    }


def load_ground_truth() -> dict[int, dict[str, str]]:
    """Load ground truth labels from the DB keyed by sample_id."""
    conn = sqlite3.connect(f"file:{SC_DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sample_id, scene_type_gt, long_action_gt, lat_action_gt, "
        "predicted_scene FROM predictions"
    )
    gt = {}
    for sid, scene, la, lat, pred in cursor.fetchall():
        gt[sid] = {
            "scene_type_gt": scene,
            "long_action_gt": la,
            "lat_action_gt": lat,
            "baseline_prediction": pred,
        }
    conn.close()
    return gt


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = (
    "You are an expert driving scene analyst. "
    "The image is 504x336 pixels. When specifying coordinates, "
    "keep x in range 0-503 and y in range 0-335."
)

USER_PROMPT_T16 = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "If you think it might be an incident or unusual situation, "
    "zoom into the area of concern to verify before making your "
    "final classification.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

USER_PROMPT_T19 = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "You can zoom into up to 3 different regions. Investigate "
    "the scene thoroughly before classifying.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

USER_PROMPT_T20 = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "If you want to examine any part of the scene more closely, "
    "you can zoom into a region.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

USER_PROMPT_T21_FREE = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "Zoom into any region you think is relevant before classifying.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

USER_PROMPT_T21_DIRECTED = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "Zoom into the center of the road ahead (approximately x=252, y=200) "
    "before classifying.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)

USER_PROMPT_T23 = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "If you think a closer look at any region would help, zoom in.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>\n"
    "CONFIDENCE: <1-5>\n"
    "(1 = very unsure, 5 = very confident)"
)

USER_PROMPT_T24 = (
    "Classify this driving scene. The possible types are: "
    "nominal, flooded, incident_zone, mounted_police, flagger. "
    "If you think a closer look at any region would help, zoom in.\n\n"
    "After your analysis, provide your answer in this format:\n"
    "FINAL_SCENE: <scene_type>"
)


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------

def make_orchestrator(
    port: int,
    max_tool_rounds: int = 1,
    max_tokens: int = 1024,
    forced_crop_size: int | None = None,
) -> ToolCallingOrchestrator:
    """Create a ToolCallingOrchestrator for the zoom experiment.

    Args:
        port: vLLM server port
        max_tool_rounds: Maximum tool call rounds
        max_tokens: Max tokens per generation
        forced_crop_size: If set, wraps zoom_region to force this crop size
    """
    if forced_crop_size is not None:
        def zoom_with_forced_crop(
            image_path: str, center_x: int, center_y: int, crop_size: int = 128
        ) -> dict[str, Any]:
            return zoom_region(
                image_path=image_path,
                center_x=center_x,
                center_y=center_y,
                crop_size=forced_crop_size,
            )
        tool_fn = zoom_with_forced_crop
    else:
        tool_fn = zoom_region

    return ToolCallingOrchestrator(
        server_url=f"http://localhost:{port}",
        tools={"zoom_region": tool_fn},
        tool_definitions=[TOOL_ZOOM],
        max_tool_rounds=max_tool_rounds,
        temperature=0,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Running a batch of samples through the orchestrator
# ---------------------------------------------------------------------------

def run_batch(
    orchestrator: ToolCallingOrchestrator,
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
    system_prompt: str,
    user_prompt: str,
    tool_choice: str = "auto",
    task_name: str = "task",
) -> list[dict[str, Any]]:
    """Run a batch of samples through the orchestrator.

    Returns a list of result dicts, one per sample.
    """
    results = []
    total = len(sample_ids)

    for i, sid in enumerate(sample_ids):
        if i % 10 == 0:
            print(f"  [{task_name}] Progress: {i}/{total}")

        # Load image
        try:
            img_path = load_sample_image(sid)
        except Exception as exc:
            results.append({
                "sample_id": sid,
                "error": f"Image load failed: {exc}",
                "ground_truth": ground_truth.get(sid, {}),
            })
            continue

        # Run orchestrator
        try:
            orch_result = orchestrator.run(
                image_path=img_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tool_choice=tool_choice,
            )
        except Exception as exc:
            results.append({
                "sample_id": sid,
                "error": f"Orchestrator failed: {exc}",
                "ground_truth": ground_truth.get(sid, {}),
            })
            continue

        gt = ground_truth.get(sid, {})
        pred = orch_result["final_prediction"]

        # Extract zoom call details
        zoom_calls = []
        for tc in orch_result.get("tool_calls", []):
            if tc["tool_name"] == "zoom_region":
                zoom_calls.append({
                    "center_x": tc["arguments"].get("center_x"),
                    "center_y": tc["arguments"].get("center_y"),
                    "crop_size": tc["arguments"].get("crop_size", 128),
                    "round": tc.get("round"),
                })

        # Strip full_conversation to save space (it contains base64 images)
        result_entry = {
            "sample_id": sid,
            "ground_truth": gt,
            "predicted_scene": pred.get("scene"),
            "baseline_prediction": gt.get("baseline_prediction"),
            "correct": pred.get("scene") == gt.get("scene_type_gt"),
            "baseline_correct": gt.get("baseline_prediction") == gt.get("scene_type_gt"),
            "num_rounds": orch_result["num_rounds"],
            "num_tool_calls": orch_result["num_tool_calls"],
            "zoom_calls": zoom_calls,
            "changed_mind": orch_result["changed_mind"],
            "final_text": orch_result["final_text"],
            "latency_ms": orch_result["latency_ms"],
            "generation_ms": orch_result["generation_ms"],
            "tool_execution_ms": orch_result["tool_execution_ms"],
            "error": orch_result.get("error"),
        }

        # Estimate token counts from text lengths (rough: 1 token ~ 4 chars)
        reasoning_text = orch_result.get("reasoning_text", "")
        result_entry["approx_total_tokens"] = len(reasoning_text) // 4

        results.append(result_entry)

    print(f"  [{task_name}] Completed: {total}/{total}")
    return results


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy metrics from a batch of results."""
    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    baseline_correct = sum(1 for r in results if r.get("baseline_correct"))
    errors = sum(1 for r in results if r.get("error"))
    used_zoom = sum(1 for r in results if r.get("num_tool_calls", 0) > 0)
    no_prediction = sum(1 for r in results if r.get("predicted_scene") is None and not r.get("error"))

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": round(baseline_correct / total, 4) if total > 0 else 0,
        "accuracy_delta": round((correct - baseline_correct) / total, 4) if total > 0 else 0,
        "used_zoom": used_zoom,
        "zoom_rate": round(used_zoom / total, 4) if total > 0 else 0,
        "errors": errors,
        "no_prediction": no_prediction,
    }


def compute_zoom_location_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze zoom locations from results (for Task 18)."""
    location_categories = {
        "center": 0,  # within 100px of image center
        "top": 0,     # y < 112
        "bottom": 0,  # y > 224
        "left": 0,    # x < 168
        "right": 0,   # x > 336
    }
    road_area_zooms = 0  # y > 168, center third of x
    total_zooms = 0
    zoom_details = []

    for r in results:
        for zc in r.get("zoom_calls", []):
            cx = zc.get("center_x")
            cy = zc.get("center_y")
            cs = zc.get("crop_size", 128)
            if cx is None or cy is None:
                continue

            total_zooms += 1
            detail = {"center_x": cx, "center_y": cy, "crop_size": cs, "categories": []}

            # Center: within 100px of (252, 168)
            if abs(cx - 252) <= 100 and abs(cy - 168) <= 100:
                location_categories["center"] += 1
                detail["categories"].append("center")

            # Positional categories
            if cy < 112:
                location_categories["top"] += 1
                detail["categories"].append("top")
            elif cy > 224:
                location_categories["bottom"] += 1
                detail["categories"].append("bottom")

            if cx < 168:
                location_categories["left"] += 1
                detail["categories"].append("left")
            elif cx > 336:
                location_categories["right"] += 1
                detail["categories"].append("right")

            # Road area: y > 168, center third of x (168-336)
            if cy > 168 and 168 <= cx <= 336:
                road_area_zooms += 1
                detail["categories"].append("road_area")

            zoom_details.append(detail)

    return {
        "total_zooms": total_zooms,
        "location_categories": location_categories,
        "road_area_zooms": road_area_zooms,
        "road_area_rate": round(road_area_zooms / total_zooms, 4) if total_zooms > 0 else 0,
        "zoom_details": zoom_details[:20],  # sample for inspection
    }


def compute_multi_zoom_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze accuracy by number of zooms used (for Task 19)."""
    by_zoom_count: dict[int, dict[str, int]] = {}
    for r in results:
        n = r.get("num_tool_calls", 0)
        if n not in by_zoom_count:
            by_zoom_count[n] = {"total": 0, "correct": 0}
        by_zoom_count[n]["total"] += 1
        if r.get("correct"):
            by_zoom_count[n]["correct"] += 1

    stats = {}
    for n_zooms in sorted(by_zoom_count.keys()):
        d = by_zoom_count[n_zooms]
        stats[f"{n_zooms}_zooms"] = {
            "total": d["total"],
            "correct": d["correct"],
            "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0,
        }

    return stats


def compute_harm_rate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure how often zoom changes a correct baseline prediction (Task 20)."""
    total = len(results)
    baseline_correct = sum(1 for r in results if r.get("baseline_correct"))
    still_correct = sum(1 for r in results if r.get("baseline_correct") and r.get("correct"))
    flipped_wrong = sum(1 for r in results if r.get("baseline_correct") and not r.get("correct") and r.get("predicted_scene") is not None)
    no_prediction = sum(1 for r in results if r.get("predicted_scene") is None and not r.get("error"))

    return {
        "total_samples": total,
        "baseline_correct": baseline_correct,
        "still_correct_after_zoom": still_correct,
        "flipped_to_wrong": flipped_wrong,
        "no_prediction": no_prediction,
        "harm_rate": round(flipped_wrong / baseline_correct, 4) if baseline_correct > 0 else 0,
        "retention_rate": round(still_correct / baseline_correct, 4) if baseline_correct > 0 else 0,
    }


def extract_confidence(text: str) -> int | None:
    """Extract confidence rating (1-5) from model output."""
    if not text:
        return None
    m = re.search(r"CONFIDENCE:\s*(\d)", text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 5:
            return val
    # Fallback: look for patterns like "confidence: 4" or "confidence of 3"
    m2 = re.search(r"confidence[:\s]+(?:of\s+)?(\d)", text, re.IGNORECASE)
    if m2:
        val = int(m2.group(1))
        if 1 <= val <= 5:
            return val
    return None


def compute_confidence_correlation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze correlation between confidence and accuracy (Task 23)."""
    by_confidence: dict[int, dict[str, int]] = {}
    no_confidence = 0

    for r in results:
        conf = extract_confidence(r.get("final_text", ""))
        if conf is None:
            no_confidence += 1
            continue
        if conf not in by_confidence:
            by_confidence[conf] = {"total": 0, "correct": 0}
        by_confidence[conf]["total"] += 1
        if r.get("correct"):
            by_confidence[conf]["correct"] += 1

    stats = {}
    for c in sorted(by_confidence.keys()):
        d = by_confidence[c]
        stats[f"confidence_{c}"] = {
            "total": d["total"],
            "correct": d["correct"],
            "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0,
        }

    return {
        "by_confidence": stats,
        "no_confidence_count": no_confidence,
    }


def compute_token_stats(all_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate token cost analysis across all tasks (Task 25)."""
    task_stats = {}
    for task_name, results in all_results.items():
        total_tokens = sum(r.get("approx_total_tokens", 0) for r in results)
        total_latency = sum(r.get("latency_ms", 0) for r in results)
        total_gen_ms = sum(r.get("generation_ms", 0) for r in results)
        total_tool_ms = sum(r.get("tool_execution_ms", 0) for r in results)
        total_zooms = sum(r.get("num_tool_calls", 0) for r in results)
        n = len(results)

        acc = compute_accuracy(results)

        task_stats[task_name] = {
            "total_samples": n,
            "total_approx_tokens": total_tokens,
            "avg_tokens_per_sample": round(total_tokens / n, 1) if n > 0 else 0,
            "avg_tokens_per_zoom": round(total_tokens / total_zooms, 1) if total_zooms > 0 else 0,
            "total_zooms": total_zooms,
            "avg_latency_ms": round(total_latency / n, 1) if n > 0 else 0,
            "avg_generation_ms": round(total_gen_ms / n, 1) if n > 0 else 0,
            "avg_tool_exec_ms": round(total_tool_ms / n, 1) if n > 0 else 0,
            "accuracy": acc["accuracy"],
            "baseline_accuracy": acc["baseline_accuracy"],
            "accuracy_delta": acc["accuracy_delta"],
        }

    return task_stats


# ---------------------------------------------------------------------------
# Per-task implementations
# ---------------------------------------------------------------------------

def task_16(
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Task 16: Zoom on incident_zone predictions (2B, 100 samples)."""
    print("\n" + "=" * 70)
    print("TASK 16: Zoom on incident_zone predictions (2B, 100 mixed samples)")
    print("=" * 70)

    orch = make_orchestrator(PORT_2B, max_tool_rounds=1)
    results = run_batch(
        orch, sample_ids, ground_truth,
        system_prompt=SYSTEM_PROMPT_BASE,
        user_prompt=USER_PROMPT_T16,
        tool_choice="auto",
        task_name="T16",
    )
    acc = compute_accuracy(results)
    print(f"  T16 Results: accuracy={acc['accuracy']:.4f} "
          f"(baseline={acc['baseline_accuracy']:.4f}, delta={acc['accuracy_delta']:+.4f})")
    print(f"  Zoom usage: {acc['used_zoom']}/{acc['total']} ({acc['zoom_rate']:.2%})")
    return {"results": results, "accuracy": acc}


def task_17(
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Task 17: Same as 16 with 8B model."""
    print("\n" + "=" * 70)
    print("TASK 17: Zoom on incident_zone predictions (8B, 100 mixed samples)")
    print("=" * 70)

    orch = make_orchestrator(PORT_8B, max_tool_rounds=1)
    results = run_batch(
        orch, sample_ids, ground_truth,
        system_prompt=SYSTEM_PROMPT_BASE,
        user_prompt=USER_PROMPT_T16,
        tool_choice="auto",
        task_name="T17",
    )
    acc = compute_accuracy(results)
    print(f"  T17 Results: accuracy={acc['accuracy']:.4f} "
          f"(baseline={acc['baseline_accuracy']:.4f}, delta={acc['accuracy_delta']:+.4f})")
    print(f"  Zoom usage: {acc['used_zoom']}/{acc['total']} ({acc['zoom_rate']:.2%})")
    return {"results": results, "accuracy": acc}


def task_18(
    t16_results: list[dict[str, Any]],
    t17_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task 18: Zoom location quality analysis (from Tasks 16-17)."""
    print("\n" + "=" * 70)
    print("TASK 18: Zoom location quality analysis")
    print("=" * 70)

    stats_2b = compute_zoom_location_stats(t16_results)
    stats_8b = compute_zoom_location_stats(t17_results)

    print("  2B zoom stats:")
    print(f"    Total zooms: {stats_2b['total_zooms']}")
    print(f"    Location distribution: {stats_2b['location_categories']}")
    print(f"    Road area zooms: {stats_2b['road_area_zooms']} ({stats_2b['road_area_rate']:.2%})")

    print("  8B zoom stats:")
    print(f"    Total zooms: {stats_8b['total_zooms']}")
    print(f"    Location distribution: {stats_8b['location_categories']}")
    print(f"    Road area zooms: {stats_8b['road_area_zooms']} ({stats_8b['road_area_rate']:.2%})")

    # Analyze 50 samples worth of detail (first 50 from t16+t17 that had zooms)
    zoom_samples_2b = [r for r in t16_results if r.get("num_tool_calls", 0) > 0][:25]
    zoom_samples_8b = [r for r in t17_results if r.get("num_tool_calls", 0) > 0][:25]

    manual_review = []
    for r in zoom_samples_2b + zoom_samples_8b:
        for zc in r.get("zoom_calls", []):
            manual_review.append({
                "sample_id": r["sample_id"],
                "model": "2B" if r in zoom_samples_2b else "8B",
                "center_x": zc.get("center_x"),
                "center_y": zc.get("center_y"),
                "crop_size": zc.get("crop_size", 128),
                "predicted_scene": r.get("predicted_scene"),
                "gt_scene": r.get("ground_truth", {}).get("scene_type_gt"),
                "correct": r.get("correct"),
            })

    return {
        "stats_2b": stats_2b,
        "stats_8b": stats_8b,
        "manual_review_50": manual_review[:50],
    }


def task_19(
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Task 19: Multiple zooms, max 3 rounds (2B, 100 samples)."""
    print("\n" + "=" * 70)
    print("TASK 19: Multiple zooms, max 3 rounds (2B, 100 samples)")
    print("=" * 70)

    orch = make_orchestrator(PORT_2B, max_tool_rounds=3)
    results = run_batch(
        orch, sample_ids, ground_truth,
        system_prompt=SYSTEM_PROMPT_BASE,
        user_prompt=USER_PROMPT_T19,
        tool_choice="auto",
        task_name="T19",
    )
    acc = compute_accuracy(results)
    multi_stats = compute_multi_zoom_stats(results)

    print(f"  T19 Results: accuracy={acc['accuracy']:.4f} "
          f"(baseline={acc['baseline_accuracy']:.4f}, delta={acc['accuracy_delta']:+.4f})")
    print("  Accuracy by zoom count:")
    for k, v in multi_stats.items():
        print(f"    {k}: {v['correct']}/{v['total']} = {v['accuracy']:.4f}")

    return {"results": results, "accuracy": acc, "multi_zoom_stats": multi_stats}


def task_20(
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Task 20: Zoom on correct predictions -- harm rate (2B, 100 correct samples)."""
    print("\n" + "=" * 70)
    print("TASK 20: Zoom on correct predictions -- harm rate (2B, 100 samples)")
    print("=" * 70)

    orch = make_orchestrator(PORT_2B, max_tool_rounds=1)
    results = run_batch(
        orch, sample_ids, ground_truth,
        system_prompt=SYSTEM_PROMPT_BASE,
        user_prompt=USER_PROMPT_T20,
        tool_choice="auto",
        task_name="T20",
    )
    acc = compute_accuracy(results)
    harm = compute_harm_rate(results)

    print(f"  T20 Results: accuracy={acc['accuracy']:.4f} "
          f"(baseline={acc['baseline_accuracy']:.4f})")
    print(f"  Harm rate: {harm['harm_rate']:.4f} "
          f"({harm['flipped_to_wrong']}/{harm['baseline_correct']} flipped)")
    print(f"  Retention rate: {harm['retention_rate']:.4f}")

    return {"results": results, "accuracy": acc, "harm_rate": harm}


def task_21(
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Task 21: Directed vs free zoom (2B, 100 false IZ, 2 conditions)."""
    print("\n" + "=" * 70)
    print("TASK 21: Directed vs free zoom (2B, 100 false IZ samples, 2 conditions)")
    print("=" * 70)

    # Condition A: free zoom
    print("  -- Condition A: Free zoom --")
    orch_a = make_orchestrator(PORT_2B, max_tool_rounds=1)
    results_a = run_batch(
        orch_a, sample_ids, ground_truth,
        system_prompt=SYSTEM_PROMPT_BASE,
        user_prompt=USER_PROMPT_T21_FREE,
        tool_choice="auto",
        task_name="T21a",
    )
    acc_a = compute_accuracy(results_a)

    # Condition B: directed zoom
    print("  -- Condition B: Directed zoom --")
    orch_b = make_orchestrator(PORT_2B, max_tool_rounds=1)
    results_b = run_batch(
        orch_b, sample_ids, ground_truth,
        system_prompt=SYSTEM_PROMPT_BASE,
        user_prompt=USER_PROMPT_T21_DIRECTED,
        tool_choice="auto",
        task_name="T21b",
    )
    acc_b = compute_accuracy(results_b)

    print(f"  T21 Condition A (free):     accuracy={acc_a['accuracy']:.4f} "
          f"(zoom_rate={acc_a['zoom_rate']:.2%})")
    print(f"  T21 Condition B (directed): accuracy={acc_b['accuracy']:.4f} "
          f"(zoom_rate={acc_b['zoom_rate']:.2%})")
    print(f"  Delta (directed - free): {acc_b['accuracy'] - acc_a['accuracy']:+.4f}")

    return {
        "condition_a_free": {"results": results_a, "accuracy": acc_a},
        "condition_b_directed": {"results": results_b, "accuracy": acc_b},
        "accuracy_delta": round(acc_b["accuracy"] - acc_a["accuracy"], 4),
    }


def task_22(
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Task 22: Crop size comparison (2B, 100 false IZ, 3 conditions)."""
    print("\n" + "=" * 70)
    print("TASK 22: Crop size comparison (2B, 100 false IZ, sizes: 64, 128, 256)")
    print("=" * 70)

    crop_results = {}
    for crop_size in [64, 128, 256]:
        print(f"  -- Crop size: {crop_size} --")
        orch = make_orchestrator(PORT_2B, max_tool_rounds=1, forced_crop_size=crop_size)
        results = run_batch(
            orch, sample_ids, ground_truth,
            system_prompt=SYSTEM_PROMPT_BASE,
            user_prompt=USER_PROMPT_T16,
            tool_choice="auto",
            task_name=f"T22-{crop_size}",
        )
        acc = compute_accuracy(results)
        crop_results[f"crop_{crop_size}"] = {"results": results, "accuracy": acc}
        print(f"  Crop {crop_size}: accuracy={acc['accuracy']:.4f} "
              f"(zoom_rate={acc['zoom_rate']:.2%})")

    return crop_results


def task_23(
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Task 23: Zoom as confidence signal (2B, 100 mixed)."""
    print("\n" + "=" * 70)
    print("TASK 23: Zoom as confidence signal (2B, 100 mixed samples)")
    print("=" * 70)

    orch = make_orchestrator(PORT_2B, max_tool_rounds=1)
    results = run_batch(
        orch, sample_ids, ground_truth,
        system_prompt=SYSTEM_PROMPT_BASE,
        user_prompt=USER_PROMPT_T23,
        tool_choice="auto",
        task_name="T23",
    )
    acc = compute_accuracy(results)
    conf_stats = compute_confidence_correlation(results)

    print(f"  T23 Results: accuracy={acc['accuracy']:.4f}")
    print("  Confidence distribution:")
    for k, v in conf_stats["by_confidence"].items():
        print(f"    {k}: {v['correct']}/{v['total']} = {v['accuracy']:.4f}")
    print(f"  Samples without confidence: {conf_stats['no_confidence_count']}")

    return {"results": results, "accuracy": acc, "confidence_stats": conf_stats}


def task_24(
    sample_ids: list[int],
    ground_truth: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Task 24: Zoom on non-incident errors (2B, 100 non-IZ errors)."""
    print("\n" + "=" * 70)
    print("TASK 24: Zoom on non-IZ errors (2B, 100 samples)")
    print("=" * 70)

    orch = make_orchestrator(PORT_2B, max_tool_rounds=1)
    results = run_batch(
        orch, sample_ids, ground_truth,
        system_prompt=SYSTEM_PROMPT_BASE,
        user_prompt=USER_PROMPT_T24,
        tool_choice="auto",
        task_name="T24",
    )
    acc = compute_accuracy(results)

    # Breakdown by error type (predicted vs GT)
    error_breakdown: dict[str, dict[str, int]] = {}
    for r in results:
        gt_scene = r.get("ground_truth", {}).get("scene_type_gt", "unknown")
        baseline = r.get("baseline_prediction", "unknown")
        key = f"{baseline}->{gt_scene}"
        if key not in error_breakdown:
            error_breakdown[key] = {"total": 0, "fixed": 0, "stayed_wrong": 0}
        error_breakdown[key]["total"] += 1
        if r.get("correct"):
            error_breakdown[key]["fixed"] += 1
        else:
            error_breakdown[key]["stayed_wrong"] += 1

    print(f"  T24 Results: accuracy={acc['accuracy']:.4f} "
          f"(baseline={acc['baseline_accuracy']:.4f}, delta={acc['accuracy_delta']:+.4f})")
    print("  Error type breakdown:")
    for k, v in sorted(error_breakdown.items(), key=lambda x: -x[1]["total"]):
        fix_rate = v["fixed"] / v["total"] if v["total"] > 0 else 0
        print(f"    {k}: {v['total']} samples, {v['fixed']} fixed ({fix_rate:.2%})")

    return {"results": results, "accuracy": acc, "error_breakdown": error_breakdown}


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("PHASE 2: Zoom Tool Deep Dive (Tasks 16-25)")
    print("=" * 70)
    start_time = time.time()

    # Load sample sets and ground truth
    print("\n--- Loading sample sets ---")
    sample_sets = load_sample_sets()
    ground_truth = load_ground_truth()
    print(f"  Ground truth loaded for {len(ground_truth)} samples")

    all_results: dict[str, Any] = {
        "sample_sets": {k: v for k, v in sample_sets.items()},
        "tasks": {},
    }

    # ------------------------------------------------------------------
    # Start 2B server (needed for most tasks)
    # ------------------------------------------------------------------
    print("\n--- Starting 2B server ---")
    if is_server_healthy(PORT_2B):
        print(f"  2B server already running on port {PORT_2B}")
    else:
        proc_2b = start_vllm_server(BASE_2B_PATH, PORT_2B, GPU_2B)
        if not wait_for_server(PORT_2B, timeout=600, proc=proc_2b):
            print("  FATAL: 2B server failed to start!")
            return

    # ------------------------------------------------------------------
    # Task 16: Zoom on incident_zone predictions (2B)
    # ------------------------------------------------------------------
    t16 = task_16(sample_sets["mixed_100"], ground_truth)
    all_results["tasks"]["task_16"] = {
        "description": "Zoom on IZ predictions (2B, 100 mixed)",
        "accuracy": t16["accuracy"],
    }
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Start 8B server for Task 17
    # ------------------------------------------------------------------
    print("\n--- Starting 8B server ---")
    server_8b_proc = None
    t17_results: str | None = None
    if is_server_healthy(PORT_8B):
        print(f"  8B server already running on port {PORT_8B}")
        t17_results = "ready"
    else:
        server_8b_proc = start_vllm_server(BASE_8B_PATH, PORT_8B, GPU_8B)
        if not wait_for_server(PORT_8B, timeout=600, proc=server_8b_proc):
            print("  WARNING: 8B server failed to start. Skipping Task 17.")
            t17_results = None
        else:
            t17_results = "ready"

    # ------------------------------------------------------------------
    # Task 17: Zoom on IZ predictions (8B)
    # ------------------------------------------------------------------
    if t17_results == "ready":
        t17 = task_17(sample_sets["mixed_100"], ground_truth)
        all_results["tasks"]["task_17"] = {
            "description": "Zoom on IZ predictions (8B, 100 mixed)",
            "accuracy": t17["accuracy"],
        }
    else:
        t17 = None
        all_results["tasks"]["task_17"] = {"description": "SKIPPED: 8B server unavailable"}
    _save_results(all_results)

    # Stop 8B server to free GPU memory (not needed for remaining tasks)
    if server_8b_proc is not None:
        print("\n--- Stopping 8B server (no longer needed) ---")
        server_8b_proc.terminate()
        try:
            server_8b_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_8b_proc.kill()
        time.sleep(5)

    # ------------------------------------------------------------------
    # Task 18: Zoom location quality analysis (from T16, T17)
    # ------------------------------------------------------------------
    t17_res = t17["results"] if t17 is not None else []
    t18 = task_18(t16["results"], t17_res)
    all_results["tasks"]["task_18"] = {
        "description": "Zoom location quality analysis",
        "stats_2b": t18["stats_2b"],
        "stats_8b": t18["stats_8b"],
        "manual_review_count": len(t18["manual_review_50"]),
    }
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Task 19: Multiple zooms, max 3 rounds (2B)
    # ------------------------------------------------------------------
    ensure_2b_server()
    t19 = task_19(sample_sets["mixed_100"], ground_truth)
    all_results["tasks"]["task_19"] = {
        "description": "Multiple zooms max 3 (2B, 100 mixed)",
        "accuracy": t19["accuracy"],
        "multi_zoom_stats": t19["multi_zoom_stats"],
    }
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Task 20: Zoom on correct predictions -- harm rate (2B)
    # ------------------------------------------------------------------
    ensure_2b_server()
    t20 = task_20(sample_sets["correct_100"], ground_truth)
    all_results["tasks"]["task_20"] = {
        "description": "Zoom harm rate on correct predictions (2B, 100)",
        "accuracy": t20["accuracy"],
        "harm_rate": t20["harm_rate"],
    }
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Task 21: Directed vs free zoom (2B)
    # ------------------------------------------------------------------
    ensure_2b_server()
    t21 = task_21(sample_sets["false_iz_100"], ground_truth)
    all_results["tasks"]["task_21"] = {
        "description": "Directed vs free zoom (2B, 100 false IZ)",
        "condition_a_accuracy": t21["condition_a_free"]["accuracy"],
        "condition_b_accuracy": t21["condition_b_directed"]["accuracy"],
        "accuracy_delta": t21["accuracy_delta"],
    }
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Task 22: Crop size comparison (2B)
    # ------------------------------------------------------------------
    ensure_2b_server()
    t22 = task_22(sample_sets["false_iz_100"], ground_truth)
    all_results["tasks"]["task_22"] = {
        "description": "Crop size comparison (2B, 100 false IZ, 64/128/256)",
        "crop_64_accuracy": t22["crop_64"]["accuracy"],
        "crop_128_accuracy": t22["crop_128"]["accuracy"],
        "crop_256_accuracy": t22["crop_256"]["accuracy"],
    }
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Task 23: Zoom as confidence signal (2B)
    # ------------------------------------------------------------------
    ensure_2b_server()
    t23 = task_23(sample_sets["mixed_100"], ground_truth)
    all_results["tasks"]["task_23"] = {
        "description": "Zoom as confidence signal (2B, 100 mixed)",
        "accuracy": t23["accuracy"],
        "confidence_stats": t23["confidence_stats"],
    }
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Task 24: Zoom on non-incident errors (2B)
    # ------------------------------------------------------------------
    ensure_2b_server()
    t24 = task_24(sample_sets["other_err_100"], ground_truth)
    all_results["tasks"]["task_24"] = {
        "description": "Zoom on non-IZ errors (2B, 100 samples)",
        "accuracy": t24["accuracy"],
        "error_breakdown": t24["error_breakdown"],
    }
    _save_results(all_results)

    # ------------------------------------------------------------------
    # Task 25: Token cost analysis (aggregated)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 25: Token cost analysis")
    print("=" * 70)

    # Collect all result lists for token analysis
    all_result_lists = {
        "T16_zoom_2B": t16["results"],
    }
    if t17 is not None:
        all_result_lists["T17_zoom_8B"] = t17["results"]
    all_result_lists["T19_multi_zoom"] = t19["results"]
    all_result_lists["T20_harm_rate"] = t20["results"]
    all_result_lists["T21a_free_zoom"] = t21["condition_a_free"]["results"]
    all_result_lists["T21b_directed_zoom"] = t21["condition_b_directed"]["results"]
    all_result_lists["T22_crop64"] = t22["crop_64"]["results"]
    all_result_lists["T22_crop128"] = t22["crop_128"]["results"]
    all_result_lists["T22_crop256"] = t22["crop_256"]["results"]
    all_result_lists["T23_confidence"] = t23["results"]
    all_result_lists["T24_non_iz_errors"] = t24["results"]

    token_stats = compute_token_stats(all_result_lists)
    all_results["tasks"]["task_25"] = {
        "description": "Token cost analysis",
        "per_task_stats": token_stats,
    }

    print(f"  {'Task':<25} {'Samples':>7} {'Tokens':>8} {'Tok/Sample':>10} "
          f"{'Tok/Zoom':>9} {'Zooms':>6} {'Accuracy':>8} {'Delta':>7}")
    print("  " + "-" * 85)
    for task_name, ts in token_stats.items():
        print(f"  {task_name:<25} {ts['total_samples']:>7} {ts['total_approx_tokens']:>8} "
              f"{ts['avg_tokens_per_sample']:>10.1f} {ts['avg_tokens_per_zoom']:>9.1f} "
              f"{ts['total_zooms']:>6} {ts['accuracy']:>8.4f} {ts['accuracy_delta']:>+7.4f}")

    _save_results(all_results)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Results saved to: {RESULTS_PATH}")
    print()

    print(f"  {'Task':>5} {'Description':<50} {'Accuracy':>8} {'Baseline':>8} {'Delta':>7}")
    print("  " + "-" * 85)

    for task_key, task_data in sorted(all_results["tasks"].items()):
        desc = task_data.get("description", "")[:50]
        acc = task_data.get("accuracy", {})
        if isinstance(acc, dict):
            accuracy = acc.get("accuracy", "N/A")
            baseline = acc.get("baseline_accuracy", "N/A")
            delta = acc.get("accuracy_delta", "N/A")
            if isinstance(accuracy, (int, float)):
                print(f"  {task_key:>5} {desc:<50} {accuracy:>8.4f} {baseline:>8.4f} {delta:>+7.4f}")
            else:
                print(f"  {task_key:>5} {desc:<50} {'N/A':>8} {'N/A':>8} {'N/A':>7}")
        else:
            print(f"  {task_key:>5} {desc:<50} {'N/A':>8} {'N/A':>8} {'N/A':>7}")

    # Kill servers
    print("\n--- Killing servers ---")
    kill_servers()
    print("Done.")


def _save_results(data: dict[str, Any]) -> None:
    """Save results to JSON (incremental saves)."""
    # Strip individual result lists (too large for summary) and save summary
    save_data = _deep_strip_results(data)
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  [saved to {RESULTS_PATH}]")


def _deep_strip_results(data: Any) -> Any:
    """Recursively strip large 'results' lists, keeping only first 5 for reference."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if k == "results" and isinstance(v, list) and len(v) > 5:
                out[k] = [_deep_strip_results(item) for item in v[:5]]
                out[f"{k}_count"] = len(v)
            elif k == "full_conversation":
                out[k] = "[stripped]"
            elif k == "zoom_details" and isinstance(v, list) and len(v) > 20:
                out[k] = v[:20]
                out[f"{k}_count"] = len(v)
            else:
                out[k] = _deep_strip_results(v)
        return out
    elif isinstance(data, list):
        return [_deep_strip_results(item) for item in data]
    return data


if __name__ == "__main__":
    main()
