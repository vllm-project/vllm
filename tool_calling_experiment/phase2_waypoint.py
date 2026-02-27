#!/usr/bin/env python3
# ruff: noqa: E501,E402
# type: ignore[import-not-found]
"""Phase 2 Waypoint Visualization Deep Dive: Tasks 26-35.

Tests whether base VLMs can reason about waypoint locations when
visualized on driving scene images.

Task 26: Self-correction via waypoint visualization (2B, 100 samples)
Task 27: Waypoint visualization for scene verification (2B, 100 false IZ)
Task 28: Predicted vs GT waypoint comparison (2B, 50 samples)
Task 29: Iterative waypoint refinement (2B, 100 samples, 3 iterations)
Task 30: Waypoint viz + zoom combined (2B, 100 samples)
Task 31: Off-road detection (2B, 100 deliberately bad waypoints)
Task 32: Waypoint spatial precision (2B, 100 samples, before/after viz)
Task 33: Waypoint viz across scene types (2B, 20 per scene x 5)
Task 34: 2B vs 8B waypoint reasoning (100 samples, both models)
Task 35: Token cost analysis

Usage:
    python tool_calling_experiment/phase2_waypoint.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any

import numpy as np
import requests

# Ensure sibling modules are importable
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from orchestrator import ToolCallingOrchestrator, parse_prediction  # noqa: I001
from visual_tools import (
    TOOL_ZOOM,
    TOOL_WAYPOINT_VIZ,
    load_sample_image,
    load_sample_metadata,
    visualize_waypoint,
    zoom_region,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(_DIR, "phase2_waypoint_results.json")
SC_DB_PATH = os.path.join(
    os.path.dirname(_DIR), "self_consistency_experiment", "self_consistency.db"
)
DATASET_PATH = "/workspace/vllm/models/dataset"

BASE_2B_MODEL = "/fsx/models/Qwen3-VL-2B-Instruct"
BASE_8B_MODEL = "/fsx/models/Qwen3-VL-8B-Instruct"

PORT_2B = 8332
PORT_8B = 8333
GPU_2B = 2
GPU_8B = 3

IMG_W = 504
IMG_H = 336
GRID_SIZE = 63

SYSTEM_PROMPT = (
    "You are a driving scene analyst. "
    "The image is 504x336 pixels. The waypoint grid is 63x63."
)

VALID_SCENES = frozenset(
    ["nominal", "flooded", "incident_zone", "mounted_police", "flagger"]
)

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def start_vllm_server(model_path: str, port: int, gpu_id: int) -> subprocess.Popen:
    """Start a vLLM server process with tool-calling support."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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

    print(f"  Starting server: model={os.path.basename(model_path)}, GPU={gpu_id}, port={port}")
    proc = subprocess.Popen(
        cmd, env=env, cwd="/tmp",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
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


def kill_servers() -> None:
    """Kill all phase2 vllm servers on ports 8332/8333."""
    subprocess.run(["pkill", "-f", "vllm serve.*833[23]"], capture_output=True)
    time.sleep(2)


# ---------------------------------------------------------------------------
# Waypoint conversion utilities
# ---------------------------------------------------------------------------

def ego_waypoint_to_grid(
    wp_x: float, wp_y: float,
    cam_intrinsics: list[float],
    cam_extrinsics: list[float],
    img_w: int = IMG_W,
    img_h: int = IMG_H,
    grid_size: int = GRID_SIZE,
) -> tuple[int, int] | None:
    """Project an ego-frame waypoint (meters) to a grid row,col using
    the camera calibration.

    Returns (row, col) in 0..grid_size-1, or None if behind camera.
    """
    # Build 4x4 extrinsic matrix (camera_from_ego)
    E = np.array(cam_extrinsics).reshape(4, 4)
    # Build 4x4 intrinsic matrix (pixel_from_camera)
    K = np.array(cam_intrinsics).reshape(4, 4)

    # Waypoint in ego frame: (x_forward, y_left, z_up=0, 1)
    pt_ego = np.array([wp_x, wp_y, 0.0, 1.0])

    # Transform to camera frame
    pt_cam = E @ pt_ego

    # Check if point is in front of camera
    if pt_cam[2] <= 0:
        return None

    # Project to pixel
    pt_proj = K @ pt_cam
    u = pt_proj[0] / pt_proj[2]
    v = pt_proj[1] / pt_proj[2]

    # The camera images are 2880x1860 originally, downscaled to 504x336
    # Scale factor
    orig_w, orig_h = 2880, 1860
    scale_x = img_w / orig_w
    scale_y = img_h / orig_h
    px = u * scale_x
    py = v * scale_y

    # Check bounds
    if px < 0 or px >= img_w or py < 0 or py >= img_h:
        return None

    # Convert to grid
    bin_w = img_w / grid_size
    bin_h = img_h / grid_size
    col = int(px / bin_w)
    row = int(py / bin_h)
    col = max(0, min(col, grid_size - 1))
    row = max(0, min(row, grid_size - 1))

    return (row, col)


def get_gt_grid_waypoint(sample_index: int) -> tuple[int, int] | None:
    """Get the ground truth waypoint as grid (row, col) for a sample.

    Uses the first GT waypoint (1 second ahead) and projects it
    through the camera calibration.
    """
    try:
        meta = load_sample_metadata(sample_index)
    except Exception:
        return None

    gt_wps = meta.get("ground_truth_waypoints", [])
    if not gt_wps:
        return None

    # Use the 1st waypoint (1s ahead)
    wp = gt_wps[0]
    wp_x, wp_y = wp[0], wp[1]

    # Get camera calibration for front_right_front_wide (image_0003)
    cals = meta.get("camera_calibration", [])
    cam_cal = None
    for cal in cals:
        if cal.get("camera_name") == "front_right_front_wide":
            cam_cal = cal
            break

    if cam_cal is None:
        return None

    return ego_waypoint_to_grid(
        wp_x, wp_y,
        cam_cal["intrinsics"],
        cam_cal["extrinsics"],
    )


def simulate_predicted_waypoint(
    gt_row: int, gt_col: int, noise_std: float = 8.0
) -> tuple[int, int]:
    """Simulate a 'predicted' waypoint with Gaussian noise from GT."""
    row = int(gt_row + np.random.normal(0, noise_std))
    col = int(gt_col + np.random.normal(0, noise_std))
    row = max(0, min(row, GRID_SIZE - 1))
    col = max(0, min(col, GRID_SIZE - 1))
    return (row, col)


def waypoint_l2(r1: int, c1: int, r2: int, c2: int) -> float:
    """L2 distance between two grid waypoints."""
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_db_samples() -> list[dict[str, Any]]:
    """Load all unique samples from the self-consistency DB."""
    conn = sqlite3.connect(f"file:{SC_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT sample_id, predicted_scene, predicted_long_action, "
        "predicted_lat_action, scene_type_gt, long_action_gt, lat_action_gt, "
        "fine_class FROM predictions WHERE sample_index = 0"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_false_iz_samples(n: int = 100) -> list[dict[str, Any]]:
    """Get samples predicted as incident_zone but GT is not incident_zone."""
    conn = sqlite3.connect(f"file:{SC_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT sample_id, predicted_scene, scene_type_gt, "
        "long_action_gt, lat_action_gt, fine_class "
        "FROM predictions "
        "WHERE predicted_scene = 'incident_zone' AND scene_type_gt != 'incident_zone' "
        "AND sample_index = 0 "
        "ORDER BY sample_id "
        f"LIMIT {n}"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_samples_by_scene(scene_type: str, n: int = 20) -> list[dict[str, Any]]:
    """Get n samples of a given scene type."""
    conn = sqlite3.connect(f"file:{SC_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT sample_id, predicted_scene, scene_type_gt, "
        "long_action_gt, lat_action_gt, fine_class "
        "FROM predictions "
        f"WHERE scene_type_gt = ? AND sample_index = 0 "
        "ORDER BY sample_id "
        f"LIMIT {n}",
        (scene_type,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_wrong_waypoint_samples(n: int = 100) -> list[dict[str, Any]]:
    """Get samples where the prediction was wrong (proxy for bad waypoints)."""
    conn = sqlite3.connect(f"file:{SC_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT sample_id, predicted_scene, scene_type_gt, "
        "long_action_gt, lat_action_gt, fine_class "
        "FROM predictions "
        "WHERE predicted_scene != scene_type_gt AND sample_index = 0 "
        "ORDER BY sample_id "
        f"LIMIT {n}"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Waypoint parsing from model output
# ---------------------------------------------------------------------------

_WAYPOINT_RE = re.compile(
    r"(?:row\s*[=:]\s*(\d+)\s*[,;]?\s*col(?:umn)?\s*[=:]\s*(\d+))"
    r"|(?:(\d+)\s*[,]\s*(\d+))",
    re.IGNORECASE,
)

_ROW_COL_RE = re.compile(
    r"(?:row|r)\s*[=:]\s*(\d+)\s*[,;/\s]+\s*(?:col|c|column)\s*[=:]\s*(\d+)",
    re.IGNORECASE,
)

_PAIR_RE = re.compile(r"\b(\d{1,2})\s*[,]\s*(\d{1,2})\b")


def parse_waypoint_from_text(text: str) -> tuple[int, int] | None:
    """Parse a row,col waypoint from model output text."""
    if not text:
        return None

    # Try row=X, col=Y format first
    m = _ROW_COL_RE.search(text)
    if m:
        row, col = int(m.group(1)), int(m.group(2))
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return (row, col)

    # Try generic pair format (N,N) where both < 63
    for m in _PAIR_RE.finditer(text):
        r, c = int(m.group(1)), int(m.group(2))
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return (r, c)

    return None


# ---------------------------------------------------------------------------
# Simple model call without orchestrator (for non-tool tasks)
# ---------------------------------------------------------------------------

def simple_chat(
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    image_path: str | None = None,
    temperature: float = 0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Make a single chat completion call (no tools)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if image_path:
        with open(image_path, "rb") as f:
            import base64
            b64 = base64.b64encode(f.read()).decode()
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    # Discover model name
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        model_name = r.json()["data"][0]["id"]
    except Exception:
        model_name = "default"

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    start = time.monotonic()
    resp = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=120
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return {
        "content": msg.get("content", ""),
        "usage": usage,
        "latency_ms": round(elapsed_ms, 1),
    }


def chat_with_image(
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[str],
    temperature: float = 0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Chat with one or more images."""
    import base64 as b64_mod
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = [{"type": "text", "text": user_prompt}]
    for ip in image_paths:
        with open(ip, "rb") as f:
            b64 = b64_mod.b64encode(f.read()).decode()
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    messages.append({"role": "user", "content": user_content})

    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        model_name = r.json()["data"][0]["id"]
    except Exception:
        model_name = "default"

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    start = time.monotonic()
    resp = requests.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=120
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return {
        "content": msg.get("content", ""),
        "usage": usage,
        "latency_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_sample_with_waypoints(
    sample_id: int, noise_std: float = 8.0
) -> dict[str, Any] | None:
    """Prepare a sample with GT and simulated predicted waypoint.

    Returns dict with sample_id, gt_row, gt_col, pred_row, pred_col,
    image_path, scene_type_gt, long_action_gt, or None if unavailable.
    """
    gt = get_gt_grid_waypoint(sample_id)
    if gt is None:
        return None

    gt_row, gt_col = gt
    pred_row, pred_col = simulate_predicted_waypoint(gt_row, gt_col, noise_std)

    try:
        img_path = load_sample_image(sample_id)
    except Exception:
        return None

    try:
        meta = load_sample_metadata(sample_id)
    except Exception:
        meta = {}

    fine_class = meta.get("fine_class", "unknown")
    scene_map = {
        "nominal": "nominal",
        "flooded": "flooded",
        "incident": "incident_zone",
        "police": "mounted_police",
        "horse": "mounted_police",
        "mounted": "mounted_police",
        "flagger": "flagger",
    }
    scene_type = "nominal"
    for key, val in scene_map.items():
        if key in fine_class.lower():
            scene_type = val
            break

    return {
        "sample_id": sample_id,
        "gt_row": gt_row,
        "gt_col": gt_col,
        "pred_row": pred_row,
        "pred_col": pred_col,
        "image_path": img_path,
        "scene_type_gt": scene_type,
        "long_action_gt": meta.get("long_action", "null"),
        "fine_class": fine_class,
    }


def collect_samples_with_waypoints(
    sample_ids: list[int], noise_std: float = 8.0, max_samples: int = 100
) -> list[dict[str, Any]]:
    """Collect up to max_samples samples with valid GT waypoints."""
    samples = []
    for sid in sample_ids:
        if len(samples) >= max_samples:
            break
        s = prepare_sample_with_waypoints(sid, noise_std)
        if s is not None:
            samples.append(s)
    return samples


# ---------------------------------------------------------------------------
# Task 26: Self-correction via waypoint visualization
# ---------------------------------------------------------------------------

def run_task26(base_url: str) -> dict[str, Any]:
    """Self-correction via waypoint visualization.

    Model sees image WITH predicted waypoint drawn. Asked if it makes sense
    and to suggest correction. Measure if correction improves distance to GT.
    """
    print("\n" + "=" * 70)
    print("TASK 26: Self-correction via waypoint visualization (2B, 100 samples)")
    print("=" * 70)

    wrong_samples = get_wrong_waypoint_samples(300)
    sample_ids = [s["sample_id"] for s in wrong_samples]
    samples = collect_samples_with_waypoints(sample_ids, noise_std=10.0, max_samples=100)
    print(f"  Prepared {len(samples)} samples with waypoints")

    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, s in enumerate(samples):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(samples)}...")

        # Draw predicted waypoint on image
        viz = visualize_waypoint(
            s["image_path"], s["pred_row"], s["pred_col"], label="predicted"
        )
        annotated_path = viz["annotated_image"]

        prompt = (
            f"A driving model predicted this vehicle should drive to the red marker. "
            f"Does this location make sense for a {s['scene_type_gt']} scene with "
            f"{s['long_action_gt']} action? If not, suggest a better waypoint location "
            f"as row,col in a 63x63 grid."
        )

        try:
            resp = simple_chat(
                base_url, SYSTEM_PROMPT, prompt,
                image_path=annotated_path, max_tokens=512,
            )
        except Exception as e:
            results.append({
                "sample_id": s["sample_id"],
                "error": str(e),
            })
            continue

        content = resp["content"]
        usage = resp.get("usage", {})
        for k in total_tokens:
            total_tokens[k] += usage.get(k, 0)

        # Parse suggested waypoint
        suggested = parse_waypoint_from_text(content)
        original_dist = waypoint_l2(s["pred_row"], s["pred_col"], s["gt_row"], s["gt_col"])

        if suggested:
            corrected_dist = waypoint_l2(suggested[0], suggested[1], s["gt_row"], s["gt_col"])
            improvement = original_dist - corrected_dist
        else:
            corrected_dist = None
            improvement = None

        results.append({
            "sample_id": s["sample_id"],
            "gt_row": s["gt_row"],
            "gt_col": s["gt_col"],
            "pred_row": s["pred_row"],
            "pred_col": s["pred_col"],
            "original_dist": round(original_dist, 2),
            "suggested_waypoint": suggested,
            "corrected_dist": round(corrected_dist, 2) if corrected_dist is not None else None,
            "improvement": round(improvement, 2) if improvement is not None else None,
            "improved": improvement is not None and improvement > 0,
            "response_snippet": content[:300],
            "latency_ms": resp["latency_ms"],
        })

    # Aggregate
    n = len(results)
    n_suggested = sum(1 for r in results if r.get("suggested_waypoint") is not None)
    n_improved = sum(1 for r in results if r.get("improved"))
    improvements = [r["improvement"] for r in results if r.get("improvement") is not None]
    original_dists = [r["original_dist"] for r in results if r.get("original_dist") is not None and "error" not in r]
    corrected_dists = [r["corrected_dist"] for r in results if r.get("corrected_dist") is not None]

    summary = {
        "task": "task26_self_correction",
        "total_samples": n,
        "n_suggested_correction": n_suggested,
        "suggestion_rate": round(n_suggested / n, 3) if n > 0 else 0,
        "n_improved": n_improved,
        "improvement_rate": round(n_improved / n_suggested, 3) if n_suggested > 0 else 0,
        "avg_original_dist": round(float(np.mean(original_dists)), 2) if original_dists else None,
        "avg_corrected_dist": round(float(np.mean(corrected_dists)), 2) if corrected_dists else None,
        "avg_improvement": round(float(np.mean(improvements)), 2) if improvements else None,
        "median_improvement": round(float(np.median(improvements)), 2) if improvements else None,
        "token_usage": total_tokens,
    }
    print(f"  TASK 26 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Task 27: Waypoint visualization for scene verification
# ---------------------------------------------------------------------------

def run_task27(base_url: str) -> dict[str, Any]:
    """Waypoint viz for scene verification on false incident_zone samples."""
    print("\n" + "=" * 70)
    print("TASK 27: Waypoint viz for scene verification (2B, 100 false IZ)")
    print("=" * 70)

    false_iz = get_false_iz_samples(200)
    print(f"  Found {len(false_iz)} false IZ samples in DB")

    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    count = 0
    for s in false_iz:
        if count >= 100:
            break

        sid = s["sample_id"]
        gt = get_gt_grid_waypoint(sid)
        if gt is None:
            continue

        gt_row, gt_col = gt
        # Use a waypoint with moderate noise (simulating fine-tuned model prediction)
        pred_row, pred_col = simulate_predicted_waypoint(gt_row, gt_col, noise_std=5.0)

        try:
            img_path = load_sample_image(sid)
        except Exception:
            continue

        viz = visualize_waypoint(img_path, pred_row, pred_col, label="predicted")
        annotated_path = viz["annotated_image"]

        prompt = (
            "A model predicted incident_zone and suggested driving to the red marker. "
            "Does the waypoint location match what you'd expect for an incident response? "
            "If the waypoint looks like normal driving, the scene might not actually be an incident. "
            "Classify: nominal, flooded, incident_zone, mounted_police, flagger."
        )

        try:
            resp = simple_chat(
                base_url, SYSTEM_PROMPT, prompt,
                image_path=annotated_path, max_tokens=512,
            )
        except Exception as e:
            results.append({"sample_id": sid, "error": str(e)})
            count += 1
            continue

        content = resp["content"]
        usage = resp.get("usage", {})
        for k in total_tokens:
            total_tokens[k] += usage.get(k, 0)

        # Parse predicted scene
        pred = parse_prediction(content)
        predicted_scene = pred.get("scene")

        gt_scene = s["scene_type_gt"]
        correct = predicted_scene == gt_scene if predicted_scene else False
        caught_misclass = (
            predicted_scene is not None
            and predicted_scene != "incident_zone"
            and gt_scene != "incident_zone"
        )

        results.append({
            "sample_id": sid,
            "gt_scene": gt_scene,
            "original_pred": "incident_zone",
            "corrected_pred": predicted_scene,
            "correct": correct,
            "caught_misclassification": caught_misclass,
            "response_snippet": content[:300],
            "latency_ms": resp["latency_ms"],
        })
        count += 1

        if count % 10 == 0:
            print(f"  Processing {count}/100...")

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    n_caught = sum(1 for r in results if r.get("caught_misclassification"))
    scene_dist = {}
    for r in results:
        s = r.get("corrected_pred") or "unparsed"
        scene_dist[s] = scene_dist.get(s, 0) + 1

    summary = {
        "task": "task27_scene_verification",
        "total_samples": n,
        "correct_after_correction": n_correct,
        "accuracy_after_correction": round(n_correct / n, 3) if n > 0 else 0,
        "caught_misclassification": n_caught,
        "catch_rate": round(n_caught / n, 3) if n > 0 else 0,
        "corrected_scene_distribution": scene_dist,
        "token_usage": total_tokens,
    }
    print(f"  TASK 27 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Task 28: Predicted vs GT waypoint comparison
# ---------------------------------------------------------------------------

def run_task28(base_url: str) -> dict[str, Any]:
    """Show TWO annotated images (Model A vs Model B) and ask which is better."""
    print("\n" + "=" * 70)
    print("TASK 28: Predicted vs GT waypoint comparison (2B, 50 samples)")
    print("=" * 70)

    all_db = load_db_samples()
    sample_ids = [s["sample_id"] for s in all_db[:300]]
    prepared = collect_samples_with_waypoints(sample_ids, noise_std=10.0, max_samples=50)
    print(f"  Prepared {len(prepared)} samples")

    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, s in enumerate(prepared):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(prepared)}...")

        # Randomly assign GT and predicted to Model A/B
        gt_is_a = random.random() < 0.5
        if gt_is_a:
            row_a, col_a = s["gt_row"], s["gt_col"]
            row_b, col_b = s["pred_row"], s["pred_col"]
        else:
            row_a, col_a = s["pred_row"], s["pred_col"]
            row_b, col_b = s["gt_row"], s["gt_col"]

        viz_a = visualize_waypoint(s["image_path"], row_a, col_a, label="Model A")
        viz_b = visualize_waypoint(s["image_path"], row_b, col_b, label="Model B")

        prompt = (
            "Two possible driving waypoints are shown in separate images. "
            "Model A's waypoint is in the first image, Model B's waypoint is in the second. "
            "Which makes more physical sense for safe driving? "
            "Answer with 'Model A' or 'Model B' and explain briefly."
        )

        try:
            resp = chat_with_image(
                base_url, SYSTEM_PROMPT, prompt,
                image_paths=[viz_a["annotated_image"], viz_b["annotated_image"]],
                max_tokens=512,
            )
        except Exception as e:
            results.append({"sample_id": s["sample_id"], "error": str(e)})
            continue

        content = resp["content"]
        usage = resp.get("usage", {})
        for k in total_tokens:
            total_tokens[k] += usage.get(k, 0)

        # Determine which model was chosen
        chose_a = "model a" in content.lower() and "model b" not in content.lower().split("model a")[0]
        chose_b = "model b" in content.lower()

        if "model a" in content.lower() and "model b" in content.lower():
            # Both mentioned - look for preference patterns
            for pattern in ["better", "more sense", "prefer", "correct", "choose", "pick"]:
                p_idx = content.lower().rfind(pattern)
                if p_idx >= 0:
                    # Which model is nearest before this pattern?
                    dist_a = p_idx - content.lower().rfind("model a", 0, p_idx)
                    dist_b = p_idx - content.lower().rfind("model b", 0, p_idx)
                    if dist_a < dist_b and dist_a < 30:
                        chose_a = True
                        chose_b = False
                    elif dist_b < dist_a and dist_b < 30:
                        chose_b = True
                        chose_a = False
                    break

        chose_gt = (gt_is_a and chose_a) or (not gt_is_a and chose_b)
        chose_pred = (not gt_is_a and chose_a) or (gt_is_a and chose_b)

        results.append({
            "sample_id": s["sample_id"],
            "gt_is_model_a": gt_is_a,
            "chose_a": chose_a,
            "chose_b": chose_b,
            "chose_gt": chose_gt,
            "chose_pred": chose_pred,
            "pred_dist_to_gt": round(waypoint_l2(s["pred_row"], s["pred_col"], s["gt_row"], s["gt_col"]), 2),
            "response_snippet": content[:300],
            "latency_ms": resp["latency_ms"],
        })

    n = len(results)
    n_chose_gt = sum(1 for r in results if r.get("chose_gt"))
    n_chose_pred = sum(1 for r in results if r.get("chose_pred"))
    n_unclear = n - n_chose_gt - n_chose_pred

    summary = {
        "task": "task28_waypoint_comparison",
        "total_samples": n,
        "chose_gt_waypoint": n_chose_gt,
        "gt_preference_rate": round(n_chose_gt / n, 3) if n > 0 else 0,
        "chose_predicted": n_chose_pred,
        "unclear_choice": n_unclear,
        "token_usage": total_tokens,
    }
    print(f"  TASK 28 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Task 29: Iterative waypoint refinement
# ---------------------------------------------------------------------------

def run_task29(base_url: str) -> dict[str, Any]:
    """Iterative refinement: predict -> visualize -> refine x3."""
    print("\n" + "=" * 70)
    print("TASK 29: Iterative waypoint refinement (2B, 100 samples, 3 iterations)")
    print("=" * 70)

    all_db = load_db_samples()
    sample_ids = [s["sample_id"] for s in all_db[:500]]
    prepared = collect_samples_with_waypoints(sample_ids, noise_std=0, max_samples=100)
    print(f"  Prepared {len(prepared)} samples")

    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, s in enumerate(prepared):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(prepared)}...")

        iterations = []
        current_wp = None

        for iteration in range(3):
            if iteration == 0:
                prompt = (
                    "Look at this driving scene. Predict where the vehicle should drive to "
                    "as a waypoint in a 63x63 grid. Output as row,col."
                )
                img_path = s["image_path"]
            else:
                if current_wp is None:
                    break
                # Visualize current prediction
                viz = visualize_waypoint(
                    s["image_path"], current_wp[0], current_wp[1],
                    label=f"iter{iteration}"
                )
                img_path = viz["annotated_image"]
                prompt = (
                    f"Your previous prediction was row={current_wp[0]}, col={current_wp[1]} "
                    f"(shown as the red marker). Does this look correct for safe driving? "
                    f"If not, refine the waypoint. Output as row,col."
                )

            try:
                resp = simple_chat(
                    base_url, SYSTEM_PROMPT, prompt,
                    image_path=img_path, max_tokens=512,
                )
            except Exception as e:
                iterations.append({"iteration": iteration, "error": str(e)})
                break

            content = resp["content"]
            usage = resp.get("usage", {})
            for k in total_tokens:
                total_tokens[k] += usage.get(k, 0)

            new_wp = parse_waypoint_from_text(content)
            dist = None
            if new_wp:
                dist = waypoint_l2(new_wp[0], new_wp[1], s["gt_row"], s["gt_col"])
                current_wp = new_wp

            iterations.append({
                "iteration": iteration,
                "predicted_wp": new_wp,
                "dist_to_gt": round(dist, 2) if dist is not None else None,
                "response_snippet": content[:200],
            })

        # Track convergence
        dists = [it["dist_to_gt"] for it in iterations if it.get("dist_to_gt") is not None]
        improved_over_iterations = False
        oscillating = False
        if len(dists) >= 2:
            improved_over_iterations = dists[-1] < dists[0]
            if len(dists) >= 3:
                # Check for oscillation: dist goes down then up or up then down
                diffs = [dists[j+1] - dists[j] for j in range(len(dists)-1)]
                oscillating = any(d > 0 for d in diffs) and any(d < 0 for d in diffs)

        results.append({
            "sample_id": s["sample_id"],
            "gt_row": s["gt_row"],
            "gt_col": s["gt_col"],
            "iterations": iterations,
            "distances_by_iter": dists,
            "improved_over_iterations": improved_over_iterations,
            "oscillating": oscillating,
            "n_valid_predictions": len(dists),
        })

    n = len(results)
    n_improved = sum(1 for r in results if r.get("improved_over_iterations"))
    n_oscillating = sum(1 for r in results if r.get("oscillating"))

    # Average distance by iteration
    iter_dists = {0: [], 1: [], 2: []}
    for r in results:
        for it in r.get("iterations", []):
            idx = it.get("iteration", 0)
            d = it.get("dist_to_gt")
            if d is not None and idx in iter_dists:
                iter_dists[idx].append(d)

    avg_dist_by_iter = {}
    for k, v in iter_dists.items():
        avg_dist_by_iter[f"iter_{k}"] = round(float(np.mean(v)), 2) if v else None

    summary = {
        "task": "task29_iterative_refinement",
        "total_samples": n,
        "n_improved": n_improved,
        "improvement_rate": round(n_improved / n, 3) if n > 0 else 0,
        "n_oscillating": n_oscillating,
        "oscillation_rate": round(n_oscillating / n, 3) if n > 0 else 0,
        "avg_dist_by_iteration": avg_dist_by_iter,
        "token_usage": total_tokens,
    }
    print(f"  TASK 29 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Task 30: Waypoint viz + zoom combined
# ---------------------------------------------------------------------------

def run_task30(base_url: str) -> dict[str, Any]:
    """Both visualize_waypoint and zoom_region tools available."""
    print("\n" + "=" * 70)
    print("TASK 30: Waypoint viz + zoom combined (2B, 100 samples)")
    print("=" * 70)

    all_db = load_db_samples()
    sample_ids = [s["sample_id"] for s in all_db[:500]]
    prepared = collect_samples_with_waypoints(sample_ids, noise_std=0, max_samples=100)
    print(f"  Prepared {len(prepared)} samples")

    orch = ToolCallingOrchestrator(
        server_url=base_url,
        tools={
            "zoom_region": zoom_region,
            "visualize_waypoint": visualize_waypoint,
        },
        tool_definitions=[TOOL_ZOOM, TOOL_WAYPOINT_VIZ],
        max_tool_rounds=5,
        temperature=0,
        max_tokens=1024,
    )

    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, s in enumerate(prepared):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(prepared)}...")

        prompt = (
            "Predict a safe driving waypoint for this scene as row,col in a 63x63 grid. "
            "Visualize it to check, and zoom in to the target area if needed."
        )

        try:
            r = orch.run(
                image_path=s["image_path"],
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                tool_choice="auto",
            )
        except Exception as e:
            results.append({"sample_id": s["sample_id"], "error": str(e)})
            continue

        # Track token usage from tool calls
        # (orchestrator doesn't expose usage directly, estimate from content)

        final_text = r.get("final_text", "")
        all_text = r.get("reasoning_text", "")
        predicted_wp = parse_waypoint_from_text(all_text)
        if predicted_wp is None:
            predicted_wp = parse_waypoint_from_text(final_text)

        dist = None
        if predicted_wp:
            dist = waypoint_l2(predicted_wp[0], predicted_wp[1], s["gt_row"], s["gt_col"])

        tools_used = [tc["tool_name"] for tc in r.get("tool_calls", [])]
        used_viz = "visualize_waypoint" in tools_used
        used_zoom = "zoom_region" in tools_used
        used_both = used_viz and used_zoom

        results.append({
            "sample_id": s["sample_id"],
            "gt_row": s["gt_row"],
            "gt_col": s["gt_col"],
            "predicted_wp": predicted_wp,
            "dist_to_gt": round(dist, 2) if dist is not None else None,
            "tools_used": tools_used,
            "used_viz": used_viz,
            "used_zoom": used_zoom,
            "used_both": used_both,
            "num_tool_calls": r["num_tool_calls"],
            "num_rounds": r["num_rounds"],
            "latency_ms": r["latency_ms"],
        })

    n = len(results)
    n_predicted = sum(1 for r in results if r.get("predicted_wp") is not None)
    dists = [r["dist_to_gt"] for r in results if r.get("dist_to_gt") is not None]
    n_used_viz = sum(1 for r in results if r.get("used_viz"))
    n_used_zoom = sum(1 for r in results if r.get("used_zoom"))
    n_used_both = sum(1 for r in results if r.get("used_both"))

    summary = {
        "task": "task30_viz_plus_zoom",
        "total_samples": n,
        "n_predicted_waypoint": n_predicted,
        "prediction_rate": round(n_predicted / n, 3) if n > 0 else 0,
        "avg_dist_to_gt": round(float(np.mean(dists)), 2) if dists else None,
        "median_dist_to_gt": round(float(np.median(dists)), 2) if dists else None,
        "used_viz": n_used_viz,
        "used_zoom": n_used_zoom,
        "used_both_tools": n_used_both,
        "token_usage": total_tokens,
    }
    print(f"  TASK 30 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Task 31: Off-road detection
# ---------------------------------------------------------------------------

def run_task31(base_url: str) -> dict[str, Any]:
    """Off-road detection with deliberately bad waypoints."""
    print("\n" + "=" * 70)
    print("TASK 31: Off-road detection (2B, 100 bad waypoints)")
    print("=" * 70)

    all_db = load_db_samples()

    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    count = 0
    for s in all_db:
        if count >= 100:
            break

        sid = s["sample_id"]
        try:
            img_path = load_sample_image(sid)
        except Exception:
            continue

        # Generate random off-road waypoint (corners of grid)
        corner = random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
        if corner == "top-left":
            r = random.randint(0, 10)
            c = random.randint(0, 10)
        elif corner == "top-right":
            r = random.randint(0, 10)
            c = random.randint(53, 62)
        elif corner == "bottom-left":
            r = random.randint(53, 62)
            c = random.randint(0, 10)
        else:
            r = random.randint(53, 62)
            c = random.randint(53, 62)

        # The GT: is this actually off-road? Corners are very likely off-road
        is_offroad_gt = True  # By design, corners are almost always off-road

        viz = visualize_waypoint(img_path, r, c, label="suggested")
        annotated_path = viz["annotated_image"]

        prompt = (
            f"A waypoint at row={r}, col={c} is suggested for driving. "
            f"Look at the red marker in the image. "
            f"Is this on the drivable road? Answer YES or NO, then explain briefly."
        )

        try:
            resp = simple_chat(
                base_url, SYSTEM_PROMPT, prompt,
                image_path=annotated_path, max_tokens=256,
            )
        except Exception as e:
            results.append({"sample_id": sid, "error": str(e)})
            count += 1
            continue

        content = resp["content"]
        usage = resp.get("usage", {})
        for k in total_tokens:
            total_tokens[k] += usage.get(k, 0)

        # Parse YES/NO
        lower = content.lower()
        said_no = "no" in lower.split()[:5] or lower.strip().startswith("no")
        said_yes = "yes" in lower.split()[:5] or lower.strip().startswith("yes")

        detected_offroad = said_no and not said_yes
        correct = detected_offroad == is_offroad_gt

        results.append({
            "sample_id": sid,
            "waypoint_row": r,
            "waypoint_col": c,
            "corner": corner,
            "is_offroad_gt": is_offroad_gt,
            "detected_offroad": detected_offroad,
            "correct": correct,
            "said_yes": said_yes,
            "said_no": said_no,
            "response_snippet": content[:200],
            "latency_ms": resp["latency_ms"],
        })
        count += 1

        if count % 10 == 0:
            print(f"  Processing {count}/100...")

    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    n_detected = sum(1 for r in results if r.get("detected_offroad"))

    # Accuracy by corner
    corner_acc = {}
    for corner in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        corner_results = [r for r in results if r.get("corner") == corner]
        if corner_results:
            acc = sum(1 for r in corner_results if r.get("correct")) / len(corner_results)
            corner_acc[corner] = round(acc, 3)

    summary = {
        "task": "task31_offroad_detection",
        "total_samples": n,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n, 3) if n > 0 else 0,
        "n_detected_offroad": n_detected,
        "detection_rate": round(n_detected / n, 3) if n > 0 else 0,
        "accuracy_by_corner": corner_acc,
        "token_usage": total_tokens,
    }
    print(f"  TASK 31 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Task 32: Waypoint spatial precision (before vs after viz)
# ---------------------------------------------------------------------------

def run_task32(base_url: str) -> dict[str, Any]:
    """Condition A: predict without viz. Condition B: predict, viz, refine."""
    print("\n" + "=" * 70)
    print("TASK 32: Waypoint spatial precision (2B, 100 samples)")
    print("=" * 70)

    all_db = load_db_samples()
    sample_ids = [s["sample_id"] for s in all_db[:500]]
    prepared = collect_samples_with_waypoints(sample_ids, noise_std=0, max_samples=100)
    print(f"  Prepared {len(prepared)} samples")

    results = []
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, s in enumerate(prepared):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(prepared)}...")

        # Condition A: No visualization
        prompt_a = (
            "Look at this driving scene. Predict where the vehicle should drive to "
            "as a waypoint in a 63x63 grid. Output as row,col."
        )
        try:
            resp_a = simple_chat(
                base_url, SYSTEM_PROMPT, prompt_a,
                image_path=s["image_path"], max_tokens=512,
            )
        except Exception as e:
            results.append({"sample_id": s["sample_id"], "error": str(e)})
            continue

        content_a = resp_a["content"]
        usage_a = resp_a.get("usage", {})
        for k in total_tokens:
            total_tokens[k] += usage_a.get(k, 0)

        wp_a = parse_waypoint_from_text(content_a)
        dist_a = waypoint_l2(wp_a[0], wp_a[1], s["gt_row"], s["gt_col"]) if wp_a else None

        # Condition B: Predict, visualize, refine
        if wp_a:
            viz = visualize_waypoint(
                s["image_path"], wp_a[0], wp_a[1], label="initial"
            )
            prompt_b = (
                f"You predicted a waypoint at row={wp_a[0]}, col={wp_a[1]} "
                f"(shown as the red marker). Does this look correct? "
                f"Refine if needed. Output final waypoint as row,col."
            )
            try:
                resp_b = simple_chat(
                    base_url, SYSTEM_PROMPT, prompt_b,
                    image_path=viz["annotated_image"], max_tokens=512,
                )
            except Exception as e:
                results.append({
                    "sample_id": s["sample_id"],
                    "dist_before_viz": round(dist_a, 2) if dist_a else None,
                    "error_condition_b": str(e),
                })
                continue

            content_b = resp_b["content"]
            usage_b = resp_b.get("usage", {})
            for k in total_tokens:
                total_tokens[k] += usage_b.get(k, 0)

            wp_b = parse_waypoint_from_text(content_b)
            dist_b = waypoint_l2(wp_b[0], wp_b[1], s["gt_row"], s["gt_col"]) if wp_b else None
        else:
            wp_b = None
            dist_b = None

        improved = (
            dist_a is not None
            and dist_b is not None
            and dist_b < dist_a
        )

        results.append({
            "sample_id": s["sample_id"],
            "gt_row": s["gt_row"],
            "gt_col": s["gt_col"],
            "wp_before": wp_a,
            "dist_before_viz": round(dist_a, 2) if dist_a is not None else None,
            "wp_after": wp_b,
            "dist_after_viz": round(dist_b, 2) if dist_b is not None else None,
            "improved": improved,
        })

    n = len(results)
    dists_before = [r["dist_before_viz"] for r in results if r.get("dist_before_viz") is not None]
    dists_after = [r["dist_after_viz"] for r in results if r.get("dist_after_viz") is not None]
    n_improved = sum(1 for r in results if r.get("improved"))
    n_both = sum(1 for r in results if r.get("dist_before_viz") is not None and r.get("dist_after_viz") is not None)

    summary = {
        "task": "task32_spatial_precision",
        "total_samples": n,
        "avg_dist_before_viz": round(float(np.mean(dists_before)), 2) if dists_before else None,
        "avg_dist_after_viz": round(float(np.mean(dists_after)), 2) if dists_after else None,
        "median_dist_before": round(float(np.median(dists_before)), 2) if dists_before else None,
        "median_dist_after": round(float(np.median(dists_after)), 2) if dists_after else None,
        "n_improved": n_improved,
        "improvement_rate": round(n_improved / n_both, 3) if n_both > 0 else 0,
        "n_both_valid": n_both,
        "token_usage": total_tokens,
    }
    print(f"  TASK 32 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Task 33: Waypoint viz across scene types
# ---------------------------------------------------------------------------

def run_task33(base_url: str) -> dict[str, Any]:
    """Does waypoint viz help equally across scene types? 20 per scene x 5."""
    print("\n" + "=" * 70)
    print("TASK 33: Waypoint viz across scene types (2B, 20 x 5)")
    print("=" * 70)

    scene_types = ["nominal", "flooded", "incident_zone", "mounted_police", "flagger"]

    all_results = {}
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for scene in scene_types:
        print(f"\n  Processing scene: {scene}")
        scene_samples = get_samples_by_scene(scene, n=80)
        sample_ids = [s["sample_id"] for s in scene_samples]
        prepared = collect_samples_with_waypoints(sample_ids, noise_std=0, max_samples=20)
        print(f"    Got {len(prepared)} samples for {scene}")

        scene_results = []
        for i, s in enumerate(prepared):
            # Condition A: no viz
            prompt_a = (
                "Predict a safe driving waypoint for this scene as row,col in a 63x63 grid."
            )
            try:
                resp_a = simple_chat(
                    base_url, SYSTEM_PROMPT, prompt_a,
                    image_path=s["image_path"], max_tokens=256,
                )
            except Exception:
                continue

            usage_a = resp_a.get("usage", {})
            for k in total_tokens:
                total_tokens[k] += usage_a.get(k, 0)

            wp_a = parse_waypoint_from_text(resp_a["content"])
            dist_a = waypoint_l2(wp_a[0], wp_a[1], s["gt_row"], s["gt_col"]) if wp_a else None

            # Condition B: with viz
            dist_b = None
            if wp_a:
                viz = visualize_waypoint(s["image_path"], wp_a[0], wp_a[1], label="initial")
                prompt_b = (
                    f"Your predicted waypoint is at row={wp_a[0]}, col={wp_a[1]} "
                    f"(red marker). Refine if needed. Output as row,col."
                )
                try:
                    resp_b = simple_chat(
                        base_url, SYSTEM_PROMPT, prompt_b,
                        image_path=viz["annotated_image"], max_tokens=256,
                    )
                    usage_b = resp_b.get("usage", {})
                    for k in total_tokens:
                        total_tokens[k] += usage_b.get(k, 0)

                    wp_b = parse_waypoint_from_text(resp_b["content"])
                    if wp_b:
                        dist_b = waypoint_l2(wp_b[0], wp_b[1], s["gt_row"], s["gt_col"])
                except Exception:
                    pass

            scene_results.append({
                "sample_id": s["sample_id"],
                "dist_before": round(dist_a, 2) if dist_a is not None else None,
                "dist_after": round(dist_b, 2) if dist_b is not None else None,
                "improved": dist_a is not None and dist_b is not None and dist_b < dist_a,
            })

        dists_b = [r["dist_before"] for r in scene_results if r.get("dist_before") is not None]
        dists_a = [r["dist_after"] for r in scene_results if r.get("dist_after") is not None]
        n_improved = sum(1 for r in scene_results if r.get("improved"))
        n_valid = sum(1 for r in scene_results if r.get("dist_before") is not None and r.get("dist_after") is not None)

        all_results[scene] = {
            "n_samples": len(scene_results),
            "avg_dist_before": round(float(np.mean(dists_b)), 2) if dists_b else None,
            "avg_dist_after": round(float(np.mean(dists_a)), 2) if dists_a else None,
            "n_improved": n_improved,
            "improvement_rate": round(n_improved / n_valid, 3) if n_valid > 0 else 0,
            "samples": scene_results,
        }

    summary = {
        "task": "task33_across_scene_types",
        "scene_summaries": {
            scene: {
                k: v for k, v in data.items() if k != "samples"
            }
            for scene, data in all_results.items()
        },
        "token_usage": total_tokens,
    }
    print(f"  TASK 33 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "by_scene": all_results}


# ---------------------------------------------------------------------------
# Task 34: 2B vs 8B waypoint reasoning
# ---------------------------------------------------------------------------

def run_task34(base_url_2b: str, base_url_8b: str) -> dict[str, Any]:
    """Same samples through both 2B and 8B models."""
    print("\n" + "=" * 70)
    print("TASK 34: 2B vs 8B waypoint reasoning (100 samples)")
    print("=" * 70)

    all_db = load_db_samples()
    sample_ids = [s["sample_id"] for s in all_db[:500]]
    prepared = collect_samples_with_waypoints(sample_ids, noise_std=0, max_samples=100)
    print(f"  Prepared {len(prepared)} samples")

    def run_model(model_url: str, model_name: str) -> list[dict]:
        model_results = []
        for i, s in enumerate(prepared):
            if i % 10 == 0:
                print(f"  [{model_name}] Processing {i}/{len(prepared)}...")

            # Step 1: predict waypoint
            prompt1 = "Predict a safe driving waypoint as row,col in a 63x63 grid."
            try:
                resp1 = simple_chat(
                    model_url, SYSTEM_PROMPT, prompt1,
                    image_path=s["image_path"], max_tokens=256,
                )
            except Exception as e:
                model_results.append({"sample_id": s["sample_id"], "error": str(e)})
                continue

            wp1 = parse_waypoint_from_text(resp1["content"])
            dist1 = waypoint_l2(wp1[0], wp1[1], s["gt_row"], s["gt_col"]) if wp1 else None

            # Step 2: visualize and refine
            dist2 = None
            wp2 = None
            if wp1:
                viz = visualize_waypoint(s["image_path"], wp1[0], wp1[1], label="predicted")
                prompt2 = (
                    f"Your waypoint at row={wp1[0]}, col={wp1[1]} is shown (red marker). "
                    f"Refine if needed. Output as row,col."
                )
                try:
                    resp2 = simple_chat(
                        model_url, SYSTEM_PROMPT, prompt2,
                        image_path=viz["annotated_image"], max_tokens=256,
                    )
                    wp2 = parse_waypoint_from_text(resp2["content"])
                    if wp2:
                        dist2 = waypoint_l2(wp2[0], wp2[1], s["gt_row"], s["gt_col"])
                except Exception:
                    pass

            model_results.append({
                "sample_id": s["sample_id"],
                "initial_wp": wp1,
                "initial_dist": round(dist1, 2) if dist1 is not None else None,
                "refined_wp": wp2,
                "refined_dist": round(dist2, 2) if dist2 is not None else None,
                "improved": dist1 is not None and dist2 is not None and dist2 < dist1,
                "usage_1": resp1.get("usage", {}),
            })
        return model_results

    results_2b = run_model(base_url_2b, "2B")
    results_8b = run_model(base_url_8b, "8B")

    def summarize_model(model_results: list, name: str) -> dict:
        n = len(model_results)
        initial_dists = [r["initial_dist"] for r in model_results if r.get("initial_dist") is not None]
        refined_dists = [r["refined_dist"] for r in model_results if r.get("refined_dist") is not None]
        n_improved = sum(1 for r in model_results if r.get("improved"))
        n_valid = sum(1 for r in model_results if r.get("initial_dist") is not None and r.get("refined_dist") is not None)
        n_predicted = sum(1 for r in model_results if r.get("initial_wp") is not None)

        total_prompt = sum(r.get("usage_1", {}).get("prompt_tokens", 0) for r in model_results)
        total_comp = sum(r.get("usage_1", {}).get("completion_tokens", 0) for r in model_results)

        return {
            "model": name,
            "n_samples": n,
            "n_predicted": n_predicted,
            "prediction_rate": round(n_predicted / n, 3) if n > 0 else 0,
            "avg_initial_dist": round(float(np.mean(initial_dists)), 2) if initial_dists else None,
            "avg_refined_dist": round(float(np.mean(refined_dists)), 2) if refined_dists else None,
            "n_improved": n_improved,
            "improvement_rate": round(n_improved / n_valid, 3) if n_valid > 0 else 0,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_comp,
        }

    summary_2b = summarize_model(results_2b, "2B")
    summary_8b = summarize_model(results_8b, "8B")

    summary = {
        "task": "task34_2b_vs_8b",
        "model_2b": summary_2b,
        "model_8b": summary_8b,
        "comparison": {
            "better_initial": "8B" if (summary_8b.get("avg_initial_dist") or 999) < (summary_2b.get("avg_initial_dist") or 999) else "2B",
            "better_refined": "8B" if (summary_8b.get("avg_refined_dist") or 999) < (summary_2b.get("avg_refined_dist") or 999) else "2B",
            "better_improvement_rate": "8B" if summary_8b.get("improvement_rate", 0) > summary_2b.get("improvement_rate", 0) else "2B",
        },
    }
    print(f"  TASK 34 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary, "results_2b": results_2b, "results_8b": results_8b}


# ---------------------------------------------------------------------------
# Task 35: Token cost analysis
# ---------------------------------------------------------------------------

def run_task35(all_results: dict[str, Any]) -> dict[str, Any]:
    """Aggregate token costs from all waypoint experiments."""
    print("\n" + "=" * 70)
    print("TASK 35: Token cost analysis")
    print("=" * 70)

    task_costs = {}
    total_prompt = 0
    total_completion = 0

    for task_key in sorted(all_results.keys()):
        if not task_key.startswith("task"):
            continue
        data = all_results[task_key]
        summary = data.get("summary", {})
        usage = summary.get("token_usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Also count from model-specific summaries (task34)
        if "model_2b" in summary:
            p2b = summary["model_2b"].get("total_prompt_tokens", 0)
            c2b = summary["model_2b"].get("total_completion_tokens", 0)
            p8b = summary["model_8b"].get("total_prompt_tokens", 0)
            c8b = summary["model_8b"].get("total_completion_tokens", 0)
            prompt_tokens = p2b + p8b
            completion_tokens = c2b + c8b
            total = prompt_tokens + completion_tokens

        n_samples = summary.get("total_samples") or summary.get("n_samples", 0)
        # For task33, compute total samples across scenes
        if "scene_summaries" in summary:
            n_samples = sum(
                v.get("n_samples", 0) for v in summary.get("scene_summaries", {}).values()
            )

        task_costs[task_key] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "n_samples": n_samples,
            "tokens_per_sample": round(total / n_samples, 1) if n_samples > 0 else 0,
        }
        total_prompt += prompt_tokens
        total_completion += completion_tokens

    grand_total = total_prompt + total_completion

    summary = {
        "task": "task35_token_costs",
        "per_task": task_costs,
        "grand_total": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": grand_total,
        },
        "note": (
            "Token counts may be partially estimated for tasks using the orchestrator "
            "(which does not always expose raw usage). The simple_chat calls provide "
            "exact token counts from the vLLM API."
        ),
    }
    print(f"  TASK 35 SUMMARY: {json.dumps(summary, indent=2)}")
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Phase 2 Waypoint Visualization tasks (26-35)."""
    print("=" * 70)
    print("PHASE 2: WAYPOINT VISUALIZATION DEEP DIVE (Tasks 26-35)")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print()

    # Start servers
    print("Starting vLLM servers...")
    proc_2b = start_vllm_server(BASE_2B_MODEL, PORT_2B, GPU_2B)
    proc_8b = start_vllm_server(BASE_8B_MODEL, PORT_8B, GPU_8B)

    base_url_2b = f"http://localhost:{PORT_2B}"
    base_url_8b = f"http://localhost:{PORT_8B}"

    try:
        print("\nWaiting for 2B server...")
        ok_2b = wait_for_server(PORT_2B, timeout=1200)
        if not ok_2b:
            print("ERROR: 2B server failed to start")
            sys.exit(1)

        print("Waiting for 8B server (may take 15+ min)...")
        ok_8b = wait_for_server(PORT_8B, timeout=1800)
        if not ok_8b:
            print("ERROR: 8B server failed to start")
            sys.exit(1)

        print("\nBoth servers healthy. Starting experiments...\n")

        all_results: dict[str, Any] = {
            "experiment": "phase2_waypoint_visualization",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "models": {
                "2b": BASE_2B_MODEL,
                "8b": BASE_8B_MODEL,
            },
            "config": {
                "grid_size": GRID_SIZE,
                "image_size": f"{IMG_W}x{IMG_H}",
                "system_prompt": SYSTEM_PROMPT,
            },
        }

        # Task 26
        t26 = run_task26(base_url_2b)
        all_results["task26"] = t26

        # Task 27
        t27 = run_task27(base_url_2b)
        all_results["task27"] = t27

        # Task 28
        t28 = run_task28(base_url_2b)
        all_results["task28"] = t28

        # Task 29
        t29 = run_task29(base_url_2b)
        all_results["task29"] = t29

        # Task 30
        t30 = run_task30(base_url_2b)
        all_results["task30"] = t30

        # Task 31
        t31 = run_task31(base_url_2b)
        all_results["task31"] = t31

        # Task 32
        t32 = run_task32(base_url_2b)
        all_results["task32"] = t32

        # Task 33
        t33 = run_task33(base_url_2b)
        all_results["task33"] = t33

        # Task 34
        t34 = run_task34(base_url_2b, base_url_8b)
        all_results["task34"] = t34

        # Task 35 (aggregation)
        t35 = run_task35(all_results)
        all_results["task35"] = t35

        # Save results
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {RESULTS_PATH}")

        # Final summary
        print("\n" + "=" * 70)
        print("PHASE 2 FINAL SUMMARY")
        print("=" * 70)
        for task_key in sorted(all_results.keys()):
            if not task_key.startswith("task"):
                continue
            data = all_results[task_key]
            s = data.get("summary", {})
            task_name = s.get("task", task_key)
            print(f"\n{task_name}:")
            for k, v in s.items():
                if k not in ("task", "samples", "per_task", "scene_summaries", "token_usage"):
                    print(f"  {k}: {v}")

    finally:
        print("\nStopping servers...")
        try:
            proc_2b.terminate()
            proc_2b.wait(timeout=15)
        except Exception:
            proc_2b.kill()
        try:
            proc_8b.terminate()
            proc_8b.wait(timeout=15)
        except Exception:
            proc_8b.kill()
        kill_servers()
        print("Servers stopped.")

    print("\nPhase 2 Waypoint Visualization complete.")


if __name__ == "__main__":
    main()
