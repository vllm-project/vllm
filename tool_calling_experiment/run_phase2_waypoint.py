#!/usr/bin/env python3
# ruff: noqa: E501,E402,I001
# type: ignore[import-not-found]
"""Phase 2: Waypoint Visualization Deep Dive. Run independently with: nohup python3 -u run_phase2_waypoint.py &

Tasks:
    26: Self-correction via waypoint visualization (2B, 100 samples)
    27: Waypoint viz for scene verification (2B, 100 false IZ)
    29: Iterative waypoint refinement (2B, 100 samples, 3 rounds)
    31: Off-road detection (2B, 100 bad waypoints)
    32: Waypoint spatial precision before/after viz (2B, 100 samples)
    34: 2B vs 8B waypoint reasoning (100 samples, both models)

Servers: GPU 2 (2B port 8402), GPU 3 (8B port 8403)
Saves to: phase2_waypoint_final.json
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from typing import Any

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
_PARENT = os.path.dirname(_DIR)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _PARENT]

from orchestrator import parse_prediction
from robust_runner import (
    SYSTEM_PROMPT,
    RobustServer,
    load_samples,
)
from visual_tools import load_sample_image, load_sample_metadata, visualize_waypoint

DB_PATH = "/workspace/vllm/self_consistency_experiment/self_consistency.db"
RESULTS_PATH = os.path.join(_DIR, "phase2_waypoint_final.json")

# Server config
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_2B = 2
GPU_8B = 3
PORT_2B = 8402
PORT_8B = 8403

GRID_SIZE = 63
IMG_W = 504
IMG_H = 336

np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_all(all_results: dict[str, Any]) -> None:
    """Save all results to disk."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  [saved to {RESULTS_PATH}]")


def _waypoint_l2(r1: int, c1: int, r2: int, c2: int) -> float:
    """L2 distance between two grid waypoints."""
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)


_ROW_COL_RE = re.compile(
    r"(?:row|r)\s*[=:]\s*(\d+)\s*[,;/\s]+\s*(?:col|c|column)\s*[=:]\s*(\d+)",
    re.IGNORECASE,
)
_PAIR_RE = re.compile(r"\b(\d{1,2})\s*[,]\s*(\d{1,2})\b")


def _parse_waypoint(text: str) -> tuple[int, int] | None:
    """Parse a row,col waypoint from model output text."""
    if not text:
        return None
    m = _ROW_COL_RE.search(text)
    if m:
        row, col = int(m.group(1)), int(m.group(2))
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return (row, col)
    for m in _PAIR_RE.finditer(text):
        r, c = int(m.group(1)), int(m.group(2))
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return (r, c)
    return None


def _ego_waypoint_to_grid(
    wp_x: float, wp_y: float,
    cam_intrinsics: list[float],
    cam_extrinsics: list[float],
) -> tuple[int, int] | None:
    """Project an ego-frame waypoint to a grid row,col."""
    E = np.array(cam_extrinsics).reshape(4, 4)
    K = np.array(cam_intrinsics).reshape(4, 4)
    pt_ego = np.array([wp_x, wp_y, 0.0, 1.0])
    pt_cam = E @ pt_ego
    if pt_cam[2] <= 0:
        return None
    pt_proj = K @ pt_cam
    u = pt_proj[0] / pt_proj[2]
    v = pt_proj[1] / pt_proj[2]
    orig_w, orig_h = 2880, 1860
    px = u * (IMG_W / orig_w)
    py = v * (IMG_H / orig_h)
    if px < 0 or px >= IMG_W or py < 0 or py >= IMG_H:
        return None
    bin_w = IMG_W / GRID_SIZE
    bin_h = IMG_H / GRID_SIZE
    col = max(0, min(int(px / bin_w), GRID_SIZE - 1))
    row = max(0, min(int(py / bin_h), GRID_SIZE - 1))
    return (row, col)


def _get_gt_waypoint(sample_index: int) -> tuple[int, int] | None:
    """Get ground truth waypoint as grid (row, col)."""
    try:
        meta = load_sample_metadata(sample_index)
    except Exception:
        return None
    gt_wps = meta.get("ground_truth_waypoints", [])
    if not gt_wps:
        return None
    wp = gt_wps[0]
    cals = meta.get("camera_calibration", [])
    cam_cal = None
    for cal in cals:
        if cal.get("camera_name") == "front_right_front_wide":
            cam_cal = cal
            break
    if cam_cal is None:
        return None
    return _ego_waypoint_to_grid(
        wp[0], wp[1], cam_cal["intrinsics"], cam_cal["extrinsics"],
    )


def _sim_pred_waypoint(gt_row: int, gt_col: int, noise_std: float = 8.0) -> tuple[int, int]:
    """Simulate a predicted waypoint with Gaussian noise."""
    row = int(gt_row + np.random.normal(0, noise_std))
    col = int(gt_col + np.random.normal(0, noise_std))
    return (max(0, min(row, GRID_SIZE - 1)), max(0, min(col, GRID_SIZE - 1)))


def _collect_wp_samples(sample_ids: list[int], noise_std: float, max_n: int) -> list[dict[str, Any]]:
    """Collect samples with valid GT waypoints."""
    samples = []
    for sid in sample_ids:
        if len(samples) >= max_n:
            break
        gt = _get_gt_waypoint(sid)
        if gt is None:
            continue
        gt_row, gt_col = gt
        pred_row, pred_col = _sim_pred_waypoint(gt_row, gt_col, noise_std)
        try:
            img_path = load_sample_image(sid)
        except Exception:
            continue
        try:
            meta = load_sample_metadata(sid)
        except Exception:
            meta = {}
        samples.append({
            "sample_id": sid,
            "gt_row": gt_row,
            "gt_col": gt_col,
            "pred_row": pred_row,
            "pred_col": pred_col,
            "image_path": img_path,
            "scene_type_gt": meta.get("fine_class", "unknown"),
            "long_action_gt": meta.get("long_action", "null"),
        })
    return samples


def _simple_chat(
    server: RobustServer,
    system_prompt: str,
    user_prompt: str,
    image_path: str | None = None,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Single chat call without tools via RobustServer."""
    import base64 as b64_mod
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if image_path:
        with open(image_path, "rb") as f:
            b64 = b64_mod.b64encode(f.read()).decode()
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    t0 = time.monotonic()
    msg = server.chat(messages, max_tokens=max_tokens)
    latency = (time.monotonic() - t0) * 1000
    if msg is None:
        return {"content": "", "latency_ms": round(latency, 1)}
    return {"content": msg.get("content", ""), "latency_ms": round(latency, 1)}


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------
def run_task26(server: RobustServer, wp_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Task 26: Self-correction via waypoint visualization."""
    print("\n=== Task 26: Self-correction via waypoint viz (2B, 100 samples) ===")
    results = []
    for i, s in enumerate(wp_samples):
        if i % 10 == 0 and i > 0:
            server.ensure_healthy()
            print(f"  [T26] {i}/{len(wp_samples)} done")
        viz = visualize_waypoint(s["image_path"], s["pred_row"], s["pred_col"], label="predicted")
        prompt = (
            "A model predicted the vehicle should drive to the red marker. "
            "Does this location make sense for safe driving? "
            "If not, suggest a better waypoint as row,col in a 63x63 grid."
        )
        resp = _simple_chat(server, SYSTEM_PROMPT, prompt, image_path=viz["annotated_image"], max_tokens=512)
        content = resp["content"]
        suggested = _parse_waypoint(content)
        orig_dist = _waypoint_l2(s["pred_row"], s["pred_col"], s["gt_row"], s["gt_col"])
        corr_dist = _waypoint_l2(suggested[0], suggested[1], s["gt_row"], s["gt_col"]) if suggested else None
        improvement = (orig_dist - corr_dist) if corr_dist is not None else None
        results.append({
            "sample_id": s["sample_id"],
            "original_dist": round(orig_dist, 2),
            "corrected_dist": round(corr_dist, 2) if corr_dist is not None else None,
            "improvement": round(improvement, 2) if improvement is not None else None,
            "improved": improvement is not None and improvement > 0,
            "latency_ms": resp["latency_ms"],
        })
    n = len(results)
    n_sug = sum(1 for r in results if r.get("corrected_dist") is not None)
    n_imp = sum(1 for r in results if r.get("improved"))
    imps = [r["improvement"] for r in results if r.get("improvement") is not None]
    summary = {
        "total": n, "n_suggested": n_sug, "n_improved": n_imp,
        "suggestion_rate": round(n_sug / n, 3) if n > 0 else 0,
        "improvement_rate": round(n_imp / n_sug, 3) if n_sug > 0 else 0,
        "avg_improvement": round(float(np.mean(imps)), 2) if imps else None,
    }
    print(f"  T26: suggested={n_sug}, improved={n_imp}, avg_imp={summary['avg_improvement']}")
    return {"summary": summary, "samples": results}


def run_task27(server: RobustServer, false_iz: list[dict[str, Any]]) -> dict[str, Any]:
    """Task 27: Waypoint viz for scene verification on false IZ."""
    print("\n=== Task 27: Waypoint viz scene verification (2B, 100 false IZ) ===")
    results = []
    for i, s in enumerate(false_iz[:100]):
        if i % 10 == 0 and i > 0:
            server.ensure_healthy()
            print(f"  [T27] {i}/100 done")
        sid = s.get("sample_id", 0)
        gt = _get_gt_waypoint(sid)
        if gt is None:
            results.append({"sample_id": sid, "error": "no_waypoint"})
            continue
        gt_row, gt_col = gt
        pred_row, pred_col = _sim_pred_waypoint(gt_row, gt_col, 5.0)
        try:
            img_path = load_sample_image(sid)
        except Exception:
            results.append({"sample_id": sid, "error": "load_failed"})
            continue
        viz = visualize_waypoint(img_path, pred_row, pred_col, label="predicted")
        prompt = (
            "A model predicted incident_zone and suggested driving to the red marker. "
            "Does the waypoint location match what you'd expect for an incident response? "
            "Classify: nominal, flooded, incident_zone, mounted_police, flagger.\n\n"
            "FINAL_SCENE: "
        )
        resp = _simple_chat(server, SYSTEM_PROMPT, prompt, image_path=viz["annotated_image"], max_tokens=512)
        pred = parse_prediction(resp["content"])
        gt_scene = s.get("scene_type_gt", "nominal")
        correct = pred.get("scene") == gt_scene if pred.get("scene") else False
        caught = pred.get("scene") is not None and pred.get("scene") != "incident_zone" and gt_scene != "incident_zone"
        results.append({
            "sample_id": sid, "gt_scene": gt_scene,
            "corrected_pred": pred.get("scene"), "correct": correct,
            "caught_misclass": caught, "latency_ms": resp["latency_ms"],
        })
    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    n_caught = sum(1 for r in results if r.get("caught_misclass"))
    summary = {
        "total": n, "correct": n_correct,
        "accuracy": round(n_correct / n, 3) if n > 0 else 0,
        "catch_rate": round(n_caught / n, 3) if n > 0 else 0,
    }
    print(f"  T27: accuracy={summary['accuracy']}, catch_rate={summary['catch_rate']}")
    return {"summary": summary, "samples": results}


def run_task29(server: RobustServer, wp_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Task 29: Iterative waypoint refinement (3 rounds)."""
    print("\n=== Task 29: Iterative refinement (2B, 100 samples, 3 rounds) ===")
    results = []
    for i, s in enumerate(wp_samples):
        if i % 10 == 0 and i > 0:
            server.ensure_healthy()
            print(f"  [T29] {i}/{len(wp_samples)} done")
        iterations = []
        current_wp = None
        for it in range(3):
            if it == 0:
                prompt = "Predict where the vehicle should drive as a waypoint in a 63x63 grid. Output as row,col."
                img_path = s["image_path"]
            else:
                if current_wp is None:
                    break
                viz = visualize_waypoint(s["image_path"], current_wp[0], current_wp[1], label=f"iter{it}")
                img_path = viz["annotated_image"]
                prompt = (
                    f"Your previous prediction was row={current_wp[0]}, col={current_wp[1]} "
                    f"(shown as the red marker). Refine if needed. Output as row,col."
                )
            resp = _simple_chat(server, SYSTEM_PROMPT, prompt, image_path=img_path, max_tokens=512)
            new_wp = _parse_waypoint(resp["content"])
            dist = _waypoint_l2(new_wp[0], new_wp[1], s["gt_row"], s["gt_col"]) if new_wp else None
            if new_wp:
                current_wp = new_wp
            iterations.append({"iteration": it, "wp": new_wp, "dist": round(dist, 2) if dist is not None else None})
        dists = [it_d["dist"] for it_d in iterations if it_d.get("dist") is not None]
        improved = len(dists) >= 2 and dists[-1] < dists[0]
        results.append({
            "sample_id": s["sample_id"], "iterations": iterations,
            "distances": dists, "improved": improved,
        })
    n = len(results)
    n_imp = sum(1 for r in results if r.get("improved"))
    iter_dists: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for r in results:
        for it_d in r.get("iterations", []):
            idx = it_d.get("iteration", 0)
            d = it_d.get("dist")
            if d is not None and idx in iter_dists:
                iter_dists[idx].append(d)
    avg_by_iter = {f"iter_{k}": round(float(np.mean(v)), 2) if v else None for k, v in iter_dists.items()}
    summary = {
        "total": n, "n_improved": n_imp,
        "improvement_rate": round(n_imp / n, 3) if n > 0 else 0,
        "avg_dist_by_iter": avg_by_iter,
    }
    print(f"  T29: improved={n_imp}/{n}, avg_by_iter={avg_by_iter}")
    return {"summary": summary, "samples": results}


def run_task31(server: RobustServer, db_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Task 31: Off-road detection with bad waypoints."""
    print("\n=== Task 31: Off-road detection (2B, 100 bad waypoints) ===")
    import random
    random.seed(42)
    results = []
    count = 0
    for s in db_samples:
        if count >= 100:
            break
        sid = s.get("sample_id", 0)
        try:
            img_path = load_sample_image(sid)
        except Exception:
            continue
        corner = random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
        if corner == "top-left":
            r, c = random.randint(0, 10), random.randint(0, 10)
        elif corner == "top-right":
            r, c = random.randint(0, 10), random.randint(53, 62)
        elif corner == "bottom-left":
            r, c = random.randint(53, 62), random.randint(0, 10)
        else:
            r, c = random.randint(53, 62), random.randint(53, 62)
        viz = visualize_waypoint(img_path, r, c, label="suggested")
        prompt = (
            f"A waypoint at row={r}, col={c} is suggested for driving. "
            f"Look at the red marker. Is this on the drivable road? "
            f"Answer YES or NO, then explain briefly."
        )
        resp = _simple_chat(server, SYSTEM_PROMPT, prompt, image_path=viz["annotated_image"], max_tokens=256)
        lower = resp["content"].lower()
        said_no = "no" in lower.split()[:5] or lower.strip().startswith("no")
        said_yes = "yes" in lower.split()[:5] or lower.strip().startswith("yes")
        detected = said_no and not said_yes
        results.append({
            "sample_id": sid, "corner": corner,
            "detected_offroad": detected, "correct": detected,
            "latency_ms": resp["latency_ms"],
        })
        count += 1
        if count % 10 == 0:
            server.ensure_healthy()
            print(f"  [T31] {count}/100 done")
    n = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    summary = {
        "total": n, "n_correct": n_correct,
        "accuracy": round(n_correct / n, 3) if n > 0 else 0,
    }
    print(f"  T31: accuracy={summary['accuracy']}")
    return {"summary": summary, "samples": results}


def run_task32(server: RobustServer, wp_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Task 32: Waypoint precision before/after viz."""
    print("\n=== Task 32: Waypoint precision before/after viz (2B, 100 samples) ===")
    results = []
    for i, s in enumerate(wp_samples):
        if i % 10 == 0 and i > 0:
            server.ensure_healthy()
            print(f"  [T32] {i}/{len(wp_samples)} done")
        # Condition A: no viz
        prompt_a = "Predict where the vehicle should drive as a waypoint in a 63x63 grid. Output as row,col."
        resp_a = _simple_chat(server, SYSTEM_PROMPT, prompt_a, image_path=s["image_path"], max_tokens=512)
        wp_a = _parse_waypoint(resp_a["content"])
        dist_a = _waypoint_l2(wp_a[0], wp_a[1], s["gt_row"], s["gt_col"]) if wp_a else None
        # Condition B: viz + refine
        wp_b = None
        dist_b = None
        if wp_a:
            viz = visualize_waypoint(s["image_path"], wp_a[0], wp_a[1], label="initial")
            prompt_b = (
                f"You predicted row={wp_a[0]}, col={wp_a[1]} (red marker). "
                f"Refine if needed. Output as row,col."
            )
            resp_b = _simple_chat(server, SYSTEM_PROMPT, prompt_b, image_path=viz["annotated_image"], max_tokens=512)
            wp_b = _parse_waypoint(resp_b["content"])
            dist_b = _waypoint_l2(wp_b[0], wp_b[1], s["gt_row"], s["gt_col"]) if wp_b else None
        improved = dist_a is not None and dist_b is not None and dist_b < dist_a
        results.append({
            "sample_id": s["sample_id"],
            "dist_before": round(dist_a, 2) if dist_a is not None else None,
            "dist_after": round(dist_b, 2) if dist_b is not None else None,
            "improved": improved,
        })
    n = len(results)
    dists_a = [r["dist_before"] for r in results if r.get("dist_before") is not None]
    dists_b = [r["dist_after"] for r in results if r.get("dist_after") is not None]
    n_imp = sum(1 for r in results if r.get("improved"))
    n_both = sum(1 for r in results if r.get("dist_before") is not None and r.get("dist_after") is not None)
    summary = {
        "total": n,
        "avg_dist_before": round(float(np.mean(dists_a)), 2) if dists_a else None,
        "avg_dist_after": round(float(np.mean(dists_b)), 2) if dists_b else None,
        "n_improved": n_imp,
        "improvement_rate": round(n_imp / n_both, 3) if n_both > 0 else 0,
    }
    print(f"  T32: before={summary['avg_dist_before']}, after={summary['avg_dist_after']}, imp_rate={summary['improvement_rate']}")
    return {"summary": summary, "samples": results}


def run_task34(
    server_2b: RobustServer,
    server_8b: RobustServer,
    wp_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Task 34: 2B vs 8B waypoint reasoning."""
    print("\n=== Task 34: 2B vs 8B waypoint reasoning (100 samples) ===")

    def _run_model(server: RobustServer, label: str) -> list[dict[str, Any]]:
        model_results = []
        for i, s in enumerate(wp_samples):
            if i % 10 == 0 and i > 0:
                server.ensure_healthy()
                print(f"  [{label}] {i}/{len(wp_samples)} done")
            # Step 1: predict
            resp1 = _simple_chat(server, SYSTEM_PROMPT,
                "Predict a safe driving waypoint as row,col in a 63x63 grid.",
                image_path=s["image_path"], max_tokens=256)
            wp1 = _parse_waypoint(resp1["content"])
            dist1 = _waypoint_l2(wp1[0], wp1[1], s["gt_row"], s["gt_col"]) if wp1 else None
            # Step 2: visualize and refine
            wp2 = None
            dist2 = None
            if wp1:
                viz = visualize_waypoint(s["image_path"], wp1[0], wp1[1], label="predicted")
                resp2 = _simple_chat(server, SYSTEM_PROMPT,
                    f"Your waypoint at row={wp1[0]}, col={wp1[1]} is shown. Refine if needed. Output as row,col.",
                    image_path=viz["annotated_image"], max_tokens=256)
                wp2 = _parse_waypoint(resp2["content"])
                if wp2:
                    dist2 = _waypoint_l2(wp2[0], wp2[1], s["gt_row"], s["gt_col"])
            improved = dist1 is not None and dist2 is not None and dist2 < dist1
            model_results.append({
                "sample_id": s["sample_id"],
                "initial_dist": round(dist1, 2) if dist1 is not None else None,
                "refined_dist": round(dist2, 2) if dist2 is not None else None,
                "improved": improved,
            })
        return model_results

    r_2b = _run_model(server_2b, "2B")
    r_8b = _run_model(server_8b, "8B")

    def _summarize(model_r: list[dict[str, Any]], name: str) -> dict[str, Any]:
        n = len(model_r)
        init = [r["initial_dist"] for r in model_r if r.get("initial_dist") is not None]
        ref = [r["refined_dist"] for r in model_r if r.get("refined_dist") is not None]
        n_imp = sum(1 for r in model_r if r.get("improved"))
        n_both = sum(1 for r in model_r if r.get("initial_dist") is not None and r.get("refined_dist") is not None)
        return {
            "model": name, "n_samples": n,
            "avg_initial_dist": round(float(np.mean(init)), 2) if init else None,
            "avg_refined_dist": round(float(np.mean(ref)), 2) if ref else None,
            "n_improved": n_imp,
            "improvement_rate": round(n_imp / n_both, 3) if n_both > 0 else 0,
        }

    s2b = _summarize(r_2b, "2B")
    s8b = _summarize(r_8b, "8B")
    summary = {"model_2b": s2b, "model_8b": s8b}
    print(f"  T34: 2B avg_init={s2b['avg_initial_dist']}, 8B avg_init={s8b['avg_initial_dist']}")
    return {"summary": summary, "results_2b": r_2b, "results_8b": r_8b}


# ---------------------------------------------------------------------------
# Token cost
# ---------------------------------------------------------------------------
def _compute_token_stats(all_task_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate latency stats across tasks."""
    stats: dict[str, Any] = {}
    for task_name, task_data in all_task_data.items():
        samples = task_data.get("samples", [])
        if not samples:
            continue
        lats = [s.get("latency_ms", 0) for s in samples if isinstance(s, dict) and "error" not in s]
        n = len(samples)
        stats[task_name] = {
            "n_samples": n,
            "avg_latency_ms": round(sum(lats) / len(lats), 1) if lats else 0,
        }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: Waypoint Visualization Deep Dive (Tasks 26, 27, 29, 31, 32, 34)")
    print("=" * 70)

    all_results: dict[str, Any] = {
        "experiment": "phase2_waypoint_final",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    task_data: dict[str, dict[str, Any]] = {}

    # --- Load samples ---
    print("\nLoading samples...")
    false_iz = load_samples(DB_PATH, "false_iz", 200)
    all_samples = load_samples(DB_PATH, "all", 500)
    print(f"Loaded {len(false_iz)} false IZ, {len(all_samples)} all samples")

    # Collect waypoint samples
    wp_sample_ids = [s.get("sample_id", 0) for s in all_samples]
    wp_samples = _collect_wp_samples(wp_sample_ids, noise_std=10.0, max_n=100)
    print(f"Waypoint samples with valid GT: {len(wp_samples)}")

    # === 2B Tasks (GPU 2) ===
    SKIP_2B = True
    if not SKIP_2B:
        print("\n--- Starting 2B server ---")
        server_2b = RobustServer(MODEL_2B, gpu_id=GPU_2B, port=PORT_2B)
        server_2b.start(timeout=1800)

        try:
            # Task 26
            t26 = run_task26(server_2b, wp_samples)
            all_results["task26"] = t26["summary"]
            task_data["task26"] = t26
            _save_all(all_results)

            # Task 27
            t27 = run_task27(server_2b, false_iz)
            all_results["task27"] = t27["summary"]
            task_data["task27"] = t27
            _save_all(all_results)

            # Task 29
            t29 = run_task29(server_2b, wp_samples)
            all_results["task29"] = t29["summary"]
            task_data["task29"] = t29
            _save_all(all_results)

            # Task 31
            t31 = run_task31(server_2b, all_samples)
            all_results["task31"] = t31["summary"]
            task_data["task31"] = t31
            _save_all(all_results)

            # Task 32
            t32 = run_task32(server_2b, wp_samples)
            all_results["task32"] = t32["summary"]
            task_data["task32"] = t32
            _save_all(all_results)

        finally:
            print("\n--- Stopping 2B server ---")
            server_2b.stop()
            time.sleep(5)

    # === 8B Task (GPU 3) -- for Task 34 ===
    print("\n--- Starting 8B server ---")
    server_8b = RobustServer(MODEL_8B, gpu_id=GPU_8B, port=PORT_8B)
    server_8b.start(timeout=1800)

    # Also restart 2B for Task 34
    print("--- Restarting 2B server ---")
    server_2b_t34 = RobustServer(MODEL_2B, gpu_id=GPU_2B, port=PORT_2B)
    server_2b_t34.start(timeout=1800)

    try:
        t34 = run_task34(server_2b_t34, server_8b, wp_samples)
        all_results["task34"] = t34["summary"]
        task_data["task34"] = t34
        _save_all(all_results)
    finally:
        print("\n--- Stopping servers ---")
        server_2b_t34.stop()
        server_8b.stop()

    # Token cost
    token_stats = _compute_token_stats(task_data)
    all_results["token_costs"] = token_stats
    _save_all(all_results)

    elapsed = time.time() - start_time
    print(f"\n=== ALL TASKS COMPLETE ({elapsed / 60:.1f} min) ===")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
