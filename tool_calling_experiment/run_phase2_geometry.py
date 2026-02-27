#!/usr/bin/env python3
# ruff: noqa: E501,E402,I001
# type: ignore[import-not-found]
"""Phase 2: Road Geometry Deep Dive. Run independently with: nohup python3 -u run_phase2_geometry.py &

Tasks:
    36: Geometry-guided waypoint prediction (2B, 100 samples)
    37: Geometry tool accuracy audit (50 samples, no model)
    39: Curvature-action consistency (2B, 100 samples)
    41: Geometry with 8B (100 samples, same as Task 36)

Servers: GPU 4 (2B port 8404), GPU 5 (8B port 8405)
Saves to: phase2_geometry_final.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
_PARENT = os.path.dirname(_DIR)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _PARENT]

from orchestrator import ToolCallingOrchestrator, parse_prediction
from robust_runner import (
    SYSTEM_PROMPT,
    RobustServer,
)
from visual_tools import (
    analyze_road_geometry,
    load_sample_image,
    load_sample_metadata,
)

DB_PATH = "/workspace/vllm/self_consistency_experiment/self_consistency.db"
RESULTS_PATH = os.path.join(_DIR, "phase2_geometry_final.json")

# Server config
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
GPU_2B = 4
GPU_8B = 5
PORT_2B = 8404
PORT_8B = 8405

GRID_SIZE = 63


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_all(all_results: dict[str, Any]) -> None:
    """Save all results to disk."""
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  [saved to {RESULTS_PATH}]")


def _get_diverse_indices(n: int, dataset_size: int = 8378) -> list[int]:
    """Evenly-spaced sample indices across the dataset."""
    step = dataset_size // n
    return [i * step for i in range(n)]


def _parse_waypoint_from_text(text: str) -> tuple[int, int] | None:
    """Extract row,col waypoint from model text."""
    if not text:
        return None
    # FINAL_WAYPOINT: row, col
    m = re.search(r"FINAL_WAYPOINT[:\s]+\[?\s*(\d+)\s*[,\s]+\s*(\d+)\s*\]?", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    # row=X, col=Y
    m = re.search(r"row\s*[=:]\s*(\d+)\s*[,;]\s*col(?:umn)?\s*[=:]\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Generic pair after keyword
    m = re.search(r"(?:waypoint|predict|output|answer)[^0-9]*(\d{1,2})\s*[,]\s*(\d{1,2})", text, re.IGNORECASE)
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return r, c
    return None


def _is_in_drivable(row: int, col: int, drivable_bins: list[list[int]]) -> bool:
    """Check if (row, col) is in the drivable bin set."""
    return any(dr == row and dc == col for dr, dc in drivable_bins)


def _build_geo_summary(geo: dict[str, Any]) -> str:
    """Build text summary of geometry analysis results."""
    bounds = geo.get("road_boundaries", {})
    left_x = bounds.get("left_x", "?")
    right_x = bounds.get("right_x", "?")
    n_bins = len(geo.get("drivable_bins", []))
    return (
        "Road geometry analysis results:\n"
        f"- Lanes detected: {geo.get('num_lanes_detected', 0)}\n"
        f"- Road curvature: {geo.get('road_curvature', 'unknown')}\n"
        f"- Vanishing point: {geo.get('vanishing_point', 'not detected')}\n"
        f"- Road boundaries: left_x={left_x}, right_x={right_x}\n"
        f"- Drivable region: {geo.get('drivable_region', 'unknown')}\n"
        f"- Drivable bins (63x63 grid): {n_bins}"
    )


def _fine_class_to_scene(fine_class: str) -> str:
    """Convert fine_class to coarse scene_type."""
    fc = fine_class.lower().strip()
    if "nominal" in fc:
        return "nominal"
    if "flood" in fc:
        return "flooded"
    if "incident" in fc:
        return "incident_zone"
    if "police" in fc or "horse" in fc or "mounted" in fc:
        return "mounted_police"
    if "flagger" in fc:
        return "flagger"
    return fc


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------
def run_task36(server_url: str) -> dict[str, Any]:
    """Task 36: Geometry-guided waypoint prediction (2B)."""
    print("\n=== Task 36: Geometry-guided waypoint (2B, 100 samples) ===")
    indices = _get_diverse_indices(100)

    print("  Pre-computing geometry...")
    geo_cache: dict[int, dict[str, Any]] = {}
    for idx in indices:
        img_path = load_sample_image(idx)
        geo_cache[idx] = analyze_road_geometry(img_path)

    orch = ToolCallingOrchestrator(
        server_url=server_url, tools={}, tool_definitions=[],
        max_tool_rounds=1, temperature=0, max_tokens=512,
    )

    results: list[dict[str, Any]] = []
    valid_wp = 0
    in_drivable = 0

    for i, idx in enumerate(indices):
        if i % 20 == 0:
            print(f"  Processing {i + 1}/100 (sample={idx})...")
        img_path = load_sample_image(idx)
        geo = geo_cache[idx]
        geo_summary = _build_geo_summary(geo)
        prompt = (
            f"{geo_summary}\n\n"
            "Use the road geometry to predict where the vehicle should drive. "
            "Your waypoint must be within the drivable region. "
            "Output as row,col in a 63x63 grid.\n\n"
            "Format: FINAL_WAYPOINT: row, col"
        )
        result = orch.run(
            image_path=img_path, system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt, tool_choice="none",
        )
        wp = _parse_waypoint_from_text(result["final_text"])
        drivable_bins = geo.get("drivable_bins", [])
        wp_in_drivable = False
        if wp is not None:
            valid_wp += 1
            wp_in_drivable = _is_in_drivable(wp[0], wp[1], drivable_bins)
            if wp_in_drivable:
                in_drivable += 1
        results.append({
            "sample_index": idx,
            "predicted_waypoint": list(wp) if wp else None,
            "in_drivable": wp_in_drivable,
            "road_curvature": geo.get("road_curvature", "unknown"),
            "latency_ms": result["latency_ms"],
        })

    drivable_rate = in_drivable / valid_wp if valid_wp > 0 else 0
    summary = {
        "total": 100, "valid_waypoints": valid_wp,
        "parse_rate": round(valid_wp / 100, 3),
        "in_drivable": in_drivable,
        "drivable_rate": round(drivable_rate, 3),
    }
    print(f"  T36: valid={valid_wp}, drivable={in_drivable}/{valid_wp} ({drivable_rate:.1%})")
    return {"summary": summary, "samples": results}


def run_task37() -> dict[str, Any]:
    """Task 37: Geometry tool accuracy audit (no model)."""
    print("\n=== Task 37: Geometry tool audit (50 samples, no model) ===")
    indices = _get_diverse_indices(50, 8378)

    results: list[dict[str, Any]] = []
    non_trivial = 0
    has_vp = 0
    curvature_dist: dict[str, int] = {}
    lane_counts: list[int] = []

    for i, idx in enumerate(indices):
        if i % 10 == 0:
            print(f"  Processing {i + 1}/50 (sample={idx})...")
        img_path = load_sample_image(idx)
        geo = analyze_road_geometry(img_path)
        meta = load_sample_metadata(idx)
        num_lanes = geo.get("num_lanes_detected", 0)
        curvature = geo.get("road_curvature", "unknown")
        vp = geo.get("vanishing_point")
        is_nt = num_lanes > 0 and curvature != "unknown"
        if is_nt:
            non_trivial += 1
        if vp is not None:
            has_vp += 1
        curvature_dist[curvature] = curvature_dist.get(curvature, 0) + 1
        lane_counts.append(num_lanes)
        results.append({
            "sample_index": idx,
            "num_lanes": num_lanes,
            "curvature": curvature,
            "vanishing_point": vp,
            "non_trivial": is_nt,
            "scene_type_gt": meta.get("fine_class", "unknown"),
        })

    import numpy as np
    avg_lanes = float(np.mean(lane_counts)) if lane_counts else 0
    summary = {
        "total": 50, "non_trivial": non_trivial,
        "non_trivial_rate": round(non_trivial / 50, 3),
        "has_vanishing_point": has_vp,
        "curvature_dist": curvature_dist,
        "avg_lanes": round(avg_lanes, 2),
    }
    print(f"  T37: non_trivial={non_trivial}/50, VP={has_vp}/50, curvature={curvature_dist}")
    return {"summary": summary, "samples": results}


def run_task39(server_url: str) -> dict[str, Any]:
    """Task 39: Curvature-action consistency (2B)."""
    print("\n=== Task 39: Curvature-action consistency (2B, 100 samples) ===")
    indices = _get_diverse_indices(100)

    orch = ToolCallingOrchestrator(
        server_url=server_url, tools={}, tool_definitions=[],
        max_tool_rounds=1, temperature=0, max_tokens=512,
    )

    results: list[dict[str, Any]] = []
    long_correct = 0
    lat_correct = 0
    total_gt = 0

    for i, idx in enumerate(indices):
        if i % 20 == 0:
            print(f"  Processing {i + 1}/100 (sample={idx})...")
        img_path = load_sample_image(idx)
        meta = load_sample_metadata(idx)
        geo = analyze_road_geometry(img_path)
        curvature = geo.get("road_curvature", "unknown")
        num_lanes = geo.get("num_lanes_detected", 0)
        bounds = geo.get("road_boundaries", {})
        bnd_text = f"left_x={bounds.get('left_x', '?')}, right_x={bounds.get('right_x', '?')}"

        prompt = (
            "The road geometry shows:\n"
            f"- Curvature: {curvature}\n"
            f"- Lanes detected: {num_lanes}\n"
            f"- Road boundaries: {bnd_text}\n\n"
            "What driving action is appropriate?\n"
            "Choose longitudinal: null, stop, slowdown, proceed\n"
            "Choose lateral: null, lc_left, lc_right\n\n"
            "Output format:\n"
            "FINAL_LONG_ACTION: <action>\n"
            "FINAL_LAT_ACTION: <action>"
        )
        result = orch.run(
            image_path=img_path, system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt, tool_choice="none",
        )
        pred = parse_prediction(result["final_text"])
        gt_long = meta.get("long_action", "null")
        gt_lat = meta.get("lat_action", "null")
        total_gt += 1
        if pred.get("long_action") == gt_long:
            long_correct += 1
        if pred.get("lat_action") == gt_lat:
            lat_correct += 1
        results.append({
            "sample_index": idx, "curvature": curvature,
            "pred_long": pred.get("long_action"),
            "pred_lat": pred.get("lat_action"),
            "gt_long": gt_long, "gt_lat": gt_lat,
            "latency_ms": result["latency_ms"],
        })

    long_acc = long_correct / total_gt if total_gt > 0 else 0
    lat_acc = lat_correct / total_gt if total_gt > 0 else 0
    summary = {
        "total": 100, "total_with_gt": total_gt,
        "long_accuracy": round(long_acc, 3),
        "lat_accuracy": round(lat_acc, 3),
    }
    print(f"  T39: long_acc={long_acc:.1%}, lat_acc={lat_acc:.1%}")
    return {"summary": summary, "samples": results}


def run_task41(server_url: str) -> dict[str, Any]:
    """Task 41: Geometry-guided waypoint prediction (8B)."""
    print("\n=== Task 41: Geometry-guided waypoint (8B, 100 samples) ===")
    indices = _get_diverse_indices(100)

    print("  Pre-computing geometry...")
    geo_cache: dict[int, dict[str, Any]] = {}
    for idx in indices:
        img_path = load_sample_image(idx)
        geo_cache[idx] = analyze_road_geometry(img_path)

    orch = ToolCallingOrchestrator(
        server_url=server_url, tools={}, tool_definitions=[],
        max_tool_rounds=1, temperature=0, max_tokens=512,
    )

    results: list[dict[str, Any]] = []
    valid_wp = 0
    in_drivable = 0

    for i, idx in enumerate(indices):
        if i % 20 == 0:
            print(f"  Processing {i + 1}/100 (sample={idx})...")
        img_path = load_sample_image(idx)
        geo = geo_cache[idx]
        geo_summary = _build_geo_summary(geo)
        prompt = (
            f"{geo_summary}\n\n"
            "Use the road geometry to predict where the vehicle should drive. "
            "Your waypoint must be within the drivable region. "
            "Output as row,col in a 63x63 grid.\n\n"
            "Format: FINAL_WAYPOINT: row, col"
        )
        result = orch.run(
            image_path=img_path, system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt, tool_choice="none",
        )
        wp = _parse_waypoint_from_text(result["final_text"])
        drivable_bins = geo.get("drivable_bins", [])
        wp_in_drivable = False
        if wp is not None:
            valid_wp += 1
            wp_in_drivable = _is_in_drivable(wp[0], wp[1], drivable_bins)
            if wp_in_drivable:
                in_drivable += 1
        results.append({
            "sample_index": idx,
            "predicted_waypoint": list(wp) if wp else None,
            "in_drivable": wp_in_drivable,
            "road_curvature": geo.get("road_curvature", "unknown"),
            "latency_ms": result["latency_ms"],
        })

    drivable_rate = in_drivable / valid_wp if valid_wp > 0 else 0
    summary = {
        "total": 100, "valid_waypoints": valid_wp,
        "parse_rate": round(valid_wp / 100, 3),
        "in_drivable": in_drivable,
        "drivable_rate": round(drivable_rate, 3),
    }
    print(f"  T41: valid={valid_wp}, drivable={in_drivable}/{valid_wp} ({drivable_rate:.1%})")
    return {"summary": summary, "samples": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start_time = time.time()
    print("=" * 70)
    print("PHASE 2: Road Geometry Deep Dive (Tasks 36, 37, 39, 41)")
    print("=" * 70)

    all_results: dict[str, Any] = {
        "experiment": "phase2_geometry_final",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # === 2B Tasks (GPU 4) ===
    SKIP_2B = True
    if not SKIP_2B:
        print("\n--- Starting 2B server ---")
        server_2b = RobustServer(MODEL_2B, gpu_id=GPU_2B, port=PORT_2B)
        server_2b.start(timeout=1800)
        url_2b = f"http://localhost:{PORT_2B}"

        try:
            # Task 36
            t36 = run_task36(url_2b)
            all_results["task36"] = t36
            _save_all(all_results)

            # Task 37 (no model needed, but run while 2B is up)
            t37 = run_task37()
            all_results["task37"] = t37
            _save_all(all_results)

            # Task 39
            t39 = run_task39(url_2b)
            all_results["task39"] = t39
            _save_all(all_results)

        finally:
            print("\n--- Stopping 2B server ---")
            server_2b.stop()
            time.sleep(5)

    # === 8B Task (GPU 5) ===
    print("\n--- Starting 8B server ---")
    server_8b = RobustServer(MODEL_8B, gpu_id=GPU_8B, port=PORT_8B)
    server_8b.start(timeout=1800)
    url_8b = f"http://localhost:{PORT_8B}"

    try:
        t41 = run_task41(url_8b)
        all_results["task41"] = t41
        _save_all(all_results)
    finally:
        print("\n--- Stopping 8B server ---")
        server_8b.stop()

    # 2B vs 8B comparison
    t36_s = all_results.get("task36", {}).get("summary", {})
    t41_s = all_results.get("task41", {}).get("summary", {})
    all_results["comparison_2b_vs_8b"] = {
        "2b_parse_rate": t36_s.get("parse_rate", 0),
        "8b_parse_rate": t41_s.get("parse_rate", 0),
        "2b_drivable_rate": t36_s.get("drivable_rate", 0),
        "8b_drivable_rate": t41_s.get("drivable_rate", 0),
    }
    _save_all(all_results)

    elapsed = time.time() - start_time
    print(f"\n=== ALL TASKS COMPLETE ({elapsed / 60:.1f} min) ===")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
