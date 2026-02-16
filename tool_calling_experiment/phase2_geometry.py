#!/usr/bin/env python3
"""Phase 2: Road Geometry Deep Dive -- Tasks 36-42.

Task 36: Geometry-guided waypoint prediction (2B, 100 samples)
Task 37: Geometry tool accuracy audit (50 samples)
Task 38: Geometry + waypoint viz combined (2B, 100 samples)
Task 39: Curvature-action consistency (2B, 100 samples)
Task 40: Geometry on difficult images (2B, 50 samples)
Task 41: Geometry with 8B (100 samples, same as Task 36)
Task 42: Token cost analysis

All results saved to phase2_geometry_results.json.
Servers are killed when done.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

# ------------------------------------------------------------------
# Path setup
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

EXPERIMENT_DIR = _DIR

from orchestrator import ToolCallingOrchestrator, parse_prediction  # noqa: E402, I001  # type: ignore[import-not-found]
from server_utils import VLLMServer  # noqa: E402, I001  # type: ignore[import-not-found]
from visual_tools import (  # noqa: E402, I001  # type: ignore[import-not-found]
    TOOL_ROAD_GEOMETRY,
    TOOL_WAYPOINT_VIZ,
    _fine_class_to_scene_type,
    analyze_road_geometry,
    load_sample_image,
    load_sample_metadata,
    visualize_waypoint,
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
PORT_2B = 8334
PORT_8B = 8335
GPU_2B = 4
GPU_8B = 5

SYSTEM_PROMPT = "The image is 504x336 pixels."
RESULTS_PATH = os.path.join(
    EXPERIMENT_DIR, "phase2_geometry_results.json"
)

# Grid size
GRID_SIZE = 63


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _print_sep(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72 + "\n")


def _parse_waypoint_from_text(
    text: str,
) -> tuple[int, int] | None:
    """Extract a row,col waypoint from model response text.

    Looks for patterns like:
      row=32, col=31
      (32, 31)
      32,31
      FINAL_WAYPOINT: 32, 31
      row 32, column 31
    """
    if not text:
        return None

    # Pattern 1: FINAL_WAYPOINT: row, col
    m = re.search(
        r"FINAL_WAYPOINT[:\s]+\[?\s*(\d+)"
        r"\s*[,\s]+\s*(\d+)\s*\]?",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), int(m.group(2))

    # Pattern 2: row=X, col=Y or row X, col Y
    m = re.search(
        r"row\s*[=:]\s*(\d+)\s*[,;]"
        r"\s*col(?:umn)?\s*[=:]\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), int(m.group(2))

    # Pattern 3: (row, col) with explicit mention
    m = re.search(
        r"(?:waypoint|position|location|prediction)"
        r"[^()]*?\(\s*(\d+)\s*,\s*(\d+)\s*\)",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), int(m.group(2))

    # Pattern 4: row X column Y or row X, column Y
    m = re.search(
        r"row\s+(\d+)\s*[,]?\s*column\s+(\d+)",
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), int(m.group(2))

    # Pattern 5: two comma-separated numbers after keyword
    m = re.search(
        r"(?:waypoint|predict|output|answer)"
        r"[^0-9]*(\d{1,2})\s*[,]\s*(\d{1,2})",
        text,
        re.IGNORECASE,
    )
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return r, c

    return None


def _is_in_drivable(
    row: int, col: int, drivable_bins: list[list[int]]
) -> bool:
    """Check if (row, col) is in the drivable bin set."""
    return any(
        dr == row and dc == col
        for dr, dc in drivable_bins
    )


def _get_diverse_sample_indices(
    n: int, dataset_size: int = 8378
) -> list[int]:
    """Get evenly-spaced sample indices across the dataset."""
    step = dataset_size // n
    indices = [i * step for i in range(n)]
    return indices[:n]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _count_conversation_tokens(
    conversation: list[dict[str, Any]],
) -> dict[str, int]:
    """Count approximate tokens in a conversation."""
    input_tokens = 0
    output_tokens = 0
    image_tokens = 0
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            toks = _estimate_tokens(content)
        elif isinstance(content, list):
            toks = 0
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        toks += _estimate_tokens(
                            part.get("text", "")
                        )
                    elif part.get("type") == "image_url":
                        # typical VLM image token count
                        image_tokens += 1120
        else:
            toks = 0

        if role in ("user", "system", "tool"):
            input_tokens += toks
        elif role == "assistant":
            output_tokens += toks

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "image_tokens": image_tokens,
        "total_tokens": (
            input_tokens + output_tokens + image_tokens
        ),
    }


def _build_geo_summary(geo: dict[str, Any]) -> str:
    """Build a text summary of geometry analysis results."""
    bounds = geo.get("road_boundaries", {})
    left_x = bounds.get("left_x", "?")
    right_x = bounds.get("right_x", "?")
    n_bins = len(geo.get("drivable_bins", []))
    return (
        "Road geometry analysis results:\n"
        f"- Lanes detected: "
        f"{geo.get('num_lanes_detected', 0)}\n"
        f"- Road curvature: "
        f"{geo.get('road_curvature', 'unknown')}\n"
        f"- Vanishing point: "
        f"{geo.get('vanishing_point', 'not detected')}\n"
        f"- Road boundaries: "
        f"left_x={left_x}, right_x={right_x}\n"
        f"- Drivable region: "
        f"{geo.get('drivable_region', 'unknown')}\n"
        f"- Drivable bins (63x63 grid): {n_bins}"
    )


def _build_boundary_text(geo: dict[str, Any]) -> str:
    """Build road boundary text from geometry dict."""
    bounds = geo.get("road_boundaries", {})
    left_x = bounds.get("left_x", "?")
    right_x = bounds.get("right_x", "?")
    return f"left_x={left_x}, right_x={right_x}"


def _accum_tokens(
    total: dict[str, int],
    conv: dict[str, int],
) -> None:
    """Accumulate token counts in-place."""
    for k in total:
        total[k] += conv[k]


# ===============================================================
# Task 36: Geometry-guided waypoint prediction (2B, 100 samples)
# ===============================================================
def run_task36(server_url: str) -> dict[str, Any]:
    """Model gets image + road geometry, predicts waypoint."""
    _print_sep(
        "TASK 36: Geometry-Guided Waypoint "
        "Prediction (2B, 100 samples)"
    )

    sample_indices = _get_diverse_sample_indices(100)

    # Pre-compute geometry for all samples
    print("  Pre-computing road geometry for 100 samples...")
    geometry_cache: dict[int, dict[str, Any]] = {}
    for idx in sample_indices:
        img_path = load_sample_image(idx)
        geometry_cache[idx] = analyze_road_geometry(img_path)

    orch = ToolCallingOrchestrator(
        server_url=server_url,
        tools={},
        tool_definitions=[],
        max_tool_rounds=1,
        temperature=0,
        max_tokens=512,
    )

    results: list[dict[str, Any]] = []
    in_drivable_count = 0
    valid_waypoint_count = 0
    total_tokens: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 0,
    }

    for i, idx in enumerate(sample_indices):
        if i % 20 == 0:
            print(f"  Processing {i+1}/100 (sample={idx})...")

        img_path = load_sample_image(idx)
        geo = geometry_cache[idx]
        geo_summary = _build_geo_summary(geo)

        prompt = (
            f"{geo_summary}\n\n"
            "Use the road geometry analysis to predict "
            "where the vehicle should drive. Your waypoint "
            "must be within the drivable region. "
            "Output as row,col in a 63x63 grid.\n\n"
            "Format: FINAL_WAYPOINT: row, col"
        )

        result = orch.run(
            image_path=img_path,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            tool_choice="none",
        )

        wp = _parse_waypoint_from_text(result["final_text"])
        drivable_bins = geo.get("drivable_bins", [])
        in_drivable = False
        if wp is not None:
            valid_waypoint_count += 1
            row, col = wp
            in_drivable = _is_in_drivable(
                row, col, drivable_bins
            )
            if in_drivable:
                in_drivable_count += 1

        conv_tokens = _count_conversation_tokens(
            result.get("full_conversation", [])
        )
        _accum_tokens(total_tokens, conv_tokens)

        results.append({
            "sample_index": idx,
            "predicted_waypoint": list(wp) if wp else None,
            "in_drivable_region": in_drivable,
            "num_drivable_bins": len(drivable_bins),
            "road_curvature": geo.get(
                "road_curvature", "unknown"
            ),
            "num_lanes": geo.get("num_lanes_detected", 0),
            "final_text": result["final_text"],
            "latency_ms": result["latency_ms"],
            "tokens": conv_tokens,
            "error": result["error"],
        })

    drivable_rate = (
        in_drivable_count / valid_waypoint_count
        if valid_waypoint_count > 0
        else 0
    )

    summary = {
        "task": "task36_geometry_waypoint_prediction",
        "model": "2B",
        "total_samples": 100,
        "valid_waypoint_predictions": valid_waypoint_count,
        "waypoint_parse_rate": round(
            valid_waypoint_count / 100, 3
        ),
        "in_drivable_count": in_drivable_count,
        "in_drivable_rate": round(drivable_rate, 3),
        "total_tokens": total_tokens,
        "samples": results,
    }

    print("\n  --- Task 36 Summary ---")
    vwc = valid_waypoint_count
    print(f"  Valid waypoints parsed: {vwc}/100")
    print(
        f"  In drivable region: "
        f"{in_drivable_count}/{vwc} ({drivable_rate:.1%})"
    )
    print(f"  Total tokens: {total_tokens['total_tokens']}")
    return summary


# ===============================================================
# Task 37: Geometry tool accuracy audit (50 samples)
# ===============================================================
def run_task37() -> dict[str, Any]:
    """Audit the geometry tool itself -- no model involved."""
    _print_sep(
        "TASK 37: Geometry Tool Accuracy Audit (50 samples)"
    )

    sample_indices = _get_diverse_sample_indices(50, 8378)

    results: list[dict[str, Any]] = []
    non_trivial_count = 0
    has_vanishing_point = 0
    curvature_dist: dict[str, int] = {}
    lane_counts: list[int] = []
    total_lines: list[int] = []

    for i, idx in enumerate(sample_indices):
        if i % 10 == 0:
            print(
                f"  Processing {i+1}/50 (sample={idx})..."
            )

        img_path = load_sample_image(idx)
        geo = analyze_road_geometry(img_path)
        meta = load_sample_metadata(idx)

        num_lanes = geo.get("num_lanes_detected", 0)
        curvature = geo.get("road_curvature", "unknown")
        vp = geo.get("vanishing_point")
        n_lines = geo.get("total_lines_detected", 0)
        n_drivable = len(geo.get("drivable_bins", []))

        is_non_trivial = (
            num_lanes > 0 and curvature != "unknown"
        )
        if is_non_trivial:
            non_trivial_count += 1
        if vp is not None:
            has_vanishing_point += 1

        curvature_dist[curvature] = (
            curvature_dist.get(curvature, 0) + 1
        )
        lane_counts.append(num_lanes)
        total_lines.append(n_lines)

        results.append({
            "sample_index": idx,
            "num_lanes_detected": num_lanes,
            "road_curvature": curvature,
            "vanishing_point": vp,
            "road_boundaries": geo.get("road_boundaries"),
            "drivable_region": geo.get("drivable_region"),
            "num_drivable_bins": n_drivable,
            "total_lines_detected": n_lines,
            "is_non_trivial": is_non_trivial,
            "scene_type_gt": meta.get(
                "fine_class", "unknown"
            ),
            "odd_label": meta.get("odd_label", "unknown"),
        })

    import numpy as np  # type: ignore[import-not-found]

    avg_lanes = (
        float(np.mean(lane_counts)) if lane_counts else 0
    )
    avg_lines = (
        float(np.mean(total_lines)) if total_lines else 0
    )

    summary = {
        "task": "task37_geometry_tool_audit",
        "total_samples": 50,
        "non_trivial_count": non_trivial_count,
        "non_trivial_rate": round(
            non_trivial_count / 50, 3
        ),
        "has_vanishing_point_count": has_vanishing_point,
        "has_vanishing_point_rate": round(
            has_vanishing_point / 50, 3
        ),
        "curvature_distribution": curvature_dist,
        "avg_lanes_detected": round(avg_lanes, 2),
        "avg_total_lines": round(avg_lines, 2),
        "lane_count_distribution": {
            "0_lanes": sum(
                1 for c in lane_counts if c == 0
            ),
            "1-5_lanes": sum(
                1 for c in lane_counts if 1 <= c <= 5
            ),
            "6-10_lanes": sum(
                1 for c in lane_counts if 6 <= c <= 10
            ),
            "10+_lanes": sum(
                1 for c in lane_counts if c > 10
            ),
        },
        "samples": results,
    }

    nt = non_trivial_count
    print("\n  --- Task 37 Summary ---")
    print(f"  Non-trivial output: {nt}/50 ({nt/50:.0%})")
    print(
        f"  Vanishing point: {has_vanishing_point}/50"
    )
    print(f"  Curvature dist: {curvature_dist}")
    print(
        f"  Avg lanes: {avg_lanes:.1f}, "
        f"Avg lines: {avg_lines:.1f}"
    )
    return summary


# ===============================================================
# Task 38: Geometry + waypoint viz combined (2B, 100 samples)
# ===============================================================
def run_task38(server_url: str) -> dict[str, Any]:
    """Both geometry and waypoint viz tools available."""
    _print_sep(
        "TASK 38: Geometry + Waypoint Viz "
        "Combined (2B, 100 samples)"
    )

    sample_indices = _get_diverse_sample_indices(100)

    orch = ToolCallingOrchestrator(
        server_url=server_url,
        tools={
            "analyze_road_geometry": analyze_road_geometry,
            "visualize_waypoint": visualize_waypoint,
        },
        tool_definitions=[
            TOOL_ROAD_GEOMETRY, TOOL_WAYPOINT_VIZ,
        ],
        max_tool_rounds=5,
        temperature=0,
        max_tokens=1024,
    )

    results: list[dict[str, Any]] = []
    used_geometry = 0
    used_viz = 0
    used_both = 0
    in_drivable_count = 0
    valid_waypoint_count = 0
    total_tokens: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 0,
    }

    for i, idx in enumerate(sample_indices):
        if i % 20 == 0:
            print(
                f"  Processing {i+1}/100 (sample={idx})..."
            )

        img_path = load_sample_image(idx)

        prompt = (
            "Analyze road geometry, then predict and "
            "visualize a waypoint. Verify it falls on "
            "the road. Output your final waypoint as:\n"
            "FINAL_WAYPOINT: row, col"
        )

        result = orch.run(
            image_path=img_path,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        tools_used = set(
            tc["tool_name"] for tc in result["tool_calls"]
        )
        did_geo = "analyze_road_geometry" in tools_used
        did_viz = "visualize_waypoint" in tools_used
        if did_geo:
            used_geometry += 1
        if did_viz:
            used_viz += 1
        if did_geo and did_viz:
            used_both += 1

        wp = _parse_waypoint_from_text(result["final_text"])

        geo = analyze_road_geometry(img_path)
        drivable_bins = geo.get("drivable_bins", [])
        in_drivable = False
        if wp is not None:
            valid_waypoint_count += 1
            row, col = wp
            in_drivable = _is_in_drivable(
                row, col, drivable_bins
            )
            if in_drivable:
                in_drivable_count += 1

        conv_tokens = _count_conversation_tokens(
            result.get("full_conversation", [])
        )
        _accum_tokens(total_tokens, conv_tokens)

        results.append({
            "sample_index": idx,
            "tools_used": list(tools_used),
            "used_geometry": did_geo,
            "used_viz": did_viz,
            "predicted_waypoint": (
                list(wp) if wp else None
            ),
            "in_drivable_region": in_drivable,
            "num_drivable_bins": len(drivable_bins),
            "num_tool_calls": result["num_tool_calls"],
            "num_rounds": result["num_rounds"],
            "final_text": result["final_text"],
            "latency_ms": result["latency_ms"],
            "tokens": conv_tokens,
            "error": result["error"],
        })

    drivable_rate = (
        in_drivable_count / valid_waypoint_count
        if valid_waypoint_count > 0
        else 0
    )

    summary = {
        "task": "task38_geometry_plus_viz",
        "model": "2B",
        "total_samples": 100,
        "used_geometry_count": used_geometry,
        "used_viz_count": used_viz,
        "used_both_count": used_both,
        "valid_waypoint_predictions": valid_waypoint_count,
        "waypoint_parse_rate": round(
            valid_waypoint_count / 100, 3
        ),
        "in_drivable_count": in_drivable_count,
        "in_drivable_rate": round(drivable_rate, 3),
        "total_tokens": total_tokens,
        "samples": results,
    }

    vwc = valid_waypoint_count
    print("\n  --- Task 38 Summary ---")
    print(f"  Used geometry: {used_geometry}/100")
    print(f"  Used viz: {used_viz}/100")
    print(f"  Used both: {used_both}/100")
    print(f"  Valid waypoints: {vwc}/100")
    print(
        f"  In drivable: "
        f"{in_drivable_count}/{vwc} ({drivable_rate:.1%})"
    )
    return summary


# ===============================================================
# Task 39: Curvature-action consistency (2B, 100 samples)
# ===============================================================
def run_task39(server_url: str) -> dict[str, Any]:
    """Give model geometry text + image, predict actions."""
    _print_sep(
        "TASK 39: Curvature-Action Consistency "
        "(2B, 100 samples)"
    )

    sample_indices = _get_diverse_sample_indices(100)

    orch = ToolCallingOrchestrator(
        server_url=server_url,
        tools={},
        tool_definitions=[],
        max_tool_rounds=1,
        temperature=0,
        max_tokens=512,
    )

    results: list[dict[str, Any]] = []
    long_correct = 0
    lat_correct = 0
    total_with_gt = 0
    curvature_action_map: dict[str, dict[str, int]] = {}
    total_tokens: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 0,
    }

    for i, idx in enumerate(sample_indices):
        if i % 20 == 0:
            print(
                f"  Processing {i+1}/100 (sample={idx})..."
            )

        img_path = load_sample_image(idx)
        meta = load_sample_metadata(idx)

        geo = analyze_road_geometry(img_path)
        curvature = geo.get("road_curvature", "unknown")
        num_lanes = geo.get("num_lanes_detected", 0)
        bnd = _build_boundary_text(geo)

        prompt = (
            "The road geometry shows:\n"
            f"- Curvature: {curvature}\n"
            f"- Lanes detected: {num_lanes}\n"
            f"- Road boundaries: {bnd}\n\n"
            "What driving action is appropriate?\n"
            "Choose longitudinal: null, stop, "
            "slowdown, proceed\n"
            "Choose lateral: null, lc_left, lc_right\n\n"
            "Output format:\n"
            "FINAL_LONG_ACTION: <action>\n"
            "FINAL_LAT_ACTION: <action>"
        )

        result = orch.run(
            image_path=img_path,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            tool_choice="none",
        )

        pred = parse_prediction(result["final_text"])
        pred_long = pred.get("long_action")
        pred_lat = pred.get("lat_action")

        gt_long = meta.get("long_action", "null")
        gt_lat = meta.get("lat_action", "null")

        has_gt = gt_long is not None
        if has_gt:
            total_with_gt += 1
            if pred_long == gt_long:
                long_correct += 1
            if pred_lat == gt_lat:
                lat_correct += 1

        if curvature not in curvature_action_map:
            curvature_action_map[curvature] = {}
        akey = (
            f"{pred_long or 'none'}|{pred_lat or 'none'}"
        )
        curvature_action_map[curvature][akey] = (
            curvature_action_map[curvature].get(akey, 0) + 1
        )

        conv_tokens = _count_conversation_tokens(
            result.get("full_conversation", [])
        )
        _accum_tokens(total_tokens, conv_tokens)

        results.append({
            "sample_index": idx,
            "curvature": curvature,
            "num_lanes": num_lanes,
            "predicted_long_action": pred_long,
            "predicted_lat_action": pred_lat,
            "gt_long_action": gt_long,
            "gt_lat_action": gt_lat,
            "long_correct": (
                pred_long == gt_long if has_gt else None
            ),
            "lat_correct": (
                pred_lat == gt_lat if has_gt else None
            ),
            "final_text": result["final_text"],
            "latency_ms": result["latency_ms"],
            "tokens": conv_tokens,
            "error": result["error"],
        })

    long_acc = (
        long_correct / total_with_gt
        if total_with_gt > 0
        else 0
    )
    lat_acc = (
        lat_correct / total_with_gt
        if total_with_gt > 0
        else 0
    )

    summary = {
        "task": "task39_curvature_action_consistency",
        "model": "2B",
        "total_samples": 100,
        "total_with_gt": total_with_gt,
        "long_action_correct": long_correct,
        "long_action_accuracy": round(long_acc, 3),
        "lat_action_correct": lat_correct,
        "lat_action_accuracy": round(lat_acc, 3),
        "curvature_action_map": curvature_action_map,
        "total_tokens": total_tokens,
        "samples": results,
    }

    twg = total_with_gt
    print("\n  --- Task 39 Summary ---")
    print(
        f"  Long accuracy: "
        f"{long_correct}/{twg} ({long_acc:.1%})"
    )
    print(
        f"  Lat accuracy: "
        f"{lat_correct}/{twg} ({lat_acc:.1%})"
    )
    print("  Curvature -> action map:")
    for curv, actions in curvature_action_map.items():
        print(f"    {curv}: {actions}")
    return summary


# ===============================================================
# Task 40: Geometry on difficult images (2B, 50 samples)
# ===============================================================
def run_task40(server_url: str) -> dict[str, Any]:
    """Select images where geometry tool gives poor results."""
    _print_sep(
        "TASK 40: Geometry on Difficult Images "
        "(2B, 50 samples)"
    )

    print("  Scanning 500 samples for difficult cases...")
    scan_indices = _get_diverse_sample_indices(500, 8378)
    difficult_indices: list[int] = []
    difficult_geo: dict[int, dict[str, Any]] = {}

    for idx in scan_indices:
        img_path = load_sample_image(idx)
        geo = analyze_road_geometry(img_path)

        num_lanes = geo.get("num_lanes_detected", 0)
        vp = geo.get("vanishing_point")
        n_lines = geo.get("total_lines_detected", 0)

        is_diff = (
            (num_lanes <= 1)
            or (vp is None)
            or (n_lines <= 3)
        )
        if is_diff:
            difficult_indices.append(idx)
            difficult_geo[idx] = geo

        if len(difficult_indices) >= 50:
            break

    n_found = len(difficult_indices)
    print(f"  Found {n_found} difficult samples")
    if n_found < 50:
        print(f"  Warning: only found {n_found}")

    selected = difficult_indices[:50]

    orch = ToolCallingOrchestrator(
        server_url=server_url,
        tools={
            "analyze_road_geometry": analyze_road_geometry,
        },
        tool_definitions=[TOOL_ROAD_GEOMETRY],
        max_tool_rounds=3,
        temperature=0,
        max_tokens=1024,
    )

    results: list[dict[str, Any]] = []
    recognizes_unreliable = 0
    blindly_trusts = 0
    uses_tool = 0
    scene_correct = 0
    total_with_scene_gt = 0
    total_tokens: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 0,
    }

    for i, idx in enumerate(selected):
        if i % 10 == 0:
            n_sel = len(selected)
            print(
                f"  Processing {i+1}/{n_sel} "
                f"(sample={idx})..."
            )

        img_path = load_sample_image(idx)
        meta = load_sample_metadata(idx)
        geo = difficult_geo.get(
            idx, analyze_road_geometry(img_path)
        )

        prompt = (
            "Analyze this driving scene. Use the road "
            "geometry tool if helpful. Classify the scene "
            "and assess the road conditions.\n\n"
            "Output format:\n"
            "FINAL_SCENE: <scene_type>\n"
            "FINAL_LONG_ACTION: <action>\n"
            "FINAL_LAT_ACTION: <action>\n"
            "Also note if the geometry analysis seems "
            "reliable or not."
        )

        result = orch.run(
            image_path=img_path,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        final = result["final_text"].lower()
        pred = parse_prediction(result["final_text"])

        tool_called = any(
            tc["tool_name"] == "analyze_road_geometry"
            for tc in result["tool_calls"]
        )
        if tool_called:
            uses_tool += 1

        unreliable_kws = [
            "unreliable", "limited", "not reliable",
            "few lanes", "no lanes", "unclear",
            "inaccurate", "imprecise",
            "unable to detect", "poor",
            "difficult", "challenging",
            "not detected", "cannot determine",
            "low confidence", "insufficient",
            "not enough", "ambiguous", "uncertain",
            "caution", "take with", "grain of salt",
        ]
        recognized = any(kw in final for kw in unreliable_kws)
        if recognized:
            recognizes_unreliable += 1

        trust_kws = [
            "the analysis shows",
            "according to the geometry",
            "the tool confirms",
            "as detected by",
            "the geometry indicates",
        ]
        trusts = (
            any(kw in final for kw in trust_kws)
            and not recognized
        )
        if trusts:
            blindly_trusts += 1

        gt_scene = meta.get("fine_class", "unknown")
        gt_scene_type = _fine_class_to_scene_type(gt_scene)
        if gt_scene_type != "unknown":
            total_with_scene_gt += 1
            if pred.get("scene") == gt_scene_type:
                scene_correct += 1

        conv_tokens = _count_conversation_tokens(
            result.get("full_conversation", [])
        )
        _accum_tokens(total_tokens, conv_tokens)

        results.append({
            "sample_index": idx,
            "tool_called": tool_called,
            "geo_num_lanes": geo.get(
                "num_lanes_detected", 0
            ),
            "geo_vanishing_point": geo.get(
                "vanishing_point"
            ),
            "geo_total_lines": geo.get(
                "total_lines_detected", 0
            ),
            "recognized_unreliable": recognized,
            "blindly_trusts": trusts,
            "predicted_scene": pred.get("scene"),
            "gt_scene_type": gt_scene_type,
            "scene_correct": (
                pred.get("scene") == gt_scene_type
            ),
            "final_text": result["final_text"],
            "latency_ms": result["latency_ms"],
            "tokens": conv_tokens,
            "error": result["error"],
        })

    n_sel = len(selected)
    unreliable_rate = (
        recognizes_unreliable / n_sel if n_sel > 0 else 0
    )
    trust_rate = (
        blindly_trusts / n_sel if n_sel > 0 else 0
    )
    scene_acc = (
        scene_correct / total_with_scene_gt
        if total_with_scene_gt > 0
        else 0
    )

    summary = {
        "task": "task40_difficult_geometry",
        "model": "2B",
        "total_samples": n_sel,
        "uses_tool_count": uses_tool,
        "recognizes_unreliable_count": recognizes_unreliable,
        "recognizes_unreliable_rate": round(
            unreliable_rate, 3
        ),
        "blindly_trusts_count": blindly_trusts,
        "blindly_trusts_rate": round(trust_rate, 3),
        "scene_correct_count": scene_correct,
        "scene_accuracy": round(scene_acc, 3),
        "total_with_scene_gt": total_with_scene_gt,
        "total_tokens": total_tokens,
        "samples": results,
    }

    ru = recognizes_unreliable
    bt = blindly_trusts
    sc = scene_correct
    tsg = total_with_scene_gt
    print("\n  --- Task 40 Summary ---")
    print(f"  Used tool: {uses_tool}/{n_sel}")
    print(
        f"  Recognized unreliable: "
        f"{ru}/{n_sel} ({unreliable_rate:.1%})"
    )
    print(
        f"  Blindly trusts: "
        f"{bt}/{n_sel} ({trust_rate:.1%})"
    )
    print(
        f"  Scene accuracy: "
        f"{sc}/{tsg} ({scene_acc:.1%})"
    )
    return summary


# ===============================================================
# Task 41: Geometry with 8B (100 samples, same as Task 36)
# ===============================================================
def run_task41(server_url: str) -> dict[str, Any]:
    """Same as Task 36 but with 8B model for comparison."""
    _print_sep(
        "TASK 41: Geometry-Guided Waypoint "
        "Prediction (8B, 100 samples)"
    )

    sample_indices = _get_diverse_sample_indices(100)

    print("  Pre-computing road geometry for 100 samples...")
    geometry_cache: dict[int, dict[str, Any]] = {}
    for idx in sample_indices:
        img_path = load_sample_image(idx)
        geometry_cache[idx] = analyze_road_geometry(img_path)

    orch = ToolCallingOrchestrator(
        server_url=server_url,
        tools={},
        tool_definitions=[],
        max_tool_rounds=1,
        temperature=0,
        max_tokens=512,
    )

    results: list[dict[str, Any]] = []
    in_drivable_count = 0
    valid_waypoint_count = 0
    total_tokens: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "image_tokens": 0,
        "total_tokens": 0,
    }

    for i, idx in enumerate(sample_indices):
        if i % 20 == 0:
            print(
                f"  Processing {i+1}/100 (sample={idx})..."
            )

        img_path = load_sample_image(idx)
        geo = geometry_cache[idx]
        geo_summary = _build_geo_summary(geo)

        prompt = (
            f"{geo_summary}\n\n"
            "Use the road geometry analysis to predict "
            "where the vehicle should drive. Your waypoint "
            "must be within the drivable region. "
            "Output as row,col in a 63x63 grid.\n\n"
            "Format: FINAL_WAYPOINT: row, col"
        )

        result = orch.run(
            image_path=img_path,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            tool_choice="none",
        )

        wp = _parse_waypoint_from_text(result["final_text"])
        drivable_bins = geo.get("drivable_bins", [])
        in_drivable = False
        if wp is not None:
            valid_waypoint_count += 1
            row, col = wp
            in_drivable = _is_in_drivable(
                row, col, drivable_bins
            )
            if in_drivable:
                in_drivable_count += 1

        conv_tokens = _count_conversation_tokens(
            result.get("full_conversation", [])
        )
        _accum_tokens(total_tokens, conv_tokens)

        results.append({
            "sample_index": idx,
            "predicted_waypoint": list(wp) if wp else None,
            "in_drivable_region": in_drivable,
            "num_drivable_bins": len(drivable_bins),
            "road_curvature": geo.get(
                "road_curvature", "unknown"
            ),
            "num_lanes": geo.get("num_lanes_detected", 0),
            "final_text": result["final_text"],
            "latency_ms": result["latency_ms"],
            "tokens": conv_tokens,
            "error": result["error"],
        })

    drivable_rate = (
        in_drivable_count / valid_waypoint_count
        if valid_waypoint_count > 0
        else 0
    )

    summary = {
        "task": "task41_geometry_waypoint_8b",
        "model": "8B",
        "total_samples": 100,
        "valid_waypoint_predictions": valid_waypoint_count,
        "waypoint_parse_rate": round(
            valid_waypoint_count / 100, 3
        ),
        "in_drivable_count": in_drivable_count,
        "in_drivable_rate": round(drivable_rate, 3),
        "total_tokens": total_tokens,
        "samples": results,
    }

    vwc = valid_waypoint_count
    print("\n  --- Task 41 Summary ---")
    print(f"  Valid waypoints: {vwc}/100")
    print(
        f"  In drivable: "
        f"{in_drivable_count}/{vwc} ({drivable_rate:.1%})"
    )
    return summary


# ===============================================================
# Task 42: Token cost analysis
# ===============================================================
def _avg_per_sample(
    total: int, n: int,
) -> float:
    """Compute average per sample, return 0 if n==0."""
    return round(total / n, 1) if n > 0 else 0


def run_task42(all_results: dict[str, Any]) -> dict[str, Any]:
    """Compile token costs from all geometry experiments."""
    _print_sep("TASK 42: Token Cost Analysis")

    task_costs: dict[str, Any] = {}
    task_keys = [
        "task36", "task37", "task38",
        "task39", "task40", "task41",
    ]

    for task_key in task_keys:
        task_data = all_results.get(task_key)
        if task_data is None:
            continue

        total_input = 0
        total_output = 0
        total_image = 0
        total_all = 0
        latencies: list[float] = []

        st = task_data.get("total_tokens")
        if st:
            total_input = st.get("input_tokens", 0)
            total_output = st.get("output_tokens", 0)
            total_image = st.get("image_tokens", 0)
            total_all = st.get("total_tokens", 0)

        samples = task_data.get("samples", [])
        n_samples = len(samples)
        for s in samples:
            latencies.append(s.get("latency_ms", 0))

        avg_lat = (
            sum(latencies) / len(latencies)
            if latencies
            else 0
        )

        task_costs[task_key] = {
            "task_name": task_data.get("task", task_key),
            "model": task_data.get("model", "N/A"),
            "num_samples": n_samples,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_image_tokens": total_image,
            "total_tokens": total_all,
            "avg_input_per_sample": _avg_per_sample(
                total_input, n_samples
            ),
            "avg_output_per_sample": _avg_per_sample(
                total_output, n_samples
            ),
            "avg_image_per_sample": _avg_per_sample(
                total_image, n_samples
            ),
            "avg_total_per_sample": _avg_per_sample(
                total_all, n_samples
            ),
            "avg_latency_ms": round(avg_lat, 1),
            "total_latency_ms": round(sum(latencies), 1),
        }

    grand_tokens = sum(
        tc.get("total_tokens", 0)
        for tc in task_costs.values()
    )
    grand_input = sum(
        tc.get("total_input_tokens", 0)
        for tc in task_costs.values()
    )
    grand_output = sum(
        tc.get("total_output_tokens", 0)
        for tc in task_costs.values()
    )
    grand_image = sum(
        tc.get("total_image_tokens", 0)
        for tc in task_costs.values()
    )
    grand_latency = sum(
        tc.get("total_latency_ms", 0)
        for tc in task_costs.values()
    )

    summary = {
        "task": "task42_token_cost_analysis",
        "per_task_costs": task_costs,
        "grand_totals": {
            "total_input_tokens": grand_input,
            "total_output_tokens": grand_output,
            "total_image_tokens": grand_image,
            "total_tokens": grand_tokens,
            "total_latency_ms": round(grand_latency, 1),
            "total_latency_minutes": round(
                grand_latency / 60000, 2
            ),
        },
    }

    print("\n  --- Task 42 Token Cost Summary ---")
    for tk, tc in task_costs.items():
        m = tc["model"]
        tt = tc["total_tokens"]
        avg = tc["avg_total_per_sample"]
        lat = tc["avg_latency_ms"]
        print(
            f"  {tk} ({m}): {tt} tokens, "
            f"{avg} avg/sample, {lat:.0f}ms avg"
        )
    print(f"\n  Grand total: {grand_tokens} tokens")
    gm = grand_latency / 60000
    print(f"  Grand latency: {gm:.1f} minutes")
    return summary


# ===============================================================
# Main
# ===============================================================
def main() -> None:
    start_time = time.time()
    all_results: dict[str, Any] = {}

    print("=" * 72)
    print("  Phase 2: Road Geometry Deep Dive (Tasks 36-42)")
    print("=" * 72)

    # --- Start servers ---
    print("\nStarting 2B server on GPU 4, port 8334...")
    server_2b = VLLMServer(
        model_path=MODEL_2B,
        port=PORT_2B,
        gpu_id=GPU_2B,
        max_model_len=8192,
        enable_tools=True,
    )
    server_2b.start(timeout=360)
    url_2b = f"http://localhost:{PORT_2B}"

    try:
        # Task 36
        try:
            all_results["task36"] = run_task36(url_2b)
        except Exception as e:
            print(f"  Task 36 failed: {e}")
            import traceback
            traceback.print_exc()
            all_results["task36"] = {"error": str(e)}

        # Task 37 (no model)
        try:
            all_results["task37"] = run_task37()
        except Exception as e:
            print(f"  Task 37 failed: {e}")
            import traceback
            traceback.print_exc()
            all_results["task37"] = {"error": str(e)}

        # Task 38
        try:
            all_results["task38"] = run_task38(url_2b)
        except Exception as e:
            print(f"  Task 38 failed: {e}")
            import traceback
            traceback.print_exc()
            all_results["task38"] = {"error": str(e)}

        # Task 39
        try:
            all_results["task39"] = run_task39(url_2b)
        except Exception as e:
            print(f"  Task 39 failed: {e}")
            import traceback
            traceback.print_exc()
            all_results["task39"] = {"error": str(e)}

        # Task 40
        try:
            all_results["task40"] = run_task40(url_2b)
        except Exception as e:
            print(f"  Task 40 failed: {e}")
            import traceback
            traceback.print_exc()
            all_results["task40"] = {"error": str(e)}

    finally:
        print("\nStopping 2B server...")
        server_2b.stop()
        time.sleep(5)

    # --- Start 8B server ---
    print("\nStarting 8B server on GPU 5, port 8335...")
    server_8b = VLLMServer(
        model_path=MODEL_8B,
        port=PORT_8B,
        gpu_id=GPU_8B,
        max_model_len=8192,
        enable_tools=True,
    )
    server_8b.start(timeout=360)
    url_8b = f"http://localhost:{PORT_8B}"

    try:
        # Task 41
        try:
            all_results["task41"] = run_task41(url_8b)
        except Exception as e:
            print(f"  Task 41 failed: {e}")
            import traceback
            traceback.print_exc()
            all_results["task41"] = {"error": str(e)}

    finally:
        print("\nStopping 8B server...")
        server_8b.stop()

    # Task 42
    try:
        all_results["task42"] = run_task42(all_results)
    except Exception as e:
        print(f"  Task 42 failed: {e}")
        all_results["task42"] = {"error": str(e)}

    # --- Save results ---
    elapsed = time.time() - start_time
    all_results["metadata"] = {
        "phase": "Phase 2: Road Geometry Deep Dive",
        "tasks": "36-42",
        "total_elapsed_seconds": round(elapsed, 1),
        "total_elapsed_minutes": round(elapsed / 60, 1),
        "model_2b": MODEL_2B,
        "model_8b": MODEL_8B,
    }

    # 2B vs 8B comparison
    t36 = all_results.get("task36", {})
    t41 = all_results.get("task41", {})
    if not isinstance(t36, dict) or "error" in t36:
        t36 = {}
    if not isinstance(t41, dict) or "error" in t41:
        t41 = {}

    all_results["comparison_2b_vs_8b"] = {
        "task36_2b_parse_rate": t36.get(
            "waypoint_parse_rate", 0
        ),
        "task41_8b_parse_rate": t41.get(
            "waypoint_parse_rate", 0
        ),
        "task36_2b_drivable_rate": t36.get(
            "in_drivable_rate", 0
        ),
        "task41_8b_drivable_rate": t41.get(
            "in_drivable_rate", 0
        ),
        "task36_2b_valid_wps": t36.get(
            "valid_waypoint_predictions", 0
        ),
        "task41_8b_valid_wps": t41.get(
            "valid_waypoint_predictions", 0
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    em = elapsed / 60
    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_PATH}")
    print(f"  Total time: {em:.1f} minutes")
    print(f"{'=' * 72}")

    # Final summary
    _print_final_summary(all_results)


def _print_final_summary(
    all_results: dict[str, Any],
) -> None:
    """Print the final summary of all tasks."""
    print("\n  === PHASE 2 FINAL SUMMARY ===\n")

    metric_keys = [
        "waypoint_parse_rate",
        "in_drivable_rate",
        "non_trivial_rate",
        "has_vanishing_point_rate",
        "long_action_accuracy",
        "lat_action_accuracy",
        "recognizes_unreliable_rate",
        "scene_accuracy",
    ]

    task_keys = [
        "task36", "task37", "task38",
        "task39", "task40", "task41",
    ]

    for key in task_keys:
        data = all_results.get(key, {})
        if isinstance(data, dict) and "error" not in data:
            tn = data.get("task", key)
            model = data.get("model", "N/A")
            print(f"  {tn} ({model}):")
            for metric in metric_keys:
                val = data.get(metric)
                if val is not None:
                    print(f"    {metric}: {val:.1%}")
        else:
            print(f"  {key}: ERROR")

    comp = all_results.get("comparison_2b_vs_8b", {})
    if comp:
        pr2 = comp.get("task36_2b_parse_rate", 0)
        pr8 = comp.get("task41_8b_parse_rate", 0)
        dr2 = comp.get("task36_2b_drivable_rate", 0)
        dr8 = comp.get("task41_8b_drivable_rate", 0)
        print("\n  2B vs 8B Waypoint Comparison:")
        print(f"    2B parse rate: {pr2:.1%}")
        print(f"    8B parse rate: {pr8:.1%}")
        print(f"    2B drivable rate: {dr2:.1%}")
        print(f"    8B drivable rate: {dr8:.1%}")


if __name__ == "__main__":
    main()
