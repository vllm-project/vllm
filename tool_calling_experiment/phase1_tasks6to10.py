#!/usr/bin/env python3
"""Phase 1 Mechanical Validation -- Tasks 6-10.

Task 6:  Base 8B tool calling comparison (vs 2B from Tasks 1-3)
Task 7:  Max tool rounds -- how many tools does 2B naturally call?
Task 8:  Image injection verification -- does the model SEE zoomed images?
Task 9:  Waypoint visualization comprehension -- bad waypoint recognition
Task 10: Road geometry tool output comprehension

All tasks use 10 samples, the ToolCallingOrchestrator, and visual tools.
Results are saved to phase1_tasks6to10_results.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Path setup -- keep tool_calling_experiment importable without
# shadowing the installed vllm package.
# ---------------------------------------------------------------------------
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

from orchestrator import ToolCallingOrchestrator  # noqa: E402, I001  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
from server_utils import VLLMServer  # noqa: E402, I001  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
from visual_tools import (  # noqa: E402, I001  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
    TOOL_ROAD_GEOMETRY,
    TOOL_WAYPOINT_VIZ,
    TOOL_ZOOM,
    analyze_road_geometry,
    load_sample_image,
    visualize_waypoint,
    zoom_region,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
PORT_2B = 8322
PORT_8B = 8323
GPU_2B = 2
GPU_8B = 3

# Same 10 sample indices used in Tasks 1-3
SAMPLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
N_SAMPLES = len(SAMPLE_INDICES)

RESULTS_PATH = os.path.join(
    EXPERIMENT_DIR, "phase1_tasks6to10_results.json"
)

# Available tools (excluding find_similar_scenes -- no FAISS index)
AVAILABLE_TOOLS_DEFS = [
    TOOL_ZOOM, TOOL_WAYPOINT_VIZ, TOOL_ROAD_GEOMETRY,
]
AVAILABLE_TOOLS_FNS = {
    "zoom_region": zoom_region,
    "visualize_waypoint": visualize_waypoint,
    "analyze_road_geometry": analyze_road_geometry,
}


def _print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"


def _is_tool_call_valid(
    name: str, args: dict[str, Any],
) -> int:
    """Return 1 if tool call arguments are valid, 0 otherwise."""
    if name == "zoom_region" and not (
        isinstance(args.get("center_x"), (int, float))
        and isinstance(args.get("center_y"), (int, float))
    ):
        return 0
    if name == "visualize_waypoint" and not (
        isinstance(args.get("waypoint_row"), (int, float))
        and isinstance(args.get("waypoint_col"), (int, float))
    ):
        return 0
    # analyze_road_geometry has no required args -- always valid
    return 1


# ===================================================================
# Task 6: Base 8B tool calling comparison
# ===================================================================
def run_task6(server_8b_url: str) -> dict[str, Any]:
    """Run 8B model with all available tools on 10 samples."""
    _print_separator("TASK 6: Base 8B Tool Calling Comparison")

    orch = ToolCallingOrchestrator(
        server_url=server_8b_url,
        tools=AVAILABLE_TOOLS_FNS,
        tool_definitions=AVAILABLE_TOOLS_DEFS,
        max_tool_rounds=5,
        temperature=0,
        max_tokens=1024,
    )

    prompt = (
        "Analyze this driving scene. You have tools to zoom in, "
        "visualize waypoints, and analyze road geometry. "
        "Use any tools you find helpful, then provide your "
        "assessment."
    )

    results: list[dict[str, Any]] = []
    total_tool_calls = 0
    valid_args_count = 0
    total_args_count = 0
    references_tool_output = 0

    for i, idx in enumerate(SAMPLE_INDICES):
        print(f"  Sample {i+1}/{N_SAMPLES} (index={idx})...")
        img_path = load_sample_image(idx)

        result = orch.run(
            image_path=img_path,
            system_prompt="You are an expert driving scene analyst.",
            user_prompt=prompt,
        )

        # Analyze tool calling quality
        n_calls = result["num_tool_calls"]
        total_tool_calls += n_calls

        # Check argument validity for each tool call
        for tc in result["tool_calls"]:
            total_args_count += 1
            valid_args_count += _is_tool_call_valid(
                tc["tool_name"], tc["arguments"]
            )

        # Check if final text references tool output
        final = result["final_text"].lower()
        tool_ref_keywords = [
            "zoom", "zoomed", "closer look", "magnified",
            "waypoint", "marker", "visualiz",
            "lane", "curvature", "geometry", "drivable",
            "tool", "analysis result",
        ]
        if any(kw in final for kw in tool_ref_keywords):
            references_tool_output += 1

        sample_result = {
            "sample_index": idx,
            "num_tool_calls": n_calls,
            "tools_used": [
                tc["tool_name"] for tc in result["tool_calls"]
            ],
            "num_rounds": result["num_rounds"],
            "final_text": result["final_text"],
            "final_prediction": result["final_prediction"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "full_conversation": result["full_conversation"],
        }
        results.append(sample_result)

        rounds = result["num_rounds"]
        print(f"    Tools called: {n_calls}, Rounds: {rounds}")
        print(f"    Tools: {sample_result['tools_used']}")
        print(f"    Final: {_truncate(result['final_text'], 200)}")
        if result["error"]:
            print(f"    ERROR: {result['error']}")

    # Summary
    arg_validity_rate = (
        valid_args_count / total_args_count
        if total_args_count > 0
        else 0
    )
    ref_rate = references_tool_output / N_SAMPLES

    summary = {
        "task": "task6_8b_comparison",
        "model": "8B",
        "total_samples": N_SAMPLES,
        "total_tool_calls": total_tool_calls,
        "avg_tool_calls": round(total_tool_calls / N_SAMPLES, 2),
        "argument_validity_rate": round(arg_validity_rate, 3),
        "valid_args": valid_args_count,
        "total_args": total_args_count,
        "references_tool_output_count": references_tool_output,
        "references_tool_output_rate": round(ref_rate, 3),
        "samples": results,
    }

    print("\n  --- Task 6 Summary ---")
    print(f"  Total tool calls: {total_tool_calls}")
    avg_tc = summary["avg_tool_calls"]
    print(f"  Avg tool calls/sample: {avg_tc}")
    print(f"  Argument validity rate: {arg_validity_rate:.1%}")
    refs = references_tool_output
    print(f"  Refs tool output: {refs}/{N_SAMPLES} ({ref_rate:.0%})")
    return summary


# ===================================================================
# Task 7: Max tool rounds
# ===================================================================
def run_task7(server_2b_url: str) -> dict[str, Any]:
    """Test how many tools 2B naturally calls with high limit."""
    _print_separator("TASK 7: Max Tool Rounds (2B, limit=8)")

    orch = ToolCallingOrchestrator(
        server_url=server_2b_url,
        tools=AVAILABLE_TOOLS_FNS,
        tool_definitions=AVAILABLE_TOOLS_DEFS,
        max_tool_rounds=8,
        temperature=0,
        max_tokens=1024,
    )

    prompt = (
        "Thoroughly analyze this driving scene. You can use any "
        "tools as many times as needed. When you're confident in "
        "your assessment, provide your final prediction."
    )

    sys_prompt = (
        "You are an expert driving scene analyst with access "
        "to visual analysis tools."
    )

    results: list[dict[str, Any]] = []
    rounds_list: list[int] = []
    calls_list: list[int] = []
    hit_limit_count = 0

    for i, idx in enumerate(SAMPLE_INDICES):
        print(f"  Sample {i+1}/{N_SAMPLES} (index={idx})...")
        img_path = load_sample_image(idx)

        result = orch.run(
            image_path=img_path,
            system_prompt=sys_prompt,
            user_prompt=prompt,
        )

        n_rounds = result["num_rounds"]
        n_calls = result["num_tool_calls"]
        hit_limit = n_rounds >= 8
        if hit_limit:
            hit_limit_count += 1

        rounds_list.append(n_rounds)
        calls_list.append(n_calls)

        # Track which tools were called and in what order
        tool_sequence = [
            tc["tool_name"] for tc in result["tool_calls"]
        ]

        sample_result = {
            "sample_index": idx,
            "num_rounds": n_rounds,
            "num_tool_calls": n_calls,
            "hit_limit": hit_limit,
            "tool_sequence": tool_sequence,
            "final_text": result["final_text"],
            "final_prediction": result["final_prediction"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "full_conversation": result["full_conversation"],
        }
        results.append(sample_result)

        status = "HIT LIMIT" if hit_limit else "stopped naturally"
        print(f"    Rounds: {n_rounds}, Calls: {n_calls} ({status})")
        print(f"    Tool sequence: {tool_sequence}")
        if result["error"]:
            print(f"    ERROR: {result['error']}")

    avg_rounds = sum(rounds_list) / N_SAMPLES
    avg_calls = sum(calls_list) / N_SAMPLES

    summary = {
        "task": "task7_max_tool_rounds",
        "model": "2B",
        "max_tool_rounds_limit": 8,
        "total_samples": N_SAMPLES,
        "avg_rounds": round(avg_rounds, 2),
        "avg_tool_calls": round(avg_calls, 2),
        "min_rounds": min(rounds_list),
        "max_rounds": max(rounds_list),
        "hit_limit_count": hit_limit_count,
        "hit_limit_rate": round(hit_limit_count / N_SAMPLES, 3),
        "rounds_distribution": rounds_list,
        "calls_distribution": calls_list,
        "samples": results,
    }

    print("\n  --- Task 7 Summary ---")
    print(f"  Avg rounds: {avg_rounds:.1f}")
    print(f"  Avg tool calls: {avg_calls:.1f}")
    rng = f"{min(rounds_list)}-{max(rounds_list)}"
    print(f"  Rounds range: {rng}")
    print(f"  Hit limit (8): {hit_limit_count}/{N_SAMPLES}")
    return summary


# ===================================================================
# Task 8: Image injection verification (MOST IMPORTANT)
# ===================================================================
def run_task8(server_2b_url: str) -> dict[str, Any]:
    """Verify model actually SEES zoomed images."""
    _print_separator(
        "TASK 8: Image Injection Verification (zoom_region)"
    )

    orch = ToolCallingOrchestrator(
        server_url=server_2b_url,
        tools={"zoom_region": zoom_region},
        tool_definitions=[TOOL_ZOOM],
        max_tool_rounds=3,
        temperature=0,
        max_tokens=1024,
    )

    # Images are roughly 504x336 based on typical SceneIQ dims
    # Center is approximately x=252, y=168
    prompt = (
        "Zoom into the center of this image (approximately "
        "x=252, y=168). Describe in detail what you see in "
        "the zoomed view. Be specific about objects, colors, "
        "and textures visible in the zoomed region."
    )

    results: list[dict[str, Any]] = []

    for i, idx in enumerate(SAMPLE_INDICES):
        print(f"  Sample {i+1}/{N_SAMPLES} (index={idx})...")
        img_path = load_sample_image(idx)

        result = orch.run(
            image_path=img_path,
            system_prompt=(
                "You are a visual analyst. When you receive "
                "tool results containing images, examine them "
                "carefully and describe what you see in the "
                "image. Focus on visual details."
            ),
            user_prompt=prompt,
        )

        # Analyze if description contains specific visual details
        final = result["final_text"]
        lower_final = final.lower()

        # Indicators that model actually looked at the zoomed image
        visual_detail_keywords = [
            "road", "lane", "car", "vehicle", "truck",
            "tree", "building", "sky", "sign", "pavement",
            "asphalt", "concrete", "grass", "white", "black",
            "gray", "green", "blue", "red", "yellow",
            "line", "marking", "barrier", "pole", "light",
            "shadow", "reflection", "texture", "surface",
            "person", "pedestrian", "sidewalk", "curb",
            "water", "puddle", "cone", "fence", "bridge",
        ]
        detail_count = sum(
            1 for kw in visual_detail_keywords
            if kw in lower_final
        )

        # Model mentions spatial details about zoom?
        zoom_kws = [
            "zoomed", "zoom", "enlarged", "closer",
            "magnified", "crop", "region", "center", "middle",
        ]
        zoom_awareness = any(
            kw in lower_final for kw in zoom_kws
        )

        # Check if tool was actually called
        tool_called = any(
            tc["tool_name"] == "zoom_region"
            for tc in result["tool_calls"]
        )

        # Check if tool result had an image
        tool_had_image = any(
            tc.get("result_has_image", False)
            for tc in result["tool_calls"]
        )

        sample_result = {
            "sample_index": idx,
            "tool_called": tool_called,
            "tool_had_image": tool_had_image,
            "num_visual_details": detail_count,
            "zoom_awareness": zoom_awareness,
            "final_text": final,
            "num_rounds": result["num_rounds"],
            "num_tool_calls": result["num_tool_calls"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "tool_calls": result["tool_calls"],
            "full_conversation": result["full_conversation"],
        }
        results.append(sample_result)

        d_flag = "GOOD" if detail_count >= 3 else "LOW"
        z_flag = "YES" if zoom_awareness else "NO"
        i_flag = "YES" if tool_had_image else "NO"
        print(f"    Tool called: {tool_called}, Img: {i_flag}")
        print(f"    Details: {detail_count} ({d_flag}), Zoom: {z_flag}")
        print(f"    Response: {_truncate(final, 300)}")
        if result["error"]:
            print(f"    ERROR: {result['error']}")

    # Aggregate
    tools_called = sum(
        1 for r in results if r["tool_called"]
    )
    images_injected = sum(
        1 for r in results if r["tool_had_image"]
    )
    high_detail = sum(
        1 for r in results if r["num_visual_details"] >= 3
    )
    zoom_aware = sum(
        1 for r in results if r["zoom_awareness"]
    )
    avg_details = (
        sum(r["num_visual_details"] for r in results) / N_SAMPLES
    )

    hd_rate = high_detail / N_SAMPLES
    summary = {
        "task": "task8_image_injection_verification",
        "model": "2B",
        "total_samples": N_SAMPLES,
        "tool_called_count": tools_called,
        "images_injected_count": images_injected,
        "high_detail_count": high_detail,
        "high_detail_rate": round(hd_rate, 3),
        "zoom_aware_count": zoom_aware,
        "zoom_aware_rate": round(zoom_aware / N_SAMPLES, 3),
        "avg_visual_details": round(avg_details, 2),
        "detail_counts": [
            r["num_visual_details"] for r in results
        ],
        "samples": results,
    }

    print("\n  --- Task 8 Summary (Image Injection) ---")
    print(f"  Tool called: {tools_called}/{N_SAMPLES}")
    print(f"  Images injected: {images_injected}/{N_SAMPLES}")
    print(f"  High detail (>=3 kw): {high_detail}/{N_SAMPLES}")
    print(f"  Zoom-aware: {zoom_aware}/{N_SAMPLES}")
    print(f"  Avg visual detail kw: {avg_details:.1f}")
    verdict = (
        "PASS -- model likely sees images"
        if high_detail >= 7
        else "NEEDS MANUAL REVIEW"
    )
    print(f"  VERDICT: {verdict}")
    return summary


# ===================================================================
# Task 9: Waypoint visualization comprehension
# ===================================================================
def run_task9(server_2b_url: str) -> dict[str, Any]:
    """Test if model recognizes deliberately bad waypoints."""
    _print_separator(
        "TASK 9: Waypoint Visualization (bad waypoints)"
    )

    orch = ToolCallingOrchestrator(
        server_url=server_2b_url,
        tools={"visualize_waypoint": visualize_waypoint},
        tool_definitions=[TOOL_WAYPOINT_VIZ],
        max_tool_rounds=3,
        temperature=0,
        max_tokens=1024,
    )

    # Deliberately bad waypoints -- off-road locations
    BAD_WAYPOINTS = [
        (5, 5),    # top-left -- sky/off-road
        (3, 3),    # top-left corner
        (2, 60),   # top-right -- sky region
        (60, 2),   # bottom-left -- roadside/shoulder
        (5, 5),    # repeated bad position
        (0, 0),    # absolute top-left corner
        (62, 62),  # absolute bottom-right corner
        (5, 31),   # top-center -- sky area
        (60, 60),  # bottom-right -- off-road
        (3, 58),   # top-right -- sky
    ]

    results: list[dict[str, Any]] = []
    recognizes_bad = 0

    for i, idx in enumerate(SAMPLE_INDICES):
        row, col = BAD_WAYPOINTS[i]
        print(
            f"  Sample {i+1}/{N_SAMPLES} "
            f"(index={idx}, wp=({row},{col}))..."
        )
        img_path = load_sample_image(idx)

        prompt = (
            f"I've predicted this vehicle should drive to "
            f"waypoint row={row}, col={col}. Use the "
            f"visualization tool to draw this on the image "
            f"and tell me if this makes sense for safe driving."
        )

        result = orch.run(
            image_path=img_path,
            system_prompt=(
                "You are an expert driving safety analyst. "
                "Evaluate whether a predicted waypoint "
                "location makes sense for safe driving. "
                "Be critical and specific about problems."
            ),
            user_prompt=prompt,
        )

        final = result["final_text"].lower()

        # Check if the model recognizes the waypoint is bad
        bad_indicators = [
            "not safe", "unsafe", "danger",
            "off-road", "off road",
            "incorrect", "wrong", "bad", "problematic",
            "doesn't make sense", "does not make sense",
            "not recommended", "not advisable",
            "sky", "outside", "out of",
            "impractical", "unreasonable",
            "not feasible", "not viable", "inappropriate",
            "collision", "crash", "accident",
            "curb", "sidewalk", "building",
            "no ", "not a valid", "invalid", "risky",
            "top-left", "top left", "corner",
            "not on the road", "not within",
            "beyond the road",
            "should not", "shouldn't",
        ]
        recognized = any(kw in final for kw in bad_indicators)
        if recognized:
            recognizes_bad += 1

        # Check if tool was called
        tool_called = any(
            tc["tool_name"] == "visualize_waypoint"
            for tc in result["tool_calls"]
        )

        sample_result = {
            "sample_index": idx,
            "bad_waypoint": {"row": row, "col": col},
            "tool_called": tool_called,
            "recognized_bad_waypoint": recognized,
            "final_text": result["final_text"],
            "num_rounds": result["num_rounds"],
            "num_tool_calls": result["num_tool_calls"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "tool_calls": result["tool_calls"],
            "full_conversation": result["full_conversation"],
        }
        results.append(sample_result)

        flag = "RECOGNIZED BAD" if recognized else "DID NOT FLAG"
        print(f"    Tool called: {tool_called}, {flag}")
        resp = _truncate(result["final_text"], 300)
        print(f"    Response: {resp}")
        if result["error"]:
            print(f"    ERROR: {result['error']}")

    recognition_rate = recognizes_bad / N_SAMPLES

    summary = {
        "task": "task9_waypoint_comprehension",
        "model": "2B",
        "total_samples": N_SAMPLES,
        "recognizes_bad_waypoint_count": recognizes_bad,
        "recognition_rate": round(recognition_rate, 3),
        "samples": results,
    }

    n_bad = recognizes_bad
    rr = recognition_rate
    print("\n  --- Task 9 Summary ---")
    print(f"  Recognized bad: {n_bad}/{N_SAMPLES} ({rr:.0%})")
    verdict = (
        "Model understands waypoint semantics"
        if n_bad >= 7
        else "Model struggles with waypoint reasoning"
    )
    print(f"  VERDICT: {verdict}")
    return summary


# ===================================================================
# Task 10: Road geometry tool output comprehension
# ===================================================================
def run_task10(server_2b_url: str) -> dict[str, Any]:
    """Test model interpretation of geometry tool output."""
    _print_separator(
        "TASK 10: Road Geometry Tool Output Comprehension"
    )

    orch = ToolCallingOrchestrator(
        server_url=server_2b_url,
        tools={"analyze_road_geometry": analyze_road_geometry},
        tool_definitions=[TOOL_ROAD_GEOMETRY],
        max_tool_rounds=3,
        temperature=0,
        max_tokens=1024,
    )

    prompt = (
        "Analyze the road geometry in this image. What do you "
        "see about the road layout, lanes, and curvature?"
    )

    results: list[dict[str, Any]] = []

    for i, idx in enumerate(SAMPLE_INDICES):
        print(f"  Sample {i+1}/{N_SAMPLES} (index={idx})...")
        img_path = load_sample_image(idx)

        result = orch.run(
            image_path=img_path,
            system_prompt=(
                "You are an expert in road infrastructure "
                "analysis. When you receive tool results, "
                "interpret the data and visual annotations "
                "carefully."
            ),
            user_prompt=prompt,
        )

        final = result["final_text"].lower()

        # Check if model references geometry data from tool
        tool_data_keywords = [
            "lane", "curvature", "straight", "gentle", "sharp",
            "vanishing point", "drivable",
            "boundary", "boundaries",
            "edge", "detected", "lines",
        ]
        data_ref_count = sum(
            1 for kw in tool_data_keywords if kw in final
        )
        references_tool_data = data_ref_count >= 2

        # Check if tool was called
        tool_called = any(
            tc["tool_name"] == "analyze_road_geometry"
            for tc in result["tool_calls"]
        )

        # Check if the tool result had image injected
        tool_had_image = any(
            tc.get("result_has_image", False)
            for tc in result["tool_calls"]
        )

        # Get actual geometry result directly for comparison
        actual_geo = analyze_road_geometry(img_path)
        actual_curv = actual_geo.get("road_curvature", "unknown")
        actual_lanes = actual_geo.get("num_lanes_detected", 0)

        # Check if model mentions the actual curvature
        mentions_curv = actual_curv.replace("_", " ") in final

        sample_result = {
            "sample_index": idx,
            "tool_called": tool_called,
            "tool_had_image": tool_had_image,
            "references_tool_data": references_tool_data,
            "data_ref_count": data_ref_count,
            "mentions_actual_curvature": mentions_curv,
            "actual_curvature": actual_curv,
            "actual_lanes": actual_lanes,
            "final_text": result["final_text"],
            "num_rounds": result["num_rounds"],
            "num_tool_calls": result["num_tool_calls"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "tool_calls": result["tool_calls"],
            "full_conversation": result["full_conversation"],
        }
        results.append(sample_result)

        d_flag = "YES" if references_tool_data else "NO"
        c_flag = "YES" if mentions_curv else "NO"
        print(f"    Tool: {tool_called}, Image: {tool_had_image}")
        print(f"    Refs data: {d_flag} ({data_ref_count} kw)")
        print(f"    Curvature: {actual_curv}, mentioned: {c_flag}")
        resp = _truncate(result["final_text"], 300)
        print(f"    Response: {resp}")
        if result["error"]:
            print(f"    ERROR: {result['error']}")

    # Aggregate
    tools_called = sum(
        1 for r in results if r["tool_called"]
    )
    refs_data = sum(
        1 for r in results if r["references_tool_data"]
    )
    refs_curv = sum(
        1 for r in results if r["mentions_actual_curvature"]
    )
    images_injected = sum(
        1 for r in results if r["tool_had_image"]
    )
    avg_data_refs = (
        sum(r["data_ref_count"] for r in results) / N_SAMPLES
    )

    summary = {
        "task": "task10_road_geometry_comprehension",
        "model": "2B",
        "total_samples": N_SAMPLES,
        "tool_called_count": tools_called,
        "images_injected_count": images_injected,
        "references_tool_data_count": refs_data,
        "references_tool_data_rate": round(
            refs_data / N_SAMPLES, 3
        ),
        "mentions_actual_curvature_count": refs_curv,
        "avg_data_ref_keywords": round(avg_data_refs, 2),
        "samples": results,
    }

    print("\n  --- Task 10 Summary ---")
    print(f"  Tool called: {tools_called}/{N_SAMPLES}")
    print(f"  Images injected: {images_injected}/{N_SAMPLES}")
    print(f"  Refs tool data (>=2 kw): {refs_data}/{N_SAMPLES}")
    print(f"  Mentions curvature: {refs_curv}/{N_SAMPLES}")
    print(f"  Avg data-ref kw: {avg_data_refs:.1f}")
    return summary


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-server", action="store_true",
        help="Skip server start/stop (use pre-started servers)",
    )
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  Phase 1 Mechanical Validation -- Tasks 6-10")
    print("=" * 70)

    server_2b = None
    server_8b = None

    if not args.no_server:
        print("\nStarting vLLM servers...")
        server_2b = VLLMServer(
            model_path=MODEL_2B,
            port=PORT_2B,
            gpu_id=GPU_2B,
            max_model_len=8192,
            gpu_memory_utilization=0.8,
            enable_tools=True,
        )
        server_8b = VLLMServer(
            model_path=MODEL_8B,
            port=PORT_8B,
            gpu_id=GPU_8B,
            max_model_len=8192,
            gpu_memory_utilization=0.8,
            enable_tools=True,
        )
        print(
            f"  Starting 2B on GPU {GPU_2B}, port {PORT_2B}..."
        )
        server_2b.start(timeout=600)
        print(
            f"  Starting 8B on GPU {GPU_8B}, port {PORT_8B}..."
        )
        server_8b.start(timeout=600)
    else:
        print("\nUsing pre-started servers...")

    url_2b = f"http://localhost:{PORT_2B}"
    url_8b = f"http://localhost:{PORT_8B}"

    all_results: dict[str, Any] = {
        "experiment": "phase1_tasks6to10",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {"2B": MODEL_2B, "8B": MODEL_8B},
        "sample_indices": SAMPLE_INDICES,
    }

    try:
        # Task 6: 8B comparison
        all_results["task6"] = run_task6(url_8b)

        # Task 7: Max tool rounds (2B)
        all_results["task7"] = run_task7(url_2b)

        # Task 8: Image injection (2B) -- MOST IMPORTANT
        all_results["task8"] = run_task8(url_2b)

        # Task 9: Waypoint comprehension (2B)
        all_results["task9"] = run_task9(url_2b)

        # Task 10: Road geometry comprehension (2B)
        all_results["task10"] = run_task10(url_2b)

    finally:
        if server_2b is not None:
            print("\nStopping servers...")
            server_2b.stop()
        if server_8b is not None:
            server_8b.stop()

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = round(elapsed, 1)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Total elapsed: {elapsed/60:.1f} minutes")

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    _print_separator("FINAL SUMMARY -- Tasks 6-10")

    t6 = all_results.get("task6", {})
    t7 = all_results.get("task7", {})
    t8 = all_results.get("task8", {})
    t9 = all_results.get("task9", {})
    t10 = all_results.get("task10", {})

    print("Task 6 (8B Tool Calling):")
    v = t6.get("avg_tool_calls", "N/A")
    print(f"  Avg tool calls: {v}")
    v = t6.get("argument_validity_rate", "N/A")
    print(f"  Argument validity: {v}")
    v = t6.get("references_tool_output_rate", "N/A")
    print(f"  Refs tool output: {v}")

    print("\nTask 7 (Max Tool Rounds, 2B):")
    v = t7.get("avg_rounds", "N/A")
    print(f"  Avg rounds: {v}")
    v = t7.get("hit_limit_count", "N/A")
    print(f"  Hit 8-round limit: {v}/{N_SAMPLES}")
    v = t7.get("rounds_distribution", "N/A")
    print(f"  Rounds dist: {v}")

    print("\nTask 8 (Image Injection -- CRITICAL):")
    v = t8.get("images_injected_count", "N/A")
    print(f"  Images injected: {v}/{N_SAMPLES}")
    v = t8.get("high_detail_count", "N/A")
    print(f"  High visual detail: {v}/{N_SAMPLES}")
    v = t8.get("zoom_aware_count", "N/A")
    print(f"  Zoom aware: {v}/{N_SAMPLES}")
    v = t8.get("avg_visual_details", "N/A")
    print(f"  Avg visual detail kw: {v}")

    print("\nTask 9 (Bad Waypoint Recognition):")
    v = t9.get("recognizes_bad_waypoint_count", "N/A")
    print(f"  Recognized bad: {v}/{N_SAMPLES}")
    v = t9.get("recognition_rate", "N/A")
    print(f"  Recognition rate: {v}")

    print("\nTask 10 (Road Geometry Comprehension):")
    v = t10.get("references_tool_data_count", "N/A")
    print(f"  Refs tool data: {v}/{N_SAMPLES}")
    v = t10.get("mentions_actual_curvature_count", "N/A")
    print(f"  Mentions curvature: {v}/{N_SAMPLES}")


if __name__ == "__main__":
    main()
