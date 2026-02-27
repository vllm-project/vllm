#!/usr/bin/env python3
# ruff: noqa: E501,E402
# type: ignore[import-not-found]
"""Phase 1 Mechanical Validation: Tasks 1-5 for visual tool-calling experiment.

Proves the basic mechanics of tool calling work before measuring reasoning quality.

Task 1: Bare tool calling smoke test (base 2B, zoom_region only, auto)
Task 2: Forced tool calling smoke test (base 2B, zoom_region only, required)
Task 3: Multiple tools available (base 2B, 3 tools, auto)
Task 4: Tool call format validation (base 2B, 3 tools, 30 samples)
Task 5: Fine-tuned model tool calling (fine-tuned 2B, zoom_region only, auto)

Usage:
    python tool_calling_experiment/phase1_tasks1to5.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import requests  # type: ignore[import-not-found]

# Ensure sibling modules are importable
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from orchestrator import ToolCallingOrchestrator  # type: ignore[import-not-found]
from visual_tools import (  # type: ignore[import-not-found]
    TOOL_ROAD_GEOMETRY,
    TOOL_WAYPOINT_VIZ,
    TOOL_ZOOM,
    analyze_road_geometry,
    load_sample_image,
    visualize_waypoint,
    zoom_region,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(_DIR, "phase1_tasks1to5_results.json")

BASE_MODEL_PATH = "/fsx/models/Qwen3-VL-2B-Instruct"
FT_MODEL_PATH = "/workspace/vllm/models/checkpoint"
DATASET_PATH = "/workspace/vllm/models/dataset"

BASE_PORT = 8320
FT_PORT = 8321
BASE_GPU = 0
FT_GPU = 1

# Image bounds (504 x 336)
IMG_W = 504
IMG_H = 336
MAX_X = IMG_W - 1  # 503
MAX_Y = IMG_H - 1  # 335

VALID_TOOL_NAMES = {"zoom_region", "visualize_waypoint", "analyze_road_geometry", "find_similar_scenes"}

# Argument validation ranges
ARG_RANGES = {
    "center_x": (0, MAX_X),
    "center_y": (0, MAX_Y),
    "crop_size": (32, 256),
    "waypoint_row": (0, 62),
    "waypoint_col": (0, 62),
    "k": (1, 5),
}

# Three tools available (excluding find_similar_scenes since no FAISS index)
THREE_TOOLS_DEFS = [TOOL_ZOOM, TOOL_WAYPOINT_VIZ, TOOL_ROAD_GEOMETRY]

TOOL_EXECUTORS: dict[str, Any] = {
    "zoom_region": zoom_region,
    "visualize_waypoint": visualize_waypoint,
    "analyze_road_geometry": analyze_road_geometry,
}


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

    log_path = f"/tmp/vllm_phase1_{port}.log"
    print(f"  Starting server: model={os.path.basename(model_path)}, GPU={gpu_id}, port={port}")
    print(f"  Log: {log_path}")
    # Keep log_file alive in Popen so fd stays open for the subprocess
    proc = subprocess.Popen(
        cmd, env=env, cwd="/tmp",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Store log path for reference (server logs go to stderr in background)
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


def kill_servers() -> None:
    """Kill all phase1 vllm servers."""
    subprocess.run(["pkill", "-f", "vllm serve.*832[01]"], capture_output=True)
    time.sleep(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_images(n: int) -> list[str]:
    """Load n sample images from the MDS dataset."""
    return [load_sample_image(i) for i in range(n)]


def validate_tool_call_json(tc: dict) -> dict[str, Any]:
    """Validate a single tool call from the orchestrator log."""
    result: dict[str, Any] = {
        "tool_name": tc.get("tool_name"),
        "valid_name": tc.get("tool_name") in VALID_TOOL_NAMES,
        "valid_json": True,  # if we got here, JSON parsed
        "arguments": tc.get("arguments", {}),
        "arg_validations": {},
    }

    args = tc.get("arguments", {})
    for arg_name, (lo, hi) in ARG_RANGES.items():
        if arg_name in args:
            val = args[arg_name]
            try:
                val_num = int(val) if isinstance(val, (int, float, str)) else None
                in_range = val_num is not None and lo <= val_num <= hi
                result["arg_validations"][arg_name] = {
                    "value": val,
                    "in_range": in_range,
                    "expected_range": [lo, hi],
                }
            except (ValueError, TypeError):
                result["arg_validations"][arg_name] = {
                    "value": val,
                    "in_range": False,
                    "expected_range": [lo, hi],
                    "error": "not a number",
                }

    return result


def print_sample_summary(idx: int, result: dict, task_name: str) -> None:
    """Print a per-sample summary line."""
    n_tools = result.get("num_tool_calls", 0)
    n_rounds = result.get("num_rounds", 0)
    error = result.get("error")
    tools_used = [tc["tool_name"] for tc in result.get("tool_calls", [])]
    status = "ERROR" if error else "OK"
    final_len = len(result.get("final_text", ""))
    print(f"  [{task_name}] Sample {idx}: {status} | rounds={n_rounds} | tool_calls={n_tools} | tools={tools_used} | final_text_len={final_len}")
    if error:
        print(f"    Error: {error}")


# ---------------------------------------------------------------------------
# Task 1: Bare tool calling smoke test
# ---------------------------------------------------------------------------

def run_task1(base_url: str, images: list[str]) -> list[dict]:
    """Task 1: zoom_region only, tool_choice=auto, base 2B model."""
    print("\n" + "=" * 70)
    print("TASK 1: Bare tool calling smoke test")
    print("  Model: Base 2B | Tools: zoom_region only | tool_choice: auto")
    print("=" * 70)

    orch = ToolCallingOrchestrator(
        server_url=base_url,
        tools={"zoom_region": zoom_region},
        tool_definitions=[TOOL_ZOOM],
        max_tool_rounds=3,
        temperature=0,
        max_tokens=1024,
    )

    prompt = "Describe this driving scene. You can zoom into any region for a closer look."
    results: list[dict] = []

    for i, img_path in enumerate(images):
        print(f"\n  --- Sample {i} ---")
        r = orch.run(
            image_path=img_path,
            system_prompt="You are a driving scene analyst with access to visual inspection tools.",
            user_prompt=prompt,
            tool_choice="auto",
        )
        made_tool_call = r["num_tool_calls"] > 0
        tool_calls_valid = []
        coords_in_bounds = []
        used_zoomed_in_response = False

        for tc in r.get("tool_calls", []):
            v = validate_tool_call_json(tc)
            tool_calls_valid.append(v)
            args = tc.get("arguments", {})
            cx = args.get("center_x")
            cy = args.get("center_y")
            if cx is not None and cy is not None:
                try:
                    x_ok = 0 <= int(cx) <= MAX_X
                    y_ok = 0 <= int(cy) <= MAX_Y
                    coords_in_bounds.append(x_ok and y_ok)
                except (ValueError, TypeError):
                    coords_in_bounds.append(False)

            if tc.get("result_has_image"):
                used_zoomed_in_response = True

        entry = {
            "sample_index": i,
            "image_path": img_path,
            "made_tool_call": made_tool_call,
            "num_tool_calls": r["num_tool_calls"],
            "num_rounds": r["num_rounds"],
            "tool_calls_detail": tool_calls_valid,
            "coords_in_bounds": coords_in_bounds,
            "all_coords_valid": all(coords_in_bounds) if coords_in_bounds else None,
            "used_zoomed_image": used_zoomed_in_response,
            "final_text": r["final_text"],
            "final_text_length": len(r.get("final_text", "")),
            "raw_tool_calls": r.get("tool_calls", []),
            "error": r.get("error"),
            "latency_ms": r.get("latency_ms"),
        }
        results.append(entry)
        print_sample_summary(i, r, "T1")
        print(f"    made_tool_call={made_tool_call}, coords_valid={coords_in_bounds}, used_zoomed={used_zoomed_in_response}")

    n = len(results)
    n_called = sum(1 for r in results if r["made_tool_call"])
    n_valid_coords = sum(1 for r in results if r.get("all_coords_valid") is True)
    n_used_zoom = sum(1 for r in results if r["used_zoomed_image"])
    n_errors = sum(1 for r in results if r["error"])

    summary = {
        "task": "task1_bare_smoke_test",
        "total_samples": n,
        "made_tool_call": n_called,
        "tool_call_rate": round(n_called / n, 3) if n > 0 else 0,
        "valid_coords": n_valid_coords,
        "used_zoomed_image": n_used_zoom,
        "errors": n_errors,
    }
    print(f"\n  TASK 1 SUMMARY: {json.dumps(summary, indent=2)}")
    return [{"summary": summary, "samples": results}]


# ---------------------------------------------------------------------------
# Task 2: Forced tool calling smoke test
# ---------------------------------------------------------------------------

def run_task2(base_url: str, images: list[str]) -> list[dict]:
    """Task 2: zoom_region only, tool_choice=required, base 2B model."""
    print("\n" + "=" * 70)
    print("TASK 2: Forced tool calling smoke test")
    print("  Model: Base 2B | Tools: zoom_region only | tool_choice: required")
    print("=" * 70)

    orch = ToolCallingOrchestrator(
        server_url=base_url,
        tools={"zoom_region": zoom_region},
        tool_definitions=[TOOL_ZOOM],
        max_tool_rounds=3,
        temperature=0,
        max_tokens=1024,
    )

    prompt = "Describe this driving scene. You can zoom into any region for a closer look."
    results: list[dict] = []

    for i, img_path in enumerate(images):
        print(f"\n  --- Sample {i} ---")
        r = orch.run(
            image_path=img_path,
            system_prompt="You are a driving scene analyst with access to visual inspection tools.",
            user_prompt=prompt,
            tool_choice="required",
        )

        made_tool_call = r["num_tool_calls"] > 0
        tool_calls_valid = []
        coords_in_bounds = []

        for tc in r.get("tool_calls", []):
            v = validate_tool_call_json(tc)
            tool_calls_valid.append(v)
            args = tc.get("arguments", {})
            cx = args.get("center_x")
            cy = args.get("center_y")
            if cx is not None and cy is not None:
                try:
                    x_ok = 0 <= int(cx) <= MAX_X
                    y_ok = 0 <= int(cy) <= MAX_Y
                    coords_in_bounds.append(x_ok and y_ok)
                except (ValueError, TypeError):
                    coords_in_bounds.append(False)

        entry = {
            "sample_index": i,
            "image_path": img_path,
            "made_tool_call": made_tool_call,
            "num_tool_calls": r["num_tool_calls"],
            "num_rounds": r["num_rounds"],
            "tool_calls_detail": tool_calls_valid,
            "coords_in_bounds": coords_in_bounds,
            "all_coords_valid": all(coords_in_bounds) if coords_in_bounds else None,
            "final_text": r["final_text"],
            "final_text_length": len(r.get("final_text", "")),
            "raw_tool_calls": r.get("tool_calls", []),
            "error": r.get("error"),
            "latency_ms": r.get("latency_ms"),
        }
        results.append(entry)
        print_sample_summary(i, r, "T2")
        print(f"    made_tool_call={made_tool_call}, coords_valid={coords_in_bounds}")

    n = len(results)
    n_called = sum(1 for r in results if r["made_tool_call"])
    n_valid_coords = sum(1 for r in results if r.get("all_coords_valid") is True)
    n_errors = sum(1 for r in results if r["error"])

    summary = {
        "task": "task2_forced_smoke_test",
        "total_samples": n,
        "made_tool_call": n_called,
        "tool_call_rate": round(n_called / n, 3) if n > 0 else 0,
        "valid_coords": n_valid_coords,
        "errors": n_errors,
    }
    print(f"\n  TASK 2 SUMMARY: {json.dumps(summary, indent=2)}")
    return [{"summary": summary, "samples": results}]


# ---------------------------------------------------------------------------
# Task 3: Multiple tools available
# ---------------------------------------------------------------------------

def run_task3(base_url: str, images: list[str]) -> list[dict]:
    """Task 3: All 3 tools available (no FAISS), tool_choice=auto, base 2B."""
    print("\n" + "=" * 70)
    print("TASK 3: Multiple tools available")
    print("  Model: Base 2B | Tools: zoom, waypoint, road_geometry | tool_choice: auto")
    print("=" * 70)

    orch = ToolCallingOrchestrator(
        server_url=base_url,
        tools=TOOL_EXECUTORS,
        tool_definitions=THREE_TOOLS_DEFS,
        max_tool_rounds=5,
        temperature=0,
        max_tokens=1024,
    )

    prompt = (
        "Analyze this driving scene thoroughly. You have tools to zoom in, "
        "visualize waypoints, and analyze road geometry."
    )
    results: list[dict] = []

    for i, img_path in enumerate(images):
        print(f"\n  --- Sample {i} ---")
        r = orch.run(
            image_path=img_path,
            system_prompt="You are a driving scene analyst with access to visual inspection tools.",
            user_prompt=prompt,
            tool_choice="auto",
        )

        tools_called = [tc["tool_name"] for tc in r.get("tool_calls", [])]
        first_tool = tools_called[0] if tools_called else None
        called_multiple = len(tools_called) > 1
        unique_tools = list(set(tools_called))
        nonsensical_tools = [t for t in tools_called if t not in VALID_TOOL_NAMES]

        entry = {
            "sample_index": i,
            "image_path": img_path,
            "num_tool_calls": r["num_tool_calls"],
            "num_rounds": r["num_rounds"],
            "first_tool_called": first_tool,
            "all_tools_called": tools_called,
            "unique_tools_called": unique_tools,
            "called_multiple_tools": called_multiple,
            "nonsensical_tool_calls": nonsensical_tools,
            "final_text": r["final_text"],
            "final_text_length": len(r.get("final_text", "")),
            "raw_tool_calls": r.get("tool_calls", []),
            "error": r.get("error"),
            "latency_ms": r.get("latency_ms"),
        }
        results.append(entry)
        print_sample_summary(i, r, "T3")
        print(f"    first_tool={first_tool}, all_tools={tools_called}, nonsensical={nonsensical_tools}")

    n = len(results)
    first_tool_dist: dict[str, int] = {}
    for r in results:
        ft = r["first_tool_called"] or "none"
        first_tool_dist[ft] = first_tool_dist.get(ft, 0) + 1

    tool_usage: dict[str, int] = {}
    for r in results:
        for t in r["all_tools_called"]:
            tool_usage[t] = tool_usage.get(t, 0) + 1

    n_multi = sum(1 for r in results if r["called_multiple_tools"])
    n_nonsensical = sum(1 for r in results if r["nonsensical_tool_calls"])

    summary = {
        "task": "task3_multiple_tools",
        "total_samples": n,
        "first_tool_distribution": first_tool_dist,
        "tool_usage_distribution": tool_usage,
        "called_multiple_tools": n_multi,
        "nonsensical_tool_calls": n_nonsensical,
    }
    print(f"\n  TASK 3 SUMMARY: {json.dumps(summary, indent=2)}")
    return [{"summary": summary, "samples": results}]


# ---------------------------------------------------------------------------
# Task 4: Tool call format validation (30 samples)
# ---------------------------------------------------------------------------

def run_task4(base_url: str) -> list[dict]:
    """Task 4: Format validation with 30 samples, all 3 tools, base 2B."""
    print("\n" + "=" * 70)
    print("TASK 4: Tool call format validation (30 samples)")
    print("  Model: Base 2B | Tools: zoom, waypoint, road_geometry | tool_choice: auto")
    print("=" * 70)

    images = load_images(30)

    orch = ToolCallingOrchestrator(
        server_url=base_url,
        tools=TOOL_EXECUTORS,
        tool_definitions=THREE_TOOLS_DEFS,
        max_tool_rounds=3,
        temperature=0,
        max_tokens=1024,
    )

    prompt = (
        "Analyze this driving scene. Use the available tools to inspect the scene. "
        "You have tools to zoom in, visualize waypoints, and analyze road geometry."
    )

    all_tool_calls: list[dict] = []
    sample_results: list[dict] = []

    for i, img_path in enumerate(images):
        if i % 5 == 0:
            print(f"  Processing sample {i}/30...")
        r = orch.run(
            image_path=img_path,
            system_prompt="You are a driving scene analyst with access to visual inspection tools.",
            user_prompt=prompt,
            tool_choice="auto",
        )

        for tc in r.get("tool_calls", []):
            validated = validate_tool_call_json(tc)
            validated["sample_index"] = i
            all_tool_calls.append(validated)

        sample_results.append({
            "sample_index": i,
            "num_tool_calls": r["num_tool_calls"],
            "error": r.get("error"),
        })
        print_sample_summary(i, r, "T4")

    total_calls = len(all_tool_calls)
    valid_json_count = sum(1 for tc in all_tool_calls if tc["valid_json"])
    valid_name_count = sum(1 for tc in all_tool_calls if tc["valid_name"])

    arg_stats: dict[str, dict[str, int]] = {}
    for tc in all_tool_calls:
        for arg_name, v in tc.get("arg_validations", {}).items():
            if arg_name not in arg_stats:
                arg_stats[arg_name] = {"total": 0, "in_range": 0, "out_of_range": 0}
            arg_stats[arg_name]["total"] += 1
            if v.get("in_range"):
                arg_stats[arg_name]["in_range"] += 1
            else:
                arg_stats[arg_name]["out_of_range"] += 1

    arg_pcts: dict[str, Any] = {}
    for arg_name, stats in arg_stats.items():
        pct = round(stats["in_range"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        arg_pcts[arg_name] = {
            "total": stats["total"],
            "in_range": stats["in_range"],
            "out_of_range": stats["out_of_range"],
            "pct_in_range": pct,
        }

    summary = {
        "task": "task4_format_validation",
        "total_samples": 30,
        "total_tool_calls": total_calls,
        "valid_json_pct": round(valid_json_count / total_calls * 100, 1) if total_calls > 0 else 0,
        "valid_name_pct": round(valid_name_count / total_calls * 100, 1) if total_calls > 0 else 0,
        "argument_validation": arg_pcts,
        "samples_with_errors": sum(1 for r in sample_results if r["error"]),
    }
    print(f"\n  TASK 4 SUMMARY: {json.dumps(summary, indent=2)}")
    return [{"summary": summary, "all_tool_calls": all_tool_calls, "samples": sample_results}]


# ---------------------------------------------------------------------------
# Task 5: Fine-tuned model tool calling
# ---------------------------------------------------------------------------

def _build_ft_prompt(sample_index: int) -> tuple[str, str]:
    """Build the fine-tuned model's prompt for a given sample.

    Returns (user_prompt_text, image_path).
    The fine-tuned model uses a structured prompt with velocity, waypoints, etc.
    We extract the text parts from the MDS dataset's messages field.
    """
    import streaming  # type: ignore[import-not-found]

    ds = streaming.StreamingDataset(local=DATASET_PATH, shuffle=False)
    sample = ds[sample_index]

    msgs = sample["messages"]
    if isinstance(msgs, bytes):
        msgs = msgs.decode()
    if isinstance(msgs, str):
        msgs = json.loads(msgs)

    user_content = msgs[0]["content"]
    text_parts = []
    for c in user_content:
        if isinstance(c, dict) and "text" in c:
            text_parts.append(c["text"])
    full_prompt = "".join(text_parts)

    img_path = load_sample_image(sample_index)
    return full_prompt, img_path


def run_task5(ft_url: str) -> list[dict]:
    """Task 5: Fine-tuned 2B model with zoom_region, tool_choice=auto."""
    print("\n" + "=" * 70)
    print("TASK 5: Fine-tuned model tool calling")
    print("  Model: Fine-tuned 2B | Tools: zoom_region only | tool_choice: auto")
    print("=" * 70)

    orch = ToolCallingOrchestrator(
        server_url=ft_url,
        tools={"zoom_region": zoom_region},
        tool_definitions=[TOOL_ZOOM],
        max_tool_rounds=3,
        temperature=0,
        max_tokens=1024,
    )

    odd_tokens = [
        "<|odd_nominal|>", "<|odd_flood|>", "<|odd_incident|>",
        "<|odd_policehorse|>", "<|odd_flagger|>",
    ]

    results: list[dict] = []
    for i in range(10):
        print(f"\n  --- Sample {i} ---")
        ft_prompt, img_path = _build_ft_prompt(i)

        r = orch.run(
            image_path=img_path,
            system_prompt="",
            user_prompt=ft_prompt,
            tool_choice="auto",
        )

        made_tool_call = r["num_tool_calls"] > 0
        tools_called = [tc["tool_name"] for tc in r.get("tool_calls", [])]

        final = r.get("final_text", "")
        has_odd_token = any(tok in final for tok in odd_tokens)
        has_actions = "<actions>" in final

        initial = r.get("initial_assessment", "")
        initial_has_odd = any(tok in initial for tok in odd_tokens)
        initial_has_actions = "<actions>" in initial

        prediction_format_preserved = has_odd_token or initial_has_odd

        entry = {
            "sample_index": i,
            "image_path": img_path,
            "made_tool_call": made_tool_call,
            "num_tool_calls": r["num_tool_calls"],
            "tools_called": tools_called,
            "final_text": final[:500],
            "initial_assessment": initial[:500],
            "has_odd_token_final": has_odd_token,
            "has_actions_final": has_actions,
            "has_odd_token_initial": initial_has_odd,
            "has_actions_initial": initial_has_actions,
            "prediction_format_preserved": prediction_format_preserved,
            "error": r.get("error"),
            "latency_ms": r.get("latency_ms"),
        }
        results.append(entry)
        print_sample_summary(i, r, "T5")
        print(f"    made_tool_call={made_tool_call}, odd_token={has_odd_token or initial_has_odd}, actions={has_actions or initial_has_actions}")
        if final:
            print(f"    final_text (first 200): {final[:200]}")

    n = len(results)
    n_called = sum(1 for r in results if r["made_tool_call"])
    n_preserved = sum(1 for r in results if r["prediction_format_preserved"])
    n_errors = sum(1 for r in results if r["error"])

    summary = {
        "task": "task5_finetuned_model",
        "total_samples": n,
        "made_tool_call": n_called,
        "tool_call_rate": round(n_called / n, 3) if n > 0 else 0,
        "prediction_format_preserved": n_preserved,
        "format_preserved_rate": round(n_preserved / n, 3) if n > 0 else 0,
        "errors": n_errors,
        "note": "Does adding tool definitions break the model's existing prediction format?",
    }
    print(f"\n  TASK 5 SUMMARY: {json.dumps(summary, indent=2)}")
    return [{"summary": summary, "samples": results}]


# ---------------------------------------------------------------------------
# Comparison: Task 1 vs Task 2
# ---------------------------------------------------------------------------

def compare_tasks_1_2(task1_results: list[dict], task2_results: list[dict]) -> dict:
    """Compare auto vs required tool_choice behavior."""
    t1 = task1_results[0]
    t2 = task2_results[0]

    s1 = t1["summary"]
    s2 = t2["summary"]

    comparison: dict[str, Any] = {
        "task1_auto_tool_call_rate": s1["tool_call_rate"],
        "task2_required_tool_call_rate": s2["tool_call_rate"],
        "difference": round(s2["tool_call_rate"] - s1["tool_call_rate"], 3),
        "task1_valid_coords": s1["valid_coords"],
        "task2_valid_coords": s2["valid_coords"],
        "interpretation": "",
    }

    if s2["tool_call_rate"] > s1["tool_call_rate"]:
        comparison["interpretation"] = (
            f"Forced mode (required) increases tool call rate from "
            f"{s1['tool_call_rate']} to {s2['tool_call_rate']}."
        )
    elif s2["tool_call_rate"] == s1["tool_call_rate"]:
        comparison["interpretation"] = (
            f"Both auto and required produce the same tool call rate "
            f"({s1['tool_call_rate']}). Model already calls tools voluntarily."
        )
    else:
        comparison["interpretation"] = (
            f"Unexpected: auto ({s1['tool_call_rate']}) > required "
            f"({s2['tool_call_rate']}). Possible issue with forced mode."
        )

    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all Phase 1 tasks."""
    print("=" * 70)
    print("PHASE 1: MECHANICAL VALIDATION (Tasks 1-5)")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print()

    # ---- Start servers ----
    print("Starting vLLM servers...")
    base_proc = start_vllm_server(BASE_MODEL_PATH, BASE_PORT, BASE_GPU)
    ft_proc = start_vllm_server(FT_MODEL_PATH, FT_PORT, FT_GPU)

    base_url = f"http://localhost:{BASE_PORT}"
    ft_url = f"http://localhost:{FT_PORT}"

    try:
        print("\nWaiting for servers to become healthy...")
        base_ok = wait_for_server(BASE_PORT, timeout=900)
        if not base_ok:
            print("ERROR: Base model server failed to start. Check /tmp/vllm_phase1_8320.log")
            sys.exit(1)

        ft_ok = wait_for_server(FT_PORT, timeout=900)
        if not ft_ok:
            print("ERROR: Fine-tuned model server failed to start. Check /tmp/vllm_phase1_8321.log")
            sys.exit(1)

        print("\nBoth servers healthy. Loading sample images...")

        images_10 = load_images(10)
        print(f"Loaded {len(images_10)} images.")

        all_results: dict[str, Any] = {
            "experiment": "phase1_mechanical_validation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "base_model": BASE_MODEL_PATH,
            "ft_model": FT_MODEL_PATH,
            "image_dimensions": {"width": IMG_W, "height": IMG_H},
        }

        # Task 1
        t1 = run_task1(base_url, images_10)
        all_results["task1"] = t1

        # Task 2
        t2 = run_task2(base_url, images_10)
        all_results["task2"] = t2

        # Task 1 vs 2 comparison
        comparison_1_2 = compare_tasks_1_2(t1, t2)
        all_results["task1_vs_task2"] = comparison_1_2
        print(f"\n  TASK 1 vs 2 COMPARISON: {json.dumps(comparison_1_2, indent=2)}")

        # Task 3
        t3 = run_task3(base_url, images_10)
        all_results["task3"] = t3

        # Task 4
        t4 = run_task4(base_url)
        all_results["task4"] = t4

        # Task 5
        t5 = run_task5(ft_url)
        all_results["task5"] = t5

        # ---- Save results ----
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {RESULTS_PATH}")

        # ---- Final summary ----
        print("\n" + "=" * 70)
        print("PHASE 1 FINAL SUMMARY")
        print("=" * 70)
        for task_key in ["task1", "task2", "task3", "task4", "task5"]:
            data = all_results.get(task_key, [])
            if data and isinstance(data, list) and "summary" in data[0]:
                s = data[0]["summary"]
                print(f"\n{s['task']}:")
                for k, v in s.items():
                    if k != "task":
                        print(f"  {k}: {v}")

        if "task1_vs_task2" in all_results:
            print(f"\nTask 1 vs 2: {all_results['task1_vs_task2']['interpretation']}")

    finally:
        print("\nStopping servers...")
        try:
            base_proc.terminate()
            base_proc.wait(timeout=15)
        except Exception:
            base_proc.kill()
        try:
            ft_proc.terminate()
            ft_proc.wait(timeout=15)
        except Exception:
            ft_proc.kill()
        kill_servers()
        print("Servers stopped.")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
