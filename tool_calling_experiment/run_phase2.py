#!/usr/bin/env python3
"""Runner for Phase 2: connects to pre-started servers.

Usage:
  # Start servers manually first, then:
  python3 run_phase2.py --mode 2b    # Tasks 36-40 (2B only)
  python3 run_phase2.py --mode 8b    # Task 41 (8B only)
  python3 run_phase2.py --mode all   # All tasks
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any

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

import requests  # noqa: E402, I001  # type: ignore[import-not-found]

from phase2_geometry import (  # noqa: E402  # type: ignore[import-not-found]
    RESULTS_PATH,
    run_task36,
    run_task37,
    run_task38,
    run_task39,
    run_task40,
    run_task41,
    run_task42,
)

MODEL_2B = "/fsx/models/Qwen3-VL-2B-Instruct"
MODEL_8B = "/fsx/models/Qwen3-VL-8B-Instruct"
PORT_2B = 8334
PORT_8B = 8335
GPU_2B = 4
GPU_8B = 5


def _wait_healthy(port: int, timeout: int = 600) -> bool:
    """Poll /health until 200 or timeout."""
    url = f"http://localhost:{port}/health"
    for i in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"  Port {port} healthy ({i}s)")
                return True
        except Exception:
            pass
        if i % 30 == 0 and i > 0:
            print(f"  Waiting for port {port}... {i}s")
        time.sleep(1)
    return False


def _start_server(
    model: str, port: int, gpu: int,
) -> subprocess.Popen:  # type: ignore[type-arg]
    """Start vllm serve in background."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        "vllm", "serve", model,
        "--trust-remote-code",
        "--max-model-len", "8192",
        "--enforce-eager",
        "--port", str(port),
        "--gpu-memory-utilization", "0.8",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    log = f"/tmp/vllm_{port}.log"
    f = open(log, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
    )
    print(f"  Started PID {proc.pid}, log: {log}")
    return proc


def _kill_server(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    """Kill a server process and its children."""
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="all",
        choices=["2b", "8b", "all", "token"],
    )
    args = parser.parse_args()

    start_time = time.time()

    # Load existing results if any
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {RESULTS_PATH}")
    else:
        all_results: dict[str, Any] = {}

    url_2b = f"http://localhost:{PORT_2B}"
    url_8b = f"http://localhost:{PORT_8B}"

    if args.mode in ("2b", "all"):
        # Check or start 2B
        if not _wait_healthy(PORT_2B, timeout=5):
            print("Starting 2B server...")
            proc_2b = _start_server(
                MODEL_2B, PORT_2B, GPU_2B,
            )
            if not _wait_healthy(PORT_2B, timeout=600):
                print("ERROR: 2B server not healthy")
                return
        else:
            proc_2b = None

        # Run 2B tasks
        for name, fn in [
            ("task36", lambda: run_task36(url_2b)),
            ("task37", lambda: run_task37()),
            ("task38", lambda: run_task38(url_2b)),
            ("task39", lambda: run_task39(url_2b)),
            ("task40", lambda: run_task40(url_2b)),
        ]:
            try:
                all_results[name] = fn()
            except Exception as e:
                print(f"  {name} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[name] = {"error": str(e)}
            # Save after each task
            with open(RESULTS_PATH, "w") as f:
                json.dump(
                    all_results, f, indent=2, default=str,
                )

        if proc_2b:
            _kill_server(proc_2b)

    if args.mode in ("8b", "all"):
        # Check or start 8B
        if not _wait_healthy(PORT_8B, timeout=5):
            print("Starting 8B server...")
            proc_8b = _start_server(
                MODEL_8B, PORT_8B, GPU_8B,
            )
            if not _wait_healthy(PORT_8B, timeout=600):
                print("ERROR: 8B server not healthy")
                # Try to save what we have
                all_results["task41"] = {
                    "error": "8B server failed to start",
                }
                with open(RESULTS_PATH, "w") as f:
                    json.dump(
                        all_results, f,
                        indent=2, default=str,
                    )
                return
        else:
            proc_8b = None

        try:
            all_results["task41"] = run_task41(url_8b)
        except Exception as e:
            print(f"  task41 failed: {e}")
            import traceback
            traceback.print_exc()
            all_results["task41"] = {"error": str(e)}

        if proc_8b:
            _kill_server(proc_8b)

    if args.mode in ("token", "all"):
        try:
            all_results["task42"] = run_task42(all_results)
        except Exception as e:
            print(f"  task42 failed: {e}")
            all_results["task42"] = {"error": str(e)}

    # Metadata and comparison
    elapsed = time.time() - start_time
    all_results["metadata"] = {
        "phase": "Phase 2: Road Geometry Deep Dive",
        "tasks": "36-42",
        "total_elapsed_seconds": round(elapsed, 1),
        "total_elapsed_minutes": round(elapsed / 60, 1),
        "model_2b": MODEL_2B,
        "model_8b": MODEL_8B,
    }

    t36 = all_results.get("task36", {})
    t41 = all_results.get("task41", {})
    if not isinstance(t36, dict) or "error" in t36:
        t36 = {}
    if not isinstance(t41, dict) or "error" in t41:
        t41 = {}
    all_results["comparison_2b_vs_8b"] = {
        "task36_2b_parse_rate": t36.get(
            "waypoint_parse_rate", 0,
        ),
        "task41_8b_parse_rate": t41.get(
            "waypoint_parse_rate", 0,
        ),
        "task36_2b_drivable_rate": t36.get(
            "in_drivable_rate", 0,
        ),
        "task41_8b_drivable_rate": t41.get(
            "in_drivable_rate", 0,
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    em = elapsed / 60
    print(f"\nResults: {RESULTS_PATH}")
    print(f"Total time: {em:.1f} minutes")


if __name__ == "__main__":
    main()
