#!/usr/bin/env -S uv run --python .venv/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileComment: Copied from NVIDIA TensorRT-LLM at commit
# 395985c025c8d1cf5aa842bc752b337ba88721b6.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark a Triton kernel using the fixed-name contract.

Standalone script -- only Python stdlib required (torch/triton needed at
runtime for GPU benchmarking, but not for --mock mode).
Outputs structured JSON to stdout.

Contract:
    The kernel file must export:
    - ``kernel_fn``: callable -- the Triton kernel wrapper
    - ``reference_fn``: callable -- reference implementation (optional)
    - ``get_inputs()``: returns a list of CUDA tensors

Usage:
    .venv/bin/python benchmark_kernel.py kernel.py [--warmup 10] [--iters 40] \
        [--timeout 120] [--mock]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

# ---------------------------------------------------------------------------
# Benchmark harness generation
# ---------------------------------------------------------------------------


def _build_benchmark_script(
    kernel_path: str,
    warmup: int,
    iters: int,
) -> str:
    """Generate a temporary benchmark harness script.

    The harness:
    1. Imports the kernel module using the fixed-name contract
    2. Calls ``get_inputs()`` for shared test tensors
    3. Benchmarks ``kernel_fn`` with CUDA events
    4. If ``reference_fn`` exists, benchmarks it too
    5. Prints a structured ``BENCHMARK:`` line to stdout
    """
    abs_kernel_path = os.path.abspath(kernel_path)
    kernel_dir = os.path.dirname(abs_kernel_path)

    return textwrap.dedent(f"""\
        import sys
        import os
        import importlib.util
        import torch

        sys.path.insert(0, {kernel_dir!r})

        # Import the kernel module
        _spec = importlib.util.spec_from_file_location(
            "kernel_module", {abs_kernel_path!r}
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        # Validate fixed-name contract
        for _attr in ("kernel_fn", "get_inputs"):
            if not hasattr(_mod, _attr):
                print(f"ERROR:kernel file missing required export: {{_attr}}")
                sys.exit(1)

        def gpu_benchmark(fn, args, warmup, iters):
            for _ in range(warmup):
                fn(*args)
            torch.accelerator.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn(*args)
            end.record()
            torch.accelerator.synchronize()
            return start.elapsed_time(end) / iters

        # Get inputs
        inputs = _mod.get_inputs()
        if not isinstance(inputs, (list, tuple)):
            print("ERROR:get_inputs() must return a list or tuple of tensors")
            sys.exit(1)

        # Clone inputs for each benchmark to avoid in-place mutation
        def _clone(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            return x

        # Benchmark kernel_fn
        kern_inputs = [_clone(t) for t in inputs]
        kernel_ms = gpu_benchmark(
            _mod.kernel_fn, kern_inputs,
            warmup={warmup}, iters={iters},
        )

        # Benchmark reference_fn if available
        if hasattr(_mod, "reference_fn"):
            ref_inputs = [_clone(t) for t in inputs]
            ref_ms = gpu_benchmark(
                _mod.reference_fn, ref_inputs,
                warmup={warmup}, iters={iters},
            )
            speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0
            print(
                f"BENCHMARK:kernel_ms={{kernel_ms:.6f}},"
                f"ref_ms={{ref_ms:.6f}},"
                f"speedup={{speedup:.4f}}"
            )
        else:
            print(f"BENCHMARK:kernel_ms={{kernel_ms:.6f}}")
    """)


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------


def benchmark_kernel(
    kernel_path: str,
    warmup: int = 10,
    iters: int = 40,
    timeout: int = 120,
) -> dict:
    """Benchmark kernel performance using the fixed-name contract.

    Args:
        kernel_path: Path to Python file exporting ``kernel_fn``,
            optionally ``reference_fn``, and ``get_inputs()``.
        warmup: Number of warmup iterations.
        iters: Number of measured iterations.
        timeout: Execution timeout in seconds.

    Returns:
        Dict with keys ``kernel_time_ms``, ``reference_time_ms``,
        ``speedup``, ``warmup_iters``, ``benchmark_iters``.
    """
    if not os.path.exists(kernel_path):
        print(f"Kernel file not found: {kernel_path}", file=sys.stderr)
        sys.exit(1)

    script = _build_benchmark_script(kernel_path, warmup, iters)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_file.write(script)
        script_path = script_file.name

    working_dir = os.path.dirname(os.path.abspath(kernel_path))

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )

        output = result.stdout + result.stderr

        if "ERROR:" in output:
            error_msg = output.split("ERROR:")[1].strip().split("\n")[0]
            print(f"Benchmark error: {error_msg}", file=sys.stderr)
            sys.exit(1)

        if "BENCHMARK:" not in output:
            print(f"Benchmark failed:\n{output[:1000]}", file=sys.stderr)
            sys.exit(1)

        result_line = [line for line in output.split("\n") if "BENCHMARK:" in line][0]
        parts = result_line.split("BENCHMARK:")[1].split(",")
        try:
            parsed = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in parts}
        except (IndexError, ValueError) as exc:
            return {
                "kernel_time_ms": None,
                "reference_time_ms": None,
                "speedup": None,
                "warmup_iters": warmup,
                "benchmark_iters": iters,
                "error": f"Failed to parse BENCHMARK line: {exc}",
            }

        result_dict: dict = {
            "kernel_time_ms": parsed["kernel_ms"],
            "warmup_iters": warmup,
            "benchmark_iters": iters,
        }

        if "ref_ms" in parsed:
            result_dict["reference_time_ms"] = parsed["ref_ms"]
            result_dict["speedup"] = parsed["speedup"]
        else:
            result_dict["reference_time_ms"] = None
            result_dict["speedup"] = None

        return result_dict

    except subprocess.TimeoutExpired:
        print(f"Benchmark timed out after {timeout} seconds", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _mock_data(warmup: int = 10, iters: int = 40) -> dict:
    """Return realistic mock benchmark data for testing."""
    return {
        "kernel_time_ms": 0.45,
        "reference_time_ms": 1.23,
        "speedup": 2.73,
        "warmup_iters": warmup,
        "benchmark_iters": iters,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Benchmark a Triton kernel using the fixed-name contract."
    )
    parser.add_argument(
        "kernel_path",
        nargs="?",
        help="Path to Python file exporting kernel_fn, reference_fn, get_inputs().",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=40,
        help="Measured iterations (default: 40).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Execution timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return mock data for testing (no GPU required).",
    )
    args = parser.parse_args()

    if args.mock:
        data = _mock_data(warmup=args.warmup, iters=args.iters)
    elif args.kernel_path:
        data = benchmark_kernel(
            kernel_path=args.kernel_path,
            warmup=args.warmup,
            iters=args.iters,
            timeout=args.timeout,
        )
    else:
        parser.error("Either --mock or kernel_path is required.")

    json.dump(data, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
