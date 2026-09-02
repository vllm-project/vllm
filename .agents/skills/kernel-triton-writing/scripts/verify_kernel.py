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
"""Verify a Triton kernel against its reference using the fixed-name contract.

Standalone script -- only Python stdlib required (torch/triton needed at
runtime for GPU verification, but not for --mock mode).
Outputs structured JSON to stdout.

Contract:
    The kernel file must export:
    - ``kernel_fn``: callable -- the Triton kernel wrapper
    - ``reference_fn``: callable -- reference implementation (same signature)
    - ``get_inputs()``: returns a list of CUDA tensors

Usage:
    .venv/bin/python verify_kernel.py kernel.py [--rtol 1e-3] [--atol 1e-3] \
        [--timeout 60] [--mock]
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
# Verification harness generation
# ---------------------------------------------------------------------------


def _build_verification_script(
    kernel_path: str,
    rtol: float,
    atol: float,
) -> str:
    """Generate a temporary verification harness script.

    The harness:
    1. Imports the kernel module using the fixed-name contract
    2. Calls ``get_inputs()`` for shared test tensors
    3. Runs ``reference_fn`` and ``kernel_fn`` with the same inputs
    4. Recursively compares outputs (handles tuples, lists, dicts, tensors,
       scalars)
    5. Prints a structured ``RESULT:`` line to stdout
    """
    abs_kernel_path = os.path.abspath(kernel_path)
    kernel_dir = os.path.dirname(abs_kernel_path)

    return textwrap.dedent(f"""\
        import sys
        import importlib.util
        import math

        sys.path.insert(0, {kernel_dir!r})

        # Import the kernel module
        _spec = importlib.util.spec_from_file_location(
            "kernel_module", {abs_kernel_path!r}
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)

        # Validate fixed-name contract
        for _attr in ("kernel_fn", "reference_fn", "get_inputs"):
            if not hasattr(_mod, _attr):
                print(f"ERROR:kernel file missing required export: {{_attr}}")
                sys.exit(1)

        import torch

        # Get shared inputs
        inputs = _mod.get_inputs()
        if not isinstance(inputs, (list, tuple)):
            print("ERROR:get_inputs() must return a list or tuple of tensors")
            sys.exit(1)

        # Clone inputs for each call to avoid in-place mutation issues
        def _clone(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            return x

        ref_inputs = [_clone(t) for t in inputs]
        kern_inputs = [_clone(t) for t in inputs]

        # Run both implementations
        ref_out = _mod.reference_fn(*ref_inputs)
        kern_out = _mod.kernel_fn(*kern_inputs)

        # Recursive comparison
        _global_max_abs = 0.0
        _global_max_rel = 0.0
        _all_correct = True
        _mismatches = []


        def _compare(ref, kern, path="root"):
            global _global_max_abs, _global_max_rel, _all_correct
            if isinstance(ref, torch.Tensor) and isinstance(kern, torch.Tensor):
                abs_diff = (kern.float() - ref.float()).abs()
                max_abs = abs_diff.max().item()
                ref_abs = ref.float().abs()
                safe_ref = torch.where(
                    ref_abs > 0, ref_abs, torch.ones_like(ref_abs)
                )
                max_rel = (abs_diff / safe_ref).max().item()
                _global_max_abs = max(_global_max_abs, max_abs)
                _global_max_rel = max(_global_max_rel, max_rel)
                if not torch.allclose(
                    kern.float(), ref.float(), rtol={rtol}, atol={atol}
                ):
                    _all_correct = False
                    _mismatches.append(
                        f"Tensor mismatch at {{path}}: "
                        f"max_abs={{max_abs:.2e}}"
                    )
            elif isinstance(ref, dict) and isinstance(kern, dict):
                for k in ref:
                    if k not in kern:
                        _all_correct = False
                        _mismatches.append(
                            f"Missing key at {{path}}: {{k!r}}"
                        )
                        return
                    _compare(ref[k], kern[k], path + f"[{{k!r}}]")
            elif isinstance(ref, (list, tuple)) and isinstance(kern, (list, tuple)):
                if len(ref) != len(kern):
                    _all_correct = False
                    _mismatches.append(
                        f"Length mismatch at {{path}}: "
                        f"{{len(ref)}} vs {{len(kern)}}"
                    )
                    return
                for i, (r, k) in enumerate(zip(ref, kern)):
                    _compare(r, k, path + f"[{{i}}]")
            elif isinstance(ref, (int, float)) and isinstance(kern, (int, float)):
                abs_d = abs(kern - ref)
                safe_r = abs(ref) if abs(ref) > 0 else 1.0
                rel_d = abs_d / safe_r
                _global_max_abs = max(_global_max_abs, abs_d)
                _global_max_rel = max(_global_max_rel, rel_d)
                if abs_d > {atol} + {rtol} * safe_r:
                    _all_correct = False
                    _mismatches.append(
                        f"Scalar mismatch at {{path}}: "
                        f"{{kern}} vs {{ref}}"
                    )
            else:
                # Non-comparable types
                if ref != kern:
                    _all_correct = False
                    _mismatches.append(
                        f"Value mismatch at {{path}}: "
                        f"{{kern!r}} vs {{ref!r}}"
                    )


        try:
            _compare(ref_out, kern_out)
            _result_parts = [
                f"passed={{_all_correct}}",
                f"max_abs={{_global_max_abs}}",
                f"max_rel={{_global_max_rel}}",
            ]
            print("RESULT:" + ",".join(_result_parts))
            for _m in _mismatches:
                print(f"MISMATCH:{{_m}}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR:{{e}}")
            sys.exit(1)
    """)


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------


def verify_kernel(
    kernel_path: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    timeout: int = 60,
) -> dict:
    """Verify kernel correctness using the fixed-name contract.

    Args:
        kernel_path: Path to Python file exporting ``kernel_fn``,
            ``reference_fn``, and ``get_inputs()``.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        timeout: Execution timeout in seconds.

    Returns:
        Dict with keys ``correct``, ``max_abs_diff``, ``max_rel_diff``,
        ``details``.
    """
    if not os.path.exists(kernel_path):
        print(f"Kernel file not found: {kernel_path}", file=sys.stderr)
        sys.exit(1)

    script = _build_verification_script(kernel_path, rtol, atol)

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

        if "RESULT:" in output:
            try:
                result_line = [
                    line for line in output.split("\n") if "RESULT:" in line
                ][0]
                parts_str = result_line.split("RESULT:")[1]
                parts = parts_str.split(",")
                passed = "True" in parts[0]
                max_abs = float(parts[1].split("=")[1])
                max_rel = float(parts[2].split("=")[1])
            except (IndexError, ValueError) as exc:
                return {
                    "correct": False,
                    "max_abs_diff": float("inf"),
                    "max_rel_diff": float("inf"),
                    "details": (
                        f"Failed to parse RESULT line: {exc}. "
                        f"Raw output: {output[:500]}"
                    ),
                }

            if passed:
                details = (
                    f"All outputs match within tolerance (rtol={rtol}, atol={atol})"
                )
            else:
                details = (
                    f"Outputs differ beyond tolerance"
                    f" (max_abs={max_abs:.2e}, rtol={rtol}, atol={atol})"
                )

            return {
                "correct": passed,
                "max_abs_diff": max_abs,
                "max_rel_diff": max_rel,
                "details": details,
            }
        elif "ERROR:" in output:
            error_msg = output.split("ERROR:")[1].strip().split("\n")[0]
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "details": f"Verification error: {error_msg}",
            }
        else:
            return {
                "correct": False,
                "max_abs_diff": float("inf"),
                "max_rel_diff": float("inf"),
                "details": f"Unexpected output: {output[:500]}",
            }

    except subprocess.TimeoutExpired:
        return {
            "correct": False,
            "max_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
            "details": f"Verification timed out after {timeout} seconds",
        }
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _mock_data(rtol: float = 1e-3, atol: float = 1e-3) -> dict:
    """Return realistic mock verification data for testing."""
    return {
        "correct": True,
        "max_abs_diff": 1.2e-7,
        "max_rel_diff": 3.4e-6,
        "details": f"All outputs match within tolerance (rtol={rtol}, atol={atol})",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for CLI invocation."""
    parser = argparse.ArgumentParser(
        description="Verify Triton kernel correctness using fixed-name contract."
    )
    parser.add_argument(
        "kernel_path",
        nargs="?",
        help="Path to Python file exporting kernel_fn, reference_fn, get_inputs().",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance (default: 1e-3).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance (default: 1e-3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Execution timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return mock data for testing (no GPU required).",
    )
    args = parser.parse_args()

    if args.mock:
        data = _mock_data(rtol=args.rtol, atol=args.atol)
    elif args.kernel_path:
        data = verify_kernel(
            kernel_path=args.kernel_path,
            rtol=args.rtol,
            atol=args.atol,
            timeout=args.timeout,
        )
    else:
        parser.error("Either --mock or kernel_path is required.")

    json.dump(data, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
