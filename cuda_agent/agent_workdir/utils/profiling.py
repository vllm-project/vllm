"""
utils/profiling.py

Benchmarks the CUDA extension model (ModelNew) against:
  1. Torch Baseline  — vanilla PyTorch eager mode
  2. Torch Compile   — torch.compile with the default backend
  3. CUDA Extension  — ModelNew using custom CUDA kernels

Do NOT modify this file.

Usage:
    python3 -m utils.profiling [--iters N] [--single-run TARGET ...]

Examples:
    python3 -m utils.profiling
    python3 -m utils.profiling --iters 20
    python3 -m utils.profiling --single-run baseline compiled extension
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from contextlib import contextmanager
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def transform_tensors(obj: Any, fn) -> Any:
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: transform_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(transform_tensors(v, fn) for v in obj)
    return obj


@contextmanager
def get_prof_ctx():
    """
    Minimal CUDA activity profiler context.
    Records device time without shape or stack tracing to minimise overhead.
    """
    yield  # Extend with torch.profiler if detailed traces are needed.


def initialize_models():
    for mod_name in ["model", "model_new"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    model_mod = importlib.import_module("model")
    model_new_mod = importlib.import_module("model_new")
    init_inputs = model_mod.get_init_inputs()

    baseline = model_mod.Model(*init_inputs).cuda().eval()
    compiled = torch.compile(model_mod.Model(*init_inputs).cuda().eval())
    extension = model_new_mod.ModelNew(*init_inputs).cuda().eval()

    return baseline, compiled, extension, model_mod


def benchmark_model(model, inputs: list, num_iters: int = 10) -> float:
    """
    Measure average device execution time over num_iters iterations.
    Returns time in microseconds.
    """
    WARMUP = 3
    with torch.no_grad():
        for _ in range(WARMUP):
            model(*inputs)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            model(*inputs)
        end.record()
        torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)  # ms
    return (elapsed_ms / num_iters) * 1e3  # convert to µs


def print_results(baseline_us: float, compiled_us: float, extension_us: float) -> None:
    speedup_vs_baseline = baseline_us / extension_us if extension_us > 0 else float("inf")
    speedup_vs_compiled = compiled_us / extension_us if extension_us > 0 else float("inf")

    print(
        f"Torch Baseline: {baseline_us:.1f}us, "
        f"Torch Compile: {compiled_us:.1f}us, "
        f"CUDA Extension: {extension_us:.1f}us"
    )
    print(
        f"  → Speedup vs Baseline: {speedup_vs_baseline:.2f}x  |  "
        f"Speedup vs torch.compile: {speedup_vs_compiled:.2f}x"
    )
    if extension_us <= compiled_us * 0.95:
        print("  ✓ PASSED: Extension achieves ≥5% speedup over torch.compile")
    else:
        print("  ✗ NOT YET: Extension does not meet the ≥5% speedup threshold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CUDA extension vs. baselines."
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Number of benchmark iterations (default: 10)"
    )
    parser.add_argument(
        "--single-run", nargs="+",
        choices=["baseline", "compiled", "extension"],
        help="Execute specified targets once without benchmarking"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_model, compiled_model, extension_model, model_mod = initialize_models()

    inputs = transform_tensors(model_mod.get_inputs(), lambda t: t.cuda())
    torch.cuda.synchronize()

    if args.single_run:
        targets = {
            "baseline": baseline_model,
            "compiled": compiled_model,
            "extension": extension_model,
        }
        with torch.no_grad():
            for target in args.single_run:
                out = targets[target](*inputs)
                print(f"[{target}] output shape: {getattr(out, 'shape', type(out))}")
        return 0

    print(f"[PROFILE] Benchmarking with {args.iters} iterations …")
    baseline_us = benchmark_model(baseline_model, inputs, args.iters)
    compiled_us = benchmark_model(compiled_model, inputs, args.iters)
    extension_us = benchmark_model(extension_model, inputs, args.iters)

    print_results(baseline_us, compiled_us, extension_us)
    return 0


if __name__ == "__main__":
    sys.exit(main())
