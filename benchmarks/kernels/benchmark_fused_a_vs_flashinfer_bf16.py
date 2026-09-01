# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the fused-A GEMM against FlashInfer CuTeDSL ``mm_bf16`` and cuBLAS.

Every ``(N, K)`` shape and token count in the Kimi-K3 and GLM-5.2 fused-A
tables is measured on all three backends, so the tables can be re-justified (or
retired) whenever the FlashInfer CuTeDSL BF16 kernel moves. The CuTeDSL backend
is measured after a FlashInfer autotuning pass, which profiles both its direct
and split-K tactic spaces rather than picking one from its shape heuristic.

Timing is CUPTI kernel time under CUDA graph replay with a cold L2, so it
excludes host-side dispatch -- which dominates an eager call to the CuTeDSL
backend by two orders of magnitude but is captured away in production.

Usage:
    python benchmarks/kernels/benchmark_fused_a_vs_flashinfer_bf16.py
    python benchmarks/kernels/benchmark_fused_a_vs_flashinfer_bf16.py --json out.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from collections.abc import Callable

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm import _custom_ops as ops
from vllm.models.deepseek_v32.nvidia.glm52_low_latency_gemm import GLM52_FUSED_A_TABLE
from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
    KIMI_K3_PROJECTIONS,
    KIMI_K3_PROJECTIONS_SM90,
    KIMI_K3_PROJECTIONS_SM100,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import autotune, flashinfer_bf16_mm

WARMUP = 10
# The CuTeDSL BF16 kernels only cover M <= 32.
MAX_TOKENS = 32

TABLES = {
    "kimi_k3_sm103": KIMI_K3_PROJECTIONS,
    "kimi_k3_sm100": KIMI_K3_PROJECTIONS_SM100,
    "kimi_k3_sm90": KIMI_K3_PROJECTIONS_SM90,
    "glm52": GLM52_FUSED_A_TABLE,
}


def all_cases() -> list[tuple[int, int, int]]:
    """Every (N, K, M) any fused-A table routes onto the kernel."""
    cases: dict[tuple[int, int], set[int]] = {}
    for table in TABLES.values():
        for shape, tokens in table.items():
            cases.setdefault(shape, set()).update(tokens)
    return sorted(
        (n, k, m) for (n, k), tokens in cases.items() for m in tokens if m <= MAX_TOKENS
    )


def bench_us(fn: Callable[[], object]) -> float:
    """Median CUPTI kernel time, warmed up so JIT never lands in the measurement."""
    for _ in range(WARMUP):
        fn()
    torch.accelerator.synchronize()
    samples = bench_gpu_time_with_cupti(fn, use_cuda_graph=True)
    return statistics.median(samples) * 1e3


def cosine(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.flatten(), dim=0
    ).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default=None, help="write rows as JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not current_platform.is_cuda():
        raise RuntimeError("CUDA is required")
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        raise RuntimeError("dsv3_fused_a_gemm was not built")
    if importlib.util.find_spec("cupti") is None:
        # Without it bench_gpu_time_with_cupti silently falls back to CUDA
        # events, which measure host dispatch rather than kernel time.
        raise RuntimeError("cupti-python is required: uv pip install cupti-python")

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    pdl = current_platform.is_arch_support_pdl()
    capability = current_platform.get_device_capability()
    print(
        f"# {torch.cuda.get_device_name(0)} "
        f"(SM{capability.major}{capability.minor}), pdl={pdl}"
    )

    rows = []
    for n, k, m in all_cases():
        x = torch.randn(m, k, dtype=torch.bfloat16)
        weight = torch.randn(n, k, dtype=torch.bfloat16)
        weight_t = weight.t()
        fused_a_out = torch.empty(m, n, dtype=torch.bfloat16)
        reference = x.float() @ weight.float().t()

        def run_cublas(x=x, weight=weight):
            return torch.nn.functional.linear(x, weight)

        def run_fused_a(x=x, weight_t=weight_t, out=fused_a_out):
            ops.dsv3_fused_a_gemm(out, x, weight_t, enable_pdl=True)
            return out

        def run_cutedsl(x=x, weight_t=weight_t):
            return flashinfer_bf16_mm(x, weight_t, None, pdl, "cute-dsl")

        # Autotune profiles both CuTeDSL tactic spaces; the choice is cached
        # per shape, so the timed calls below reuse it.
        with autotune(True):
            run_cutedsl()
        torch.accelerator.synchronize()

        candidates = {
            "cublas": run_cublas,
            "fused_a": run_fused_a,
            "cutedsl": run_cutedsl,
        }
        timings = {}
        for name, fn in candidates.items():
            similarity = cosine(fn(), reference)
            if similarity <= 0.999:
                raise RuntimeError(f"{name} wrong at {(n, k, m)}: cosine {similarity}")
            timings[name] = bench_us(fn)

        flops = 2 * m * n * k
        row = {
            "n": n,
            "k": k,
            "m": m,
            **{f"{name}_us": us for name, us in timings.items()},
            "cutedsl_tflops": flops / (timings["cutedsl"] * 1e6),
            "cutedsl_vs_fused_a": timings["fused_a"] / timings["cutedsl"],
            "best": min(timings, key=timings.get),
        }
        rows.append(row)
        print(
            f"N={n:6d} K={k:6d} M={m:3d} | "
            f"cublas {timings['cublas']:7.2f}us  "
            f"fused_a {timings['fused_a']:7.2f}us  "
            f"cutedsl {timings['cutedsl']:7.2f}us  "
            f"({row['cutedsl_tflops']:6.1f} TFLOP/s)  "
            f"cutedsl/fused_a speedup {row['cutedsl_vs_fused_a']:5.2f}x  "
            f"best={row['best']}"
        )

    wins = sum(row["best"] == "cutedsl" for row in rows)
    fused_a_wins = sum(row["best"] == "fused_a" for row in rows)
    print(
        f"\n# {len(rows)} cases: cutedsl best in {wins}, "
        f"fused_a best in {fused_a_wins}, "
        f"cublas best in {len(rows) - wins - fused_a_wins}"
    )
    speedups = sorted(row["cutedsl_vs_fused_a"] for row in rows)
    print(
        f"# cutedsl vs fused_a speedup: min {speedups[0]:.2f}x  "
        f"median {statistics.median(speedups):.2f}x  max {speedups[-1]:.2f}x"
    )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
