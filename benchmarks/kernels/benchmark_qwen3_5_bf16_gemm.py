# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Qwen3.5 BF16 decode GEMMs on ROCm.

The default shapes come from a Qwen3.8-27B MTP=5 decode trace.  ``M=1`` and
``M=24`` are draft forwards; ``M=6`` and ``M=144`` are the corresponding
target forwards.  The projection dimensions are ordered by frequency in the
model (65, 65, 65, 48, 48, and 17 calls per target forward respectively).

Example:
    python benchmarks/kernels/benchmark_qwen3_5_bf16_gemm.py \
        --m 1 6 24 144 --providers current torch hipblaslt aiter triton
"""

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units

PROJECTIONS = {
    "mlp_up": (34816, 5120),
    "mlp_down": (5120, 17408),
    "common_qkv": (6144, 5120),
    "gdn_in": (16384, 5120),
    "gdn_aux": (96, 5120),
    "full_attn_qkv": (14336, 5120),
}


def _provider(
    name: str, x: torch.Tensor, weight: torch.Tensor
) -> Callable[[], torch.Tensor]:
    if name == "current":
        return lambda: rocm_unquantized_gemm_impl(x, weight)
    if name == "torch":
        return lambda: F.linear(x, weight)
    if name == "wvsplitk":
        if not 0 < x.shape[0] <= 6:
            raise ValueError("wvSplitK supports M in [1, 6]")
        return lambda: ops.wvSplitK(weight, x, num_compute_units())
    if name == "hipblaslt":
        from aiter.tuned_gemm import hipb_gemm

        return lambda: hipb_gemm(x, weight, -1)
    if name == "aiter":
        from aiter.tuned_gemm import tgemm

        return lambda: tgemm.mm(x, weight)
    if name == "triton":
        from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

        return lambda: gemm_a16w16(x, weight)
    raise ValueError(f"unknown provider: {name}")


def _bench(fn: Callable[[], torch.Tensor], warmup_ms: int, rep_ms: int) -> float:
    return triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        return_mode="median",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 6, 24, 144])
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=PROJECTIONS,
        default=list(PROJECTIONS),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["current", "torch", "wvsplitk", "hipblaslt", "aiter", "triton"],
        default=["current", "torch", "hipblaslt", "aiter", "triton"],
    )
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    print("projection M N K provider latency_us speedup_vs_current max_abs")
    for projection in args.projections:
        n, k = PROJECTIONS[projection]
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        for m in args.m:
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            reference = F.linear(x, weight)
            results: dict[str, tuple[float, float]] = {}
            for provider in args.providers:
                try:
                    fn = _provider(provider, x, weight)
                    output = fn()
                    torch.cuda.synchronize()
                    max_abs = (output.float() - reference.float()).abs().max().item()
                    results[provider] = (
                        _bench(fn, args.warmup_ms, args.rep_ms) * 1000,
                        max_abs,
                    )
                except (
                    ImportError,
                    OSError,
                    RuntimeError,
                    ValueError,
                    AssertionError,
                ) as exc:
                    print(f"{projection} {m} {n} {k} {provider} SKIP {exc}")
            baseline = results.get("current", (float("nan"), 0.0))[0]
            for provider, (latency_us, max_abs) in results.items():
                print(
                    f"{projection} {m} {n} {k} {provider} "
                    f"{latency_us:.3f} {baseline / latency_us:.3f} {max_abs:.6g}"
                )
        del weight


if __name__ == "__main__":
    main()
