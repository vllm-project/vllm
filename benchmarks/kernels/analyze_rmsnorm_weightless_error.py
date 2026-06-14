# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""10-seed error analysis: weightless CUDA RMSNorm kernels vs native IR.

Usage (CUDA source build required):

    .venv/bin/python benchmarks/kernels/analyze_rmsnorm_weightless_error.py \\
        --dtype bfloat16 --hidden-size 4096 --seeds 10
"""

from __future__ import annotations

import argparse
import statistics

import torch

from vllm import _custom_ops as ops
from vllm import ir


def _error_stats(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    diff = (out - ref).float().abs()
    ref_abs = ref.float().abs().clamp(min=1e-8)
    rel = diff / ref_abs
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs": max(r["max_abs"] for r in rows),
        "mean_abs": statistics.mean(r["mean_abs"] for r in rows),
        "max_rel": max(r["max_rel"] for r in rows),
    }


def analyze(
    *,
    hidden_size: int,
    tokens: list[int],
    dtype: torch.dtype,
    epsilon: float,
    seeds: int,
    device: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not hasattr(torch.ops._C, "rms_norm_weightless"):
        raise RuntimeError(
            "torch.ops._C.rms_norm_weightless not found — rebuild from source"
        )

    native_rms = ir.ops.rms_norm.impls["native"].impl_fn
    native_fused = ir.ops.fused_add_rms_norm.impls["native"].impl_fn
    tol = ir.ops.rms_norm.get_tolerance(dtype)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"dtype={dtype}, hidden_size={hidden_size}, epsilon={epsilon}")
    print(f"seeds={seeds}, IR tolerance: atol={tol['atol']}, rtol={tol['rtol']}")
    print()

    for num_tokens in tokens:
        rms_rows: list[dict[str, float]] = []
        fused_out_rows: list[dict[str, float]] = []
        fused_res_rows: list[dict[str, float]] = []

        for seed in range(seeds):
            torch.manual_seed(seed)
            x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            residual = torch.randn_like(x)

            ref = native_rms(x, None, epsilon)
            out = torch.empty_like(x)
            ops.rms_norm_weightless(out, x.clone(), epsilon)
            rms_rows.append(_error_stats(ref, out))
            torch.testing.assert_close(out, ref, **tol)

            x_k = x.clone()
            r_k = residual.clone()
            ref_out, ref_res = native_fused(x.clone(), residual.clone(), None, epsilon)
            ops.fused_add_rms_norm_weightless(x_k, r_k, epsilon)
            fused_out_rows.append(_error_stats(ref_out, x_k))
            fused_res_rows.append(_error_stats(ref_res, r_k))
            torch.testing.assert_close(x_k, ref_out, **tol)
            torch.testing.assert_close(r_k, ref_res, **tol)

        rms_agg = _aggregate(rms_rows)
        fused_out_agg = _aggregate(fused_out_rows)
        fused_res_agg = _aggregate(fused_res_rows)

        print(f"tokens={num_tokens} ({seeds} seeds)")
        print(
            f"  rms_norm_weightless:           "
            f"max_abs={rms_agg['max_abs']:.6f}  "
            f"mean_abs={rms_agg['mean_abs']:.6f}  "
            f"max_rel={rms_agg['max_rel']:.6f}"
        )
        print(
            f"  fused_add_rms_norm (output): "
            f"max_abs={fused_out_agg['max_abs']:.6f}  "
            f"mean_abs={fused_out_agg['mean_abs']:.6f}  "
            f"max_rel={fused_out_agg['max_rel']:.6f}"
        )
        print(
            f"  fused_add_rms_norm (residual): "
            f"max_abs={fused_res_agg['max_abs']:.6f}  "
            f"mean_abs={fused_res_agg['mean_abs']:.6f}  "
            f"max_rel={fused_res_agg['max_rel']:.6f}"
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 1024])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    analyze(
        hidden_size=args.hidden_size,
        tokens=args.tokens,
        dtype=getattr(torch, args.dtype),
        epsilon=args.epsilon,
        seeds=args.seeds,
        device="cuda",
    )


if __name__ == "__main__":
    main()
