# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark FlashInfer SM90 vs CUDA vs Triton per-token-group FP8 quantization.

Compares the three code paths that ``per_token_group_quant_fp8()`` can take:
  - FI_SM90:   FlashInfer TMA-based ``fp8_quantize_1x128`` (only Hopper)
  - CUDA:      vLLM custom ``torch.ops._C.per_token_group_fp8_quant``
  - Triton:    Python fallback kernel

The FI_SM90 path requires: Hopper, group_size=128, col-major scales,
TMA-aligned scales, bf16 contiguous 2D input, and no UE8M0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import get_tma_aligned_size

# MiniMax M2.7 key dimensions
DEFAULT_K = 3072  # hidden_size
DEFAULT_INTERMEDIATE = 1536
DEFAULT_ATTN_DIM = 128 * 48  # head_dim * num_q_heads = 6144

# Decode / prefill token counts
DECODE_TOKENS = [1, 2, 4, 8, 16, 32, 64]
PREFILL_TOKENS = [256, 512, 1024, 2048, 4096, 8192]


@contextmanager
def _force_triton():
    with patch("vllm.platforms.current_platform.is_cuda", return_value=False):
        yield


@contextmanager
def _force_cuda():
    """Force CUDA path by making FlashInfer unavailable."""
    with patch(
        "vllm.model_executor.layers.quantization.utils.fp8_utils._flashinfer_sm90_per_token_group_quant_fp8",
        return_value=None,
    ):
        yield


@contextmanager
def _force_fi():
    """Force FlashInfer path — just let the normal function run."""
    yield


def _time_ms(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000  # ms


def _run_comparison(
    num_tokens: int,
    hidden_dim: int,
    group_size: int = 128,
    warmup: int = 10,
    iters: int = 200,
) -> dict[str, Any] | None:
    device = torch.device("cuda")
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16) * 8

    timings: dict[str, float] = {}

    # --- FI_SM90 path ---
    fi_q, fi_s = None, None
    try:
        result = fp8_utils._flashinfer_sm90_per_token_group_quant_fp8(
            x,
            group_size,
            column_major_scales=True,
            tma_aligned_scales=True,
            use_ue8m0=False,
            fp8_dtype=current_platform.fp8_dtype(),
            out_q=None,
            x_s=fp8_utils.create_per_token_group_quant_fp8_output_scale(
                x.shape,
                device,
                group_size,
                column_major_scales=True,
                tma_aligned_scales=True,
                scale_ue8m0=False,
            ),
        )
        if result is not None:
            fi_q, fi_s = result

            def fi_impl():
                return fp8_utils._flashinfer_sm90_per_token_group_quant_fp8(
                    x,
                    group_size,
                    column_major_scales=True,
                    tma_aligned_scales=True,
                    use_ue8m0=False,
                    fp8_dtype=current_platform.fp8_dtype(),
                    out_q=torch.empty_like(x, dtype=current_platform.fp8_dtype()),
                    x_s=fp8_utils.create_per_token_group_quant_fp8_output_scale(
                        x.shape,
                        device,
                        group_size,
                        column_major_scales=True,
                        tma_aligned_scales=True,
                        scale_ue8m0=False,
                    ),
                )  # type: ignore[misc]

            timings["FI_SM90"] = _time_ms(fi_impl, warmup, iters)
    except Exception:
        pass

    # --- CUDA path ---
    try:

        def cuda_impl():
            with _force_fi():
                with patch.object(
                    fp8_utils,
                    "_flashinfer_sm90_per_token_group_quant_fp8",
                    return_value=None,
                ):
                    return fp8_utils.per_token_group_quant_fp8(
                        x,
                        group_size,
                        column_major_scales=True,
                        tma_aligned_scales=True,
                        use_ue8m0=False,
                    )

        timings["CUDA"] = _time_ms(cuda_impl, warmup, iters)
    except Exception:
        pass

    # --- Triton path ---
    try:

        def triton_impl():
            with _force_triton():
                return fp8_utils.per_token_group_quant_fp8(
                    x,
                    group_size,
                    column_major_scales=True,
                    tma_aligned_scales=True,
                    use_ue8m0=False,
                )

        timings["Triton"] = _time_ms(triton_impl, warmup, iters)
    except Exception:
        pass

    if not timings:
        return None

    # Verify correctness if FI_SM90 succeeded
    if fi_q is not None and "CUDA" in timings:
        ref_q, ref_s = cuda_impl()
        fi_match = torch.equal(fi_q, ref_q)
        # scales may have negligible float differences
        s_close = torch.allclose(fi_s, ref_s, atol=0, rtol=0)
        correct = fi_match and s_close
    else:
        correct = None

    return {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "group_size": group_size,
        "fi_ms": timings.get("FI_SM90"),
        "cuda_ms": timings.get("CUDA"),
        "triton_ms": timings.get("Triton"),
        "fi_vs_cuda": (timings.get("FI_SM90") / timings["CUDA"] - 1) * 100
        if "FI_SM90" in timings and "CUDA" in timings
        else None,
        "correct": correct,
    }


def _fmt_speedup(baseline_ms: float | None, target_ms: float | None) -> str:
    if baseline_ms is None or target_ms is None or baseline_ms == 0:
        return "-"
    ratio = baseline_ms / target_ms
    sign = "+" if ratio > 1 else ""
    color = "\033[92m" if ratio > 1.05 else "\033[91m" if ratio < 0.95 else ""
    return f"{color}{sign}{ratio - 1:.1%}\033[0m"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tokens", type=str, default="1,2,4,8,16,32,64,256,512,1024,2048,4096,8192")
    p.add_argument("--hidden-dim", type=str, default=f"{DEFAULT_K},{DEFAULT_ATTN_DIM}")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    if not current_platform.is_cuda():
        raise RuntimeError("CUDA device required")

    args = parse_args()
    num_tokens_list = [int(n) for n in args.num_tokens.split(",")]
    hidden_dims = [int(d) for d in args.hidden_dim.split(",")]

    results: list[dict[str, Any]] = []

    col_hdr = (
        f"{'Shape':>14} | {'FI_SM90 (ms)':>13} | {'CUDA (ms)':>13} | "
        f"{'Triton (ms)':>13} | {'FI vs CUDA':>12} | {'Correct':>8}"
    )
    print(col_hdr)
    print("-" * len(col_hdr))

    for hidden_dim in hidden_dims:
        for num_tokens in num_tokens_list:
            r = _run_comparison(
                num_tokens, hidden_dim, args.group_size, args.warmup, args.iters
            )
            if r is None:
                continue
            results.append(r)

            shape_str = f"({num_tokens}, {hidden_dim})"
            fi_s = f"{r['fi_ms']:10.3f} ms" if r["fi_ms"] else "       n/a   "
            cu_s = f"{r['cuda_ms']:10.3f} ms" if r["cuda_ms"] else "       n/a   "
            tr_s = f"{r['triton_ms']:10.3f} ms" if r["triton_ms"] else "       n/a   "
            fi_vs = (
                f"{r['fi_vs_cuda']:+8.1f}%"
                if r["fi_vs_cuda"] is not None
                else "     -"
            )
            cor = "OK" if r["correct"] else ("MISMATCH" if r["correct"] is False else "-")
            print(
                f"{shape_str:>14} | {fi_s:>13} | {cu_s:>13} | {tr_s:>13} | "
                f"{fi_vs:>12} | {cor:>8}"
            )

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(results, indent=2))
