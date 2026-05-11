# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark FlashInfer SM90 vs CUDA per-token-group FP8 quantization.

Run on H100/H200 with ``group_size=128`` and ``col_major_scales=True``.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any
from unittest.mock import patch

import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.platforms import current_platform


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _run_comparison(
    num_tokens: int,
    hidden_dim: int,
    group_size: int = 128,
    warmup: int = 10,
    iters: int = 200,
) -> dict[str, Any] | None:
    device = torch.device("cuda")
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)

    timings: dict[str, float] = {}
    q_fi = None
    s_fi = None

    # --- FI path ---
    try:
        def fi_impl():
            return fp8_utils.per_token_group_quant_fp8(
                x, group_size, column_major_scales=True, tma_aligned_scales=True,
                use_ue8m0=False,
            )

        # First call triggers JIT compile; warmup absorbs it
        q_fi, s_fi = fi_impl()
        timings["FI_SM90"] = _time_ms(fi_impl, warmup, iters)
    except Exception as e:
        print(f"  FI init failed: {e}")

    # --- CUDA path (skip FI via mock) ---
    try:
        with patch.object(
            fp8_utils,
            "_flashinfer_sm90_per_token_group_quant_fp8",
            return_value=None,
        ):

            def cuda_impl():
                return fp8_utils.per_token_group_quant_fp8(
                    x, group_size, column_major_scales=True, tma_aligned_scales=True,
                    use_ue8m0=False,
                )

            q_cuda, s_cuda = cuda_impl()
            timings["CUDA"] = _time_ms(cuda_impl, warmup, iters)
    except Exception as e:
        print(f"  CUDA init failed: {e}")

    if not timings:
        return None

    result: dict[str, Any] = {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "group_size": group_size,
        "fi_ms": timings.get("FI_SM90"),
        "cuda_ms": timings.get("CUDA"),
    }

    if "FI_SM90" in timings and "CUDA" in timings:
        result["speedup"] = timings["CUDA"] / timings["FI_SM90"]
        correct = (
            q_fi is not None
            and q_cuda is not None
            and torch.allclose(s_fi.float(), s_cuda.float(), rtol=1e-5)
            and torch.allclose(q_fi.float(), q_cuda.float(), rtol=0.15, atol=0.005)
        )
        result["correct"] = correct
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--num-tokens", type=str,
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384",
    )
    p.add_argument("--hidden-dim", type=str, default="3072,6144")
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
    header = (
        f"{'Shape':>14} | {'FI_SM90 (ms)':>13} | {'CUDA (ms)':>13} | "
        f"{'Speedup':>9} | {'Correct':>8}"
    )
    print(header)
    print("-" * len(header))

    for hidden_dim in hidden_dims:
        for num_tokens in num_tokens_list:
            r = _run_comparison(num_tokens, hidden_dim, args.group_size,
                                args.warmup, args.iters)
            if r is None:
                continue
            results.append(r)
            shape_s = f"({num_tokens}, {hidden_dim})"
            fi_s = f"{r['fi_ms']:10.3f} ms" if r.get("fi_ms") else "       n/a   "
            cu_s = f"{r['cuda_ms']:10.3f} ms" if r.get("cuda_ms") else "       n/a   "
            ratio_s = f"{r['speedup']:6.3f}x" if "speedup" in r else "    -"
            cor_s = "OK" if r.get("correct") else ("MISMATCH" if r.get("correct") is False else "-")
            print(f"{shape_s:>14} | {fi_s:>13} | {cu_s:>13} | {ratio_s:>9} | {cor_s:>8}")

    if args.json:
        print(json.dumps(results, indent=2))
