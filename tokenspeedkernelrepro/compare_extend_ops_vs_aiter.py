#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare TokenSpeed native extend numerics with AITER unified attention.

This is an op-level diagnostic for the vLLM ROCM_TOKENSPEED_MHA integration.
It builds synthetic request-level extend batches over the same paged KV cache
and compares:

* TokenSpeed native extend
* AITER unified attention, matching the ROCM_AITER_UNIFIED_ATTN op path

The script does not modify tokenspeed_kernel_amd.

By default, sliding-window cases use the corrected integration convention:
TokenSpeed receives sliding_window - 1 for the same vLLM semantic window where
AITER receives window_size=(sliding_window - 1, 0). Use
--tokenspeed-sliding-delta 0 only to reproduce the old mismatch.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from aiter.ops.triton.unified_attention import unified_attention

from vllm.v1.attention.ops.rocm_tokenspeed_mha import rocm_tokenspeed_mha_extend


@dataclass(frozen=True)
class Case:
    name: str
    query_lens: list[int]
    final_seq_lens: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-sinks", action="store_true")
    parser.add_argument("--random-pages", action="store_true")
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument(
        "--tokenspeed-sliding-delta",
        type=int,
        default=-1,
        help=(
            "Offset applied only to the TokenSpeed sliding_window argument. "
            "Default -1 matches the corrected vLLM integration convention. "
            "Use 0 to reproduce the old raw-window mismatch."
        ),
    )
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    return parser.parse_args()


def make_cases(include_large: bool) -> list[Case]:
    cases = [
        Case("q1_decode_like", [1, 1, 1, 1], [64, 128, 512, 1024]),
        Case("short_ragged", [1, 2, 7, 16], [65, 130, 257, 1024]),
        Case("q8_batch", [8, 8, 8, 8], [1024, 1024, 1024, 1024]),
        Case("q16_batch", [16, 16, 16, 16], [1024, 2048, 3072, 4096]),
        Case("q64_batch", [64, 64], [2048, 4096]),
    ]
    if include_large:
        cases.extend(
            [
                Case("gptoss_q128", [128, 128, 128, 128], [2048, 4096, 6144, 8192]),
                Case("gptoss_q256", [256, 256], [4096, 8192]),
                Case("gptoss_ragged_large", [64, 257, 513], [4096, 6145, 8192]),
            ]
        )
    return cases


def compare(
    name: str, lhs: torch.Tensor, rhs: torch.Tensor, atol: float, rtol: float
) -> None:
    torch.cuda.synchronize()
    diff = (lhs.float() - rhs.float()).abs()
    flat = diff.flatten()
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.5, 0.9, 0.99, 0.999], device=flat.device),
    )
    print(
        f"{name}: allclose={torch.allclose(lhs, rhs, atol=atol, rtol=rtol)} "
        f"max_abs={diff.max().item():.8g} "
        f"mean_abs={diff.mean().item():.8g} "
        f"p50={quantiles[0].item():.8g} "
        f"p90={quantiles[1].item():.8g} "
        f"p99={quantiles[2].item():.8g} "
        f"p999={quantiles[3].item():.8g}",
        flush=True,
    )


def run_case(
    case: Case,
    sliding_window: int,
    args: argparse.Namespace,
    dtype: torch.dtype,
    device: torch.device,
    sinks: torch.Tensor | None,
) -> None:
    batch = len(case.query_lens)
    total_q = sum(case.query_lens)
    max_q = max(case.query_lens)
    max_seq = max(case.final_seq_lens)
    pages_per_req = math.ceil(max_seq / args.block_size)
    num_pages = batch * pages_per_req

    query_start_loc_cpu = torch.zeros(batch + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.tensor(case.query_lens, dtype=torch.int32).cumsum(0)
    query_start_loc = query_start_loc_cpu.to(device)
    seq_lens = torch.tensor(case.final_seq_lens, device=device, dtype=torch.int32)

    query = torch.randn(total_q, args.heads, args.head_dim, device=device, dtype=dtype)
    key_cache = torch.randn(
        num_pages,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)

    physical_pages = torch.arange(num_pages, device=device, dtype=torch.int32)
    if args.random_pages:
        physical_pages = physical_pages[torch.randperm(num_pages, device=device)]
    block_table = physical_pages.reshape(batch, pages_per_req)

    native = torch.empty_like(query)
    aiter = torch.empty_like(query)
    tokenspeed_sliding_window = (
        sliding_window
        if sliding_window < 0
        else sliding_window + args.tokenspeed_sliding_delta
    )

    rocm_tokenspeed_mha_extend(
        query=query,
        query_start_loc=query_start_loc,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=max_seq,
        max_query_len=max_q,
        sliding_window=tokenspeed_sliding_window,
        sinks=sinks,
        output=native,
    )

    aiter_window = (-1, -1) if sliding_window < 0 else (sliding_window - 1, 0)
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=aiter,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_q,
        seqused_k=seq_lens,
        max_seqlen_k=max_seq,
        softmax_scale=1.0 / math.sqrt(args.head_dim),
        causal=True,
        window_size=aiter_window,
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
        output_scale=None,
    )

    mode = "full" if sliding_window < 0 else f"sliding{sliding_window}"
    print(
        f"\ncase={case.name} mode={mode} batch={batch} total_q={total_q} "
        f"max_q={max_q} max_seq={max_seq} "
        f"vllm_sliding_window={sliding_window} "
        f"tokenspeed_kernel_sliding_window={tokenspeed_sliding_window} "
        f"aiter_window={aiter_window}",
        flush=True,
    )
    compare("tokenspeed_native_extend_vs_aiter", native, aiter, args.atol, args.rtol)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires a ROCm/CUDA-visible GPU.")
    if args.heads % args.kv_heads != 0:
        raise ValueError("--heads must be divisible by --kv-heads")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    sinks = (
        None
        if args.no_sinks
        else torch.randn(args.heads, device=device, dtype=torch.float32)
    )

    print(
        "device="
        f"{torch.cuda.get_device_name(device)} "
        f"arch={torch.cuda.get_device_properties(device).gcnArchName} "
        f"dtype={dtype} sinks={sinks is not None} "
        f"random_pages={args.random_pages} "
        f"tokenspeed_sliding_delta={args.tokenspeed_sliding_delta} "
        f"atol={args.atol} rtol={args.rtol}",
        flush=True,
    )

    for case in make_cases(args.include_large):
        run_case(case, -1, args, dtype, device, sinks)
        run_case(case, 128, args, dtype, device, sinks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
