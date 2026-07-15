#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check fixed-input repeatability for TokenSpeed MHA extend.

This isolates whether native extend itself produces different outputs across
repeated launches on identical inputs. It compares:

* TokenSpeed native extend
* AITER unified attention

The script does not modify tokenspeed_kernel_amd.

For sliding-window cases, --sliding-window is the vLLM semantic window. The
TokenSpeed kernels are called with sliding_window - 1, matching AITER's
window_size=(sliding_window - 1, 0).
"""

from __future__ import annotations

import argparse
import math

import torch
from aiter.ops.triton.unified_attention import unified_attention

from vllm.v1.attention.ops.rocm_tokenspeed_mha import rocm_tokenspeed_mha_extend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--query-lens",
        default="1,2,7,16,64,128",
        help="Comma-separated extend query lengths.",
    )
    parser.add_argument(
        "--seq-lens",
        default="65,130,257,1024,4096,8192",
        help="Comma-separated final sequence lengths.",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=-1,
        help="vLLM semantic sliding window; use -1 for full attention.",
    )
    parser.add_argument("--random-pages", action="store_true")
    parser.add_argument("--no-sinks", action="store_true")
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    return parser.parse_args()


def tokenspeed_decode_extend_window(sliding_window: int) -> int:
    """Translate vLLM semantic sliding window to TokenSpeed kernel argument."""
    return -1 if sliding_window < 0 else sliding_window - 1


def parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def summarize(name: str, outputs: list[torch.Tensor], ref: torch.Tensor) -> None:
    exact = 0
    allclose = 0
    max_abs = 0.0
    mean_abs = 0.0
    for out in outputs:
        torch.cuda.synchronize()
        if torch.equal(out, ref):
            exact += 1
        if torch.allclose(out, ref, atol=0.01, rtol=0.01):
            allclose += 1
        diff = (out.float() - ref.float()).abs()
        max_abs = max(max_abs, diff.max().item())
        mean_abs = max(mean_abs, diff.mean().item())
    print(
        f"{name}: exact_equal={exact}/{len(outputs)} "
        f"allclose_1e-2={allclose}/{len(outputs)} "
        f"max_abs_vs_first={max_abs:.8g} "
        f"max_mean_abs_vs_first={mean_abs:.8g}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")

    query_lens = parse_ints(args.query_lens)
    seq_lens_cpu = parse_ints(args.seq_lens)
    if len(query_lens) != len(seq_lens_cpu):
        raise ValueError("--query-lens and --seq-lens must have same length")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    batch = len(query_lens)
    total_q = sum(query_lens)
    max_q = max(query_lens)
    max_seq = max(seq_lens_cpu)
    pages_per_req = math.ceil(max_seq / args.block_size)
    num_pages = batch * pages_per_req

    query_start_loc_cpu = torch.zeros(batch + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    query_start_loc = query_start_loc_cpu.to(device)
    seq_lens = torch.tensor(seq_lens_cpu, device=device, dtype=torch.int32)

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

    pages = torch.arange(num_pages, device=device, dtype=torch.int32)
    if args.random_pages:
        pages = pages[torch.randperm(num_pages, device=device)]
    block_table = pages.reshape(batch, pages_per_req)
    sinks = None if args.no_sinks else torch.randn(args.heads, device=device)

    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"arch={torch.cuda.get_device_properties(device).gcnArchName} "
        f"dtype={dtype} iters={args.iters} batch={batch} total_q={total_q} "
        f"max_q={max_q} max_seq={max_seq} "
        f"vllm_sliding_window={args.sliding_window} "
        f"tokenspeed_kernel_sliding_window="
        f"{tokenspeed_decode_extend_window(args.sliding_window)} "
        f"random_pages={args.random_pages} sinks={sinks is not None}",
        flush=True,
    )

    native_outputs: list[torch.Tensor] = []
    aiter_outputs: list[torch.Tensor] = []
    aiter_window = (-1, -1) if args.sliding_window < 0 else (args.sliding_window - 1, 0)
    ts_window = tokenspeed_decode_extend_window(args.sliding_window)

    for _ in range(args.iters):
        native = torch.empty_like(query)
        rocm_tokenspeed_mha_extend(
            query=query,
            query_start_loc=query_start_loc,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq,
            max_query_len=max_q,
            sliding_window=ts_window,
            sinks=sinks,
            output=native,
        )
        native_outputs.append(native.clone())

        aiter = torch.empty_like(query)
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
        aiter_outputs.append(aiter.clone())

    summarize("tokenspeed_native_extend_repeat", native_outputs, native_outputs[0])
    summarize("aiter_unified_repeat", aiter_outputs, aiter_outputs[0])

    def compare_pair(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
        diff = (lhs.float() - rhs.float()).abs()
        print(
            f"{name}: allclose={torch.allclose(lhs, rhs, atol=args.atol, rtol=args.rtol)} "
            f"max_abs={diff.max().item():.8g} mean_abs={diff.mean().item():.8g}",
            flush=True,
        )

    compare_pair("native_vs_aiter_first", native_outputs[0], aiter_outputs[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
