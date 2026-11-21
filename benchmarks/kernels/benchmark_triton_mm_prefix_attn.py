#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the Triton unified attention kernel on a prefix-LM
(multimodal bidirectional) prompt with many ``mm_prefix_range`` spans.

Times one prefill step of one sliding-window layer: ``--query`` new tokens
attending into a ``--context`` token prefix, with ``--spans`` frame spans of
``--span-width`` tokens spread over the sequence, as a long video prompt on
Gemma 4 produces (one span per frame). The default shape is one chunk of a
7 minute 5 fps video on Gemma 4 31B (head_dim 256, window 1024).

Calls ``triton.testing.do_bench``, which clears the L2 cache before every
measurement iteration.

Output: one JSON line to stdout.
"""

import argparse
import json

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=150_000)
    p.add_argument("--query", type=int, default=8192)
    p.add_argument("--spans", type=int, default=2100)
    p.add_argument("--span-width", type=int, default=66)
    p.add_argument("--span-gap", type=int, default=5)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--sliding-window", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--no-clamp",
        action="store_true",
        help="use the unclamped (Gemma 3 style) prefix mask",
    )
    return p.parse_args()


def _spans(num: int, seq_len: int, width: int, gap: int) -> list[tuple[int, int]]:
    spans = []
    pos = 10
    for _ in range(num):
        if pos + width > seq_len:
            break
        spans.append((pos, pos + width - 1))
        pos += width + gap
    return spans


def main() -> None:
    args = _parse()
    device = torch.device("cuda")
    seq_len = args.context + args.query
    spans = _spans(args.spans, seq_len, args.span_width, args.span_gap)
    num_blocks = (seq_len + args.block_size - 1) // args.block_size

    torch.manual_seed(0)
    q = torch.randn(
        args.query, args.heads, args.head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)[None]
    cu_seqlens_q = torch.tensor([0, args.query], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
    mm_prefix_range = (
        torch.tensor([spans], dtype=torch.int32, device=device) if spans else None
    )

    def run() -> None:
        unified_attention(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            args.query,
            seqused_k,
            seq_len,
            args.head_dim**-0.5,
            True,
            (args.sliding_window - 1, 0),
            block_table,
            0.0,
            None,
            None,
            None,
            mm_prefix_range=mm_prefix_range,
            mm_prefix_clamp_sliding_window=not args.no_clamp,
        )

    ms = triton.testing.do_bench(run, warmup=2, rep=5)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "context": args.context,
                "query": args.query,
                "spans": len(spans),
                "clamp_sliding_window": not args.no_clamp,
                "ms": round(ms, 3),
            }
        )
    )


if __name__ == "__main__":
    main()
