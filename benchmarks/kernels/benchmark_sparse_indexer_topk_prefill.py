# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare in-tree vs AITER sparse-indexer prefill top-k at GLM-5.3-Flash shapes.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_sparse_indexer_topk_prefill.py \
        --num-rows 1024 --num-cols 131072 --topk 512

"""

import argparse
import os
import time

import torch

from vllm import _custom_ops  # noqa: F401


def _make_inputs(num_rows, num_cols, causal, device):
    logits = torch.randn(num_rows, num_cols, dtype=torch.float32, device=device)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    if causal:
        ramp = torch.arange(1, num_rows + 1, device=device, dtype=torch.float32)
        row_ends = (ramp * (num_cols / num_rows)).ceil().to(torch.int32)
    else:
        row_ends = torch.full((num_rows,), num_cols, dtype=torch.int32, device=device)
    return logits, row_starts, row_ends


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def _selected_set(indices, row_starts):
    valid = indices >= 0
    return [set(r[m].tolist()) for r, m in zip(indices.cpu(), valid.cpu())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rows", type=int, default=1024)
    parser.add_argument("--num-cols", type=int, default=131072)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    logits, row_starts, row_ends = _make_inputs(
        args.num_rows, args.num_cols, args.causal, device
    )
    print(
        f"rows={args.num_rows} cols={args.num_cols} k={args.topk} "
        f"causal={args.causal} logits={logits.numel() * 4 / 2**20:.0f} MiB "
        f"path={os.environ.get('TOPK_FORCE_PATH', 'auto')}"
    )

    native = torch.empty(args.num_rows, args.topk, dtype=torch.int32, device=device)

    def run_native():
        torch.ops._C.top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            native,
            args.num_rows,
            logits.stride(0),
            logits.stride(1),
            args.topk,
        )

    native_ms = _time(run_native, args.iters, args.warmup)
    print(f"in-tree  topKPerRowPrefill : {native_ms:8.3f} ms")

    try:
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
            _launch_aiter_top_k_per_row_prefill,
        )
    except ImportError as exc:
        print(f"AITER path unavailable: {exc}")
        return

    aiter_out = torch.empty(args.num_rows, args.topk, dtype=torch.int32, device=device)

    def run_aiter():
        _launch_aiter_top_k_per_row_prefill(
            logits, row_starts, row_ends, aiter_out, args.topk
        )

    try:
        aiter_ms = _time(run_aiter, args.iters, args.warmup)
    except Exception as exc:
        print(f"AITER path failed: {type(exc).__name__}: {exc}")
        return
    print(f"AITER    top_k_per_row_prefill: {aiter_ms:8.3f} ms")
    print(f"speedup  : {native_ms / aiter_ms:.2f}x")

    ref = _selected_set(native, row_starts)
    got = _selected_set(aiter_out, row_starts)
    mismatched = sum(1 for a, b in zip(ref, got) if a != b)
    worst = max((len(a ^ b) for a, b in zip(ref, got)), default=0)
    print(f"rows differing: {mismatched}/{args.num_rows} (worst sym-diff {worst})")


if __name__ == "__main__":
    main()
