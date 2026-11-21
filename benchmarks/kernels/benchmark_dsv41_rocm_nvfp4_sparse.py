#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark gfx950 sparse decode with FP8 versus FP4 compressed KV."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_ops

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
BLOCK_SIZE = 64
SWA_TOKENS = 128
TOPK = 512


def _identity_cos_sin(context: int, device: torch.device) -> torch.Tensor:
    return torch.cat(
        (
            torch.ones(context, ROPE_DIM // 2, device=device),
            torch.zeros(context, ROPE_DIM // 2, device=device),
        ),
        dim=-1,
    )


def _capture(call) -> torch.cuda.CUDAGraph:
    call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.accelerator.synchronize()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iterations)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    q = (
        torch.randn(
            args.batch,
            args.heads,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    main_kv = (
        torch.randn(
            args.batch * SWA_TOKENS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    main_cache = torch.empty(
        (args.batch * SWA_TOKENS // BLOCK_SIZE, BLOCK_SIZE, 584),
        dtype=torch.uint8,
        device=device,
    )
    main_slots = torch.arange(
        args.batch * SWA_TOKENS,
        dtype=torch.int64,
        device=device,
    )
    quantize_and_insert_k_cache(
        main_kv,
        main_cache,
        main_slots,
        block_size=BLOCK_SIZE,
        use_fnuz=False,
    )

    local_positions = (
        torch.arange(TOPK, device=device, dtype=torch.int64)
        * max(1, args.context // TOPK)
    ).clamp_max(args.context - 1)
    request_offsets = (
        torch.arange(args.batch, device=device, dtype=torch.int64) * args.context
    )
    extra_slots = (request_offsets[:, None] + local_positions[None, :]).flatten()
    positions = local_positions.repeat(args.batch)
    extra_kv = (
        torch.randn(
            args.batch * TOPK,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    num_extra_blocks = (args.batch * args.context + BLOCK_SIZE - 1) // BLOCK_SIZE
    fp8_cache = torch.empty(
        (num_extra_blocks, BLOCK_SIZE, 584),
        dtype=torch.uint8,
        device=device,
    )
    fp4_width = 272 if args.cache_format == "mxfp4" else 288
    fp4_cache = torch.empty(
        (num_extra_blocks, BLOCK_SIZE, fp4_width),
        dtype=torch.uint8,
        device=device,
    )
    cos_sin = _identity_cos_sin(args.context, device)
    rope_quant_insert(extra_kv, positions, cos_sin, fp8_cache, extra_slots, 1)
    rope_quant_insert(extra_kv, positions, cos_sin, fp4_cache, extra_slots, 1)

    main_indices = main_slots.to(torch.int32)
    main_indptr = (
        torch.arange(args.batch + 1, dtype=torch.int32, device=device) * SWA_TOKENS
    )
    extra_indices = extra_slots.to(torch.int32)
    extra_indptr = torch.arange(args.batch + 1, dtype=torch.int32, device=device) * TOPK
    sink = torch.zeros(args.heads, dtype=torch.float32, device=device)
    fp8_out = torch.empty_like(q)
    fp4_out = torch.empty_like(q)

    def fp8_call() -> torch.Tensor:
        return sparse_ops._rocm_sparse_attn_decode_ragged_triton(
            q,
            main_cache,
            main_indices,
            main_indptr,
            HEAD_DIM**-0.5,
            sink,
            NOPE_DIM,
            ROPE_DIM,
            extra_cache=fp8_cache,
            extra_indices=extra_indices,
            extra_indptr=extra_indptr,
            out=fp8_out,
            extra_cache_nan_free=True,
            adaptive_splits=True,
        )

    def fp4_call() -> torch.Tensor:
        return sparse_ops._rocm_sparse_attn_decode_ragged_triton(
            q,
            main_cache,
            main_indices,
            main_indptr,
            HEAD_DIM**-0.5,
            sink,
            NOPE_DIM,
            ROPE_DIM,
            extra_cache=fp4_cache,
            extra_indices=extra_indices,
            extra_indptr=extra_indptr,
            out=fp4_out,
            adaptive_splits=True,
        )

    original_split_selector = sparse_ops._decode_gfx950_num_splits
    if args.fp8_splits:
        sparse_ops._decode_gfx950_num_splits = lambda *unused: args.fp8_splits
    fp8_graph = _capture(fp8_call)
    sparse_ops._decode_gfx950_num_splits = original_split_selector
    if args.fp4_splits:
        sparse_ops._decode_gfx950_num_splits = lambda *unused: args.fp4_splits
    fp4_graph = _capture(fp4_call)
    sparse_ops._decode_gfx950_num_splits = original_split_selector
    fp8_samples = []
    fp4_samples = []
    for repeat in range(args.repeats):
        order = (
            (("fp8", fp8_graph), ("fp4", fp4_graph))
            if repeat % 2 == 0
            else (("fp4", fp4_graph), ("fp8", fp8_graph))
        )
        for name, graph in order:
            latency = _time_graph(graph, args.warmup, args.iterations)
            (fp8_samples if name == "fp8" else fp4_samples).append(latency)

    fp8_call()
    fp4_call()
    torch.accelerator.synchronize()
    fp8_f32 = fp8_out.float()
    fp4_f32 = fp4_out.float()
    cosine = torch.nn.functional.cosine_similarity(
        fp8_f32.flatten(),
        fp4_f32.flatten(),
        dim=0,
    )
    fp8_median = statistics.median(fp8_samples)
    fp4_median = statistics.median(fp4_samples)
    return {
        "batch": args.batch,
        "heads": args.heads,
        "context": args.context,
        "swa_tokens": SWA_TOKENS,
        "compressed_topk": TOPK,
        "fp8_cache_bytes": fp8_cache.numel(),
        "cache_format": args.cache_format,
        "fp4_cache_bytes": fp4_cache.numel(),
        "fp8_us": fp8_samples,
        "fp4_us": fp4_samples,
        "fp8_median_us": fp8_median,
        "fp4_median_us": fp4_median,
        "speedup": fp8_median / fp4_median,
        "fp8_splits": args.fp8_splits or "auto",
        "fp4_splits": args.fp4_splits or "auto",
        "cosine_fp8_fp4": float(cosine),
        "max_abs_fp8_fp4": float((fp8_f32 - fp4_f32).abs().max()),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--heads", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--context", type=int, choices=(8192, 131072), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--fp8-splits", type=int, default=0)
    parser.add_argument("--fp4-splits", type=int, default=0)
    parser.add_argument(
        "--cache-format",
        choices=("nvfp4", "mxfp4"),
        default="nvfp4",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
