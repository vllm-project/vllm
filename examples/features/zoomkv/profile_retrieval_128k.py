#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile ZoomKV's single-layer retrieval at 128K CUDA Graph geometry."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

# Stage flags are read when ZoomKV modules are imported.
os.environ.setdefault("VLLM_ZOOMKV_RETRIEVE_STAGE_TIMER", "1")
os.environ.setdefault("VLLM_ZOOMKV_STAGE_TIMER", "1")
os.environ.setdefault("VLLM_ZOOMKV_NVTX", "1")

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=131072)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--nvtx-iters", type=int, default=20)
    parser.add_argument("--output-json")
    return parser.parse_args()


def timed_cuda(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    from vllm.v1.attention.ops.zoomkv import stage_timer
    from vllm.v1.attention.ops.zoomkv.retriever import (
        ZoomKVRetriever,
        ZoomKVRuntimeConfig,
    )
    from vllm.v1.attention.ops.zoomkv.state import ZoomKVBlockSummary

    device = torch.device("cuda")
    cfg = ZoomKVRuntimeConfig(max_model_len=args.context_len)
    block_size = cfg.chunk_size
    start_block = cfg.sink_size // block_size
    local_blocks = (cfg.local_size + block_size - 1) // block_size
    total_blocks = (args.context_len + block_size - 1) // block_size
    n_chunks = total_blocks - start_block - local_blocks
    if n_chunks <= 0:
        raise ValueError("context is too short for sparse retrieval")

    summary = ZoomKVBlockSummary(
        n_chunks,
        args.kv_heads,
        args.head_dim,
        block_size,
        device,
        dtype=torch.bfloat16,
    )
    keys = torch.randn(
        n_chunks,
        block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    physical = torch.arange(n_chunks, device=device, dtype=torch.int32)
    summary.update_blocks_from_key_cache(keys, physical)
    del keys

    retriever = ZoomKVRetriever(cfg)
    q = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    physical_ids = physical.view(1, -1).expand(args.batch_size, -1).contiguous()
    actual = torch.full(
        (args.batch_size,), n_chunks, device=device, dtype=torch.int32
    )
    topk = torch.empty(
        args.batch_size,
        args.kv_heads,
        cfg.final_topk,
        device=device,
        dtype=torch.int64,
    )

    def eager() -> torch.Tensor:
        return retriever._retrieve_topk_physical(
            q,
            summary,
            physical_ids,
            n_chunks,
            cfg.sink_size,
            actual_num_chunks=actual,
            topk_out=topk,
        )

    # Prime all lazy extension and scratch allocations before capture.
    eager()
    eager()
    torch.cuda.synchronize()
    stage_timer.dump_and_reset("warmup")

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        eager()
        eager()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=capture_stream):
        eager()

    eager_ms = timed_cuda(eager, args.warmup, args.iters)
    stage_report = stage_timer.dump_and_reset(
        "128k eager retrieval", decode_tokens=args.iters
    )

    def replay() -> None:
        graph.replay()

    graph_ms = timed_cuda(replay, args.warmup, args.iters)
    torch.cuda.synchronize()

    # Stable ranges for nsys: `nsys profile -t cuda,nvtx ... --nvtx-iters 100`.
    torch.cuda.nvtx.range_push("zoomkv.retrieval_graph_profile")
    for _ in range(args.nvtx_iters):
        torch.cuda.nvtx.range_push("zoomkv.retrieval_graph_replay")
        graph.replay()
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    result = {
        "context_len": args.context_len,
        "batch_size": args.batch_size,
        "n_chunks": n_chunks,
        "chunk_candidates": cfg.chunk_candidates,
        "kivi_width": cfg.kivi_width,
        "eager_ms_mean": statistics.mean(eager_ms),
        "eager_ms_p50": statistics.median(eager_ms),
        "graph_ms_mean": statistics.mean(graph_ms),
        "graph_ms_p50": statistics.median(graph_ms),
        "graph_speedup": statistics.mean(eager_ms) / statistics.mean(graph_ms),
        "topk_valid_fraction": float((topk >= 0).float().mean().item()),
    }
    print(stage_report)
    print(json.dumps(result, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as out:
            json.dump({**result, "stage_report": stage_report}, out, indent=2)


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"wall_s={time.perf_counter() - started:.3f}")
