# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx1151 native-AITER vs skinny FP8 blockscale GEMM microbenchmark.

Times *already quantized* GEMM calls on an identical native-AITER FP8
weight view, including the ROCm production FP8 row padding. CUDA Graph
replay amortizes Python dispatch/launch overhead. E2E data is separate.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

import vllm._custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms.rocm import on_gfx1151

CANDIDATE_M = (1, 2, 3, 4, 5, 6)
ALL_M = (1, 2, 3, 4, 5, 6, 8)


def make_case(m: int, n: int, pad_cols: int):
    dtype = torch.float8_e4m3fn
    k = 4096
    # Quantized synthetic payloads, same physical weight and scales for both.
    a = (torch.randn((m, k), device="cuda") * 0.1).to(dtype).contiguous()
    w = (torch.randn((n, k), device="cuda") * 0.1).to(dtype).contiguous()
    if pad_cols:
        w = torch.nn.functional.pad(w, (0, pad_cols))[:, :k]
    a_scale = torch.full((m, 32), 0.1, device="cuda", dtype=torch.float32)
    w_scale = torch.full((n // 128, 32), 0.1, device="cuda", dtype=torch.float32)
    assert w.stride() == (k + pad_cols, 1)
    return a, w, a_scale, w_scale


def bench_graph(fn, *, graph_calls: int, replays: int) -> float:
    """Return average device time per GEMM in microseconds."""
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_calls):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (graph_calls * replays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pad-cols", type=int, choices=(0, 256), default=256)
    parser.add_argument("--graph-calls", type=int, default=16)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    if not on_gfx1151():
        raise RuntimeError("This benchmark is only for gfx1151")
    if args.graph_calls <= 0 or args.replays <= 0:
        parser.error("graph-calls and replays must be positive")
    cu = torch.cuda.get_device_properties("cuda").multi_processor_count
    rows = []

    for n in (1536, 4096):
        for m in ALL_M:
            a, w, a_scale, w_scale = make_case(m, n, args.pad_cols)

            def native_aiter(
                a=a,
                w=w,
                a_scale=a_scale,
                w_scale=w_scale,
            ):
                return rocm_aiter_ops.triton_gemm_a8w8_blockscale(
                    a,
                    w,
                    a_scale,
                    w_scale,
                    [128, 128],
                    output_dtype=torch.bfloat16,
                )

            def skinny(
                a=a,
                w=w,
                a_scale=a_scale,
                w_scale=w_scale,
            ):
                return ops.wvSplitKQBlockScale(w, a, a_scale, w_scale, cu)

            aiter_us = bench_graph(
                native_aiter, graph_calls=args.graph_calls, replays=args.replays
            )
            skinny_us = None
            if m in CANDIDATE_M:
                reference = native_aiter()
                actual = skinny()
                torch.testing.assert_close(actual, reference, rtol=5e-2, atol=1e-1)
                skinny_us = bench_graph(
                    skinny, graph_calls=args.graph_calls, replays=args.replays
                )
            row = {
                "n": n,
                "m": m,
                "k": 4096,
                "pad_cols": args.pad_cols,
                "weight_stride_n": w.stride(0),
                "aiter_us": round(aiter_us, 4),
                "skinny_us": None if skinny_us is None else round(skinny_us, 4),
                "speedup": (
                    None if skinny_us is None else round(aiter_us / skinny_us, 4)
                ),
                "skinny_faster": None if skinny_us is None else skinny_us < aiter_us,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
