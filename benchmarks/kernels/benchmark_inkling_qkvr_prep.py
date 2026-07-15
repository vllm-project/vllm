# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import statistics

import torch
from tabulate import tabulate

from vllm.models.inkling.nvidia.ops import qkvr_prep
from vllm.utils.argparse_utils import FlexibleArgumentParser


def make_inputs(tokens: int, tp_size: int, is_local: bool):
    torch.manual_seed(0)
    num_q_heads = 64 // tp_size
    num_kv_heads = (16 if is_local else 8) // tp_size
    head_dim = 128
    d_rel = 16
    rel_extent = 512 if is_local else 1024
    page_size = 16
    num_blocks = (tokens + page_size - 1) // page_size
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    r_width = num_q_heads * d_rel
    device = "cuda"

    qkvr = torch.randn(
        tokens,
        q_width + 2 * kv_width + r_width,
        device=device,
        dtype=torch.bfloat16,
    )
    k_weight = torch.randn(kv_width, 4, device=device, dtype=torch.bfloat16)
    v_weight = torch.randn_like(k_weight)
    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16)
    k_norm_weight = torch.randn_like(q_norm_weight)
    rel_proj = torch.randn(d_rel, rel_extent, device=device, dtype=torch.bfloat16)
    conv_cache = torch.zeros(
        num_blocks,
        num_kv_heads,
        page_size,
        2 * head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key_cache = torch.empty(
        num_blocks,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.empty_like(key_cache)
    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
    seq_idx = torch.zeros(tokens, device=device, dtype=torch.int32)
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    query_start = torch.zeros(tokens, device=device, dtype=torch.int32)
    log_scaling = None
    if not is_local:
        effective_n = (positions + 1).to(torch.float32)
        log_scaling = 1.0 + 0.1 * torch.log(torch.clamp(effective_n / 128000, min=1.0))
    return (
        qkvr,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        rel_proj,
        1e-6,
        num_q_heads,
        num_kv_heads,
        head_dim,
        d_rel,
        conv_cache,
        key_cache,
        value_cache,
        positions,
        block_table,
        seq_idx,
        slots,
        query_start,
        slots,
        0,
        head_dim,
        page_size,
        log_scaling,
    )


def capture(implementation, inputs):
    outputs = []

    def run():
        outputs[:] = implementation.fused_qkvr_prep(*inputs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.accelerator.synchronize()
    return graph, outputs


def time_graph(graph: torch.cuda.CUDAGraph, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def benchmark(inputs, args) -> float:
    graph, _ = capture(qkvr_prep, inputs)
    return statistics.median(
        time_graph(graph, args.warmup, args.repeats) for _ in range(args.trials)
    )


@torch.inference_mode()
def main(args):
    rows = []
    for tp_size in args.tp_sizes:
        for tokens in args.tokens:
            for is_local in (True, False):
                triton_us = benchmark(make_inputs(tokens, tp_size, is_local), args)
                rows.append(
                    [
                        tp_size,
                        tokens,
                        "local" if is_local else "global",
                        triton_us,
                    ]
                )

    print("Inkling QKVR prep (CUDA graph, median latency)")
    print(
        tabulate(
            rows,
            headers=[
                "TP",
                "tokens",
                "scope",
                "Triton (us)",
            ],
            floatfmt=("d", "d", "", ".2f"),
        )
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1 << power for power in range(15)],
    )
    parser.add_argument("--tp-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    main(parser.parse_args())
