#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark MXFP4 MoE activation quant kernels at GLM-5.3-shaped sizes.

Times ``mxfp4_experts_quant`` (hidden 6144) and ``silu_and_mul_mxfp4_experts_quant``
(FFN intermediate) with and without ``sorted_expert_ids``. Expert count is swept;
routing uses 32 rows (2 tokens on each of the first 16 experts), matching a small
decode batch. Routing metadata and GEMMs are outside the timed region.

Run from a CUDA-built vLLM checkout, for example:

    .venv/bin/python benchmarks/kernels/benchmark_mxfp4_experts_activation_quant.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import vllm._custom_ops  # noqa: E402, F401 -- registers torch.ops._C kernels

# GLM-5.3 MXFP4 MoE activation quant shapes (see cutlass MoE path).
ROUTED_ROWS = 32
HIDDEN_SIZE = 6144
INTERMEDIATE_IN = 1024
INTERMEDIATE_OUT = 512
DEFAULT_EXPERT_COUNTS = (16, 32, 64, 128, 256)


@dataclass(frozen=True)
class Case:
    kernel: str
    output_width: int
    fused_silu_mul: bool
    experts: int


def routing_offsets(
    num_experts: int,
    rows: int = ROUTED_ROWS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """32 rows, two assignments on experts 0..15 (production-like small decode)."""
    if rows != ROUTED_ROWS:
        raise ValueError(f"expected {ROUTED_ROWS} routed rows, got {rows}")
    if num_experts < 16:
        raise ValueError("expert sweep must use at least 16 experts")

    counts = [0] * num_experts
    counts[:16] = [2] * 16

    row_offsets = [0]
    scale_offsets = [0]
    expert_ids: list[int] = []
    for expert, count in enumerate(counts):
        expert_ids.extend([expert] * count)
        row_offsets.append(row_offsets[-1] + count)
        scale_offsets.append(scale_offsets[-1] + ((count + 127) // 128) * 128)
    assert row_offsets[-1] == rows
    return (
        torch.tensor(row_offsets, dtype=torch.int32, device="cuda"),
        torch.tensor(scale_offsets, dtype=torch.int32, device="cuda"),
        torch.tensor(expert_ids, dtype=torch.int32, device="cuda"),
        scale_offsets[-1],
    )


def benchmark_graph(
    invoke,
    graph_nodes: int,
    samples: int,
) -> float:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(5):
            invoke()
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_nodes):
            invoke()
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()

    elapsed_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed_us.append(start.elapsed_time(end) * 1_000 / graph_nodes)
    return statistics.median(elapsed_us)


def benchmark_mxfp4(
    case: Case,
    inputs: torch.Tensor,
    graph_nodes: int,
    samples: int,
) -> tuple[float, float]:
    expert_offsets, blockscale_offsets, expert_ids, scale_rows = routing_offsets(
        case.experts
    )
    output = torch.empty(
        (ROUTED_ROWS, case.output_width // 2), dtype=torch.uint8, device="cuda"
    )
    scale_cols_int32 = (case.output_width // 32 + 3) // 4
    output_scales = torch.zeros(
        (scale_rows, scale_cols_int32), dtype=torch.int32, device="cuda"
    )
    op = (
        torch.ops._C.silu_and_mul_mxfp4_experts_quant
        if case.fused_silu_mul
        else torch.ops._C.mxfp4_experts_quant
    )

    def invoke(sorted_ids: torch.Tensor | None) -> None:
        op(
            output,
            output_scales,
            inputs,
            expert_offsets,
            blockscale_offsets,
            case.experts,
            sorted_ids,
        )

    baseline_us = benchmark_graph(lambda: invoke(None), graph_nodes, samples)
    baseline_output = output.clone()
    baseline_scales = output_scales.clone()

    direct_id_us = benchmark_graph(lambda: invoke(expert_ids), graph_nodes, samples)
    if not (
        torch.equal(output, baseline_output)
        and torch.equal(output_scales, baseline_scales)
    ):
        raise AssertionError(
            f"MXFP4 output mismatch for {case.kernel} E={case.experts}"
        )
    return baseline_us, direct_id_us


def benchmark_nvfp4(
    inputs: torch.Tensor,
    graph_nodes: int,
    samples: int,
) -> float:
    rows, width = inputs.shape
    output = torch.empty((rows, width // 2), dtype=torch.uint8, device="cuda")
    output_scales = torch.empty((rows, width // 16), dtype=torch.uint8, device="cuda")
    global_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    def invoke() -> None:
        torch.ops._C.scaled_fp4_quant.out(
            inputs,
            global_scale,
            False,
            output=output,
            output_scale=output_scales,
        )

    return benchmark_graph(invoke, graph_nodes, samples)


def verify_sorted_id_writer(num_experts: int) -> None:
    """Check the routing sort writes the expected sorted expert ID at each row."""
    topk_ids = torch.arange(ROUTED_ROWS, device="cuda", dtype=torch.int32) % num_experts
    topk_ids = topk_ids.view(4, 8)
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device="cuda")
    blockscale_offsets = torch.empty_like(expert_offsets)
    problem_sizes1 = torch.empty(num_experts, 3, dtype=torch.int32, device="cuda")
    problem_sizes2 = torch.empty_like(problem_sizes1)
    input_permutation = torch.empty(ROUTED_ROWS, dtype=torch.int32, device="cuda")
    output_permutation = torch.empty_like(input_permutation)
    sorted_expert_ids = torch.empty_like(input_permutation)
    torch.ops._C.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        64,
        64,
        blockscale_offsets,
        True,
        sorted_expert_ids,
    )
    expected = torch.sort(topk_ids.flatten()).values
    torch.accelerator.synchronize()
    if not torch.equal(sorted_expert_ids, expected):
        raise AssertionError("routing sort produced incorrect sorted expert IDs")


def parse_expert_counts(raw: str) -> tuple[int, ...]:
    counts = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not counts or any(c < 16 for c in counts):
        raise ValueError(
            "expert counts must be a comma-separated list of integers >= 16"
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-nodes", type=int, default=128)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--experts",
        type=parse_expert_counts,
        default=DEFAULT_EXPERT_COUNTS,
        help=(
            "comma-separated expert counts (default: "
            f"{','.join(map(str, DEFAULT_EXPERT_COUNTS))})"
        ),
    )
    parser.add_argument("--skip-metadata-check", action="store_true")
    parser.add_argument("--skip-nvfp4", action="store_true")
    args = parser.parse_args()
    if args.graph_nodes < 1 or args.samples < 1:
        parser.error("--graph-nodes and --samples must be positive")
    if not torch.cuda.is_available():
        parser.error("requires a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability[0] not in (10, 11):
        parser.error(f"requires an SM10x/11x GPU, got SM{capability[0]}{capability[1]}")

    torch.manual_seed(0)
    hidden_input = torch.randn(
        (ROUTED_ROWS, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
    )
    intermediate_input = torch.randn(
        (ROUTED_ROWS, INTERMEDIATE_IN), device="cuda", dtype=torch.bfloat16
    )

    if not args.skip_metadata_check:
        verify_sorted_id_writer(num_experts=256)

    cases: list[tuple[Case, torch.Tensor]] = []
    for experts in args.experts:
        cases.append(
            (
                Case(
                    "mxfp4_experts_quant",
                    HIDDEN_SIZE,
                    False,
                    experts,
                ),
                hidden_input,
            )
        )
        cases.append(
            (
                Case(
                    "silu_and_mul_mxfp4_experts_quant",
                    INTERMEDIATE_OUT,
                    True,
                    experts,
                ),
                intermediate_input,
            )
        )

    print(f"GPU: {torch.cuda.get_device_name()} (SM{capability[0]}{capability[1]})")
    print(
        f"shapes: hidden={ROUTED_ROWS}x{HIDDEN_SIZE}, "
        f"intermediate={ROUTED_ROWS}x{INTERMEDIATE_IN}->{INTERMEDIATE_OUT}"
    )
    print("kernel,experts,variant,median_us,speedup_vs_offset_search")
    for case, inputs in cases:
        baseline_us, direct_id_us = benchmark_mxfp4(
            case, inputs, args.graph_nodes, args.samples
        )
        print(f"{case.kernel},{case.experts},offset_search,{baseline_us:.4f},1.0000")
        print(
            f"{case.kernel},{case.experts},sorted_expert_id,"
            f"{direct_id_us:.4f},{baseline_us / direct_id_us:.4f}"
        )

    if not args.skip_nvfp4:
        latency = benchmark_nvfp4(hidden_input, args.graph_nodes, args.samples)
        print(f"scaled_fp4_quant,-,standalone,{latency:.4f},")


if __name__ == "__main__":
    main()
