#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark MXFP4 MoE activation quant at GLM-5.3-shaped sizes.

Two timed regions (``--timed-region``):

* ``quant`` (default): ``mxfp4_experts_quant`` / ``silu_and_mul_mxfp4_experts_quant``
  only, with synthetic routing offsets (kernel isolation).
* ``metadata_and_quant``: ``get_cutlass_moe_mm_data`` → ``shuffle_rows`` → quant,
  matching the Cutlass MXFP4 MoE preparation + activation quant sequence.
* ``both``: run ``quant`` then ``metadata_and_quant``.

Expert count is swept; routing uses 32 rows (two tokens on experts 0..15). GEMMs
are outside the timed region.

Run from a CUDA-built vLLM checkout, for example:

    .venv/bin/python benchmarks/kernels/benchmark_mxfp4_experts_activation_quant.py
    .venv/bin/python benchmarks/kernels/benchmark_mxfp4_experts_activation_quant.py \\
        --timed-region metadata_and_quant --experts 256
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import vllm._custom_ops  # noqa: E402, F401 -- registers torch.ops._C kernels
from vllm import _custom_ops as ops  # noqa: E402

# GLM-5.3 MXFP4 MoE activation quant shapes (see cutlass MoE path).
ROUTED_ROWS = 32
TOPK = 1
HIDDEN_SIZE = 6144
INTERMEDIATE_IN = 1024
INTERMEDIATE_OUT = 512
DEFAULT_EXPERT_COUNTS = (16, 32, 64, 128, 256)


class TimedRegion(str, Enum):
    QUANT = "quant"
    METADATA_AND_QUANT = "metadata_and_quant"
    BOTH = "both"


@dataclass(frozen=True)
class Case:
    kernel: str
    output_width: int
    fused_silu_mul: bool
    experts: int
    mm_n: int
    mm_k: int
    is_gated: bool


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


def decode_topk_ids(rows: int = ROUTED_ROWS) -> torch.Tensor:
    """Two tokens per expert 0..15; token order shuffled (fixed seed)."""
    expert_ids: list[int] = []
    for expert in range(16):
        expert_ids.extend([expert, expert])
    assert len(expert_ids) == rows
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    perm = torch.randperm(rows, generator=gen, device="cuda")
    ordered = torch.tensor(expert_ids, device="cuda", dtype=torch.int32)
    return ordered[perm].unsqueeze(1)


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


def _quant_tensors(
    case: Case,
    inputs: torch.Tensor,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_rows = int(blockscale_offsets[-1].item())
    scale_cols_int32 = (case.output_width // 32 + 3) // 4
    output = torch.empty(
        (ROUTED_ROWS, case.output_width // 2), dtype=torch.uint8, device="cuda"
    )
    output_scales = torch.zeros(
        (scale_rows, scale_cols_int32), dtype=torch.int32, device="cuda"
    )
    return output, output_scales


def benchmark_mxfp4_quant_only(
    case: Case,
    inputs: torch.Tensor,
    graph_nodes: int,
    samples: int,
    sorted_first: bool,
) -> tuple[float, float]:
    expert_offsets, blockscale_offsets, expert_ids, _ = routing_offsets(case.experts)
    output, output_scales = _quant_tensors(
        case, inputs, expert_offsets, blockscale_offsets
    )
    op = (
        torch.ops._C.silu_and_mul_mxfp4_experts_quant
        if case.fused_silu_mul
        else torch.ops._C.mxfp4_experts_quant
    )

    def invoke_offset() -> None:
        op(
            output,
            output_scales,
            inputs,
            expert_offsets,
            blockscale_offsets,
            case.experts,
            None,
        )

    def invoke_sorted() -> None:
        op(
            output,
            output_scales,
            inputs,
            expert_offsets,
            blockscale_offsets,
            case.experts,
            expert_ids,
        )

    if sorted_first:
        direct_id_us = benchmark_graph(invoke_sorted, graph_nodes, samples)
        baseline_us = benchmark_graph(invoke_offset, graph_nodes, samples)
    else:
        baseline_us = benchmark_graph(invoke_offset, graph_nodes, samples)
        baseline_output = output.clone()
        baseline_scales = output_scales.clone()
        direct_id_us = benchmark_graph(invoke_sorted, graph_nodes, samples)
        if not (
            torch.equal(output, baseline_output)
            and torch.equal(output_scales, baseline_scales)
        ):
            raise AssertionError(
                f"MXFP4 output mismatch for {case.kernel} E={case.experts}"
            )
    return baseline_us, direct_id_us


def benchmark_mxfp4_metadata_and_quant(
    case: Case,
    inputs: torch.Tensor,
    graph_nodes: int,
    samples: int,
    sorted_first: bool,
) -> tuple[float, float]:
    num_experts = case.experts
    topk_ids = decode_topk_ids()
    expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device="cuda")
    blockscale_offsets = torch.empty_like(expert_offsets)
    problem_sizes1 = torch.empty(num_experts, 3, dtype=torch.int32, device="cuda")
    problem_sizes2 = torch.empty_like(problem_sizes1)
    input_permutation = torch.empty(ROUTED_ROWS, dtype=torch.int32, device="cuda")
    output_permutation = torch.empty_like(input_permutation)
    sorted_expert_ids = torch.empty_like(input_permutation)

    # Warmup metadata once to size scale buffers.
    torch.ops._C.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        case.mm_n,
        case.mm_k,
        blockscale_offsets,
        case.is_gated,
        sorted_expert_ids,
    )
    output, output_scales = _quant_tensors(
        case, inputs, expert_offsets, blockscale_offsets
    )
    op = (
        torch.ops._C.silu_and_mul_mxfp4_experts_quant
        if case.fused_silu_mul
        else torch.ops._C.mxfp4_experts_quant
    )

    def invoke_offset() -> None:
        torch.ops._C.get_cutlass_moe_mm_data(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            input_permutation,
            output_permutation,
            num_experts,
            case.mm_n,
            case.mm_k,
            blockscale_offsets,
            case.is_gated,
            None,
        )
        sorted_inputs = ops.shuffle_rows(inputs, input_permutation)
        op(
            output,
            output_scales,
            sorted_inputs,
            expert_offsets,
            blockscale_offsets,
            num_experts,
            None,
        )

    def invoke_sorted() -> None:
        torch.ops._C.get_cutlass_moe_mm_data(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            input_permutation,
            output_permutation,
            num_experts,
            case.mm_n,
            case.mm_k,
            blockscale_offsets,
            case.is_gated,
            sorted_expert_ids,
        )
        sorted_inputs = ops.shuffle_rows(inputs, input_permutation)
        op(
            output,
            output_scales,
            sorted_inputs,
            expert_offsets,
            blockscale_offsets,
            num_experts,
            sorted_expert_ids,
        )

    if sorted_first:
        direct_id_us = benchmark_graph(invoke_sorted, graph_nodes, samples)
        baseline_us = benchmark_graph(invoke_offset, graph_nodes, samples)
    else:
        baseline_us = benchmark_graph(invoke_offset, graph_nodes, samples)
        baseline_output = output.clone()
        baseline_scales = output_scales.clone()
        direct_id_us = benchmark_graph(invoke_sorted, graph_nodes, samples)
        if not (
            torch.equal(output, baseline_output)
            and torch.equal(output_scales, baseline_scales)
        ):
            raise AssertionError(
                f"MXFP4 output mismatch for {case.kernel} E={case.experts}"
            )
    return baseline_us, direct_id_us


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


def parse_timed_region(raw: str) -> TimedRegion:
    try:
        return TimedRegion(raw)
    except ValueError:
        choices = ", ".join(r.value for r in TimedRegion)
        raise ValueError(f"timed-region must be one of: {choices}") from None


def build_cases(experts: tuple[int, ...]) -> list[tuple[Case, torch.Tensor]]:
    cases: list[tuple[Case, torch.Tensor]] = []
    for expert_count in experts:
        cases.append(
            (
                Case(
                    "mxfp4_experts_quant",
                    HIDDEN_SIZE,
                    False,
                    expert_count,
                    mm_n=4096,
                    mm_k=HIDDEN_SIZE,
                    is_gated=False,
                ),
                torch.randn(
                    (ROUTED_ROWS, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
                ),
            )
        )
        cases.append(
            (
                Case(
                    "silu_and_mul_mxfp4_experts_quant",
                    INTERMEDIATE_OUT,
                    True,
                    expert_count,
                    mm_n=HIDDEN_SIZE,
                    mm_k=INTERMEDIATE_OUT,
                    is_gated=True,
                ),
                torch.randn(
                    (ROUTED_ROWS, INTERMEDIATE_IN), device="cuda", dtype=torch.bfloat16
                ),
            )
        )
    return cases


def run_region(
    region: TimedRegion,
    cases: list[tuple[Case, torch.Tensor]],
    graph_nodes: int,
    samples: int,
    sorted_first: bool,
) -> None:
    label = region.value
    print(f"timed_region={label}")
    print("kernel,experts,variant,median_us,speedup_vs_offset_search")
    bench_fn = (
        benchmark_mxfp4_metadata_and_quant
        if region == TimedRegion.METADATA_AND_QUANT
        else benchmark_mxfp4_quant_only
    )
    for case, inputs in cases:
        baseline_us, direct_id_us = bench_fn(
            case, inputs, graph_nodes, samples, sorted_first
        )
        print(f"{case.kernel},{case.experts},offset_search,{baseline_us:.4f},1.0000")
        print(
            f"{case.kernel},{case.experts},sorted_expert_id,"
            f"{direct_id_us:.4f},{baseline_us / direct_id_us:.4f}"
        )
    print()


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
    parser.add_argument(
        "--timed-region",
        type=parse_timed_region,
        default=TimedRegion.QUANT,
        help=(
            "quant: activation quant only; metadata_and_quant: mm_data + shuffle + "
            "quant; both: run each region"
        ),
    )
    parser.add_argument(
        "--sorted-id-first",
        action="store_true",
        help="time sorted-expert-ID path before offset search (default: offset first)",
    )
    parser.add_argument("--skip-metadata-check", action="store_true")
    args = parser.parse_args()
    if args.graph_nodes < 1 or args.samples < 1:
        parser.error("--graph-nodes and --samples must be positive")
    if not torch.cuda.is_available():
        parser.error("requires a CUDA GPU")
    capability = torch.cuda.get_device_capability()
    if capability[0] not in (10, 11):
        parser.error(f"requires an SM10x/11x GPU, got SM{capability[0]}{capability[1]}")

    torch.manual_seed(0)
    if not args.skip_metadata_check:
        verify_sorted_id_writer(num_experts=256)

    cases = build_cases(args.experts)
    print(f"GPU: {torch.cuda.get_device_name()} (SM{capability[0]}{capability[1]})")
    print(
        f"shapes: hidden={ROUTED_ROWS}x{HIDDEN_SIZE}, "
        f"intermediate={ROUTED_ROWS}x{INTERMEDIATE_IN}->{INTERMEDIATE_OUT}, "
        f"topk={TOPK}"
    )

    regions: list[TimedRegion]
    if args.timed_region == TimedRegion.BOTH:
        regions = [TimedRegion.QUANT, TimedRegion.METADATA_AND_QUANT]
    else:
        regions = [args.timed_region]

    for region in regions:
        run_region(
            region,
            cases,
            args.graph_nodes,
            args.samples,
            args.sorted_id_first,
        )


if __name__ == "__main__":
    main()
