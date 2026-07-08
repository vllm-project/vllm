# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.mla.sparse_utils import (
    merge_topology_tail_indices,
    merge_topology_tail_indices_reference,
)


def format_markdown_row(result: dict[str, float | int]) -> str:
    return (
        f"| {int(result['rows'])} |"
        f" {int(result['topk'])} |"
        f" {int(result['topology_width'])} |"
        f" {int(result['learned_keep'])} |"
        f" {int(result['max_replacements'])} |"
        f" {float(result['triton_us']):.3f} |"
        f" {float(result['reference_us']):.3f} |"
        f" {float(result['speedup']):.2f}x |"
    )


def benchmark_case(
    rows: int,
    topk: int,
    topology_width: int,
    learned_keep: int,
    max_replacements: int,
) -> dict[str, float | int]:
    learned_cpu, topology_cpu = make_inputs(rows, topk, topology_width, "cpu")
    learned_cuda = learned_cpu.cuda()
    topology_cuda = topology_cpu.cuda()

    expected = merge_topology_tail_indices_reference(
        learned_cpu,
        topology_cpu,
        learned_keep,
        max_replacements,
    ).cuda()
    actual = merge_topology_tail_indices(
        learned_cuda,
        topology_cuda,
        learned_keep,
        max_replacements,
    )
    if not torch.equal(actual, expected):
        raise AssertionError("Triton topology tail merge disagrees with reference")

    quantiles = [0.5, 0.2, 0.8]
    triton_ms, _, _ = triton.testing.do_bench(
        lambda: merge_topology_tail_indices(
            learned_cuda,
            topology_cuda,
            learned_keep,
            max_replacements,
        ),
        quantiles=quantiles,
    )
    reference_ms, _, _ = triton.testing.do_bench(
        lambda: merge_topology_tail_indices_reference(
            learned_cpu,
            topology_cpu,
            learned_keep,
            max_replacements,
        ),
        quantiles=quantiles,
    )
    triton_us = triton_ms * 1000.0
    reference_us = reference_ms * 1000.0
    return {
        "rows": rows,
        "topk": topk,
        "topology_width": topology_width,
        "learned_keep": learned_keep,
        "max_replacements": max_replacements,
        "triton_us": triton_us,
        "reference_us": reference_us,
        "speedup": reference_us / max(triton_us, 1e-12),
    }


def make_inputs(
    rows: int,
    topk: int,
    topology_width: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    learned = torch.arange(rows * topk, dtype=torch.int32, device=device).reshape(
        rows, topk
    )
    learned = learned % (topk * 8)
    topology = torch.arange(
        rows * topology_width,
        dtype=torch.int32,
        device=device,
    ).reshape(rows, topology_width)
    topology = (topology * 17 + topk // 2) % (topk * 8)
    if topology_width > 1:
        topology[:, -1] = -1
    return learned, topology


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark sparse MLA topology tail index merge."
    )
    parser.add_argument("--rows", type=int, nargs="*", default=[128, 512, 2048])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--topology-width", type=int, default=64)
    parser.add_argument("--learned-keep", type=int, default=1536)
    parser.add_argument("--max-replacements", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print(
        "| rows | topk | topology width | learned keep | max replacements | "
        "triton us | reference us | speedup |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rows in args.rows:
        print(
            format_markdown_row(
                benchmark_case(
                    rows=rows,
                    topk=args.topk,
                    topology_width=args.topology_width,
                    learned_keep=args.learned_keep,
                    max_replacements=args.max_replacements,
                )
            )
        )


if __name__ == "__main__":
    main()
