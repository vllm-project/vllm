# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cost of the sparse MLA topology witness pass, against a full row copy.

The pass rewrites the tail of an existing top-k buffer, so the floor it has to
be judged against is copying that buffer -- not the eager fallback, which is a
Python loop and would flatter any kernel.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.attention.backends.mla.sparse_utils import apply_topology_witnesses


def make_inputs(
    rows: int, topk: int, max_context: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    context_lens = torch.randint(1, max_context + 1, (rows,), generator=generator)
    learned = torch.randint(0, max_context, (rows, topk), generator=generator)
    learned = (learned % context_lens.unsqueeze(1)).int()
    return context_lens.int().cuda(), learned.cuda()


def benchmark_case(
    rows: int, topk: int, segments: int, max_context: int, seed: int
) -> dict[str, float | int]:
    context_lens, learned = make_inputs(rows, topk, max_context, seed)
    learned_keep = max(0, topk - segments)

    expected = apply_topology_witnesses(
        learned.cpu(), context_lens.cpu(), learned_keep, segments, segments
    )
    actual = apply_topology_witnesses(
        learned, context_lens, learned_keep, segments, segments
    )
    if not torch.equal(actual.cpu(), expected):
        raise AssertionError("fused witness kernel disagrees with the torch reference")

    quantiles = [0.5, 0.2, 0.8]
    witness_ms, _, _ = triton.testing.do_bench(
        lambda: apply_topology_witnesses(
            learned, context_lens, learned_keep, segments, segments
        ),
        quantiles=quantiles,
    )
    copy_ms, _, _ = triton.testing.do_bench(
        lambda: learned.clone(), quantiles=quantiles
    )
    return {
        "rows": rows,
        "topk": topk,
        "segments": segments,
        "witness_us": witness_ms * 1000.0,
        "copy_us": copy_ms * 1000.0,
        "ratio": witness_ms / max(copy_ms, 1e-12),
    }


def format_markdown_row(result: dict[str, float | int]) -> str:
    return (
        f"| {int(result['rows'])} |"
        f" {int(result['topk'])} |"
        f" {int(result['segments'])} |"
        f" {float(result['witness_us']):.3f} |"
        f" {float(result['copy_us']):.3f} |"
        f" {float(result['ratio']):.2f}x |"
    )


def main() -> None:
    parser = FlexibleArgumentParser(
        description="Benchmark the sparse MLA topology witness pass."
    )
    parser.add_argument("--rows", type=int, nargs="*", default=[32, 128, 512])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print("| rows | topk | segments | witness us | row copy us | x copy |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for rows in args.rows:
        print(
            format_markdown_row(
                benchmark_case(
                    rows=rows,
                    topk=args.topk,
                    segments=args.segments,
                    max_context=args.max_context,
                    seed=args.seed,
                )
            )
        )


if __name__ == "__main__":
    main()
