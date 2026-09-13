# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark sparse indexer decode TopK dispatch.

This targets ``torch.ops._C.persistent_topk``, which is used by the sparse MLA
indexer decode path after fp8/fp4 paged MQA logits are produced.

Example:
  python benchmarks/kernels/benchmark_sparse_indexer_topk.py --grid glm
  python benchmarks/kernels/benchmark_sparse_indexer_topk.py --grid full \
    --output sparse_indexer_topk.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


TOPK = 2048
RADIX_THRESHOLD = 32768
DEFAULT_WORKSPACE_BYTES = 1024 * 1024


@dataclasses.dataclass(frozen=True)
class Shape:
    batch_size: int
    next_n: int
    stride: int
    pattern: str

    @property
    def num_rows(self) -> int:
        return self.batch_size * self.next_n


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_str_list(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def legacy_route(num_rows: int, effective_max_len: int) -> str:
    if effective_max_len <= TOPK:
        return "persistent"
    return "filtered" if num_rows > 32 else "persistent"


def heuristic_route(num_rows: int, effective_max_len: int) -> str:
    if effective_max_len <= TOPK:
        return "persistent"
    if effective_max_len <= RADIX_THRESHOLD:
        return "filtered"
    return "filtered" if num_rows > 32 else "persistent"


def estimate_max_seq_len(shape: Shape) -> int:
    if shape.pattern == "short_padded":
        return min(shape.stride, max(TOPK + 1, 8192))
    return shape.stride


def make_lengths(shape: Shape, device: torch.device) -> torch.Tensor:
    batch_size = shape.batch_size
    stride = shape.stride
    next_n = shape.next_n

    if shape.pattern == "uniform":
        seq_lens = torch.full((batch_size,), stride, dtype=torch.int32, device=device)
    elif shape.pattern == "short_padded":
        length = min(stride, max(TOPK + 1, 8192))
        seq_lens = torch.full((batch_size,), length, dtype=torch.int32, device=device)
    elif shape.pattern == "mixed_clustered":
        short = min(stride, max(TOPK + 1, stride // 4))
        long = stride
        seq_lens = torch.full((batch_size,), long, dtype=torch.int32, device=device)
        seq_lens[::2] = short
    elif shape.pattern == "mixed_interleaved":
        values = torch.linspace(TOPK + 1, stride, batch_size, device=device)
        seq_lens = values.round().to(torch.int32)
    elif shape.pattern == "trivial_heavy":
        seq_lens = torch.full((batch_size,), TOPK, dtype=torch.int32, device=device)
        seq_lens[-1:] = stride
    elif shape.pattern == "ramp":
        values = torch.arange(1, batch_size + 1, device=device, dtype=torch.float32)
        values = TOPK + values * (stride - TOPK) / max(batch_size, 1)
        seq_lens = values.round().to(torch.int32)
    else:
        raise ValueError(f"unknown length pattern: {shape.pattern}")

    seq_lens = seq_lens.clamp(min=1, max=stride)
    if next_n == 1:
        return seq_lens

    offsets = torch.arange(next_n, device=device, dtype=torch.int32)
    lengths = seq_lens.unsqueeze(1) - next_n + 1 + offsets
    return lengths.clamp(min=1, max=stride).flatten()


def make_logits(shape: Shape, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits = torch.randn(
        (shape.num_rows, shape.stride), dtype=torch.float32, device=device
    )
    row_ids = torch.arange(shape.stride, dtype=torch.int32, device=device)
    logits.masked_fill_(row_ids.unsqueeze(0) >= lengths.unsqueeze(1), float("-inf"))
    return logits


def topk_values_match(
    logits: torch.Tensor,
    output: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    atol: float = 1e-4,
) -> bool:
    for row in range(logits.shape[0]):
        length = int(lengths[row].item())
        k = min(topk, length)
        if k <= 0:
            continue
        actual_indices = output[row, :k].to(torch.long)
        actual_values = logits[row, actual_indices].sort(descending=True).values
        expected_values = logits[row, :length].topk(k).values.sort(descending=True).values
        if not torch.allclose(actual_values, expected_values, atol=atol, rtol=atol):
            return False
    return True


def benchmark_one(
    shape: Shape,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, int | float | str | bool]:
    lengths = make_lengths(shape, device=device)
    logits = make_logits(shape, lengths, device=device)
    output = torch.empty((shape.num_rows, TOPK), dtype=torch.int32, device=device)
    workspace = torch.empty(args.workspace_bytes, dtype=torch.uint8, device=device)

    if args.max_seq_len_mode == "stride":
        max_seq_len = shape.stride
    else:
        max_seq_len = int(lengths.max().item())
    effective_max_len = max(1, min(shape.stride, max_seq_len))

    def run() -> None:
        torch.ops._C.persistent_topk(
            logits, lengths, output, workspace, TOPK, max_seq_len
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    times_us: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        start.record()
        run()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000.0)

    correct = True
    if args.check:
        correct = topk_values_match(logits, output, lengths, TOPK)

    times = sorted(times_us)
    return {
        "batch_size": shape.batch_size,
        "next_n": shape.next_n,
        "num_rows": shape.num_rows,
        "topk": TOPK,
        "stride": shape.stride,
        "pattern": shape.pattern,
        "max_seq_len": max_seq_len,
        "effective_max_len": effective_max_len,
        "legacy_route": legacy_route(shape.num_rows, effective_max_len),
        "heuristic_route": heuristic_route(shape.num_rows, effective_max_len),
        "route_changed": legacy_route(shape.num_rows, effective_max_len)
        != heuristic_route(shape.num_rows, effective_max_len),
        "correct": correct,
        "min_us": min(times),
        "p20_us": times[max(0, math.floor(0.20 * (len(times) - 1)))],
        "median_us": statistics.median(times),
        "p80_us": times[max(0, math.floor(0.80 * (len(times) - 1)))],
        "max_us": max(times),
    }


def build_shapes(args: argparse.Namespace) -> list[Shape]:
    if args.grid == "glm":
        batch_sizes = [8, 16, 32]
        next_ns = [1, 2]
        strides = [8192, 9216, 16384, 32768]
        patterns = ["uniform", "short_padded", "mixed_clustered"]
    elif args.grid == "quick":
        batch_sizes = [1, 8, 16, 32, 64]
        next_ns = [1, 2]
        strides = [8192, 32768, 163840]
        patterns = ["uniform", "short_padded", "trivial_heavy"]
    elif args.grid == "full":
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        next_ns = [1, 2, 3, 4]
        strides = [4096, 8192, 16384, 32768, 65536, 131072, 163840]
        patterns = [
            "uniform",
            "short_padded",
            "mixed_clustered",
            "mixed_interleaved",
            "trivial_heavy",
            "ramp",
        ]
    else:
        batch_sizes = parse_int_list(args.batch_sizes)
        next_ns = parse_int_list(args.next_ns)
        strides = parse_int_list(args.strides)
        patterns = parse_str_list(args.patterns)

    shapes = [
        Shape(batch_size=bs, next_n=next_n, stride=stride, pattern=pattern)
        for bs, next_n, stride, pattern in itertools.product(
            batch_sizes, next_ns, strides, patterns
        )
    ]

    filtered: list[Shape] = []
    for shape in shapes:
        input_gb = shape.num_rows * shape.stride * 4 / 1e9
        if input_gb <= args.max_input_gb:
            filtered.append(shape)
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", choices=["quick", "glm", "full", "custom"], default="quick"
    )
    parser.add_argument("--batch-sizes", default="1,8,16,32,64")
    parser.add_argument("--next-ns", default="1,2")
    parser.add_argument("--strides", default="8192,32768,163840")
    parser.add_argument(
        "--patterns", default="uniform,short_padded,mixed_clustered,trivial_heavy"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--max-input-gb", type=float, default=2.0)
    parser.add_argument(
        "--max-seq-len-mode", choices=["actual", "stride"], default="actual"
    )
    parser.add_argument("--workspace-bytes", type=int, default=DEFAULT_WORKSPACE_BYTES)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    shapes = build_shapes(args)
    route_changed_shapes = sum(
        heuristic_route(s.num_rows, estimate_max_seq_len(s))
        != legacy_route(s.num_rows, estimate_max_seq_len(s))
        for s in shapes
    )
    print(
        f"grid={args.grid} shapes={len(shapes)} topk={TOPK} "
        f"max_input_gb={args.max_input_gb}"
    )
    print(f"route_changed_shapes={route_changed_shapes}")

    if args.dry_run:
        for shape in shapes:
            print(dataclasses.asdict(shape))
        return 0

    global torch
    import torch as torch_module
    torch = torch_module

    import vllm._custom_ops  # noqa: F401
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        raise RuntimeError("benchmark_sparse_indexer_topk requires CUDA")

    device = torch.device("cuda")
    rows: list[dict[str, int | float | str | bool]] = []
    for idx, shape in enumerate(shapes, start=1):
        row = benchmark_one(shape, args, device)
        rows.append(row)
        print(
            f"[{idx:03d}/{len(shapes):03d}] bs={shape.batch_size} "
            f"next_n={shape.next_n} stride={shape.stride} pattern={shape.pattern} "
            f"route={row['heuristic_route']} median_us={row['median_us']:.3f} "
            f"correct={row['correct']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
