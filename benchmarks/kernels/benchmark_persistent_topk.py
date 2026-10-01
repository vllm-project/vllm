# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark sparse-indexer decode top-k on variable-length FP32 rows.

Run on a reserved GPU, for example:
    chg run -- .venv/bin/python benchmarks/kernels/benchmark_persistent_topk.py

Use --capacity to emulate graphs captured at a larger maximum context length.
Run the same command on the base and candidate builds to compare JSON results.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm._custom_ops  # noqa: F401
from vllm.platforms import current_platform


def benchmark(rows: int, width: int, k: int, args) -> dict:
    capacity = args.capacity or width
    if capacity < width:
        raise ValueError("Capacity must be at least the valid-length upper bound")
    torch.manual_seed(args.seed)
    logits = torch.randn(rows, capacity, device="cuda", dtype=torch.float32)
    if args.full_length:
        lengths = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    else:
        lengths = torch.randint(
            int(width * 0.8), width + 1, (rows,), device="cuda", dtype=torch.int32
        )
    invalid = torch.arange(capacity, device="cuda")[None] >= lengths[:, None]
    logits.masked_fill_(invalid, float("nan"))
    out = torch.empty((rows, k), device="cuda", dtype=torch.int32)
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
    cooperative = (
        args.backend == "auto"
        and rows <= 64
        and capacity % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    op = torch.ops._C.cooperative_topk if cooperative else torch.ops._C.persistent_topk

    def run():
        op(logits, lengths, out, workspace, k, capacity)

    run()
    valid = torch.arange(k, device="cuda")[None] < lengths[:, None]
    assert torch.all(out[~valid] == -1)
    assert torch.all(((out >= 0) & (out < lengths[:, None])) == valid)
    sorted_indices = out.sort(dim=1).values
    assert torch.all(
        (sorted_indices[:, 1:] != sorted_indices[:, :-1])
        | (sorted_indices[:, 1:] == -1)
    )
    selected = logits.gather(1, out.clamp_min(0).long()).masked_fill(
        ~valid, -float("inf")
    )
    reference = logits.masked_fill(invalid, -float("inf")).topk(k, dim=1).values
    torch.testing.assert_close(
        selected.sort(dim=1, descending=True).values, reference, atol=0, rtol=0
    )
    trials = [
        statistics.median(
            bench_gpu_time_with_cupti(
                run,
                use_cuda_graph=True,
                cold_l2_cache=not args.warm,
                dry_run_iters=5,
                repeat_iters=args.repeat,
            )
        )
        * 1000
        for _ in range(args.trials)
    ]
    us = statistics.median(trials)
    return dict(
        rows=rows,
        width=width,
        capacity=capacity,
        k=k,
        backend="cooperative" if cooperative else "persistent",
        us=us,
        trials_us=trials,
        # Minimum useful traffic: one valid-score read and one index write.
        effective_gbps=(int(lengths.sum()) * 4 + out.numel() * 4) / (us * 1000),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 32, 128, 256, 1024])
    parser.add_argument("--widths", type=int, nargs="+", default=[8192, 65536, 163840])
    parser.add_argument("--top-k", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--capacity", type=int)
    parser.add_argument(
        "--full-length", action="store_true", help="Use exactly --widths valid scores"
    )
    parser.add_argument("--backend", choices=["auto", "persistent"], default="auto")
    parser.add_argument(
        "--warm", action="store_true", help="Keep L2 warm between replays"
    )
    parser.add_argument("--repeat", type=int, default=60)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = dict(
        gpu=str(torch.cuda.get_device_properties(0)),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        args=vars(args),
    )
    print(json.dumps(metadata, default=str))
    results = []
    for rows, width, k in itertools.product(args.rows, args.widths, args.top_k):
        result = benchmark(rows, width, k, args)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(
            json.dumps(dict(metadata=metadata, results=results), indent=2, default=str)
            + "\n"
        )


if __name__ == "__main__":
    main()
