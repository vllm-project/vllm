# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare HiSparse compact gather across builds, including sparse MTP plans."""

import json
import statistics

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.math_utils import cdiv


def benchmark(args, num_rows: int) -> dict:
    block_size = 64
    host_rows = num_rows * args.top_k
    host = torch.ones((host_rows, args.row_bytes), dtype=torch.uint8, pin_memory=True)
    hot = torch.empty(
        (cdiv(host_rows, block_size), block_size, args.row_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    rows = torch.arange(host_rows, dtype=torch.int32, device="cuda").view(
        num_rows, args.top_k
    )
    counts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    misses = int(args.top_k * args.miss_ratio)
    counts[:: args.active_every] = misses

    def run():
        torch.ops._C_cache_ops.hisparse_gather_compact(host, hot, rows, rows, counts)

    samples = [
        triton.testing.do_bench_cudagraph(run, rep=args.rep_ms)
        for _ in range(args.repeats)
    ]
    median_ms = statistics.median(samples)
    bytes_copied = cdiv(num_rows, args.active_every) * misses * args.row_bytes
    return {
        "rows": num_rows,
        "top_k": args.top_k,
        "row_bytes": args.row_bytes,
        "miss_ratio": args.miss_ratio,
        "active_every": args.active_every,
        "latency_us": [ms * 1000 for ms in samples],
        "median_us": median_ms * 1000,
        "host_to_device_GB_s": bytes_copied / median_ms / 1e6,
    }


def main():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 8, 32, 64, 65, 128, 256]
    )
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--row-bytes", type=int, default=576)
    parser.add_argument("--miss-ratio", type=float, default=1.0)
    parser.add_argument("--active-every", type=int, default=1)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if (
        min(
            *args.rows,
            args.top_k,
            args.row_bytes,
            args.active_every,
            args.rep_ms,
            args.repeats,
        )
        <= 0
        or args.row_bytes % 16
        or not 0 <= args.miss_ratio <= 1
    ):
        parser.error(
            "dimensions/repetitions must be positive, row bytes a multiple of 16, "
            "and miss ratio in [0, 1]"
        )
    if not current_platform.is_cuda():
        parser.error("HiSparse gather requires NVIDIA CUDA")
    current_platform.import_kernels()
    for num_rows in args.rows:
        print(json.dumps(benchmark(args, num_rows)), flush=True)


if __name__ == "__main__":
    main()
