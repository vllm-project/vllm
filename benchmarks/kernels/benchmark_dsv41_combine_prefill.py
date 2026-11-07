# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DeepSeek-V4.1 prefill index combination.

Compares the current kernel, which writes its own padding sentinels, with the
previous implementation that cleared the full output workspace first.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_dsv41_combine_prefill.py
    .venv/bin/python benchmarks/kernels/benchmark_dsv41_combine_prefill.py --graph

"""

import argparse
import json
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.deepseek_v41.common.ops.cache_utils import (
    combine_topk_swa_indices,
)

WINDOW_SIZE = 128
TOPK_WIDTH = 512
PREFILL_ALIGNMENT = 128


def _make_inputs(
    seq_lens: list[int],
    query_lens: list[int],
    compress_ratio: int,
    logical_topk: int,
    seed: int,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    num_reqs = len(seq_lens)
    num_tokens = sum(query_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    gather_lens = seq_lens_t.clone()
    ratio = max(compress_ratio, 1)
    compressed_len = (max(seq_lens) + ratio - 1) // ratio
    kv_len = 2 * compressed_len + int(gather_lens.max()) + 8
    topk_indices = torch.randint(
        0,
        compressed_len,
        (num_tokens, TOPK_WIDTH),
        dtype=torch.int32,
        device=device,
    )
    return (
        topk_indices,
        query_start_loc,
        seq_lens_t,
        gather_lens,
        kv_len,
        compressed_len,
    )


def _run(
    args: tuple[torch.Tensor | int, ...],
    out: tuple[torch.Tensor, torch.Tensor],
    *,
    fill_sentinels: bool,
) -> None:
    if fill_sentinels:
        out[0].fill_(-1)
    combine_topk_swa_indices(*args, out=out)


def benchmark_case(case: dict, args) -> dict:
    inputs = _make_inputs(
        case["seq_lens"],
        case["query_lens"],
        case["compress_ratio"],
        case["logical_topk"],
        args.seed,
    )
    topk_indices, query_start_loc, seq_lens, gather_lens, M, N = inputs
    num_tokens = topk_indices.shape[0]
    width = (
        (case["logical_topk"] + WINDOW_SIZE + PREFILL_ALIGNMENT - 1)
        // PREFILL_ALIGNMENT
        * PREFILL_ALIGNMENT
    )
    indices = torch.empty((num_tokens, width), dtype=torch.int32, device="cuda")
    lens = torch.empty(num_tokens, dtype=torch.int32, device="cuda")
    out = (indices, lens)
    combine_args = (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        WINDOW_SIZE,
        case["compress_ratio"],
        case["logical_topk"],
        M,
        N,
    )

    expected_indices, expected_lens = combine_topk_swa_indices(*combine_args, out=None)
    _run(combine_args, out, fill_sentinels=False)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(lens, expected_lens)

    arms = {
        "memset_combine": lambda: _run(combine_args, out, fill_sentinels=True),
        "kernel_sentinels": lambda: _run(combine_args, out, fill_sentinels=False),
    }
    samples = {name: [] for name in arms}
    for trial in range(args.trials):
        order = list(arms)
        if trial % 2:
            order.reverse()
        for name in order:
            times = bench_gpu_time_with_cupti(
                arms[name],
                repeat_time_ms=args.repeat_time_ms,
                cold_l2_cache=True,
                use_cuda_graph=args.graph,
            )
            samples[name].append(statistics.median(times) * 1000)

    medians = {name: statistics.median(times) for name, times in samples.items()}
    return {
        **case,
        "num_tokens": num_tokens,
        "width": width,
        "device": torch.cuda.get_device_name(),
        "trials_us": samples,
        "median_us": medians,
        "speedup": medians["memset_combine"] / medians["kernel_sentinels"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeat-time-ms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()

    cases = [
        dict(
            name="short-12-swa",
            seq_lens=[12],
            query_lens=[12],
            compress_ratio=0,
            logical_topk=0,
        ),
        dict(
            name="8k-swa",
            seq_lens=[8192],
            query_lens=[8192],
            compress_ratio=0,
            logical_topk=0,
        ),
        dict(
            name="8k-c2",
            seq_lens=[8192],
            query_lens=[8192],
            compress_ratio=2,
            logical_topk=512,
        ),
        dict(
            name="8k-c4",
            seq_lens=[8192],
            query_lens=[8192],
            compress_ratio=4,
            logical_topk=512,
        ),
        dict(
            name="8k-c128",
            seq_lens=[8192],
            query_lens=[8192],
            compress_ratio=128,
            logical_topk=512,
        ),
        dict(
            name="8k-c2-8req",
            seq_lens=[1024] * 8,
            query_lens=[1024] * 8,
            compress_ratio=2,
            logical_topk=512,
        ),
        dict(
            name="16k-c128",
            seq_lens=[16384],
            query_lens=[16384],
            compress_ratio=128,
            logical_topk=512,
        ),
    ]

    metadata = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "args": vars(args),
    }
    print(json.dumps(metadata), flush=True)
    for case in cases:
        print(json.dumps(benchmark_case(case, args)), flush=True)


if __name__ == "__main__":
    main()
