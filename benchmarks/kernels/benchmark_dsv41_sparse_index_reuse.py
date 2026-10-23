# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the upper bound from reusing DeepSeek-V4.1 derived sparse indices.

The baseline runs one index build per compressed layer. The reuse arm runs
one per ``index_source_layer_ids`` entry. Correctness is covered by the model
unit tests; this script measures only the kernel work removed from a layer
sweep.
"""

import argparse
import json
import statistics

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.deepseek_v41.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
)

TOP_K = 512
BLOCK_SIZE = 64
WINDOW_SIZE = 128


def _decode_inputs(num_tokens: int, device: torch.device):
    topk_indices = torch.randint(
        0, 4096, (num_tokens, TOP_K), dtype=torch.int32, device=device
    )
    token_to_req = torch.arange(num_tokens, dtype=torch.int32, device=device)
    block_table = torch.arange(num_tokens * 64, dtype=torch.int32, device=device).view(
        num_tokens, 64
    )
    is_valid = torch.ones(num_tokens, dtype=torch.bool, device=device)
    return topk_indices, token_to_req, block_table, is_valid


def _prefill_inputs(num_tokens: int, device: torch.device):
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    gather_lens = seq_lens.clone()
    topk_indices = torch.randint(
        0, 4096, (num_tokens, TOP_K), dtype=torch.int32, device=device
    )
    compressed_len = (num_tokens + 1) // 2
    kv_len = 2 * compressed_len + num_tokens + 8
    width = (TOP_K + WINDOW_SIZE + 127) // 128 * 128
    indices = torch.empty((num_tokens, width), dtype=torch.int32, device=device)
    lens = torch.empty(num_tokens, dtype=torch.int32, device=device)
    return (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        kv_len,
        compressed_len,
        width,
        indices,
        lens,
    )


def _bench(fn, input_args: tuple, repeat_time_ms: int):
    times = bench_gpu_time_with_cupti(
        fn,
        repeat_time_ms=repeat_time_ms,
        cold_l2_cache=True,
        use_cuda_graph=True,
        input_args=input_args,
    )
    return statistics.median(times) * 1000, len(times)


def benchmark_decode(num_tokens: int, baseline_layers: int, reuse_layers: int, args):
    device = torch.device("cuda")
    topk_indices, token_to_req, block_table, is_valid = _decode_inputs(
        num_tokens, device
    )
    inputs = (topk_indices, token_to_req, block_table, BLOCK_SIZE, is_valid)

    def sweep(layers: int):
        def run(*args):
            for _ in range(layers):
                compute_global_topk_indices_and_lens(*args)

        return run

    baseline_us, baseline_n = _bench(
        sweep(baseline_layers), inputs, args.repeat_time_ms
    )
    reuse_us, reuse_n = _bench(sweep(reuse_layers), inputs, args.repeat_time_ms)
    return {
        "phase": "decode",
        "num_tokens": num_tokens,
        "baseline_layers": baseline_layers,
        "reuse_layers": reuse_layers,
        "baseline_us": baseline_us,
        "reuse_us": reuse_us,
        "speedup": baseline_us / reuse_us,
        "samples": [baseline_n, reuse_n],
    }


def benchmark_prefill(num_tokens: int, baseline_layers: int, reuse_layers: int, args):
    device = torch.device("cuda")
    (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        kv_len,
        compressed_len,
        width,
        indices,
        lens,
    ) = _prefill_inputs(num_tokens, device)
    inputs = (
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        WINDOW_SIZE,
        2,
        TOP_K,
        kv_len,
        compressed_len,
    )

    def sweep(layers: int, fill_cache: bool):
        def run(*run_inputs):
            for _ in range(layers):
                if fill_cache:
                    indices.fill_(-1)
                combine_topk_swa_indices(*run_inputs, out=(indices, lens))

        return run

    baseline_us, baseline_n = _bench(
        sweep(baseline_layers, fill_cache=False), inputs, args.repeat_time_ms
    )
    reuse_us, reuse_n = _bench(
        sweep(reuse_layers, fill_cache=True), inputs, args.repeat_time_ms
    )
    return {
        "phase": "prefill",
        "num_tokens": num_tokens,
        "width": width,
        "baseline_layers": baseline_layers,
        "reuse_layers": reuse_layers,
        "baseline_us": baseline_us,
        "reuse_us": reuse_us,
        "speedup": baseline_us / reuse_us,
        "samples": [baseline_n, reuse_n],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-time-ms", type=int, default=300)
    parser.add_argument("--decode-tokens", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument("--prefill-tokens", type=int, nargs="+", default=[4096, 8192])
    args = parser.parse_args()

    metadata = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": vars(args),
    }
    print(json.dumps(metadata), flush=True)
    for num_tokens in args.decode_tokens:
        print(json.dumps(benchmark_decode(num_tokens, 38, 8, args)), flush=True)
    for num_tokens in args.prefill_tokens:
        print(json.dumps(benchmark_prefill(num_tokens, 40, 9, args)), flush=True)


if __name__ == "__main__":
    main()
