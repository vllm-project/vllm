# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
SEQ_LENS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 163840]
TOP_K_VALUES = [2048]


def create_decode_data(batch_size: int, seq_len: int, dtype: torch.dtype):
    logits = torch.randn((batch_size, seq_len), dtype=dtype, device="cuda")

    min_len = int(seq_len * 0.8)
    seq_lens = torch.randint(
        min_len, seq_len + 1, (batch_size,), dtype=torch.int32, device="cuda"
    )

    positions = torch.arange(seq_len, device="cuda", dtype=torch.int32).unsqueeze(0)
    mask = positions >= seq_lens.unsqueeze(1)
    logits = logits.masked_fill(mask, float("-inf"))

    return logits, seq_lens


def get_decode_configs():
    configs = list(itertools.product(BATCH_SIZES, SEQ_LENS, TOP_K_VALUES))
    valid_configs = [(b, s, k) for b, s, k in configs if s >= k]

    if len(valid_configs) < len(configs):
        removed = len(configs) - len(valid_configs)
        print(f"Note: Filtered out {removed} invalid configs where seq_len < topk")

    return valid_configs


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size", "seq_len", "topk"],
        x_vals=[list(_) for _ in get_decode_configs()],
        line_arg="provider",
        line_vals=["vllm_decode", "large_context_topk", "flashinfer"],
        line_names=[
            "vLLM top_k_per_row_decode",
            "vLLM large_context_topk",
            "FlashInfer radix_topk",
        ],
        styles=[("blue", "--"), ("orange", "-."), ("green", "-")],
        ylabel="Latency (μs)",
        plot_name="top-k-decode",
        args={},
    )
)
def bench_decode(batch_size, seq_len, topk, provider):
    dtype = torch.float32
    next_n = 1

    logits, seq_lens = create_decode_data(batch_size, seq_len, dtype)
    lengths = seq_lens.clone()
    indices = torch.empty((batch_size, topk), dtype=torch.int32, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm_decode":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                indices,
                batch_size,
                logits.stride(0),
                logits.stride(1),
                topk,
            ),
            quantiles=quantiles,
            rep=200,
        )
    elif provider == "large_context_topk":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.ops._C.large_context_topk(logits, indices, lengths, None),
            quantiles=quantiles,
            rep=200,
        )
    else:  # flashinfer
        workspace = torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda")
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.ops._C.flashinfer_radix_topk(
                logits, lengths, indices, workspace, topk
            ),
            quantiles=quantiles,
            rep=200,
        )

    return ms * 1000, max_ms * 1000, min_ms * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark top-k decode kernels")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save benchmark results",
    )

    args = parser.parse_args()

    if not current_platform.is_cuda():
        raise RuntimeError("This benchmark requires CUDA")

    print("\n" + "=" * 70)
    print("TOP-K DECODE KERNEL BENCHMARKS")
    print("=" * 70)
    print(
        "Kernels: top_k_per_row_decode vs large_context_topk vs flashinfer_radix_topk"
    )
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Sequence lengths: {SEQ_LENS}")
    print(f"Top-k values: {TOP_K_VALUES}")
    if args.save_path:
        print(f"Saving results to: {args.save_path}")
    print("=" * 70 + "\n")

    bench_decode.run(print_data=True, save_path=args.save_path)

    print("\n" + "=" * 70)
    print("Benchmarking complete!")
    print("=" * 70)
