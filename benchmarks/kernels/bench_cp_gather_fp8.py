# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import math

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton

# DeepSeek V3 MLA dimensions
NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 576 BF16 output elements per token
ENTRY_BYTES = 656  # 512 FP8 + 16 scales + 128 BF16 RoPE
BLOCK_SIZE = 64  # tokens per physical cache block - get_supported_kernel_block_sizes

# Realistic prefill scenarios:
#   - 1 long prefill: single request, 16K-96K tokens
#   - 4 medium prefills: 4 requests, 4K-24K tokens each
#   - 16 shorter prefills: 16 requests, 1K-6K tokens each
SCENARIOS = [
    # (label, num_reqs, total_tokens_list)
    ("1-req", 1, [8192, 16384, 32768, 65536, 98304]),
    ("4-reqs", 4, [8192, 16384, 32768, 65536, 98304]),
    ("16-reqs", 16, [8192, 16384, 32768, 65536, 98304]),
]


def make_inputs(total_tokens, num_reqs, block_size):
    """Create synthetic FP8 cache, block table, and output buffer.

    Fills the cache with random bytes (we only measure throughput,
    not correctness). Block table maps each request to contiguous
    physical blocks.
    """
    # Divide tokens evenly across requests
    base_len = total_tokens // num_reqs
    remainder = total_tokens % num_reqs
    seq_lens = [base_len + (1 if r < remainder else 0) for r in range(num_reqs)]

    # workspace_starts: cumulative sum of seq_lens
    workspace_starts = [0] * num_reqs
    for r in range(1, num_reqs):
        workspace_starts[r] = workspace_starts[r - 1] + seq_lens[r - 1]

    # Physical blocks needed per request
    blocks_per_req = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(blocks_per_req)
    max_blocks = max(blocks_per_req)

    # Allocate cache with random data (content doesn't matter for perf)
    cache = torch.randint(
        0,
        256,
        (total_blocks, block_size, ENTRY_BYTES),
        dtype=torch.uint8,
        device="cuda",
    )

    # Block table: contiguous block assignments
    block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device="cuda")
    block_idx = 0
    for r in range(num_reqs):
        for b in range(blocks_per_req[r]):
            block_table[r, b] = block_idx
            block_idx += 1

    # Output workspace
    dst = torch.zeros(total_tokens, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    workspace_starts_t = torch.tensor(
        workspace_starts, dtype=torch.int32, device="cuda"
    )

    return cache, dst, block_table, seq_lens_t, workspace_starts_t


def bench_scenario(label, num_reqs, total_tokens_list, save_path):
    """Run benchmark for a specific (num_reqs, total_tokens) scenario."""

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["total_tokens"],
            x_vals=total_tokens_list,
            line_arg="provider",
            line_vals=["cuda_kernel"],
            line_names=["cp_gather_fp8 (CUDA)"],
            styles=[("green", "-")],
            ylabel="Latency (us)",
            plot_name=f"cp_gather_fp8-{label}-bs{BLOCK_SIZE}",
            args={"num_reqs": num_reqs},
        )
    )
    def bench_fn(total_tokens, provider, num_reqs):
        cache, dst, block_table, seq_lens_t, ws_starts = make_inputs(
            total_tokens, num_reqs, BLOCK_SIZE
        )

        quantiles = [0.5, 0.2, 0.8]

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: ops.cp_gather_and_upconvert_fp8_kv_cache(
                cache, dst, block_table, seq_lens_t, ws_starts, num_reqs
            ),
            quantiles=quantiles,
            rep=500,
        )

        return ms * 1000, max_ms * 1000, min_ms * 1000  # us

    seq_len_per_req = total_tokens_list[0] // num_reqs
    seq_len_per_req_max = total_tokens_list[-1] // num_reqs
    print(
        f"\n--- {label}: {num_reqs} request(s), "
        f"~{seq_len_per_req}-{seq_len_per_req_max} tokens/req ---"
    )
    bench_fn.run(print_data=True, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark cp_gather_and_upconvert_fp8_kv_cache"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save benchmark results as CSV",
    )
    args = parser.parse_args()

    # Print data volume info for bandwidth analysis
    read_per_token = ENTRY_BYTES  # 656 bytes from cache
    write_per_token = HEAD_DIM * 2  # 576 * 2 = 1152 bytes to workspace
    total_per_token = read_per_token + write_per_token  # 1808 bytes

    print("\n" + "=" * 70)
    print("CP_GATHER_AND_UPCONVERT_FP8_KV_CACHE BENCHMARKS")
    print("=" * 70)
    print(f"Cache entry: {ENTRY_BYTES} bytes (512 FP8 + 16 scales + 128 RoPE)")
    print(f"Output row:  {HEAD_DIM} BF16 = {HEAD_DIM * 2} bytes")
    print(f"Per token:   {total_per_token} bytes (read + write)")
    print(f"Block size:  {BLOCK_SIZE} tokens/block")
    print("=" * 70)

    for label, num_reqs, total_tokens_list in SCENARIOS:
        bench_scenario(label, num_reqs, total_tokens_list, args.save_path)

    print("\n" + "=" * 70)
    print("Benchmarking complete!")
    print("=" * 70)
