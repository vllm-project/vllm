import argparse
import random
import time
from typing import Optional

import torch

from vllm._C import ops
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, create_kv_caches_with_random

NUM_BLOCKS = 1024
PARTITION_SIZE = 512


@torch.inference_mode()
def main(
    version: str,
    num_seqs: int,
    context_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "cuda",
    kv_cache_dtype: Optional[str] = None,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device=device)

    context_lens = [context_len for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device=device)

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)

    # Create the KV cache.
    key_caches, value_caches = create_kv_caches_with_random(NUM_BLOCKS,
                                                            block_size,
                                                            1,
                                                            num_kv_heads,
                                                            head_size,
                                                            kv_cache_dtype,
                                                            dtype,
                                                            device=device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v2":
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) //
                          PARTITION_SIZE)
        tmp_output = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        # Using default kv_scale
        kv_scale = 1.0

        for _ in range(num_iters):
            if version == "v1":
                ops.paged_attention_v1(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    num_kv_heads,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_context_len,
                    alibi_slopes,
                    kv_cache_dtype,
                    kv_scale,
                )
            elif version == "v2":
                ops.paged_attention_v2(
                    output,
                    exp_sums,
                    max_logits,
                    tmp_output,
                    query,
                    key_cache,
                    value_cache,
                    num_kv_heads,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_context_len,
                    alibi_slopes,
                    kv_cache_dtype,
                    kv_scale,
                )
            else:
                raise ValueError(f"Invalid version: {version}")
        torch.cuda.synchronize()

        end_time = time.perf_counter()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return (end_time - start_time) / num_iters

    # Warmup.
    print("Warming up...")
    run_benchmark = run_cuda_benchmark
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    print(f"Kernel running time: {latency * 1000000:.3f} us")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["v1", "v2"],
                        default="v2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--num-query-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 256],
                        default=128)
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    main(
        version=args.version,
        num_seqs=args.batch_size,
        context_len=args.context_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        use_alibi=args.use_alibi,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
        kv_cache_dtype=args.kv_cache_dtype,
    )
