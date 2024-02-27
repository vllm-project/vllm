from typing import Optional
import argparse
import random
import time

import numpy as np
import torch

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_func, flash_attn_with_kvcache = None, None

from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm._C import cache_ops
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, create_kv_caches_with_random

NUM_BLOCKS = 1024


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

    use_flash_attn = version in ["flash-attn", "flash-attn-kvcache"]
    if use_flash_attn:
        if dtype not in [torch.half, torch.bfloat16
                         ] or kv_cache_dtype != "auto":
            raise ValueError(
                "skip: flash-attn requires dtype and kv_cache_dtype to be half or bfloat16"
            )

    context_lens = [context_len for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens_tensor = torch.tensor(context_lens,
                                       dtype=torch.int,
                                       device=device)
    zero_context_lens_tensor = torch.zeros_like(context_lens_tensor)

    scale = float(1.0 / (head_size**0.5))
    qkv = torch.empty(num_seqs,
                      max_context_len,
                      num_query_heads + 2 * num_kv_heads,
                      head_size,
                      dtype=dtype,
                      device=device)
    qkv.uniform_(-scale, scale)
    query, key, value = qkv.split(
        [num_query_heads, num_kv_heads, num_kv_heads], dim=2)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads

    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device=device)

    # Create the block tables.
    if use_flash_attn:
        block_size = ((block_size + 256 - 1) // 256) * 256
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables, slot_mapping = [], []
    for seq_idx in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
        slot_mapping.append([])
        for i in range(context_lens[seq_idx]):
            block_number = block_table[i // block_size]
            block_offset = i % block_size
            slot = block_number * block_size + block_offset
            slot_mapping[-1].append(slot)
        for _ in range(max_context_len - context_lens[seq_idx]):
            slot_mapping[-1].append(-1)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=device)

    # Create the KV cache.
    key_caches, value_caches = create_kv_caches_with_random(
        NUM_BLOCKS,
        block_size,
        1,
        num_kv_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        device=device,
        use_flash_attn=use_flash_attn)
    key_cache, value_cache = key_caches[0], value_caches[0]

    if version == "xformers":
        attn_bias = BlockDiagonalCausalMask.from_seqlens(context_lens)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            key_repeated = torch.repeat_interleave(key,
                                                   num_queries_per_kv,
                                                   dim=2)
            value_repeated = torch.repeat_interleave(value,
                                                     num_queries_per_kv,
                                                     dim=2)
        else:
            key_repeated = key
            value_repeated = value

    def run_cuda_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            if version == "xformers":
                cache_ops.reshape_and_cache(
                    key.reshape(-1, *key.shape[2:]),
                    value.reshape(-1, *key.shape[2:]),
                    key_cache,
                    value_cache,
                    slot_mapping.flatten(),
                    kv_cache_dtype,
                )
                output = xops.memory_efficient_attention_forward(
                    query.reshape(1, -1, *query.shape[2:]),
                    key_repeated.reshape(1, -1, *key_repeated.shape[2:]),
                    value_repeated.reshape(1, -1, *value_repeated.shape[2:]),
                    attn_bias=attn_bias,
                    p=0.0,
                    scale=scale,
                )
                output = output.reshape(query.shape)
            elif version == "flash-attn":
                flat_slot_mapping = slot_mapping.flatten()
                slot_block_index = flat_slot_mapping // block_size
                slot_block_offset = flat_slot_mapping % block_size
                key_cache[slot_block_index,
                          slot_block_offset, :, :] = key.reshape(
                              -1, *key.shape[2:])
                value_cache[slot_block_index,
                            slot_block_offset, :, :] = value.reshape(
                                -1, *key.shape[2:])
                output = flash_attn_func(
                    q=query,
                    k=key,
                    v=value,
                    softmax_scale=scale,
                    causal=True,
                    alibi_slopes=alibi_slopes,
                )
            elif version == "flash-attn-kvcache":
                output = flash_attn_with_kvcache(
                    q=query,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    k=key,
                    v=value,
                    cache_seqlens=zero_context_lens_tensor,
                    block_table=block_tables,
                    softmax_scale=scale,
                    causal=True,
                    alibi_slopes=alibi_slopes,
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
    print(
        f"Version: {version}, Context Length: {context_len}, Batch size: {num_seqs}, Kernel running time: {latency * 1000000:.3f} us"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument(
        "--version",
        type=str,
        choices=["xformers", "flash-attn", "flash-attn-kvcache"],
        default="xformers")
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
        choices=["auto", "fp8_e5m2"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    parser.add_argument("--device", type=str, choices=["cuda"], default="cuda")
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
