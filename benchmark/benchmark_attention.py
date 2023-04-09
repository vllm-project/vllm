import functools
import random
import time
from typing import List

from flash_attn.flash_attn_interface import _flash_attn_forward
import torch

from cacheflow import attention_ops


def benchmark(name, f, num_warmup = 10, num_iters = 100):
    for _ in range(num_warmup):
        f()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        f()
    torch.cuda.synchronize()
    end = time.time()
    print(f'{name}: {(end - start) / num_iters * 1000:.3f} ms')


@torch.inference_mode()
def benchmark_multi_query_cached_kv_attention(
    query_lens: List[int],
    context_lens: List[int],
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
) -> None:
    print(f'query_lens: {query_lens}, context_lens: {context_lens}, '
          f'num_heads: {num_heads}, head_size: {head_size}, block_size: '
          f'{block_size}, num_blocks: {num_blocks}, dtype: {dtype}')
    # Create query tensor.
    num_queries = len(query_lens)
    cu_query_lens = [0]
    for query_len in query_lens:
        cu_query_lens.append(cu_query_lens[-1] + query_len)
    num_total_tokens = cu_query_lens[-1]
    qkv = torch.randn(
        num_total_tokens, 3, num_heads, head_size, dtype=dtype, device='cuda')
    query, _, _ = qkv.unbind(dim=1)

    # Create key and value cache.
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (num_heads, head_size // x, block_size, x)
    key_cache = torch.randn(
        size=(num_blocks, *key_block_shape), dtype=dtype, device='cuda')
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.randn(
        size=(num_blocks, *value_block_shape), dtype=dtype, device='cuda')

    # Create block tables.
    max_context_len = max(context_lens)
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_queries):
        block_table = [
            random.randint(0, num_blocks - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')

    # Create input and output data structures.
    cu_query_lens = torch.tensor(cu_query_lens, dtype=torch.int, device='cuda')
    context_len_tensor = torch.tensor(context_lens, dtype=torch.int, device='cuda')
    scale = float(1.0 / (head_size ** 0.5))
    output = torch.empty(
        num_total_tokens, num_heads, head_size, dtype=dtype, device='cuda')

    # Run our implementation.
    def run_ours():
        attention_ops.multi_query_cached_kv_attention(
            cu_query_lens,
            output,
            query,
            key_cache,
            value_cache,
            scale,
            block_tables,
            context_len_tensor,
            block_size,
            max_context_len,
        )
    benchmark('Ours', run_ours)

    # Upper bound: Flash attention.
    # Becuase Flash attention cannot read our own cache,
    # we make key and value tensors contiguous.
    num_kv_tokens = sum(context_lens)
    cu_context_lens = [0]
    for context_len in context_lens:
        cu_context_lens.append(cu_context_lens[-1] + context_len)
    cu_context_lens = torch.tensor(cu_context_lens, dtype=torch.int, device='cuda')
    qkv = torch.randn(
        num_kv_tokens, 3, num_heads, head_size, dtype=dtype, device='cuda')
    _, key, value = qkv.unbind(dim=1)
    ref_output = torch.empty_like(output)

    # Run Flash attention.
    def run_flash_attn():
        _flash_attn_forward(
            query,
            key,
            value,
            ref_output,
            cu_query_lens,
            cu_context_lens,
            max(query_lens),
            max_context_len,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True,
            return_softmax=False,
        )
    benchmark('Flash attention', run_flash_attn)


if __name__ == '__main__':
    BLOCK_SIZE = 8
    NUM_BLOCKS = 1024
    DTYPE = torch.half

    # LLaMA-13B and OPT-13B
    NUM_HEADS = 40
    HEAD_SIZE = 128

    run_benchmark = functools.partial(
        benchmark_multi_query_cached_kv_attention,
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        dtype=DTYPE,
    )

    run_benchmark(
        query_lens=[64] * 1,
        context_lens=[64] * 1,
    )
    run_benchmark(
        query_lens=[128] * 1,
        context_lens=[128] * 1,
    )
    run_benchmark(
        query_lens=[64] * 8,
        context_lens=[64] * 8,
    )
    run_benchmark(
        query_lens=[128] * 8,
        context_lens=[128] * 8,
    )
    run_benchmark(
        query_lens=[64, 32, 16],
        context_lens=[128, 256, 64],
    )
    run_benchmark(
        query_lens=[1024],
        context_lens=[1024],
    )
