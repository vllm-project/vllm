import functools
import random
import time

import torch

from cacheflow import cache_ops


def benchmark(name, f, size: int, num_warmup = 10, num_iters = 100):
    for _ in range(num_warmup):
        f()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        f()
    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / num_iters
    print(f'[Latency] {name}: {avg_time * 1000:.3f} ms')
    print(f'[Throughput] {name}: {size / avg_time / 2 ** 30:.3f} GB/s')


@torch.inference_mode()
def test_gather_cached_kv(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
) -> None:
    print(f'num_tokens: {num_tokens}, num_heads: {num_heads}, '
          f'head_size: {head_size}, block_size: {block_size}, '
          f'num_blocks: {num_blocks}, dtype: {dtype}')

    num_slots = block_size * num_blocks
    slot_mapping = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int, device='cuda')

    qkv = torch.randn(
        num_tokens, 3, num_heads, head_size, dtype=dtype, device='cuda')
    _, key, value = qkv.unbind(dim=1)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_cache = torch.randn(size=key_cache_shape, dtype=dtype, device='cuda')

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_cache = torch.randn(
        size=value_cache_shape, dtype=dtype, device='cuda')

    # Run Flash attention.
    def run():
        cache_ops.gather_cached_kv(key, value, key_cache, value_cache, slot_mapping)

    benchmark('gather_cached_kv', run,
              size=num_tokens * num_heads * head_size * 2 * qkv.element_size())


if __name__ == '__main__':
    BLOCK_SIZE = 8
    NUM_BLOCKS = 1024
    DTYPE = torch.half

    # LLaMA-13B and OPT-13B
    NUM_HEADS = 40
    HEAD_SIZE = 128

    run_benchmark = functools.partial(
        test_gather_cached_kv,
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        block_size=BLOCK_SIZE,
        num_blocks=NUM_BLOCKS,
        dtype=DTYPE,
    )

    for i in range(6, 12):
        run_benchmark(num_tokens=2 ** i)
