import random
from typing import List, Optional, Tuple

import time
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm import attention_ops
from vllm.utils import get_max_shared_memory_bytes

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
NUM_BLOCKS = 128  # Arbitrary values for testing

DTYPES = [torch.half]
NUM_GEN_SEQS = [1, 4, 16, 64, 128]  # Arbitrary values for testing
CONTEXT_LENS = [1024, 2048, 4096, 8192, 16384]  # Arbitrary values for testing
NUM_HEADS = [(40, 40)]  # Arbitrary values for testing
HEAD_SIZES = [128]
BLOCK_SIZES = [16]
USE_ALIBI = [False]
SEEDS = [0]

def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=dtype,
                                device='cuda')
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=dtype,
                                  device='cuda')
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches


@torch.inference_mode()
def test_single_query_cached_kv_attention(
    num_seqs: int,
    context_len: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    version: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
        num_queries_per_kv)
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    context_lens = [random.randint(1, context_len) for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    PARTITION_SIZE = 512
    num_partitions = (max_context_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    def f():
        num_seqs, num_heads, head_size = output.shape
        use_v1 = num_partitions == 1 or num_seqs * num_heads > 512
        if version == 1 or use_v1:
            attention_ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                alibi_slopes,
            )
        else:
            assert PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            attention_ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_context_len,
                alibi_slopes,
            )

    for _ in range(3):
        f()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        f()
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time: {(end - start) / 100 * 1000:.3f} ms")


if __name__ == "__main__":
    for context_len in CONTEXT_LENS:
        for num_seqs in NUM_GEN_SEQS:
            for num_heads in NUM_HEADS:
                    for dtype in DTYPES:
                        for version in [1, 2]:
                            print(
                                f"Testing: V{version} {num_seqs}, {context_len}"
                            )
                            test_single_query_cached_kv_attention(
                                num_seqs,
                                context_len,
                                num_heads,
                                128,
                                False,
                                16,
                                dtype,
                                0,
                                version,
                            )
