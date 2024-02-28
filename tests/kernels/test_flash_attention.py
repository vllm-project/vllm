import random
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.attention import (
    flash_attn_with_kvcache_paged, )
from vllm.utils import get_max_shared_memory_bytes

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
NUM_BLOCKS = 128  # Arbitrary values for testing
PARTITION_SIZE = 512

DTYPES = [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [3, 6, 17]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3, 6, 17]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing
NUM_HEADS_SMALL = NUM_HEADS
# head size should be bigger than or equal to block size.
HEAD_SIZES = [256]
# TODO(sang): https://github.com/Dao-AILab/flash-attention/pull/824
# should fix the block size. But right now, the block size should be
# divisible by 256.
BLOCK_SIZES = [256]
USE_ALIBI = [False, True]
SEEDS = [0]
# PAD_CONFIGS = [(0, 0), (8, MAX_SEQ_LEN - 1000), (16, MAX_SEQ_LEN - 2000)]
PAD_CONFIGS = [(0, 0)]


def pad_attention_inputs(
    pad_config: Tuple[int, int],
    block_size: int,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    max_context_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad the attention inputs to the specified batch size and context length.
    """
    pad_batch_size, pad_max_context_len = pad_config
    if pad_batch_size == 0:
        return query, block_tables, context_lens, max_context_len
    target_batch_size = (
        (query.shape[0] - 1) % pad_batch_size + 1) * pad_batch_size
    target_block_size = pad_max_context_len // block_size + 1
    padded_query = F.pad(query,
                         (0, 0, 0, 0, 0, target_batch_size - query.shape[0]))
    padded_block_table = F.pad(block_tables,
                               (0, target_block_size - block_tables.shape[1],
                                0, target_batch_size - block_tables.shape[0]))
    padded_context_lens = F.pad(context_lens,
                                (0, target_batch_size - context_lens.shape[0]))
    return padded_query, padded_block_table, padded_context_lens, pad_max_context_len


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    flash_style: bool = False,
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[-2]
    head_size = value_cache.shape[-1]
    block_size = value_cache.shape[-3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            if flash_style:
                k = key_cache[block_number, block_offset, :, :]
            else:
                k = key_cache[block_number, :, :, block_offset, :]
                k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            if flash_style:
                v = value_cache[block_number, block_offset, :, :]
            else:
                v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len, device="cuda").int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", [False, True])
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("pad_config", PAD_CONFIGS)
@torch.inference_mode()
def test_flash_paged_attention(
    kv_cache_factory,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    pad_config: Tuple[int, int],
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
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    max_seq_len = MAX_SEQ_LEN if not pad_config[0] else (pad_config[1] - 1000)
    context_lens = [random.randint(1, max_seq_len) for _ in range(num_seqs)]
    context_lens[-1] = max_seq_len
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
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS,
                                                block_size,
                                                1,
                                                num_kv_heads,
                                                head_size,
                                                dtype,
                                                seed,
                                                flash_style=True)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Call the paged attention kernel.
    output = torch.empty_like(query)

    padded_query, padded_block_table, padded_context_lens, pad_max_context_len = \
        pad_attention_inputs(pad_config, block_size, query,
                             block_tables, context_lens, max_context_len)

    output = flash_attn_with_kvcache_paged(
        padded_query,
        key_cache,
        value_cache,
        scale,
        padded_block_table,
        padded_context_lens,
        block_size,
        alibi_slopes,
    )

    # Run the reference implementation.
    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        alibi_slopes,
        flash_style=True,
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)
