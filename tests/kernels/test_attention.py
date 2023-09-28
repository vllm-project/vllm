import random
from typing import List, Optional, Tuple

import pytest
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm import attention_ops

MAX_SEQ_LEN = 8192
NUM_BLOCKS = 128  # Arbitrary values for testing

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [1, 3, 7]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
BLOCK_SIZES = [8, 16, 32]
USE_ALIBI = [False, True]
SEEDS = [0]


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
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
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

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

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

def ref_single_query_cached_kv_attention_quantized(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    k_scale: float,
    k_zp: float,
    v_scale: float,
    v_zp: float,
) -> None:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            k = k.to(torch.float32)
            k = k * k_scale + k_zp
            k = k.to(q.dtype)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            v = v.to(torch.float32)
            v = v * v_scale + v_zp
            v = v.to(q.dtype)
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_size**0.5)
        out = ref_masked_attention(q, keys, values, scale)
        out = out.view(num_heads, head_size)
        output[i].copy_(out, non_blocking=True)


def ref_multi_query_kv_attention(
    cu_seq_lens: List[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    head_size = query.shape[-1]
    scale = 1.0 / (head_size**0.5)

    num_seqs = len(cu_seq_lens) - 1
    ref_outputs = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask.
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype),
                               diagonal=1)
        attn_mask = attn_mask * torch.finfo(dtype).min
        attn_mask = attn_mask.to(dtype=dtype, device='cuda')

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output


def ref_multi_query_cached_kv_attention(
    cu_query_lens: List[int],
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    scale = 1.0 / (head_size**0.5)

    num_queries = len(cu_query_lens) - 1
    ref_outputs = []
    for i in range(num_queries):
        start_idx = cu_query_lens[i]
        end_idx = cu_query_lens[i + 1]
        query_len = end_idx - start_idx
        context_len = int(context_lens[i])
        block_table = block_tables[i]

        # Create attention mask
        attn_mask = torch.triu(torch.ones(query_len, context_len),
                               diagonal=context_len - query_len + 1) * -1e5
        attn_mask = attn_mask.to(dtype=dtype, device='cuda')

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            keys,
            values,
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_single_query_cached_kv_attention(
    kv_cache_factory,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
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

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
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
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    attention_ops.single_query_cached_kv_attention(
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
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


def ref_multi_query_kv_attention(
    cu_seq_lens: List[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_seqs = len(cu_seq_lens) - 1
    ref_outputs = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask.
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype),
                               diagonal=1)
        attn_mask = attn_mask * torch.finfo(dtype).min
        attn_mask = attn_mask.to(dtype=dtype, device="cuda")

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output


# TODO(woosuk): Add tests for USE_ALIBI=True.
@pytest.mark.parametrize("num_seqs", NUM_PREFILL_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def run_single_query_cached_kv_attention_quantized(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    num_kv_heads: int = None,
    k_scale: float = 1e-2,
    k_zp: float = 0.0,
    v_scale: float = 1e-2,
    v_zp: float = 0.0,
) -> None:
    qkv = torch.empty(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    qkv.uniform_(-1e-3, 1e-3)
    query, _, _ = qkv.unbind(dim=1)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (num_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=(num_blocks, *key_block_shape),
                            dtype=torch.int8, ## fixed this to int8
                            device='cuda')
    key_cache.random_(-1, 2) ## change data range
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.empty(size=(num_blocks, *value_block_shape),
                              dtype=torch.int8, ## fixed this to int8
                              device='cuda')
    value_cache.random_(-1, 2) ## change data range

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')
    head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda")

    scale = float(1.0 / (head_size**0.5))

    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    assert num_heads % num_kv_heads == 0
    num_queries_per_kv = num_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
                     num_queries_per_kv)

    output = torch.empty(num_tokens,
                         num_heads,
                         head_size,
                         dtype=dtype,
                         device='cuda')
    attention_ops.single_query_cached_kv_quantized_attention(
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
        None,  # ALiBi slopes.
        k_scale,
        k_zp,
        v_scale,
        v_zp,
    )

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention_quantized(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        k_scale,
        k_zp,
        v_scale,
        v_zp,
    )
    # NOTE(woosuk): Due to the difference in the data types the two
    # implementations use for attention softmax logits and accumulation,
    # there is a small difference in the final outputs.
    # We should use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)

def test_single_query_cached_kv_attention_quantized() -> None:
    # FIXME: set TEST_SEED
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    for dtype in [
                  torch.half, 
                  torch.bfloat16, 
                  torch.float,
                  ]:
        for block_size in [8, 
                           16,
                           ]:
            for head_size in [64, 
                              80, 
                              96, 
                              112,  
                              128, 
                              256,
                              ]:
                print(f'Testing single_query_cached_kv_attention with '
                      f'dtype={dtype}, block_size={block_size}, '
                      f'head_size={head_size}')
                run_single_query_cached_kv_attention_quantized(
                    num_tokens=37,
                    num_heads=3,
                    head_size=head_size,
                    block_size=block_size,
                    num_blocks=1024,
                    dtype=dtype,
                )

@pytest.mark.parametrize("num_seqs", NUM_PREFILL_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def run_multi_query_kv_attention(
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    seq_lens = random.sample(range(1, MAX_SEQ_LEN), num_seqs)
    num_tokens = sum(seq_lens)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    qkv = torch.empty(num_tokens,
                      num_query_heads + 2 * num_kv_heads,
                      head_size,
                      dtype=dtype,
                      device="cuda")
    qkv.uniform_(-scale, scale)
    query, key, value = qkv.split(
        [num_query_heads, num_kv_heads, num_kv_heads], dim=1)

    num_queries_per_kv = num_query_heads // num_kv_heads
    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)
    attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_lens)
    output = xops.memory_efficient_attention_forward(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
    )
    output = output.squeeze(0)

    cu_seq_lens = [0]
    for seq_len in seq_lens:
        cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
    ref_output = ref_multi_query_kv_attention(
        cu_seq_lens,
        query,
        key,
        value,
        scale,
        dtype,
    )
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)

def test_multi_query_kv_attention() -> None:
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        for head_size in [64, 80, 96, 112, 128, 256]:
            print(f'Testing multi_query_kv_attention with dtype={dtype}, '
                  f'head_size={head_size}')
            run_multi_query_kv_attention(
                num_seqs=5,
                num_heads=3,
                head_size=head_size,
                dtype=dtype,
            )
