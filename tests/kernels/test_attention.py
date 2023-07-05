import random
from typing import List, Optional

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm import attention_ops

MAX_SEQ_LEN = 4096
TEST_SEED = 0


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = torch.einsum('qhd,khd->hqk', query, key)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum('hqk,khd->qhd', attn, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
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
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
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


@torch.inference_mode()
def run_single_query_cached_kv_attention(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
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
                            dtype=dtype,
                            device='cuda')
    key_cache.uniform_(-1e-3, 1e-3)
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.empty(size=(num_blocks, *value_block_shape),
                              dtype=dtype,
                              device='cuda')
    value_cache.uniform_(-1e-3, 1e-3)

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

    scale = float(1.0 / (head_size**0.5))
    output = torch.empty(num_tokens,
                         num_heads,
                         head_size,
                         dtype=dtype,
                         device='cuda')
    attention_ops.single_query_cached_kv_attention(
        output,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        None,  # ALiBi slopes.
    )

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
    )
    # NOTE(woosuk): Due to the difference in the data types the two
    # implementations use for attention softmax logits and accumulation,
    # there is a small difference in the final outputs.
    # We should use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


@torch.inference_mode()
def run_multi_query_kv_attention(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    seq_lens = random.sample(range(1, MAX_SEQ_LEN), num_seqs)
    num_tokens = sum(seq_lens)

    scale = float(1.0 / (head_size**0.5))
    qkv = torch.empty(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    qkv.uniform_(-1e-3, 1e-3)
    query, key, value = qkv.unbind(dim=1)

    attn_op = xops.fmha.cutlass.FwOp()
    attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_lens)
    output = xops.memory_efficient_attention_forward(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
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
        dtype,
    )
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


def test_single_query_cached_kv_attention() -> None:
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        for block_size in [8, 16, 32]:
            for head_size in [64, 80, 96, 128]:
                print(f'Testing single_query_cached_kv_attention with '
                      f'dtype={dtype}, block_size={block_size}, '
                      f'head_size={head_size}')
                run_single_query_cached_kv_attention(
                    num_tokens=37,
                    num_heads=3,
                    head_size=head_size,
                    block_size=block_size,
                    num_blocks=1024,
                    dtype=dtype,
                )


def test_multi_query_kv_attention() -> None:
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        for head_size in [64, 80, 96, 128]:
            print(f'Testing multi_query_kv_attention with dtype={dtype}, '
                  f'head_size={head_size}')
            run_multi_query_kv_attention(
                num_seqs=5,
                num_heads=3,
                head_size=head_size,
                dtype=dtype,
            )
