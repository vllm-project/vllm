import random
from typing import List, Optional

from flash_attn.flash_attn_interface import _flash_attn_forward
import torch

from cacheflow import attention_ops

MAX_SEQ_LEN = 4096


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

        scale = 1.0 / (head_size ** 0.5)
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
    scale = 1.0 / (head_size ** 0.5)

    num_seqs = len(cu_seq_lens) - 1
    ref_outputs = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask
        attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e5
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
    scale = 1.0 / (head_size ** 0.5)

    num_queries = len(cu_query_lens) - 1
    ref_outputs = []
    for i in range(num_queries):
        start_idx = cu_query_lens[i]
        end_idx = cu_query_lens[i + 1]
        query_len = end_idx - start_idx
        context_len = int(context_lens[i])
        block_table = block_tables[i]

        # Create attention mask
        attn_mask = torch.triu(
            torch.ones(query_len, context_len), diagonal=context_len - query_len + 1) * -1e5
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


def test_single_query_cached_kv_attention(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
) -> None:
    qkv = torch.randn(
        num_tokens, 3, num_heads, head_size, dtype=dtype, device='cuda')
    query, _, _ = qkv.unbind(dim=1)
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (num_heads, head_size // x, block_size, x)
    key_cache = torch.randn(
        size=(num_blocks, *key_block_shape), dtype=dtype, device='cuda')
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.randn(
        size=(num_blocks, *value_block_shape), dtype=dtype, device='cuda')

    # Adjust the range of the values to reduce precision errors.
    query = query / (head_size ** 0.5)
    key_cache = key_cache / (head_size ** 0.5)
    value_cache = value_cache / (head_size ** 0.5)

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

    scale = float(1.0 / (head_size ** 0.5))
    output = torch.empty(
        num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
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


def test_multi_query_kv_attention(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    seq_lens = random.sample(range(1, MAX_SEQ_LEN), num_seqs)
    max_seq_len = max(seq_lens)
    num_tokens = sum(seq_lens)

    cu_seq_lens = [0]
    for seq_len in seq_lens:
        cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
    cu_seq_lens = torch.tensor(cu_seq_lens, dtype=torch.int, device='cuda')

    scale = float(1.0 / (head_size ** 0.5))
    qkv = torch.randn(
        num_tokens, 3, num_heads, head_size, dtype=dtype, device='cuda')
    # Adjust the range of the values to reduce precision errors.
    qkv = qkv / (head_size ** 0.5)

    query, key, value = qkv.unbind(dim=1)
    output = torch.empty(
        num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    _flash_attn_forward(
        query,
        key,
        value,
        output,
        cu_seq_lens,
        cu_seq_lens,
        max_seq_len,
        max_seq_len,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=True,
        return_softmax=False,
    )

    cu_seq_lens = cu_seq_lens.cpu().tolist()
    ref_output = ref_multi_query_kv_attention(
        cu_seq_lens,
        query,
        key,
        value,
        dtype,
    )
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


def test_multi_query_cached_kv_attention(
    num_queries: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
) -> None:
    query_lens = random.sample(range(1, MAX_SEQ_LEN), num_queries)
    cu_query_lens = [0]
    for query_len in query_lens:
        cu_query_lens.append(cu_query_lens[-1] + query_len)
    num_total_tokens = cu_query_lens[-1]

    query = torch.randn(
        num_total_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (num_heads, head_size // x, block_size, x)
    key_cache = torch.randn(
        size=(num_blocks, *key_block_shape), dtype=dtype, device='cuda')
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.randn(
        size=(num_blocks, *value_block_shape), dtype=dtype, device='cuda')

    cu_query_lens = torch.tensor(cu_query_lens, dtype=torch.int, device='cuda')
    context_lens = [
        query_len + random.randint(0, MAX_SEQ_LEN - query_len)
        for query_len in query_lens
    ]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_queries):
        block_table = [
            random.randint(0, num_blocks - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')

    scale = float(1.0 / (head_size ** 0.5))
    output = torch.empty_like(query)

    attention_ops.multi_query_cached_kv_attention(
        cu_query_lens,
        output,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
    )

    ref_output = ref_multi_query_cached_kv_attention(
        cu_query_lens,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        dtype,
    )
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


@torch.inference_mode()
def test_attention(seed: int) -> None:
    # NOTE(woosuk): Even when the seed is fixed, there is a chance that
    # the test fails due to the precision issue. Re-run the test if it fails.
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    for dtype in [torch.half, torch.float]:
        for block_size in [8, 16]:
            for head_size in [32, 64, 80, 96, 128, 160, 192, 256]:
                print(f'Testing single_query_cached_kv_attention with '
                      f'dtype={dtype}, block_size={block_size}, '
                      f'head_size={head_size}')
                test_single_query_cached_kv_attention(
                    num_tokens=37,
                    num_heads=3,
                    head_size=head_size,
                    block_size=block_size,
                    num_blocks=1024,
                    dtype=dtype,
                )

    # NOTE(siyuan): Same as above. Re-run the test if it fails. Also
    # note that the test is also more likely to fail due to the much
    # larger amount of tokens in the input may increase the variance.
    for dtype in [torch.half, torch.float]:
        for block_size in [8, 16]:
            for head_size in [32, 64, 80, 96, 128, 160, 192, 256]:
                print(f'Testing multi_query_cached_kv_attention with '
                      f'dtype={dtype}, block_size={block_size}, '
                      f'head_size={head_size}')
                test_multi_query_cached_kv_attention(
                    num_queries=11,
                    num_heads=3,
                    head_size=head_size,
                    block_size=block_size,
                    num_blocks=1024,
                    dtype=dtype,
                )

    # NOTE(woosuk): FlashAttention does not support FP32.
    for dtype in [torch.half]:
        # NOTE(woosuk): FlashAttention does not support head_size > 128.
        for head_size in [64, 80, 96, 128]:
            print(f'Testing multi_query_kv_attention with dtype={dtype}, '
                  f'head_size={head_size}')
            test_multi_query_kv_attention(
                num_seqs=11,
                num_heads=3,
                head_size=head_size,
                dtype=dtype,
            )


if __name__ == '__main__':
    test_attention(seed=0)
