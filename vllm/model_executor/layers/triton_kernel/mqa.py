"""
Paged multi-query attention.
"""
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    query,  # [bsz, num_query_tokens, num_heads, head_size]
    output,  # [bsz, num_query_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables,  # [bsz, max_num_blocks]
    context_lens,  # [bsz]
    stride_query_0,
    stride_query_1,
    stride_query_2,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    stride_key_0,
    stride_key_1,
    stride_key_2,
    stride_key_3,
    stride_value_0,
    stride_value_1,
    stride_value_2,
    stride_block_tables_0,
    head_size: tl.constexpr,
    num_query_tokens: tl.constexpr,
    padded_num_query_tokens: tl.constexpr,
    x: tl.constexpr,
    block_size: tl.constexpr,
    sm_scale: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_head_idx = head_idx // num_queries_per_kv
    ctx_len = tl.load(context_lens + batch_idx)

    offs_num_query_tokens = tl.arange(0, padded_num_query_tokens)
    offs_block_size = tl.arange(0, block_size)
    offs_head_size = tl.arange(0, head_size)

    # load q from block ptr of shape [padded_num_query_tokens, head_size]
    offs_q = batch_idx * stride_query_0 + \
        offs_num_query_tokens[:, None] * stride_query_1 + head_idx * \
        stride_query_2 + offs_head_size[None, :]
    q = tl.load(query + offs_q,
                mask=offs_num_query_tokens[:, None] < num_query_tokens,
                other=0.0)

    m_i = tl.zeros([padded_num_query_tokens], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([padded_num_query_tokens], dtype=tl.float32)
    acc = tl.zeros([padded_num_query_tokens, head_size], dtype=tl.float32)

    for block_idx in range(0, tl.cdiv(ctx_len, block_size)):
        token_offs = (block_idx * block_size + offs_block_size)
        block_offs = tl.load(block_tables + batch_idx * stride_block_tables_0 +
                             token_offs // block_size,
                             mask=token_offs < ctx_len)
        # load k from block ptr of shape [head_size, block_size]
        offs_k = block_offs[None, :] * stride_key_0 + kv_head_idx * stride_key_1 + \
            (offs_head_size[:, None] // x) * stride_key_2 + \
            offs_block_size[None, :] * stride_key_3 + \
            offs_head_size[:, None] % x
        k = tl.load(
            key_cache + offs_k,
            mask=block_idx * block_size + offs_block_size[None, :] < ctx_len,
            other=0.0)
        # calculate qk
        qk = tl.zeros([padded_num_query_tokens, block_size], dtype=tl.float32)
        qk += tl.dot(q, k)
        # apply causal mask
        offs_a = ctx_len - num_query_tokens + offs_num_query_tokens[:, None]
        offs_b = block_idx * block_size + offs_block_size[None, :]
        mask = offs_a >= offs_b
        qk = qk * sm_scale + tl.where(mask, 0, -1.0e6)
        # calculate m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        # update m_i and l_i
        alpha = tl.exp(m_i - m_ij)
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        # rescale previous acc
        acc = acc * alpha[:, None]
        # load v from block ptr of shape [block_size, head_size]
        offs_v = block_offs[:, None] * stride_value_0 + kv_head_idx * stride_value_1 + \
            offs_head_size[None, :] * stride_value_2 + offs_block_size[:, None]
        v = tl.load(value_cache + offs_v)
        # update acc
        acc += tl.dot(p.to(v.dtype), v)
    # divide acc by row sum
    acc = acc / l_i[:, None]
    # update output
    offs_output = batch_idx * stride_output_0 + \
        offs_num_query_tokens[:, None] * stride_output_1 + head_idx * \
        stride_output_2 + offs_head_size[None, :]
    tl.store(output + offs_output,
             acc.to(output.type.element_ty),
             mask=offs_num_query_tokens[:, None] < num_query_tokens)
    return


def paged_multi_query_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> None:
    assert alibi_slopes is None, "Alibi is currently not supported."
    # Note that triton requires A.shape(0) to be at least 16 in tl.dot(A, B).
    padded_num_query_tokens = 16
    num_tokens, num_heads, head_size = query.shape
    bsz = context_lens.shape[0]
    num_query_tokens = num_tokens // bsz
    block_size = value_cache.shape[3]
    # Here we limit num_query_tokens <= 16 since it's rarely useful to evaluate more than 16
    # tokens in speculative decoding.
    assert num_query_tokens <= padded_num_query_tokens, \
        f"The num_query_tokens ({num_query_tokens}) must be <= {padded_num_query_tokens}"
    assert num_heads % num_kv_heads == 0
    num_queries_per_kv = num_heads // num_kv_heads
    x = key_cache.shape[-1]
    grid = (bsz, num_heads)
    query = query.view(bsz, num_query_tokens, num_heads, head_size)
    output = output.view(bsz, num_query_tokens, num_heads, head_size)
    _fwd_kernel[grid](
        query,  # [bsz, num_query_tokens, num_heads, head_size]
        output,  # [bsz, num_query_tokens, num_heads, head_size]
        key_cache,  # [num_blocks, num_kv_heads, head_size/x, block_size, x]
        value_cache,  # [num_blocks, num_kv_heads, head_size, block_size]
        block_tables,  # [bsz, max_num_blocks]
        context_lens,  # [bsz]
        query.stride(0),
        query.stride(1),
        query.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        head_size,
        num_query_tokens,
        padded_num_query_tokens,
        x,
        block_size,
        scale,
        num_queries_per_kv,
    )
