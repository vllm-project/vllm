# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn.functional as F


def ceil_div(a, b):
    return (a + b - 1) // b


def pad_to_multiple(a, b):
    return ceil_div(a, b) * b


def pad_to_next_power_of_2(a):
    return 2**int(a - 1).bit_length() if a > 0 else 0


def is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0


class BlockDiagonalCausalFromBottomRightMask:

    @staticmethod
    def _from_seqlens(query_lens, seq_lens, block_size=None):
        from torch import logical_and, logical_or

        contexted = block_size is None
        context_lens = seq_lens - query_lens
        n_queries = query_lens.sum().item()
        num_seqs = len(query_lens)
        if contexted:
            key_lens_blockaligned = seq_lens
        else:
            n_blocks_per_seq = (context_lens + block_size - 1) // block_size
            offset_per_seq = n_blocks_per_seq * block_size
            key_lens_blockaligned = offset_per_seq[:num_seqs]
        n_keys = key_lens_blockaligned.sum().item()

        a = (torch.arange(n_queries).reshape(n_queries,
                                             1).expand(n_queries, n_keys))
        b = torch.arange(n_keys).reshape(1, n_keys).expand(n_queries, n_keys)
        q_cumsum = F.pad(query_lens, (1, 0)).cumsum(dim=0)
        k_cumsum = F.pad(key_lens_blockaligned, (1, 0)).cumsum(dim=0)

        prior_mask = torch.zeros(n_queries, n_keys)
        new_masks: list[torch.Tensor] = []
        for seq_id in range(num_seqs):
            ri = q_cumsum[seq_id]
            ci = k_cumsum[seq_id]
            nr = query_lens[seq_id]

            if contexted:
                nc = seq_lens[seq_id]
                a_offset = ci + nc - ri - nr
                new_mask = (a + a_offset) >= b
            else:
                nc = context_lens[seq_id]
                a_offset = ci + nc - 1
                new_mask = a_offset >= b

            left_mask = b >= ci
            top_mask = a >= ri
            bottom_mask = a < (ri + nr)

            new_mask = logical_and(
                logical_and(logical_and(new_mask, left_mask), top_mask),
                bottom_mask,
            )
            prior_mask = logical_or(prior_mask, new_mask)
            new_masks = new_masks + [new_mask]
        return prior_mask

    @staticmethod
    def from_seqlens(query_lens, seq_lens, block_size=None):
        contexted = block_size is None
        if contexted:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens)
            active_mask = None
        else:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens, block_size)
            active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, query_lens)
        return prior_mask, active_mask


def ref_softmax(x: torch.Tensor,
                dim: int,
                mixed_precision=False,
                return_max_reduce=False):
    max_value = torch.amax(x, dim=dim, keepdims=True)
    exp = torch.exp(x - max_value)
    if mixed_precision:
        sum_value = torch.sum(exp.astype(torch.float32),
                              dim=dim,
                              keepdims=True).astype(x.dtype)
    else:
        sum_value = torch.sum(exp, dim=dim, keepdims=True)
    if return_max_reduce:
        return exp / sum_value, max_value, torch.reciprocal(sum_value)
    return exp / sum_value


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
    return_max_reduce: Optional[bool] = False,
) -> torch.Tensor:
    scaled_qk = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        masked_score = scaled_qk + attn_mask.float()
    if return_max_reduce:
        norm_score, cached_max, cached_sum_reciprocal = ref_softmax(
            masked_score, dim=-1, return_max_reduce=True)
    else:
        norm_score = ref_softmax(masked_score, dim=-1)
    out = torch.einsum("hqk,khd->qhd", norm_score.to(value.dtype), value)
    if return_max_reduce:
        return (
            out,
            cached_max,
            cached_sum_reciprocal,
            norm_score,
            masked_score,
            scaled_qk,
        )
    else:
        return (out, )


def is_int_tensor(x):
    return isinstance(x, torch.Tensor) and (x.dtype == torch.int32
                                            or x.dtype == torch.int64)


def ref_context_attention(
    query,
    key,
    value,
    query_lens,
    seq_lens,
    head_size,
    num_queries_per_kv,
    return_max_reduce=False,
):
    assert is_int_tensor(query_lens) and is_int_tensor(seq_lens)
    scale = float(1.0 / (head_size**0.5))
    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    attn_mask, _ = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        query_lens, seq_lens)

    # convert binary mask to -inf values
    attn_mask = torch.logical_not(attn_mask)
    attn_mask = attn_mask.float() * -30000

    output, *debug_tensors = ref_masked_attention(
        query,
        key,
        value,
        scale,
        attn_mask,
        return_max_reduce=return_max_reduce,
    )

    return output, debug_tensors


def _sample_lengths(num, min_len, max_len):
    return torch.randint(min_len, max_len + 1, size=(num, ))


def sample_input_sizes(
    prefill_batch_size,
    decode_batch_size,
    min_query_len,
    max_query_len,
    min_ctx_len,
    max_ctx_len,
):
    batch_size = prefill_batch_size + decode_batch_size
    prefill_query_lens = _sample_lengths(prefill_batch_size, min_query_len,
                                         max_query_len)
    decode_query_lens = torch.ones(decode_batch_size, dtype=torch.long)
    query_lens = torch.cat([prefill_query_lens, decode_query_lens])
    if max_ctx_len == 0:
        ctx_lens = torch.zeros(batch_size, dtype=torch.long)
    else:
        ctx_lens = _sample_lengths(batch_size, min_ctx_len, max_ctx_len)
    return query_lens, ctx_lens


def sample_paged_attention_inputs(
    query_lens,
    ctx_lens,
    max_block_per_request,
    num_blocks_in_cache,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    dtype,
    init_tensor=True,
):
    # max_model_len = (max_query_len + max_ctx_len) * 4
    query_lens = torch.tensor(query_lens, dtype=torch.long)
    ctx_lens = torch.tensor(ctx_lens, dtype=torch.long)
    seq_lens = query_lens + ctx_lens
    num_query_tokens = query_lens.sum()
    num_seq_tokens = seq_lens.sum()

    def _sample_tensor(shape, dtype):
        tensor = torch.empty(shape, dtype=dtype)
        if init_tensor:
            tensor.uniform_(-1, 1)
        return tensor

    # prepare query key value for vanilla (dense and contiguous) attention
    query = _sample_tensor(shape=(num_query_tokens, num_heads, head_size),
                           dtype=dtype)
    all_keys = _sample_tensor(shape=(num_seq_tokens, num_kv_heads, head_size),
                              dtype=dtype)
    all_values = _sample_tensor(shape=(num_seq_tokens, num_kv_heads,
                                       head_size),
                                dtype=dtype)

    # prepare for paged attention

    # sample block table
    batch_size = query_lens.shape[0]
    block_table = torch.randperm(num_blocks_in_cache)[:batch_size *
                                                      max_block_per_request]
    block_table = block_table.view(batch_size, max_block_per_request)

    k_cache = _sample_tensor(
        shape=(num_blocks_in_cache, block_size, num_kv_heads, head_size),
        dtype=dtype,
    )
    v_cache = _sample_tensor(
        shape=(num_blocks_in_cache, block_size, num_kv_heads, head_size),
        dtype=dtype,
    )

    if not init_tensor:
        # does not need to initialize
        k_new = torch.empty_like(query)
        v_new = torch.empty_like(query)
        return (
            query,
            k_new,
            v_new,
            k_cache,
            v_cache,
            block_table,
            all_keys,
            all_values,
        )

    # use broadcast add to convert block indices to token indices
    token_indices = (block_table * block_size).reshape(
        (batch_size, max_block_per_request,
         1)) + torch.arange(block_size).reshape((1, 1, -1))
    token_indices = token_indices.reshape((batch_size, -1))

    # prepare gather/scatter indices to set key/value into kv cache
    seq_starts = F.pad(torch.cumsum(seq_lens, dim=0), (1, 0))
    ctx_token_indices = []
    new_token_indices = []
    ctx_to_cache_indices = []
    for seq_id, (offset, q_len,
                 c_len) in enumerate(zip(seq_starts, query_lens, ctx_lens)):
        ctx_token_indices.append(torch.arange(c_len) + offset)
        new_token_indices.append(torch.arange(q_len) + (offset + c_len))
        ctx_to_cache_indices.append(token_indices[seq_id, :c_len])
    ctx_token_indices = torch.cat(ctx_token_indices)
    new_token_indices = torch.cat(new_token_indices)
    ctx_to_cache_indices = torch.cat(ctx_to_cache_indices)

    k_new = all_keys[new_token_indices]
    v_new = all_values[new_token_indices]
    if len(ctx_to_cache_indices) > 0:
        k_ctx = all_keys[ctx_token_indices].view(ctx_lens.sum(), -1)
        v_ctx = all_values[ctx_token_indices].view(ctx_lens.sum(), -1)
        k_cache.view(num_blocks_in_cache * block_size,
                     -1)[ctx_to_cache_indices] = k_ctx
        v_cache.view(num_blocks_in_cache * block_size,
                     -1)[ctx_to_cache_indices] = v_ctx

    return (
        query,
        k_new,
        v_new,
        k_cache,
        v_cache,
        block_table,
        all_keys,
        all_values,
    )


def get_active_block_tables(block_tables, context_lens, block_size,
                            num_blocks):
    blocks_per_seq = ceil_div(context_lens, block_size)
    num_seqs = len(context_lens)
    active_blocks: list[int] = []
    for seq_id in range(num_seqs):
        active_blocks.append(block_tables[seq_id, :blocks_per_seq[seq_id]])
    active_blocks = torch.cat(active_blocks)
    return F.pad(
        active_blocks,
        (0, num_blocks - len(active_blocks)),
        "constant",
        0,
    )


def convert_to_kernel_input_format(
    query_lens,
    context_lens,
    block_table,
    k_cache,
    v_cache,
    query,
    k_active,
    v_active,
    block_size,
    LARGE_TILE_SZ,
    max_num_queries,
):
    # calculate input shapes
    num_active_blocks = ceil_div(context_lens, block_size).sum().item()
    num_active_blocks = pad_to_multiple(num_active_blocks,
                                        LARGE_TILE_SZ // block_size)
    context_kv_len = num_active_blocks * block_size
    assert (context_kv_len %
            LARGE_TILE_SZ == 0), f"invalid context_kv_len={context_kv_len}"
    assert (max_num_queries
            >= query.shape[0]), f"invalid max_num_queries={max_num_queries}"

    # pad QKV tensors
    pad_dims = (
        0,
        0,
        0,
        0,
        0,
        max_num_queries - query.shape[0],
    )
    query = F.pad(query, pad_dims, "constant", 0)
    k = F.pad(k_active, pad_dims, "constant", 0)
    v = F.pad(v_active, pad_dims, "constant", 0)

    # permute QKV tensors
    # query: (1, n_heads, d, seq_q)
    # key:   (1, n_kv_heads, d, seq_k)
    # value: (1, n_kv_heads, seq_v, d)
    query = query.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    k = k.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    v = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    # permute KV cache tensors
    # cache layout: (n_blocks, n_kv_heads, block_size, d)
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    v_cache = v_cache.permute(0, 2, 1, 3).contiguous()

    # transform block table to active block tables
    active_block_table = get_active_block_tables(
        block_table,
        context_lens,
        block_size,
        num_active_blocks,
    ).to(torch.int32)  # kernel require int32

    # Build attention masks
    seq_lens = query_lens + context_lens
    prior_mask, active_mask = (
        BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            query_lens, seq_lens, block_size=block_size))
    active_mask_padded = F.pad(
        active_mask,
        (
            0,
            max_num_queries - active_mask.shape[1],
            0,
            max_num_queries - active_mask.shape[0],
        ),
        "constant",
        0,
    ).bool()
    if prior_mask.numel() > 0:
        prior_mask_padded = F.pad(
            prior_mask,
            (
                0,
                context_kv_len - prior_mask.shape[1],
                0,
                max_num_queries - prior_mask.shape[0],
            ),
            "constant",
            0,
        ).bool()
        attn_mask = torch.cat([prior_mask_padded, active_mask_padded], dim=1)
    else:
        attn_mask = active_mask_padded
    return query, k, v, k_cache, v_cache, active_block_table, attn_mask
