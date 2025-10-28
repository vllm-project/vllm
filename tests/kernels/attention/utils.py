# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None = None,
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
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: torch.Tensor | None,
    attn_masks: list[torch.Tensor | None] | None = None,
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    assert attn_masks is None or len(attn_masks) == num_seqs

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: list[torch.Tensor] = []
        values_lst: list[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        seq_attn_mask = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)
            seq_attn_mask = alibi_bias

        if attn_masks is not None:
            if seq_attn_mask is not None:
                if attn_masks[i] is not None:
                    seq_attn_mask += attn_masks[i]  # add mask to alibi bias
            else:
                seq_attn_mask = attn_masks[i]

        out = ref_masked_attention(q, keys, values, scale, seq_attn_mask)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


def ref_multi_query_kv_attention(
    cu_seq_lens: list[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    alibi_bias: list[torch.Tensor] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_seqs = len(cu_seq_lens) - 1
    ref_outputs: list[torch.Tensor] = []
    if alibi_bias:
        assert len(alibi_bias) == num_seqs
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask. ALiBi already includes a tril causal mask.
        if alibi_bias:
            attn_mask = alibi_bias[i]
        else:
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=dtype), diagonal=1
            )
            attn_mask = attn_mask * torch.finfo(dtype).min
            attn_mask = attn_mask.to(dtype=dtype)

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)

    return torch.cat(ref_outputs, dim=0)
