###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import habana_frameworks.torch as htorch
from typing import List, Optional, Tuple

def silu_and_mul(output, input):
    htorch.core.mark_step()
    d = input.shape[-1] // 2
    silu = torch.nn.SiLU().to(input.device)
    x, y = torch.split(input, d, dim=-1)
    output.copy_(silu(x) * y)
    htorch.core.mark_step()

def gelu_new(output, input):
    raise NotImplementedError

def gelu_fast(output, input):
    raise NotImplementedError

def paged_attention_v1(query_in, key_cache_in, value_cache_in, head_mapping, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, attn_masks=None)  -> None:
    query = query_in.bfloat16()
    key_cache = key_cache_in.bfloat16()
    value_cache = value_cache_in.bfloat16()
    num_kv_heads = value_cache[0].shape[0]
    head_size = value_cache[0].shape[1]
    block_size = value_cache[0].shape[2]
    num_seqs = query.shape[0]
    num_query_heads = query.shape[1]
    max_num_blocks_per_seq = block_tables.shape[1]

    if alibi_slopes or num_query_heads != num_kv_heads: #or attn_masks is None:
        raise NotImplementedError

    attn_weights_blocks = []
    value_blocks = []
    seq_index = torch.tensor([0], dtype=torch.int64, device="hpu")

    for i in range(0, max_num_blocks_per_seq):
        # FIXME: dynamic hard override for filler. These blocks would contribute nothing to the output due to zero attention_probs and
        #  will clog up compute resources. The override itself makes the code unsuitable for graph precompilation
        if (i - 2) * block_size > torch.max(context_lens):
            break
        attn_weights = torch.full((num_seqs, num_query_heads, 1, block_size), torch.finfo(query.dtype).min, dtype=query.dtype, device="hpu")
        values = torch.zeros((num_seqs, num_query_heads, head_size, block_size), dtype=query.dtype, device="hpu")
        for seq_id in range(num_seqs):
            seq_index.fill_(seq_id)
            if i * block_size < context_lens[seq_id]:

                q =  torch.index_select(query, 0, seq_index).transpose(0, 1)
                key = torch.index_select(key_cache, 0, block_tables[seq_id][i]).squeeze(0)
                attn_weight = scale * torch.matmul(q, key)

                if attn_masks is not None:
                    attn_mask = torch.index_select(attn_masks[i], 0, seq_index)
                    attn_weight = torch.masked_fill(attn_weight, ~(attn_mask.unsqueeze(0).to(torch.bool)), torch.finfo(attn_weight.dtype).min)

                # FIXME: these dynamic checks serve to ensure the -inf default value is not overwritten with fillers that would cause errors
                #  in logsoftmax computation. A change to custom block multiplication code is required to avoid incurring extra costs here
                if context_lens[seq_id] < (i + 1) * block_size:
                    if context_lens[seq_id] - i*block_size < 0:
                        attn_weight = torch.finfo(query.dtype).min
                    else:
                        attn_weight[:, :, context_lens[seq_id] - i*block_size:] = torch.finfo(query.dtype).min
                attn_weights.index_copy_(0, seq_index, attn_weight.unsqueeze(0))
            value = torch.index_select(value_cache, 0, block_tables[seq_id][i])
            # FIXME: these checks concern filler values in the V cache and should be removed once the underlying issue is addressed
            value = torch.nan_to_num(value)
            value[value < -1.0e+30] = 0.0
            values.index_copy_(0, seq_index, value)
            torch.hpu.synchronize()

        attn_weights_blocks.append(attn_weights.reshape(num_seqs * num_query_heads, 1, block_size))
        value_blocks.append(values.reshape(num_seqs * num_query_heads, head_size, block_size).transpose(1, 2))

    exp_sum = torch.zeros((*attn_weights_blocks[0].shape[:2], 1), dtype=attn_weights_blocks[0].dtype, device="hpu")
    for x in attn_weights_blocks:
        exp_sum.add_(torch.exp(x).sum(dim=-1, keepdim=True))

    output = torch.zeros_like(query)
    for i in range(len(attn_weights_blocks)):
        attention_probs = torch.exp(attn_weights_blocks[i]) / exp_sum
        value = value_blocks[i]
        out = torch.matmul(attention_probs.to(value.dtype), value).reshape(num_seqs, num_query_heads, head_size)
        output.add_(out)
    htorch.core.mark_step()
    return output.to(dtype=query_in.dtype)

def rms_norm(out, hidden_states, weight, eps):
    htorch.core.mark_step()
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    out.copy_(weight * hidden_states.to(input_dtype))
    htorch.core.mark_step()

def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)
    return q_embed, k_embed


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style):
    # FIXME: the below code is unused legacy code not meant to be used. Use FusedRoPE
    #  on HPU and delete this once coverage is verified
    raise NotImplementedError

def awq_gemm(*args):
    raise NotImplementedError
