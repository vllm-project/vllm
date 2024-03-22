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

import vllm.hpu.utils as hpu_utils


def silu_and_mul(output, input):
    d = input.shape[-1] // 2
    silu = torch.nn.SiLU().to(input.device)
    x, y = torch.split(input, d, dim=-1)
    output.copy_(silu(x) * y)


def gelu_new(output, input):
    raise NotImplementedError


def gelu_fast(output, input):
    raise NotImplementedError


def fetch_from_cache(cache, blocks):
    _, seq_len = blocks.shape
    return torch.cat([cache.index_select(0, blocks[:, i]) for i in range(seq_len)], dim=-1)


def scaled_dot_product_attention(query, key, value, scale, mask):
    bs = query.size(0)
    min_inf = torch.finfo(query.dtype).min
    value.masked_fill_(mask, min_inf)
    return (torch.matmul(query, key)
            .mul_(scale)
            .masked_fill_(mask, min_inf)
            .softmax(dim=-1)
            .matmul(value.transpose(-1, -2)))


@hpu_utils.with_mark_steps
def paged_attention_v1(query, key_cache, value_cache, head_mapping, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, attn_masks=None)  -> None:
    if alibi_slopes is not None:
        raise NotImplementedError
    if attn_masks is not None:
        raise NotImplementedError

    key = fetch_from_cache(key_cache, block_tables)
    value = fetch_from_cache(value_cache, block_tables)
    query = query.unsqueeze(-2)
    batch_size, query_heads, head_dim, _ = query.shape
    _, kv_heads, _, seq_len = key.shape

    mask = (torch.arange(0, seq_len, dtype=torch.int32, device=key.device)
            .view(1, -1)
            .expand(batch_size, seq_len)
            .ge(context_lens.view(-1, 1))
            .view(batch_size, 1, 1, -1))

    if query_heads != kv_heads:
        attn_weights = scaled_dot_product_attention(
            query.unflatten(1, (kv_heads, -1)),
            key.unsqueeze(2),
            value.unsqueeze(2),
            scale,
            mask.unsqueeze(2))
        attn_weights = attn_weights.flatten(1, 2)
    else:
        attn_weights = scaled_dot_product_attention(query, key, value, scale, mask)
    attn_weights = attn_weights.squeeze(-2)

    return attn_weights


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
