###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import habana_frameworks.torch as htorch
from typing import List, Optional, Tuple

import vllm.hpu.utils as hpu_utils

PA_SPLIT_VALUE = (os.environ.get('PA_SPLIT_VALUE', '0') == '1')


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
    return [cache.index_select(0, blocks[:, i]) for i in range(blocks.size(1))]


@hpu_utils.with_mark_steps
def paged_attention_v1(query, key_cache, value_cache, head_mapping, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes, kv_cache_dtype=None)  -> None:
    seq_len = block_tables.size(1)
    batch_size, query_heads, _ = query.shape
    _, kv_heads, _, _ = key_cache.shape
    min_inf = torch.finfo(query.dtype).min
    mask = (torch.arange(0, seq_len * block_size, dtype=torch.int32, device=key_cache.device)
            .view(1, -1)
            .expand(batch_size, -1)
            .ge(context_lens.view(-1, 1))
            .view(batch_size, 1, 1, -1))
    query = query.unsqueeze(-2)
    keys = fetch_from_cache(key_cache, block_tables)
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        keys = [k.unflatten(1, (kv_heads, 1)) for k in keys]
        mask = mask.unsqueeze(2)

    attn_weights = [torch.matmul(query, k) for k in keys]
    attn_weights = (torch.cat(attn_weights, dim=-1)
                    .mul_(scale)
                    .masked_fill(mask, min_inf)
                    .softmax(dim=-1))

    values = fetch_from_cache(value_cache, block_tables)
    if PA_SPLIT_VALUE:
        attn_weights = attn_weights.split(block_size, dim=-1)
    else:
        values = [torch.cat(values, dim=-1)]
        attn_weights = [attn_weights]
    if query_heads != kv_heads:
        values = [v.unflatten(1, (kv_heads, 1)) for v in values]
    attn_weights = [torch.matmul(a, v.transpose(-1, -2)).squeeze(-2) for a, v in zip(attn_weights, values)]
    if query_heads != kv_heads:
        attn_weights = [a.flatten(1, 2) for a in attn_weights]
    attn_weights = sum(attn_weights)

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


def awq_gemm(*args):
    raise NotImplementedError
