###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
import os
from typing import Optional

import habana_frameworks.torch as htorch
import torch
import torch.nn.functional as F

import vllm.hpu.utils as hpu_utils

PA_SPLIT_VALUE = (os.environ.get('PA_SPLIT_VALUE', '1') == '1')


def silu_and_mul(output, input):
    d = input.shape[-1] // 2
    silu = torch.nn.SiLU().to(input.device)
    x, y = torch.split(input, d, dim=-1)
    output.copy_(silu(x) * y)


def fetch_from_cache(cache, blocks, permutations):
    return [
        cache.index_select(0, blocks[:, i]).permute(permutations)
        for i in range(blocks.size(1))
    ]


@hpu_utils.with_mark_steps
def paged_attention_v1(query,
                       key_cache,
                       value_cache,
                       head_mapping,
                       scale,
                       block_tables,
                       context_lens,
                       block_size,
                       alibi_slopes=None,
                       kv_cache_dtype=None) -> None:
    seq_len = block_tables.size(1)
    batch_size, query_heads, _ = query.shape
    _, _, kv_heads, _ = key_cache.shape
    min_inf = torch.finfo(query.dtype).min
    mask = (torch.arange(0,
                         seq_len * block_size,
                         dtype=torch.int32,
                         device=key_cache.device).view(1, -1).expand(
                             batch_size, -1).ge(context_lens.view(-1, 1)).view(
                                 batch_size, 1, 1, -1))
    query.mul_(scale)
    query = query.unsqueeze(-2)
    keys = fetch_from_cache(key_cache, block_tables, (0, 2, 3, 1))
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        keys = [k.unflatten(1, (kv_heads, 1)) for k in keys]
        mask = mask.unsqueeze(2)

    attn_weights = [torch.matmul(query, k) for k in keys]
    attn_weights = torch.cat(attn_weights, dim=-1)
    if alibi_slopes is not None:
        attn_weights.add_(alibi_slopes[:, :, -attn_weights.size(2):,
                                       -attn_weights.size(3):])
    attn_weights = (attn_weights.masked_fill(mask, min_inf).softmax(dim=-1))

    values = fetch_from_cache(value_cache, block_tables, (0, 2, 1, 3))
    if PA_SPLIT_VALUE:
        attn_weights = attn_weights.split(block_size, dim=-1)
    else:
        values = [torch.cat(values, dim=-2)]
        attn_weights = [attn_weights]
    if query_heads != kv_heads:
        values = [v.unflatten(1, (kv_heads, 1)) for v in values]
    attn_weights = [torch.matmul(a, v) for a, v in zip(attn_weights, values)]
    if query_heads != kv_heads:
        attn_weights = [a.flatten(1, 2) for a in attn_weights]
    attn_weights = sum(attn_weights)
    return attn_weights.squeeze(-2)


def silu_and_mul_wrapper(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    silu_and_mul(out, x)
    return out


def static_fused_moe(hidden_states, w1, w2, score, topk):
    B, D = hidden_states.shape
    num_experts = w1.shape[0]
    routing_weights = F.softmax(score, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights,
                                                   topk,
                                                   dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((1, B, D),
                                      dtype=hidden_states.dtype,
                                      device=hidden_states.device)
    padded_weights = torch.zeros((B, num_experts),
                                 dtype=hidden_states.dtype,
                                 device=hidden_states.device)
    padded_weights.scatter_(-1, selected_experts, routing_weights)
    padded_weights = padded_weights.reshape(-1, B, w1.shape[0])
    padded_weights = padded_weights.permute(2, 0, 1).unsqueeze(-1)

    htorch.core.mark_step()

    for expert_idx in range(num_experts):
        padded_weight = padded_weights[expert_idx]
        current_state_static = hidden_states.reshape(-1, D)
        w_output = silu_and_mul_wrapper(
            torch.matmul(current_state_static, w1[expert_idx].transpose(0, 1)))
        w_output = torch.matmul(w_output, w2[expert_idx].transpose(0, 1))
        current_hidden_states_static = w_output * padded_weight
        final_hidden_states += current_hidden_states_static
        htorch.core.mark_step()

    return final_hidden_states.view(-1, D)


@hpu_utils.with_mark_steps
def prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        attn_bias = attn_bias.unsqueeze(2)
    attn_weights = torch.matmul(query * scale, key.transpose(-1, -2))
    if attn_bias is not None:
        attn_weights.add_(attn_bias)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = torch.matmul(attn_weights, value)
    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights
