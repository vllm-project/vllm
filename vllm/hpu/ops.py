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

from vllm.logger import init_logger

logger = init_logger(__name__)
HPUFusedRMSNorm = None
try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
    HPUFusedRMSNorm = FusedRMSNorm
except ImportError:
    logger.warning("Could not import HPU FusedRMSNorm kernel. "
                   "vLLM will use forward_native implementation of RMSNorm.")
HPUFusedSDPA = None   
try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    HPUFusedSDPA = FusedSDPA
except ImportError:
    logger.warning("Could not import HPU FusedSDPA kernel. vLLM will use native implementation.")

PA_SPLIT_VALUE = (os.environ.get('PA_SPLIT_VALUE', '1') == '1')


def fetch_from_cache(cache, blocks, permutations):
    return [
        cache.index_select(0, blocks[:, i]).permute(permutations)
        for i in range(blocks.size(1))
    ]


def paged_attention_v1(query,
                       key_cache,
                       value_cache,
                       head_mapping,
                       scale,
                       block_tables,
                       context_lens,
                       block_size,
                       alibi_slopes=None,
                       kv_cache_dtype=None,
                       matmul_qk_op=torch.matmul,
                       softmax_op=torch.softmax,
                       matmul_av_op=torch.matmul,
                       k_cache_cls=None,
                       v_cache_cls=None) -> None:
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
    fetch_keys = fetch_from_cache if k_cache_cls is None else \
                 k_cache_cls.fetch_from_cache
    keys = fetch_keys(key_cache, block_tables, (0, 2, 3, 1))
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        keys = [k.unflatten(1, (kv_heads, 1)) for k in keys]
        mask = mask.unsqueeze(2)

    attn_weights = torch.cat([matmul_qk_op(query, k) for k in keys], dim=-1)
    if alibi_slopes is not None:
        attn_weights.add_(alibi_slopes[:, :, -attn_weights.size(2):,
                                       -attn_weights.size(3):])
    attn_weights = softmax_op(attn_weights.masked_fill(mask, min_inf), dim=-1)

    fetch_values = fetch_from_cache if v_cache_cls is None else \
                   v_cache_cls.fetch_from_cache
    values = fetch_values(value_cache, block_tables, (0, 2, 1, 3))
    if PA_SPLIT_VALUE:
        attn_weights = attn_weights.split(block_size, dim=-1)
    else:
        values = [torch.cat(values, dim=-2)]
        attn_weights = [attn_weights]
    if query_heads != kv_heads:
        values = [v.unflatten(1, (kv_heads, 1)) for v in values]
    attn_weights = [matmul_av_op(a, v) for a, v in zip(attn_weights, values)]
    if query_heads != kv_heads:
        attn_weights = [a.flatten(1, 2) for a in attn_weights]
    attn_weights = sum(attn_weights)
    return attn_weights.squeeze(-2)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


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
        w_output = torch.matmul(hidden_states, w1[expert_idx].transpose(0, 1))
        w_output = silu_and_mul(w_output)
        w_output = torch.matmul(w_output, w2[expert_idx].transpose(0, 1))
        final_hidden_states += w_output * padded_weights[expert_idx]
        htorch.core.mark_step()

    return final_hidden_states.view(-1, D)

#TODO: remove after SW-195415 fix
def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = kv.shape
    if n_rep == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen,
                                     head_dim)
    return kv.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@hpu_utils.with_mark_steps
def prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    matmul_qk_op=torch.matmul,
    softmax_op=torch.softmax,
    matmul_av_op=torch.matmul,
    valid_seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if attn_bias is not None or HPUFusedSDPA:
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
    else:
        #TODO: remove after SW-195415 fix
        if query_heads != kv_heads:
            key = repeat_kv(key, int(query_heads // kv_heads))
            value = repeat_kv(value, int(query_heads // kv_heads))
        softmax_mode = 'fast'
        recompute_mode = True
        attn_weights = FusedSDPA.apply(query, key, value, None, 0.0, True,
                                       scale, softmax_mode, recompute_mode,
                                       valid_seq_lengths, 'right')
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights
