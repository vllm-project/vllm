###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
from typing import Optional

import habana_frameworks.torch as htorch
import torch
import torch.nn.functional as F

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
    logger.warning("Could not import HPU FusedSDPA kernel. "
                   "vLLM will use native implementation.")


def batch2block(tensor, block_mapping):
    shape = tuple(tensor.shape)
    return (block_mapping @ tensor.view(shape[0], -1)).view(-1, *shape[1:])


def block2batch(tensor, block_mapping):
    shape = tuple(tensor.shape)
    return (block_mapping.t() @ tensor.view(shape[0], -1)).view(-1, *shape[1:])


def block_softmax(batch_size, attn, block_mapping):
    attn.sub_(10.0)
    attn = attn.exp_()
    sums = attn.sum(dim=-1).unsqueeze(-1)
    sums = block2batch(sums, block_mapping)
    sums = batch2block(sums, block_mapping)
    sums.add_(1.0e-12)
    attn.div_(sums)
    return attn


def flat_pa(query, key_cache, value_cache, block_list, block_mapping,
            block_bias, scale, matmul_qk_op, matmul_av_op, keys_fetch_func,
            values_fetch_func):
    batch_size = query.size(0)
    q_heads = query.size(1)
    kv_heads = key_cache.size(2)

    query = batch2block(scale * query, block_mapping).unsqueeze(-2)
    key = keys_fetch_func(key_cache, block_list).transpose(1, 2)
    value = values_fetch_func(value_cache, block_list).transpose(1, 2)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)

    if kv_heads != q_heads:
        block_bias = block_bias.unsqueeze(1)
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        key = key.transpose(3, 4)
    else:
        key = key.transpose(2, 3)

    attn = matmul_qk_op(query, key) + block_bias
    attn = block_softmax(batch_size, attn, block_mapping)
    attn = matmul_av_op(attn, value)
    attn = block2batch(attn, block_mapping)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn


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

    return final_hidden_states.view(-1, D)


#TODO: remove after fusedsdpa fix for query_head != kv_head
def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The kv go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = kv.shape
    if n_rep == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen,
                                     head_dim)
    return kv.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
    if attn_bias is not None or HPUFusedSDPA is None:
        if query_heads != kv_heads:
            query = query.unflatten(1, (kv_heads, -1))
            key = key.unflatten(1, (kv_heads, 1))
            value = value.unflatten(1, (kv_heads, 1))
            if attn_bias is not None:
                attn_bias = attn_bias.unsqueeze(2)
        attn_weights = matmul_qk_op(query * scale, key.transpose(-1, -2))
        if attn_bias is not None:
            attn_weights.add_(attn_bias)
        attn_weights = softmax_op(attn_weights, dim=-1)
        attn_weights = matmul_av_op(attn_weights, value)
        if query_heads != kv_heads:
            attn_weights = attn_weights.flatten(1, 2)
    else:
        #TODO: remove after fusedsdpa fix for query_heads != kv_heads
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


class LoraMask:
    lora_mask = None

    @staticmethod
    def setLoraMask(mask):
        LoraMask.lora_mask = mask

    @staticmethod
    def getLoraMask():
        return LoraMask.lora_mask


def dispatch_bgmv_linear(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    indices: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    """
    `wa_t_all` and `wb_t_all` contains all LoRA A and LoRA B weight matrices
    stacked at dimension 0 into single tensors, assuming same rank. `wa` is the
    reshaped and transposed version of `wa_t_all` of shape
    (h_in, max_loras * lora_rank) and `wb` is the transposed and reshaped
    version of `wb_t_all` of shape (max_loras * lora_rank, h_out).

    Matmul input `x` with `wa`. Multiply `x` with a mask to zero-out inputs of
    inactive LoRA indices. Matmul masked output with `wb` and scale it to get
    the final output.
    """

    assert layer_idx == 0, f'layer_idx should be 0, but got {layer_idx}'
    mask = LoraMask.getLoraMask()

    wa = wa_t_all[:, 0, :, :]
    wb = wb_t_all[:, 0, :, :].transpose(1, 2)
    wa = wa.reshape(wa.shape[0] * wa.shape[1], wa.shape[2]).transpose(0, 1)
    wb = wb.reshape(wb.shape[0] * wb.shape[1], wb.shape[2])

    out = x @ wa
    assert (out.shape == mask.shape)
    out = out * mask
    out = out @ wb
    y += out * scale


def dispatch_bgmv_embedding(
    y: torch.Tensor,
    x: torch.Tensor,
    wb_t_all: torch.Tensor,
    indices: torch.LongTensor,
    layer_idx: int,
    scale: float,
):
    """
    `wb_t_all` contains all LoRA-B weight matrices stacked at dimension 0 into
    a single tensor, assuming same rank. `wb` is the transposed and reshaped
    version of `wb_t_all` of shape (num_loras * lora_rank, embedding_dim).

    Output of LoRA-A embedding (tensor x) is repeated max_loras times to match
    the shape of `wb`. Multiply `x` with a mask to zero-out inputs of inactive
    LoRA indices. Matmul masked output with `wb` and scale it to get the final
    output.
    """

    assert layer_idx == 0, f'layer_idx should be 0, but got {layer_idx}'
    max_loras = wb_t_all.size(0)

    x = x.repeat(1, max_loras)
    x = x * LoraMask.getLoraMask()
    wb = wb_t_all[:, 0, :, :].transpose(1, 2)
    wb = wb.reshape(wb.shape[0] * wb.shape[1], wb.shape[2])
    out = x @ wb
    y += out * scale
