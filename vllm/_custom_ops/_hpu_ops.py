###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
import os
from typing import Optional, Tuple

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
                       qk_matmul_op=torch.matmul,
                       softmax_op=torch.softmax,
                       av_matmul_op=torch.matmul,
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
    fetch_keys = fetch_from_cache if k_cache_cls is None else k_cache_cls.fetch_from_cache
    keys = fetch_keys(key_cache, block_tables, (0, 2, 3, 1))
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        keys = [k.unflatten(1, (kv_heads, 1)) for k in keys]
        mask = mask.unsqueeze(2)

    attn_weights = [qk_matmul_op(query, k) for k in keys]
    attn_weights = torch.cat(attn_weights, dim=-1)
    if alibi_slopes is not None:
        attn_weights.add_(alibi_slopes[:, :, -attn_weights.size(2):,
                                       -attn_weights.size(3):])
    attn_weights = softmax_op(attn_weights.masked_fill(mask, min_inf), dim=-1)

    fetch_values = fetch_from_cache if v_cache_cls is None else k_cache_cls.fetch_from_cache
    values = fetch_values(value_cache, block_tables, (0, 2, 1, 3))
    if PA_SPLIT_VALUE:
        attn_weights = attn_weights.split(block_size, dim=-1)
    else:
        values = [torch.cat(values, dim=-2)]
        attn_weights = [attn_weights]
    if query_heads != kv_heads:
        values = [v.unflatten(1, (kv_heads, 1)) for v in values]
    attn_weights = [av_matmul_op(a, v) for a, v in zip(attn_weights, values)]
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


def prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    qk_matmul_op = torch.matmul,
    softmax_op = torch.softmax,
    av_matmul_op = torch.matmul,
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
    attn_weights = qk_matmul_op(query * scale, key.transpose(-1, -2))
    if attn_bias is not None:
        attn_weights.add_(attn_bias)
    attn_weights = softmax_op(attn_weights, dim=-1)
    attn_weights = av_matmul_op(attn_weights, value)
    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights




def reshape_and_cache(key,
                      value,
                      key_cache,
                      value_cache,
                      slot_mapping,
                      dtype,
                      is_prompt=False):
    num_blocks = key_cache.size(0)
    block_size = key_cache.size(1)
    slot_mapping = slot_mapping.flatten()
    indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offsets = torch.fmod(slot_mapping, block_size)
    num_slots_requested = slot_mapping.size(0)
    num_slots_available = num_blocks * block_size
    # NOTE(kzawora): HPU PT bridge crashes with
    # RuntimeError: Invalid inputs for scatter_nd_onnx
    # on index_put when num_slots_requested > num_slots_available.
    # This case might occur when we have little kv cache blocks and
    # lots of padding, or are doing warmup.
    # This loop is a workaround for this issue. Please remove it
    # once key_cache.index_put_(indices, offsets), key) works.
    num_kv_cache_passes = torch.div(num_slots_requested,
                                    num_slots_available).ceil().int().item()
    for i in range(num_kv_cache_passes):
        start_idx = i * num_slots_available
        end_idx = (i + 1) * num_slots_available
        key_cache.index_put_(
            (indices[start_idx:end_idx], offsets[start_idx:end_idx]),
            key[start_idx:end_idx])
        value_cache.index_put_(
            (indices[start_idx:end_idx], offsets[start_idx:end_idx]),
            value[start_idx:end_idx])


def prepare_to_cache(cache, slot_mapping):
    num_blocks = cache.size(0)
    block_size = cache.size(1)
    slot_mapping = slot_mapping.flatten()
    indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offsets = torch.fmod(slot_mapping, block_size)
    num_slots_requested = slot_mapping.size(0)
    num_slots_available = num_blocks * block_size
    # NOTE(kzawora): HPU PT bridge crashes with
    # RuntimeError: Invalid inputs for scatter_nd_onnx
    # on index_put when num_slots_requested > num_slots_available.
    # This case might occur when we have little kv cache blocks and
    # lots of padding, or are doing warmup.
    # This loop is a workaround for this issue. Please remove it
    # once key_cache.index_put_(indices, offsets), key) works.
    num_kv_cache_passes = torch.div(num_slots_requested,
                                    num_slots_available).ceil().int().item()

    return num_kv_cache_passes, num_slots_available, indices, offsets


def insert_or_update_cache(input, cache, num_kv_cache_passes, num_slots_available, block_indices, block_offsets):
    for i in range(num_kv_cache_passes):
        start_idx = i * num_slots_available
        end_idx = (i + 1) * num_slots_available
        cache.index_put_(
            (block_indices[start_idx:end_idx], block_offsets[start_idx:end_idx]),
            input[start_idx:end_idx])


def swap_blocks(src, dst, block_mapping):
    index_src = torch.zeros((1, ), dtype=torch.int32, device=src.device)
    index_dst = torch.zeros((1, ), dtype=torch.int32, device=dst.device)
    for src_idx, dst_idx in block_mapping.items():
        index_src[0] = src_idx
        index_dst[0] = dst_idx
        dst.index_put_([index_dst], src.index_select(0, index_src))
    if dst.device.type == 'hpu':
        htorch.core.mark_step()
        torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, block_mapping):
    index_src = torch.zeros((1, ),
                            dtype=torch.int32,
                            device=key_caches[0].device)
    index_dst = torch.zeros((1, ),
                            dtype=torch.int32,
                            device=key_caches[0].device)
    for src, dsts in block_mapping.items():
        index_src[0] = src
        for dst in dsts:
            index_dst[0] = dst
            for key_cache in key_caches:
                key_cache.index_copy_(0, index_dst,
                                      key_cache.index_select(0, index_src))
            for value_cache in value_caches:
                value_cache.index_copy_(0, index_dst,
                                        value_cache.index_select(0, index_src))
        if key_caches[0].device.type == 'hpu':
            htorch.core.mark_step()


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    batch_dim_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensor for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic 
            per token case
        batch_dim_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token 
            in the dynamic quantization case.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    if batch_dim_padding:
        shape = (max(batch_dim_padding, input.shape[0]), *input.shape[1:])
        output = torch.empty(shape,
                             device=input.device,
                             dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    if scale is None:
        raise "dynamic scaled_fp8_quant not implemented for HPU"
        #TODO: calculate scale to match gaudi2 240 range instead of 448
        if use_per_token_if_dynamic:
            scale = torch.empty((input.numel() // input.shape[-1], 1),
                                device=input.device,
                                dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        output = torch.ops.hpu.cast_to_fp8_v2(input, 1/scale, False, False, dtype=torch.float8_e4m3fn)[0]

    return output, scale
