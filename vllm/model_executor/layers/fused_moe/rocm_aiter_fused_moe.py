# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

import torch

import vllm.envs as envs
from vllm.platforms import current_platform


def is_rocm_aiter_moe_enabled() -> bool:
    return current_platform.is_rocm() \
        and envs.VLLM_ROCM_USE_AITER_MOE \
        and envs.VLLM_ROCM_USE_AITER \


def is_rocm_aiter_block_scaled_moe_enabled() -> bool:
    return is_rocm_aiter_moe_enabled() and \
        envs.VLLM_ROCM_USE_AITER_FP8_BLOCK_SCALED_MOE


def is_rocm_aiter_channel_scaled_moe_enabled() -> bool:
    return is_rocm_aiter_moe_enabled() and \
        envs.VLLM_ROCM_USE_AITER_FP8_CHANNEL_SCALED_MOE


def asm_moe_tkw1_impl(sorted_ids: torch.Tensor,
                      sorted_weights: torch.Tensor,
                      sorted_expert_ids: torch.Tensor,
                      num_valid_ids: torch.Tensor,
                      moe_buf: torch.Tensor,
                      hidden_states: torch.Tensor,
                      w1: torch.Tensor,
                      w2: torch.Tensor,
                      topk_weight: torch.Tensor,
                      topk_ids: torch.Tensor,
                      fc1_scale: Optional[torch.Tensor] = None,
                      fc2_scale: Optional[torch.Tensor] = None,
                      fc1_smooth_scale: Optional[torch.Tensor] = None,
                      fc2_smooth_scale: Optional[torch.Tensor] = None,
                      activation_str: str = "silu") -> None:
    import aiter as rocm_aiter

    if activation_str == "silu":
        activation = rocm_aiter.ActivationType.Silu
    elif activation_str == "gelu":
        activation = rocm_aiter.ActivationType.Gelu
    else:
        activation = rocm_aiter.ActivationType.Silu

    E, model_dim, _ = w2.shape
    M, topk = topk_ids.shape
    device = topk_ids.device

    a8_type = (w1.dtype if w1.dtype != torch.int32 and w1.dtype != torch.uint32
               else torch.float8_e4m3fnuz)
    a8 = torch.empty((M, model_dim), dtype=a8_type, device=device)
    a8_scale = torch.empty(M, dtype=torch.float, device=device)
    rocm_aiter.dynamic_per_token_scaled_fp8_quant(a8, hidden_states, a8_scale)
    fmoe_func = rocm_aiter.fmoe_g1u1_tkw1
    fmoe_func(moe_buf, a8, w1, w2, sorted_ids, sorted_weights,
              sorted_expert_ids, num_valid_ids, topk, a8_scale, fc1_scale,
              fc2_scale, fc2_smooth_scale, activation)


def rocm_aiter_fused_experts(
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_fp8_w8a8: bool = False,
        apply_router_weight_on_input: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        expert_mask: Optional[torch.Tensor] = None,
        activation: str = "silu",
        **kwagrs  # Ignore additional keyword arguments
) -> torch.Tensor:

    import aiter as rocm_aiter
    import aiter.fused_moe_bf16_asm as rocm_aiter_asm_fmoe

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8)
    
    if apply_router_weight_on_input:
        _, topk = topk_weights.shape
        assert (
            topk == 1
        ), "Only support topk=1 when `apply_router_weight_on_input` is True"

        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

    if is_rocm_aiter_block_scaled_moe_enabled() and use_fp8_w8a8:
        assert not apply_router_weight_on_input, (
            "apply_router_weight_on_input is not supported for block scaled moe"
        )

        assert w1_scale is not None
        assert w2_scale is not None

        local_E = E = w1.shape[0]
        if expert_mask is not None:
            E = expert_mask.numel()

        topk = topk_ids.shape[1]
        model_dim = w1.shape[-1]
        dtype = hidden_states.dtype
        # The default block sizes are 128 in AITER.
        if block_shape is None:
            block_shape = [128, 128]

        scale_blk_k = block_shape[1]

        (
            sorted_token_ids,
            sorted_weight_buf,
            sorted_expert_ids,
            num_valid_ids,
            out_asm,
        ) = rocm_aiter_asm_fmoe.moe_sorting_ck(topk_ids,
                                               topk_weights,
                                               E,
                                               model_dim,
                                               dtype,
                                               expert_mask=expert_mask)

        a1, a1_scale = per_token_group_quant_fp8(hidden_states, scale_blk_k)
        rocm_aiter.fmoe_fp8_blockscale_g1u1(
            out_asm,
            a1,
            w1,
            w2,
            sorted_token_ids,
            sorted_weight_buf,
            sorted_expert_ids,
            num_valid_ids,
            topk,
            w1_scale.view(local_E, -1),
            w2_scale.view(local_E, -1),
            a1_scale.t().contiguous(),
            block_shape[0],
            block_shape[1],
            None,
        )
        return out_asm
    
    elif is_rocm_aiter_channel_scaled_moe_enabled() and use_fp8_w8a8:
        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(torch.int32)

        E, model_dim, _ = w2.shape
        dtype = hidden_states.dtype

        if expert_mask is not None:
            E = expert_mask.numel()

        (
            sorted_token_ids,
            sorted_weight_buf,
            sorted_expert_ids,
            num_valid_ids,
            out_asm,
        ) = rocm_aiter_asm_fmoe.moe_sorting_ck(topk_ids,
                                               topk_weights,
                                               E,
                                               model_dim,
                                               dtype,
                                               expert_mask=expert_mask)
        
        asm_moe_tkw1_impl(
            sorted_ids=sorted_token_ids,
            sorted_weights=sorted_weight_buf,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            moe_buf=out_asm,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            fc1_scale=w1_scale,
            fc2_scale=w2_scale,
            fc1_smooth_scale=None,
            fc2_smooth_scale=None,
            activation_str=activation)

        return out_asm

    elif use_fp8_w8a8:
        assert not apply_router_weight_on_input, (
            "apply_router_weight_on_input is not supported for fp8_w8a8")
        
        return rocm_aiter_asm_fmoe.asm_moe(hidden_states=hidden_states,
                                           w1=w1,
                                           w2=w2,
                                           topk_weight=topk_weights,
                                           topk_ids=topk_ids,
                                           fc1_scale=w1_scale,
                                           fc2_scale=w2_scale,
                                           fc1_smooth_scale=None,
                                           fc2_smooth_scale=None,
                                           a16=False)

    return rocm_aiter.ck_moe(hidden_states=hidden_states,
                             w1=w1,
                             w2=w2,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids)


def rocm_aiter_topk_softmax(topk_weights: torch.Tensor,
                            topk_indices: torch.Tensor,
                            token_expert_indices: torch.Tensor,
                            gating_output: torch.Tensor,
                            renormalize: bool) -> tuple[torch.Tensor, ...]:
    import aiter as rocm_aiter
    rocm_aiter.topk_softmax(topk_weights, topk_indices, token_expert_indices,
                            gating_output, renormalize)

    return topk_weights, topk_indices


def shuffle_weights(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Applies shuffle_weight function from AITER to each 
    input tensor and returns them.

    Args:
    *tensors: Variable number of torch.Tensor objects.

    Returns:
    A tuple of shuffled tensors.
    """
    from aiter.ops.shuffle import shuffle_weight

    return tuple(shuffle_weight(tensor) for tensor in tensors)


def expand_weights(*tensors: torch.Tensor,
                   expansion_dims: list[int]) -> tuple[torch.Tensor, ...]:
    """
    Expands the dimensions of input tensors.

    Args:
        *tensors: A variable number of torch.Tensor objects.
        expansion_dims: A list of expansion dimensions 
        corresponding to each tensor.

    Returns:
        A tuple of tensors with expanded dimensions.
    """

    assert len(tensors) == len(expansion_dims), \
    "Number of tensors must match the number of expansion dimensions."

    return tuple(
        tensor.unsqueeze(-1).unsqueeze(-1).expand((-1, dim, -1))
        for tensor, dim in zip(tensors, expansion_dims))
