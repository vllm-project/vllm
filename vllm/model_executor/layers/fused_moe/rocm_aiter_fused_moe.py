# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

import torch

import vllm.envs as envs
from vllm.platforms import current_platform


def is_rocm_aiter_moe_enabled() -> bool:
    return current_platform.is_rocm() \
        and envs.VLLM_ROCM_USE_AITER_MOE \
        and envs.VLLM_ROCM_USE_AITER


def rocm_aiter_asm_moe_tkw1(hidden_states,
                            w1,
                            w2,
                            topk_weight,
                            topk_ids,
                            fc1_scale=None,
                            fc2_scale=None,
                            fc1_smooth_scale=None,
                            fc2_smooth_scale=None,
                            a16=False,
                            per_tensor_quant_scale=None,
                            expert_mask=None,
                            activation_str: str = "silu") -> None:

    from aiter import ActivationType
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1

    activation = \
        ActivationType.Gelu if activation_str == "gelu" else ActivationType.Silu

    return asm_moe_tkw1(hidden_states,
                        w1,
                        w2,
                        topk_weight,
                        topk_ids,
                        fc1_scale=fc1_scale,
                        fc2_scale=fc2_scale,
                        fc1_smooth_scale=fc1_smooth_scale,
                        fc2_smooth_scale=fc2_smooth_scale,
                        a16=a16,
                        per_tensor_quant_scale=per_tensor_quant_scale,
                        expert_mask=expert_mask,
                        activation=activation)


def rocm_aiter_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    allow_deep_gemm: bool = False,
) -> torch.Tensor:

    import aiter as rocm_aiter
    import aiter.fused_moe_bf16_asm as rocm_aiter_asm_fmoe

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8)

    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    if (block_shape is not None) and use_fp8_w8a8:
        assert not apply_router_weight_on_input, (
            "apply_router_weight_on_input is not supported for block scaled moe"
        )

        assert w1_scale is not None
        assert w2_scale is not None

        local_E = E = w1.shape[0]
        if expert_map is not None:
            E = expert_map.numel()

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
                                               expert_mask=expert_map)

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

    elif per_channel_quant and apply_router_weight_on_input and use_fp8_w8a8:
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`
        # This applies topk_weights on the GEMM output of the first FC layer
        #  rather than the second FC.
        assert (topk_weights.dim() == 2
                ), "`topk_weights` should be in shape (num_tokens, topk)"
        assert topk_weights.shape[-1] == 1, (
            "Only support topk=1 when"
            " `apply_router_weight_on_input` is True")

        return rocm_aiter_asm_moe_tkw1(hidden_states,
                                       w1,
                                       w2,
                                       topk_weights,
                                       topk_ids,
                                       fc1_scale=w1_scale,
                                       fc2_scale=w2_scale,
                                       fc1_smooth_scale=None,
                                       fc2_smooth_scale=None,
                                       a16=False,
                                       per_tensor_quant_scale=None,
                                       expert_mask=expert_map,
                                       activation_str=activation)

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

    if apply_router_weight_on_input:
        assert (topk_weights.dim() == 2
                ), "`topk_weights` should be in shape (num_tokens, topk)"
        _, topk = topk_weights.shape
        assert (
            topk == 1
        ), "Only support topk=1 when `apply_router_weight_on_input` is True"

        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

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
