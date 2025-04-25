# SPDX-License-Identifier: Apache-2.0
from functools import cache
from typing import List, Optional, Tuple

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


@cache
def is_rocm_aiter_moe_enabled() -> bool:
    return current_platform.is_rocm() \
        and envs.VLLM_ROCM_USE_AITER_MOE \
        and envs.VLLM_ROCM_USE_AITER


def rocm_aiter_asm_moe_tkw1_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weight: torch.Tensor,
        topk_ids: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
        fc1_smooth_scale: Optional[torch.Tensor] = None,
        fc2_smooth_scale: Optional[torch.Tensor] = None,
        a16: bool = False,
        per_tensor_quant_scale: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        activation_str: str = "silu") -> torch.Tensor:

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


def rocm_aiter_asm_moe_tkw1_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weight: torch.Tensor,
        topk_ids: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
        fc1_smooth_scale: Optional[torch.Tensor] = None,
        fc2_smooth_scale: Optional[torch.Tensor] = None,
        a16: bool = False,
        per_tensor_quant_scale: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        activation_str: str = "silu") -> torch.Tensor:
    return torch.empty_like(hidden_states)


def rocm_aiter_ck_moe_impl(hidden_states: torch.Tensor, w1: torch.Tensor,
                           w2: torch.Tensor, topk_weights: torch.Tensor,
                           topk_ids: torch.Tensor) -> torch.Tensor:
    from aiter import ck_moe
    return ck_moe(hidden_states=hidden_states,
                  w1=w1,
                  w2=w2,
                  topk_weights=topk_weights,
                  topk_ids=topk_ids)


def rocm_aiter_ck_moe_fake(hidden_states: torch.Tensor, w1: torch.Tensor,
                           w2: torch.Tensor, topk_weights: torch.Tensor,
                           topk_ids: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def rocm_aiter_fmoe_fp8_blockscale_g1u1_impl(
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        hidden_states_dtype: torch.dtype,
        expert_mask: torch.Tensor,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1_scale: torch.Tensor,
        block_shape: List[int],
        smooth_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    from aiter import fmoe_fp8_blockscale_g1u1
    from aiter.fused_moe_bf16_asm import moe_sorting_ck

    topk = topk_ids.shape[1]
    model_dim = w1.shape[-1]
    local_E = E = w1.shape[0]
    if expert_mask is not None:
        E = expert_mask.numel()

    (
        sorted_token_ids,
        sorted_weight_buf,
        sorted_expert_ids,
        num_valid_ids,
        out_asm,
    ) = moe_sorting_ck(topk_ids,
                       topk_weights,
                       E,
                       model_dim,
                       hidden_states_dtype,
                       expert_mask=expert_mask)

    fmoe_fp8_blockscale_g1u1(out_asm, a1, w1, w2, sorted_token_ids,
                             sorted_weight_buf, sorted_expert_ids,
                             num_valid_ids, topk, w1_scale.view(local_E, -1),
                             w2_scale.view(local_E, -1),
                             a1_scale.t().contiguous(), *block_shape,
                             smooth_scale)

    return out_asm


def rocm_aiter_fmoe_fp8_blockscale_g1u1_fake(
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        hidden_states_dtype: torch.dtype,
        expert_mask: torch.Tensor,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1_scale: torch.Tensor,
        block_shape: List[int],
        smooth_scale: Optional[torch.Tensor] = None) -> torch.Tensor:

    return torch.empty_like(a1, dtype=torch.bf16)


def rocm_aiter_asm_moe_impl(hidden_states: torch.Tensor,
                            w1: torch.Tensor,
                            w2: torch.Tensor,
                            topk_weight: torch.Tensor,
                            topk_ids: torch.Tensor,
                            fc1_scale: Optional[torch.Tensor] = None,
                            fc2_scale: Optional[torch.Tensor] = None,
                            fc1_smooth_scale: Optional[torch.Tensor] = None,
                            fc2_smooth_scale: Optional[torch.Tensor] = None,
                            a16: bool = False,
                            activation: str = "silu") -> torch.Tensor:
    import aiter.fused_moe_bf16_asm as rocm_aiter_asm_fmoe
    from aiter import ActivationType

    assert activation in ["silu", "gelu"], "The given activation:" \
                                          f" {activation}"         \
                                           " is not supported in" \
                                           " AITER."
    if activation == "silu":
        aiter_activation = ActivationType.Silu
    else:
        aiter_activation = ActivationType.Gelu

    return rocm_aiter_asm_fmoe.asm_moe(hidden_states=hidden_states,
                                       w1=w1,
                                       w2=w2,
                                       topk_weight=topk_weight,
                                       topk_ids=topk_ids,
                                       fc1_scale=fc1_scale,
                                       fc2_scale=fc2_scale,
                                       fc1_smooth_scale=fc1_smooth_scale,
                                       fc2_smooth_scale=fc2_smooth_scale,
                                       a16=a16,
                                       activation=aiter_activation)


def rocm_aiter_asm_moe_fake(hidden_states: torch.Tensor,
                            w1: torch.Tensor,
                            w2: torch.Tensor,
                            topk_weight: torch.Tensor,
                            topk_ids: torch.Tensor,
                            fc1_scale: Optional[torch.Tensor] = None,
                            fc2_scale: Optional[torch.Tensor] = None,
                            fc1_smooth_scale: Optional[torch.Tensor] = None,
                            fc2_smooth_scale: Optional[torch.Tensor] = None,
                            a16: bool = False,
                            activation: str = "silu") -> torch.Tensor:
    return torch.empty_like(hidden_states)


def rocm_aiter_topk_softmax_impl(topk_weights: torch.Tensor,
                                 topk_indices: torch.Tensor,
                                 token_expert_indices: torch.Tensor,
                                 gating_output: torch.Tensor,
                                 renormalize: bool) -> None:
    from aiter import topk_softmax
    topk_softmax(topk_weights, topk_indices, token_expert_indices,
                 gating_output, renormalize)


def rocm_aiter_topk_softmax_fake(topk_weights: torch.Tensor,
                                 topk_indices: torch.Tensor,
                                 token_expert_indices: torch.Tensor,
                                 gating_output: torch.Tensor,
                                 renormalize: bool) -> None:
    pass


if current_platform.is_rocm():

    direct_register_custom_op(
        op_name="rocm_aiter_asm_moe_tkw1",
        op_func=rocm_aiter_asm_moe_tkw1_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_asm_moe_tkw1_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_ck_moe",
        op_func=rocm_aiter_ck_moe_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_ck_moe_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_fmoe_fp8_blockscale_g1u1",
        op_func=rocm_aiter_fmoe_fp8_blockscale_g1u1_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_fmoe_fp8_blockscale_g1u1_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_asm_moe",
        op_func=rocm_aiter_asm_moe_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_asm_moe_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_topk_softmax",
        op_func=rocm_aiter_topk_softmax_impl,
        mutates_args=["topk_weights", "topk_indices", "token_expert_indices"],
        fake_impl=rocm_aiter_topk_softmax_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def rocm_aiter_fused_experts(hidden_states: torch.Tensor,
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
                             allow_deep_gemm: bool = False) -> torch.Tensor:

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8)

    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # w8a8 block-scaled
    if block_shape is not None and use_fp8_w8a8:
        assert not apply_router_weight_on_input, (
            "apply_router_weight_on_input is not supported for block scaled moe"
        )
        assert w1_scale is not None
        assert w2_scale is not None

        # The default block sizes are 128 in AITER.
        block_shape = [128, 128] if block_shape is None else block_shape

        a1, a1_scale = per_token_group_quant_fp8(hidden_states, block_shape[1])

        return torch.ops.vllm.rocm_aiter_fmoe_fp8_blockscale_g1u1(
            topk_ids, topk_weights, hidden_states.dtype, expert_map, a1, w1,
            w2, w1_scale, w2_scale, a1_scale, block_shape, None)

    # w8a8 per-channel quantization
    elif per_channel_quant and apply_router_weight_on_input and use_fp8_w8a8:
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`
        # This applies topk_weights on the GEMM output of the first FC layer
        #  rather than the second FC.
        assert (topk_weights.dim() == 2
                ), "`topk_weights` should be in shape (num_tokens, topk)"
        assert topk_weights.shape[-1] == 1, (
            "Only support topk=1 when"
            " `apply_router_weight_on_input` is True")

        return torch.ops.vllm.rocm_aiter_asm_moe_tkw1(
            hidden_states,
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

    # w8a8 per-tensor activation per-tensor weight
    elif use_fp8_w8a8:
        assert not apply_router_weight_on_input, (
            "apply_router_weight_on_input is not supported for fp8_w8a8")
        return torch.ops.vllm.rocm_aiter_asm_moe(hidden_states=hidden_states,
                                                 w1=w1,
                                                 w2=w2,
                                                 topk_weight=topk_weights,
                                                 topk_ids=topk_ids,
                                                 fc1_scale=w1_scale,
                                                 fc2_scale=w2_scale,
                                                 fc1_smooth_scale=None,
                                                 fc2_smooth_scale=None,
                                                 a16=False,
                                                 activation=activation)
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

    # w16a16 fallback to rocm_aiter_ck_moe w16a16
    return torch.ops.vllm.rocm_aiter_ck_moe(hidden_states=hidden_states,
                                            w1=w1,
                                            w2=w2,
                                            topk_weights=topk_weights,
                                            topk_ids=topk_ids)


def rocm_aiter_topk_softmax(topk_weights: torch.Tensor,
                            topk_indices: torch.Tensor,
                            token_expert_indices: torch.Tensor,
                            gating_output: torch.Tensor,
                            renormalize: bool) -> Tuple[torch.Tensor, ...]:
    torch.ops.vllm.rocm_aiter_topk_softmax(topk_weights, topk_indices,
                                           token_expert_indices, gating_output,
                                           renormalize)
    return topk_weights, topk_indices


def shuffle_weights(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Applies shuffle_weight function from AITER to each 
    input tensor and returns them.

    Args:
    *tensors: Variable number of torch.Tensor objects.

    Returns:
    A Tuple of shuffled tensors.
    """
    from aiter.ops.shuffle import shuffle_weight
    return tuple(shuffle_weight(tensor) for tensor in tensors)


def expand_weights(*tensors: torch.Tensor,
                   expansion_dims: list[int]) -> Tuple[torch.Tensor, ...]:
    """
    Expands the dimensions of input tensors.

    Args:
        *tensors: A variable number of torch.Tensor objects.
        expansion_dims: A list of expansion dimensions 
        corresponding to each tensor.

    Returns:
        A Tuple of tensors with expanded dimensions.
    """

    assert len(tensors) == len(expansion_dims), \
    "Number of tensors must match the number of expansion dimensions."

    return tuple(
        tensor.unsqueeze(-1).unsqueeze(-1).expand((-1, dim, -1))
        for tensor, dim in zip(tensors, expansion_dims))
