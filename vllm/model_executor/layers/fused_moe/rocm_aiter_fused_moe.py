# SPDX-License-Identifier: Apache-2.0
from enum import IntEnum
from functools import cache
from typing import Optional

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


class QuantMethod(IntEnum):
    # This allows interfacing with AITER QuantType Enum
    # without importing the QuantType from AITER globally.

    # Note that these quantization methods are
    # supported in AITER package. However,
    # not all are used in this module.

    NO = 0  # a16w16
    PER_TENSOR = 1  # w8a8 (pre_Tensor)
    PER_TOKEN = 2  # w8a8/w8a4 (per_Token)
    BLOCK_1X128 = 3  # block quantized w8a8 (per_1x128)
    BLOCK_128x128 = 4  # block quantized w8a8 (per_128x128)


class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1


@cache
def is_rocm_aiter_moe_enabled() -> bool:
    return current_platform.is_rocm() \
        and envs.VLLM_ROCM_USE_AITER_MOE \
        and envs.VLLM_ROCM_USE_AITER


def rocm_aiter_asm_moe_tkw1_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
        fc1_smooth_scale: Optional[torch.Tensor] = None,
        fc2_smooth_scale: Optional[torch.Tensor] = None,
        a16: bool = False,
        per_tensor_quant_scale: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        activation_method: int = ActivationMethod.SILU.value) -> torch.Tensor:

    from aiter import ActivationType
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1

    activation = ActivationType(activation_method)

    return asm_moe_tkw1(hidden_states,
                        w1,
                        w2,
                        topk_weights,
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
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
        fc1_smooth_scale: Optional[torch.Tensor] = None,
        fc2_smooth_scale: Optional[torch.Tensor] = None,
        a16: bool = False,
        per_tensor_quant_scale: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        activation_method: int = ActivationMethod.SILU.value) -> torch.Tensor:
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


def rocm_aiter_biased_grouped_topk_impl(
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        routed_scaling_factor: float = 1.0  # mul to topk_weights
) -> None:

    from aiter import biased_grouped_topk

    biased_grouped_topk(gating_output, correction_bias, topk_weights, topk_ids,
                        num_expert_group, topk_group, need_renorm,
                        routed_scaling_factor)


def rocm_aiter_biased_grouped_topk_fake(
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        routed_scaling_factor: float = 1.0  # mul to topk_weights
) -> None:
    pass


def rocm_aiter_grouped_topk_impl(
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0  # mul to topk_weights
) -> None:

    from aiter import grouped_topk

    grouped_topk(gating_output, topk_weights, topk_ids, num_expert_group,
                 topk_group, need_renorm, scoring_func, routed_scaling_factor)


def rocm_aiter_grouped_topk_fake(
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0  # mul to topk_weights
) -> None:
    pass


def rocm_aiter_fused_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation_method: int = ActivationMethod.SILU.value,
    quant_method: int = QuantMethod.NO.value,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    activation = ActivationType(activation_method)
    quant_type = QuantType(quant_method)

    return fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, expert_mask,
                     activation, quant_type, doweight_stage1, w1_scale,
                     w2_scale, a1_scale, a2_scale)


def rocm_aiter_fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation_method: int = ActivationMethod.SILU.value,
    quant_method: int = QuantMethod.NO.value,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


if current_platform.is_rocm():

    direct_register_custom_op(
        op_name="rocm_aiter_asm_moe_tkw1",
        op_func=rocm_aiter_asm_moe_tkw1_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_asm_moe_tkw1_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_fused_moe",
        op_func=rocm_aiter_fused_moe_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_fused_moe_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_topk_softmax",
        op_func=rocm_aiter_topk_softmax_impl,
        mutates_args=["topk_weights", "topk_indices", "token_expert_indices"],
        fake_impl=rocm_aiter_topk_softmax_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_biased_grouped_topk",
        op_func=rocm_aiter_biased_grouped_topk_impl,
        mutates_args=["topk_weights", "topk_ids"],
        fake_impl=rocm_aiter_biased_grouped_topk_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_grouped_topk",
        op_func=rocm_aiter_grouped_topk_impl,
        mutates_args=["topk_weights", "topk_ids"],
        fake_impl=rocm_aiter_grouped_topk_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def rocm_aiter_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    token = hidden_states.shape[0]
    device = hidden_states.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk),
                               dtype=torch.float32,
                               device=device)

    if e_score_correction_bias is not None:
        torch.ops.vllm.rocm_aiter_biased_grouped_topk(
            gating_output,
            e_score_correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
        )
    else:
        assert (scoring_func == "softmax" or scoring_func == "sigmoid")
        torch.ops.vllm.rocm_aiter_grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scoring_func,
        )

    return topk_weights, topk_ids


def rocm_aiter_fused_experts(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_fp8_w8a8: bool = False,
        per_channel_quant: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[list[int]] = None) -> torch.Tensor:

    activation_method = (ActivationMethod.SILU
                         if activation == "silu" else ActivationMethod.GELU)
    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # w8a8 per-channel quantization
    if per_channel_quant and apply_router_weight_on_input and use_fp8_w8a8:
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
            expert_mask=None,
            activation_method=activation_method)

    else:
        quant_method = QuantMethod.NO.value

        # w8a8 block-scaled
        if block_shape is not None and use_fp8_w8a8:
            assert not apply_router_weight_on_input, (
                "apply_router_weight_on_input is\
                not supported for block scaled moe")
            assert w1_scale is not None
            assert w2_scale is not None
            quant_method = QuantMethod.BLOCK_128x128.value
        elif use_fp8_w8a8:
            # Currently only per tensor quantization method is enabled.
            quant_method = QuantMethod.PER_TENSOR.value

        if apply_router_weight_on_input:
            assert (topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"

        return torch.ops.vllm.rocm_aiter_fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_method=quant_method,
            activation_method=activation_method,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            doweight_stage1=apply_router_weight_on_input)


def rocm_aiter_topk_softmax(topk_weights: torch.Tensor,
                            topk_indices: torch.Tensor,
                            token_expert_indices: torch.Tensor,
                            gating_output: torch.Tensor,
                            renormalize: bool) -> tuple[torch.Tensor, ...]:
    torch.ops.vllm.rocm_aiter_topk_softmax(topk_weights, topk_indices,
                                           token_expert_indices, gating_output,
                                           renormalize)
    return topk_weights, topk_indices


def shuffle_weights(
    *tensors: torch.Tensor, layout: tuple[int, int] = (16, 16)
) -> tuple[torch.Tensor, ...]:
    """
    Applies shuffle_weight function from AITER to each 
    input tensor and returns them.
    
    Rearranges (shuffles) the input tensor/s
    into a specified block layout for optimized computation.

    Args:
        *tensors: Variable number of torch.Tensor objects.
        layout: A pair of integers specifying the 
        block sizes used to divide the tensors during shuffling.
        Default is (16, 16).

    Returns:
    A Tuple of shuffled tensors.
    """
    from aiter.ops.shuffle import shuffle_weight

    return tuple(shuffle_weight(tensor, layout=layout) for tensor in tensors)
