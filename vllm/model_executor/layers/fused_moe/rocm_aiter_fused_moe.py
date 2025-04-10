# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


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


# 1. Register rocm_aiter.ck_moe
def rocm_aiter_ck_moe_impl(hidden_states: torch.Tensor, w1: torch.Tensor,
                           w2: torch.Tensor, topk_weights: torch.Tensor,
                           topk_ids: torch.Tensor) -> torch.Tensor:
    import aiter as rocm_aiter
    return rocm_aiter.ck_moe(hidden_states=hidden_states,
                             w1=w1,
                             w2=w2,
                             topk_weights=topk_weights,
                             topk_ids=topk_ids)


def rocm_aiter_ck_moe_fake(hidden_states: torch.Tensor, w1: torch.Tensor,
                           w2: torch.Tensor, topk_weights: torch.Tensor,
                           topk_ids: torch.Tensor) -> torch.Tensor:
    return torch.empty((topk_ids.size(0), hidden_states.size(1)),
                       dtype=hidden_states.dtype,
                       device=hidden_states.device)


# 2. Register rocm_aiter_asm_fmoe.moe_sorting_ck
def moe_sorting_ck_impl(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    model_dim: int,
    dtype: torch.dtype,
    expert_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    import aiter.fused_moe_bf16_asm as rocm_aiter_asm_fmoe
    return rocm_aiter_asm_fmoe.moe_sorting_ck(topk_ids,
                                              topk_weights,
                                              E,
                                              model_dim,
                                              dtype,
                                              expert_mask=expert_mask)


def moe_sorting_ck_fake(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    E: int,
    model_dim: int,
    dtype: torch.dtype,
    expert_mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    # Create fake tensors with appropriate shapes
    sorted_token_ids = torch.zeros_like(topk_ids)
    sorted_weight_buf = torch.zeros_like(topk_weights)
    sorted_expert_ids = torch.zeros_like(topk_ids)
    num_valid_ids = torch.zeros(1, dtype=torch.int32, device=topk_ids.device)
    out_asm = torch.zeros((topk_ids.size(0), model_dim),
                          dtype=dtype,
                          device=topk_ids.device)
    return sorted_token_ids, sorted_weight_buf, \
        sorted_expert_ids, num_valid_ids, out_asm


# 3. Register rocm_aiter.fmoe_fp8_blockscale_g1u1
def fmoe_fp8_blockscale_g1u1_impl(
        out_asm: torch.Tensor,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        sorted_weight_buf: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        topk: int,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1_scale: torch.Tensor,
        block_m: int,
        block_n: int,
        smooth_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    import aiter as rocm_aiter
    rocm_aiter.fmoe_fp8_blockscale_g1u1(out_asm, a1, w1, w2, sorted_token_ids,
                                        sorted_weight_buf, sorted_expert_ids,
                                        num_valid_ids, topk, w1_scale,
                                        w2_scale, a1_scale, block_m, block_n,
                                        smooth_scale)
    return out_asm


def fmoe_fp8_blockscale_g1u1_fake(
        out_asm: torch.Tensor,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        sorted_weight_buf: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        topk: int,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        a1_scale: torch.Tensor,
        block_m: int,
        block_n: int,
        smooth_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    return out_asm


def asm_moe_tkw1_impl(hidden_states: torch.Tensor,
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
    import aiter as rocm_aiter
    import aiter.fused_moe_bf16_asm as rocm_aiter_asm_fmoe

    if activation == "silu":
        aiter_activation = rocm_aiter.ActivationType.Silu
    elif activation == "gelu":
        aiter_activation = rocm_aiter.ActivationType.Gelu
    else:
        raise ValueError(f"The given activation: {activation}"
                         " is not supported in AITER.")

    return rocm_aiter_asm_fmoe.asm_moe_tkw1(hidden_states=hidden_states,
                                            w1=w1,
                                            w2=w2,
                                            topk_weight=topk_weight,
                                            topk_ids=topk_ids,
                                            fc1_scale=fc1_scale,
                                            fc2_scale=fc1_scale,
                                            fc1_smooth_scale=None,
                                            fc2_smooth_scale=None,
                                            a16=a16,
                                            activation=aiter_activation)


def asm_moe_tkw1_fake(hidden_states: torch.Tensor,
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


# 4. Register rocm_aiter_asm_fmoe.asm_moe
def asm_moe_impl(hidden_states: torch.Tensor,
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
    import aiter as rocm_aiter
    import aiter.fused_moe_bf16_asm as rocm_aiter_asm_fmoe

    if activation == "silu":
        aiter_activation = rocm_aiter.ActivationType.Silu
    elif activation == "gelu":
        aiter_activation = rocm_aiter.ActivationType.Gelu
    else:
        raise ValueError(f"The given activation: {activation}"
                         " is not supported in AITER.")

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


def asm_moe_fake(hidden_states: torch.Tensor,
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


# 5. Register rocm_aiter.topk_softmax
def aiter_topk_softmax_impl(topk_weights: torch.Tensor,
                            topk_indices: torch.Tensor,
                            token_expert_indices: torch.Tensor,
                            gating_output: torch.Tensor,
                            renormalize: bool) -> None:
    import aiter as rocm_aiter
    rocm_aiter.topk_softmax(topk_weights, topk_indices, token_expert_indices,
                            gating_output, renormalize)


def topk_softmax_fake(topk_weights: torch.Tensor, topk_indices: torch.Tensor,
                      token_expert_indices: torch.Tensor,
                      gating_output: torch.Tensor, renormalize: bool) -> None:
    pass  # No-op for fake implementation


# 6. Register aiter.ops.shuffle.shuffle_weight
def shuffle_weight_impl(tensor: torch.Tensor) -> torch.Tensor:
    from aiter.ops.shuffle import shuffle_weight
    return shuffle_weight(tensor)


def shuffle_weight_fake(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone()  # Just return a copy of the tensor


# Register all custom ops if on ROCm platform
if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_shuffle_Weight",
        op_func=shuffle_weight_impl,
        mutates_args=[],
        fake_impl=shuffle_weight_fake,
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
        op_name="rocm_aiter_moe_sorting_ck",
        op_func=moe_sorting_ck_impl,
        mutates_args=[],
        fake_impl=moe_sorting_ck_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_fmoe_fp8_blockscale_g1u1",
        op_func=fmoe_fp8_blockscale_g1u1_impl,
        mutates_args=["out_asm"],
        fake_impl=fmoe_fp8_blockscale_g1u1_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_asm_moe",
        op_func=asm_moe_impl,
        mutates_args=[],
        fake_impl=asm_moe_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_asm_moe_tkw1",
        op_func=asm_moe_tkw1_impl,
        mutates_args=[],
        fake_impl=asm_moe_tkw1_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_topk_softmax",
        op_func=aiter_topk_softmax_impl,
        mutates_args=["topk_weights", "topk_indices"],
        fake_impl=topk_softmax_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_shuffle_weight",
        op_func=shuffle_weight_impl,
        mutates_args=[],
        fake_impl=shuffle_weight_fake,
        dispatch_key=current_platform.dispatch_key,
    )


def rocm_aiter_fused_experts(
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        use_fp8_w8a8: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[List[int]] = None,
        expert_mask: Optional[torch.Tensor] = None,
        activation: str = "silu",
        **kwargs  # Keep this for backward compatibility
) -> torch.Tensor:

    if is_rocm_aiter_block_scaled_moe_enabled() and use_fp8_w8a8:
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

        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8)

        (
            sorted_token_ids,
            sorted_weight_buf,
            sorted_expert_ids,
            num_valid_ids,
            out_asm,
        ) = torch.ops.vllm.rocm_aiter_moe_sorting_ck(topk_ids, topk_weights, E,
                                                     model_dim, dtype,
                                                     expert_mask)

        a1, a1_scale = per_token_group_quant_fp8(hidden_states, scale_blk_k)

        torch.ops.vllm.rocm_aiter_fmoe_fp8_blockscale_g1u1(
            out_asm, a1, w1, w2, sorted_token_ids, sorted_weight_buf,
            sorted_expert_ids, num_valid_ids, topk, w1_scale.view(local_E, -1),
            w2_scale.view(local_E, -1),
            a1_scale.t().contiguous(), block_shape[0], block_shape[1], None)
        return out_asm

    elif is_rocm_aiter_channel_scaled_moe_enabled() and use_fp8_w8a8:
        return torch.ops.vllm.rocm_aiter_asm_moe_tkw1(
            hidden_states=hidden_states,
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

    elif use_fp8_w8a8:
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
    torch.ops.vllm.topk_softmax(topk_weights, topk_indices,
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
    shuffle_weigth_func = torch.ops.vllm.rocm_aiter_shuffle_weight

    return tuple(shuffle_weigth_func(tensor) for tensor in tensors)


def expand_weights(*tensors: torch.Tensor,
                   expansion_dims: list[int]) -> Tuple[torch.Tensor, ...]:
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
