# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _minimax_moe_topk_sigmoid_quant_kernel(
    hidden_states_ptr,
    router_logits_ptr,
    e_score_correction_bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    a1q_ptr,
    a1q_scale_ptr,
    fp8_max_val,
    hidden_stride_m: tl.constexpr,
    logits_stride_m: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0)
    group_id = tl.program_id(1)

    hidden_offsets = group_id * GROUP_SIZE + tl.arange(0, BLOCK_H)
    hidden_mask = (hidden_offsets < HIDDEN_SIZE) & (
        hidden_offsets < (group_id + 1) * GROUP_SIZE
    )
    x = tl.load(
        hidden_states_ptr + token_id * hidden_stride_m + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0e-10)
    scale = tl.math.div_rn(absmax, fp8_max_val)
    x_q = tl.clamp(
        tl.math.div_rn(x, scale), fp8_min, fp8_max
    ).to(a1q_ptr.dtype.element_ty)

    tl.store(
        a1q_ptr + token_id * hidden_stride_m + hidden_offsets,
        x_q,
        mask=hidden_mask,
    )
    tl.store(a1q_scale_ptr + token_id * NUM_GROUPS + group_id, scale)

    if group_id == 0:
        expert_offsets = tl.arange(0, BLOCK_E)
        expert_mask = expert_offsets < NUM_EXPERTS
        logits = tl.load(
            router_logits_ptr + token_id * logits_stride_m + expert_offsets,
            mask=expert_mask,
            other=-float("inf"),
        ).to(tl.float32)

        sigmoid_scores = 1.0 / (1.0 + tl.exp(-logits))
        sigmoid_scores = tl.where(sigmoid_scores == sigmoid_scores, sigmoid_scores, 0.0)
        bias = tl.load(
            e_score_correction_bias_ptr + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        scores_for_choice = sigmoid_scores + bias
        scores_for_choice = tl.where(expert_mask, scores_for_choice, -float("inf"))

        selected_sum = tl.full((), 0.0, tl.float32)
        for k_idx in tl.static_range(0, TOP_K):
            selected_id = tl.argmax(scores_for_choice, axis=0)
            selected_weight = tl.sum(
                tl.where(expert_offsets == selected_id, sigmoid_scores, 0.0),
                axis=0,
            )
            tl.store(topk_ids_ptr + token_id * TOP_K + k_idx, selected_id)
            tl.store(topk_weights_ptr + token_id * TOP_K + k_idx, selected_weight)
            selected_sum += selected_weight
            scores_for_choice = tl.where(
                expert_offsets == selected_id,
                -float("inf"),
                scores_for_choice,
            )

        denom = tl.where(selected_sum > 0.0, selected_sum, 1.0)
        for k_idx in tl.static_range(0, TOP_K):
            weight = tl.load(topk_weights_ptr + token_id * TOP_K + k_idx)
            tl.store(topk_weights_ptr + token_id * TOP_K + k_idx, weight / denom)


def _allocate_outputs(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hidden_size = hidden_states.shape
    num_groups = triton.cdiv(hidden_size, block_k)
    topk_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=hidden_states.device
    )
    a1q = torch.empty(
        hidden_states.shape,
        dtype=current_platform.fp8_dtype(),
        device=hidden_states.device,
    )
    a1q_scale = torch.empty(
        (num_tokens, num_groups), dtype=torch.float32, device=hidden_states.device
    )
    return topk_weights, topk_ids, a1q, a1q_scale


def _validate_inputs(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    block_k: int,
) -> None:
    assert hidden_states.dim() == 2
    assert router_logits.dim() == 2
    assert hidden_states.shape[0] == router_logits.shape[0]
    assert hidden_states.stride(-1) == 1
    assert router_logits.stride(-1) == 1
    assert block_k > 0
    assert router_logits.shape[-1] == e_score_correction_bias.shape[0]


def _minimax_moe_topk_sigmoid_quant_triton_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_inputs(hidden_states, router_logits, e_score_correction_bias, block_k)

    num_tokens, hidden_size = hidden_states.shape
    num_experts = router_logits.shape[-1]
    num_groups = triton.cdiv(hidden_size, block_k)
    topk_weights, topk_ids, a1q, a1q_scale = _allocate_outputs(
        hidden_states, router_logits, top_k, block_k
    )

    if num_tokens == 0:
        return topk_weights, topk_ids, a1q, a1q_scale

    fp8_min, fp8_max = get_fp8_min_max()
    _minimax_moe_topk_sigmoid_quant_kernel[(num_tokens, num_groups)](
        hidden_states,
        router_logits,
        e_score_correction_bias,
        topk_weights,
        topk_ids,
        a1q,
        a1q_scale,
        fp8_max,
        hidden_states.stride(0),
        router_logits.stride(0),
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        TOP_K=top_k,
        NUM_EXPERTS=num_experts,
        HIDDEN_SIZE=hidden_size,
        GROUP_SIZE=block_k,
        NUM_GROUPS=num_groups,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_H=triton.next_power_of_2(block_k),
        num_warps=8,
    )

    return topk_weights, topk_ids, a1q, a1q_scale


def _minimax_moe_topk_sigmoid_quant_impl(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _minimax_moe_topk_sigmoid_quant_triton_impl(
        hidden_states,
        router_logits,
        e_score_correction_bias,
        top_k,
        block_k,
    )


def _minimax_moe_topk_sigmoid_quant_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, hidden_size = hidden_states.shape
    num_groups = (hidden_size + block_k - 1) // block_k
    topk_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=hidden_states.device
    )
    a1q = torch.empty(
        hidden_states.shape,
        dtype=current_platform.fp8_dtype(),
        device=hidden_states.device,
    )
    a1q_scale = torch.empty(
        (num_tokens, num_groups), dtype=torch.float32, device=hidden_states.device
    )
    return topk_weights, topk_ids, a1q, a1q_scale


direct_register_custom_op(
    "minimax_moe_topk_sigmoid_quant",
    _minimax_moe_topk_sigmoid_quant_impl,
    fake_impl=_minimax_moe_topk_sigmoid_quant_fake,
)


def minimax_moe_topk_sigmoid_quant(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.minimax_moe_topk_sigmoid_quant(
        hidden_states,
        router_logits,
        e_score_correction_bias,
        top_k,
        block_k,
    )


def minimax_moe_topk_sigmoid_quant_triton(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _minimax_moe_topk_sigmoid_quant_triton_impl(
        hidden_states,
        router_logits,
        e_score_correction_bias,
        top_k,
        block_k,
    )
