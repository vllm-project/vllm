# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

_TOPK = 6

# Adapted from:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/moe_fused_gate.py


def can_use_dsv4_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    indices_dtype: torch.dtype,
) -> bool:
    return (
        current_platform.is_cuda()
        and gating_output.dtype == torch.float32
        and gating_output.ndim == 2
        and gating_output.shape[1] in (256, 384)
        and gating_output.is_contiguous()
        and correction_bias is not None
        and correction_bias.dtype == torch.float32
        and correction_bias.shape == (gating_output.shape[1],)
        and correction_bias.is_contiguous()
        and topk == _TOPK
        and renormalize
        and indices_dtype in (torch.int32, torch.uint32, torch.int64)
    )


if current_platform.is_cuda():

    @triton.jit
    def _dsv4_topk_kernel(
        gating_output_ptr,
        correction_bias_ptr,
        topk_weights_ptr,
        topk_ids_ptr,
        routed_scaling_factor,
        NUM_EXPERTS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        row = tl.program_id(0)
        expert_offsets = tl.arange(0, BLOCK_N)
        expert_mask = expert_offsets < NUM_EXPERTS
        bias = tl.load(
            correction_bias_ptr + expert_offsets, mask=expert_mask, other=0.0
        ).to(tl.float32)

        if launch_pdl:
            tl.extra.cuda.gdc_wait()

        logits = tl.load(
            gating_output_ptr + row * NUM_EXPERTS + expert_offsets,
            mask=expert_mask,
            other=0.0,
        ).to(tl.float32)
        weights = tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))))
        current = tl.where(expert_mask, weights + bias, -float("inf"))
        current = tl.where(current == current, current, -1e30)

        topk_offsets = tl.arange(0, 8)
        selected_weights = tl.zeros([8], dtype=tl.float32)
        selected_ids = tl.zeros([8], dtype=tl.int32)
        for slot in tl.static_range(6):
            max_value = tl.max(current, axis=0)
            candidate = tl.where(current == max_value, expert_offsets, NUM_EXPERTS)
            expert_id = tl.min(candidate, axis=0).to(tl.int32)
            selected_weight = tl.sum(
                tl.where(expert_offsets == expert_id, weights, 0.0), axis=0
            )
            is_slot = topk_offsets == slot
            selected_weights = tl.where(is_slot, selected_weight, selected_weights)
            selected_ids = tl.where(is_slot, expert_id, selected_ids)
            current = tl.where(expert_offsets == expert_id, -float("inf"), current)

        weight_sum = tl.sum(selected_weights, axis=0)
        selected_weights *= routed_scaling_factor / tl.where(
            weight_sum > 0.0, weight_sum, 1.0
        )
        output_mask = topk_offsets < 6
        output_offsets = row * 6 + topk_offsets

        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()

        tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
        tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)


def dsv4_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    indices_dtype: torch.dtype,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = gating_output.shape
    shape = (num_tokens, _TOPK)
    topk_weights = gating_output.new_empty(shape, dtype=torch.float32)
    topk_ids = gating_output.new_empty(shape, dtype=indices_dtype)
    if num_tokens > 0:
        _dsv4_topk_kernel[(num_tokens,)](
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            routed_scaling_factor,
            NUM_EXPERTS=num_experts,
            BLOCK_N=triton.next_power_of_2(num_experts),
            num_warps=1,
            launch_pdl=current_platform.is_arch_support_pdl(),
        )
    return topk_weights, topk_ids
