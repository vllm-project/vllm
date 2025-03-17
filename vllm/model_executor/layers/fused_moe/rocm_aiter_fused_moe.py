# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)


def fused_experts(
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
        **kwagrs  # Ignore additional keyword arguments
) -> torch.Tensor:

    import aiter as rocm_aiter
    import aiter.fused_moe_bf16_asm as rocm_aiter_asm_fmoe

    if envs.VLLM_ROCM_USE_AITER_FP8_BLOCK_SCALED_MOE and use_fp8_w8a8:
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

    elif use_fp8_w8a8:
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
