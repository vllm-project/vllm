# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.distributed import get_ep_group
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    _moe_unpermute_and_reduce)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)

def swizzle_sf(unswizzled_sf: torch.Tensor,
               original_row: int,
               original_col: int,
               scaling_vector_size: int = 16) -> torch.Tensor:
    """
    Converts an unswizzled tensor back to swizzled form.
    
    Args:
        unswizzled_sf: Tensor of shape [row, col // scaling_vector_size].
        original_row: Original row dimension (e.g., 120).
        original_col: Original column dimension (e.g., 64).
        scaling_vector_size: Scaling factor (default 16).
    
    Returns:
        Swizzled tensor of shape [padded_row, padded_col].
    """
    factor = scaling_vector_size * 4
    padded_row = ((original_row + 128 - 1) // 128) * 128  # Next multiple of 128
    padded_col = ((original_col + factor - 1) // factor) * factor  # Next multiple of 64
    
    # Pad the input tensor to [padded_row, padded_col // scaling_vector_size]
    pad_rows = padded_row - original_row
    pad_cols = (padded_col - original_col) // scaling_vector_size
    padded_sf = torch.nn.functional.pad(
        unswizzled_sf,
        (0, pad_cols, 0, pad_rows),
        mode='constant',
        value=0
    )
    
    # Reshape and transpose to reverse unswizzle_sf
    num_m_tiles = padded_row // 128
    num_k_tiles = padded_col // factor
    sf_reshaped = padded_sf.view(num_m_tiles, 4, 32, num_k_tiles, 4)  # Reverse reshape
    sf_swizzled = sf_reshaped.transpose(1, 3)  # Reverse transpose [num_m_tiles, num_k_tiles, 32, 4, 4]
    sf_swizzled = sf_swizzled.reshape(padded_row, padded_col // scaling_vector_size)  # Flatten to [128, 64]
    
    return sf_swizzled.contiguous()

class FlashInferCutlassMoEPrepareAndFinalizeNoEP(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        quant_dtype: Optional[torch.dtype] = None,
        per_channel_quant: bool = False,
        block_shape: Optional[list[int]] = None,
    ):
        super().__init__()
        self.per_channel_quant = per_channel_quant
        self.block_shape = block_shape
        self.quant_dtype = quant_dtype

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool = False,
        use_dp: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1.mul_(topk_weights.to(a1.dtype))

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            a1_scale,
            self.quant_dtype,
            self.per_channel_quant,
            self.block_shape,
            is_sf_swizzled_layout=not use_dp,  # Needs swizzling after communication
        )        
        if use_dp:
            topk_weights, topk_ids = get_ep_group().dispatch(topk_weights, topk_ids)
            # TODO(shuw): Improve by efficient all-gather
            a1q, a1q_scale = get_ep_group().dispatch(a1q, a1q_scale)
            a1_m, a1_n = a1q.shape
            a1q_scale = swizzle_sf(a1q_scale, a1_m, a1_n * 2)

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        use_dp: bool = False,
    ) -> None:
        if use_dp:
            # TODO(shuw): Improve by efficient reduce-scatter
            fused_expert_output = get_ep_group().combine(fused_expert_output)
        output.copy_(fused_expert_output)
