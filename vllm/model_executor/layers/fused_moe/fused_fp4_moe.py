# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused MoE utilities for NVFP4 Quantization"""
import functools
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size, try_get_optimal_moe_config)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils import direct_register_custom_op

def fp4_moe_gemm(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    hs_blockscale: torch.Tensor,
    w_blockscale: torch.Tensor,
    gating_output: torch.Tensor,
    num_topk: int,
    block_size: int,
) -> torch.Tensor:
    """
    This function computes the gemm used in MoE by utilizing the 
    cutlass_fp4_groupgemm op. The input hidden states and weights
    are NVFP4 quantized and each has a corresponding blockscale 
    as well as a global scale alpha. 

    The MoE gemm uses the sparse top k experts to pass the 
    corresponding weights.

    Parameters:
    hidden_states: Shape[M, K]
    expert_weights: Shape[E, K, N] (w13 or w2)
    hs_blockscale: Hidden State blockscale
    weights_blockscale: Weights block scale
    Alpha
    gating_output
    num_topk(int): The number of top_k experts to select.


    Returns:
    torch.Tensor: The output tensor after applying the MoE layer.
    """
    
    M, K = hidden_states.shape
    assert(expert_weights.shape[1] == K), (
        "Contracting dimension size must match")
    
    E = expert_weights.shape[0]
    N = expert_weights.shape[2] // 2

    topk_weights, topk_ids = fused_topk(hidden_states=hidden_states,
                                        gating_output=gating_output,
                                        topk=num_topk)

    sorted_token_ids, expert_ids, num_tokens_post_padded =(
        moe_align_block_size(topk_ids, block_size, E)
    ) 
    # Do the permutation of hidden_states based on sorted_token_ids
    

    # Do the FP4 quantization of hidden states

    
    # Weight swizzling for FP4 weights(move it to process weights after loading)
    

    # Call FP4 group gemm with Fp4 weights and scales.


def fused_moe_fp4(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
) -> torch.Tensor:
    





def fused_moe_fp4_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="fused_moe_fp4",
    op_func=fused_moe_fp4,
    mutates_args=[],
    fake_impl=fused_moe_fp4_fake,
)
