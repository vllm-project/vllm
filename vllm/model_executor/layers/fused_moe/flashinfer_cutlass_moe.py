# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib.util
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    _moe_permute)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (
    FlashInferCutlassMoEPrepareAndFinalizeNoEP)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache, per_token_group_quant_fp8)
from vllm.utils import round_up

logger = init_logger(__name__)

from typing import TYPE_CHECKING
try:
    from flashinfer.fused_moe import cutlass_fused_moe as cutlass_fused_moe
    from flashinfer import fp4_quantize as fp4_quantize
except ImportError:
    if not TYPE_CHECKING:
        cutlass_fused_moe = None



has_flashinfer_cutlass_fused_moe = cutlass_fused_moe is not None
def _valid_flashinfer_fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor,
                     w2: torch.Tensor) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    # TODO(shuw): add data type check!
    if not has_flashinfer_cutlass_fused_moe:
        logger.debug("FlashInferExperts disabled: flashinfer_cutlass_fused_moe not available.")
        return False
    return True

class FlashInferExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(self):
        super().__init__()


    def supports_chunking(self) -> bool:
        return True

    def workspace_shapes(
        self, a: torch.Tensor, aq: torch.Tensor, M: int, N: int, K: int,
        topk: int, global_num_experts: int, local_num_experts: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # We use global_num_experts due to how moe_align_block_size handles
        # expert_maps.
        # TODO(shuw): Just 0 Here
        num_experts = global_num_experts
        block_m = self.block_shape[0]
        M_sum = (M * topk) + num_experts * (block_m - 1)
        M_sum = round_up(M_sum, block_m)
        workspace1 = (M_sum, max(N * 2, K))
        workspace2 = (M_sum, max(N, K))
        output = (M * topk, K)
        return (workspace1, workspace2, output, a.dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation: str,
        global_num_experts: int,
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        expert_num_tokens: Optional[torch.Tensor],
        g1_alphas: torch.Tensor,
        g2_alphas: torch.Tensor,
        input_sf: torch.Tensor,
        out_dtype: torch.dtype,
        ep_rank: Optional[int] = None,
        ep_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ):
        # Flashinfer CUTLASS kernel takes scalar global scales,
        # min because inv_scale. 
        a1_gs = torch.min(a1_scale)
        a2_gs = torch.min(a2_scale)
        w1_blockscale=w1_scale                                                                 
        w2_blockscale=w2_scale                              
        
        quant_scales=[
            a1_gs,
            w1_blockscale.view(torch.int32),
            g1_alphas,
            a2_gs,
            w2_blockscale.view(torch.int32),
            g2_alphas,
        ]
        # TRTLLM Cutlass moe takes in activations in BF16/Half/nvfp4 precision
        # and fp4 quantized weights loaded from the checkpoint
        # TODO(shuw): do quantization here
        # out_dtype = x.dtype
        # x, input_sf = fp4_quantize(x, a1_gs)
        # print("calling flashinfer cutlass fused moe\n"*100)
        output = cutlass_fused_moe(
            hidden_states,
            topk_ids.to(torch.int),                                                               
            topk_weights,                                                                                              
            w1.view(torch.long),                                                                         
            w2.view(torch.long),                                                                          
            out_dtype,                                                                  
            quant_scales=quant_scales,
            input_sf=input_sf,
            ep_size=ep_size,
            ep_rank=ep_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )[0]

class FlashInferCutlassKernels(mk.FusedMoEModularKernel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        expert_num_tokens: Optional[torch.Tensor] = None,
        g1_alphas: torch.Tensor = None,
        g2_alphas: torch.Tensor = None,
        input_sf: torch.Tensor = None,
        out_dtype: torch.dtype = None,
        ep_rank: Optional[int] = 0,
        ep_size: Optional[int] = 1,
        tp_rank: Optional[int] = 0,
        tp_size: Optional[int] = 1,
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        has_nvfp4 = False
        if self.prepare_finalize.quant_dtype == torch.uint8:
            has_nvfp4 = True
        a1 = hidden_states
        output = a1 if inplace else torch.zeros_like(a1)
        
		# flashinfer kernel don't need partition topk_ids and top_weights
        # just to quantization in prepare
        
        (a1q, a1q_scale, expert_num_tokens, _expert_topk_ids,
         _expert_topk_weights) = self.prepare_finalize.prepare(
             a1, a1_scale, a2_scale, topk_weights, topk_ids,
             global_num_experts, expert_map, apply_router_weight_on_input)

        fused_out = output
        # TODO(shuw): no chunk atm
        self.fused_experts.apply(
            fused_out,
            a1q,
            w1,
            w2,
            topk_ids,
            topk_weights,
            activation,
            global_num_experts,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1_scale,
            a2_scale,
            expert_num_tokens,
            g1_alphas,
            g2_alphas,
            a1q_scale,
            out_dtype,
            ep_rank,
            ep_size,
            tp_rank,
            tp_size,
        )
        self.prepare_finalize.finalize(output, fused_out, topk_weights,
                                       topk_ids, apply_router_weight_on_input)
        return output

def flashinfer_cutlass_fused_moe_nvfp4(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    global_num_experts: int = -1,
    ep_rank: Optional[int] = 0,
    ep_size: Optional[int] = 1,
    tp_rank: Optional[int] = 0,
    tp_size: Optional[int] = 1,
    apply_router_weight_on_input=False,
)-> torch.Tensor:
    fn = FlashInferCutlassKernels(
        FlashInferCutlassMoEPrepareAndFinalizeNoEP(
            quant_dtype=torch.uint8, #meaning 2x e2m1 packed in one
        ),
        FlashInferExperts(),
    )
    # quant_scales computed in the prepare
    return fn(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        inplace,
        activation,
        global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        ep_rank=ep_rank,
        ep_size=ep_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
        out_dtype=hidden_states.dtype,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )