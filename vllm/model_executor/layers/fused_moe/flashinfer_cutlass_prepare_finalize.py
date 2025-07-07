# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
from flashinfer import fp4_swizzle_blockscale
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)

def get_local_sizes():
    cu_sizes = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
    sizes = [cu_sizes[0].item()]
    for i in range(1, len(cu_sizes)):
        sizes.append((cu_sizes[i] - cu_sizes[i - 1]).item())
    return sizes


class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
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

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

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
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        use_dp: Optional[bool] = True,
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
            quant_config.quant_dtype,
            self.per_channel_quant,
            self.block_shape,
            is_fp4_scalar_swizzled=
            not use_dp,  # Needs swizzling after communication
        )
        if use_dp:
            topk_weights, topk_ids, a1q, a1q_scale = \
                get_dp_group().all_gatherv([topk_weights, topk_ids, a1q, a1q_scale],
                                           dim=0,
                                           sizes=get_local_sizes())
            a1_m, a1_n = a1q.shape
            a1q_scale = fp4_swizzle_blockscale(a1q_scale, a1_m, a1_n * 2)

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
            fused_expert_output = get_dp_group().reduce_scatter(
                fused_expert_output,
                dim=0,
                sizes=get_local_sizes(),
            )
        output.copy_(fused_expert_output)
