# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    extract_required_args, moe_kernel_quantize_input)
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def get_local_sizes():
    return get_forward_context().dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        quant_dtype: Optional[torch.dtype] = None,
        per_channel_quant: bool = False,
        block_shape: Optional[list[int]] = None,
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.per_channel_quant = per_channel_quant
        self.block_shape = block_shape
        self.quant_dtype = quant_dtype
        self.num_dispatchers_ = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],  # Not used
        a2_scale: Optional[torch.Tensor],  # Not used
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        extra_prepare_args: Optional[dict[str, Any]]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:

        assert not apply_router_weight_on_input

        (a1_gscale, use_dp, local_tokens) = extract_required_args(
            extra_prepare_args, ['a1_gscale', 'use_dp', 'local_tokens'])

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            a1_gscale,
            quant_config.quant_dtype,
            self.per_channel_quant,
            self.block_shape,
            is_fp4_scale_swizzled=not use_dp,  # Swizzling after communication
        )
        if use_dp:
            topk_weights, topk_ids, a1q, a1q_scale = \
                get_dp_group().all_gatherv([topk_weights, topk_ids, a1q, a1q_scale], # noqa: E501
                                           dim=0,
                                           sizes=get_local_sizes())
            a1_m, a1_n = a1q.shape
            a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool,
                 weight_and_reduce_impl: mk.TopKWeightAndReduce,
                 extra_finalize_args: Optional[dict[str, Any]]) -> None:

        (use_dp,
         local_tokens) = extract_required_args(extra_finalize_args,
                                               ['use_dp', 'local_tokens'])
        if use_dp:
            fused_expert_output = get_dp_group().reduce_scatterv(
                fused_expert_output, dim=0, sizes=get_local_sizes())
        output.copy_(fused_expert_output)
