# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


def _quantize_input(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Defer input quant to moe kernel for backends (e.g. AITER, FI)
    # which use a single kernel call for quant + experts.
    if defer_input_quant:
        return a1, None

    input_sf = (
        quant_config.a1_gscale if quant_config.use_nvfp4_w4a4 else quant_config.a1_scale
    )
    a1q, a1q_scale = moe_kernel_quantize_input(
        a1,
        input_sf,
        quant_dtype=quant_config.quant_dtype,
        per_act_token_quant=quant_config.per_act_token_quant,
        block_shape=quant_config.block_shape,
        is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,
    )

    return a1q, a1q_scale


class MoEPrepareAndFinalizeNoDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            # Note: do not use inplace for shared experts overlap
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, a1q_scale = _quantize_input(a1, quant_config, defer_input_quant)

        return a1q, a1q_scale, None, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        weight_and_reduce_impl.apply(
            output=output,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class MoEPrepareAndFinalizeNoDPEPMonolithic(mk.FusedMoEPrepareAndFinalizeMonolithic):
    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        a1q, a1q_scale = _quantize_input(a1, quant_config, defer_input_quant)
        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        return fused_expert_output


def make_moe_prepare_and_finalize_no_dp_ep(
    use_monolithic: bool,
) -> MoEPrepareAndFinalizeNoDPEPModular | MoEPrepareAndFinalizeNoDPEPMonolithic:
    return (
        MoEPrepareAndFinalizeNoDPEPMonolithic()
        if use_monolithic
        else MoEPrepareAndFinalizeNoDPEPModular()
    )
