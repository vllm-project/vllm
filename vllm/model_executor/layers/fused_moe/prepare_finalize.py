# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


class MoEPrepareAndFinalizeNaiveEP(mk.FusedMoEPrepareAndFinalize):
    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

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
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            # Note: do not use inplace for shared experts overlap
            a1 = a1 * topk_weights.to(a1.dtype)

        # Defer input quantization to the MoE kernel.
        use_nvfp4 = quant_config.use_nvfp4_w4a4
        if defer_input_quant:
            a1q = a1
            a1q_scale = None
        else:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_gscale if use_nvfp4 else quant_config.a1_scale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                # NOTE: swizzling pads the scales to multiple of 128
                # which makes the scales tensor different shape than
                # the hidden states, breaking the A2A kernel. So, we
                # delay the swizzling until after the A2A.
                is_fp4_scale_swizzled=False,
            )

        # Skip gathering scales if we have static quantization
        # (the scale is a scalar, replicated on all ranks) or
        # if quantization is deferred.
        skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
        scales = None if skip_gather_scales else [a1q_scale]

        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )
        if skip_gather_scales:
            a1q, topk_weights, topk_ids = res
        else:
            a1q, topk_weights, topk_ids, scales = res
            assert scales is not None and len(scales) == 1
            a1q_scale = scales[0]
            if quant_config.quant_dtype == "nvfp4":
                assert a1q_scale is not None
                if a1q_scale.element_size() == 1:
                    a1q_scale = a1q_scale.view(torch.uint8)
                a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

        return a1q, a1q_scale, None, topk_ids, topk_weights

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

        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        output.copy_(
            get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
        )


class MoEPrepareAndFinalizeNoDPEP(mk.FusedMoEPrepareAndFinalize):
    def __init__(self, defer_input_quant: bool = False) -> None:
        super().__init__()
        self.defer_input_quant = defer_input_quant

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

        # Defer input quant to moe kernel for backends (e.g. AITER, FI)
        # which use a single kernel call for quant + experts.
        if defer_input_quant:
            return a1, None, None, None, None

        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            input_sf,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )

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
