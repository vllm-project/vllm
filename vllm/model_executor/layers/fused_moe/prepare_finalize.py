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


class MoEPrepareAndFinalizeNaiveEPBase(mk.FusedMoEPrepareAndFinalizeBase):
    """
    Base class for Naive Prepare/Finalize for Dp/Ep with two subclasses:
    * Modular Case
    * Monolithic Case

    In modular case, a separate router runs *before* and we dispatch
    the topk weights and ids.

    In monolithic case, the router runs *inside* the MoE kernel so we
    dispatch the router logits.

    In both cases, the quantization of X happens prior to dispatching.
    """

    @staticmethod
    def make(
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
        use_monolithic: bool = False,
    ) -> "MoEPrepareAndFinalizeNaiveEPBase":
        cls = (
            MoEPrepareAndFinalizeNaiveEPMonolithic
            if use_monolithic
            else MoEPrepareAndFinalizeNaiveEP
        )
        return cls(
            is_sequence_parallel=is_sequence_parallel, num_dispatchers=num_dispatchers
        )

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

    def _quantize_and_setup_dispatch(
        self,
        a1: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        # Defer input quantization to the MoE kernel.
        if defer_input_quant:
            a1q = a1
            a1q_scale = None
        else:
            input_sf = (
                quant_config.a1_gscale
                if quant_config.use_nvfp4_w4a4
                else quant_config.a1_scale
            )

            # NOTE: swizzling pads the scales to multiple of 128
            # which makes the scales tensor different shape than
            # the hidden states, breaking the A2A kernel. So, we
            # delay the swizzling until after the A2A.
            a1q, a1q_scale = a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                input_sf,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
                is_fp4_scale_swizzled=False,
            )

        # Skip gathering scales if we have static quantization
        # (the scale is a scalar, replicated on all ranks) or
        # if quantization is deferred.
        skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
        scales = None if skip_gather_scales else [a1q_scale]

        return a1q, scales

    def _unwrap_scale_and_prepare_for_moe(
        self,
        scales: list[torch.Tensor] | None,
        quant_config: FusedMoEQuantConfig,
    ) -> torch.Tensor:
        assert scales is not None and len(scales) == 1
        a1q_scale = scales[0]
        # Apply swizzling after a2a if the MoE kernel needs it.
        if quant_config.quant_dtype == "nvfp4" and quant_config.is_nvfp4_scale_swizzled:
            assert a1q_scale is not None
            if a1q_scale.element_size() == 1:
                a1q_scale = a1q_scale.view(torch.uint8)
            a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

        return a1q_scale


class MoEPrepareAndFinalizeNaiveEP(
    MoEPrepareAndFinalizeNaiveEPBase, mk.FusedMoEPrepareAndFinalize
):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the topk weights and ids.
    """

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
        """Quantize and Dispatch Topk Weights and Topk Ids."""

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            # Note: do not use inplace for shared experts overlap
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, scales = self._quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            a1q, topk_weights, topk_ids = res
        else:
            a1q, topk_weights, topk_ids, scales = res
            a1q_scale = self._unwrap_scale_and_prepare_for_moe(scales, quant_config)

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


class MoEPrepareAndFinalizeNaiveEPMonolithic(
    MoEPrepareAndFinalizeNaiveEPBase, mk.FusedMoEPrepareAndFinalizeMonolithic
):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the router logits (the MoE kernel runs the router internally).
    """

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        """Quantize and Dispatch Router Logits."""

        a1q, scales = self._quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        res = get_ep_group().dispatch_router_logits(
            a1q,
            router_logits,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            a1q, router_logits = res
            a1q_scale = None
        else:
            a1q, router_logits, scales = res
            a1q_scale = self._unwrap_scale_and_prepare_for_moe(scales, quant_config)

        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        out = get_ep_group().combine(
            fused_expert_output, is_sequence_parallel=self.is_sequence_parallel
        )
        return out


class MoEPrepareAndFinalizeNoEPBase(mk.FusedMoEPrepareAndFinalizeBase):
    """
    Base class for TP case Prepare/Finalize.
    * prepare: applies input quantization
    * finalize: applies the reduction (if needed)
    """

    @staticmethod
    def make(use_monolithic: bool) -> "MoEPrepareAndFinalizeNoEPBase":
        return (
            MoEPrepareAndFinalizeNoEPMonolithic()
            if use_monolithic
            else MoEPrepareAndFinalizeNoEP()
        )

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

    def _quantize_input(
        self,
        a1: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Defer input quant to moe kernel for backends (e.g. AITER, FI)
        # which use a single kernel call for quant + experts.
        if defer_input_quant:
            return a1, None

        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
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


class MoEPrepareAndFinalizeNoEP(
    mk.FusedMoEPrepareAndFinalize, MoEPrepareAndFinalizeNoEPBase
):
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

        a1q, a1q_scale = self._quantize_input(a1, quant_config, defer_input_quant)

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


class MoEPrepareAndFinalizeNoEPMonolithic(
    mk.FusedMoEPrepareAndFinalizeMonolithic, MoEPrepareAndFinalizeNoEPBase
):
    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        a1q, a1q_scale = self._quantize_input(a1, quant_config, defer_input_quant)
        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        return fused_expert_output
