# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.hw_agnostic.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.hw_agnostic.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.topk_weight_and_reduce import (  # noqa: E501
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.hw_agnostic.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
)


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
    if defer_input_quant:
        return a1, None, None

    a1q, a1q_scale = moe_kernel_quantize_input(
        a1,
        quant_config.a1_scale,
        quant_dtype=quant_config.quant_dtype,
        per_act_token_quant=quant_config.per_act_token_quant,
        block_shape=quant_config.block_shape,
    )

    # Static (scalar) scales are replicated on every rank — skip gathering.
    skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
    scales = None if skip_gather_scales else [a1q_scale]
    return a1q, scales, a1q_scale


class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the topk weights and ids.
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers
        # Set by FusedMoEWithLoRA.set_mapping() when LoRA is active. When
        # present, prepare() dispatches the per-token LoRA mapping alongside
        # hidden_states and writes the gathered result back to the context so
        # experts can use the per-rank-local mapping.
        self._lora_context = None

    def set_lora_context(self, ctx) -> None:
        self._lora_context = ctx

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
        """Quantize and Dispatch Topk Weights and Topk Ids."""

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        # When LoRA is active, dispatch the per-token LoRA id along with
        # hidden_states so every rank receives the correct mapping for the
        # tokens it ends up processing. The punica_wrapper stores indices as
        # int64 but the moe_lora_align_block_size kernel expects int32, so
        # pull the pre-cast view from token_mapping_meta.
        lora_ctx = self._lora_context
        local_token_lora_mapping = None
        if lora_ctx is not None:
            local_token_lora_mapping = (
                lora_ctx.punica_wrapper.token_mapping_meta.token_lora_mapping[
                    : a1.shape[0]
                ]
            )

        extra_tensors: list[torch.Tensor] | None = None
        if scales is not None:
            extra_tensors = list(scales)
        if local_token_lora_mapping is not None:
            if extra_tensors is None:
                extra_tensors = []
            extra_tensors.append(local_token_lora_mapping)

        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

        if extra_tensors is None:
            assert len(res) == 3
            a1q, topk_weights, topk_ids = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 4
            a1q, topk_weights, topk_ids, gathered_extras = res
            gathered_extras = list(gathered_extras)
            if local_token_lora_mapping is not None:
                dispatched_lora_mapping = gathered_extras.pop()
                assert lora_ctx is not None
                lora_ctx.local_token_lora_mapping = dispatched_lora_mapping
            if scales is not None:
                assert len(gathered_extras) == 1
                a1q_scale = gathered_extras[0]
            else:
                a1q_scale = a1q_scale_orig

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


def make_moe_prepare_and_finalize_naive_dp_ep(
    is_sequence_parallel: bool = False,
    num_dispatchers: int = 1,
) -> MoEPrepareAndFinalizeNaiveDPEPModular:
    return MoEPrepareAndFinalizeNaiveDPEPModular(
        is_sequence_parallel=is_sequence_parallel,
        num_dispatchers=num_dispatchers,
    )
