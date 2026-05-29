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
        is_scale_swizzled=quant_config.is_scale_swizzled,
        mx_alignment=quant_config.mx_alignment,
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

    def supports_prepared_inputs(self) -> bool:
        return True

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
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, a1q_scale = _quantize_input(a1, quant_config, defer_input_quant)

        return a1q, a1q_scale, None, None, None

    def prepare_prepared_input(
        self,
        a1: torch.Tensor,
        prepared_a1q: torch.Tensor,
        prepared_a1q_scale: torch.Tensor | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if (
            defer_input_quant
            or apply_router_weight_on_input
            or not quant_config.is_quantized
        ):
            return self.prepare(
                a1,
                topk_weights,
                topk_ids,
                num_experts,
                expert_map,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant,
            )

        if prepared_a1q.shape != a1.shape:
            raise ValueError(
                "prepared_a1q must have the same shape as the MoE input, "
                f"got {prepared_a1q.shape} and {a1.shape}."
            )
        if prepared_a1q.device != a1.device:
            raise ValueError(
                "prepared_a1q must be on the same device as the MoE input, "
                f"got {prepared_a1q.device} and {a1.device}."
            )
        if prepared_a1q_scale is not None and prepared_a1q_scale.device != a1.device:
            raise ValueError(
                "prepared_a1q_scale must be on the same device as the MoE input, "
                f"got {prepared_a1q_scale.device} and {a1.device}."
            )

        block_shape = quant_config.block_shape
        if block_shape is not None:
            block_k = block_shape[1]
            expected_scale_shape = (
                *a1.shape[:-1],
                (a1.shape[-1] + block_k - 1) // block_k,
            )
            if prepared_a1q_scale is None:
                raise ValueError(
                    "prepared_a1q_scale is required for block quantized MoE."
                )
            if prepared_a1q_scale.shape != expected_scale_shape:
                raise ValueError(
                    "prepared_a1q_scale has an unexpected shape, got "
                    f"{prepared_a1q_scale.shape}, expected {expected_scale_shape}."
                )

        return prepared_a1q, prepared_a1q_scale, None, None, None

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
