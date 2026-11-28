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
        is_scale_transposed=quant_config.is_scale_transposed,
        mx_alignment=quant_config.mx_alignment,
    )

    return a1q, a1q_scale


def _validate_mhc_fp8_pair(
    prequantized_input: torch.Tensor | None,
    prequantized_input_scale: torch.Tensor | None,
    bf16_input: torch.Tensor,
) -> None:
    """Validate a contiguous E4M3FN input and row-major FP32 g128 scales."""
    if prequantized_input is None and prequantized_input_scale is None:
        return
    if (prequantized_input is None) != (prequantized_input_scale is None):
        raise ValueError(
            "prequantized_input and prequantized_input_scale must be provided "
            "as a pair (both None = not provided)"
        )
    assert prequantized_input is not None and prequantized_input_scale is not None
    if bf16_input.dim() < 1:
        raise ValueError(
            "the BF16 main input must have at least one dimension for the "
            "mHC FP8 pair contract"
        )
    if bf16_input.dtype != torch.bfloat16 or not bf16_input.is_contiguous():
        raise ValueError(
            "the BF16 main input must be contiguous bfloat16 (got "
            f"{bf16_input.dtype}, contiguous={bf16_input.is_contiguous()})"
        )
    H = bf16_input.shape[-1]
    if H % 128 != 0:
        raise ValueError(
            f"prequantized mHC FP8 input requires hidden_size divisible by 128, got {H}"
        )
    if (
        tuple(prequantized_input.shape) != tuple(bf16_input.shape)
        or prequantized_input.dtype != torch.float8_e4m3fn
        or not prequantized_input.is_contiguous()
        or prequantized_input.device != bf16_input.device
    ):
        raise ValueError(
            "prequantized_input must be a contiguous float8_e4m3fn tensor "
            f"with the same shape/device as the BF16 input "
            f"(got {tuple(prequantized_input.shape)} "
            f"{prequantized_input.dtype} on {prequantized_input.device}, "
            f"expected {tuple(bf16_input.shape)} on {bf16_input.device})"
        )
    expected_scale_shape = tuple(bf16_input.shape[:-1]) + (H // 128,)
    if (
        tuple(prequantized_input_scale.shape) != expected_scale_shape
        or prequantized_input_scale.dtype != torch.float32
        or not prequantized_input_scale.is_contiguous()
        or prequantized_input_scale.device != bf16_input.device
    ):
        raise ValueError(
            "prequantized_input_scale must be a contiguous row-major FP32 "
            f"tensor of shape {expected_scale_shape} on "
            f"{bf16_input.device} (got "
            f"{tuple(prequantized_input_scale.shape)} "
            f"{prequantized_input_scale.dtype} on "
            f"{prequantized_input_scale.device})"
        )


def _quant_config_accepts_mhc_fp8_pair(quant_config) -> bool:
    """Check the activation format consumed by the FP8 prepare path."""
    from vllm.platforms import current_platform

    return (
        quant_config.quant_dtype == current_platform.fp8_dtype()
        and quant_config.block_shape == [1, 128]
        and not quant_config.is_scale_transposed
    )


class MoEPrepareAndFinalizeNoDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    def supports_deferred_moe_finalize(self) -> bool:
        # finalize() here is the top-k weight-and-reduce and nothing else,
        # which is exactly the work a deferring consumer takes over.
        return True

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
        *,
        prequantized_input: torch.Tensor | None = None,
        prequantized_input_scale: torch.Tensor | None = None,
    ) -> mk.PrepareResultType:
        if prequantized_input is not None or prequantized_input_scale is not None:
            # Reuse the upstream quantization for w13.
            _validate_mhc_fp8_pair(prequantized_input, prequantized_input_scale, a1)
            if apply_router_weight_on_input:
                raise ValueError(
                    "prequantized mHC FP8 input is incompatible with "
                    "apply_router_weight_on_input"
                )
            if quant_config is None or not _quant_config_accepts_mhc_fp8_pair(
                quant_config
            ):
                raise ValueError(
                    "prequantized mHC FP8 input requires the FP8 dynamic "
                    "group-128 recipe (got quant_config="
                    f"{type(quant_config).__name__ if quant_config else None})"
                )
            return prequantized_input, prequantized_input_scale, None, None, None

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
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
    def supports_deferred_moe_finalize(self) -> bool:
        return True

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
        a1_scale: torch.Tensor | None = None,
    ) -> mk.PrepareMonolithicResultType:
        if a1_scale is not None:
            # Already quantized by the caller, before the PCP all-gather.
            assert not defer_input_quant, "a quantized input cannot defer its quant"
            return a1, a1_scale, router_logits
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
