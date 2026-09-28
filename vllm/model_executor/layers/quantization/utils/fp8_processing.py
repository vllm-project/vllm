# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeatable FP8 conversion, independent of parameter and kernel installation."""

from dataclasses import dataclass
from typing import Literal

import torch


def compute_fp8_moe_per_tensor_scales(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute CUTLASS/HPC scale values without installing runtime bindings."""
    return {
        "g1_alphas": w1_scale * a1_scale,
        "g2_alphas": w2_scale * a2_scale,
        "a1_gscale": 1.0 / a1_scale,
        "a2_gscale": 1.0 / a2_scale,
    }


def compute_fp8_moe_trtllm_scales(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    is_act_and_mul: bool,
) -> dict[str, torch.Tensor]:
    """Compute the per-tensor scales cached by TRTLLM monolithic experts."""
    g1 = (w1_scale * a1_scale).squeeze()
    return {
        "_g1_alphas": g1,
        "_g2_alphas": (w2_scale * a2_scale).squeeze(),
        "_g1_scale_c": (
            g1 / a2_scale if is_act_and_mul else torch.ones_like(g1) / a2_scale
        ),
    }


@dataclass(frozen=True)
class Fp8MoEWeights:
    """Local checkpoint inputs or converted outputs, never a live layer.

    Conversion may consume input tensors in place. Callers retaining checkpoint
    values must supply working copies. Runtime installation is the caller's job.
    """

    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_input_scale: torch.Tensor | None
    w2_input_scale: torch.Tensor | None


@dataclass(frozen=True)
class Fp8MoEProcessingPlan:
    """Cold-load structural decisions reused for each new set of weights.

    This plan deliberately contains neither tensors nor expert placement.
    Inputs have already been sharded using the current round's expert mapping.
    """

    backend: Literal[
        "deep_gemm",
        "flashinfer_cutlass",
        "flashinfer_trtllm",
        "triton",
        "vllm_cutlass",
        "hpc",
        "cpu",
        "aiter",
        "xpu",
    ]
    block_shape: tuple[int, ...] | None
    is_act_and_mul: bool
    is_gated: bool
    shard_size: int
    num_experts: int
    static_input: bool
    enable_eplb: bool
    use_e8m0: bool
    fnuz: bool = False

    def process(self, weights: Fp8MoEWeights) -> Fp8MoEWeights:
        """Convert fresh checkpoint-layout inputs without modifying a module."""
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            deepgemm_post_process_fp8_weight_block,
            process_fp8_input_tensor_strategy_moe,
            process_fp8_weight_tensor_strategy_moe,
        )

        w13, w2 = weights.w13, weights.w2
        s13, s2 = weights.w13_scale, weights.w2_scale
        a13, a2 = weights.w13_input_scale, weights.w2_input_scale
        if self.fnuz:
            from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
                normalize_e4m3fn_to_e4m3fnuz,
            )

            w13, s13, a13 = normalize_e4m3fn_to_e4m3fnuz(w13, s13, a13)
            w2, s2, a2 = normalize_e4m3fn_to_e4m3fnuz(w2, s2, a2)
        if self.static_input:
            assert a13 is not None and a2 is not None
            a13, a2 = process_fp8_input_tensor_strategy_moe(a13, a2, self.enable_eplb)
        if self.block_shape is None:
            w13, s13 = process_fp8_weight_tensor_strategy_moe(
                w13, s13, self.shard_size, self.num_experts, self.is_act_and_mul
            )

        if self.backend == "deep_gemm":
            assert self.block_shape is not None
            w13, s13 = deepgemm_post_process_fp8_weight_block(
                w13, s13, self.block_shape, self.use_e8m0
            )
            w2, s2 = deepgemm_post_process_fp8_weight_block(
                w2, s2, self.block_shape, self.use_e8m0
            )
        elif self.backend in ("flashinfer_cutlass", "flashinfer_trtllm"):
            from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
                convert_fp8_moe_weights_for_fi,
            )

            w13, w2, s13, s2 = convert_fp8_moe_weights_for_fi(
                w13,
                w2,
                s13,
                a13,
                s2,
                a2,
                block_quant=self.block_shape is not None,
                is_act_and_mul=self.is_act_and_mul,
                is_gated=self.is_gated,
                is_trtllm=self.backend == "flashinfer_trtllm",
            )
        elif self.backend == "cpu":
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                prepare_fp8_moe_layer_for_cpu,
            )

            w13, w2 = prepare_fp8_moe_layer_for_cpu(w13, w2)
        elif self.backend == "aiter":
            from vllm._aiter_ops import rocm_aiter_ops

            w13, w2 = rocm_aiter_ops.shuffle_weights(w13, w2)
            w13.is_shuffled = True
            w2.is_shuffled = True
        elif self.backend == "xpu":
            from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
                prepare_fp8_moe_layer_for_xpu,
            )

            w13, s13, w2, s2 = prepare_fp8_moe_layer_for_xpu(w13, s13, w2, s2)
        return Fp8MoEWeights(w13, w2, s13, s2, a13, a2)

    def derived_scales(self, weights: Fp8MoEWeights) -> dict[str, torch.Tensor]:
        """Refresh weight-dependent caches, never kernels or quant configs."""
        if self.block_shape is not None or self.backend not in (
            "flashinfer_cutlass",
            "flashinfer_trtllm",
            "hpc",
        ):
            return {}
        a1, a2 = weights.w13_input_scale, weights.w2_input_scale
        assert a1 is not None and a2 is not None
        if self.backend == "flashinfer_trtllm":
            return compute_fp8_moe_trtllm_scales(
                weights.w13_scale, weights.w2_scale, a1, a2, self.is_act_and_mul
            )
        return compute_fp8_moe_per_tensor_scales(
            weights.w13_scale, weights.w2_scale, a1, a2
        )
