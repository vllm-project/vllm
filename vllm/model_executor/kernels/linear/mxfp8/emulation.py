# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    dequant_mxfp8_to_bf16,
)

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class EmulationMxfp8LinearKernel(Mxfp8LinearKernel):
    """Software emulation fallback for MXFP8 (dequant to BF16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE

        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()

        # Dequantize MXFP8 -> BF16 ONCE here, at load time, so apply_weights runs
        # a plain BF16 linear with no per-step dequant -- i.e. run as if from a
        # BF16 checkpoint. The 1-byte MXFP8 weight is replaced by BF16 (2x its
        # size, but linear weights are small vs the MoE experts); the tiny E8M0
        # scale is kept for the dtype/ndim asserts but is otherwise unused.
        # Opt out (VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=0) to keep the MXFP8
        # weight and dequant per-step in apply_weights instead.
        import vllm.envs as envs

        if envs.VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD:
            weight = dequant_mxfp8_to_bf16(weight.contiguous(), weight_scale)
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        # Load-time dequant path: weights are already BF16/FP16 (>= 2-byte), so
        # run a plain linear -- no per-step dequant. (MXFP8 weights are 1-byte.)
        if weight.element_size() >= 2:
            # F.linear requires x and weight share a dtype; .to() is a no-op when
            # they already match (e.g. both BF16).
            output = torch.nn.functional.linear(x, weight.to(x.dtype), bias)
            return output.to(x.dtype)

        # Fallback: weights still in MXFP8 -- dequant on the fly (other archs /
        # if a future caller skips the load-time conversion above).
        weight_scale = layer.weight_scale
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Emulation backend requires {MXFP8_SCALE_DTYPE} "
                f"weight_scale dtype, got {weight_scale.dtype}."
            )
        if weight_scale.ndim != 2:
            raise ValueError(
                f"Emulation backend requires 2D weight_scale, "
                f"got {weight_scale.ndim}D. "
                f"Ensure process_weights_after_loading was called."
            )

        # Cast to x's dtype: dequant yields BF16, but F.linear needs both operands
        # to match (e.g. an FP16 model). No-op when x is already BF16.
        weight_bf16 = dequant_mxfp8_to_bf16(weight, weight_scale).to(x.dtype)
        output = torch.nn.functional.linear(x, weight_bf16, bias)
        return output.to(x.dtype)
