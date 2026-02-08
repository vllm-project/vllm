# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    flashinfer_scaled_fp8_mm,
    has_flashinfer,
)

from .ScaledMMLinearKernel import (
    FP4ScaledMMLinearKernel,
    FP4ScaledMMLinearLayerConfig,
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)


class FlashInferFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."

        if not has_flashinfer():
            return False, "requires FlashInfer to be installed."

        if compute_capability is not None and compute_capability < 100:
            return False, "requires compute capability 100 and above."

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return False, "requires per tensor activation and weight scales."

        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        return flashinfer_scaled_fp8_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )


class FlashInferFP4ScaledMMLinearKernel(FP4ScaledMMLinearKernel):
    """FlashInfer FP4 GEMM kernel implementation"""

    def __init__(
        self,
        c: FP4ScaledMMLinearLayerConfig,
        layer_param_names: Sequence[str],
        backend: str = "cutlass",
    ):
        """
        Args:
            c: Configuration for the FP4 layer
            layer_param_names: Names of layer parameters
            backend: FlashInfer backend variant ("cutlass", "trtllm", or "cudnn")
        """
        super().__init__(c, layer_param_names)
        self.backend = backend

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Requires CUDA."

        if compute_capability is not None and compute_capability < 100:
            return False, "NVFP4 requires compute capability of 10.0 (Blackwell)"

        if not has_flashinfer():
            return False, "FlashInfer not available"

        return True, None

    @classmethod
    def can_implement(cls, c: FP4ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def apply_fp4_mm(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
        input_scale_inv: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list[int],
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Apply FlashInfer FP4 matmul."""
        output = flashinfer_scaled_fp4_mm(
            x,
            weight,
            weight_scale,
            weight_global_scale,
            input_scale_inv,
            alpha,
            layer.output_size_per_partition,
            backend=self.backend,
        )

        if bias is not None:
            output = output + bias

        return output.view(*output_shape)
