# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
)
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    flashinfer_scaled_fp8_mm,
    has_flashinfer,
    is_flashinfer_fp8_blockscale_gemm_supported,
)
from vllm.utils.torch_utils import direct_register_custom_op

from ..base import DynamicMMLinearKernel, FP8Params
from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from .cutlass import CutlassFp8BlockScaledMMKernel
from .deep_gemm import DeepGemmFp8BlockScaledMMKernel
from .ScaledMMLinearKernel import (
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


class FlashInferFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.input_quant_op = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=is_deep_gemm_e8m0_used(),
        )

    @classmethod
    def ordered_fallback_kernels(cls) -> list[type["Fp8BlockScaledMMLinearKernel"]]:
        # TODO This import is to avoid circular import
        # this import can be global
        # after all scaled MM kernels inherit from base
        from .triton import TritonFp8BlockScaledMMKernel

        return [
            DeepGemmFp8BlockScaledMMKernel,
            CutlassFp8BlockScaledMMKernel,
            TritonFp8BlockScaledMMKernel,
        ]

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "only cuda devices are supported."

        if not is_flashinfer_fp8_blockscale_gemm_supported():
            return False, "FlashInfer block-scale FP8 GEMM is not available."

        return True, None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run FlashInfer FP8 block-scale GEMM.

        This backend uses TensorRT-LLM's FP8 block-scale GEMM kernels
        and supports FP8+FP8 (W8A8 full quantization) on SM90+ (Hopper).
        """

        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale_inv = params.weight_scale_inv
        input_scale = params.input_scale

        # View input as 2D matrix for fp8 methods
        input_2d = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], weight.shape[0]]
        output_dtype = x.dtype

        output = self.apply_block_scaled_mm(
            input_2d, weight, output_dtype, input_scale, weight_scale_inv
        )

        if bias is not None:
            output = output + bias

        return output.to(dtype=x.dtype).view(*output_shape)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run FlashInfer FP8 block-scale GEMM.

        This backend uses TensorRT-LLM's FP8 block-scale GEMM kernels
        and supports FP8+FP8 (W8A8 full quantization) on SM90+ (Hopper).
        """

        return torch.ops.vllm.flashinfer_fp8_blockscale_gemm(
            A,  # BF16 input
            B,  # FP8 weight
            Bs,  # Weight scales
        )


class FlashInferFp8DeepGEMMDynamicBlockScaledKernel(
    DynamicMMLinearKernel[
        FP8ScaledMMLinearLayerConfig,
        FP8Params,
        FlashInferFp8BlockScaledMMKernel,
        DeepGemmFp8BlockScaledMMKernel,
    ]
):
    """
    Conditional FlashInfer FP8 blockscale GEMM with batch-size-dependent selection.

    This function switches between two optimized kernels based on the input batch size:
    - For small batches (M < 32): Uses FlashInfer's DeepGEMM swapAB optimization.
    - For larger batches (M >= 32): Uses the official DeepGEMM kernel.

    The conditional logic must use torch.cond() instead of a simple if-else statement
    to maintain compatibility with torch.compile graph compilation.

    This batch-size-dependent selection is essential for maintaining model accuracy.
    Benchmarks on GSM8K show a significant accuracy gap (88% vs 95%) for DeepSeek-V3.1
    when using FlashInfer's DeepGEMM on M>=32. The M < 32 strategy fixes the accurracy
    drop.

    """

    base_type = type[FlashInferFp8BlockScaledMMKernel]
    fallback_type = type[DeepGemmFp8BlockScaledMMKernel]

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # deepgemm might require post processing.
        # both flashinfer and deepgemm kernels
        # work on the same layer parameter tensor layouts
        self.fallback.process_weights_after_loading(layer)

    def predicate(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ):
        input_2d = x.view(-1, x.shape[-1])
        return input_2d.shape[0] < 32


def _flashinfer_fp8_blockscale_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return flashinfer_fp8_blockscale_gemm(
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
    )


def _flashinfer_fp8_blockscale_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Required fake/meta implementation for torch.compile graph tracing.
    """
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "flashinfer_fp8_blockscale_gemm",
    _flashinfer_fp8_blockscale_gemm_impl,
    fake_impl=_flashinfer_fp8_blockscale_gemm_fake,
)
