# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_cutedsl
from vllm.utils.math_utils import round_up

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

logger = init_logger(__name__)


def _check_mm_mxfp8_shape(weight_shape: tuple[int, int]) -> tuple[bool, str | None]:
    N, K = weight_shape
    if K < 128:
        return False, f"mm_mxfp8 requires K >= 128, got K={K}."
    if K % MXFP8_BLOCK_SIZE != 0:
        return (
            False,
            f"mm_mxfp8 requires K to be divisible by {MXFP8_BLOCK_SIZE}, got K={K}.",
        )
    if N < 128:
        return False, f"mm_mxfp8 requires N >= 128, got N={N}."
    return True, None


def _pad_n(
    weight: torch.Tensor, weight_scale: torch.Tensor, multiple: int, kernel: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad N, the rows of `weight` [N, K] and of its unswizzled scale
    [N, K / 32], to a multiple of `multiple`. The padded rows compute zeros,
    which the kernel slices off its output."""
    n = weight.shape[0]
    padded_n = round_up(n, multiple)
    if padded_n == n:
        return weight, weight_scale
    logger.info_once(
        "Padding N from %d to %d for the %s MXFP8 linear kernel.", n, padded_n, kernel
    )
    padded_weight = weight.new_zeros((padded_n, weight.shape[1]))
    padded_weight[:n] = weight
    padded_scale = weight_scale.new_zeros((padded_n, weight_scale.shape[1]))
    padded_scale[:n] = weight_scale
    return padded_weight, padded_scale


class FlashInferCutlassMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CUTLASS (SM100+)."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda() and current_platform.has_device_capability(100)
        ):
            return False, "requires >=sm_100 (Blackwell)"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        can_implement, reason = _check_mm_mxfp8_shape(c.weight_shape)
        if not can_implement:
            return False, reason
        N, _ = c.weight_shape
        if current_platform.is_device_capability_family(120) and N % 32 != 0:
            return (
                False,
                f"mm_mxfp8 on SM12x requires N to be divisible by 32, got N={N}.",
            )
        return True, None

    def input_quant_key(self) -> QuantKey:
        # Activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic

    # The CUTLASS MXFP8 GEMM requires N (output features) to be a multiple of 32.
    _N_ALIGNMENT = 32

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight, weight_scale_2d = _pad_n(
            weight, weight_scale_2d, self._N_ALIGNMENT, "FlashInfer CUTLASS"
        )
        weight_scale_swizzled = swizzle_mxfp8_scale(
            weight_scale_2d, M=weight.shape[0], K=K
        )

        replace_parameter(layer, "weight", weight.contiguous())
        replace_parameter(layer, "weight_scale", weight_scale_swizzled.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # [N padded to a multiple of 32, K]
        weight_scale = layer.weight_scale
        padded_n, K = weight.shape
        N = layer.output_size_per_partition

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = layer.params_dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                x.view(-1, K), is_sf_swizzled_layout=True
            )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cutlass",
        )
        if padded_n != N:
            output = output[:, :N].contiguous()

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferCutedslMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer CuTe-DSL (SM100/SM103)."""

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires sm_100/sm_103 (Blackwell)"
        if not has_flashinfer_cutedsl():
            return False, "requires FlashInfer CuTe-DSL module"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return _check_mm_mxfp8_shape(c.weight_shape)

    def input_quant_key(self) -> QuantKey:
        # Activations use FlashInfer's F8_128x4 swizzled scale layout.
        return kMxfp8Dynamic

    # The CuTe-DSL MXFP8 GEMM stores its bf16/fp16 output with TMA, which needs
    # 16-byte aligned rows, so N must be a multiple of 8.
    _N_ALIGNMENT = 8

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale_2d = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight, weight_scale_2d = _pad_n(
            weight, weight_scale_2d, self._N_ALIGNMENT, "FlashInfer CuTe-DSL"
        )
        weight_scale_swizzled = swizzle_mxfp8_scale(
            weight_scale_2d, M=weight.shape[0], K=K
        )

        # Store weight column-major [K, N] as mm_mxfp8 expects for operand B.
        replace_parameter(layer, "weight", weight.contiguous().t())
        replace_parameter(layer, "weight_scale", weight_scale_swizzled.contiguous())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight  # [K, N padded to a multiple of 8], column-major
        weight_scale = layer.weight_scale
        K, padded_n = weight.shape
        N = layer.output_size_per_partition

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            input_mxfp8, input_scale = qa.data, qa.scale
            out_dtype, input_shape = qa.orig_dtype, qa.orig_shape
        else:
            assert isinstance(x, torch.Tensor)
            out_dtype, input_shape = layer.params_dtype, x.shape
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                x.view(-1, K), is_sf_swizzled_layout=True
            )

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight,
            input_scale,
            weight_scale,
            out_dtype=out_dtype,
            backend="cute-dsl",
        )
        if padded_n != N:
            output = output[:, :N].contiguous()

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)


class FlashInferTrtllmMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 GEMM via FlashInfer's TensorRT-LLM wrapper."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False, "requires SM100-family GPU"
        if not has_flashinfer():
            return False, "requires FlashInfer"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

        if hasattr(layer, "_mxfp8_trtllm_output_size") and layer.weight_scale.ndim == 1:
            return

        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        if K % 256 != 0:
            raise ValueError(
                f"FlashInfer TRTLLM MXFP8 requires K to be divisible by 256, got K={K}."
            )

        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        weight, weight_scale = _pad_n(weight, weight_scale, 128, "FlashInfer TRT-LLM")
        padded_n = weight.shape[0]

        replace_parameter(
            layer,
            "weight",
            shuffle_matrix_a(weight.contiguous(), 128).reshape(padded_n, K),
        )
        replace_parameter(
            layer,
            "weight_scale",
            shuffle_matrix_sf_a(
                weight_scale,
                128,
                num_elts_per_sf=MXFP8_BLOCK_SIZE,
            ).reshape(-1),
        )
        layer._mxfp8_trtllm_output_size = N

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dtype == torch.bfloat16, (
            f"FlashInfer TRTLLM MXFP8 requires bfloat16 activations, got {x.dtype}."
        )

        weight = layer.weight  # shuffled [padded N, K]
        weight_scale = layer.weight_scale
        _, K = weight.shape
        output_size = layer._mxfp8_trtllm_output_size
        input_shape = x.shape
        input_2d = x.view(-1, K)

        input_mxfp8, input_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_8x4(
            input_2d
        )
        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=x.dtype,
            backend="trtllm",
            use_8x4_sf_layout=True,
        )
        if output.shape[-1] != output_size:
            output = output[:, :output_size].contiguous()

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], output_size)
        return output.view(output_shape)
