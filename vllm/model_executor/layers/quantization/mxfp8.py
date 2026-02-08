# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MXFP8 (Microscaling FP8) Quantization for Linear layers.

This module implements online MXFP8 quantization with dynamic activation
scaling. MXFP8 uses block-based scaling where each block of 32 elements
shares a single scale factor (torch.float8_e8m0fnu format).

Note: MXFP8 with native cuBLAS MX GEMM requires SM100+ (Blackwell).
"""

from typing import Any, Optional

import torch
from torch.nn import Module

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import ModelWeightParameter

logger = init_logger(__name__)

MIN_M_REQUIRED = 32


def _mxfp8_linear_forward_impl(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    w_scale_blocked: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """MXFP8 linear forward pass implementation.

    Args:
        x: Input tensor of shape [..., K] in bfloat16.
        weight_fp8: Quantized weight tensor of shape [N, K] in float8_e4m3fn.
        w_scale_blocked: Weight scale in cuBLAS blocked layout.
        bias: Optional bias tensor of shape [N].

    Returns:
        Output tensor of shape [..., N] in bfloat16.
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    M = x.numel() // K
    N = weight_fp8.shape[0]

    # Reshape input to 2D: [..., K] -> [M, K]
    x_2d = x.reshape(M, K)

    # PAD M dimension to minimum of 32 for cuBLAS bias support
    # (cuBLAS MX GEMM with bias requires M >= 17)
    # Always pad/unpad to keep control flow static for torch.compile/cudagraph
    padded_M = max(M, MIN_M_REQUIRED)
    padded_shape = [padded_M, K]
    padded_x = torch.zeros(padded_shape, device=x_2d.device, dtype=x_2d.dtype)
    padded_x[0:M, ...].copy_(x_2d)

    # Quantize input activation to MXFP8 with swizzled scale layout
    x_fp8, x_scale_blocked = mxfp8_quantize(padded_x)

    # Assertions for contiguity (matching torchao pattern)
    assert x_fp8.is_contiguous(), "Input must be contiguous"
    assert weight_fp8.is_contiguous(), "Weight must be contiguous"

    # Use torch._scaled_mm with block scales
    # a (x_fp8): [M_padded, K] row-major (contiguous)
    # b (weight_fp8.t()): [K, N] column-major (weight_fp8 is [N, K] row-major,
    #   so weight_fp8.t() is [K, N] with stride [1, K] = column-major)
    output = torch._scaled_mm(
        x_fp8,
        weight_fp8.t(),  # [K, N] column-major
        x_scale_blocked,
        w_scale_blocked,
        bias=bias,
        out_dtype=torch.bfloat16,
    )

    # Remove padding from output (always slice to keep control flow static)
    output = output[0:M, ...]

    output_shape = orig_shape[:-1] + (N,)
    output = output.view(*output_shape)

    return output


class Mxfp8Config(QuantizationConfig):
    """Config class for MXFP8 (Microscaling FP8) quantization.

    MXFP8 uses block-based quantization where each block of 32 elements
    shares a single torch.float8_e8m0fnu scale factor. This provides a balance
    between per-tensor and per-channel quantization granularity.

    Args:
        ignored_layers: List of layer name patterns to skip quantization.
    """

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # MXFP8 with native cuBLAS MX GEMM requires SM100+ (Blackwell)
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Mxfp8Config":
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(ignored_layers=ignored_layers)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            return Mxfp8LinearMethod(self)
        return None


class Mxfp8LinearMethod(LinearMethodBase):
    """Linear method for MXFP8 online quantization.

    Implements online (dynamic) MXFP8 quantization where:
    - Weights are quantized to torch.float8_e4m3fn with block-wise
      torch.float8_e8m0fnu scales
    - Activations are dynamically quantized at runtime

    This is the simplest MXFP8 implementation for linear layers.

    Args:
        quant_config: The MXFP8 quantization config.
    """

    def __init__(self, quant_config: Mxfp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weight parameters for MXFP8 linear layer.

        For online quantization, weights are stored in the original dtype
        and quantized after loading.

        Note: This implementation only supports online quantization.
        Pre-quantized MXFP8 checkpoints are not supported.
        """
        # Assert bfloat16 for online quantization
        assert params_dtype == torch.bfloat16, (
            "MXFP8 only supports bfloat16 params_dtype for online quantization. "
            f"Got params_dtype={params_dtype}"
        )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # Create weight in original dtype for online quantization
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: Module) -> None:
        """Quantize weights to MXFP8 format after loading.

        Converts weights from original dtype to torch.float8_e4m3fn with
        block-wise torch.float8_e8m0fnu scale factors. Weight scales are
        pre-converted to cuBLAS blocked layout for efficient runtime usage.

        Weight is stored as [N, K] row-major. When passed to _scaled_mm,
        we use weight.t() which gives [K, N] column-major (as required by cuBLAS).
        """
        weight = layer.weight.data

        # Ensure weight is contiguous before quantization
        weight = weight.contiguous()

        # Quantize weight to MXFP8 format with swizzled scale layout
        weight_fp8, w_scale_blocked = mxfp8_quantize(weight)

        # Store weight as [N, K] row-major (standard contiguous layout)
        # When we call weight.t() in apply(), we get [K, N] column-major
        # which is what cuBLAS _scaled_mm expects for the second argument
        layer.weight = torch.nn.Parameter(weight_fp8, requires_grad=False)
        # Store pre-blocked weight scale in torch.float8_e8m0fnu format
        layer.weight_scale = torch.nn.Parameter(w_scale_blocked, requires_grad=False)
        layer.orig_dtype = layer.orig_dtype

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply MXFP8 quantized linear operation.

        Quantizes input activation dynamically and performs FP8 matmul
        using torch._scaled_mm with block scales.

        Note: Requires SM100+ (Blackwell) for native cuBLAS MX GEMM support.

        Args:
            layer: The linear layer with quantized weights.
            x: Input tensor of shape [..., K].
            bias: Optional bias tensor.

        Returns:
            Output tensor of shape [..., N].
        """
        return _mxfp8_linear_forward_impl(
            x,
            layer.weight,
            layer.weight_scale,
            bias,
        )
