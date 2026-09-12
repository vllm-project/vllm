# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM120 W6A8 linear kernel backed by the optional mxfp6 package."""

import importlib
from types import ModuleType

import torch
import torch.nn.functional as F
from packaging.version import InvalidVersion, Version
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .base import MxFp6LinearKernel, MxFp6LinearLayerConfig
from .emulation import EmulationMxfp6LinearKernel

logger = init_logger(__name__)

_REQUIRED_API = ("is_available", "load_library", "pack_scales")
_MINIMUM_VERSION = Version("0.2.1")


def _import_mxfp6() -> ModuleType:
    return importlib.import_module("mxfp6")


def is_mxfp6_sm120_available() -> bool:
    """Return whether the optional native W6A8 extension can run."""
    if not current_platform.is_cuda():
        return False
    if not current_platform.is_device_capability(120):
        return False

    try:
        mxfp6 = _import_mxfp6()
        if not all(hasattr(mxfp6, name) for name in _REQUIRED_API):
            return False
        if Version(mxfp6.__version__) < _MINIMUM_VERSION:
            return False
        return bool(mxfp6.is_available()) and hasattr(torch.ops.mxfp6, "gemm_w6a8")
    except (AttributeError, ImportError, InvalidVersion, OSError, RuntimeError):
        return False


def _mxfp6_sm120_gemm_impl(
    quantized_x: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    mxfp6 = _import_mxfp6()
    mxfp6.load_library()
    rows, input_features = quantized_x.shape
    return torch.ops.mxfp6.gemm_w6a8(
        quantized_x.view(torch.uint8),
        weight,
        input_scale,
        weight_scale,
        rows,
        output_features,
        input_features,
        1.0,
        output_dtype,
    )


def _mxfp6_sm120_gemm_fake(
    quantized_x: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    del input_scale, weight, weight_scale
    return torch.empty(
        (quantized_x.shape[0], output_features),
        dtype=output_dtype,
        device=quantized_x.device,
    )


direct_register_custom_op(
    op_name="mxfp6_sm120_gemm",
    op_func=_mxfp6_sm120_gemm_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_gemm_fake,
)


def mxfp6_sm120_gemm(
    quantized_x: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Run native W6A8 GEMM with pre-quantized MXFP8 activations."""
    return torch.ops.vllm.mxfp6_sm120_gemm(
        quantized_x,
        input_scale,
        weight,
        weight_scale,
        output_features,
        output_dtype,
    )


class Mxfp6Sm120LinearKernel(MxFp6LinearKernel):
    """MXFP6 E3M2 weight and dynamic MXFP8 activation GEMM on SM120."""

    def __init__(self, config: MxFp6LinearLayerConfig) -> None:
        super().__init__(config)
        self.emulation = EmulationMxfp6LinearKernel(config)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is not None and compute_capability != 120:
            return False, "requires SM120"
        if not is_mxfp6_sm120_available():
            return False, ("requires an SM120 GPU and the mxfp6-sm120 native extension")
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp6LinearLayerConfig) -> tuple[bool, str | None]:
        if config.weight_quant_key != kMxfp6E3M2Static:
            return False, "only supports static MXFP6 E3M2 weights"
        if config.activation_quant_key != kMxfp8Dynamic:
            return False, "only supports dynamic MXFP8 E4M3 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_mxfp6_sm120_processed", False):
            return

        weight = layer.weight.data
        weight_scale = layer.weight_scale.data
        if weight.ndim != 2 or weight_scale.ndim != 2:
            raise ValueError("mxfp6-sm120 weights and scales must both be 2D")
        if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8:
            raise ValueError("mxfp6-sm120 weights and scales must both be uint8")

        output_features, scale_columns = weight_scale.shape
        input_features = scale_columns * 32
        expected_packed_columns = input_features * 3 // 4
        if weight.shape != (output_features, expected_packed_columns):
            raise ValueError(
                "mxfp6-sm120 weight shape does not match its scale tensor: "
                f"weight={tuple(weight.shape)}, scale={tuple(weight_scale.shape)}"
            )

        layer._mxfp6_sm120_output_features = output_features
        layer._mxfp6_sm120_input_features = input_features
        layer._mxfp6_sm120_padded_input_features = (input_features + 127) // 128 * 128
        layer._mxfp6_sm120_native = output_features % 8 == 0
        if not layer._mxfp6_sm120_native:
            logger.warning_once(
                "mxfp6-sm120 requires N divisible by 8, but received N=%d. "
                "Falling back to emulation.",
                output_features,
            )
            self.emulation.process_weights_after_loading(layer)
            layer._mxfp6_sm120_processed = True
            return

        padded_input_features = layer._mxfp6_sm120_padded_input_features
        if padded_input_features != input_features:
            padded_weight = torch.zeros(
                (output_features, padded_input_features * 3 // 4),
                dtype=weight.dtype,
                device=weight.device,
            )
            padded_weight[:, : weight.shape[1]] = weight
            weight = padded_weight

            padded_scale = torch.full(
                (output_features, padded_input_features // 32),
                127,
                dtype=weight_scale.dtype,
                device=weight_scale.device,
            )
            padded_scale[:, : weight_scale.shape[1]] = weight_scale
            weight_scale = padded_scale

        mxfp6 = _import_mxfp6()
        packed_scale = mxfp6.pack_scales(weight_scale.contiguous())
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(packed_scale, requires_grad=False)
        layer._mxfp6_sm120_processed = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not layer._mxfp6_sm120_native:
            return self.emulation.apply_weights(layer, x, bias)

        input_features = layer._mxfp6_sm120_input_features
        padded_input_features = layer._mxfp6_sm120_padded_input_features
        output_features = layer._mxfp6_sm120_output_features
        input_shape = x.shape
        input_2d = x.reshape(-1, input_features).contiguous()
        if padded_input_features != input_features:
            input_2d = F.pad(input_2d, (0, padded_input_features - input_features))
        quantized_x, input_scale = mxfp8_e4m3_quantize(
            input_2d, is_sf_swizzled_layout=True
        )
        output = mxfp6_sm120_gemm(
            quantized_x,
            input_scale,
            layer.weight,
            layer.weight_scale,
            output_features,
            x.dtype,
        )
        if bias is not None:
            output = output + bias
        return output.view(*input_shape[:-1], output_features)
