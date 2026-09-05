# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
A QuantizedActivation is a pre-quantized activation produced by a fused kernel
and consumed directly by a linear layer, letting the layer skip its own input
quantization. A linear advertises the key its kernel can consume, and the
accessor for the scales that key needs, via expose_input_quant_key; the kernel
validates and reads the activation via as_quantized_activation.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


@dataclass
class QuantizedActivation:
    """A quantized activation paired with its scale and original metadata.

    The quant_key describes how data and scale are to be interpreted (dtype,
    scale granularity, value packing). Details the key does not capture, such
    as blockscale layout or activation padding, must follow the consumer
    kernel's convention.

    TODO(mgoin): Encode layout and padding requirements in the contract so
    producers can match consumer kernels without relying on convention.
    """

    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey


@dataclass(frozen=True)
class InputQuantScales:
    """Consumer scales needed by an upstream quantization kernel.

    static_scale is the FP8 quantization divisor (and dequantization scale).
    global_scale_inv is the NVFP4 global quantization multiplier. Dynamic
    per-token or per-block scales are produced by the quantizer, not stored here.
    """

    static_scale: torch.Tensor | None = None
    global_scale_inv: torch.Tensor | None = None


def expose_input_quant_key(layer: torch.nn.Module, kernel) -> None:
    """Advertise the kernel's pre-quantized input key and scale accessor, if any.

    This is the bridge from a kernel's input_quant_key() to the
    layer.input_quant_key attribute that fusion call sites read. The attribute
    is left unset when the kernel quantizes its own input, so non-supporting
    backends never receive a QuantizedActivation.

    Scales are resolved through the kernel at use time: weight processing may
    create or replace the parameters after this bridge is called.
    """
    key = kernel.input_quant_key()
    if key is not None:
        layer.input_quant_key = key
        layer._input_quant_scales = kernel.input_quant_scales


def get_input_quant_scales(layer: torch.nn.Module) -> InputQuantScales:
    """Read current scales from a layer advertising pre-quantized input support."""
    return layer._input_quant_scales(layer)


def as_quantized_activation(
    x: "torch.Tensor | QuantizedActivation", expected_key: QuantKey | None
) -> "QuantizedActivation | None":
    """Validate and narrow a pre-quantized activation for a consumer kernel.

    Returns the QuantizedActivation when x is one whose key matches the
    kernel's declared expected_key, and None when x is a plain tensor (the
    caller quantizes in-kernel). Raises on a key mismatch so a wrongly routed
    activation fails loudly instead of being silently re-quantized.
    """
    if not isinstance(x, QuantizedActivation):
        return None
    assert x.quant_key == expected_key, (
        f"QuantizedActivation key {x.quant_key} != consumer kernel "
        f"input_quant_key {expected_key}"
    )
    return x
