# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
A QuantizedActivation is a pre-quantized activation produced by a fused kernel
and consumed directly by a linear layer, letting the layer skip its own input
quantization. A linear advertises the key its kernel can consume via
expose_input_quant_key; the kernel validates and reads the activation via
as_quantized_activation.
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


def expose_input_quant_key(layer: torch.nn.Module, kernel) -> None:
    """Advertise the kernel's pre-quantized input key on the layer, if any.

    This is the bridge from a kernel's input_quant_key() to the
    layer.input_quant_key attribute that fusion call sites read. The attribute
    is left unset when the kernel quantizes its own input, so non-supporting
    backends never receive a QuantizedActivation.

    TODO(mgoin): Producers also need the consumer's quantization scales (e.g.
    static input scale, global scale). Expose those here as well so producers
    do not reach into kernel-specific layer attributes.
    """
    key = kernel.input_quant_key()
    if key is not None:
        layer.input_quant_key = key


def try_expose_input_quant_key(layer: torch.nn.Module) -> bool:
    """Try to expose input_quant_key on a layer by finding its kernel.

    This is a convenience function for model code that wants to enable
    manual activation+quant fusion without knowing the internal structure
    of the quantization method.

    Returns True if input_quant_key was set, False otherwise.
    """
    # Check common patterns for accessing the kernel
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is None:
        return False

    # FP8 linear methods store kernel as fp8_linear
    kernel = getattr(quant_method, "fp8_linear", None)
    if kernel is None:
        # NVFP4 and other methods store kernel as kernel
        kernel = getattr(quant_method, "kernel", None)
    if kernel is None:
        return False

    # Check if kernel supports input_quant_key
    input_quant_key_fn = getattr(kernel, "input_quant_key", None)
    if input_quant_key_fn is None:
        return False

    key = input_quant_key_fn()
    if key is not None:
        layer.input_quant_key = key
        return True
    return False


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
