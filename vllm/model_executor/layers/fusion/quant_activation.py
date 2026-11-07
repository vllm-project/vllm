# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A QuantizedActivation is a pre-quantized activation produced by a fused kernel
and consumed directly by a linear layer, letting the layer skip its own input
quantization. A linear advertises the key its kernel can consume via
expose_input_quant_key; the kernel validates and reads the activation via
as_quantized_activation. Producers query the effective capability through
get_input_quant_key, which hides the key when another consumer branch needs the
original activation.
"""

from dataclasses import dataclass, replace

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
)


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

    def weak_ref(self) -> "QuantizedActivation":
        """Return a copy with non-owning tensor references for CUDA graph replay."""
        from vllm.utils.torch_utils import weak_ref_tensor

        return replace(
            self,
            data=weak_ref_tensor(self.data),
            scale=weak_ref_tensor(self.scale),
        )


def expose_input_quant_key(
    layer: torch.nn.Module,
    kernel,
    scale_attr: str | None = None,
) -> None:
    """Store the kernel's pre-quantized input key and scale attribute on the layer.

    This is the bridge from a kernel's input_quant_key() to the layer capability
    that fusion call sites read through get_input_quant_key and
    get_input_quant_scale. The raw key is left unset when the kernel quantizes
    its own input.
    """
    key = kernel.input_quant_key()
    if key is not None:
        layer._input_quant_key = key
        if scale_attr is not None:
            layer._input_quant_scale_attr = scale_attr


def get_input_quant_key(layer: torch.nn.Module) -> QuantKey | None:
    """Return the pre-quantized input key the complete layer can consume."""
    if getattr(layer, "requires_unquantized_input", False):
        return None
    return getattr(layer, "_input_quant_key", None)


def get_input_quant_scale(layer: torch.nn.Module) -> torch.Tensor | None:
    """Return the consumer's static or global input quantization scale, if any.

    Producers call this instead of reaching into kernel-specific attributes
    like ``layer.input_scale`` or ``layer.input_global_scale_inv``. Because
    scales are finalized in ``process_weights_after_loading`` (after
    ``expose_input_quant_key`` runs in ``create_weights``), the scale tensor
    is resolved dynamically from the layer.
    """
    key = get_input_quant_key(layer)
    if key is None:
        return None
    scale_attr = getattr(layer, "_input_quant_scale_attr", None)
    if scale_attr is not None:
        return getattr(layer, scale_attr, None)
    if key == kNvfp4Dynamic:
        return getattr(layer, "input_global_scale_inv", None)
    if key.scale.static:
        return getattr(layer, "input_scale", None)
    return None


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
