# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A QuantizedActivation is a pre-quantized activation produced by a fused kernel
and consumed directly by a linear layer, letting the layer skip its own input
quantization. A linear advertises the key its kernel can consume via
expose_input_quant_key; the kernel validates and reads the activation via
as_quantized_activation. Activation producers query get_fused_act_quant_key;
get_input_quant_key exposes the underlying consumer capability. Both hide the
key when another consumer branch needs the original activation.
"""

from dataclasses import dataclass, replace

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

    def weak_ref(self) -> "QuantizedActivation":
        """Return a copy with non-owning tensor references for CUDA graph replay."""
        from vllm.utils.torch_utils import weak_ref_tensor

        return replace(
            self,
            data=weak_ref_tensor(self.data),
            scale=weak_ref_tensor(self.scale),
        )


def expose_input_quant_key(layer: torch.nn.Module, kernel) -> None:
    """Store the kernel's input key and optional automatic-fusion policy.

    This is the bridge from a kernel's input_quant_key() to the layer capability
    that fusion call sites read through get_fused_act_quant_key. Re-exposure replaces
    both fields, including clearing stale capabilities for unsupported kernels.
    Kernels without an activation policy retain unrestricted producer selection.

    TODO(mgoin): Producers also need the consumer's quantization scales (e.g.
    static input scale, global scale). Expose those here as well so producers
    do not reach into kernel-specific layer attributes.
    """
    key = kernel.input_quant_key()
    get_types = getattr(kernel, "input_quant_activation_types", None)
    types = get_types() if key is not None and get_types is not None else None
    layer._input_quant_key = key
    layer._input_quant_activation_types = tuple(types) if types is not None else None


def get_input_quant_key(layer: torch.nn.Module) -> QuantKey | None:
    """Return the consumer key, without selecting an activation producer."""
    if getattr(layer, "requires_unquantized_input", False):
        return None
    return getattr(layer, "_input_quant_key", None)


def get_fused_act_quant_key(
    layer: torch.nn.Module, act_fn: torch.nn.Module
) -> QuantKey | None:
    """Return the key only for an allowed automatic activation producer."""
    allowed = getattr(layer, "_input_quant_activation_types", None)
    if allowed is not None and type(act_fn) not in allowed:
        return None
    return get_input_quant_key(layer)


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
