# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
A ``QuantizedActivation`` is a pre-quantized activation produced by a fused kernel
and consumed directly by a linear, so the module skips its own input quantization.
A linear advertises the key its kernel can consume via ``expose_input_quant_key``;
the kernel validates and reads the activation via ``as_quantized_activation``.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


@dataclass
class QuantizedActivation:
    data: torch.Tensor
    scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: torch.Size
    quant_key: QuantKey


def expose_input_quant_key(layer: torch.nn.Module, kernel) -> None:
    """Advertise the kernel's pre-quantized input key on ``layer``, if any.

    The single bridge from a kernel's ``input_quant_key()`` to the
    ``layer.input_quant_key`` the fusion helper reads. Left unset when the kernel
    quantizes its own input, so non-supporting backends never receive a
    ``QuantizedActivation``.
    """
    key = kernel.input_quant_key()
    if key is not None:
        layer.input_quant_key = key


def as_quantized_activation(
    x: "torch.Tensor | QuantizedActivation", expected_key: QuantKey | None
) -> "QuantizedActivation | None":
    """Validate and narrow a pre-quantized activation for a consumer kernel.

    Returns the ``QuantizedActivation`` when ``x`` is one whose key matches the
    kernel's declared ``expected_key``; returns ``None`` when ``x`` is a plain
    tensor (the caller quantizes in-kernel). The caller then reads whichever
    fields its ``quant_key`` defines. Raises on a key mismatch so a wrongly
    routed activation fails loudly instead of being silently re-quantized.
    """
    if not isinstance(x, QuantizedActivation):
        return None
    assert x.quant_key == expected_key, (
        f"QuantizedActivation key {x.quant_key} != consumer kernel "
        f"input_quant_key {expected_key}"
    )
    return x
