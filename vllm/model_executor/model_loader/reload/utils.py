# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import BoundArguments

import torch

from .types import LayerReloadingInfo, LayerTensors

__all__ = [
    "get_layer_tensors",
    "get_layer_params_buffers",
    "get_layer_size",
    "has_device_tensors",
    "get_info_size",
]


def get_layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get all parameters and buffers from a module as a dict."""
    params, buffers = get_layer_params_buffers(layer)
    return params | buffers


def get_layer_params_buffers(layer: torch.nn.Module) -> LayerTensors:
    """Get all parameters and buffers of a module as a tuple of dicts."""
    return (
        {name: param for name, param in layer._parameters.items() if param is not None},
        {name: buffer for name, buffer in layer._buffers.items() if buffer is not None},
    )


def get_layer_size(layer: torch.nn.Module) -> int:
    """Calculate total number of elements across loadable tensors in a layer.

    Excludes SKIP_TENSORS (e.g. _expert_map) which are never moved to meta
    device and never loaded via weight_loader during layerwise reload.
    """
    from .meta import SKIP_TENSORS

    return sum(
        tensor.numel()
        for name, tensor in get_layer_tensors(layer).items()
        if name not in SKIP_TENSORS
    )


def has_device_tensors(bound_args: BoundArguments) -> bool:
    """
    Return True if the loaded weights exist on an accelerator device

    :param bound_args: args to load weights
    :return: True if weights are on accelerator device
    """
    return any(
        isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
        for value in bound_args.arguments.values()
    )


def get_info_size(info: LayerReloadingInfo) -> int:
    """
    Calculate the number of bytes used by loaded weights for a given layer

    :param info: layerwise info to get size of
    :return: number of bytes used by loaded weights
    """
    return sum(
        value.nbytes
        for _, args in info.loaded_weights
        for value in args.arguments.values()
        if isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
    )
