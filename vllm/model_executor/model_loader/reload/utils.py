# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import BoundArguments

import torch

from .types import LayerReloadingInfo, LayerTensors

__all__ = [
    "get_layer_tensors",
    "get_layer_params_buffers",
    "get_loadable_layer_tensors",
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


def get_loadable_layer_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get the tensors of a layer that a weight loader may write to, excluding
    SKIP_TENSORS (e.g. _expert_map), which reload never moves to meta."""
    from .meta import SKIP_TENSORS

    params, buffers = get_layer_params_buffers(layer)
    non_persistent = getattr(layer, "_non_persistent_buffers_set", set())

    # A non-persistent buffer is absent from ``state_dict`` and so is derived
    # state, not a checkpoint destination, unless it carries its own loader.
    loadable_buffers = {
        name: buffer
        for name, buffer in buffers.items()
        if name not in non_persistent
        or getattr(buffer, "weight_loader", None) is not None
    }
    return {
        name: tensor
        for name, tensor in (params | loadable_buffers).items()
        if name not in SKIP_TENSORS
    }


def has_device_tensors(bound_args: BoundArguments) -> bool:
    """
    Return True if the loaded weights exist on an accelerator device

    Args:
        bound_args: args to load weights

    Returns:
        True if weights are on accelerator device
    """
    return any(
        isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
        for value in bound_args.arguments.values()
    )


def get_info_size(info: LayerReloadingInfo) -> int:
    """
    Calculate the number of bytes used by loaded weights for a given layer

    Args:
        info: layerwise info to get size of

    Returns:
        number of bytes used by loaded weights
    """
    return sum(
        value.nbytes
        for _, args in info.loaded_weights
        for value in args.arguments.values()
        if isinstance(value, torch.Tensor) and value.device.type not in ("meta", "cpu")
    )
