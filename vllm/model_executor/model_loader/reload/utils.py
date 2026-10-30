# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import BoundArguments

import torch
from torch import nn

from .types import LayerReloadingInfo, LayerTensors

__all__ = [
    "get_layer_tensors",
    "get_layer_params_buffers",
    "get_layer_size",
    "get_tensor_load_numel",
    "has_device_tensors",
    "get_info_size",
    "adopt_fused_tensor",
]


def adopt_fused_tensor(
    module: nn.Module, attr: str, value: torch.Tensor
) -> torch.Tensor:
    """Adopt ``value`` into ``module``'s existing ``attr`` tensor when possible.

    Some models stack per-layer weights into fused buffers that are then read
    through addresses baked into captured CUDA graphs. Those buffers are plain
    attributes rather than registered buffers, so the layerwise reload path
    does not preserve their storage. Rebuilding them on a reload re-allocates
    new tensors and leaves the captured graph reading the old freed storage,
    even though the reload itself restored the correct values.

    When ``attr`` already holds a tensor with matching shape, dtype and device,
    copy ``value`` into it in place so the address stays stable across rebuilds.
    A cold load has no prior tensor and adopts ``value`` directly, so first-build
    behaviour is unchanged; a shape change still re-allocates.

    Args:
        module: module holding the fused buffer as a plain attribute.
        attr: name of the attribute to adopt into.
        value: freshly built tensor.

    Returns:
        The tensor now held at ``module.attr``.

    """
    old = getattr(module, attr, None)
    if (
        isinstance(old, torch.Tensor)
        and old.shape == value.shape
        and old.dtype == value.dtype
        and old.device == value.device
    ):
        old.copy_(value)
        return old
    setattr(module, attr, value)
    return value


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


def get_tensor_load_numel(tensor: torch.Tensor) -> int:
    """Count checkpoint elements, excluding padding declared by the weight creator."""
    return getattr(tensor, "weight_loader_numel", tensor.numel())


def get_layer_size(layer: torch.nn.Module) -> int:
    """Calculate total number of elements across loadable tensors in a layer.

    Excludes SKIP_LOAD_TENSORS (e.g. _expert_map) which are never loaded via
    weight_loader during layerwise reload.
    """
    from .meta import SKIP_LOAD_TENSORS

    return sum(
        get_tensor_load_numel(tensor)
        for name, tensor in get_layer_tensors(layer).items()
        if name not in SKIP_LOAD_TENSORS
    )


def has_device_tensors(bound_args: BoundArguments) -> bool:
    """Return True if the loaded weights exist on an accelerator device.

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
    """Calculate the number of bytes used by loaded weights for a given layer.

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
