# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .sanitize import restore_layer_refs, sanitize_layer_refs
from .types import LayerReloadingInfo, LayerTensors
from .utils import get_layer_params_buffers, get_layer_tensors

__all__ = [
    "to_meta_tensor",
    "materialize_meta_tensor",
    "capture_layer_to_meta",
    "restore_layer_on_meta",
    "materialize_layer",
    "get_numel_loaded",
]

SKIP_MODULES: set[str] = {"HadamardTransform"}

SKIP_TENSORS: set[str] = {
    "_expert_map",
    "expert_mask",
    "expert_global_to_physical",
    "expert_physical_to_global",
    "expert_local_to_global",
}


def to_meta_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a tensor to a meta tensor while preserving class and attributes."""
    meta_tensor = tensor.data.to("meta")
    meta_tensor.__class__ = tensor.__class__
    meta_tensor.__dict__ = tensor.__dict__.copy()
    return meta_tensor


def materialize_meta_tensor(meta_tensor: torch.Tensor) -> torch.Tensor:
    """
    Materialize a meta tensor into an actual tensor on the current device.
    Should be called within the torch device context for the given rank.
    """
    tensor = torch.empty_strided(
        size=tuple(meta_tensor.size()),
        stride=tuple(meta_tensor.stride()),
        dtype=meta_tensor.dtype,
        requires_grad=False,
    )
    tensor.__class__ = meta_tensor.__class__
    tensor.__dict__ = meta_tensor.__dict__.copy()
    return tensor


def capture_layer_to_meta(layer: torch.nn.Module) -> LayerTensors:
    if layer.__class__.__name__ in SKIP_MODULES:
        return ({}, {})

    params, buffers = get_layer_params_buffers(layer)
    return (
        {
            name: sanitize_layer_refs(to_meta_tensor(param), layer)
            for name, param in params.items()
            if name not in SKIP_TENSORS
        },
        {
            name: sanitize_layer_refs(to_meta_tensor(buffer), layer)
            for name, buffer in buffers.items()
            if name not in SKIP_TENSORS
        },
    )


def restore_layer_on_meta(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Restore a layer to model format with tensors on the meta device"""
    if layer.__class__.__name__ in SKIP_MODULES:
        return

    for name in get_layer_tensors(layer):
        if name not in SKIP_TENSORS:
            delattr(layer, name)

    restore_params, restore_buffers = info.restore_metadata
    for name, param in restore_params.items():
        if name not in SKIP_TENSORS:
            param = restore_layer_refs(param, layer)
            layer.register_parameter(name, param)

    for name, buffer in restore_buffers.items():
        if name not in SKIP_TENSORS:
            buffer = restore_layer_refs(buffer, layer)
            layer.register_buffer(name, buffer)


def materialize_layer(layer: torch.nn.Module) -> None:
    """Materialize all meta tensors in a layer to actual tensors."""
    if layer.__class__.__name__ in SKIP_MODULES:
        return

    for name, tensor in get_layer_tensors(layer).items():
        if name not in SKIP_TENSORS:
            setattr(layer, name, materialize_meta_tensor(tensor))


class MetaCopyCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`.

    Useful for keeping track of weight loading where underlying weights can be
    arbitrarily transformed (such as with `narrow`) before calling copy.

    Note: Assumes that copy kwargs are not used.
    """

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is torch.ops.aten.copy_.default and args[0].device.type == "meta":
            assert args[0].numel() == args[1].numel()
            self.copied_numel += args[0].numel()

        return func(*args, **kwargs)


def get_numel_loaded(
    weight_loader: Callable, args: inspect.BoundArguments
) -> tuple[int, object]:
    """
    Determine how many elements would be loaded by a weight loader call.

    :param weight loader: used to load weights
    :param args: bound arguments to weight loader
    :return: number of elements loaded by the weight loader, the return value of the
        weight loader
    """
    assert args.arguments["param"].device.type == "meta"
    with MetaCopyCounter() as counter:
        return_value = weight_loader(*args.args, **args.kwargs)
    return counter.copied_numel, return_value
