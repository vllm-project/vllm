# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import UninitializedParameter

from .sanitize import restore_layer_refs, sanitize_layer_refs
from .types import LayerReloadingInfo, LayerTensors
from .utils import get_layer_params_buffers, get_layer_tensors

__all__ = [
    "to_meta_tensor",
    "materialize_meta_tensor",
    "capture_layer_to_meta",
    "restore_layer_on_meta",
    "materialize_layer",
]

# Modules whose tensors are never moved to, or materialized from, the meta device.
SKIP_MODULES: set[str] = {"HadamardTransform"}

# Tensors never loaded by a weight loader, so the layerwise trigger ignores them.
SKIP_LOAD_TENSORS: set[str] = {
    "_expert_map",
    "expert_mask",
    "expert_global_to_physical",
    "expert_physical_to_global",
    "expert_local_to_global",
    "e_score_correction_bias",
}

# Tensors which are never moved to, or materialized from, the meta device.
# `bias` is built after create_weights(), so it is never on meta to begin with.
SKIP_TENSORS: set[str] = SKIP_LOAD_TENSORS | {"bias"}


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


def _is_non_persistent_parameter_alias_buffer(
    layer: torch.nn.Module,
    name: str,
    buffer: torch.Tensor,
    parameter_storage_ptrs: set[int],
) -> bool:
    if name not in layer._non_persistent_buffers_set:
        return False

    buffer_storage_ptr = _tensor_storage_ptr(buffer)
    return (
        buffer_storage_ptr is not None and buffer_storage_ptr in parameter_storage_ptrs
    )


def _tensor_storage_ptr(tensor: torch.Tensor) -> int | None:
    if isinstance(tensor, UninitializedParameter):
        return None

    try:
        return tensor.untyped_storage().data_ptr()
    except (RuntimeError, ValueError):
        return None


def _parameter_storage_ptrs(layer: torch.nn.Module) -> set[int]:
    return {
        storage_ptr
        for param in layer.parameters(recurse=True)
        if (storage_ptr := _tensor_storage_ptr(param)) is not None
    }


def capture_layer_to_meta(layer: torch.nn.Module) -> LayerTensors:
    if layer.__class__.__name__ in SKIP_MODULES:
        return ({}, {})

    params, buffers = get_layer_params_buffers(layer)
    parameter_storage_ptrs = _parameter_storage_ptrs(layer)
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
            and not _is_non_persistent_parameter_alias_buffer(
                layer, name, buffer, parameter_storage_ptrs
            )
        },
    )


def restore_layer_on_meta(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Restore a layer to model format with tensors on the meta device"""
    if layer.__class__.__name__ in SKIP_MODULES:
        return

    non_persistent = set(layer._non_persistent_buffers_set)
    for name in get_layer_tensors(layer):
        if name not in SKIP_TENSORS:
            delattr(layer, name)

    restore_params, restore_buffers = info.restore_metadata
    for name, param in restore_params.items():
        layer.register_parameter(name, restore_layer_refs(param, layer))

    for name, buffer in restore_buffers.items():
        layer.register_buffer(
            name,
            restore_layer_refs(buffer, layer),
            persistent=name not in non_persistent,
        )


def materialize_layer(layer: torch.nn.Module, info: LayerReloadingInfo):
    """Materialize all meta tensors in a layer to actual tensors."""
    if layer.__class__.__name__ in SKIP_MODULES:
        return

    with info.restore_device:
        for name, tensor in get_layer_tensors(layer).items():
            if name not in SKIP_TENSORS and tensor.is_meta:
                setattr(layer, name, materialize_meta_tensor(tensor))
