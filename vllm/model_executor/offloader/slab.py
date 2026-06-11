# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slab layout helpers for packed prefetch offloading."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SlabTensorSpec:
    """Storage metadata for one tensor packed into a byte slab."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    offset_bytes: int
    storage_bytes: int


@dataclass(frozen=True)
class SlabLayout:
    """Byte-slab layout for a group of tensors."""

    specs: tuple[SlabTensorSpec, ...]
    total_bytes: int


def build_slab_layout(
    tensors: list[tuple[str, torch.Tensor]],
    *,
    alignment_bytes: int = 16,
) -> SlabLayout:
    """Build a byte-packed slab layout preserving shape/stride metadata."""
    offset_bytes = 0
    specs: list[SlabTensorSpec] = []

    for name, tensor in tensors:
        itemsize = tensor.element_size()
        alignment = max(alignment_bytes, itemsize)
        offset_bytes = _align_up(offset_bytes, alignment)
        storage_bytes = storage_size_in_bytes(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
        specs.append(
            SlabTensorSpec(
                name=name,
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                offset_bytes=offset_bytes,
                storage_bytes=storage_bytes,
            )
        )
        offset_bytes += storage_bytes

    return SlabLayout(specs=tuple(specs), total_bytes=offset_bytes)


def build_slab_layout_from_specs(
    tensor_specs: list[tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype]],
    *,
    alignment_bytes: int = 16,
) -> SlabLayout:
    """Build a byte-packed slab layout from tensor metadata."""
    offset_bytes = 0
    specs: list[SlabTensorSpec] = []

    for name, shape, stride, dtype in tensor_specs:
        itemsize = _dtype_itemsize(dtype)
        alignment = max(alignment_bytes, itemsize)
        offset_bytes = _align_up(offset_bytes, alignment)
        storage_bytes = storage_size_in_bytes(shape, stride, dtype)
        specs.append(
            SlabTensorSpec(
                name=name,
                shape=shape,
                stride=stride,
                dtype=dtype,
                offset_bytes=offset_bytes,
                storage_bytes=storage_bytes,
            )
        )
        offset_bytes += storage_bytes

    return SlabLayout(specs=tuple(specs), total_bytes=offset_bytes)


def view_slab_tensor(slab: torch.Tensor, spec: SlabTensorSpec) -> torch.Tensor:
    """Create a typed tensor view from a byte slab."""
    assert slab.dtype == torch.uint8, "Slab storage must use uint8."
    itemsize = _dtype_itemsize(spec.dtype)
    assert spec.offset_bytes % itemsize == 0, (
        f"Spec {spec.name} offset {spec.offset_bytes} is not aligned to "
        f"dtype itemsize {itemsize}."
    )
    typed_storage = slab[
        spec.offset_bytes : spec.offset_bytes + spec.storage_bytes
    ].view(spec.dtype)
    return typed_storage.as_strided(spec.shape, spec.stride)


def storage_size_in_bytes(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> int:
    """Return the number of bytes needed for a tensor's strided storage."""
    return storage_size_in_elements(shape, stride) * _dtype_itemsize(dtype)


def storage_size_in_elements(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> int:
    """Return the underlying storage span in elements for a strided tensor."""
    if not shape:
        return 1
    if 0 in shape:
        return 0
    return 1 + sum((dim - 1) * step for dim, step in zip(shape, stride))


def _dtype_itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _align_up(value: int, alignment: int) -> int:
    return int(math.ceil(value / alignment) * alignment)
