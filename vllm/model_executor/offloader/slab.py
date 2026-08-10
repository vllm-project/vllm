# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slab layout helpers for packed prefetch offloading."""

import math
from dataclasses import dataclass

import torch

PINNED_CPU_SLAB_CHUNK_BYTES = 2 * 1024**3
PINNED_CPU_SLAB_MIN_TAIL_BYTES = 128 * 1024**2


@dataclass(frozen=True)
class CpuSlabChunk:
    """One pinned CPU byte range and its offset in the GPU slab."""

    offset_bytes: int
    data: torch.Tensor


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


def build_slab_chunk_ranges(
    total_bytes: int,
    *,
    chunk_bytes: int = PINNED_CPU_SLAB_CHUNK_BYTES,
    min_tail_bytes: int = PINNED_CPU_SLAB_MIN_TAIL_BYTES,
) -> tuple[tuple[int, int], ...]:
    """Split a slab into bounded ranges without leaving a tiny tail."""
    if total_bytes < 0:
        raise ValueError("total_bytes must be non-negative")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if not 0 <= min_tail_bytes < chunk_bytes:
        raise ValueError("min_tail_bytes must be in [0, chunk_bytes)")
    if total_bytes == 0:
        return ()

    full_chunks, tail_bytes = divmod(total_bytes, chunk_bytes)
    if tail_bytes == 0 or tail_bytes >= min_tail_bytes or full_chunks == 0:
        return tuple(
            (start, min(start + chunk_bytes, total_bytes))
            for start in range(0, total_bytes, chunk_bytes)
        )

    ranges = [
        (index * chunk_bytes, (index + 1) * chunk_bytes)
        for index in range(full_chunks - 1)
    ]
    penultimate_start = (full_chunks - 1) * chunk_bytes
    tail_start = total_bytes - min_tail_bytes
    ranges.append((penultimate_start, tail_start))
    ranges.append((tail_start, total_bytes))
    return tuple(ranges)


def build_cpu_slab_chunks(
    layout: SlabLayout,
    tensors: dict[str, torch.Tensor],
    *,
    chunk_bytes: int = PINNED_CPU_SLAB_CHUNK_BYTES,
    min_tail_bytes: int = PINNED_CPU_SLAB_MIN_TAIL_BYTES,
    pin_memory: bool,
) -> tuple[CpuSlabChunk, ...]:
    """Pack one logical slab into bounded pinned CPU allocations.

    The GPU slab remains contiguous; only its CPU source is split into byte
    ranges. Copying raw storage bytes supports strided tensors and parameters
    that cross a chunk boundary.
    """
    chunks: list[CpuSlabChunk] = []
    for chunk_start, chunk_end in build_slab_chunk_ranges(
        layout.total_bytes,
        chunk_bytes=chunk_bytes,
        min_tail_bytes=min_tail_bytes,
    ):
        data = torch.empty(
            chunk_end - chunk_start,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )

        for spec in layout.specs:
            spec_start = spec.offset_bytes
            spec_end = spec_start + spec.storage_bytes
            copy_start = max(chunk_start, spec_start)
            copy_end = min(chunk_end, spec_end)
            if copy_start >= copy_end:
                continue

            tensor = tensors[spec.name]
            assert tensor.storage_offset() == 0, (
                f"Slab tensor {spec.name} must have storage_offset=0."
            )
            itemsize = tensor.element_size()
            assert spec.storage_bytes % itemsize == 0
            storage = tensor.as_strided(
                (spec.storage_bytes // itemsize,),
                (1,),
                storage_offset=0,
            ).view(torch.uint8)
            source_start = copy_start - spec_start
            source_end = copy_end - spec_start
            data[copy_start - chunk_start : copy_end - chunk_start].copy_(
                storage[source_start:source_end]
            )

        chunks.append(CpuSlabChunk(offset_bytes=chunk_start, data=data))

    return tuple(chunks)


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
