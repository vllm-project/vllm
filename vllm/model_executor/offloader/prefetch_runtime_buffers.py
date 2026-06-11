# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime GPU buffer pools for prefetch weight offloading.

A prefetched parameter ends up in one of three runtime buffer kinds:

1. **Slab-backed** (preferred): one packed byte slab per module-layout fits all
   simply-strided parameters with ``storage_offset == 0`` and unique storage.
   Allocated by :class:`StaticBufferPool`.
2. **Storage-group fallback**: parameters that share underlying storage (e.g.
   sliced views) are kept aliased through :class:`StorageGroupBufferPool`.
3. **Direct fallback**: anything that cannot be packed (sparse, etc.) gets a
   per-parameter standalone GPU buffer.

The split for one module is captured in :class:`RuntimeBufferPlan`.

``ParamInfo`` lives in :mod:`prefetch` so the new buffer-pool plumbing here
can reuse the upstream-shaped dataclass without redefining it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.slab import (
    SlabLayout,
    build_slab_layout_from_specs,
    view_slab_tensor,
)

if TYPE_CHECKING:
    from vllm.model_executor.offloader.prefetch import ParamInfo

logger = init_logger(__name__)


@dataclass(frozen=True)
class StorageGroupViewSpec:
    """Metadata for one parameter view backed by a shared storage group."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    storage_offset: int


@dataclass(frozen=True)
class StorageGroupInfo:
    """Metadata for one underlying storage shared by one or more parameters."""

    storage_numel: int
    dtype: torch.dtype
    cpu_source: torch.Tensor
    view_specs: tuple[StorageGroupViewSpec, ...]

    @property
    def key(
        self,
    ) -> tuple[
        int,
        torch.dtype,
        tuple[tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype, int], ...],
    ]:
        return (
            self.storage_numel,
            self.dtype,
            tuple(
                (
                    spec.name,
                    spec.shape,
                    spec.stride,
                    spec.dtype,
                    spec.storage_offset,
                )
                for spec in self.view_specs
            ),
        )


@dataclass(frozen=True)
class RuntimeBufferPlan:
    """Per-module split between slab-packable and fallback parameters."""

    slab_param_names: tuple[str, ...]
    storage_group_infos: tuple[StorageGroupInfo, ...]
    direct_param_names: tuple[str, ...]
    fallback_reasons: tuple[str, ...]


def view_storage_group_tensor(
    storage: torch.Tensor,
    spec: StorageGroupViewSpec,
) -> torch.Tensor:
    """Create a parameter view from a storage-group buffer."""
    assert storage.dtype == spec.dtype, (
        f"Storage dtype {storage.dtype} does not match spec dtype {spec.dtype}."
    )
    return storage.as_strided(spec.shape, spec.stride, spec.storage_offset)


def build_storage_group_infos(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> tuple[StorageGroupInfo, ...]:
    """Group tensors by underlying storage to preserve aliasing semantics."""
    grouped_specs: dict[int, list[StorageGroupViewSpec]] = {}
    group_order: list[int] = []
    cpu_sources: dict[int, torch.Tensor] = {}
    storage_numels: dict[int, int] = {}
    storage_dtypes: dict[int, torch.dtype] = {}

    for name, tensor in named_tensors:
        assert tensor.layout == torch.strided, (
            f"Offloaded parameter {name} must use strided layout, got {tensor.layout}."
        )
        storage_ptr = tensor.untyped_storage().data_ptr()
        storage_numel = tensor.untyped_storage().nbytes() // tensor.element_size()

        if storage_ptr not in grouped_specs:
            group_order.append(storage_ptr)
            grouped_specs[storage_ptr] = []
            storage_numels[storage_ptr] = storage_numel
            storage_dtypes[storage_ptr] = tensor.dtype
            cpu_sources[storage_ptr] = tensor.as_strided(
                (storage_numel,), (1,), storage_offset=0
            )
        else:
            assert storage_numels[storage_ptr] == storage_numel, (
                f"Shared storage for {name} changed size unexpectedly."
            )
            assert storage_dtypes[storage_ptr] == tensor.dtype, (
                f"Shared storage for {name} changed dtype unexpectedly."
            )

        grouped_specs[storage_ptr].append(
            StorageGroupViewSpec(
                name=name,
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                dtype=tensor.dtype,
                storage_offset=tensor.storage_offset(),
            )
        )

    return tuple(
        StorageGroupInfo(
            storage_numel=storage_numels[ptr],
            dtype=storage_dtypes[ptr],
            cpu_source=cpu_sources[ptr],
            view_specs=tuple(grouped_specs[ptr]),
        )
        for ptr in group_order
    )


def build_runtime_buffer_plan(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> RuntimeBufferPlan:
    """Split one module into slab-packable tensors and fallback groups."""
    grouped_tensors: dict[int, list[tuple[str, torch.Tensor]]] = {}
    group_order: list[int] = []
    direct_param_names: list[str] = []
    fallback_reasons: list[str] = []

    for name, tensor in named_tensors:
        if tensor.layout != torch.strided:
            direct_param_names.append(name)
            fallback_reasons.append(f"{name} uses non-strided layout {tensor.layout}")
            continue
        storage_ptr = tensor.untyped_storage().data_ptr()
        if storage_ptr not in grouped_tensors:
            group_order.append(storage_ptr)
            grouped_tensors[storage_ptr] = []
        grouped_tensors[storage_ptr].append((name, tensor))

    slab_param_names: list[str] = []
    storage_group_infos: list[StorageGroupInfo] = []

    for storage_ptr in group_order:
        group_tensors = grouped_tensors[storage_ptr]

        if len(group_tensors) == 1 and group_tensors[0][1].storage_offset() == 0:
            slab_param_names.append(group_tensors[0][0])
            continue

        if len(group_tensors) == 1:
            name, tensor = group_tensors[0]
            fallback_reasons.append(
                f"{name} has storage_offset={tensor.storage_offset()}"
            )
        else:
            fallback_reasons.append(
                ", ".join(name for name, _ in group_tensors)
                + " share underlying storage"
            )

        storage_group_infos.extend(build_storage_group_infos(group_tensors))

    return RuntimeBufferPlan(
        slab_param_names=tuple(slab_param_names),
        storage_group_infos=tuple(storage_group_infos),
        direct_param_names=tuple(direct_param_names),
        fallback_reasons=tuple(fallback_reasons),
    )


class StaticBufferPool:
    """Pre-allocated GPU buffer pool for offloaded parameters.

    Allocates one packed byte slab per module-layout, allowing for
    double/triple buffering during prefetch. Buffer slots are reused
    circularly: layer N uses slot (N % slot_capacity).

    Parameters with simply-strided storage and ``storage_offset == 0`` are
    packed into the slab so a whole layer is staged with a single H2D copy;
    parameters that share storage or use unusual layouts are handled via
    :class:`StorageGroupBufferPool` or per-parameter direct fallback buffers.

    The legacy :meth:`get_buffer` adapter exposes the original
    per-parameter ``(name, shape, stride, dtype, slot_idx) -> Tensor`` API
    backed by the same slab buffers.
    """

    def __init__(
        self,
        module_param_infos: list[list["ParamInfo"]],
        slot_capacity: int,
        device: torch.device,
    ):
        self.slot_capacity = slot_capacity
        self.total_bytes = 0
        self._device = device

        self._slab_layouts: dict[tuple, SlabLayout] = {}
        self._slab_buffers: dict[tuple, list[torch.Tensor]] = {}
        for param_infos in module_param_infos:
            layout_key = tuple(info.key for info in param_infos)
            if layout_key in self._slab_layouts:
                continue
            layout = build_slab_layout_from_specs(list(layout_key))
            self._slab_layouts[layout_key] = layout
            self._slab_buffers[layout_key] = [
                torch.empty(layout.total_bytes, dtype=torch.uint8, device=device)
                for _ in range(slot_capacity)
            ]
        self.total_bytes = sum(
            layout.total_bytes * slot_capacity for layout in self._slab_layouts.values()
        )
        logger.debug(
            "[StaticBufferPool] Allocated %d module slab layout(s), total %.4f GB",
            len(self._slab_layouts),
            self.total_bytes / 1e9,
        )

    def get_slab_assignment(
        self,
        param_infos: list["ParamInfo"],
        slot_idx: int,
    ) -> tuple[SlabLayout, torch.Tensor, dict[str, torch.Tensor]]:
        """Get slab-backed assignment data for one module layout."""
        layout_key = tuple(info.key for info in param_infos)
        layout = self._slab_layouts[layout_key]
        slab = self._slab_buffers[layout_key][slot_idx % self.slot_capacity]
        return (
            layout,
            slab,
            {spec.name: view_slab_tensor(slab, spec) for spec in layout.specs},
        )

    def get_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
        slot_idx: int,
    ) -> torch.Tensor:
        """Legacy adapter: return one parameter's view from the matching slab.

        Provides the original ``(name, shape, stride, dtype, slot_idx)`` API
        backed by the slab-packed buffers used internally.
        """
        for layout_key, layout in self._slab_layouts.items():
            for spec in layout.specs:
                if (spec.name, spec.shape, spec.stride, spec.dtype) == (
                    name,
                    shape,
                    stride,
                    dtype,
                ):
                    slab = self._slab_buffers[layout_key][slot_idx % self.slot_capacity]
                    return view_slab_tensor(slab, spec)
        raise KeyError(f"No slab buffer for ({name=}, {shape=}, {stride=}, {dtype=}).")


class StorageGroupBufferPool:
    """Pre-allocated fallback buffers for alias-preserving storage groups."""

    def __init__(
        self,
        module_storage_group_infos: list[tuple[StorageGroupInfo, ...]],
        slot_capacity: int,
        device: torch.device,
    ):
        self.slot_capacity = slot_capacity
        self.total_bytes = 0
        self._device = device

        self._group_layouts: dict[tuple, tuple[StorageGroupInfo, ...]] = {}
        self._group_buffers: dict[tuple, list[list[torch.Tensor]]] = {}

        for group_infos in module_storage_group_infos:
            layout_key = tuple(group_info.key for group_info in group_infos)
            if layout_key in self._group_layouts:
                continue
            self._group_layouts[layout_key] = group_infos
            self._group_buffers[layout_key] = [
                [
                    torch.empty(
                        group_info.storage_numel,
                        dtype=group_info.dtype,
                        device=device,
                    )
                    for group_info in group_infos
                ]
                for _ in range(slot_capacity)
            ]

        self.total_bytes = sum(
            sum(
                group_info.storage_numel
                * torch.empty((), dtype=group_info.dtype).element_size()
                for group_info in group_infos
            )
            * slot_capacity
            for group_infos in self._group_layouts.values()
        )
        logger.debug(
            "[StorageGroupBufferPool] Allocated %d fallback layout(s), total %.4f GB",
            len(self._group_layouts),
            self.total_bytes / 1e9,
        )

    def get_assignment(
        self,
        group_infos: tuple[StorageGroupInfo, ...],
        slot_idx: int,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        """Get fallback buffers and parameter views for one module layout."""
        layout_key = tuple(group_info.key for group_info in group_infos)
        buffers = self._group_buffers[layout_key][slot_idx % self.slot_capacity]
        param_views: dict[str, torch.Tensor] = {}
        for group_info, buffer in zip(group_infos, buffers):
            for spec in group_info.view_specs:
                param_views[spec.name] = view_storage_group_tensor(buffer, spec)
        return buffers, param_views
