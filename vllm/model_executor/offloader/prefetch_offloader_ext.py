# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auxiliary lifecycle / instrumentation helpers for ``PrefetchOffloader``.

These free functions are imported by :class:`prefetch.PrefetchOffloader` and
:class:`prefetch._ModuleOffloader` so the class bodies themselves stay close
to the upstream prefetch implementation.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import should_pin_memory
from vllm.model_executor.offloader.prefetch_diagnostics import (
    should_log_transfer_stats,
)
from vllm.model_executor.offloader.prefetch_helpers import nvtx_range
from vllm.model_executor.offloader.prefetch_runtime_buffers import (
    StaticBufferPool,
    StorageGroupBufferPool,
    StorageGroupInfo,
    build_runtime_buffer_plan,
)
from vllm.model_executor.offloader.slab import SlabLayout, view_slab_tensor

if TYPE_CHECKING:
    from vllm.model_executor.offloader.prefetch import (
        ParamInfo,
        PrefetchOffloader,
        _ModuleOffloader,
    )

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# PrefetchOffloader-level helpers.
# ---------------------------------------------------------------------------


def allocate_runtime_buffers(
    self: "PrefetchOffloader",
    device: torch.device,
    module_param_infos: list[list["ParamInfo"]],
    module_storage_group_infos: list[tuple[StorageGroupInfo, ...]],
) -> int:
    """Allocate slab + storage-group runtime buffer pools and return total bytes."""
    # Resolve the pool classes through the public ``prefetch`` module so that
    # tests can monkeypatch ``prefetch.StaticBufferPool`` /
    # ``prefetch.StorageGroupBufferPool`` to inject fake pool implementations.
    from vllm.model_executor.offloader import prefetch as prefetch_module

    static_buffer_pool_cls = prefetch_module.StaticBufferPool
    storage_group_buffer_pool_cls = prefetch_module.StorageGroupBufferPool

    if module_param_infos:
        self.buffer_pool = static_buffer_pool_cls(
            module_param_infos=module_param_infos,
            slot_capacity=self.prefetch_step,
            device=device,
        )
    if module_storage_group_infos:
        self.storage_group_pool = storage_group_buffer_pool_cls(
            module_storage_group_infos=module_storage_group_infos,
            slot_capacity=self.prefetch_step,
            device=device,
        )

    assert self.runtime is not None, "Runtime controller not initialized"
    for runtime_unit, offloader in zip(self.runtime.units, self.module_offloaders):
        offloader.assign_buffer_slot(
            self.buffer_pool, self.storage_group_pool, runtime_unit.slot_idx
        )

    runtime_buffer_bytes = 0
    if self.buffer_pool is not None:
        runtime_buffer_bytes += self.buffer_pool.total_bytes
    if self.storage_group_pool is not None:
        runtime_buffer_bytes += self.storage_group_pool.total_bytes
    runtime_buffer_bytes += sum(
        offloader.direct_buffer_bytes for offloader in self.module_offloaders
    )
    return runtime_buffer_bytes


def collect_module_buffer_infos(
    module_offloaders: list["_ModuleOffloader"],
) -> tuple[
    list[list["ParamInfo"]],
    list[tuple[StorageGroupInfo, ...]],
    torch.device | None,
]:
    """Collect slab + storage-group descriptors and the common device."""
    module_param_infos: list[list[ParamInfo]] = []
    module_storage_group_infos: list[tuple[StorageGroupInfo, ...]] = []
    device: torch.device | None = None
    for offloader in module_offloaders:
        if device is None:
            device = offloader.device
        if offloader.uses_slab_buffers:
            module_param_infos.append(offloader.get_param_infos())
        if offloader.uses_storage_group_fallback:
            module_storage_group_infos.append(offloader.storage_group_infos)
    return module_param_infos, module_storage_group_infos, device


def start_initial_prefetches(self: "PrefetchOffloader") -> None:
    """Start the pre-forward prefetch window from a clean runtime state."""
    with nvtx_range("weight_offload.initial_prefetch"):
        assert self.runtime is not None, "Runtime controller not initialized"
        for runtime_unit in self.runtime.initial_prefetches():
            previous_owner = self.runtime.begin_prefetch(runtime_unit.unit_idx)
            assert previous_owner is None, (
                "Initial prefetches should not need slot handoff"
            )
            self.runtime.mark_prefetch_started(
                runtime_unit.unit_idx,
                in_capture=self.module_offloaders[
                    runtime_unit.unit_idx
                ].start_onload_to_static(),
            )


def reset_runtime_state(self: "PrefetchOffloader") -> None:
    """Reset transient prefetch state and restart initial prefetches."""
    if self.runtime is None or not self.module_offloaders:
        return
    self.sync_prev_onload()
    self.runtime.reset()
    self.transfer_stats.reset()
    for offloader in self.module_offloaders:
        offloader.reset_runtime_tracking()
    start_initial_prefetches(self)


def begin_forward_stats(self: "PrefetchOffloader") -> None:
    if not should_log_transfer_stats():
        return
    self.transfer_stats.reset()


def end_forward_stats(self: "PrefetchOffloader") -> None:
    if not should_log_transfer_stats():
        return
    torch.cuda.current_stream().synchronize()
    self.copy_stream.synchronize()
    flush_transfer_timings(self)
    snap = self.transfer_stats.forward_snapshot()
    logger.info(
        "[PrefetchOffloader] forward_stats: "
        "h2d_gb=%.2f h2d_copy_ops=%d "
        "gpu_copy_time_s=%.6f gpu_wait_time_s=%.6f "
        "gpu_copy_bandwidth_gb_s=%.2f",
        snap["h2d_gb"],
        snap["h2d_copy_ops"],
        snap["gpu_copy_time_s"],
        snap["gpu_wait_time_s"],
        snap["gpu_copy_bandwidth_gb_s"],
    )
    self.transfer_stats.reset()


def record_current_stream_wait(
    self: "PrefetchOffloader",
    wait_fn: Callable[[torch.cuda.Stream], None],
) -> None:
    stream = torch.cuda.current_stream()
    if not should_log_transfer_stats():
        wait_fn(stream)
        return
    wait_start = torch.cuda.Event(enable_timing=True)
    wait_end = torch.cuda.Event(enable_timing=True)
    wait_start.record(stream)
    wait_fn(stream)
    wait_end.record(stream)
    self.transfer_stats.record_wait(wait_start, wait_end)


def flush_transfer_timings(
    self: "PrefetchOffloader",
    *,
    skip_query: bool = False,
) -> None:
    self.transfer_stats.flush_copy_timings(skip_query=skip_query)
    self.transfer_stats.flush_wait_timings(skip_query=skip_query)


# ---------------------------------------------------------------------------
# _ModuleOffloader-level helpers.
# ---------------------------------------------------------------------------


def refresh_runtime_buffer_strategy(self: "_ModuleOffloader") -> None:
    """Choose slab-backed tensors, storage-group fallback and direct fallback."""
    named_tensors: list[tuple[str, torch.Tensor]] = []
    for name, offloader in self._param_offloaders.items():
        cpu_storage = offloader._cpu_storage
        assert cpu_storage is not None, f"CPU storage for {name} not initialized"
        named_tensors.append((name, cpu_storage))

    plan = build_runtime_buffer_plan(named_tensors)
    self._slab_param_names = plan.slab_param_names
    self._storage_group_infos = plan.storage_group_infos
    self._direct_param_names = plan.direct_param_names
    self._fallback_reasons = plan.fallback_reasons

    if not plan.fallback_reasons:
        logger.debug(
            "[PrefetchOffloader] Layer %d uses slab-backed runtime buffers.",
            self.layer_idx,
        )
        return

    logger.info(
        "[PrefetchOffloader] Layer %d uses mixed runtime buffers: "
        "%d slab param(s), %d storage-group fallback group(s), "
        "%d direct fallback param(s): %s",
        self.layer_idx,
        len(plan.slab_param_names),
        len(plan.storage_group_infos),
        len(plan.direct_param_names),
        "; ".join(plan.fallback_reasons),
    )


def assign_module_buffer_slot(
    self: "_ModuleOffloader",
    pool: StaticBufferPool | None,
    storage_group_pool: StorageGroupBufferPool | None,
    slot_idx: int,
) -> None:
    """Assign module to a buffer slot, dispatching across the three tiers."""
    self._buffer_slot_idx = slot_idx
    self._direct_buffer_bytes = 0

    if self.uses_slab_buffers:
        assert pool is not None, "Slab-backed runtime buffers require a pool"
        self._buffer_pool = pool
        param_infos = self.get_param_infos()
        slab_layout, gpu_slab, slab_views = pool.get_slab_assignment(
            param_infos, slot_idx
        )
        self._slab_layout = slab_layout
        self._gpu_slab = gpu_slab
        self._cpu_slab = _build_cpu_slab(self, slab_layout)
        for name in self._slab_param_names:
            self._param_offloaders[name].assign_static_buffer(slab_views[name])
    else:
        self._buffer_pool = None
        self._slab_layout = None
        self._cpu_slab = None
        self._gpu_slab = None

    if self.uses_storage_group_fallback:
        assert storage_group_pool is not None, (
            "Storage-group fallback runtime buffers require a pool"
        )
        self._storage_group_buffers, param_views = storage_group_pool.get_assignment(
            self._storage_group_infos, slot_idx
        )
        for name, offloader in self._param_offloaders.items():
            if name in param_views:
                offloader.assign_static_buffer(param_views[name])
    else:
        self._storage_group_buffers = []

    self._direct_buffers = {}
    for name in self._direct_param_names:
        cpu_storage = self._param_offloaders[name]._cpu_storage
        assert cpu_storage is not None, f"CPU storage for {name} not initialized"
        gpu_buffer = torch.empty_like(cpu_storage, device=self.device)
        self._direct_buffers[name] = gpu_buffer
        self._direct_buffer_bytes += gpu_buffer.numel() * gpu_buffer.element_size()
        self._param_offloaders[name].assign_static_buffer(gpu_buffer)


def _build_cpu_slab(
    self: "_ModuleOffloader",
    slab_layout: SlabLayout,
) -> torch.Tensor:
    """Build one pinned CPU slab from the current parameter CPU storages."""
    cpu_slab = torch.empty(
        slab_layout.total_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=should_pin_memory(),
    )
    for spec in slab_layout.specs:
        cpu_storage = self._param_offloaders[spec.name]._cpu_storage
        assert cpu_storage is not None, f"CPU storage for {spec.name} not initialized"
        view_slab_tensor(cpu_slab, spec).copy_(cpu_storage)
    return cpu_slab
