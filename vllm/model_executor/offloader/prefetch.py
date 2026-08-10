# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Prefetch-based CPU offloading with async prefetching.

Uses static buffers and event-based stream forking for torch.compile +
CUDA graph compatibility. Events allow the copy stream to join CUDA
graph captures, ensuring H2D copies are properly captured.
"""

import threading
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# Import prefetch_ops to register custom ops at module load time
import vllm.model_executor.offloader.prefetch_ops  # noqa: F401
from vllm.config.offload import PrefetchOffloadSelector
from vllm.logger import init_logger
from vllm.model_executor.offloader import prefetch_offloader_ext as ext
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory
from vllm.model_executor.offloader.planner import OffloadUnit, should_offload_module
from vllm.model_executor.offloader.prefetch_diagnostics import (
    PrefetchCopySegment,
    PrefetchScheduleRow,  # noqa: F401  (re-exported)
    PrefetchTransferStats,
    build_prefetch_copy_segments,
    log_prefetch_manifest,
    log_prefetch_offload_plan,
    log_prefetch_schedule,
    should_collect_prefetch_debug_metadata,
)
from vllm.model_executor.offloader.prefetch_helpers import (
    maybe_bind_process_to_current_gpu_numa,
    maybe_retarget_offload_unit,
    nvtx_range,
    pick_dependency_tensor,
    pick_output_dependency_tensor,
)
from vllm.model_executor.offloader.prefetch_onload import run_onload_to_static
from vllm.model_executor.offloader.prefetch_runtime_buffers import (
    StaticBufferPool,
    StorageGroupBufferPool,
    StorageGroupInfo,
)
from vllm.model_executor.offloader.prefetch_tail_copy import (
    TailCopyScheduler,
    is_wraparound_prefetch,
)
from vllm.model_executor.offloader.runtime import PrefetchRuntimeController
from vllm.model_executor.offloader.selectors import select_module_parameters
from vllm.model_executor.offloader.slab import CpuSlabChunk, SlabLayout
from vllm.utils.torch_utils import get_dtype_size

logger = init_logger(__name__)


@dataclass
class ParamInfo:
    """Metadata about an offloaded parameter."""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype

    @property
    def key(self) -> tuple[str, tuple[int, ...], tuple[int, ...], torch.dtype]:
        """Unique key for buffer pool grouping.

        Includes parameter name to prevent different parameters with the same
        shape from sharing buffers within the same layer. Parameters with the
        same name across different layers will share buffers (via slots).

        Includes stride because parameters with same shape but different
        strides need separate buffers to preserve memory layout.
        """
        return (self.name, self.shape, self.stride, self.dtype)

    @property
    def num_bytes(self) -> int:
        """Size in bytes."""
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * get_dtype_size(self.dtype)


class PrefetchOffloader(BaseOffloader):
    """Prefetching-based offloader with group-based layer selection.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    Uses static buffers and stream synchronization for torch.compile and
    CUDA graph compatibility.

    Args:
        group_size: Group every N layers together.
        num_in_group: Offload this many layers per group (last N of each group).
        prefetch_step: Number of layers to prefetch ahead.
        comm_aware: Pace H2D copies around TP collectives.
        mode: Offload mode ("cpu" is currently supported).
    """

    # Class-level default so the attribute is defined even for instances built
    # without __init__ (the offloader is heavyweight to construct).
    comm_aware: bool = False

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        offload_params: set[str] | None = None,
        offload_selectors: set[PrefetchOffloadSelector] | None = None,
        comm_aware: bool = False,
        mode: str = "cpu",
    ):
        maybe_bind_process_to_current_gpu_numa()

        self.group_size = group_size
        self.num_in_group = num_in_group
        self.prefetch_step = prefetch_step
        self.offload_params = offload_params or set()
        self.offload_selectors = offload_selectors or set()
        self.comm_aware = comm_aware
        self.mode = mode

        # Copy stream for async H2D transfers
        self.copy_stream = torch.cuda.Stream()
        self.tail_copy_scheduler = TailCopyScheduler(
            device=torch.accelerator.current_device_index(),
            copy_stream=self.copy_stream,
        )

        # Module offloaders and buffer pools (populated in wrap_modules/post_init)
        self.module_offloaders: list[_ModuleOffloader] = []
        self.runtime: PrefetchRuntimeController | None = None
        self.buffer_pool: StaticBufferPool | None = None
        self.storage_group_pool: StorageGroupBufferPool | None = None
        self.total_offloaded_bytes = 0
        self._static_runtime_buffer_bytes = 0
        self.transfer_stats = PrefetchTransferStats()
        self._diagnostic_modules: tuple[nn.Module, ...] = ()
        self._diagnostic_plan_units: tuple[OffloadUnit, ...] = ()

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with prefetch offloading logic."""
        assert len(self.module_offloaders) == 0, (
            "wrap_modules should only be called once"
        )

        modules: list[nn.Module] = []
        units: list[OffloadUnit] = []

        for module_index, module in enumerate(modules_generator):
            modules.append(module)

            # Select layers to offload based on group pattern
            # Offload last num_in_group layers of each group_size
            if not should_offload_module(
                module_index,
                group_size=self.group_size,
                num_in_group=self.num_in_group,
            ):
                continue

            param_names = select_module_parameters(
                module,
                selectors=self.offload_selectors,
                include_names=self.offload_params,
            )
            if not param_names:
                continue  # skip layers with no matching params

            unit_module, unit_param_names = maybe_retarget_offload_unit(
                module,
                param_names,
                selectors=self.offload_selectors,
                include_names=self.offload_params,
            )
            unit = OffloadUnit(
                module_index=module_index,
                module=unit_module,
                param_names=unit_param_names,
            )
            units.append(unit)
            self.module_offloaders.append(
                _ModuleOffloader(
                    mode=self.mode,
                    module=unit.module,
                    copy_stream=self.copy_stream,
                    tail_copy_scheduler=self.tail_copy_scheduler,
                    whitelist_param_names=list(unit.param_names),
                    layer_idx=len(units) - 1,
                    module_index=unit.module_index,
                    transfer_stats=self.transfer_stats,
                    comm_aware=self.comm_aware,
                )
            )

        self.runtime = PrefetchRuntimeController(
            unit_count=len(units),
            prefetch_step=self.prefetch_step,
        )
        if should_collect_prefetch_debug_metadata():
            self._diagnostic_modules = tuple(modules)
            self._diagnostic_plan_units = tuple(units)
        log_prefetch_offload_plan(units)
        log_prefetch_schedule(units, self.runtime, module_count=len(modules))

        for runtime_unit, unit in zip(self.runtime.units, units):
            self._hook_module_forward(runtime_unit.unit_idx, unit.module)

        return modules

    def _hook_module_forward(self, index: int, module: nn.Module):
        """Hook module's forward with torch.compile-compatible sync."""
        original_forward = module.forward
        assert self.runtime is not None, "Runtime controller not initialized"
        next_unit = self.runtime.prefetch_after(index)
        next_unit_idx = None if next_unit is None else next_unit.unit_idx

        def forward(*args, **kwargs):
            # Temporarily restore original forward to avoid recursion
            module.forward = original_forward

            # Wait for this layer's prefetch to complete.
            # mutates_args on the main activation tensor creates the scheduling
            # dependency for torch.compile. Prefer hidden_states when present;
            # otherwise pick the first floating-point tensor instead of
            # blindly using args[0] (which can be metadata like positions).
            positional_tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
            input_tensor = pick_dependency_tensor(
                positional_tensors,
                preferred_tensor=kwargs.get("hidden_states"),
            )
            torch.ops.vllm.wait_prefetch(input_tensor, index)

            # No parameter swapping needed - parameters already point to
            # GPU static buffers (set in assign_static_buffer)
            output = original_forward(*args, **kwargs)

            # Start prefetch for next layer (circular)
            # mutates_args on output_tensor creates ordering dependency
            if next_unit_idx is not None:
                output_tensor = pick_output_dependency_tensor(output)
                is_tail_prefetch = is_wraparound_prefetch(index, next_unit_idx)
                torch.ops.vllm.start_prefetch(
                    output_tensor, next_unit_idx, is_tail_prefetch
                )

            # No explicit offload needed - static buffers are reused implicitly

            # Restore hooked forward
            module.forward = forward
            return output

        module.forward = forward

    def _wait_for_layer(self, layer_idx: int):
        """Called by custom op - wait for copy to complete.

        Synchronization strategy:
        - During CUDA graph capture: use event-based wait (graph-compatible)
        - Outside capture (warmup/eager): use wait_stream (more robust)

        During capture, we skip wait for pre-capture prefetches because:
        1. sync_before_graph_capture() ensures pre-capture work is complete
        2. We can't wait on pre-capture events during capture (isolation error)
        """
        offloader = self.module_offloaders[layer_idx]
        with nvtx_range(
            f"weight_offload.wait unit={layer_idx} "
            f"position={getattr(offloader, 'module_index', -1)} "
            f"slot={getattr(offloader, '_buffer_slot_idx', -1)}"
        ):
            assert self.runtime is not None, "Runtime controller not initialized"

            if torch.cuda.is_current_stream_capturing():
                # During capture, skip wait for pre-capture prefetches.
                # sync_before_graph_capture() ensures pre-capture work is complete.
                if not self.runtime.is_pending_in_capture(layer_idx):
                    return
                # Event-based wait for in-capture prefetches (graph-compatible)
                torch.cuda.current_stream().wait_event(offloader._copy_done_event)
                ext.flush_transfer_timings(self, skip_query=True)
                # Mark that this prefetch has been waited on (joined).
                self.runtime.mark_waited(layer_idx)
            else:
                if offloader._event_valid_for_eager:
                    offloader.wait_until_copy_done_event_recorded()
                    # Use per-layer event to only wait for THIS layer's copy,
                    # allowing other layers' prefetches to run concurrently.
                    ext.record_current_stream_wait(
                        self,
                        lambda stream: stream.wait_event(offloader._copy_done_event),
                    )
                else:
                    # Event not usable (unrecorded or recorded during capture).
                    # Fall back to wait_stream to drain all copy_stream work.
                    ext.record_current_stream_wait(
                        self,
                        lambda stream: stream.wait_stream(self.copy_stream),
                    )
                ext.flush_transfer_timings(self)

    def sync_prev_onload(self):
        """Sync previous onload operations.

        Ensures any H2D copies in flight on copy_stream complete before
        the compute stream continues. Call this before CUDA graph
        capture/replay or when synchronization is needed.
        """
        for offloader in self.module_offloaders:
            offloader.wait_until_copy_done_event_recorded()
        ext.record_current_stream_wait(
            self, lambda stream: stream.wait_stream(self.copy_stream)
        )
        ext.flush_transfer_timings(self)

    def _start_prefetch(
        self,
        layer_idx: int,
        is_tail_prefetch: bool = False,
    ):
        """Called by custom op - start async copy to static buffer."""
        offloader = self.module_offloaders[layer_idx]
        with nvtx_range(
            f"weight_offload.start_prefetch unit={layer_idx} "
            f"position={getattr(offloader, 'module_index', -1)} "
            f"slot={getattr(offloader, '_buffer_slot_idx', -1)} "
            f"bytes={getattr(offloader, '_h2d_bytes_per_prefetch', 0)} "
            f"segments={len(getattr(offloader, '_copy_segments', ()))} "
            f"tail={int(is_tail_prefetch)}"
        ):
            assert self.runtime is not None, "Runtime controller not initialized"
            previous_owner = self.runtime.begin_prefetch(layer_idx)
            if previous_owner is not None:
                previous = self.module_offloaders[previous_owner.unit_idx]
                previous.ensure_cpu_master_freshness()
                previous.release_runtime_buffer_tracking()
            self.runtime.mark_prefetch_started(
                layer_idx,
                in_capture=offloader.start_onload_to_static(
                    allow_paced_chunking=is_tail_prefetch,
                ),
            )

    def join_after_forward(self):
        """Join copy_stream after model forward completes.

        Call this after the model forward pass but before CUDA graph capture
        ends. This ensures copy_stream is rejoined for any prefetches started
        during the forward pass.

        We join ALL layers that have capture-started prefetches, meaning their
        prefetch was started during capture but not yet waited on (joined).
        This handles both full and piecewise cudagraph modes correctly:
        - Full mode: joins any wraparound prefetches started by later layers
        - Piecewise mode: joins only layers prefetched by THIS subgraph's layers
        """
        if not self.module_offloaders or self.runtime is None:
            return
        # Join all layers whose prefetch was started in capture but not waited on
        for runtime_unit in self.runtime.pending_capture_prefetches():
            torch.cuda.current_stream().wait_event(
                self.module_offloaders[runtime_unit.unit_idx]._copy_done_event
            )
            ext.flush_transfer_timings(self, skip_query=True)
            self.runtime.mark_waited(runtime_unit.unit_idx)

    @property
    def gates_collectives(self) -> bool:
        return self.comm_aware

    @contextmanager
    def gate_h2d_for_collective(self) -> Generator[None, None, None]:
        """Gate paced copies while a TP collective is active on the GPU."""
        if not self.comm_aware or torch.cuda.is_current_stream_capturing():
            yield
            return
        with self.tail_copy_scheduler.gate_for_collective():
            yield

    def post_init(self):
        """Allocate static buffer pool and start initial prefetches.

        Note: Parameters have already been offloaded to CPU during wrap_modules()
        (in _CpuParamOffloader.__init__), so GPU memory is available for the
        static buffer pool.
        """
        self._static_runtime_buffer_bytes = 0
        # Sync CPU storage with current param.data BEFORE collecting param info.
        # This is needed because process_weights_after_loading may have:
        # 1. Transformed weights (quantization, transpose, etc.)
        # 2. Created new CPU tensors via device_loading_context
        # Our _cpu_storage would be stale otherwise.
        for offloader in self.module_offloaders:
            offloader.sync_cpu_storage()

        module_param_infos, module_storage_group_infos, device = (
            ext.collect_module_buffer_infos(self.module_offloaders)
        )
        if device is None:
            return  # No modules to offload

        runtime_buffer_bytes = ext.allocate_runtime_buffers(
            self, device, module_param_infos, module_storage_group_infos
        )

        # Collect offloaded bytes
        for offloader in self.module_offloaders:
            offloader.post_init()
            self.total_offloaded_bytes += offloader.offloaded_bytes

        self._static_runtime_buffer_bytes = runtime_buffer_bytes
        logger.info_once(
            f"[PrefetchOffloader] Initialized {len(self.module_offloaders)} modules. "
            f"Total GPU memory saved: {self.total_offloaded_bytes / 1e9:.4f} GB, "
            f"Static runtime buffers: "
            f"{self.static_runtime_buffer_bytes / 1e9:.4f} GB "
            f"(group_size={self.group_size}, num_in_group={self.num_in_group}, "
            f"prefetch_step={self.prefetch_step}, mode={self.mode})"
        )

        assert self.runtime is not None, "Runtime controller not initialized"
        log_prefetch_manifest(
            getattr(self, "_diagnostic_plan_units", ()),
            self.runtime,
            self.module_offloaders,
            getattr(self, "_diagnostic_modules", ()),
            group_size=self.group_size,
            num_in_group=self.num_in_group,
            prefetch_step=self.prefetch_step,
            selectors=getattr(self, "offload_selectors", set()),
            include_names=getattr(self, "offload_params", set()),
            comm_aware=self.comm_aware,
            total_offloaded_bytes=self.total_offloaded_bytes,
            runtime_buffer_bytes=self.static_runtime_buffer_bytes,
        )

        self._start_initial_prefetches()

    @property
    def static_runtime_buffer_bytes(self) -> int:
        return self._static_runtime_buffer_bytes

    # ---- Lifecycle / instrumentation: thin wrappers around prefetch_offloader_ext ----

    def _start_initial_prefetches(self) -> None:
        ext.start_initial_prefetches(self)

    def reset_runtime_state(self) -> None:
        ext.reset_runtime_state(self)

    def begin_forward_stats(self) -> None:
        ext.begin_forward_stats(self)

    def end_forward_stats(self) -> None:
        ext.end_forward_stats(self)


class _ModuleOffloader:
    """Manages offloading for a single module.

    Uses static buffers from a shared pool instead of dynamic allocation.
    """

    # See PrefetchOffloader.comm_aware.
    comm_aware: bool = False

    def __init__(
        self,
        mode: str,
        module: nn.Module,
        copy_stream: torch.cuda.Stream,
        tail_copy_scheduler: TailCopyScheduler,
        whitelist_param_names: list[str],
        layer_idx: int,
        module_index: int,
        transfer_stats: PrefetchTransferStats,
        comm_aware: bool = False,
    ):
        self.mode = mode
        self.module = module
        self.device = next(module.parameters()).device
        self.copy_stream = copy_stream
        self._tail_copy_scheduler = tail_copy_scheduler
        self.comm_aware = comm_aware
        self.layer_idx = layer_idx
        self.module_index = module_index
        self.offloaded_bytes = 0
        self.transfer_stats = transfer_stats
        self._copy_segments: tuple[PrefetchCopySegment, ...] = ()
        self._h2d_bytes_per_prefetch = 0

        # Event to signal when H2D copy to static buffer is complete.
        # Used for per-layer synchronization (both eager and capture modes).
        self._copy_done_event = torch.cuda.Event()

        # Track whether _copy_done_event is valid for eager-mode wait_event.
        # False when: (1) never recorded, or (2) last recorded during a
        # cudagraph capture (events become invalid after capture ends).
        # In these cases we fall back to wait_stream.
        self._event_valid_for_eager = False
        self._copy_done_event_recorded = threading.Event()
        self._copy_done_event_recorded.set()
        self._copy_thread_error: Exception | None = None

        assert self.device != torch.device("cpu"), (
            "Module parameters should not already be on CPU "
            "(offloader handles CPU placement)"
        )

        # Buffer pool and slot (assigned in assign_buffer_slot)
        self._buffer_pool: StaticBufferPool | None = None
        self._buffer_slot_idx: int = 0
        # Three-tier runtime buffer state, finalized by _refresh_runtime_buffer_strategy
        self._slab_param_names: tuple[str, ...] = ()
        self._slab_layout: SlabLayout | None = None
        self._cpu_slab_chunks: tuple[CpuSlabChunk, ...] = ()
        self._gpu_slab: torch.Tensor | None = None
        self._storage_group_infos: tuple[StorageGroupInfo, ...] = ()
        self._storage_group_buffers: list[torch.Tensor] = []
        self._direct_param_names: tuple[str, ...] = ()
        self._direct_buffers: dict[str, torch.Tensor] = {}
        self._direct_buffer_bytes = 0
        self._fallback_reasons: tuple[str, ...] = ()
        self._use_slab_copy = True

        param_dict = dict(self.module.named_parameters())
        assert all(name in param_dict for name in whitelist_param_names), (
            f"Whitelist params {whitelist_param_names} not found in module params "
            f"{list(param_dict.keys())}"
        )

        self._param_offloaders = {
            name: _BaseParamOffloader.create(mode, module=module, param_name=name)
            for name in whitelist_param_names
        }

    def post_init(self):
        """Collect total offloaded bytes (offloading already done in __init__)."""
        for param_offloader in self._param_offloaders.values():
            param_offloader.post_init()
            self.offloaded_bytes += param_offloader.offloaded_bytes
        if should_collect_prefetch_debug_metadata():
            self._copy_segments = build_prefetch_copy_segments(self)
            self._h2d_bytes_per_prefetch = sum(
                segment.num_bytes for segment in self._copy_segments
            )

    def sync_cpu_storage(self):
        """Sync CPU storage with current param.data.

        Called after process_weights_after_loading to ensure _cpu_storage
        contains the final processed weights, not stale pre-loading data.

        Parameters whose underlying nn.Parameter was deleted by
        process_weights_after_loading (e.g. transient KV-cache scale params)
        are pruned from self._param_offloaders so they do not participate in
        buffer-pool allocation or prefetching.
        """
        for param_offloader in self._param_offloaders.values():
            param_offloader.sync_cpu_storage()

        # Remove offloaders whose parameter was deleted during
        # process_weights_after_loading (e.g. k_scale / v_scale).
        deleted = [
            name
            for name, offloader in self._param_offloaders.items()
            if getattr(offloader, "_param_deleted", False)
        ]
        if deleted:
            logger.debug(
                "Pruning %d transient offloaded param(s) that were deleted "
                "by process_weights_after_loading: %s",
                len(deleted),
                deleted,
            )
            for name in deleted:
                del self._param_offloaders[name]

        ext.refresh_runtime_buffer_strategy(self)

    @property
    def uses_slab_buffers(self) -> bool:
        return bool(self._slab_param_names)

    @property
    def uses_storage_group_fallback(self) -> bool:
        return bool(self._storage_group_infos)

    @property
    def uses_direct_fallback(self) -> bool:
        return bool(self._direct_param_names)

    @property
    def storage_group_infos(self) -> tuple[StorageGroupInfo, ...]:
        return self._storage_group_infos

    @property
    def direct_buffer_bytes(self) -> int:
        return self._direct_buffer_bytes

    def get_param_infos(self) -> list[ParamInfo]:
        """Get parameter metadata for buffer pool allocation.

        Note: sync_cpu_storage() must be called before this method to ensure
        _cpu_storage reflects the final processed weights (after quantization).
        """
        assert self.uses_slab_buffers, "No slab-backed parameters for this module"
        infos: list[ParamInfo] = []
        for name in self._slab_param_names:
            cpu_storage = self._param_offloaders[name]._cpu_storage
            assert cpu_storage is not None, "CPU storage not initialized"
            infos.append(
                ParamInfo(
                    name=name,
                    shape=tuple(cpu_storage.shape),
                    stride=tuple(cpu_storage.stride()),
                    dtype=cpu_storage.dtype,
                )
            )
        return infos

    def assign_buffer_slot(
        self,
        pool: StaticBufferPool | None,
        storage_group_pool: StorageGroupBufferPool | None,
        slot_idx: int,
    ):
        """Assign this module to a buffer slot in the pool.

        Also assigns static GPU buffers to each parameter offloader,
        which moves the parameter data to point to the GPU buffer.
        """
        ext.assign_module_buffer_slot(self, pool, storage_group_pool, slot_idx)

    def start_onload_to_static(
        self,
        *,
        allow_paced_chunking: bool = False,
    ) -> bool:
        """Start async copy from CPU storage to GPU buffer.

        Uses event-based forking to join copy_stream to CUDA graph capture.
        This ensures H2D copies are properly captured when recording a graph.

        IMPORTANT: We must wait for the compute stream before copying, because
        the previous layer's forward may still be using the buffer (GPU ops are
        async). Without this sync, we could overwrite the buffer while it's
        being read.
        """
        return run_onload_to_static(self, allow_paced_chunking=allow_paced_chunking)

    def wait_until_copy_done_event_recorded(self) -> None:
        self._copy_done_event_recorded.wait()
        if self._copy_thread_error is not None:
            raise RuntimeError(
                "Paced prefetch H2D copy thread failed."
            ) from self._copy_thread_error

    def ensure_cpu_master_freshness(self) -> None:
        for offloader in self._param_offloaders.values():
            offloader.ensure_cpu_master_freshness()

    def release_runtime_buffer_tracking(self) -> None:
        for offloader in self._param_offloaders.values():
            offloader.release_runtime_buffer_tracking()

    def reset_runtime_tracking(self) -> None:
        """Clear transient runtime metadata before restarting prefetch."""
        self._event_valid_for_eager = False
        self.wait_until_copy_done_event_recorded()
        self.release_runtime_buffer_tracking()


class _BaseParamOffloader(ABC):
    """Base class for parameter offloading strategies."""

    # CPU storage for offloaded parameters (set by subclasses)
    _cpu_storage: torch.Tensor | None
    # GPU buffer reference (set by subclasses when using static buffers)
    _gpu_buffer: torch.Tensor | None

    @staticmethod
    def create(mode: str, **kwargs) -> "_BaseParamOffloader":
        """Factory method to create appropriate offloader for mode."""
        if mode == "cpu":
            return _CpuParamOffloader(**kwargs)
        else:
            raise ValueError(f"Unknown offload mode: {mode}")

    def __init__(self, module: nn.Module, param_name: str):
        self._module = module
        self._param_name = param_name
        self.offloaded_bytes = 0
        self._cpu_storage = None
        self._gpu_buffer = None

    @property
    def _param(self) -> nn.Parameter:
        """Get the parameter being offloaded.

        Supports dotted names (e.g. 'self_attn.qkv_proj.weight') by
        traversing the module hierarchy.
        """
        obj: Any = self._module
        for attr in self._param_name.split("."):
            obj = getattr(obj, attr)
        return obj

    def post_init(self):
        """Initialize offloading (move parameter to storage)."""
        return

    # ---- CPU master tracking hooks (overridden by _CpuParamOffloader) ----

    def mark_cpu_master_synced(self) -> None:
        return

    def ensure_cpu_master_freshness(self) -> None:
        return

    def release_runtime_buffer_tracking(self) -> None:
        return

    def mark_cpu_master_stale(self, reason: str) -> None:
        return

    def sync_cpu_master_from_runtime(self) -> None:
        return

    @abstractmethod
    def sync_cpu_storage(self) -> None:
        """Sync CPU storage with current param.data.

        Called after process_weights_after_loading to update _cpu_storage
        with the final processed weights.
        """
        pass

    @abstractmethod
    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None:
        """Point parameter data to GPU static buffer."""
        pass


class _CpuParamOffloader(_BaseParamOffloader):
    """Offload parameter to pinned CPU memory.

    Uses GPU static buffers as the actual parameter, with CPU storage
    kept separately. This ensures torch.compile sees GPU tensors at trace time.

    The offloading happens in two phases:
    1. __init__() - copies GPU data to CPU, frees GPU memory immediately
    2. assign_static_buffer() - points param.data to GPU static buffer
    """

    def __init__(self, module: nn.Module, param_name: str):
        super().__init__(module, param_name)
        self._cpu_storage: torch.Tensor | None = None
        self._gpu_buffer: torch.Tensor | None = None  # Store reference to GPU buffer
        self._cpu_master_stale: bool = False
        self._cpu_master_stale_reason: str | None = None
        self._expected_gpu_buffer_version: int | None = None
        self._expected_gpu_buffer_ptr: int | None = None
        # Set to True if the underlying nn.Parameter was deleted by
        # process_weights_after_loading (e.g. transient KV-cache scale params
        # such as k_scale/v_scale created by BaseKVCacheMethod.create_weights
        # and deleted after copying into permanent _k_scale buffers).
        self._param_deleted: bool = False

        # Offload to CPU immediately to free GPU memory during model loading
        self._offload_to_cpu_internal()

    def _offload_to_cpu_internal(self):
        """Copy parameter data to pinned CPU storage and free GPU memory.

        This replaces param.data with CPU storage, allowing weight loading
        to continue writing to CPU memory. GPU memory is freed when the
        original GPU tensor is garbage collected.
        """
        param = self._param
        pin_memory = should_pin_memory()

        # Create pinned CPU storage and copy current GPU data
        self._cpu_storage = torch.empty_strided(
            size=param.data.size(),
            stride=param.data.stride(),
            dtype=param.data.dtype,
            layout=param.data.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._cpu_storage.copy_(param.data)

        self.offloaded_bytes = (
            self._cpu_storage.numel() * self._cpu_storage.element_size()
        )

        # Point param.data to CPU storage - this allows weight loading to work
        # and frees GPU memory when the original GPU tensor is garbage collected
        param.data = self._cpu_storage

    def _update_cpu_storage_from_param(self) -> None:
        """Update _cpu_storage from current param.data, ensuring pinned memory.

        After process_weights_after_loading, device_loading_context creates
        non-pinned CPU tensors via `p.data = p.data.to("cpu")`. Using
        non-pinned memory with `copy_(src, non_blocking=True)` causes CUDA to
        perform a stream synchronization before the copy, breaking the
        event-based fork synchronization and potentially allowing the copy
        to overwrite the GPU buffer while the compute stream still reads it.

        This method ensures _cpu_storage always uses pinned memory when
        available, re-pinning if necessary.
        """
        param = self._param

        if param.data.device.type == "cpu":
            if (
                self._gpu_buffer is not None
                and param.data.data_ptr() == self._gpu_buffer.data_ptr()
            ):
                # The runtime parameter still aliases the GPU buffer; refresh
                # CPU master in place so we do not overwrite live GPU memory.
                assert self._cpu_storage is not None
                if self._cpu_storage.data_ptr() == param.data.data_ptr():
                    self._cpu_storage = torch.empty_strided(
                        size=param.data.size(),
                        stride=param.data.stride(),
                        dtype=param.data.dtype,
                        layout=param.data.layout,
                        device="cpu",
                        pin_memory=should_pin_memory(),
                    )
                self._cpu_storage.copy_(param.data)
            elif should_pin_memory() and not param.data.is_pinned():
                pinned = torch.empty_strided(
                    size=param.data.size(),
                    stride=param.data.stride(),
                    dtype=param.data.dtype,
                    layout=param.data.layout,
                    device="cpu",
                    pin_memory=True,
                )
                pinned.copy_(param.data)
                self._cpu_storage = pinned
            else:
                self._cpu_storage = param.data
        else:
            # param.data is on GPU - copy to existing CPU storage
            assert self._cpu_storage is not None
            self._cpu_storage.copy_(param.data)

    def assign_static_buffer(self, gpu_buffer: torch.Tensor) -> None:
        """Point parameter data to GPU static buffer.

        This is called after weight loading AND process_weights_after_loading
        complete. At this point:
        - param.data may have been replaced by device_loading_context
          (which creates new CPU tensors after quantization processing)
        - We need to update _cpu_storage to point to current param.data
          so that prefetch copies the processed weights, not stale data
        - Then point param.data to the GPU buffer for torch.compile
        """
        assert self._cpu_storage is not None, (
            "_offload_to_cpu_internal() must be called before assign_static_buffer()"
        )

        # Get current parameter (may have been replaced by
        # process_weights_after_loading)
        param = self._param

        # Update _cpu_storage to current param.data. This is critical because:
        # 1. process_weights_after_loading may transform weights (quantization)
        # 2. device_loading_context creates NEW CPU tensors when moving back
        # 3. Our old _cpu_storage would have pre-processed or stale data
        self._update_cpu_storage_from_param()

        # Store reference to GPU buffer for use in start_onload
        self._gpu_buffer = gpu_buffer
        self._cpu_master_stale = False
        self._cpu_master_stale_reason = None
        self._expected_gpu_buffer_version = None
        self._expected_gpu_buffer_ptr = None

        # Point parameter to static GPU buffer - this is what torch.compile sees
        param.data = gpu_buffer

    def sync_cpu_storage(self) -> None:
        """Sync CPU storage with current param.data.

        Called after process_weights_after_loading to update _cpu_storage
        with the final processed weights. This is critical because:
        1. process_weights_after_loading may transform weights (quantization)
        2. device_loading_context creates NEW CPU tensors when moving back
        3. Our old _cpu_storage would have pre-processed or stale data

        If the parameter no longer exists on the module (e.g. transient
        KV-cache scale parameters such as k_scale/v_scale that are created
        by BaseKVCacheMethod.create_weights() and then deleted by
        process_weights_after_loading() after copying their values into
        permanent _k_scale buffers), the offloader marks itself as deleted
        and skips the sync.  The caller (_ModuleOffloader.sync_cpu_storage)
        is responsible for removing these stale entries.
        """
        try:
            self._update_cpu_storage_from_param()
        except AttributeError:
            # The parameter was deleted by process_weights_after_loading.
            # Drop the now-stale CPU storage so this offloader can be pruned.
            self._param_deleted = True
            self._cpu_storage = None

    def post_init(self):
        """No-op: offloading done in offload_to_cpu/assign_static_buffer."""
        pass

    # ---- CPU master tracking ----

    def mark_cpu_master_synced(self) -> None:
        if self._gpu_buffer is None:
            return
        self._cpu_master_stale = False
        self._cpu_master_stale_reason = None
        self._expected_gpu_buffer_version = self._gpu_buffer._version
        self._expected_gpu_buffer_ptr = self._gpu_buffer.data_ptr()

    def mark_cpu_master_stale(self, reason: str) -> None:
        self._cpu_master_stale = True
        self._cpu_master_stale_reason = reason

    def sync_cpu_master_from_runtime(self) -> None:
        self._update_cpu_storage_from_param()
        self.mark_cpu_master_synced()

    def release_runtime_buffer_tracking(self) -> None:
        self._expected_gpu_buffer_version = None
        self._expected_gpu_buffer_ptr = None

    def ensure_cpu_master_freshness(self) -> None:
        if self._cpu_master_stale:
            reason = self._cpu_master_stale_reason or "unknown reason"
            raise RuntimeError(
                f"Offloaded parameter {self._param_name} CPU master copy is "
                f"stale: {reason}."
            )
        if self._gpu_buffer is None or self._expected_gpu_buffer_version is None:
            return

        param = self._param
        if param.data.data_ptr() != self._expected_gpu_buffer_ptr:
            self._cpu_master_stale = True
            self._cpu_master_stale_reason = (
                "runtime parameter no longer points to the managed runtime buffer"
            )
            raise RuntimeError(
                f"Offloaded parameter {self._param_name} no longer points to the "
                "managed runtime buffer; CPU master copy is stale."
            )
        if self._gpu_buffer._version != self._expected_gpu_buffer_version:
            self._cpu_master_stale = True
            self._cpu_master_stale_reason = (
                "runtime parameter was mutated after CPU master synchronization"
            )
            raise RuntimeError(
                f"Offloaded parameter {self._param_name} was mutated after CPU "
                "master synchronization; CPU master copy is stale."
            )
