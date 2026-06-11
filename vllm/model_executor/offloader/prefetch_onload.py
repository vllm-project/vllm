# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H2D copy orchestration for one prefetched module.

Encapsulates the multi-tier (slab / storage-group / direct) buffer copy
sequence so the core :class:`prefetch._ModuleOffloader` can keep a small,
upstream-shaped surface.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.offloader.base import should_pin_memory
from vllm.model_executor.offloader.prefetch_diagnostics import (
    PrefetchTransferStats,
    should_log_transfer_stats,
)
from vllm.model_executor.offloader.prefetch_helpers import nvtx_range
from vllm.model_executor.offloader.prefetch_tail_copy import (
    TAIL_PREFETCH_H2D_CHUNK_BYTES,
    TailCopyJob,
    TensorCopyItem,
    iter_chunked_tensor_views,
)

if TYPE_CHECKING:
    from vllm.model_executor.offloader.prefetch import _ModuleOffloader


def _make_copy_recorder(
    module_offloader: "_ModuleOffloader",
    *,
    in_capture: bool,
    log_stats: bool,
):
    """Return a function that copies and (optionally) records timing events."""
    transfer_stats: PrefetchTransferStats = module_offloader.transfer_stats
    copy_stream = module_offloader.copy_stream

    def copy_and_record(dst: torch.Tensor, src: torch.Tensor, num_bytes: int) -> None:
        if not log_stats:
            dst.copy_(src, non_blocking=True)
            return
        if in_capture:
            dst.copy_(src, non_blocking=True)
            transfer_stats.record_copy(num_bytes)
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(copy_stream)
        dst.copy_(src, non_blocking=True)
        end_event.record(copy_stream)
        transfer_stats.record_copy(num_bytes, start_event, end_event)

    return copy_and_record


def run_onload_to_static(
    module_offloader: "_ModuleOffloader",
    *,
    allow_paced_chunking: bool,
) -> bool:
    """Drive one module's CPU→GPU prefetch and return ``in_capture``.

    See :meth:`_ModuleOffloader.start_onload_to_static` for the full contract.
    """
    in_capture = torch.cuda.is_current_stream_capturing()

    with nvtx_range(
        "weight_offload.h2d_copy "
        f"unit={module_offloader.layer_idx} "
        f"slot={module_offloader._buffer_slot_idx} "
        f"capture={int(in_capture)} paced={int(allow_paced_chunking)}"
    ):
        module_offloader.wait_until_copy_done_event_recorded()
        module_offloader._copy_done_event_recorded.clear()
        module_offloader._copy_thread_error = None

        log_stats = should_log_transfer_stats()
        copy_and_record = _make_copy_recorder(
            module_offloader, in_capture=in_capture, log_stats=log_stats
        )

        use_paced_chunks = allow_paced_chunking and not in_capture and not log_stats
        paced_copy_items: list[TensorCopyItem] = []

        def copy_or_defer(dst: torch.Tensor, src: torch.Tensor, num_bytes: int) -> None:
            if not use_paced_chunks:
                copy_and_record(dst, src, num_bytes)
                return
            paced_copy_items.extend(
                iter_chunked_tensor_views(
                    dst, src, num_bytes, TAIL_PREFETCH_H2D_CHUNK_BYTES
                )
            )

        # Fork: record event on compute stream, copy_stream waits on it.
        # This joins copy_stream to any active CUDA graph capture.
        fork_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(fork_event)
        if not use_paced_chunks:
            module_offloader.copy_stream.wait_event(fork_event)

        with torch.cuda.stream(module_offloader.copy_stream):
            for offloader in module_offloader._param_offloaders.values():
                offloader.ensure_cpu_master_freshness()

            if module_offloader.uses_slab_buffers and module_offloader._use_slab_copy:
                cpu_slab = module_offloader._cpu_slab
                gpu_slab = module_offloader._gpu_slab
                assert cpu_slab is not None and gpu_slab is not None
                assert not should_pin_memory() or cpu_slab.is_pinned(), (
                    "CPU slab is not pinned! "
                    "non_blocking=True H2D copy from non-pinned memory "
                    "causes stream synchronization that breaks "
                    "event-based fork synchronization."
                )
                copy_or_defer(gpu_slab, cpu_slab, cpu_slab.numel())
            elif module_offloader.uses_slab_buffers:
                for name in module_offloader._slab_param_names:
                    p = module_offloader._param_offloaders[name]
                    cpu_storage = p._cpu_storage
                    gpu_buffer = p._gpu_buffer
                    assert cpu_storage is not None and gpu_buffer is not None
                    assert not should_pin_memory() or cpu_storage.is_pinned(), (
                        f"CPU storage for {name} is not pinned!"
                    )
                    copy_or_defer(
                        gpu_buffer,
                        cpu_storage,
                        cpu_storage.numel() * cpu_storage.element_size(),
                    )

            if module_offloader.uses_storage_group_fallback:
                for group_info, gpu_buffer in zip(
                    module_offloader._storage_group_infos,
                    module_offloader._storage_group_buffers,
                ):
                    cpu_source = group_info.cpu_source
                    assert not should_pin_memory() or cpu_source.is_pinned(), (
                        "CPU storage-group source is not pinned!"
                    )
                    copy_or_defer(
                        gpu_buffer,
                        cpu_source,
                        cpu_source.numel() * cpu_source.element_size(),
                    )

            for name in module_offloader._direct_param_names:
                p = module_offloader._param_offloaders[name]
                cpu_storage = p._cpu_storage
                gpu_buffer = p._gpu_buffer
                assert cpu_storage is not None and gpu_buffer is not None
                copy_or_defer(
                    gpu_buffer,
                    cpu_storage,
                    cpu_storage.numel() * cpu_storage.element_size(),
                )

            if not paced_copy_items:
                for offloader in module_offloader._param_offloaders.values():
                    offloader.mark_cpu_master_synced()

        if paced_copy_items:
            module_offloader._tail_copy_scheduler.submit(
                TailCopyJob(
                    module_offloader=module_offloader,
                    fork_event=fork_event,
                    copy_items=tuple(paced_copy_items),
                )
            )
        else:
            # Record completion event for _wait_for_layer to use.
            module_offloader._copy_done_event.record(module_offloader.copy_stream)
            module_offloader._copy_done_event_recorded.set()
        # Event is only valid for eager wait_event if recorded outside capture.
        # Events recorded during capture become invalid after capture ends.
        module_offloader._event_valid_for_eager = not in_capture
    return in_capture
