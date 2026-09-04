# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse worker-side host/hot state and data movement."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.utils.torch_utils import current_stream
from vllm.v1.core.kv_cache_utils import (
    HISPARSE_HOT_SUFFIX,
    HISPARSE_RESIDENT_SUFFIX,
    KVCacheBlockCopy,
)
from vllm.v1.hisparse.runtime import HiSparseCacheHandle, release_pinned_state
from vllm.v1.hisparse.types import SparseKVPageTransfer, SparseKVRowMirror
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    KVCacheConfig,
)
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnectorMetadata,
    )
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


class _DMADescriptors(NamedTuple):
    src: torch.Tensor
    dst: torch.Tensor
    sizes: torch.Tensor
    src_np: np.ndarray
    dst_np: np.ndarray
    sizes_np: np.ndarray


@dataclass
class _SlotMappingStaging:
    stream: torch.Stream
    event: torch.Event
    slots: torch.Tensor
    candidates: tuple[SparseKVRowMirror, ...] = ()
    num_tokens: int = 0
    source_index: int = 0


def _allocate_dma_descriptors(size: int) -> _DMADescriptors:
    src, dst, sizes = (torch.empty(size, dtype=torch.int64) for _ in range(3))
    return _DMADescriptors(
        src,
        dst,
        sizes,
        src.numpy(),
        dst.numpy(),
        sizes.numpy(),
    )


def _get_hisparse_cache(
    forward_context: dict[str, Any], layer_name: str
) -> HiSparseCacheHandle:
    attention_layer = forward_context[layer_name]
    hisparse_cache = attention_layer.hisparse_cache
    assert hisparse_cache is not None
    return hisparse_cache


def _flatten_row_mirrors(
    row_mirrors: Mapping[str, tuple[SparseKVRowMirror, ...]],
    request_ids: Sequence[str] | None,
) -> tuple[SparseKVRowMirror, ...]:
    ordered_ids = row_mirrors if request_ids is None else request_ids
    return tuple(
        mirror
        for request_id in ordered_ids
        for mirror in row_mirrors.get(request_id, ())
    )


def _select_written_row_mirrors(
    candidates: tuple[SparseKVRowMirror, ...],
    source_slots: np.ndarray,
    source_index: int,
) -> tuple[SparseKVRowMirror, ...]:
    """Select and coalesce GPU-written rows from a scheduler-owned envelope."""
    source_slots = source_slots[source_slots >= 0]
    if source_slots.size == 0:
        return ()
    if not candidates:
        raise RuntimeError("HiSparse GPU slots have no scheduler mapping envelope.")
    starts = np.asarray([mirror.source_starts for mirror in candidates], dtype=np.int64)
    counts = np.fromiter((mirror.num_rows for mirror in candidates), dtype=np.int64)
    destinations = np.fromiter(
        [mirror.destination_start for mirror in candidates], dtype=np.int64
    )
    if source_index >= starts.shape[1]:
        raise RuntimeError("HiSparse row-mirror source index is out of range.")
    source_starts = starts[:, source_index]
    order = np.argsort(source_starts)
    matches = np.searchsorted(source_starts[order], source_slots, side="right") - 1
    candidate_ids = order[np.maximum(matches, 0)]
    offsets = source_slots - source_starts[candidate_ids]
    valid = (matches >= 0) & (offsets < counts[candidate_ids])
    if not np.all(valid):
        # A page the scheduler could not map has no host destination, so its
        # rows are simply not mirrorable this step.
        source_slots = source_slots[valid]
        candidate_ids = candidate_ids[valid]
        offsets = offsets[valid]
        if source_slots.size == 0:
            return ()
    sources = starts[candidate_ids] + offsets[:, None]
    destinations = destinations[candidate_ids] + offsets
    boundaries = (
        np.flatnonzero(
            (candidate_ids[1:] != candidate_ids[:-1])
            | (np.diff(destinations) != 1)
            | np.any(np.diff(sources, axis=0) != 1, axis=1)
        )
        + 1
    )
    run_starts = np.concatenate(([0], boundaries))
    run_ends = np.concatenate((boundaries, [destinations.size]))
    return tuple(
        SparseKVRowMirror(
            source_starts=tuple(int(value) for value in sources[start]),
            destination_start=int(destinations[start]),
            num_rows=int(end - start),
        )
        for start, end in zip(run_starts, run_ends, strict=True)
    )


def _is_hisparse_host_writer(
    shared_host_region: SharedOffloadRegion | None,
) -> bool:
    return shared_host_region is None or get_tensor_model_parallel_rank() == 0


def _create_hisparse_host_events(
    shared_host_region: SharedOffloadRegion | None,
    is_host_writer: bool,
    device: torch.device,
) -> tuple[torch.Event, torch.Event]:
    if shared_host_region is None:
        return torch.Event(), torch.Event()

    events: tuple[torch.Event, torch.Event] | None = None
    ipc_handles = None
    if is_host_writer:
        events = (
            torch.cuda.Event(interprocess=True),
            torch.cuda.Event(interprocess=True),
        )
        stream = current_stream()
        for event in events:
            event.record(stream)
        ipc_handles = tuple(event.ipc_handle() for event in events)
    ipc_handles = get_tp_group().broadcast_object(ipc_handles, src=0)
    if events is None:
        events = (
            torch.cuda.Event.from_ipc_handle(device, ipc_handles[0]),
            torch.cuda.Event.from_ipc_handle(device, ipc_handles[1]),
        )
    return events


class HiSparseConnectorWorker:
    """Own HiSparse host/hot state and execute its worker-side transfers."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self._initialized = False

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        forward_context = self.vllm_config.compilation_config.static_forward_context
        cache_handles: list[HiSparseCacheHandle] = []
        cache_layer_names: list[str] = []
        for group in self.kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, HiSparseHotSpec):
                continue
            for cache_name in group.layer_names:
                assert cache_name.endswith(HISPARSE_HOT_SUFFIX)
                layer_name = cache_name[: -len(HISPARSE_HOT_SUFFIX)]
                cache_handles.append(_get_hisparse_cache(forward_context, layer_name))
                cache_layer_names.append(layer_name)

        if not cache_handles:
            raise RuntimeError("HiSparse connector found no hot-cache handles.")
        hot_backings: dict[int, torch.Tensor] = {}
        registered_host_pools: dict[int, torch.Tensor] = {}
        for cache in cache_handles:
            hot_backing = cache.runtime.hot_backing
            hot_backings[hot_backing.untyped_storage().data_ptr()] = hot_backing
            registered_pool = cache.runtime.registered_host_pool
            registered_host_pools[registered_pool.data_ptr()] = registered_pool
        if len(hot_backings) != 1:
            raise RuntimeError("HiSparse hot tensors must share one GPU backing.")
        hot_backing = next(iter(hot_backings.values()))
        pinned_host_pools = list(registered_host_pools.values())

        resident = cache_handles[0].view
        assert resident is not None
        host_num_blocks = self.kv_cache_config.hisparse_host_num_blocks
        assert host_num_blocks is not None
        try:
            self.initialize(
                cache_handles,
                cache_layer_names,
                hot_backing,
                self.vllm_config.scheduler_config.max_num_seqs,
                host_num_blocks,
                hot_backing.device,
                pinned_host_pools,
            )
        except Exception:
            release_pinned_state(
                [cache.runtime for cache in cache_handles],
                pinned_host_pools,
            )
            raise

    def initialize(
        self,
        cache_handles: list[HiSparseCacheHandle],
        cache_layer_names: list[str],
        hot_backing: torch.Tensor,
        max_num_reqs: int,
        host_num_blocks: int,
        device: torch.device,
        pinned_host_pools: list[torch.Tensor],
        *,
        shared_host_region: SharedOffloadRegion | None = None,
        is_host_writer: bool = True,
    ) -> None:
        if self._initialized:
            raise RuntimeError("HiSparse connector worker is already initialized.")
        resident = cache_handles[0].view
        assert resident is not None
        self.kernel_block_size = resident.block_size
        self.host_num_blocks = host_num_blocks
        self.pinned_host_pools = pinned_host_pools
        self.shared_host_region = shared_host_region
        self.is_host_writer = is_host_writer
        self.dma_stream = (
            torch.cuda.Stream(device=device) if self.is_host_writer else None
        )
        self._slot_mapping_staging = None
        if self.is_host_writer:
            max_mirror_rows = (
                self.vllm_config.scheduler_config.max_num_batched_tokens
                + max_num_reqs * (self.vllm_config.num_lookahead_tokens + 1)
            )
            self._slot_mapping_staging = _SlotMappingStaging(
                stream=torch.Stream(device=device),
                event=torch.Event(),
                slots=torch.empty(
                    max_mirror_rows,
                    dtype=torch.int64,
                    pin_memory=True,
                ),
            )
        self._dma_free_descriptors: list[_DMADescriptors] = []
        self._pending_dma_descriptors: deque[tuple[torch.Event, _DMADescriptors]] = (
            deque()
        )
        self._dma_submitted = False
        self._per_layer_mirrored: set[int] = set()
        self._submitted_mirror_layers: set[int] = set()
        self._layer_ready_events = tuple(torch.Event() for _ in cache_handles)
        self._forward_ready_event = torch.Event()
        self._set_row_mirrors(())
        self.cache_layer_names = cache_layer_names
        self.host_write_events = _create_hisparse_host_events(
            shared_host_region, is_host_writer, device
        )
        self.host_write_event = self.host_write_events[1]
        self._next_host_write_event = 0
        self.cache_handles = cache_handles
        self.leader_runtimes = [
            cache.runtime for cache in cache_handles if cache.runtime.is_group_leader
        ]
        request_state_indices = {
            indices.data_ptr(): indices
            for cache in cache_handles
            if (indices := cache.runtime.request_state_indices) is not None
        }
        if len(request_state_indices) != 1:
            raise RuntimeError(
                "HiSparse runtimes must share one request-state mapping."
            )
        self.request_state_indices = next(iter(request_state_indices.values()))
        if self.request_state_indices.numel() != max_num_reqs:
            raise RuntimeError(
                "HiSparse request-state mapping does not match max_num_seqs."
            )
        self.hot_backing = hot_backing
        self._pending_invalid_block_ids: list[int] = []
        self._post_forward_transfers: list[SparseKVPageTransfer] = []
        self._enqueued_transfer_ids: list[int] = []
        self._pending_transfer_events: deque[tuple[torch.Event, tuple[int, ...]]] = (
            deque()
        )
        self._init_dma()
        if self.is_host_writer:
            for layer_index, handle in enumerate(cache_handles):
                handle.submit_layer_mirror = partial(
                    self._enqueue_layer_mirror, layer_index
                )
        self._initialized = True

    def set_request_state_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() > self.request_state_indices.numel():
            raise ValueError(
                "HiSparse request-state mapping exceeds max_num_seqs: "
                f"{indices.numel()} > {self.request_state_indices.numel()}."
            )
        if torch.cuda.is_current_stream_capturing():
            return
        # Attention indexes persistent request state by input-batch row. Refresh
        # that indirection after every batch compaction or reorder.
        self.request_state_indices.fill_(-1)
        self.request_state_indices[: indices.numel()].copy_(indices)
        if self._pending_invalid_block_ids:
            self.invalidate_blocks(self._pending_invalid_block_ids, indices)
            self._pending_invalid_block_ids.clear()

    def _init_dma(self) -> None:
        host_caches = tuple(cache.runtime.host_cache for cache in self.cache_handles)
        resident_caches = []
        for cache in self.cache_handles:
            assert cache.view is not None and cache.slot_mapping is not None
            resident_caches.append(cache.view.cache)
        self.host_caches = host_caches
        self.resident_caches = tuple(resident_caches)

        for resident_cache, host_cache in zip(resident_caches, host_caches):
            row_bytes = resident_cache.shape[-1] * resident_cache.element_size()
            if (
                resident_cache.ndim != 3
                or resident_cache.shape[1] != self.kernel_block_size
                or resident_cache.stride(1) * resident_cache.element_size() != row_bytes
            ):
                raise RuntimeError("HiSparse DMA requires contiguous resident rows.")
            if (
                host_cache.ndim != 2
                or not host_cache.is_contiguous()
                or host_cache.shape[1] * host_cache.element_size() != row_bytes
            ):
                raise RuntimeError("HiSparse DMA requires contiguous host rows.")

    def start_step(
        self,
        metadata: HiSparseConnectorMetadata,
        request_state_indices: torch.Tensor | None,
        request_ids: list[str] | None = None,
    ) -> None:
        previous_host_write_event = self.host_write_event
        self.host_write_event = self.host_write_events[self._next_host_write_event]
        self._next_host_write_event ^= 1
        current_stream().wait_event(previous_host_write_event)
        self._release_completed_dma_descriptors()
        mirrors = _flatten_row_mirrors(metadata.row_mirrors, request_ids)
        if self._slot_mapping_staging is not None:
            self._slot_mapping_staging.candidates = mirrors
        self._set_row_mirrors(mirrors)
        self._dma_submitted = False
        self._clear_forward_mirror_state()
        for handle in self.cache_handles:
            handle.all_context_pages_resident = metadata.all_context_pages_resident
            handle.mirror_from_resident = True
        self._copy_host_blocks(metadata.host_block_copies, previous_host_write_event)
        transfers = (
            metadata.command.page_transfers if metadata.command is not None else []
        )
        self._post_forward_transfers = [
            transfer for transfer in transfers if transfer.after_forward
        ]
        self._submit_transfers(
            [transfer for transfer in transfers if not transfer.after_forward]
        )
        self._pending_invalid_block_ids.extend(metadata.source_block_ids)
        if request_state_indices is not None:
            self.set_request_state_indices(request_state_indices)

    def _clear_forward_mirror_state(self) -> None:
        self._per_layer_mirrored.clear()
        self._submitted_mirror_layers.clear()
        for handle in self.cache_handles:
            handle.decode_batch = False
            handle.host_mirror_required = False
            handle.num_actual_tokens = 0
            handle.num_decode_tokens = 0
            handle.req_id_per_token = None

    def stage_row_mirror_mapping(
        self, slot_mappings: Mapping[str, torch.Tensor], num_tokens: int
    ) -> None:
        if not self.is_host_writer:
            return
        state = self._slot_mapping_staging
        assert state is not None
        mapping = next(
            (
                (
                    slot_mappings[layer_name + HISPARSE_RESIDENT_SUFFIX],
                    handle.runtime.resident_source_index,
                )
                for layer_name, handle in zip(
                    self.cache_layer_names, self.cache_handles, strict=True
                )
                if layer_name + HISPARSE_RESIDENT_SUFFIX in slot_mappings
            ),
            None,
        )
        if mapping is None:
            return
        slots, source_index = mapping
        if slots.ndim != 1:
            raise ValueError("HiSparse requires per-layer slot mappings.")
        start = state.num_tokens
        end = start + num_tokens
        if end > state.slots.shape[0]:
            raise ValueError(
                "HiSparse row mapping exceeds staging capacity: "
                f"{end} > {state.slots.shape[0]}."
            )
        if start and state.source_index != source_index:
            raise ValueError("HiSparse mirror phase mixed resident cache groups.")
        main_stream = current_stream()
        state.stream.wait_stream(main_stream)
        with torch.cuda.stream(state.stream):
            state.slots[start:end].copy_(
                slots[:num_tokens],
                non_blocking=True,
            )
            state.event.record(state.stream)
        state.num_tokens = end
        state.source_index = source_index

    def _resolve_row_mirrors(self, state: _SlotMappingStaging) -> None:
        """Compute the final row mirrors from the staged GPU slot mapping."""
        self._set_row_mirrors(
            _select_written_row_mirrors(
                state.candidates,
                state.slots[: state.num_tokens].numpy(),
                state.source_index,
            )
        )
        state.num_tokens = 0

    def _copy_host_blocks(
        self,
        host_block_copies: Sequence[KVCacheBlockCopy],
        previous_host_write_event: torch.Event,
    ) -> None:
        if self.shared_host_region is not None and host_block_copies:
            if get_tensor_model_parallel_rank() == 0:
                copy_kv_cache_blocks_inplace(
                    self.host_caches,
                    self.host_num_blocks,
                    host_block_copies,
                    previous_host_write_event,
                )
            get_tp_group().barrier()
        else:
            copy_kv_cache_blocks_inplace(
                self.host_caches,
                self.host_num_blocks,
                host_block_copies,
                previous_host_write_event,
            )

    def invalidate_blocks(
        self, block_ids: list[int], request_state_indices: torch.Tensor
    ) -> None:
        """Invalidate recycled host slots in this worker's leader runtimes."""
        if not block_ids:
            return
        device = self.cache_handles[0].runtime.device
        staging = torch.tensor(block_ids, dtype=torch.int32, pin_memory=True)
        blocks = staging.to(device, dtype=torch.int32, non_blocking=True)
        offsets = torch.arange(self.kernel_block_size, dtype=torch.int32, device=device)
        slots = (blocks[:, None] * self.kernel_block_size + offsets[None, :]).flatten()
        sorted_slots = torch.sort(slots).values
        state_indices = request_state_indices.to(device=device, dtype=torch.long)
        for runtime in self.leader_runtimes:
            runtime.invalidate_sorted_slots(sorted_slots, state_indices)

    def reset_hot_state(self) -> None:
        for runtime in self.leader_runtimes:
            runtime.reset_hot_state()

    def finish_step(self) -> None:
        self._finish_mirror_phase()
        transfers = self._post_forward_transfers
        self._post_forward_transfers = []
        self._submit_transfers(transfers)
        if self.is_host_writer and self._dma_submitted:
            current_stream().wait_event(self.host_write_event)
            self._dma_submitted = False
        self._release_completed_dma_descriptors()

    def _release_completed_dma_descriptors(self) -> None:
        pending = self._pending_dma_descriptors
        while pending and pending[0][0].query():
            _, descriptors = pending.popleft()
            self._dma_free_descriptors.append(descriptors)

    def _acquire_dma_descriptors(self, size: int) -> _DMADescriptors:
        for index, descriptors in enumerate(self._dma_free_descriptors):
            if descriptors.src.numel() >= size:
                return self._dma_free_descriptors.pop(index)
        return _allocate_dma_descriptors(size)

    def _submit_dma_descriptors(
        self,
        descriptors: _DMADescriptors,
        descriptor_count: int,
        transfer_ids: tuple[int, ...] = (),
        ready_event: torch.Event | None = None,
    ) -> None:
        stream = self.dma_stream
        assert stream is not None
        if ready_event is None:
            stream.wait_stream(current_stream())
        else:
            stream.wait_event(ready_event)
        completion_event = torch.Event()
        with torch.cuda.stream(stream):
            ops.swap_blocks_batch(
                descriptors.src[:descriptor_count],
                descriptors.dst[:descriptor_count],
                descriptors.sizes[:descriptor_count],
            )
            self.host_write_event.record(stream)
            completion_event.record(stream)
        self._pending_dma_descriptors.append((completion_event, descriptors))
        self._dma_submitted = True
        if transfer_ids:
            self._pending_transfer_events.append((completion_event, transfer_ids))
            self._enqueued_transfer_ids.extend(transfer_ids)

    def _set_row_mirrors(self, mirrors: tuple[SparseKVRowMirror, ...]) -> None:
        self._row_mirrors = mirrors
        self._row_mirror_destination_starts = np.fromiter(
            (mirror.destination_start for mirror in mirrors),
            dtype=np.int64,
            count=len(mirrors),
        )
        self._row_mirror_counts = np.fromiter(
            (mirror.num_rows for mirror in mirrors),
            dtype=np.int64,
            count=len(mirrors),
        )
        if mirrors:
            self._row_mirror_source_starts = np.asarray(
                [mirror.source_starts for mirror in mirrors], dtype=np.int64
            )
            if self._row_mirror_source_starts.ndim != 2:
                raise RuntimeError("HiSparse DMA source mappings must be rectangular.")
        else:
            self._row_mirror_source_starts = np.empty((0, 0), dtype=np.int64)
        self._row_mirror_num_rows = int(self._row_mirror_counts.sum())

    def _enqueue_row_dma(
        self, layer_indices: Sequence[int], ready_event: torch.Event | None = None
    ) -> None:
        if (
            not layer_indices
            or not self._row_mirror_num_rows
            or not self.is_host_writer
        ):
            return
        mirrors = self._row_mirrors
        num_layers = len(layer_indices)
        descriptor_count = len(mirrors) * num_layers
        descriptors = self._acquire_dma_descriptors(descriptor_count)
        destination_starts = self._row_mirror_destination_starts
        row_counts = self._row_mirror_counts
        for descriptor_offset, layer_index in enumerate(layer_indices):
            cache = self.cache_handles[layer_index]
            source_index = cache.runtime.resident_source_index
            if source_index >= self._row_mirror_source_starts.shape[1]:
                raise RuntimeError("HiSparse row DMA source index is out of range.")
            source_rows = self._row_mirror_source_starts[:, source_index]
            source = self.resident_caches[layer_index]
            destination = self.host_caches[layer_index]
            row_bytes = source.shape[-1] * source.element_size()
            if (
                source.stride(1) * source.element_size() != row_bytes
                or destination.shape[1] * destination.element_size() != row_bytes
            ):
                raise RuntimeError("HiSparse row DMA requires contiguous rows.")
            source_blocks, source_row_offsets = np.divmod(
                source_rows, self.kernel_block_size
            )
            source_out_of_range = (
                np.any(source_blocks < 0)
                or np.any(source_blocks >= source.shape[0])
                or np.any(source_row_offsets + row_counts > self.kernel_block_size)
            )
            if source_out_of_range:
                raise RuntimeError("HiSparse row DMA source is out of range.")
            if np.any(destination_starts < 0) or np.any(
                destination_starts + row_counts > destination.shape[0]
            ):
                raise RuntimeError("HiSparse row DMA index is out of range.")
            descriptor_slice = slice(descriptor_offset, descriptor_count, num_layers)
            descriptors.src_np[descriptor_slice] = (
                source.data_ptr()
                + source_blocks * source.stride(0) * source.element_size()
                + source_row_offsets * row_bytes
            )
            descriptors.dst_np[descriptor_slice] = (
                destination.data_ptr() + destination_starts * row_bytes
            )
            descriptors.sizes_np[descriptor_slice] = row_counts * row_bytes

        self._submit_dma_descriptors(
            descriptors, descriptor_count, ready_event=ready_event
        )

    def _enqueue_layer_mirror(self, layer_index: int) -> None:
        handle = self.cache_handles[layer_index]
        if not handle.host_mirror_required:
            return
        if layer_index in self._per_layer_mirrored:
            raise RuntimeError(f"HiSparse layer {layer_index} mirrored twice.")
        state = self._slot_mapping_staging
        if state is not None and state.num_tokens:
            # The staging stream waited on the compute stream before the
            # forward launched, so this only drains pre-forward work that is
            # already queued ahead of the layer kernels: no GPU bubble.
            state.event.synchronize()
            self._resolve_row_mirrors(state)
        self._per_layer_mirrored.add(layer_index)
        next_layer = layer_index + 1
        if (
            next_layer < len(self.cache_handles)
            and self.cache_handles[next_layer].runtime.resident_source_index
            == handle.runtime.resident_source_index
        ):
            return
        ready_event = self._layer_ready_events[layer_index]
        ready_event.record()
        pending_layers = tuple(
            sorted(self._per_layer_mirrored - self._submitted_mirror_layers)
        )
        self._enqueue_row_dma(
            pending_layers,
            ready_event=ready_event,
        )
        self._submitted_mirror_layers.update(pending_layers)

    def _record_transfer_completion(
        self, transfers: list[SparseKVPageTransfer]
    ) -> None:
        if not transfers or not self.is_host_writer:
            return
        stream = self.dma_stream
        assert stream is not None
        completion_event = torch.Event()
        completion_event.record(stream)
        transfer_ids = tuple(transfer.transfer_id for transfer in transfers)
        self._pending_transfer_events.append((completion_event, transfer_ids))
        self._enqueued_transfer_ids.extend(transfer_ids)

    def _submit_transfers(self, transfers: list[SparseKVPageTransfer]) -> None:
        if self.cache_handles[0].runtime.eager_host_mirror:
            self._record_transfer_completion(transfers)
        else:
            self._enqueue_transfers(transfers)

    def _enqueue_transfers(self, transfers: list[SparseKVPageTransfer]) -> None:
        if not transfers or not self.is_host_writer:
            return
        num_layers = len(self.cache_handles)
        descriptor_count = len(transfers) * num_layers
        descriptors = self._acquire_dma_descriptors(descriptor_count)
        destination_rows = np.fromiter(
            (
                transfer.destination_block_id * self.kernel_block_size
                for transfer in transfers
            ),
            dtype=np.int64,
            count=len(transfers),
        )
        source_blocks_by_transfer = np.asarray(
            [transfer.source_block_ids for transfer in transfers], dtype=np.int64
        )
        if source_blocks_by_transfer.ndim != 2:
            raise RuntimeError(
                "HiSparse spill DMA source mappings must be rectangular."
            )
        for layer_index, cache in enumerate(self.cache_handles):
            source_index = cache.runtime.resident_source_index
            if source_index >= source_blocks_by_transfer.shape[1]:
                raise RuntimeError("HiSparse spill DMA source index is out of range.")
            source_blocks = source_blocks_by_transfer[:, source_index]
            source = self.resident_caches[layer_index]
            destination = self.host_caches[layer_index]
            if np.any(source_blocks < 0) or np.any(source_blocks >= source.shape[0]):
                raise RuntimeError("HiSparse spill DMA source is out of range.")
            if np.any(destination_rows < 0) or np.any(
                destination_rows + self.kernel_block_size > destination.shape[0]
            ):
                raise RuntimeError("HiSparse spill DMA destination is out of range.")
            row_bytes = source.shape[-1] * source.element_size()
            descriptor_slice = slice(layer_index, descriptor_count, num_layers)
            descriptors.src_np[descriptor_slice] = (
                source.data_ptr()
                + source_blocks * source.stride(0) * source.element_size()
            )
            descriptors.dst_np[descriptor_slice] = (
                destination.data_ptr() + destination_rows * row_bytes
            )
            descriptors.sizes_np[descriptor_slice] = self.kernel_block_size * row_bytes
        self._submit_dma_descriptors(
            descriptors,
            descriptor_count,
            transfer_ids=tuple(transfer.transfer_id for transfer in transfers),
        )

    def _enqueue_host_mirror(
        self,
        ready_event: torch.Event | None = None,
    ) -> None:
        active = [
            (index, handle)
            for index, handle in enumerate(self.cache_handles)
            if handle.num_actual_tokens != 0
        ]
        if not active:
            return

        def mirror_key(handle: HiSparseCacheHandle) -> tuple:
            slots = handle.mirror_slot_mapping
            return (
                handle.num_actual_tokens,
                handle.num_decode_tokens,
                handle.decode_batch,
                handle.host_mirror_required,
                handle.runtime.eager_host_mirror,
                None if slots is None else slots.data_ptr(),
            )

        keys = {mirror_key(handle) for _, handle in active}
        if len(keys) > 1:
            raise RuntimeError(
                f"HiSparse cache layers disagree on mirror metadata: {list(keys)}."
            )
        cache = active[0][1]
        if not cache.host_mirror_required:
            return
        dst_slots = cache.mirror_slot_mapping
        if dst_slots is None:
            raise RuntimeError("HiSparse host mirror has no source slot mapping.")
        num_rows = min(cache.num_actual_tokens, dst_slots.numel())
        if num_rows == 0:
            return
        if self.is_host_writer:
            expected_layers = {index for index, _ in active}
            if self._per_layer_mirrored and self._per_layer_mirrored != expected_layers:
                raise RuntimeError(
                    "HiSparse per-layer DMA did not mirror every active layer: "
                    f"expected {sorted(expected_layers)}, got "
                    f"{sorted(self._per_layer_mirrored)}."
                )
            pending_layers = tuple(
                sorted(expected_layers - self._submitted_mirror_layers)
            )
            if pending_layers:
                self._enqueue_row_dma(pending_layers, ready_event=ready_event)
                self._submitted_mirror_layers.update(pending_layers)
        assert cache.req_id_per_token is not None
        for _, handle in active:
            if handle.runtime.is_group_leader:
                handle.runtime.invalidate_written_slots(
                    dst_slots[:num_rows],
                    cache.req_id_per_token[:num_rows],
                )

    def _finish_mirror_phase(self, ready_event: torch.Event | None = None) -> None:
        state = self._slot_mapping_staging
        if state is not None and state.num_tokens:
            # Measured free: blocking here costs no throughput (26.7 vs 27.6
            # gen tok/s) and lets every step mirror exactly the written rows.
            state.event.synchronize()
            self._resolve_row_mirrors(state)
        self._enqueue_host_mirror(ready_event)
        self._clear_forward_mirror_state()

    def finish_forward(self) -> None:
        compute_stream = current_stream()
        self._forward_ready_event.record()
        self._finish_mirror_phase(self._forward_ready_event)
        if self.is_host_writer and not self._dma_submitted:
            self.host_write_event.record(compute_stream)

    def take_transfer_updates(self) -> tuple[list[int], list[int]]:
        enqueued = self._enqueued_transfer_ids
        self._enqueued_transfer_ids = []
        completed: list[int] = []
        while self._pending_transfer_events:
            event, transfer_ids = self._pending_transfer_events[0]
            if not event.query():
                break
            self._pending_transfer_events.popleft()
            completed.extend(transfer_ids)
        return enqueued, completed

    def shutdown(self) -> None:
        if not self._initialized:
            return
        if self._slot_mapping_staging is not None:
            self._slot_mapping_staging.stream.synchronize()
        if self.dma_stream is not None:
            self.dma_stream.synchronize()
        release_pinned_state(
            [cache.runtime for cache in self.cache_handles], self.pinned_host_pools
        )
        self._initialized = False
