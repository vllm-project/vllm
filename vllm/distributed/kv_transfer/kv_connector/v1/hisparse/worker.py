# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse worker-side host/hot state and data movement."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
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
    KVCacheBlockCopy,
    get_unique_kv_cache_group_id,
)
from vllm.v1.hisparse.runtime import HiSparseCacheHandle, release_pinned_state
from vllm.v1.hisparse.types import SparseKVPageTransfer, SparseKVRowMirror
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    KVCacheConfig,
    KVCacheGroupRole,
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
        for group in self.kv_cache_config.kv_cache_groups:
            if not isinstance(group.kv_cache_spec, HiSparseHotSpec):
                continue
            for cache_name in group.layer_names:
                assert cache_name.endswith(HISPARSE_HOT_SUFFIX)
                layer_name = cache_name[: -len(HISPARSE_HOT_SUFFIX)]
                cache_handles.append(_get_hisparse_cache(forward_context, layer_name))

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

        source_group_id = get_unique_kv_cache_group_id(
            self.kv_cache_config, KVCacheGroupRole.HISPARSE_SOURCE
        )
        source_block_size = self.kv_cache_config.kv_cache_groups[
            source_group_id
        ].kv_cache_spec.block_size
        resident = cache_handles[0].view
        assert resident is not None
        assert source_block_size % resident.block_size == 0
        host_num_blocks = self.kv_cache_config.hisparse_host_num_blocks
        assert host_num_blocks is not None
        try:
            self.initialize(
                cache_handles,
                hot_backing,
                self.vllm_config.scheduler_config.max_num_seqs,
                source_block_size // resident.block_size,
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
        hot_backing: torch.Tensor,
        max_num_reqs: int,
        pages_per_host_block: int,
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
        self.pages_per_host_block = pages_per_host_block
        self.host_num_blocks = host_num_blocks
        self.pinned_host_pools = pinned_host_pools
        self.shared_host_region = shared_host_region
        self.is_host_writer = is_host_writer
        self.dma_stream = (
            torch.cuda.Stream(device=device) if self.is_host_writer else None
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
        self._layer_mirror_callbacks = tuple(
            partial(self._enqueue_layer_mirror, layer_index)
            for layer_index in range(len(cache_handles))
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
        mirror_caches = []
        for cache in self.cache_handles:
            mirror_cache = cache.mirror_staging_cache
            if mirror_cache is None:
                raise RuntimeError("HiSparse prefill mirror staging is not bound.")
            mirror_caches.append(mirror_cache)
        self.mirror_caches = tuple(mirror_caches)

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
        host_write_events = getattr(self, "host_write_events", None)
        if host_write_events is not None:
            self.host_write_event = host_write_events[self._next_host_write_event]
            self._next_host_write_event ^= 1
        current_stream().wait_event(previous_host_write_event)
        self._release_completed_dma_descriptors()
        mirrors = _flatten_row_mirrors(metadata.row_mirrors, request_ids)
        self._set_row_mirrors(mirrors)
        self._row_mirrors_from_resident = metadata.row_mirrors_from_resident
        self._dma_submitted = False
        self._per_layer_mirrored.clear()
        self._submitted_mirror_layers.clear()
        for handle in self.cache_handles:
            handle.decode_batch = False
            handle.all_context_pages_resident = getattr(
                metadata, "all_context_pages_resident", False
            )
            handle.host_mirror_required = False
            handle.num_actual_tokens = 0
            handle.num_decode_tokens = 0
            handle.req_id_per_token = None
            handle.submit_layer_mirror = None
        self._copy_host_blocks(metadata.host_block_copies, previous_host_write_event)
        transfers = (
            metadata.command.page_transfers if metadata.command is not None else []
        )
        if transfers:
            pre_forward_transfers = [
                transfer for transfer in transfers if not transfer.after_forward
            ]
            self._post_forward_transfers = [
                transfer for transfer in transfers if transfer.after_forward
            ]
            self._enqueue_pre_forward_transfers(pre_forward_transfers)
        else:
            self._post_forward_transfers.clear()
        if self.is_host_writer and self._row_mirror_num_rows:
            for handle, callback in zip(
                self.cache_handles, self._layer_mirror_callbacks
            ):
                handle.submit_layer_mirror = callback
        self._pending_invalid_block_ids.extend(metadata.source_block_ids)
        if request_state_indices is not None:
            self.set_request_state_indices(request_state_indices)

    def _copy_host_blocks(
        self,
        host_block_copies: Sequence[KVCacheBlockCopy],
        previous_host_write_event: torch.Event,
    ) -> None:
        if getattr(self, "shared_host_region", None) is not None and host_block_copies:
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
        if self.is_host_writer and self._dma_submitted:
            current_stream().wait_event(self.host_write_event)
            self._dma_submitted = False
        self._release_completed_dma_descriptors()

    def _release_completed_dma_descriptors(self) -> None:
        pending = getattr(self, "_pending_dma_descriptors", None)
        if pending is None:
            return
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

    def _require_row_mirrors(self, num_rows: int) -> tuple[SparseKVRowMirror, ...]:
        mirror_rows = self._row_mirror_num_rows
        from_resident = getattr(self, "_row_mirrors_from_resident", False)
        if mirror_rows < num_rows or (not from_resident and mirror_rows != num_rows):
            raise RuntimeError(
                "HiSparse DMA metadata does not match the forward: "
                f"{mirror_rows} rows for {num_rows} tokens."
            )
        return self._row_mirrors

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
        decode_batch = self.cache_handles[layer_indices[0]].decode_batch
        from_resident = decode_batch or getattr(
            self, "_row_mirrors_from_resident", False
        )
        staging_starts = np.cumsum(row_counts, dtype=np.int64) - row_counts
        for descriptor_offset, layer_index in enumerate(layer_indices):
            cache = self.cache_handles[layer_index]
            source_index = cache.runtime.resident_source_index
            if source_index >= self._row_mirror_source_starts.shape[1]:
                raise RuntimeError("HiSparse row DMA source index is out of range.")
            source_rows = (
                self._row_mirror_source_starts[:, source_index]
                if from_resident
                else staging_starts
            )
            source = (
                self.resident_caches[layer_index]
                if from_resident
                else self.mirror_caches[layer_index]
            )
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
            source_out_of_range = np.any(source_blocks < 0) or np.any(
                source_blocks >= source.shape[0]
            )
            if from_resident:
                source_out_of_range |= np.any(
                    source_row_offsets + row_counts > self.kernel_block_size
                )
            else:
                source_out_of_range |= np.any(
                    source_rows + row_counts > source.shape[0] * self.kernel_block_size
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
        self._require_row_mirrors(handle.num_actual_tokens)
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

    def _enqueue_pre_forward_transfers(
        self, transfers: list[SparseKVPageTransfer]
    ) -> None:
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
                (
                    transfer.destination_block_id * self.pages_per_host_block
                    + transfer.destination_page_offset
                )
                * self.kernel_block_size
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
        active_layer_indices = tuple(
            index
            for index, handle in enumerate(self.cache_handles)
            if handle.num_actual_tokens != 0
        )
        if not active_layer_indices:
            return
        cache = self.cache_handles[active_layer_indices[0]]
        dst_slots = cache.mirror_slot_mapping
        active_handles = [self.cache_handles[index] for index in active_layer_indices]
        mismatch = next(
            (
                (index, handle)
                for index, handle in zip(
                    active_layer_indices[1:], active_handles[1:], strict=True
                )
                if handle.num_actual_tokens != cache.num_actual_tokens
                or handle.num_decode_tokens != cache.num_decode_tokens
                or handle.decode_batch != cache.decode_batch
                or handle.host_mirror_required != cache.host_mirror_required
                or handle.runtime.eager_host_mirror != cache.runtime.eager_host_mirror
                or handle.mirror_slot_mapping is None
                or dst_slots is None
                or handle.mirror_slot_mapping.data_ptr() != dst_slots.data_ptr()
            ),
            None,
        )
        if mismatch is not None:
            index, handle = mismatch
            expected = (
                cache.num_actual_tokens,
                cache.num_decode_tokens,
                cache.decode_batch,
                cache.host_mirror_required,
                cache.runtime.eager_host_mirror,
                None if dst_slots is None else dst_slots.data_ptr(),
            )
            actual = (
                handle.num_actual_tokens,
                handle.num_decode_tokens,
                handle.decode_batch,
                handle.host_mirror_required,
                handle.runtime.eager_host_mirror,
                None
                if handle.mirror_slot_mapping is None
                else handle.mirror_slot_mapping.data_ptr(),
            )
            raise RuntimeError(
                "HiSparse cache layers disagree on mirror metadata: "
                f"cache 0={expected}, cache {index}={actual}."
            )
        if not cache.host_mirror_required:
            return
        if dst_slots is None:
            raise RuntimeError("HiSparse host mirror has no source slot mapping.")
        num_rows = min(
            cache.num_actual_tokens,
            dst_slots.numel(),
        )
        if num_rows == 0:
            return
        self._require_row_mirrors(num_rows)
        if self.is_host_writer:
            expected_layers = set(active_layer_indices)
            if self._per_layer_mirrored:
                if self._per_layer_mirrored != expected_layers:
                    raise RuntimeError(
                        "HiSparse per-layer DMA did not mirror every active layer: "
                        f"expected {sorted(expected_layers)}, got "
                        f"{sorted(self._per_layer_mirrored)}."
                    )
                submitted_layers = getattr(
                    self, "_submitted_mirror_layers", expected_layers
                )
                pending_layers = tuple(sorted(expected_layers - submitted_layers))
                if pending_layers:
                    self._enqueue_row_dma(
                        pending_layers,
                        ready_event=ready_event,
                    )
                    self._submitted_mirror_layers.update(pending_layers)
            else:
                self._enqueue_row_dma(active_layer_indices, ready_event=ready_event)
        num_decode_tokens = min(cache.num_decode_tokens, num_rows)
        if num_decode_tokens:
            assert cache.req_id_per_token is not None
            for handle in active_handles:
                if handle.runtime.is_group_leader:
                    handle.runtime.invalidate_written_slots(
                        dst_slots[:num_decode_tokens],
                        cache.req_id_per_token[:num_decode_tokens],
                    )

    def finish_forward(self) -> None:
        compute_stream = current_stream()
        self._forward_ready_event.record()
        for handle in self.cache_handles:
            handle.submit_layer_mirror = None
        transfers = self._post_forward_transfers
        self._post_forward_transfers = []
        self._enqueue_host_mirror(self._forward_ready_event)
        if self.cache_handles[0].runtime.eager_host_mirror:
            self._record_transfer_completion(transfers)
        else:
            self._enqueue_transfers(transfers)
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
        dma_stream = getattr(self, "dma_stream", None)
        if dma_stream is not None:
            dma_stream.synchronize()
        release_pinned_state(
            [cache.runtime for cache in self.cache_handles], self.pinned_host_pools
        )
        self._initialized = False
