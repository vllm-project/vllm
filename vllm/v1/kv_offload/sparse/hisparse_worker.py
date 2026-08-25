# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse worker-side host/hot state and data movement."""

from __future__ import annotations

from collections import deque
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import (
    HISPARSE_HOT_SUFFIX,
    HISPARSE_INDEXER_SOURCE_SUFFIX,
    HISPARSE_RESIDENT_SUFFIX,
    get_unique_kv_cache_group_id,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.sparse.base import (
    SparseKVOffloadCommand,
    SparseKVPageTransfer,
)
from vllm.v1.kv_offload.sparse.hisparse_runtime import (
    HiSparseCacheHandle,
    release_pinned_state,
)
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.block_table import BlockTables


def _expand_source_block_ids(
    source_blocks: list[int], blocks_per_kv_block: int, count: int
) -> np.ndarray:
    """Expand logical host block IDs into kernel-page IDs."""
    num_source_blocks = cdiv(count, blocks_per_kv_block)
    logical_ids = np.asarray(source_blocks[:num_source_blocks], dtype=np.int32)
    offsets = np.arange(blocks_per_kv_block, dtype=np.int32)
    return (logical_ids[:, None] * blocks_per_kv_block + offsets).reshape(-1)[:count]


def _get_hisparse_cache(
    forward_context: dict[str, Any], layer_name: str
) -> HiSparseCacheHandle:
    attention_layer = forward_context[layer_name]
    hisparse_cache = attention_layer.hisparse_cache
    assert hisparse_cache is not None
    return hisparse_cache


class HiSparseWorker:
    """Own HiSparse host/hot state and execute its worker-side transfers."""

    def __init__(
        self,
        cache_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        cache_handles: list[HiSparseCacheHandle],
        hot_backing: torch.Tensor,
        max_num_reqs: int,
        max_model_len: int,
        max_concurrent_batches: int,
        blocks_per_kv_block: int,
        host_num_blocks: int,
        source_group_id: int,
        indexer_group_id: int,
        device: torch.device,
        pinned_host_pools: list[torch.Tensor],
        indexer_source_layers: list[Any],
    ) -> None:
        self.cache_pairs = cache_pairs
        kernel_block_size = cache_pairs[0][0].shape[1]
        self.kernel_block_size = kernel_block_size
        self.blocks_per_kv_block = blocks_per_kv_block
        self.host_num_blocks = host_num_blocks
        self.source_group_id = source_group_id
        self.indexer_group_id = indexer_group_id
        self.pinned_host_pools = pinned_host_pools
        self.indexer_source_layers = indexer_source_layers
        capacity = max_num_reqs * cdiv(max_model_len, kernel_block_size)
        self.src_cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.dst_cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.src_gpu = torch.empty(capacity, dtype=torch.int32, device=device)
        self.dst_gpu = torch.empty(capacity, dtype=torch.int32, device=device)
        self._restore_staging_event: torch.Event | None = None
        self.cache_handles = cache_handles
        self.leader_runtimes = [
            cache.runtime for cache in cache_handles if cache.runtime.is_group_leader
        ]
        self.request_state_indices = torch.full(
            (max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        for cache in cache_handles:
            cache.runtime.request_state_indices = self.request_state_indices
        self.hot_backing = hot_backing
        self._block_staging: torch.Tensor | None = None
        self._block_staging_event: torch.Event | None = None
        self._pending_invalid_block_ids: list[int] = []
        self._post_forward_transfers: list[SparseKVPageTransfer] = []
        self._enqueued_transfer_ids: list[int] = []
        self._pending_transfer_events: deque[tuple[torch.Event, tuple[int, ...]]] = (
            deque()
        )
        self._init_backup_plan(device, max_model_len, max_concurrent_batches)

    def set_request_state_indices(self, indices: torch.Tensor) -> None:
        if indices.numel() > self.request_state_indices.numel():
            raise ValueError(
                "HiSparse request-state mapping exceeds max_num_seqs: "
                f"{indices.numel()} > {self.request_state_indices.numel()}."
            )
        if torch.cuda.is_current_stream_capturing():
            return
        self.request_state_indices.fill_(-1)
        self.request_state_indices[: indices.numel()].copy_(indices)
        if self._pending_invalid_block_ids:
            self.invalidate_blocks(self._pending_invalid_block_ids, indices)
            self._pending_invalid_block_ids.clear()

    def _init_backup_plan(
        self,
        device: torch.device,
        max_model_len: int,
        max_concurrent_batches: int,
    ) -> None:
        entries = [cache.runtime.backup_caches() for cache in self.cache_handles]
        hot_caches, host_caches = zip(*entries)
        self.host_caches = (*host_caches, *(source for source, _ in self.cache_pairs))

        layouts = []
        for hot_cache, host_cache in zip(hot_caches, host_caches):
            if host_cache.ndim != 2 or not host_cache.is_contiguous():
                raise RuntimeError("HiSparse host caches must be contiguous 2D.")
            row_bytes = hot_cache.shape[-1] * hot_cache.element_size()
            layouts.append(
                (
                    row_bytes,
                    hot_cache.shape[1],
                    hot_cache.stride(0) * hot_cache.element_size(),
                    hot_cache.shape[0] * hot_cache.shape[1],
                    host_cache.shape[0],
                )
            )
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise RuntimeError(
                "HiSparse all-layer backup requires a uniform cache layout."
            )
        src_block_size, src_block_stride, src_rows = layouts[0][1:4]
        backing_ptr = self.hot_backing.data_ptr()

        self.backup_layer_offsets = torch.tensor(
            [cache.data_ptr() - backing_ptr for cache in hot_caches],
            dtype=torch.int64,
            device=device,
        )
        self.backup_host_cache_ptrs = torch.tensor(
            [cache.data_ptr() for cache in host_caches],
            dtype=torch.uint64,
            device=device,
        )
        self.backup_host_anchor = host_caches[0]
        self.backup_src_block_stride = src_block_stride
        self.backup_src_block_size = src_block_size
        self.backup_src_rows = src_rows
        self.host_write_event = torch.Event()
        self.spill_row_capacity = max_model_len
        spill_staging_count = max_concurrent_batches + 1
        self.spill_src_cpu = torch.empty(
            (
                spill_staging_count,
                len(self.cache_handles),
                self.spill_row_capacity,
            ),
            dtype=torch.int64,
            pin_memory=True,
        )
        self.spill_dst_cpu = torch.empty(
            (spill_staging_count, self.spill_row_capacity),
            dtype=torch.int64,
            pin_memory=True,
        )
        self.spill_src_gpu = torch.empty(
            (len(self.cache_handles), self.spill_row_capacity),
            dtype=torch.int64,
            device=device,
        )
        self.spill_dst_gpu = torch.empty(
            self.spill_row_capacity, dtype=torch.int64, device=device
        )
        self.spill_src_indices_ptrs = torch.tensor(
            [row.data_ptr() for row in self.spill_src_gpu],
            dtype=torch.uint64,
            device=device,
        )
        self._spill_staging_index = 0
        self._spill_staging_events = [torch.Event() for _ in range(spill_staging_count)]

    def apply_scheduler_output(
        self,
        command: SparseKVOffloadCommand | None,
        scheduler_output: SchedulerOutput,
    ) -> None:
        host_copies = [
            copy
            for copy in scheduler_output.kv_cache_block_copies or ()
            if copy.block_pool_id is None
        ]
        copy_kv_cache_blocks_inplace(
            self.host_caches,
            self.host_num_blocks,
            host_copies,
            self.host_write_event,
        )
        transfers = command.page_transfers if command is not None else []
        if transfers:
            self._post_forward_transfers = [
                transfer for transfer in transfers if transfer.after_forward
            ]
            self._enqueue_transfers(
                [transfer for transfer in transfers if not transfer.after_forward]
            )
        else:
            self._post_forward_transfers.clear()
        block_ids = [
            block_id
            for request in scheduler_output.scheduled_new_reqs
            for block_id in request.block_ids[self.source_group_id]
        ]
        for new_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
            if new_block_ids is not None:
                block_ids.extend(new_block_ids[self.source_group_id])

        self._pending_invalid_block_ids.extend(block_ids)
        indexer_ready_req_ids = command.indexer_ready_req_ids if command else ()
        self.restore_prefix(scheduler_output, indexer_ready_req_ids)

    def invalidate_blocks(
        self, block_ids: list[int], request_state_indices: torch.Tensor
    ) -> None:
        """Invalidate recycled host slots in this worker's leader runtimes."""
        if not block_ids:
            return
        device = self.cache_handles[0].runtime.device
        num_blocks = len(block_ids)
        if self._block_staging is None or self._block_staging.shape[0] < num_blocks:
            size = 1 << max(10, (num_blocks - 1).bit_length())
            self._block_staging = torch.empty(size, dtype=torch.long, pin_memory=True)
            self._block_staging_event = None
        if self._block_staging_event is not None:
            self._block_staging_event.synchronize()
        staging = self._block_staging[:num_blocks]
        staging.copy_(torch.from_numpy(np.asarray(block_ids, dtype=np.int64)))
        blocks = staging.to(device, dtype=torch.int32, non_blocking=True)
        if self._block_staging_event is None:
            self._block_staging_event = torch.Event()
        self._block_staging_event.record(torch.accelerator.current_stream(device))
        offsets = torch.arange(self.kernel_block_size, dtype=torch.int32, device=device)
        slots = (blocks[:, None] * self.kernel_block_size + offsets[None, :]).flatten()
        for runtime in self.leader_runtimes:
            runtime.invalidate_slots(slots, request_state_indices)

    def reset_hot_state(self) -> None:
        for runtime in self.leader_runtimes:
            runtime.reset_hot_state()

    def _enqueue_transfers(self, transfers: list[SparseKVPageTransfer]) -> None:
        if not transfers:
            return
        transfers_per_batch = self.spill_row_capacity // self.kernel_block_size
        if transfers_per_batch == 0:
            raise RuntimeError("HiSparse spill staging cannot hold one cache page.")
        offsets = np.arange(self.kernel_block_size, dtype=np.int64)
        for batch_start in range(0, len(transfers), transfers_per_batch):
            batch = transfers[batch_start : batch_start + transfers_per_batch]
            num_rows = len(batch) * self.kernel_block_size
            staging_idx = self._spill_staging_index
            staging_event = self._spill_staging_events[staging_idx]
            if not staging_event.query():
                staging_event.synchronize()
            src_staging = self.spill_src_cpu[staging_idx]
            dst_staging = self.spill_dst_cpu[staging_idx]
            src = src_staging.numpy()
            dst = dst_staging.numpy()
            for transfer_idx, transfer in enumerate(batch):
                start = transfer_idx * self.kernel_block_size
                end = start + self.kernel_block_size
                for cache_idx, cache in enumerate(self.cache_handles):
                    block_id = transfer.source_block_ids[
                        cache.runtime.resident_source_index
                    ]
                    src[cache_idx, start:end] = (
                        block_id * self.kernel_block_size + offsets
                    )
                host_page = (
                    transfer.destination_block_id * self.blocks_per_kv_block
                    + transfer.destination_page_offset
                )
                dst[start:end] = host_page * self.kernel_block_size + offsets
            self.spill_src_gpu[:, :num_rows].copy_(
                src_staging[:, :num_rows], non_blocking=True
            )
            self.spill_dst_gpu[:num_rows].copy_(
                dst_staging[:num_rows], non_blocking=True
            )
            current_stream = torch.accelerator.current_stream(self.hot_backing.device)
            staging_event.record(current_stream)
            self._spill_staging_index = (staging_idx + 1) % len(
                self._spill_staging_events
            )
            torch.ops._C_cache_ops.hisparse_backup_layers(
                self.hot_backing,
                self.backup_layer_offsets,
                self.spill_src_indices_ptrs,
                self.backup_host_anchor,
                self.backup_host_cache_ptrs,
                self.spill_dst_gpu,
                num_rows,
                self.backup_src_block_stride,
                self.backup_src_block_size,
                self.backup_src_rows,
            )
            self.host_write_event.record(current_stream)
            transfer_ids = tuple(transfer.transfer_id for transfer in batch)
            completion_event = torch.Event()
            completion_event.record(current_stream)
            self._pending_transfer_events.append((completion_event, transfer_ids))
            self._enqueued_transfer_ids.extend(transfer_ids)

    def finish_forward(self) -> None:
        current_stream = torch.accelerator.current_stream(self.hot_backing.device)
        transfers = self._post_forward_transfers
        self._post_forward_transfers = []
        self._enqueue_transfers(transfers)
        self.host_write_event.record(current_stream)

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
        for layer in self.indexer_source_layers:
            layer.hisparse_indexer_source = None
        self.indexer_source_layers.clear()
        release_pinned_state(
            [cache.runtime for cache in self.cache_handles], self.pinned_host_pools
        )

    def restore_prefix(
        self,
        scheduler_output: SchedulerOutput,
        indexer_ready_req_ids: Collection[str],
    ) -> None:
        if self._restore_staging_event is not None:
            self._restore_staging_event.synchronize()
        src = self.src_cpu.numpy()
        dst = self.dst_cpu.numpy()
        num_pairs = 0

        def append_pairs(block_ids: tuple[list[int], ...], num_tokens: int) -> None:
            nonlocal num_pairs
            if num_tokens <= 0:
                return
            source_blocks = block_ids[self.source_group_id]
            indexer_blocks = block_ids[self.indexer_group_id]
            # The host manager returns logical scheduler blocks; the GPU
            # indexer manager returns already-split kernel-page blocks.
            count = min(
                cdiv(num_tokens, self.kernel_block_size),
                len(source_blocks) * self.blocks_per_kv_block,
                len(indexer_blocks),
            )
            end = num_pairs + count
            if end > src.size:
                raise RuntimeError(
                    "HiSparse prefix restore exceeded its preallocated block-ID "
                    f"capacity ({end} > {src.size})."
                )
            src[num_pairs:end] = _expand_source_block_ids(
                source_blocks, self.blocks_per_kv_block, count
            )
            dst[num_pairs:end] = indexer_blocks[:count]
            num_pairs = end

        for request in scheduler_output.scheduled_new_reqs:
            if request.req_id not in indexer_ready_req_ids:
                append_pairs(request.block_ids, request.num_computed_tokens)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, block_ids, num_tokens in zip(
            cached.req_ids, cached.new_block_ids, cached.num_computed_tokens
        ):
            if req_id in cached.resumed_req_ids and req_id not in indexer_ready_req_ids:
                assert block_ids is not None
                append_pairs(block_ids, num_tokens)

        if num_pairs == 0:
            return
        self.src_gpu[:num_pairs].copy_(self.src_cpu[:num_pairs], non_blocking=True)
        self.dst_gpu[:num_pairs].copy_(self.dst_cpu[:num_pairs], non_blocking=True)
        for source_cache, indexer_cache in self.cache_pairs:
            torch.ops._C_cache_ops.hisparse_copy_blocks(
                source_cache,
                indexer_cache,
                self.src_gpu[:num_pairs],
                self.dst_gpu[:num_pairs],
            )
        if self._restore_staging_event is None:
            self._restore_staging_event = torch.Event()
        self._restore_staging_event.record(
            torch.accelerator.current_stream(self.src_gpu.device)
        )


def init_hisparse_worker(
    *,
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    raw_tensors: dict[str, torch.Tensor],
    kv_caches: dict[str, torch.Tensor],
    block_tables: BlockTables,
    max_num_reqs: int,
    max_model_len: int,
    max_concurrent_batches: int,
    device: torch.device,
    pinned_host_pools: list[torch.Tensor],
) -> HiSparseWorker:
    tensor_configs = {
        name: tensor_config
        for tensor_config in kv_cache_config.kv_cache_tensors
        for name in tensor_config.layers
    }
    groups = kv_cache_config.kv_cache_groups
    source_group_id = get_unique_kv_cache_group_id(
        kv_cache_config, KVCacheGroupRole.HISPARSE_SOURCE
    )
    indexer_group_id = get_unique_kv_cache_group_id(
        kv_cache_config, KVCacheGroupRole.HISPARSE_INDEXER
    )
    source_group = groups[source_group_id]
    indexer_group = groups[indexer_group_id]
    source_specs = cast(UniformTypeKVCacheSpecs, source_group.kv_cache_spec)
    indexer_specs = cast(UniformTypeKVCacheSpecs, indexer_group.kv_cache_spec)
    num_blocks_by_pool = kv_cache_config.num_blocks_by_pool
    host_num_blocks = kv_cache_config.hisparse_host_num_blocks
    assert host_num_blocks is not None

    cache_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    indexer_source_layers: list[Any] = []
    for layer_name in indexer_group.layer_names:
        cache_name = f"{layer_name}{HISPARSE_INDEXER_SOURCE_SUFFIX}"
        tensor_config = tensor_configs[cache_name]
        source_spec = source_specs.kv_cache_specs[cache_name]
        gpu_indexer_spec = indexer_specs.kv_cache_specs[layer_name]
        assert isinstance(source_spec, AttentionSpec)
        kernel_block_size = gpu_indexer_spec.num_states
        source_block_size = source_spec.num_states
        assert source_block_size % kernel_block_size == 0
        blocks_per_kv_block = source_block_size // kernel_block_size
        kernel_page_stride = (
            gpu_indexer_spec.page_size_bytes // source_spec.dtype.itemsize
        )
        assert source_spec.page_size_bytes == (
            blocks_per_kv_block * gpu_indexer_spec.page_size_bytes
        )
        source_cache = torch.as_strided(
            raw_tensors[cache_name].view(source_spec.dtype),
            size=(
                host_num_blocks * blocks_per_kv_block,
                kernel_block_size,
                source_spec.head_size,
            ),
            stride=(
                kernel_page_stride,
                source_spec.head_size,
                1,
            ),
        )
        indexer_source_layer = forward_context[layer_name]
        indexer_source_layer.hisparse_indexer_source = (
            source_cache,
            block_tables.slot_mappings[source_group_id],
        )
        indexer_source_layers.append(indexer_source_layer)
        kv_caches[cache_name] = source_cache
        cache_pairs.append((source_cache, kv_caches[layer_name]))

    resident_source_index = 0
    for group_id, group in enumerate(groups):
        if not isinstance(group.kv_cache_spec, HiSparseResidentSpec):
            continue
        for cache_name in group.layer_names:
            assert cache_name.endswith(HISPARSE_RESIDENT_SUFFIX)
            layer_name = cache_name[: -len(HISPARSE_RESIDENT_SUFFIX)]
            tensor_config = tensor_configs[cache_name]
            assert tensor_config.block_pool_id is not None
            cache_handle = _get_hisparse_cache(forward_context, layer_name)
            cache_handle.bind_cache(
                raw_tensors[cache_name],
                byte_offset=tensor_config.offset,
                block_stride=tensor_config.block_stride,
                num_blocks=num_blocks_by_pool[tensor_config.block_pool_id],
                block_size=group.kv_cache_spec.block_size,
                block_table=block_tables.input_block_tables[group_id],
                slot_mapping=block_tables.slot_mappings[group_id],
            )
            assert cache_handle.view is not None
            kv_caches[cache_name] = cache_handle.view.cache
            cache_handle.runtime.resident_source_index = resident_source_index
        resident_source_index += 1

    hot_backing: torch.Tensor | None = None
    cache_handles: list[HiSparseCacheHandle] = []
    for group_id, group in enumerate(groups):
        if not isinstance(group.kv_cache_spec, HiSparseHotSpec):
            continue
        for cache_name in group.layer_names:
            assert cache_name.endswith(HISPARSE_HOT_SUFFIX)
            raw_tensor = raw_tensors[cache_name]
            if hot_backing is None:
                hot_backing = raw_tensor
            elif hot_backing.untyped_storage().data_ptr() != (
                raw_tensor.untyped_storage().data_ptr()
            ):
                raise RuntimeError("HiSparse hot tensors must share one GPU backing.")
            layer_name = cache_name[: -len(HISPARSE_HOT_SUFFIX)]
            cache_handle = _get_hisparse_cache(forward_context, layer_name)
            tensor_config = tensor_configs[cache_name]
            assert tensor_config.block_pool_id is not None
            cache_handle.runtime.bind_hot_cache(
                raw_tensor,
                byte_offset=tensor_config.offset,
                block_stride=tensor_config.block_stride,
                num_blocks=num_blocks_by_pool[tensor_config.block_pool_id],
                block_size=group.kv_cache_spec.block_size,
                block_table=block_tables.input_block_tables[group_id],
            )
            resident = cache_handle.view
            hot = cache_handle.runtime.hot
            assert resident is not None
            if (
                resident.cache.untyped_storage().data_ptr()
                != hot.cache.untyped_storage().data_ptr()
                or resident.cache.stride() != hot.cache.stride()
            ):
                raise RuntimeError("HiSparse resident and hot layouts must match.")
            cache_handle.runtime.bind_source_cache(
                kv_caches[layer_name], explicitly_registered=True
            )
            cache_handles.append(cache_handle)

    block_size = source_group.kv_cache_spec.block_size
    indexer_block_size = indexer_group.kv_cache_spec.block_size
    assert block_size % indexer_block_size == 0
    if not cache_handles or hot_backing is None:
        raise RuntimeError("HiSparse worker found no hot-cache handles.")
    return HiSparseWorker(
        cache_pairs,
        cache_handles,
        hot_backing,
        max_num_reqs,
        max_model_len,
        max_concurrent_batches,
        block_size // indexer_block_size,
        host_num_blocks,
        source_group_id,
        indexer_group_id,
        device,
        pinned_host_pools,
        indexer_source_layers,
    )
