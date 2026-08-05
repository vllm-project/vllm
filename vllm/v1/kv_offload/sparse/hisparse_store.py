# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiSparse worker-side store for host/hot state and data movement."""

from __future__ import annotations

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
from vllm.v1.kv_offload.sparse.hisparse_layer import (
    HiSparseLayer,
    register_host_write_event,
    register_indexer_source,
    release_pinned_state,
)
from vllm.v1.metrics.stats import HiSparseStats

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.block_table import BlockTables


_METRICS_INTERVAL = 2000


def _expand_source_block_ids(
    source_blocks: list[int], blocks_per_kv_block: int, count: int
) -> np.ndarray:
    """Expand logical host block IDs into kernel-page IDs."""
    num_source_blocks = cdiv(count, blocks_per_kv_block)
    logical_ids = np.asarray(source_blocks[:num_source_blocks], dtype=np.int32)
    offsets = np.arange(blocks_per_kv_block, dtype=np.int32)
    return (logical_ids[:, None] * blocks_per_kv_block + offsets).reshape(-1)[:count]


def _get_hisparse_layer(
    forward_context: dict[str, Any], layer_name: str
) -> HiSparseLayer:
    layer = forward_context[layer_name]
    hisparse_layer = getattr(layer, "hisparse_layer", None)
    if hisparse_layer is None:
        hisparse_layer = layer.impl.hisparse_layer
    assert hisparse_layer is not None
    return hisparse_layer


class HiSparseOffloadStore:
    """Own HiSparse host/hot state and execute its worker-side transfers."""

    def __init__(
        self,
        cache_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        layers: list[HiSparseLayer],
        hot_backing: torch.Tensor,
        max_num_reqs: int,
        max_model_len: int,
        max_concurrent_batches: int,
        blocks_per_kv_block: int,
        device: torch.device,
    ) -> None:
        self.cache_pairs = cache_pairs
        kernel_block_size = cache_pairs[0][0].shape[1]
        self.kernel_block_size = kernel_block_size
        self.blocks_per_kv_block = blocks_per_kv_block
        capacity = max_num_reqs * cdiv(max_model_len, kernel_block_size)
        self.src_cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.dst_cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.src_gpu = torch.empty(capacity, dtype=torch.int32, device=device)
        self.dst_gpu = torch.empty(capacity, dtype=torch.int32, device=device)
        self.layers = layers
        self.lru_layers = [layer for layer in layers if layer.offload.leader is None]
        self.hot_backing = hot_backing
        self._block_staging: torch.Tensor | None = None
        self._block_staging_event: torch.Event | None = None
        self._post_forward_transfers: list[SparseKVPageTransfer] = []
        self._completed_transfer_ids: list[int] = []
        self._metrics_calls = 0
        self._metrics_last = HiSparseStats()
        self._init_backup_plan(device, max_model_len, max_concurrent_batches)

    def _init_backup_plan(
        self,
        device: torch.device,
        max_model_len: int,
        max_concurrent_batches: int,
    ) -> None:
        entries = [layer.offload.backup_caches() for layer in self.layers]
        hot_caches, host_caches = zip(*entries)

        def host_layout(cache: torch.Tensor) -> tuple[int, int, int]:
            if cache.ndim not in (2, 3):
                raise RuntimeError("HiSparse host caches must be 2D or 3D.")
            block_size = cache.shape[1] if cache.ndim == 3 else 1
            rows = cache.numel() // cache.shape[-1]
            return rows, block_size, cache.stride(0) * cache.element_size()

        layouts = []
        for item, hot_cache, host_cache in zip(self.layers, hot_caches, host_caches):
            row_bytes = hot_cache.shape[-1] * hot_cache.element_size()
            layouts.append(
                (
                    row_bytes,
                    hot_cache.shape[1],
                    hot_cache.stride(0) * hot_cache.element_size(),
                    hot_cache.shape[0] * hot_cache.shape[1],
                    *host_layout(host_cache),
                    item.offload.row_value_bytes or 0,
                )
            )
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise RuntimeError(
                "HiSparse all-layer backup requires a uniform cache layout."
            )
        src_block_size, src_block_stride, src_rows = layouts[0][1:4]
        row_value_bytes = layouts[0][-1]
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
        self.backup_row_value_bytes = row_value_bytes
        self.host_write_event = torch.Event()
        register_host_write_event(device, self.host_write_event)
        self.spill_row_capacity = max_model_len
        spill_staging_count = max_concurrent_batches + 1
        self.spill_src_cpu = torch.empty(
            (
                spill_staging_count,
                len(self.layers),
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
            (len(self.layers), self.spill_row_capacity),
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

    def prepare_step(
        self,
        command: SparseKVOffloadCommand,
        scheduler_output: SchedulerOutput,
    ) -> None:
        self.set_fully_resident_batch(command.fully_resident)
        transfers = command.page_transfers
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
            for block_id in request.block_ids[0]
        ]
        for new_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
            if new_block_ids is not None:
                block_ids.extend(new_block_ids[0])

        self.invalidate_blocks(block_ids)
        self.restore_prefix(scheduler_output)

    def invalidate_blocks(self, block_ids: list[int]) -> None:
        """Invalidate recycled host slots in only the layers owned by this store."""
        if not block_ids:
            return
        device = self.layers[0].offload.device
        num_blocks = len(block_ids)
        if self._block_staging is None or self._block_staging.shape[0] < num_blocks:
            size = 1 << max(10, (num_blocks - 1).bit_length())
            self._block_staging = torch.empty(size, dtype=torch.long, pin_memory=True)
            self._block_staging_event = None
        if self._block_staging_event is not None:
            self._block_staging_event.synchronize()
        staging = self._block_staging[:num_blocks]
        staging.copy_(torch.from_numpy(np.asarray(block_ids, dtype=np.int64)))
        blocks = staging.to(device, non_blocking=True)
        if self._block_staging_event is None:
            self._block_staging_event = torch.Event()
        self._block_staging_event.record(torch.accelerator.current_stream(device))
        offsets = torch.arange(self.kernel_block_size, dtype=torch.long, device=device)
        slots = (blocks[:, None] * self.kernel_block_size + offsets[None, :]).flatten()
        for layer in self.lru_layers:
            layer.offload.invalidate_slots(slots)

    def set_fully_resident_batch(self, fully_resident: bool) -> None:
        for layer in self.layers:
            layer.fully_resident = fully_resident

    @property
    def fully_resident_batch(self) -> bool:
        return self.layers[0].fully_resident

    def reset_hot_state(self) -> None:
        for layer in self.lru_layers:
            layer.offload.reset_hot_state()

    def _enqueue_transfers(self, transfers: list[SparseKVPageTransfer]) -> None:
        if not transfers:
            return
        num_rows = len(transfers) * self.kernel_block_size
        if num_rows > self.spill_row_capacity:
            raise RuntimeError(
                "HiSparse spill exceeded its preallocated row capacity "
                f"({num_rows} > {self.spill_row_capacity})."
            )
        staging_idx = self._spill_staging_index
        staging_event = self._spill_staging_events[staging_idx]
        if not staging_event.query():
            raise RuntimeError(
                "HiSparse exceeded its preallocated in-flight spill staging."
            )
        src_staging = self.spill_src_cpu[staging_idx]
        dst_staging = self.spill_dst_cpu[staging_idx]
        src = src_staging.numpy()
        dst = dst_staging.numpy()
        offsets = np.arange(self.kernel_block_size, dtype=np.int64)
        for transfer_idx, transfer in enumerate(transfers):
            start = transfer_idx * self.kernel_block_size
            end = start + self.kernel_block_size
            for layer_idx, layer in enumerate(self.layers):
                block_id = transfer.source_block_ids[
                    layer.offload.resident_source_index
                ]
                src[layer_idx, start:end] = block_id * self.kernel_block_size + offsets
            host_page = (
                transfer.destination_block_id * self.blocks_per_kv_block
                + transfer.destination_page_offset
            )
            dst[start:end] = host_page * self.kernel_block_size + offsets
        self.spill_src_gpu[:, :num_rows].copy_(
            src_staging[:, :num_rows], non_blocking=True
        )
        self.spill_dst_gpu[:num_rows].copy_(dst_staging[:num_rows], non_blocking=True)
        staging_event.record(torch.accelerator.current_stream(self.hot_backing.device))
        self._spill_staging_index = (staging_idx + 1) % len(self._spill_staging_events)
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
            self.backup_row_value_bytes,
        )
        self.host_write_event.record(
            torch.accelerator.current_stream(self.hot_backing.device)
        )
        self._completed_transfer_ids.extend(
            transfer.transfer_id for transfer in transfers
        )

    def finish_forward(self) -> None:
        current_stream = torch.accelerator.current_stream(self.hot_backing.device)
        transfers = self._post_forward_transfers
        self._post_forward_transfers = []
        self._enqueue_transfers(transfers)
        self.host_write_event.record(current_stream)

    def finish_step(self) -> HiSparseStats | None:
        self._metrics_calls += 1
        if self._metrics_calls % _METRICS_INTERVAL != 0:
            return None

        current = HiSparseStats()
        for layer in self.lru_layers:
            hits, misses = layer.offload._swap_stats.cpu().tolist()
            current.cache_hits += hits
            current.cache_misses += misses
            current.host_to_device_bytes += misses * layer.offload.stats_row_bytes

        delta = HiSparseStats(
            cache_hits=current.cache_hits - self._metrics_last.cache_hits,
            cache_misses=current.cache_misses - self._metrics_last.cache_misses,
            host_to_device_bytes=(
                current.host_to_device_bytes - self._metrics_last.host_to_device_bytes
            ),
        )
        self._metrics_last = current
        if delta.cache_hits == 0 and delta.cache_misses == 0:
            return None
        return delta

    def take_completed_transfer_ids(self) -> list[int] | None:
        completed = self._completed_transfer_ids
        self._completed_transfer_ids = []
        return completed or None

    def shutdown(self) -> None:
        release_pinned_state([layer.offload for layer in self.layers])

    def restore_prefix(self, scheduler_output: SchedulerOutput) -> None:
        src = self.src_cpu.numpy()
        dst = self.dst_cpu.numpy()
        num_pairs = 0

        def append_pairs(block_ids: tuple[list[int], ...], num_tokens: int) -> None:
            nonlocal num_pairs
            if num_tokens <= 0:
                return
            source_blocks = block_ids[0]
            indexer_blocks = block_ids[1]
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
            append_pairs(request.block_ids, request.num_computed_tokens)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, block_ids, num_tokens in zip(
            cached.req_ids, cached.new_block_ids, cached.num_computed_tokens
        ):
            if req_id in cached.resumed_req_ids:
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


def init_hisparse_store(
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
) -> HiSparseOffloadStore:
    tensor_configs = {
        name: tensor_config
        for tensor_config in kv_cache_config.kv_cache_tensors
        for name in tensor_config.shared_by
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

    cache_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_name in indexer_group.layer_names:
        cache_name = f"{layer_name}{HISPARSE_INDEXER_SOURCE_SUFFIX}"
        tensor_config = tensor_configs[cache_name]
        source_spec = source_specs.kv_cache_specs[cache_name]
        gpu_indexer_spec = indexer_specs.kv_cache_specs[layer_name]
        assert isinstance(source_spec, AttentionSpec)
        kernel_block_size = getattr(
            gpu_indexer_spec, "storage_block_size", gpu_indexer_spec.block_size
        )
        source_block_size = getattr(
            source_spec, "storage_block_size", source_spec.block_size
        )
        assert source_block_size % kernel_block_size == 0
        blocks_per_kv_block = source_block_size // kernel_block_size
        source_cache = torch.as_strided(
            raw_tensors[cache_name].view(source_spec.dtype),
            size=(
                num_blocks_by_pool[tensor_config.block_pool_id] * blocks_per_kv_block,
                kernel_block_size,
                source_spec.head_size,
            ),
            stride=(
                source_spec.page_size_bytes // source_spec.dtype.itemsize,
                source_spec.head_size,
                1,
            ),
        )
        register_indexer_source(
            layer_name,
            source_cache,
            block_tables.slot_mappings[source_group_id],
        )
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
            layer = _get_hisparse_layer(forward_context, layer_name)
            layer.bind_cache(
                raw_tensors[cache_name],
                byte_offset=tensor_config.offset,
                block_stride=tensor_config.block_stride,
                num_blocks=num_blocks_by_pool[tensor_config.block_pool_id],
                block_size=group.kv_cache_spec.block_size,
                block_table=block_tables.input_block_tables[group_id],
                slot_mapping=block_tables.slot_mappings[group_id],
            )
            layer.offload.resident_source_index = resident_source_index
        resident_source_index += 1

    hot_backing: torch.Tensor | None = None
    layers: list[HiSparseLayer] = []
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
            layer = _get_hisparse_layer(forward_context, layer_name)
            tensor_config = tensor_configs[cache_name]
            layer.offload.bind_hot_cache(
                raw_tensor,
                byte_offset=tensor_config.offset,
                block_stride=tensor_config.block_stride,
                num_blocks=num_blocks_by_pool[tensor_config.block_pool_id],
                block_size=group.kv_cache_spec.block_size,
                block_table=block_tables.input_block_tables[group_id],
            )
            resident = layer.view
            hot = layer.offload.hot
            assert resident is not None and hot is not None
            if (
                resident.cache.untyped_storage().data_ptr()
                != hot.cache.untyped_storage().data_ptr()
                or resident.cache.stride() != hot.cache.stride()
            ):
                raise RuntimeError("HiSparse resident and hot layouts must match.")
            layer.offload.bind_source_cache(kv_caches[layer_name])
            layers.append(layer)

    block_size = source_group.kv_cache_spec.block_size
    indexer_block_size = indexer_group.kv_cache_spec.block_size
    assert block_size % indexer_block_size == 0
    if not layers or hot_backing is None:
        raise RuntimeError("HiSparse runtime found no hot-cache layers.")
    return HiSparseOffloadStore(
        cache_pairs,
        layers,
        hot_backing,
        max_num_reqs,
        max_model_len,
        max_concurrent_batches,
        block_size // indexer_block_size,
        device,
    )
