# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.hisparse import (
    HiSparseCoordinator,
    invalidate_blocks,
    register_indexer_source,
    release_pinned_state,
)
from vllm.v1.core.kv_cache_utils import (
    HISPARSE_HOT_SUFFIX,
    HISPARSE_INDEXER_SOURCE_SUFFIX,
    HISPARSE_RESIDENT_SUFFIX,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    HiSparseSpill,
    KVCacheConfig,
    KVCacheGroupRole,
    UniformTypeKVCacheSpecs,
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


class HiSparseRuntime:
    def __init__(
        self,
        cache_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        coordinators: list[HiSparseCoordinator],
        hot_backing: torch.Tensor,
        max_num_reqs: int,
        max_model_len: int,
        max_concurrent_batches: int,
        block_size: int,
        blocks_per_kv_block: int,
        device: torch.device,
    ) -> None:
        self.cache_pairs = cache_pairs
        self.block_size = block_size
        kernel_block_size = cache_pairs[0][0].shape[1]
        assert block_size % kernel_block_size == 0
        self.kernel_block_size = kernel_block_size
        self.blocks_per_kv_block = blocks_per_kv_block
        capacity = max_num_reqs * cdiv(max_model_len, kernel_block_size)
        self.src_cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.dst_cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.src_gpu = torch.empty(capacity, dtype=torch.int32, device=device)
        self.dst_gpu = torch.empty(capacity, dtype=torch.int32, device=device)
        self.coordinators = coordinators
        self.hot_backing = hot_backing
        self._post_forward_spills: list[HiSparseSpill] = []
        self._enqueued_spill_ids: list[int] = []
        self._metrics_calls = 0
        self._metrics_last = HiSparseStats()
        self._init_backup_plan(device, max_model_len, max_concurrent_batches)

    def _init_backup_plan(
        self,
        device: torch.device,
        max_model_len: int,
        max_concurrent_batches: int,
    ) -> None:
        entries = [coordinator.backup_caches() for coordinator in self.coordinators]
        hot_caches, host_caches = zip(*entries)
        first_hot = hot_caches[0]
        first_host = host_caches[0]

        def host_layout(cache: torch.Tensor) -> tuple[int, int, int]:
            if cache.ndim == 2:
                row_bytes = cache.shape[1] * cache.element_size()
                return cache.shape[0], 1, row_bytes
            if cache.ndim == 3:
                return (
                    cache.shape[0] * cache.shape[1],
                    cache.shape[1],
                    cache.stride(0) * cache.element_size(),
                )
            raise RuntimeError("HiSparse host caches must be 2D or 3D.")

        row_bytes = first_hot.shape[-1] * first_hot.element_size()
        src_block_size = first_hot.shape[1]
        src_block_stride = first_hot.stride(0) * first_hot.element_size()
        src_rows = first_hot.shape[0] * src_block_size
        host_rows, host_block_size, host_block_stride = host_layout(first_host)
        row_value_bytes = self.coordinators[0].row_value_bytes or 0
        backing_ptr = self.hot_backing.data_ptr()

        for coordinator, hot_cache, host_cache in zip(
            self.coordinators, hot_caches, host_caches
        ):
            cache_host_rows, cache_host_block_size, cache_host_stride = host_layout(
                host_cache
            )
            if hot_cache.untyped_storage().data_ptr() != (
                self.hot_backing.untyped_storage().data_ptr()
            ):
                raise RuntimeError("HiSparse hot caches must share one GPU backing.")
            if (
                hot_cache.shape[-1] * hot_cache.element_size() != row_bytes
                or hot_cache.shape[1] != src_block_size
                or hot_cache.stride(0) * hot_cache.element_size() != src_block_stride
                or hot_cache.shape[0] * src_block_size != src_rows
                or cache_host_rows != host_rows
                or cache_host_block_size != host_block_size
                or cache_host_stride != host_block_stride
                or host_cache.shape[-1] * host_cache.element_size() != row_bytes
                or (coordinator.row_value_bytes or 0) != row_value_bytes
            ):
                raise RuntimeError(
                    "HiSparse all-layer backup requires a uniform cache layout."
                )

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
        self.backup_host_anchor = first_host
        self.backup_src_block_stride = src_block_stride
        self.backup_src_block_size = src_block_size
        self.backup_src_rows = src_rows
        self.backup_row_value_bytes = row_value_bytes
        self.host_write_event = torch.Event()
        for coordinator in self.coordinators:
            coordinator._host_write_event = self.host_write_event
        self.spill_row_capacity = max_model_len
        spill_staging_count = max_concurrent_batches + 1
        self.spill_src_cpu = torch.empty(
            (
                spill_staging_count,
                len(self.coordinators),
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
            (len(self.coordinators), self.spill_row_capacity),
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

    def pre_step(self, scheduler_output: SchedulerOutput) -> None:
        self.set_fully_resident_batch(scheduler_output.hisparse_fully_resident)
        spills = scheduler_output.hisparse_spills or []
        self._post_forward_spills = [spill for spill in spills if spill.after_forward]
        self._enqueue_spills([spill for spill in spills if not spill.after_forward])
        block_ids = [
            block_id
            for request in scheduler_output.scheduled_new_reqs
            for block_id in request.block_ids[0]
        ]
        for new_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
            if new_block_ids is not None:
                block_ids.extend(new_block_ids[0])

        invalidate_blocks(block_ids, self.kernel_block_size)
        self.restore_prefix(scheduler_output)

    def set_fully_resident_batch(self, fully_resident: bool) -> None:
        for coordinator in self.coordinators:
            coordinator.fully_resident_batch = fully_resident

    def set_request_state_indices(self, indices: torch.Tensor) -> None:
        self.coordinators[0].set_request_state_indices(indices, force=True)

    def reset_hot_state(self) -> None:
        for coordinator in self.coordinators:
            coordinator.reset_hot_state()

    def _enqueue_spills(self, spills: list[HiSparseSpill]) -> None:
        if not spills:
            return
        num_rows = len(spills) * self.kernel_block_size
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
        for spill_idx, spill in enumerate(spills):
            start = spill_idx * self.kernel_block_size
            end = start + self.kernel_block_size
            resident_blocks = dict(spill.resident_block_ids)
            for layer_idx, coordinator in enumerate(self.coordinators):
                block_id = resident_blocks[coordinator.resident_group_id]
                src[layer_idx, start:end] = block_id * self.kernel_block_size + offsets
            host_page = (
                spill.host_block_id * self.blocks_per_kv_block + spill.host_page_offset
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
        self._enqueued_spill_ids.extend(spill.spill_id for spill in spills)

    def post_forward(self) -> None:
        current_stream = torch.accelerator.current_stream(self.hot_backing.device)
        self._enqueue_spills(self._post_forward_spills)
        self._post_forward_spills = []
        self.host_write_event.record(current_stream)

    def post_step(self) -> HiSparseStats | None:
        self._metrics_calls += 1
        if self._metrics_calls % _METRICS_INTERVAL != 0:
            return None

        current = HiSparseStats()
        for coordinator in self.coordinators:
            hits, misses = coordinator._swap_stats.cpu().tolist()
            current.cache_hits += hits
            current.cache_misses += misses
            current.host_to_device_bytes += misses * coordinator.stats_row_bytes

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

    def take_spill_completions(self) -> list[int] | None:
        if not self._enqueued_spill_ids:
            return None
        spill_ids = self._enqueued_spill_ids
        self._enqueued_spill_ids = []
        return spill_ids

    def shutdown(self) -> None:
        release_pinned_state()

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


def init_hisparse_runtime(
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
) -> HiSparseRuntime | None:
    def get_coordinator(layer_name: str) -> HiSparseCoordinator:
        layer = forward_context[layer_name]
        coordinator = getattr(layer, "hisparse_coordinator", None)
        if coordinator is None:
            coordinator = layer.impl.hisparse_coordinator
        assert coordinator is not None
        return coordinator

    tensor_configs = {
        name: tensor_config
        for tensor_config in kv_cache_config.kv_cache_tensors
        for name in tensor_config.shared_by
    }
    group_ids = {
        name: group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        for name in group.layer_names
    }
    source_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if group.role is KVCacheGroupRole.HISPARSE_SOURCE
    ]
    indexer_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if group.role is KVCacheGroupRole.HISPARSE_INDEXER
    ]
    if len(source_group_ids) != 1 or len(indexer_group_ids) != 1:
        raise ValueError(
            "HiSparse requires exactly one source and one indexer cache group; "
            f"found source={source_group_ids}, indexer={indexer_group_ids}."
        )
    source_group_id = source_group_ids[0]
    indexer_group_id = indexer_group_ids[0]
    num_blocks_by_pool = kv_cache_config.num_blocks_by_pool
    hot_backing: torch.Tensor | None = None
    coordinators: list[HiSparseCoordinator] = []
    seen_coordinators: set[int] = set()
    for cache_name, raw_tensor in raw_tensors.items():
        group_id = group_ids[cache_name]
        group_spec = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        tensor_config = tensor_configs[cache_name]
        if cache_name.endswith(HISPARSE_INDEXER_SOURCE_SUFFIX):
            layer_name = cache_name[: -len(HISPARSE_INDEXER_SOURCE_SUFFIX)]
            source_group_spec = cast(UniformTypeKVCacheSpecs, group_spec)
            source_spec = source_group_spec.kv_cache_specs[cache_name]
            assert isinstance(source_spec, AttentionSpec)
            indexer_spec = cast(
                UniformTypeKVCacheSpecs,
                kv_cache_config.kv_cache_groups[indexer_group_id].kv_cache_spec,
            )
            gpu_indexer_spec = indexer_spec.kv_cache_specs[layer_name]
            kernel_block_size = getattr(
                gpu_indexer_spec, "storage_block_size", gpu_indexer_spec.block_size
            )
            source_storage_block_size = getattr(
                source_spec, "storage_block_size", source_spec.block_size
            )
            assert source_storage_block_size % kernel_block_size == 0
            blocks_per_kv_block = source_storage_block_size // kernel_block_size
            source_cache = torch.as_strided(
                raw_tensor.view(source_spec.dtype),
                size=(
                    num_blocks_by_pool[tensor_config.block_pool_id]
                    * blocks_per_kv_block,
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
            continue
        if cache_name.endswith(HISPARSE_RESIDENT_SUFFIX):
            layer_name = cache_name[: -len(HISPARSE_RESIDENT_SUFFIX)]
            resident_spec = group_spec
            assert isinstance(resident_spec, HiSparseResidentSpec)
            coordinator = get_coordinator(layer_name)
            coordinator.bind_resident_cache(
                raw_tensor,
                byte_offset=tensor_config.offset,
                block_stride=tensor_config.block_stride,
                num_blocks=num_blocks_by_pool[tensor_config.block_pool_id],
                block_size=resident_spec.block_size,
                block_table=block_tables.input_block_tables[group_id],
                slot_mapping=block_tables.slot_mappings[group_id],
                group_id=group_id,
            )
            continue
        if not cache_name.endswith(HISPARSE_HOT_SUFFIX):
            continue
        if hot_backing is None:
            hot_backing = raw_tensor
        elif hot_backing.untyped_storage().data_ptr() != (
            raw_tensor.untyped_storage().data_ptr()
        ):
            raise RuntimeError("HiSparse hot tensors must share one GPU backing.")
        layer_name = cache_name[: -len(HISPARSE_HOT_SUFFIX)]
        coordinator = get_coordinator(layer_name)
        hot_spec = group_spec
        assert isinstance(hot_spec, HiSparseHotSpec)
        coordinator.bind_hot_cache(
            raw_tensor,
            byte_offset=tensor_config.offset,
            block_stride=tensor_config.block_stride,
            num_blocks=num_blocks_by_pool[tensor_config.block_pool_id],
            block_size=hot_spec.block_size,
        )
        coordinator.bind_hot_block_table(block_tables.input_block_tables[group_id])
        if id(coordinator) not in seen_coordinators:
            coordinator.bind_source_cache(kv_caches[layer_name])
            coordinators.append(coordinator)
            seen_coordinators.add(id(coordinator))

    indexer_group = kv_cache_config.kv_cache_groups[indexer_group_id]
    cache_pairs = [
        (
            kv_caches[f"{layer_name}{HISPARSE_INDEXER_SOURCE_SUFFIX}"],
            kv_caches[layer_name],
        )
        for layer_name in indexer_group.layer_names
    ]

    block_size = kv_cache_config.kv_cache_groups[
        source_group_id
    ].kv_cache_spec.block_size
    indexer_block_size = kv_cache_config.kv_cache_groups[
        indexer_group_id
    ].kv_cache_spec.block_size
    assert block_size % indexer_block_size == 0
    if not coordinators or hot_backing is None:
        raise RuntimeError("HiSparse runtime found no hot-cache coordinators.")
    return HiSparseRuntime(
        cache_pairs,
        coordinators,
        hot_backing,
        max_num_reqs,
        max_model_len,
        max_concurrent_batches,
        block_size,
        block_size // indexer_block_size,
        device,
    )
