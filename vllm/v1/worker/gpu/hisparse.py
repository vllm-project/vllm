# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.hisparse import (
    HiSparseCoordinator,
    bind_indexer_source_slot_mapping,
    get_indexer_source,
    invalidate_blocks,
    register_indexer_source,
    release_pinned_state,
    take_hisparse_stats,
)
from vllm.v1.core.kv_cache_utils import (
    HISPARSE_HOT_SUFFIX,
    HISPARSE_INDEXER_SOURCE_SUFFIX,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiSparseHotSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.metrics.stats import HiSparseStats

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


class HiSparseRuntime:
    def __init__(
        self,
        cache_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        coordinators: list[HiSparseCoordinator],
        hot_backing: torch.Tensor,
        backup_dst_slots: torch.Tensor,
        max_num_reqs: int,
        max_model_len: int,
        block_size: int,
        device: torch.device,
    ) -> None:
        self.cache_pairs = cache_pairs
        self.block_size = block_size
        kernel_block_size = cache_pairs[0][0].shape[1]
        assert block_size % kernel_block_size == 0
        self.kernel_block_size = kernel_block_size
        self.blocks_per_kv_block = block_size // kernel_block_size
        capacity = max_num_reqs * cdiv(max_model_len, kernel_block_size)
        self.src_cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.dst_cpu = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
        self.src_gpu = torch.empty(capacity, dtype=torch.int32, device=device)
        self.dst_gpu = torch.empty(capacity, dtype=torch.int32, device=device)
        self.coordinators = coordinators
        self.hot_backing = hot_backing
        self.backup_dst_slots = backup_dst_slots
        self._init_backup_plan(device)

    def _init_backup_plan(self, device: torch.device) -> None:
        entries = [
            coordinator.prepare_deferred_backup() for coordinator in self.coordinators
        ]
        hot_caches, source_indices, host_caches = zip(*entries)
        first_hot = hot_caches[0]
        first_host = host_caches[0]
        row_bytes = first_hot.shape[-1] * first_hot.element_size()
        src_block_size = first_hot.shape[1]
        src_block_stride = first_hot.stride(0) * first_hot.element_size()
        src_rows = first_hot.shape[0] * src_block_size
        host_rows = first_host.shape[0]
        backing_ptr = self.hot_backing.data_ptr()

        for hot_cache, host_cache in zip(hot_caches, host_caches):
            if hot_cache.untyped_storage().data_ptr() != (
                self.hot_backing.untyped_storage().data_ptr()
            ):
                raise RuntimeError("HiSparse hot caches must share one GPU backing.")
            if (
                hot_cache.shape[-1] * hot_cache.element_size() != row_bytes
                or hot_cache.shape[1] != src_block_size
                or hot_cache.stride(0) * hot_cache.element_size() != src_block_stride
                or hot_cache.shape[0] * src_block_size != src_rows
                or host_cache.shape[0] != host_rows
                or host_cache.shape[1] * host_cache.element_size() != row_bytes
                or not host_cache.is_contiguous()
            ):
                raise RuntimeError(
                    "HiSparse all-layer backup requires a uniform cache layout."
                )

        self.backup_layer_offsets = torch.tensor(
            [cache.data_ptr() - backing_ptr for cache in hot_caches],
            dtype=torch.int64,
            device=device,
        )
        self.backup_src_indices_ptrs = torch.tensor(
            [indices.data_ptr() for indices in source_indices],
            dtype=torch.uint64,
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
        self.host_write_event = torch.Event()
        for coordinator in self.coordinators:
            coordinator._host_write_event = self.host_write_event
        self.backup_num_items = 0

    def pre_step(self, scheduler_output: SchedulerOutput) -> None:
        scheduled_tokens = scheduler_output.num_scheduled_tokens
        self.backup_num_items = (
            len(scheduled_tokens)
            if scheduled_tokens
            and all(num_tokens == 1 for num_tokens in scheduled_tokens.values())
            else 0
        )
        block_ids = [
            block_id
            for request in scheduler_output.scheduled_new_reqs
            for block_id in request.block_ids[0]
        ]
        for new_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
            if new_block_ids is not None:
                block_ids.extend(new_block_ids[0])

        invalidate_blocks(block_ids, self.block_size)
        self.restore_prefix(scheduler_output)

    def post_forward(self) -> None:
        current_stream = torch.accelerator.current_stream(self.hot_backing.device)
        if self.backup_num_items > 0:
            torch.ops._C_cache_ops.hisparse_backup_layers(
                self.hot_backing,
                self.backup_layer_offsets,
                self.backup_src_indices_ptrs,
                self.backup_host_anchor,
                self.backup_host_cache_ptrs,
                self.backup_dst_slots,
                self.backup_num_items,
                self.backup_src_block_stride,
                self.backup_src_block_size,
                self.backup_src_rows,
            )
        self.host_write_event.record(current_stream)

    def post_step(self) -> HiSparseStats | None:
        return take_hisparse_stats()

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
    device: torch.device,
) -> HiSparseRuntime | None:
    tensor_configs = {
        name: tensor_config
        for tensor_config in kv_cache_config.kv_cache_tensors
        for name in tensor_config.shared_by
    }
    group_specs = {
        name: (group.kv_cache_spec, group_id)
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        for name in group.layer_names
    }
    hot_backing: torch.Tensor | None = None
    coordinators: list[HiSparseCoordinator] = []
    seen_coordinators: set[int] = set()
    for cache_name, raw_tensor in raw_tensors.items():
        if cache_name.endswith(HISPARSE_INDEXER_SOURCE_SUFFIX):
            layer_name = cache_name[: -len(HISPARSE_INDEXER_SOURCE_SUFFIX)]
            source_group_spec, _ = group_specs[cache_name]
            assert isinstance(source_group_spec, UniformTypeKVCacheSpecs)
            source_spec = source_group_spec.kv_cache_specs[cache_name]
            assert isinstance(source_spec, AttentionSpec)
            assert kv_cache_config.num_blocks_by_pool is not None
            indexer_spec, _ = group_specs[layer_name]
            assert isinstance(indexer_spec, UniformTypeKVCacheSpecs)
            kernel_block_size = indexer_spec.kv_cache_specs[layer_name].block_size
            assert source_spec.block_size % kernel_block_size == 0
            blocks_per_kv_block = source_spec.block_size // kernel_block_size
            tensor_config = tensor_configs[cache_name]
            source_cache = raw_tensor.view(source_spec.dtype).view(
                kv_cache_config.num_blocks_by_pool[tensor_config.block_pool_id]
                * blocks_per_kv_block,
                kernel_block_size,
                source_spec.head_size,
            )
            register_indexer_source(layer_name, source_cache)
            kv_caches[cache_name] = source_cache
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
        coordinator = forward_context[layer_name].impl.hisparse_coordinator
        tensor_config = tensor_configs[cache_name]
        hot_spec, hot_group_id = group_specs[cache_name]
        assert isinstance(hot_spec, HiSparseHotSpec)
        assert kv_cache_config.num_blocks_by_pool is not None
        coordinator.bind_hot_cache(
            raw_tensor,
            byte_offset=tensor_config.offset,
            block_stride=tensor_config.block_stride,
            num_blocks=kv_cache_config.num_blocks_by_pool[tensor_config.block_pool_id],
            block_size=hot_spec.block_size,
            hot_group_id=hot_group_id,
        )
        coordinator.bind_hot_block_table(block_tables.input_block_tables[hot_group_id])
        if id(coordinator) not in seen_coordinators:
            coordinator.bind_source_cache(kv_caches[layer_name])
            coordinators.append(coordinator)
            seen_coordinators.add(id(coordinator))

    source_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if any(
            name.endswith(HISPARSE_INDEXER_SOURCE_SUFFIX) for name in group.layer_names
        )
    ]
    indexer_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if group.layer_names
        and all(get_indexer_source(name) is not None for name in group.layer_names)
    ]
    if len(source_group_ids) != 1 or len(indexer_group_ids) != 1:
        raise RuntimeError(
            "HiSparse requires exactly one source group and one indexer group; "
            f"found source={source_group_ids}, indexer={indexer_group_ids}."
        )
    source_group_id = source_group_ids[0]
    indexer_group = kv_cache_config.kv_cache_groups[indexer_group_ids[0]]
    cache_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_name in indexer_group.layer_names:
        source = get_indexer_source(layer_name)
        assert source is not None
        cache_pairs.append((source[0], kv_caches[layer_name]))
        bind_indexer_source_slot_mapping(
            layer_name, block_tables.slot_mappings[source_group_id]
        )

    block_size = kv_cache_config.kv_cache_groups[
        source_group_id
    ].kv_cache_spec.block_size
    if not coordinators or hot_backing is None:
        raise RuntimeError("HiSparse runtime found no hot-cache coordinators.")
    return HiSparseRuntime(
        cache_pairs,
        coordinators,
        hot_backing,
        block_tables.slot_mappings[source_group_id],
        max_num_reqs,
        max_model_len,
        block_size,
        device,
    )
