# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.v1.hisparse.layout import (
    HISPARSE_HOT_SUFFIX,
    HISPARSE_RESIDENT_SUFFIX,
)
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    allocate_pinned_host_pool,
    check_hisparse_host_memory,
    release_pinned_state,
)
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables


def _allocate_hisparse_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[int, torch.Tensor]]:
    host_sizes: dict[int, int] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        if tensor.block_pool_id != kv_cache_config.host_block_pool_id:
            continue
        previous_size = host_sizes.setdefault(tensor.block_pool_id, tensor.size)
        assert previous_size == tensor.size
    check_hisparse_host_memory(sum(host_sizes.values()))

    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    host_backings: dict[int, torch.Tensor] = {}
    device_backings: dict[int, torch.Tensor] = {}
    raw_tensors: dict[str, torch.Tensor] = {}
    kv_caches: dict[str, torch.Tensor] = {}
    pinned_host_pools: dict[int, torch.Tensor] = {}

    for tensor in kv_cache_config.kv_cache_tensors:
        if tensor.block_pool_id == kv_cache_config.host_block_pool_id:
            backing = host_backings.get(tensor.block_pool_id)
            if backing is None:
                backing, registered_pool = allocate_pinned_host_pool(tensor.size)
                host_backings[tensor.block_pool_id] = backing
                pinned_host_pools[backing.data_ptr()] = registered_pool
            else:
                assert backing.numel() == tensor.size
            num_blocks = kv_cache_config.block_pools[tensor.block_pool_id].num_blocks
        else:
            backing = device_backings.get(tensor.block_pool_id)
            if backing is None:
                backing = torch.zeros(tensor.size, dtype=torch.int8, device=device)
                device_backings[tensor.block_pool_id] = backing
            else:
                assert backing.numel() == tensor.size
            num_blocks = kv_cache_config.num_blocks

        for layer_name in tensor.layers:
            raw_tensors[layer_name] = backing

        first_layer = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if first_layer in group.layer_names
        )
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[first_layer]
        if isinstance(spec, (HiSparseHotSpec, HiSparseResidentSpec)):
            continue
        views = create_kv_cache_views(
            backing,
            spec,
            num_blocks,
            layout,
            tensor,
            kernel_block_size=kernel_block_sizes[group_id],
        )
        kv_caches.update(zip(tensor.layers, views))

    return kv_caches, raw_tensors, pinned_host_pools


def _get_hisparse_cache(
    forward_context: dict[str, Any], layer_name: str
) -> HiSparseCacheHandle:
    attention_layer = forward_context[layer_name]
    hisparse_cache = attention_layer.hisparse_cache
    assert hisparse_cache is not None
    return hisparse_cache


def release_hisparse_profiling_cache(forward_context: dict[str, Any]) -> None:
    cache_handles = [
        cache
        for layer in forward_context.values()
        if (cache := getattr(layer, "hisparse_cache", None)) is not None
    ]
    runtimes = {
        id(cache.runtime): cache.runtime
        for cache in cache_handles
        if hasattr(cache.runtime, "_host_cache")
    }
    if not runtimes:
        return

    registered_pools = list(
        {
            runtime.registered_host_pool.data_ptr(): runtime.registered_host_pool
            for runtime in runtimes.values()
        }.values()
    )
    release_pinned_state(list(runtimes.values()), registered_pools)
    for cache in cache_handles:
        cache.mirror_staging_cache = None
        cache.mirror_staging_slots = None


def _bind_hisparse_kv_caches(
    *,
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    raw_tensors: dict[str, torch.Tensor],
    kv_caches: dict[str, torch.Tensor],
    block_tables: "BlockTables",
    pinned_host_pools: dict[int, torch.Tensor],
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> None:
    tensor_configs = {
        name: tensor_config
        for tensor_config in kv_cache_config.kv_cache_tensors
        for name in tensor_config.layers
    }
    resident_source_index = 0
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if not isinstance(group.kv_cache_spec, HiSparseResidentSpec):
            continue
        for cache_name in group.layer_names:
            assert cache_name.endswith(HISPARSE_RESIDENT_SUFFIX)
            layer_name = cache_name[: -len(HISPARSE_RESIDENT_SUFFIX)]
            tensor_config = tensor_configs[cache_name]
            cache_handle = _get_hisparse_cache(forward_context, layer_name)
            cache_handle.bind_cache(
                raw_tensors[cache_name],
                byte_offset=tensor_config.offset,
                block_stride=tensor_config.block_stride,
                num_blocks=kv_cache_config.num_blocks,
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
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
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
            cache_handle.runtime.bind_hot_cache(
                raw_tensor,
                byte_offset=tensor_config.offset,
                block_stride=tensor_config.block_stride,
                num_blocks=kv_cache_config.num_blocks,
                block_size=group.kv_cache_spec.block_size,
                block_table=block_tables.input_block_tables[group_id],
            )
            resident = cache_handle.view
            hot = cache_handle.runtime.hot
            assert resident is not None
            if (
                resident.cache.untyped_storage().data_ptr()
                != hot.cache.untyped_storage().data_ptr()
                or resident.cache.data_ptr() != hot.cache.data_ptr()
                or resident.cache.stride() != hot.cache.stride()
            ):
                raise RuntimeError("HiSparse resident and hot layouts must match.")
            source_tensor = raw_tensors[layer_name]
            cache_handle.runtime.bind_source_cache(
                kv_caches[layer_name],
                registered_host_pool=pinned_host_pools[source_tensor.data_ptr()],
            )
            cache_handles.append(cache_handle)

    if hot_backing is None or not cache_handles:
        raise RuntimeError("HiSparse found no hot-cache handles.")
    request_state_indices = torch.full(
        (max_num_reqs,), -1, dtype=torch.int32, device=hot_backing.device
    )
    for cache_handle in cache_handles:
        cache_handle.runtime.request_state_indices = request_state_indices
    (source_group_id,) = kv_cache_config.host_group_ids
    source_block_table = block_tables.input_block_tables[source_group_id]
    source_slot_mapping = block_tables.slot_mappings[source_group_id]
    resident = cache_handles[0].view
    assert resident is not None
    staging_blocks = (
        max_num_batched_tokens + resident.block_size - 1
    ) // resident.block_size
    mirror_staging_caches = torch.empty(
        (
            len(cache_handles),
            staging_blocks,
            resident.block_size,
            resident.cache.shape[-1],
        ),
        dtype=resident.cache.dtype,
        device=hot_backing.device,
    )
    mirror_staging_slots = torch.arange(
        max_num_batched_tokens, dtype=torch.int64, device=hot_backing.device
    )
    for layer_index, cache_handle in enumerate(cache_handles):
        cache_handle.source_block_table = source_block_table
        cache_handle.mirror_slot_mapping = source_slot_mapping
        cache_handle.mirror_staging_cache = mirror_staging_caches[layer_index]
        cache_handle.mirror_staging_slots = mirror_staging_slots


def init_hisparse_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
    forward_context: dict[str, Any],
    block_tables: "BlockTables",
) -> dict[str, torch.Tensor]:
    kv_caches, raw_tensors, pinned_host_pools = _allocate_hisparse_kv_cache(
        kv_cache_config, device, kernel_block_sizes, vllm_config
    )
    _bind_hisparse_kv_caches(
        forward_context=forward_context,
        kv_cache_config=kv_cache_config,
        raw_tensors=raw_tensors,
        kv_caches=kv_caches,
        block_tables=block_tables,
        pinned_host_pools=pinned_host_pools,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
    )
    return kv_caches
