# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side wiring of HiSparse caches to attention layers."""

from collections.abc import Mapping
from copy import copy
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.v1.hisparse.layout import (
    HISPARSE_HOT_SUFFIX,
    HISPARSE_RESIDENT_SUFFIX,
)
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    HiSparseHostPool,
    initialize_hisparse_runtime_buffers,
    release_pinned_state,
)
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)
from vllm.v1.worker.utils import allocate_kv_cache, select_common_block_size

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.worker.gpu.block_table import BlockTables


def resolve_hisparse_block_size(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    attn_layers: Mapping[str, "AttentionLayerBase"],
) -> None:
    """Resolve a common kernel block size in-place for HiSparse MLA specs."""
    if vllm_config.attention_config.hisparse_config is None:
        return
    mla_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if isinstance(spec, MLAAttentionSpec)
    }
    if not mla_specs:
        return
    block_sizes = {spec.block_size for spec in mla_specs.values()}
    if len(block_sizes) != 1:
        raise ValueError("HiSparse requires one scheduler block size.")
    backends = [attn_layers[name].get_attn_backend() for name in mla_specs]
    try:
        block_size = select_common_block_size(block_sizes.pop(), backends)
    except ValueError as error:
        raise ValueError(
            "HiSparse requires a GPU block size supported by every sparse "
            f"attention and indexer backend: {error}"
        ) from error
    kv_cache_spec.update(
        (name, spec.copy_with_new_block_size(block_size))
        for name, spec in mla_specs.items()
    )


def allocate_hisparse_kv_caches(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    layout: KVCacheLayout,
    kernel_block_sizes: list[int],
    host_pool: HiSparseHostPool,
) -> dict[str, torch.Tensor]:
    """Allocate the host pool separately from the shared device backing."""
    device_config = copy(kv_cache_config)
    device_config.kv_cache_tensors = [
        tensor
        for tensor in kv_cache_config.kv_cache_tensors
        if not tensor.host_resident
    ]
    kv_caches = allocate_kv_cache(device_config, device, layout, kernel_block_sizes)
    host_tensors = [
        tensor for tensor in kv_cache_config.kv_cache_tensors if tensor.host_resident
    ]
    (host_size,) = {tensor.size for tensor in host_tensors}
    backing = host_pool.allocate(host_size)
    (host_group_id,) = kv_cache_config.host_group_ids
    host_spec = kv_cache_config.kv_cache_groups[host_group_id].kv_cache_spec
    for tensor in host_tensors:
        spec = (
            host_spec.kv_cache_specs[tensor.layers[0]]
            if isinstance(host_spec, UniformTypeKVCacheSpecs)
            else host_spec
        )
        kernel_block_size = kernel_block_sizes[host_group_id]
        if isinstance(spec, MLAAttentionSpec) and spec.storage_block_size is not None:
            kernel_block_size = spec.storage_block_size
        views = create_kv_cache_views(
            backing,
            spec,
            kv_cache_config.num_blocks_of(tensor),
            layout,
            tensor,
            kernel_block_size=kernel_block_size,
        )
        kv_caches.update(zip(tensor.layers, views))
    return kv_caches


def init_hisparse_kv_cache(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    kernel_block_sizes: list[int],
    vllm_config: VllmConfig,
    forward_context: dict[str, Any],
    block_tables: "BlockTables",
) -> dict[str, torch.Tensor]:
    """Allocate and bind HiSparse caches within the caller's allocation context."""
    host_pool = HiSparseHostPool()
    kv_caches = allocate_hisparse_kv_caches(
        kv_cache_config,
        device,
        vllm_config.cache_config.get_resolved_kv_cache_layout(),
        kernel_block_sizes,
        host_pool,
    )
    cache_handles = bind_hisparse_kv_caches(
        forward_context=forward_context,
        kv_cache_config=kv_cache_config,
        kv_caches=kv_caches,
        block_tables=block_tables,
        host_pool=host_pool,
    )
    initialize_hisparse_runtime_buffers(
        cache_handles,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
    )
    return kv_caches


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


def bind_hisparse_kv_caches(
    *,
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
    block_tables: "BlockTables",
    host_pool: HiSparseHostPool,
) -> list[HiSparseCacheHandle]:
    """Bind existing cache storage and block tables; return the bound handles."""
    assert host_pool.registered is not None
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
            assert not tensor_config.host_resident
            cache_handle = _get_hisparse_cache(forward_context, layer_name)
            cache_handle.bind_cache(
                kv_caches[cache_name],
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
            raw_tensor = kv_caches[cache_name]
            if hot_backing is None:
                hot_backing = raw_tensor
            elif hot_backing.untyped_storage().data_ptr() != (
                raw_tensor.untyped_storage().data_ptr()
            ):
                raise RuntimeError("HiSparse hot tensors must share one GPU backing.")
            layer_name = cache_name[: -len(HISPARSE_HOT_SUFFIX)]
            cache_handle = _get_hisparse_cache(forward_context, layer_name)
            tensor_config = tensor_configs[cache_name]
            assert not tensor_config.host_resident
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
            source_cache = kv_caches[layer_name]
            assert source_cache.untyped_storage().data_ptr() == (
                host_pool.registered.untyped_storage().data_ptr()
            )
            cache_handle.runtime.bind_source_cache(
                source_cache,
                registered_host_pool=host_pool.registered,
            )
            cache_handles.append(cache_handle)
            # The hot slab is raw storage owned by the runtime, not a layer
            # cache: nothing downstream binds or registers it.
            del kv_caches[cache_name]

    if hot_backing is None or not cache_handles:
        raise RuntimeError("HiSparse found no hot-cache handles.")
    (source_group_id,) = kv_cache_config.host_group_ids
    for cache_handle in cache_handles:
        cache_handle.source_block_table = block_tables.input_block_tables[
            source_group_id
        ]
        cache_handle.mirror_slot_mapping = block_tables.slot_mappings[source_group_id]
    return cache_handles
