# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side wiring of HiSparse caches to attention layers."""

from collections import defaultdict
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import replace
from math import gcd, lcm
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.abstract import (
    get_mamba_prefill_checkpoint_alignment,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    HiSparseHotManager,
    HiSparseResidentManager,
)
from vllm.v1.hisparse.backend import get_hisparse_backend
from vllm.v1.hisparse.layout import (
    HISPARSE_HOT_SUFFIX,
    HISPARSE_RESIDENT_SUFFIX,
    get_hisparse_page_layer_configs,
    is_hisparse_source_layer,
)
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    HiSparseHostPool,
    drop_hisparse_host_pool,
    get_hisparse_host_pool,
    initialize_hisparse_runtime_buffers,
    release_pinned_state,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheLayerConfig,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    SparseFullAttentionSpec,
    create_kv_cache_views,
)
from vllm.v1.worker.utils import allocate_kv_cache, select_common_block_size

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.worker.gpu.block_table import BlockTables


def resolve_hisparse_specs(
    vllm_config: VllmConfig,
    layer_configs: dict[str, KVCacheLayerConfig],
    attn_layers: Mapping[str, "AttentionLayerBase"],
) -> dict[str, KVCacheLayerConfig]:
    """Resolve sparse source and indexer block sizes, derive resident and hot
    caches, and fit align-mode Mamba blocks to the resulting attention blocks."""
    sparse_layers = {
        name: lc for name, lc in layer_configs.items() if is_hisparse_source_layer(lc)
    }
    if not sparse_layers:
        return layer_configs
    block_sizes = {lc.spec.block_size for lc in sparse_layers.values()}
    if len(block_sizes) != 1:
        raise ValueError("HiSparse requires one scheduler block size.")
    backends = [attn_layers[name].get_attn_backend() for name in sparse_layers]
    specs = [lc.spec for lc in sparse_layers.values()]
    try:
        block_size = select_common_block_size(block_sizes.pop(), backends, specs)
    except ValueError as error:
        raise ValueError(
            "HiSparse requires a GPU block size supported by every sparse "
            f"attention and indexer backend: {error}"
        ) from error
    indexer_specs = {
        name: lc.spec
        for name, lc in sparse_layers.items()
        if isinstance(lc.spec, MLAAttentionSpec)
        and lc.spec.cache_role is SparseCacheRole.INDEXER
    }
    indexer_block_size = _largest_exact_kernel_block_size(
        block_size,
        [attn_layers[name].get_attn_backend() for name in indexer_specs],
        list(indexer_specs.values()),
    )
    sparse_layers = {
        name: replace(
            lc,
            spec=lc.spec.copy_with_new_block_size(
                indexer_block_size if name in indexer_specs else block_size
            ),
        )
        for name, lc in sparse_layers.items()
    }
    page_layer_configs = get_hisparse_page_layer_configs(vllm_config, sparse_layers)
    return fit_mamba_blocks_to_attention(
        vllm_config, layer_configs | sparse_layers | page_layer_configs, attn_layers
    )


def _largest_exact_kernel_block_size(
    block_size: int,
    backends: list[type["AttentionBackend"]],
    specs: list[MLAAttentionSpec],
) -> int:
    """Largest exact kernel page all backends accept, as a multiple of block_size."""
    sizes: set[int] | None = None
    for backend, spec in zip(backends, specs, strict=True):
        exact = {
            size
            for size in backend.get_supported_kernel_block_sizes(spec)
            if isinstance(size, int)
        }
        sizes = exact if sizes is None else sizes & exact
    return max(
        (size for size in sizes or () if size % block_size == 0), default=block_size
    )


def fit_mamba_blocks_to_attention(
    vllm_config: VllmConfig,
    layer_configs: dict[str, KVCacheLayerConfig],
    layers: Mapping[str, "AttentionLayerBase"],
) -> dict[str, KVCacheLayerConfig]:
    """Shrink align-mode Mamba blocks that would grow the scheduler block.

    Mamba blocks start at `cache_config.block_size`, which resized attention
    specs may no longer use. Rebuilds the Mamba specs at the largest size up to
    it that divides the attention blocks' lcm.
    """
    if not any(isinstance(lc.spec, MambaSpec) for lc in layer_configs.values()):
        return layer_configs
    cache_config = vllm_config.cache_config
    if cache_config.mamba_cache_mode != "align":
        return layer_configs
    attn_block_sizes = [
        lc.spec.block_size
        for lc in layer_configs.values()
        if not isinstance(lc.spec, MambaSpec)
        and lc.manager_cls.prefix_cacheable(lc.manager_config)
    ]
    if not attn_block_sizes:
        return layer_configs
    scheduler_block_size = lcm(*attn_block_sizes)
    step = lcm(
        gcd(*attn_block_sizes),
        get_mamba_prefill_checkpoint_alignment(vllm_config) or 1,
    )
    block_size = max(
        (
            size
            for size in range(
                step, min(cache_config.block_size, scheduler_block_size) + 1, step
            )
            if scheduler_block_size % size == 0
        ),
        default=cache_config.mamba_block_size,
    )
    if block_size == cache_config.mamba_block_size:
        return layer_configs
    cache_config.mamba_block_size = block_size
    resized = dict(layer_configs)
    for name, lc in layer_configs.items():
        if isinstance(lc.spec, MambaSpec):
            new_config = layers[name].get_kv_cache_config(vllm_config)
            assert new_config is not None and isinstance(new_config.spec, MambaSpec)
            resized[name] = replace(new_config, is_draft=lc.is_draft)
    return resized


def allocate_hisparse_kv_caches(
    kv_cache_config: KVCacheConfig,
    device: torch.device,
    layout: KVCacheLayout,
    kernel_block_sizes: list[int],
    host_pool: HiSparseHostPool,
    allocate: Callable[[int], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Allocate the host pool separately from the shared device backing."""
    device_config = copy(kv_cache_config)
    device_config.kv_cache_tensors = [
        tensor
        for tensor in kv_cache_config.kv_cache_tensors
        if not tensor.host_resident
    ]
    kv_caches = allocate_kv_cache(
        device_config, device, layout, kernel_block_sizes, allocate=allocate
    )
    host_tensors = [
        tensor for tensor in kv_cache_config.kv_cache_tensors if tensor.host_resident
    ]
    (host_size,) = {tensor.size for tensor in host_tensors}
    backing = host_pool.allocate(host_size)
    (host_group_id,) = kv_cache_config.host_group_ids
    host_specs = kv_cache_config.kv_cache_groups[host_group_id].kv_cache_specs
    for tensor in host_tensors:
        spec = host_specs[tensor.layers[0]]
        views = create_kv_cache_views(
            backing,
            spec,
            kv_cache_config.num_blocks_of(tensor),
            layout,
            tensor,
            kernel_block_size=kernel_block_sizes[host_group_id],
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
    allocate: Callable[[int], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Allocate and bind HiSparse caches within the caller's allocation context."""
    backend = get_hisparse_backend()
    backend.validate_kv_cache_config(kv_cache_config)
    host_pool = get_hisparse_host_pool(vllm_config, kv_cache_config)
    try:
        kv_caches = allocate_hisparse_kv_caches(
            kv_cache_config,
            device,
            vllm_config.cache_config.get_resolved_kv_cache_layout(),
            kernel_block_sizes,
            host_pool,
            allocate=allocate,
        )
        backend.prepare_kv_cache_views(kv_cache_config, kv_caches, forward_context)
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
        )
        return kv_caches
    except Exception:
        release_pinned_state(host_pool.pinned, host_pool.shared_region)
        unbind_hisparse_caches(forward_context)
        drop_hisparse_host_pool(vllm_config)
        raise


def _get_hisparse_cache(
    forward_context: dict[str, Any], layer_name: str
) -> HiSparseCacheHandle:
    attention_layer = forward_context[layer_name]
    hisparse_cache = attention_layer.hisparse_cache
    assert hisparse_cache is not None
    return hisparse_cache


def get_kv_sharing_readers(
    forward_context: dict[str, Any],
) -> defaultdict[str, list[HiSparseCacheHandle]]:
    """Cross-layer KV-sharing layers by target: they own no cache, so HiSparse
    binds them to their target's resident, hot and host buffers."""
    readers: defaultdict[str, list[HiSparseCacheHandle]] = defaultdict(list)
    for layer in forward_context.values():
        target = getattr(layer, "kv_sharing_target_layer_name", None)
        cache = getattr(layer, "hisparse_cache", None)
        if target is not None and cache is not None:
            readers[target].append(cache)
    return readers


def unbind_hisparse_caches(forward_context: dict[str, Any]) -> None:
    """Detach HiSparse cache bindings without unregistering the host pool."""
    for layer in forward_context.values():
        if (cache := getattr(layer, "hisparse_cache", None)) is not None:
            cache.unbind_cache()


def _tensor_placements(
    kv_cache_config: KVCacheConfig,
) -> dict[str, tuple[KVCacheTensor, int]]:
    """Each cache's tensor placement and byte offset."""
    return {
        name: (
            tensor_config,
            tensor_config.offset + layer_index * tensor_config.layer_stride,
        )
        for tensor_config in kv_cache_config.kv_cache_tensors
        for layer_index, name in enumerate(tensor_config.layers)
    }


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
    (source_group_id,) = kv_cache_config.host_group_ids
    source_specs = kv_cache_config.kv_cache_groups[source_group_id].kv_cache_specs
    kv_sharing_readers = get_kv_sharing_readers(forward_context)
    tensor_configs = _tensor_placements(kv_cache_config)
    resident_groups = [
        (group_id, group)
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if issubclass(group.manager_cls, HiSparseResidentManager)
    ]
    residency: torch.Tensor | None = None
    if resident_groups:
        table_shapes = {
            block_tables.input_block_tables[group_id].shape
            for group_id, _ in resident_groups
        }
        assert len(table_shapes) == 1
        max_num_reqs, max_num_pages = table_shapes.pop()
        residency = torch.zeros(
            (len(resident_groups), max_num_reqs, max_num_pages),
            dtype=torch.int32,
            device=block_tables.input_block_tables[resident_groups[0][0]].device,
        )
    for resident_source_index, (group_id, group) in enumerate(resident_groups):
        assert residency is not None
        for cache_name in group.layer_names:
            assert cache_name.endswith(HISPARSE_RESIDENT_SUFFIX)
            layer_name = cache_name[: -len(HISPARSE_RESIDENT_SUFFIX)]
            tensor_config, byte_offset = tensor_configs[cache_name]
            assert not tensor_config.host_resident
            cache_handle = _get_hisparse_cache(forward_context, layer_name)
            for handle in (cache_handle, *kv_sharing_readers[layer_name]):
                handle.bind_cache(
                    kv_caches[cache_name],
                    byte_offset=byte_offset,
                    block_stride=tensor_config.block_stride,
                    num_blocks=kv_cache_config.num_blocks,
                    block_size=group.block_size,
                    block_table=residency[resident_source_index],
                    slot_mapping=block_tables.slot_mappings[group_id],
                )
                handle.residency = residency
                handle.runtime.resident_source_index = resident_source_index
            assert cache_handle.view is not None
            kv_caches[cache_name] = cache_handle.view.cache

    hot_backing: torch.Tensor | None = None
    cache_handles: list[HiSparseCacheHandle] = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if not issubclass(group.manager_cls, HiSparseHotManager):
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
            tensor_config, byte_offset = tensor_configs[cache_name]
            assert not tensor_config.host_resident
            readers = kv_sharing_readers[layer_name]
            for handle in (cache_handle, *readers):
                handle.runtime.bind_hot_cache(
                    raw_tensor,
                    byte_offset=byte_offset,
                    block_stride=tensor_config.block_stride,
                    num_blocks=kv_cache_config.num_blocks,
                    block_size=group.block_size,
                    block_table=block_tables.input_block_tables[group_id],
                    owner=None if handle is cache_handle else cache_handle.runtime,
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
            if isinstance(source_specs[layer_name], SparseFullAttentionSpec):
                token_rows = source_cache.transpose(1, 2)
                try:
                    source_cache = token_rows.view(*token_rows.shape[:2], -1)
                except RuntimeError as error:
                    raise ValueError(
                        "HiSparse sparse full attention requires unpadded, "
                        "contiguous token rows in the host LBNHC layout."
                    ) from error
            for handle in (cache_handle, *readers):
                handle.draft_layer = forward_context[layer_name].is_draft_layer
                handle.runtime.host_pool = host_pool
                handle.runtime.bind_source_cache(
                    source_cache,
                    registered_host_pool=host_pool.registered,
                )
            cache_handles.extend((cache_handle, *readers))
            # The hot slab is raw storage owned by the runtime, not a layer
            # cache: nothing downstream binds or registers it.
            del kv_caches[cache_name]

    if hot_backing is None or not cache_handles:
        raise RuntimeError("HiSparse found no hot-cache handles.")
    for cache_handle in cache_handles:
        cache_handle.source_block_table = block_tables.input_block_tables[
            source_group_id
        ]
        cache_handle.mirror_slot_mapping = block_tables.slot_mappings[source_group_id]
    return cache_handles


def rebind_hisparse_device_caches(
    *,
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
) -> None:
    """Rebind the resident and hot views over resized raw device storage.

    kv_caches holds each resident and hot cache's raw storage, as allocation
    returns it; as in `bind_hisparse_kv_caches`, resident entries become their
    views and hot entries are dropped.
    """
    tensor_configs = _tensor_placements(kv_cache_config)
    kv_sharing_readers = get_kv_sharing_readers(forward_context)
    for group in kv_cache_config.kv_cache_groups:
        for cache_name in group.layer_names:
            if issubclass(group.manager_cls, HiSparseResidentManager):
                layer_name = cache_name[: -len(HISPARSE_RESIDENT_SUFFIX)]
                cache_handle = _get_hisparse_cache(forward_context, layer_name)
                tensor_config, byte_offset = tensor_configs[cache_name]
                assert cache_handle.block_table is not None
                assert cache_handle.slot_mapping is not None
                for handle in (cache_handle, *kv_sharing_readers[layer_name]):
                    handle.bind_cache(
                        kv_caches[cache_name],
                        byte_offset=byte_offset,
                        block_stride=tensor_config.block_stride,
                        num_blocks=kv_cache_config.num_blocks,
                        block_size=group.block_size,
                        block_table=cache_handle.block_table,
                        slot_mapping=cache_handle.slot_mapping,
                    )
                assert cache_handle.view is not None
                kv_caches[cache_name] = cache_handle.view.cache
            elif issubclass(group.manager_cls, HiSparseHotManager):
                layer_name = cache_name[: -len(HISPARSE_HOT_SUFFIX)]
                runtime = _get_hisparse_cache(forward_context, layer_name).runtime
                tensor_config, byte_offset = tensor_configs[cache_name]
                raw_tensor = kv_caches.pop(cache_name)
                readers = kv_sharing_readers[layer_name]
                for reader_runtime in (runtime, *(r.runtime for r in readers)):
                    reader_runtime.bind_hot_cache(
                        raw_tensor,
                        byte_offset=byte_offset,
                        block_stride=tensor_config.block_stride,
                        num_blocks=kv_cache_config.num_blocks,
                        block_size=group.block_size,
                        block_table=runtime.hot_block_table,
                        owner=None if reader_runtime is runtime else runtime,
                    )
