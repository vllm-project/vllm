# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from typing import cast

from vllm.config import VllmConfig
from vllm.config.kv_transfer import hisparse_host_pool_gib
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import select_common_block_size_from_constraints
from vllm.v1.hisparse.runtime import ResolvedHiSparseConfig
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheBlockPoolSpec,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCachePlacement,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)

HISPARSE_HOT_SUFFIX = ".hisparse_hot"
HISPARSE_RESIDENT_SUFFIX = ".hisparse_resident"

# HiSparse pages sparse-MLA KV from a host pool onto the shared device pool.
DEVICE_BLOCK_POOL_ID = 0
HOST_BLOCK_POOL_ID = 1


@dataclass(frozen=True)
class HiSparseLayout:
    source_group: KVCacheGroupSpec
    device_groups: list[KVCacheGroupSpec]
    host_tensors: list[KVCacheTensor]
    host_num_blocks: int

    @property
    def host_block_pool(self) -> KVCacheBlockPoolSpec:
        return KVCacheBlockPoolSpec(self.host_num_blocks, KVCachePlacement.HOST)


def get_hisparse_host_pool_bytes(vllm_config: VllmConfig) -> int | None:
    host_pool_gib = hisparse_host_pool_gib(vllm_config.kv_transfer_config)
    hisparse_enabled = vllm_config.attention_config.hisparse_config is not None
    if hisparse_enabled != (host_pool_gib is not None):
        raise ValueError(
            "HiSparse requires both attention_config.hisparse_config and "
            "HiSparseConnector with host_pool_gib"
        )
    return int(host_pool_gib * 2**30) if host_pool_gib is not None else None


def _partition_hisparse_specs(
    groups: list[KVCacheGroupSpec],
) -> tuple[dict[str, MLAAttentionSpec], dict[str, MLAAttentionSpec]]:
    group_spec = groups[0].kv_cache_spec
    if not isinstance(group_spec, UniformTypeKVCacheSpecs):
        raise ValueError("HiSparse requires uniform sparse-MLA cache specs.")

    specs = group_spec.kv_cache_specs
    if any(
        isinstance(spec, MLAAttentionSpec) and spec.model_version == "deepseek_v4"
        for spec in specs.values()
    ):
        raise ValueError("HiSparse does not support DeepSeek V4.")
    if not all(isinstance(spec, MLAAttentionSpec) for spec in specs.values()):
        raise ValueError("HiSparse requires its first cache group to contain MLA only.")

    source_specs = {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, MLAAttentionSpec)
        and spec.cache_role is SparseCacheRole.SPARSE
    }
    indexer_specs = {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, MLAAttentionSpec)
        and spec.cache_role is SparseCacheRole.INDEXER
    }
    if not source_specs or not indexer_specs:
        raise ValueError("HiSparse requires sparse-MLA and indexer cache specs.")
    return source_specs, indexer_specs


def get_hisparse_gpu_memory_usage(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int | None:
    if vllm_config.attention_config.hisparse_config is None or not kv_cache_groups:
        return None

    _, indexer_specs = _partition_hisparse_specs(kv_cache_groups)
    return sum(
        spec.max_memory_usage_bytes(vllm_config) for spec in indexer_specs.values()
    ) + sum(
        group.kv_cache_spec.max_memory_usage_bytes(vllm_config)
        for group in kv_cache_groups[1:]
    )


def create_hisparse_layout(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    host_budget: int,
) -> HiSparseLayout:
    source_specs, indexer_specs = _partition_hisparse_specs(groups)
    scheduler_block_sizes = {spec.block_size for spec in source_specs.values()}
    scheduler_block_sizes.update(spec.block_size for spec in indexer_specs.values())
    if len(scheduler_block_sizes) != 1:
        raise ValueError("HiSparse requires one scheduler block size.")
    scheduler_block_size = scheduler_block_sizes.pop()
    constraints = [
        spec.supported_kernel_block_sizes
        for spec in (*source_specs.values(), *indexer_specs.values())
        if isinstance(spec, AttentionSpec)
    ]
    try:
        gpu_block_size = select_common_block_size_from_constraints(
            scheduler_block_size, constraints
        )
    except ValueError as error:
        raise ValueError(
            "HiSparse requires a GPU block size supported by every sparse "
            f"attention and indexer backend: {error}"
        ) from error

    config = ResolvedHiSparseConfig.from_vllm_config(
        vllm_config,
        vllm_config.model_config.hf_config.index_topk,
        gpu_block_size,
    )
    assert config is not None
    source_specs = {
        name: spec.copy_with_new_block_size(gpu_block_size)
        for name, spec in source_specs.items()
    }
    indexer_specs = {
        name: spec.copy_with_new_block_size(gpu_block_size)
        for name, spec in indexer_specs.items()
    }

    indexer_group_spec = UniformTypeKVCacheSpecs.from_specs(
        cast(dict[str, KVCacheSpec], indexer_specs)
    )
    assert indexer_group_spec is not None
    indexer_group = KVCacheGroupSpec(
        list(indexer_specs),
        indexer_group_spec,
        block_pool_id=0,
        enable_prefix_caching=True,
        enable_kv_transfer=True,
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )

    indexer_page = sum(spec.page_size_bytes for spec in indexer_specs.values())
    hot_blocks_per_request = cdiv(config.device_buffer_size, gpu_block_size)
    hot_units: list[list[tuple[str, MLAAttentionSpec]]] = []
    for layer_name, layer_spec in source_specs.items():
        if layer_spec.is_index_group_leader or not hot_units:
            hot_units.append([])
        hot_units[-1].append((f"{layer_name}{HISPARSE_HOT_SUFFIX}", layer_spec))

    resident_groups: list[KVCacheGroupSpec] = []
    hot_groups: list[KVCacheGroupSpec] = []

    def append_hot_group(layers: list[tuple[str, MLAAttentionSpec]]) -> None:
        page_sizes = {spec.page_size_bytes for _, spec in layers}
        if len(page_sizes) != 1:
            raise ValueError(
                "HiSparse hot-cache groups require one page size, got "
                f"{sorted(page_sizes)}."
            )
        page_size = page_sizes.pop()
        names = [name for name, _ in layers]
        resident_groups.append(
            KVCacheGroupSpec(
                [
                    name[: -len(HISPARSE_HOT_SUFFIX)] + HISPARSE_RESIDENT_SUFFIX
                    for name in names
                ],
                HiSparseResidentSpec(
                    block_size=gpu_block_size,
                    page_size=page_size,
                ),
                block_pool_id=0,
                enable_prefix_caching=False,
                enable_kv_transfer=False,
            )
        )
        hot_groups.append(
            KVCacheGroupSpec(
                names,
                HiSparseHotSpec(
                    block_size=gpu_block_size,
                    page_size=page_size,
                    blocks_per_request=hot_blocks_per_request,
                ),
                block_pool_id=0,
                enable_prefix_caching=False,
                enable_kv_transfer=False,
            )
        )

    current: list[tuple[str, MLAAttentionSpec]] = []
    current_page = 0
    for unit in hot_units:
        unit_page = sum(spec.page_size_bytes for _, spec in unit)
        if current and current_page + unit_page > indexer_page:
            append_hot_group(current)
            current = []
            current_page = 0
        current.extend(unit)
        current_page += unit_page
    if current:
        append_hot_group(current)

    source_group_spec = UniformTypeKVCacheSpecs.from_specs(
        cast(dict[str, KVCacheSpec], source_specs)
    )
    assert source_group_spec is not None
    source_group = KVCacheGroupSpec(
        list(source_specs),
        source_group_spec,
        block_pool_id=HOST_BLOCK_POOL_ID,
        enable_kv_transfer=True,
    )
    regular_groups = [replace(group, block_pool_id=0) for group in groups[1:]]
    gpu_groups = [indexer_group, *resident_groups, *hot_groups, *regular_groups]

    host_num_blocks = host_budget // sum(
        spec.page_size_bytes for spec in source_specs.values()
    )
    if host_num_blocks <= 0:
        raise ValueError("HiSparse has no allocatable host blocks.")

    tensors = [
        KVCacheTensor(
            size=spec.page_size_bytes * host_num_blocks,
            layers=[name],
            layer_stride=spec.page_size_bytes * host_num_blocks,
            block_stride=spec.page_size_bytes,
            block_pool_id=HOST_BLOCK_POOL_ID,
        )
        for name, spec in source_specs.items()
    ]
    return HiSparseLayout(
        source_group=source_group,
        device_groups=gpu_groups,
        host_tensors=tensors,
        host_num_blocks=host_num_blocks,
    )
