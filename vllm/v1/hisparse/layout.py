# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from dataclasses import dataclass, replace

import regex as re

from vllm.config import VllmConfig
from vllm.config.kv_transfer import hisparse_host_pool_gib
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.hisparse.runtime import (
    ResolvedHiSparseConfig,
    get_hisparse_host_block_stride,
    use_shared_hisparse_host_pool,
)
from vllm.v1.hisparse.types import ACTIVE_TAIL_PAGES
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

logger = init_logger(__name__)

HISPARSE_HOT_SUFFIX = ".hisparse_hot"
HISPARSE_RESIDENT_SUFFIX = ".hisparse_resident"


@dataclass(frozen=True)
class HiSparseLayout:
    source_group: KVCacheGroupSpec
    device_groups: list[KVCacheGroupSpec]
    host_num_blocks: int
    host_block_stride: int
    shared_host_pool: bool


def dsv41_offloaded_sources(vllm_config: VllmConfig) -> set[int]:
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    extra = transfer.kv_connector_extra_config if transfer is not None else {}
    sources = set(extra.get("dsv41_sources", [20]))
    if not sources or not sources <= {2, 8, 14, 20}:
        raise ValueError("dsv41_sources must select KV owners 2, 8, 14, or 20")
    return sources


def is_hisparse_spec(
    vllm_config: VllmConfig, spec: KVCacheSpec, layer_name: str | None = None
) -> bool:
    if not isinstance(spec, MLAAttentionSpec):
        return False
    if (
        getattr(vllm_config.model_config.hf_config, "model_type", None)
        == "deepseek_v41"
    ):
        sources = dsv41_offloaded_sources(vllm_config)
        if sources == {20}:
            return spec.tokens_per_state == 1
        if spec.cache_role is SparseCacheRole.INDEXER:
            return True
        if layer_name is None:
            return spec.tokens_per_state in ({1} if sources == {20} else {1, 2})
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        return match is not None and int(match[1]) in sources
    return True


def get_hisparse_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec] | None:
    """The groups HiSparse allocates: the host source group, then the GPU
    groups sharing one device pool. Derived resident/hot specs in
    `kv_cache_spec` (see `resolve_hisparse_specs`) are laid out again
    from the attention specs they belong to."""
    attention_config = getattr(vllm_config, "attention_config", None)
    if attention_config is None or attention_config.hisparse_config is None:
        return None

    mla_specs: dict[str, KVCacheSpec] = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if is_hisparse_spec(vllm_config, spec, name)
    }
    other_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not is_hisparse_spec(vllm_config, spec, name)
        and not isinstance(spec, (HiSparseResidentSpec, HiSparseHotSpec))
    }
    if not mla_specs:
        return None

    from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

    mla_group_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)
    assert mla_group_spec is not None
    mla_group = KVCacheGroupSpec(list(mla_specs), mla_group_spec)
    regular_groups = (
        get_kv_cache_groups(vllm_config, other_specs) if other_specs else []
    )
    return _lay_out_hisparse_groups(vllm_config, [mla_group, *regular_groups])


def get_hisparse_host_pool_bytes(vllm_config: VllmConfig) -> int:
    host_pool_gib = hisparse_host_pool_gib(vllm_config.kv_transfer_config)
    if host_pool_gib is None:
        raise ValueError("HiSparse requires HiSparseConnector with host_pool_gib")
    return int(host_pool_gib * 2**30)


def _partition_hisparse_specs(
    groups: list[KVCacheGroupSpec], vllm_config: VllmConfig
) -> tuple[dict[str, MLAAttentionSpec], dict[str, MLAAttentionSpec]]:
    group_spec = groups[0].kv_cache_spec
    if not isinstance(group_spec, UniformTypeKVCacheSpecs):
        raise ValueError("HiSparse requires uniform sparse-MLA cache specs.")

    specs = group_spec.kv_cache_specs
    if (
        any(
            isinstance(spec, MLAAttentionSpec) and spec.model_version == "deepseek_v4"
            for spec in specs.values()
        )
        and getattr(vllm_config.model_config.hf_config, "model_type", None)
        != "deepseek_v41"
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


def _lay_out_hisparse_groups(
    vllm_config: VllmConfig, groups: list[KVCacheGroupSpec]
) -> list[KVCacheGroupSpec]:
    source_specs, indexer_specs = _partition_hisparse_specs(groups, vllm_config)
    block_sizes = {
        spec.block_size for spec in (*source_specs.values(), *indexer_specs.values())
    }
    if len(block_sizes) != 1:
        raise ValueError("HiSparse requires one resolved GPU block size.")
    gpu_block_size = block_sizes.pop()

    config = ResolvedHiSparseConfig.from_vllm_config(
        vllm_config,
        vllm_config.model_config.hf_config.index_topk,
        gpu_block_size,
    )
    assert config is not None
    device_specs: dict[str, KVCacheSpec] = dict(indexer_specs)
    regular_groups = []
    for group in groups[1:]:
        spec = group.kv_cache_spec
        candidate = (
            dict(spec.kv_cache_specs)
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else dict.fromkeys(group.layer_names, spec)
        )
        merged = device_specs | candidate
        if UniformTypeKVCacheSpecs.is_uniform_type(merged):
            device_specs = merged
        else:
            regular_groups.append(group)
    indexer_group_spec = UniformTypeKVCacheSpecs.from_specs(device_specs)
    assert indexer_group_spec is not None
    indexer_group = KVCacheGroupSpec(
        list(device_specs),
        indexer_group_spec,
        enable_kv_transfer=True,
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )

    indexer_page = sum(spec.page_size_bytes for spec in device_specs.values())
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
        physical_block_size = layers[0][1].num_states
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
                enable_kv_transfer=False,
            )
        )
        hot_groups.append(
            KVCacheGroupSpec(
                names,
                HiSparseHotSpec(
                    block_size=gpu_block_size,
                    page_size=page_size,
                    blocks_per_request=cdiv(
                        config.device_buffer_size, physical_block_size
                    ),
                    supports_strided_attention=(
                        getattr(vllm_config.model_config.hf_config, "model_type", None)
                        == "deepseek_v41"
                    ),
                ),
                enable_kv_transfer=False,
            )
        )

    current: list[tuple[str, MLAAttentionSpec]] = []
    current_page = 0
    for unit in hot_units:
        unit_page = sum(spec.page_size_bytes for _, spec in unit)
        if current and (
            current_page + unit_page > indexer_page
            or current[0][1].page_size_bytes != unit[0][1].page_size_bytes
        ):
            append_hot_group(current)
            current = []
            current_page = 0
        current.extend(unit)
        current_page += unit_page
    if current:
        append_hot_group(current)

    source_group_spec = UniformTypeKVCacheSpecs.from_specs(
        {
            name: replace(spec, alignment=None, page_size_padded=None)
            for name, spec in source_specs.items()
        }
    )
    assert source_group_spec is not None
    source_group = KVCacheGroupSpec(
        list(source_specs),
        source_group_spec,
        role=KVCacheGroupRole.HISPARSE_SOURCE,
        host_resident=True,
        enable_kv_transfer=True,
    )
    return [
        source_group,
        indexer_group,
        *resident_groups,
        *hot_groups,
        *(
            split
            for group in regular_groups
            for split in _split_group_to_block(group, indexer_page)
        ),
    ]


def _split_group_to_block(
    group: KVCacheGroupSpec, block_bytes: int
) -> list[KVCacheGroupSpec]:
    """Keep regular groups from padding every block to their original width."""
    spec = group.kv_cache_spec
    specs = (
        dict(spec.kv_cache_specs)
        if isinstance(spec, UniformTypeKVCacheSpecs)
        else dict.fromkeys(group.layer_names, spec)
    )
    page_size_layers: defaultdict[int, list[str]] = defaultdict(list)
    for name, layer_spec in specs.items():
        page_size_layers[layer_spec.page_size_bytes].append(name)
    if len(set(map(len, page_size_layers.values()))) != 1:
        return [group]
    repeats = list(zip(*page_size_layers.values()))
    per_group = max(block_bytes // sum(page_size_layers), 1)
    num_groups = cdiv(len(repeats), per_group)
    if num_groups == 1:
        return [group]
    result = []
    for i in range(num_groups):
        names = [name for repeat in repeats[i::num_groups] for name in repeat]
        split_spec = spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            uniform_spec = UniformTypeKVCacheSpecs.from_specs(
                {name: specs[name] for name in names}
            )
            assert uniform_spec is not None
            split_spec = uniform_spec
        result.append(replace(group, layer_names=names, kv_cache_spec=split_spec))
    return result


def create_hisparse_layout(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroupSpec],
    host_budget: int,
) -> HiSparseLayout:
    """Size the host pool for groups laid out by `get_hisparse_kv_cache_groups`."""
    (source_group,) = [group for group in groups if group.host_resident]
    shared_host_pool = use_shared_hisparse_host_pool(vllm_config)
    host_block_stride = get_hisparse_host_block_stride(
        source_group.kv_cache_spec.page_size_bytes,
        use_shared_host_pool=shared_host_pool,
    )
    host_num_blocks = host_budget // host_block_stride
    if host_num_blocks <= 0:
        raise ValueError("HiSparse has no allocatable host blocks.")
    # Every computed page needs a host block, so one request at max_model_len
    # must fit alongside the pool's null block and a copy-on-write tail.
    gpu_block_size = source_group.kv_cache_spec.block_size
    min_host_blocks = cdiv(vllm_config.model_config.max_model_len, gpu_block_size) + 2
    if host_num_blocks < min_host_blocks:
        raise ValueError(
            f"HiSparse host pool has {host_num_blocks} blocks but max_model_len "
            f"needs {min_host_blocks}; increase host_pool_gib."
        )

    return HiSparseLayout(
        source_group=source_group,
        device_groups=[group for group in groups if not group.host_resident],
        host_num_blocks=host_num_blocks,
        host_block_stride=host_block_stride,
        shared_host_pool=shared_host_pool,
    )


def _build_hisparse_kv_cache_tensors(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_blocks: int,
    size: int,
    layout: KVCacheLayout,
    bytes_per_block: int,
    *,
    host_resident: bool = False,
) -> list[KVCacheTensor]:
    interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None
    tensors: list[KVCacheTensor] = []
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            for layer_name, spec in group_spec.kv_cache_specs.items():
                layers_by_spec[spec].append(layer_name)
        elif group.layer_names:
            layers_by_spec[group_spec].extend(group.layer_names)

        byte_offset = 0
        for spec, layer_names in layers_by_spec.items():
            if isinstance(spec, (HiSparseHotSpec, HiSparseResidentSpec)):
                if not layout.is_block_outermost:
                    raise ValueError("HiSparse requires a block-outermost KV layout.")
                layer_stride = spec.page_size_bytes
                block_stride = bytes_per_block
            else:
                layer_stride, block_stride, _, _, _ = compute_layout_strides(
                    spec,
                    num_blocks,
                    len(layer_names),
                    layout,
                    fixed_strides=(None, interleaved_block_stride, None, None, None),
                )
            offset = (
                byte_offset
                * max(layer_stride, spec.page_size_bytes)
                // spec.page_size_bytes
            )
            tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=layer_names,
                    layer_stride=layer_stride,
                    block_stride=block_stride,
                    offset=offset,
                    host_resident=host_resident,
                )
            )
            byte_offset += len(layer_names) * spec.page_size_bytes
    return tensors


def get_hisparse_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
    host_budget: int,
) -> KVCacheConfig:
    from vllm.v1.core.kv_cache_planning import (
        _get_kv_cache_bytes_per_block,
        may_override_num_blocks,
        validate_kv_cache_layout,
    )

    hisparse_layout = create_hisparse_layout(vllm_config, kv_cache_groups, host_budget)
    device_groups = hisparse_layout.device_groups
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    validate_kv_cache_layout(layout, device_groups)
    bytes_per_block = _get_kv_cache_bytes_per_block(device_groups, layout)
    num_blocks = may_override_num_blocks(
        vllm_config, available_memory // bytes_per_block
    )
    size = bytes_per_block * num_blocks
    kv_cache_tensors = _build_hisparse_kv_cache_tensors(
        device_groups, num_blocks, size, layout, bytes_per_block
    )

    host_groups = [hisparse_layout.source_group]
    host_layout = KVCacheLayout.LBNHC
    validate_kv_cache_layout(host_layout, host_groups)
    host_bytes_per_block = _get_kv_cache_bytes_per_block(host_groups, host_layout)
    host_size = host_bytes_per_block * hisparse_layout.host_num_blocks
    kv_cache_tensors[:0] = _build_hisparse_kv_cache_tensors(
        host_groups,
        hisparse_layout.host_num_blocks,
        host_size,
        host_layout,
        host_bytes_per_block,
        host_resident=True,
    )
    logger.info_once(
        "HiSparse HMA: %.1f GiB host source (%d blocks), %.1f GiB shared "
        "GPU indexer/resident/hot pool (%d blocks).",
        host_size / 2**30,
        hisparse_layout.host_num_blocks,
        size / 2**30,
        num_blocks,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=[*host_groups, *device_groups],
        hisparse_host_num_blocks=hisparse_layout.host_num_blocks,
        hisparse_host_block_stride=hisparse_layout.host_block_stride,
        hisparse_shared_host_pool=hisparse_layout.shared_host_pool,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )


def get_hisparse_steady_state_concurrency(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """Max concurrency at max_model_len once running requests read from host.

    A request reading from host pins only its active tail of each resident
    group, but admitting one still takes its full in-flight window, so the
    bound is all-but-one requests at steady state plus one being admitted.
    """
    admission_blocks = 0
    steady_blocks = 0
    host_blocks = 0
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        required = cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
        if group.host_resident:
            host_blocks += required
            continue
        admission_blocks += required
        if isinstance(spec, HiSparseResidentSpec):
            required = min(required, ACTIVE_TAIL_PAGES)
        steady_blocks += required
    assert kv_cache_config.hisparse_host_num_blocks is not None
    num_blocks = kv_cache_config.num_blocks
    if num_blocks < admission_blocks:
        gpu_concurrency = num_blocks / admission_blocks
    else:
        gpu_concurrency = 1 + (num_blocks - admission_blocks) / steady_blocks
    return min(gpu_concurrency, kv_cache_config.hisparse_host_num_blocks / host_blocks)
