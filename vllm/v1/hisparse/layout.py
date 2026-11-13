# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from dataclasses import dataclass, replace

from vllm.config import VllmConfig
from vllm.config.kv_transfer import hisparse_host_pool_gib
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    HiSparseHotManager,
    HiSparseResidentManager,
    HiSparseSourceManager,
)
from vllm.v1.hisparse.runtime import (
    ResolvedHiSparseConfig,
    get_hisparse_host_block_stride,
    use_shared_hisparse_host_pool,
)
from vllm.v1.hisparse.types import ACTIVE_TAIL_PAGES
from vllm.v1.kv_cache_interface import (
    HiSparseHotPageSpec,
    HiSparsePageSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVCacheGroupRole,
    KVCacheLayerConfig,
    KVCacheSpec,
    KVCacheTensor,
    ManagerConfig,
    MLAAttentionSpec,
    SparseCacheRole,
    SparseFullAttentionSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

logger = init_logger(__name__)

HISPARSE_HOT_SUFFIX = ".hisparse_hot"
HISPARSE_RESIDENT_SUFFIX = ".hisparse_resident"
# DeepGEMM reads indexer pages through TMA.
_INDEXER_PAGE_ALIGNMENT = 16

_SourceSpec = MLAAttentionSpec | SparseFullAttentionSpec


@dataclass(frozen=True)
class HiSparseLayout:
    source_group: KVCacheGroup
    device_groups: list[KVCacheGroup]
    host_num_blocks: int
    host_block_stride: int
    shared_host_pool: bool


def is_hisparse_source_layer(layer_config: KVCacheLayerConfig) -> bool:
    """Whether the layer participates in HiSparse as a sparse source or indexer."""
    return isinstance(layer_config.spec, _SourceSpec) and issubclass(
        layer_config.manager_cls, FullAttentionManager
    )


def get_hisparse_kv_cache_groups(
    vllm_config: VllmConfig, layer_configs: dict[str, KVCacheLayerConfig]
) -> list[KVCacheGroup] | None:
    """The groups HiSparse allocates: the host source group, then the GPU
    groups sharing one device pool. Derived resident/hot specs in
    `layer_configs` (see `resolve_hisparse_specs`) are laid out again
    from the attention specs they belong to."""
    attention_config = getattr(vllm_config, "attention_config", None)
    if attention_config is None or attention_config.hisparse_config is None:
        return None

    sparse_layers = {
        name: lc for name, lc in layer_configs.items() if is_hisparse_source_layer(lc)
    }
    other_specs = {
        name: lc
        for name, lc in layer_configs.items()
        if name not in sparse_layers and not isinstance(lc.spec, HiSparsePageSpec)
    }
    if not sparse_layers:
        return None

    from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

    regular_groups = (
        get_kv_cache_groups(vllm_config, other_specs) if other_specs else []
    )
    return _lay_out_hisparse_groups(vllm_config, sparse_layers, regular_groups)


def _lay_out_hisparse_groups(
    vllm_config: VllmConfig,
    sparse_layers: dict[str, KVCacheLayerConfig],
    regular_groups: list[KVCacheGroup],
) -> list[KVCacheGroup]:
    """Lay out the groups to fill the largest regular page. When a block-outer
    layout's pool block is not whole indexer pages, either the block grows to
    whole pages or the pages spread out to divide the block, whichever takes
    less memory per request. Before the layout is resolved, only the per-layer
    specs are used."""
    from vllm.v1.core.kv_cache_planning import _pool_bytes_per_block

    block_bytes = max((g.page_size_bytes for g in regular_groups), default=0)
    groups = _fill_hisparse_groups(
        vllm_config, sparse_layers, regular_groups, block_bytes
    )
    if not block_bytes or vllm_config.cache_config.kv_cache_layout is None:
        return groups
    if _pool_bytes_per_block(vllm_config, groups) == block_bytes:
        return groups
    other_groups = [
        group for group in groups if group.role is not KVCacheGroupRole.HISPARSE_INDEXER
    ]
    candidates = [
        groups,
        _fill_hisparse_groups(
            vllm_config,
            sparse_layers,
            regular_groups,
            _pool_bytes_per_block(vllm_config, other_groups),
            spread_indexer_pages=True,
        ),
    ]
    return min(
        candidates,
        key=lambda groups: (
            _pool_bytes_per_block(vllm_config, groups)
            * sum(
                group.max_allocated_blocks_per_request(vllm_config)
                for group in groups
                if not group.host_resident
            )
        ),
    )


def get_hisparse_host_pool_bytes(vllm_config: VllmConfig) -> int:
    host_pool_gib = hisparse_host_pool_gib(vllm_config.kv_transfer_config)
    if host_pool_gib is None:
        raise ValueError("HiSparse requires HiSparseConnector with host_pool_gib")
    return int(host_pool_gib * 2**30)


def _partition_hisparse_specs(
    specs: dict[str, KVCacheSpec],
) -> tuple[dict[str, _SourceSpec], dict[str, MLAAttentionSpec]]:
    if any(
        isinstance(spec, MLAAttentionSpec) and spec.model_version == "deepseek_v4"
        for spec in specs.values()
    ):
        raise ValueError("HiSparse does not support DeepSeek V4.")
    if not all(isinstance(spec, _SourceSpec) for spec in specs.values()):
        raise ValueError("HiSparse requires sparse attention cache specs.")

    source_specs = {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, SparseFullAttentionSpec)
        or (
            isinstance(spec, MLAAttentionSpec)
            and spec.cache_role is SparseCacheRole.SPARSE
        )
    }
    indexer_specs = {
        name: spec
        for name, spec in specs.items()
        if isinstance(spec, MLAAttentionSpec)
        and spec.cache_role is SparseCacheRole.INDEXER
    }
    if not source_specs or not indexer_specs:
        raise ValueError("HiSparse requires sparse attention and indexer cache specs.")
    return source_specs, indexer_specs


def get_hisparse_page_layer_configs(
    vllm_config: VllmConfig, sparse_layers: dict[str, KVCacheLayerConfig]
) -> dict[str, KVCacheLayerConfig]:
    """The resident and hot layers HiSparse derives from its source layers.
    They are per-layer, so they don't depend on the KV cache layout that
    groups them."""
    source_specs, _ = _partition_hisparse_specs(
        {name: lc.spec for name, lc in sparse_layers.items()}
    )
    block_sizes = {spec.block_size for spec in source_specs.values()}
    if len(block_sizes) != 1:
        raise ValueError("HiSparse requires one resolved GPU block size.")
    gpu_block_size = block_sizes.pop()
    config = ResolvedHiSparseConfig.from_vllm_config(
        vllm_config,
        max(
            spec.topk_buffer_width
            if spec.topk_buffer_width is not None
            else vllm_config.model_config.hf_text_config.index_topk
            for spec in source_specs.values()
        ),
        gpu_block_size,
    )
    assert config is not None
    resident_manager = HiSparseResidentManager.config()
    hot_manager = HiSparseHotManager.config(
        blocks_per_request=cdiv(config.device_buffer_size, gpu_block_size)
    )
    layer_configs: dict[str, KVCacheLayerConfig] = {}
    for name, spec in source_specs.items():
        layer_configs[f"{name}{HISPARSE_RESIDENT_SUFFIX}"] = KVCacheLayerConfig(
            resident_manager,
            HiSparsePageSpec(block_size=gpu_block_size, page_size=spec.page_size_bytes),
        )
        layer_configs[f"{name}{HISPARSE_HOT_SUFFIX}"] = KVCacheLayerConfig(
            hot_manager,
            HiSparseHotPageSpec(
                block_size=gpu_block_size,
                page_size=spec.page_size_bytes,
                block_stride_alignment=spec.block_stride_alignment,
            ),
        )
    return layer_configs


def _fill_hisparse_groups(
    vllm_config: VllmConfig,
    sparse_layers: dict[str, KVCacheLayerConfig],
    regular_groups: list[KVCacheGroup],
    block_bytes: int,
    spread_indexer_pages: bool = False,
) -> list[KVCacheGroup]:
    source_specs, indexer_specs = _partition_hisparse_specs(
        {name: lc.spec for name, lc in sparse_layers.items()}
    )
    page_layer_configs = get_hisparse_page_layer_configs(vllm_config, sparse_layers)
    if len({spec.block_size for spec in indexer_specs.values()}) != 1:
        raise ValueError("HiSparse requires one indexer block size.")
    indexer_page_bytes = sum(spec.page_size_bytes for spec in indexer_specs.values())
    page_stride = None
    if spread_indexer_pages:
        page_bytes = indexer_page_bytes // len(indexer_specs)
        page_stride = block_bytes // next(
            pages
            for pages in range(block_bytes // page_bytes, 0, -1)
            if block_bytes % (pages * _INDEXER_PAGE_ALIGNMENT) == 0
        )
        indexer_page_bytes = page_stride * len(indexer_specs)
    pages_per_block = max(block_bytes // indexer_page_bytes, 1)
    indexer_group = KVCacheGroup.from_layers(
        {
            name: replace(
                sparse_layers[name],
                spec=replace(
                    spec.copy_with_new_block_size(spec.block_size * pages_per_block),
                    page_size_padded=page_stride and page_stride * pages_per_block,
                    block_stride_alignment=page_stride or spec.block_stride_alignment,
                ),
            )
            for name, spec in indexer_specs.items()
        },
        enable_kv_transfer=True,
        role=KVCacheGroupRole.HISPARSE_INDEXER,
    )

    hot_group_page_limit = max(block_bytes, indexer_group.page_size_bytes)
    hot_units: list[list[tuple[str, _SourceSpec]]] = []
    for layer_name, layer_spec in source_specs.items():
        if layer_spec.is_index_group_leader or not hot_units:
            hot_units.append([])
        hot_units[-1].append((layer_name, layer_spec))

    resident_groups: list[KVCacheGroup] = []
    hot_groups: list[KVCacheGroup] = []

    def append_hot_group(layers: list[tuple[str, _SourceSpec]]) -> None:
        page_sizes = {spec.page_size_bytes for _, spec in layers}
        if len(page_sizes) != 1:
            raise ValueError(
                "HiSparse hot-cache groups require one page size, got "
                f"{sorted(page_sizes)}."
            )
        alignments = {spec.block_stride_alignment for _, spec in layers}
        if len(alignments) != 1:
            raise ValueError(
                "HiSparse hot-cache groups require one block stride alignment, "
                f"got {alignments}."
            )
        for suffix, groups in (
            (HISPARSE_RESIDENT_SUFFIX, resident_groups),
            (HISPARSE_HOT_SUFFIX, hot_groups),
        ):
            groups.append(
                KVCacheGroup.from_layers(
                    {
                        f"{name}{suffix}": page_layer_configs[f"{name}{suffix}"]
                        for name, _ in layers
                    },
                    enable_kv_transfer=False,
                )
            )

    current: list[tuple[str, _SourceSpec]] = []
    current_page = 0
    for unit in hot_units:
        unit_page = sum(spec.page_size_bytes for _, spec in unit)
        if current and current_page + unit_page > hot_group_page_limit:
            append_hot_group(current)
            current = []
            current_page = 0
        current.extend(unit)
        current_page += unit_page
    if current:
        append_hot_group(current)

    source_group = KVCacheGroup.from_layers(
        {
            name: replace(
                sparse_layers[name],
                manager_config=ManagerConfig(
                    HiSparseSourceManager, sparse_layers[name].manager_config.params
                ),
            )
            for name in source_specs
        },
        host_resident=True,
        enable_kv_transfer=True,
    )
    return [
        source_group,
        indexer_group,
        *resident_groups,
        *hot_groups,
        *regular_groups,
    ]


def create_hisparse_layout(
    vllm_config: VllmConfig,
    groups: list[KVCacheGroup],
    host_budget: int,
) -> HiSparseLayout:
    """Size the host pool for groups laid out by `get_hisparse_kv_cache_groups`."""
    (source_group,) = [group for group in groups if group.host_resident]
    sparse_full_specs = [
        spec
        for spec in source_group.kv_cache_specs.values()
        if isinstance(spec, SparseFullAttentionSpec)
    ]
    shared_host_pool = not sparse_full_specs and use_shared_hisparse_host_pool(
        vllm_config
    )
    host_block_stride = get_hisparse_host_block_stride(
        source_group.page_size_bytes,
        use_shared_host_pool=shared_host_pool,
    )
    logical_block_bytes = host_block_stride
    if sparse_full_specs:
        total_num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        logical_block_bytes = sum(
            spec.num_states * total_num_kv_heads * spec.state_content_size_bytes
            for spec in sparse_full_specs
        )
    host_num_blocks = host_budget // logical_block_bytes
    if host_num_blocks <= 0:
        raise ValueError("HiSparse has no allocatable host blocks.")
    # Every computed page needs a host block, so one request at max_model_len
    # must fit alongside the pool's null block and a copy-on-write tail.
    gpu_block_size = source_group.block_size
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
    kv_cache_groups: list[KVCacheGroup],
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
        layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
        for layer_name, spec in group.kv_cache_specs.items():
            layers_by_spec[spec].append(layer_name)

        byte_offset = 0
        for spec, layer_names in layers_by_spec.items():
            if isinstance(spec, HiSparsePageSpec):
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
    kv_cache_groups: list[KVCacheGroup],
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
        required = group.max_allocated_blocks_per_request(vllm_config)
        if group.host_resident:
            host_blocks += required
            continue
        admission_blocks += required
        if issubclass(group.manager_cls, HiSparseResidentManager):
            required = min(required, ACTIVE_TAIL_PAGES)
        steady_blocks += required
    assert kv_cache_config.hisparse_host_num_blocks is not None
    num_blocks = kv_cache_config.num_blocks
    if num_blocks < admission_blocks:
        gpu_concurrency = num_blocks / admission_blocks
    else:
        gpu_concurrency = 1 + (num_blocks - admission_blocks) / steady_blocks
    return min(gpu_concurrency, kv_cache_config.hisparse_host_num_blocks / host_blocks)
