# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache planning: grouping, sizing, and memory-fit logic."""

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from functools import partial, reduce
from typing import Any

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_config_builder import KVCacheConfigBuilder
from vllm.v1.core.kv_cache_utils import _kv_cache_device_name
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CircularBufferManager,
    FullAttentionManager,
    HiSparseHotManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.hisparse.layout import (
    get_hisparse_host_pool_bytes,
    get_hisparse_kv_cache_config,
    get_hisparse_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayerConfig,
    KVCacheLayout,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    compute_layout_strides,
)

logger = init_logger(__name__)


def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            "(this flag also controls CPU memory reservation on the CPU "
            "backend, despite its name). "
            "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            "for more details."
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
            f"Try increasing `gpu_memory_utilization` (which also controls "
            f"CPU memory on the CPU backend) or decreasing `max_model_len` "
            f"when initializing the engine. "
            f"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
            f"for more details."
        )


def max_memory_usage_bytes(
    vllm_config: VllmConfig, layer_configs: Iterable[KVCacheLayerConfig]
) -> int:
    """Get the maximum memory usage in bytes for the given layers."""
    return sum(layer.max_memory_usage_bytes(vllm_config) for layer in layer_configs)


def estimate_max_model_len(
    vllm_config: VllmConfig,
    layer_configs: dict[str, KVCacheLayerConfig],
    available_memory: int,
) -> int:
    """Estimates the maximum model length that can fit in the available memory
    using binary search.

    This function temporarily modifies max_model_len during estimation but
    restores the original value before returning, ensuring no side effects.

    Args:
        vllm_config: The global VllmConfig
        layer_configs: The KV cache config of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.

    """
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, layer_configs.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len


def check_enough_kv_cache_memory(
    vllm_config: VllmConfig,
    layer_configs: dict[str, KVCacheLayerConfig],
    available_memory: int,
):
    """Checks whether `available_memory` is enough for the KV cache to hold at
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        layer_configs: The KV cache config of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.

    """
    # No need to check for available memory if there are no KV cache layers
    if layer_configs:
        # Reserve the null block BlockPool permanently holds back, so the check
        # plans against usable blocks, as in get_kv_cache_configs. Group a copy
        # since grouping may unify the layers in-place.
        groups = get_kv_cache_groups(vllm_config, dict(layer_configs))
        check_memory = (
            available_memory - _pool_bytes_per_block(vllm_config, groups)
            if groups
            else available_memory
        )
        _check_enough_kv_cache_memory(
            check_memory,
            lambda: max_memory_usage_bytes(vllm_config, layer_configs.values()),
            vllm_config.model_config.max_model_len,
            lambda am: estimate_max_model_len(vllm_config, layer_configs, am),
        )


def can_share_block_table(layer_configs: dict[str, KVCacheLayerConfig]) -> bool:
    """Whether the layers can form one KV cache group: one manager policy, block
    size and DCP sharding, with layouts the manager accepts side by side."""
    first = next(iter(layer_configs.values()))
    specs = [lc.spec for lc in layer_configs.values()]
    return (
        all(lc.manager_config == first.manager_config for lc in layer_configs.values())
        and len({(spec.block_size, spec.dcp_sharded) for spec in specs}) == 1
        and first.manager_cls.can_share_block_table(specs)
    )


def _bucket_layers(
    layer_configs: dict[str, KVCacheLayerConfig],
) -> list[dict[str, KVCacheLayerConfig]]:
    """Greedily bucket layers that can share a block table, in layer order."""
    buckets: list[dict[str, KVCacheLayerConfig]] = []
    for name, layer_config in layer_configs.items():
        for bucket in buckets:
            if can_share_block_table({**bucket, name: layer_config}):
                bucket[name] = layer_config
                break
        else:
            buckets.append({name: layer_config})
    return buckets


def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    """Override the number of kv cache blocks if `num_gpu_blocks_override` is set.
    The override is logged once, at the call site in `get_kv_cache_configs`.
    """
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks


def _pool_bytes_per_block(
    vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroupSpec]
) -> int:
    """Bytes consumed by one block in the worker's shared KV cache pool, mirroring
    the divisor used by `get_kv_cache_config_from_groups` to convert
    `available_memory` into `num_blocks`. Used to compute the effective KV cache
    capacity once `num_gpu_blocks_override` is applied.
    """
    return _get_kv_cache_bytes_per_block(
        [group for group in kv_cache_groups if not group.host_resident],
        vllm_config.cache_config.get_resolved_kv_cache_layout(),
    )


def _pool_bytes_per_block_from_builder(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Bytes consumed by one block in the worker's device KV cache pool.

    Core places one block with the builder and reads the size of its single
    device backing. A placement with several device allocations declares the
    value through ``KVCacheConfigBuilder._get_pool_bytes_per_block``. HiSparse
    keeps the pure computation, since probing its placement would check the host
    pool against ``max_model_len`` before auto-fit shortens it.
    ``num_gpu_blocks_override`` is suppressed so it cannot inflate the probe.
    """
    if not kv_cache_groups:
        return 0
    if vllm_config.attention_config.hisparse_config is not None:
        return _pool_bytes_per_block(vllm_config, kv_cache_groups)
    pool_bytes = builder._get_pool_bytes_per_block(vllm_config, kv_cache_groups)
    if pool_bytes is not None:
        return pool_bytes
    saved_override = vllm_config.cache_config.num_gpu_blocks_override
    vllm_config.cache_config.num_gpu_blocks_override = None
    try:
        probe = builder.get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, num_blocks=1
        )
    finally:
        vllm_config.cache_config.num_gpu_blocks_override = saved_override
    backing_sizes = {
        tensor.size for tensor in probe.kv_cache_tensors if not tensor.host_resident
    }
    if len(backing_sizes) != 1:
        raise ValueError(
            "KV cache tensors must share one device backing allocation, but got "
            f"sizes {sorted(backing_sizes)}. Override "
            "KVCacheConfigBuilder._get_pool_bytes_per_block() for placements "
            "with several device allocations."
        )
    return backing_sizes.pop()


def get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int:
    """Get the page size of the KV cache."""
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1
    return page_sizes.pop()


def unify_kv_cache_spec_page_size(
    layer_configs: dict[str, KVCacheLayerConfig],
) -> dict[str, KVCacheLayerConfig]:
    """Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size. Two
    cases cannot be unified by block size alone and pad their physical page to
    the maximum instead: Mamba layers, whose page size comes from state shapes
    and is independent of block size; and non-MLA attention layers whose page
    does not evenly divide the maximum (the padded page is read through a
    strided view). MLA is excluded because sparse MLA indexes the cache in
    whole token rows (see ``flat_kv_row_view``), so its block stride can only
    be padded by its own row-aligned ``alignment``, not to an arbitrary page
    size. Raise NotImplementedError if failed to unify the page size;
    ``get_kv_cache_groups`` catches it to try the full-allocation fallback
    (e.g. MLA next to an incompatible sliding-window draft).

    Args:
        layer_configs: The KV cache config of each attention layer in the model

    Returns:
        The updated layer configs whose specs share one page_size_bytes.

    """
    page_sizes = {layer.spec.page_size_bytes for layer in layer_configs.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return layer_configs

    max_page_size = max(page_sizes)
    new_layer_configs = {}
    for layer_name, layer_config in layer_configs.items():
        layer_spec = layer_config.spec
        if layer_spec.page_size_bytes == max_page_size:
            new_layer_configs[layer_name] = layer_config
        elif isinstance(layer_spec, MambaSpec):
            # MambaSpec's page size is determined by its state shapes and does
            # not scale with block_size, so pad the page instead. This is the
            # same padding mechanism the platform uses to align Mamba pages
            # with the main model's attention page size; it is needed here
            # when another layer (e.g. from a draft model) has a larger page
            # than the already-aligned Mamba page.
            new_spec: KVCacheSpec = replace(layer_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size
            new_layer_configs[layer_name] = replace(layer_config, spec=new_spec)
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif isinstance(layer_spec, AttentionSpec) and not isinstance(
                layer_spec, MLAAttentionSpec
            ):
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for non-MLA attention layers."
                )
            assert new_spec.page_size_bytes == max_page_size
            new_layer_configs[layer_name] = replace(layer_config, spec=new_spec)
    return new_layer_configs


def is_kv_cache_type_attention_free(configs: dict[str, KVCacheLayerConfig]) -> bool:
    # configs is an empty dict for attention free models
    return not configs


def _get_kv_cache_groups_uniform_page_size(
    layer_configs: dict[str, KVCacheLayerConfig],
) -> list[KVCacheGroupSpec]:
    """Generates the KV cache groups for hybrid models with multiple
    attention types but still with a uniform page size (physical memory per
    block per layer) for all layers.

    Detailed explanation about kv cache management of hybrid models:
    The layers in the models are repeated with some patterns, e.g., a model
    with 10 full attention layers and 20 sliding window attention layers can be
    regarded as repeating the pattern (1 * full, 2 * sw) 10 times.
    The KVCacheManager allocates different block tables for each of the 3 layers
    in the pattern, and repeats each of them 10 times to generate the
    block_table for the 30 layers in the model.
    Therefore, we can group the layers in the model into 3 kv_cache_groups, each
    of which contains 10 layers in the model.
    The KVCacheManager allocates the block_table for each group based on its
    kv_cache spec, and the model runner applies the block table to each layer
    in the group.
    For example:
    1. A model only uses full attention. The pattern is
    (num_hidden_layers * full), so there is only one group and the block table
    is shared by all layers.
    2. A model with 10 full attention layers and 20 sliding window
    attention layers. There are 3 layers in the pattern (1 * full, 2 * sw), so
    there are 3 kv_cache_groups, each of which represents 10 layers.

    To simplify the implementation, we make the following assumptions:
    1. Physical memory per block: Must be the same across all KV cache groups.
    Breaking this assumption is non-trivial due to memory fragmentation concerns
    when allocating blocks of different sizes.
    2. Tokens per block (block_size): Currently, we directly use
    `CacheConfig.block_size` for all layers. It can be extended to vary by KV
    cache group, but within each KV cache group, all layers must share the same
    block size.
    3. Physical memory per token per layer: This property is decided by model
    config. Currently we only support models that have the same physical memory
    per token per layer for all layers. Can be relaxed with a simple extension,
    but still need to keep physical memory per block the same for all groups.
    4. Number of layers per group: Currently assumed the same for all layers.
    Can be relaxed with a simple extension, but still need to keep physical
    memory per block the same for all groups.
    5. Attention type within groups: All layers in a group must share the same
    manager policy. When `--disable-hybrid-kv-cache-manager` is true, sliding
    window and LLaMA 4 local attention layers take the full attention policy.
    See `unify_hybrid_kv_cache_specs` for more details.
    6. Support for multiple attention types: The design for most components is
    general to an arbitrary number of attention types. But
    `find_longest_cache_hit` only supports one attention type or two
    types of full-attention plus exactly one another type. The general
    implementation of this function is feasible but we don't know how to
    implement it cleanly yet.

    As we assume tokens per block, physical memory per token per layer, and
    number of layers per group are the same now, we can ensure that physical
    memory per block is the same for all groups.

    Args:
        layer_configs: The KV cache config of each attention layer in the model
    Returns:
        The generated KVCacheGroupSpecs

    """
    # Group all layers that can share a block table.
    # E.g., 2 full attention layers and 3 sliding window attention layers,
    # -> (full.0, full.1), (sw.0, sw.1, sw.2).
    layer_buckets = [list(bucket) for bucket in _bucket_layers(layer_configs)]

    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in layer_buckets])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers
    grouped_layers = []
    for layers in layer_buckets:
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # the 3 groups in stage 0 will be (full.0), (sw.0, sw.1), (empty group)
        # and it will be padded to (full.0, padding), (sw.0, sw.1),
        # (padding, padding) to ensure the number of layers in each group is
        # the same and will cause memory waste.
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])
    return [
        KVCacheGroupSpec.from_layers({name: layer_configs[name] for name in names})
        for names in grouped_layers
    ]


def _get_layer_block_stride(spec: KVCacheSpec) -> int:
    """A layer-outer block stride: the page padded to its alignment."""
    return round_up(spec.page_size_bytes, spec.get_block_stride_alignment())


def _get_kv_cache_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec], layout: KVCacheLayout
) -> int:
    """Return the largest cache group's bytes per block."""
    if not layout.is_block_outermost:
        # Each layer's block stride is its own padded page.
        return max(
            sum(_get_layer_block_stride(spec) for spec in group.kv_cache_specs.values())
            for group in kv_cache_groups
        )
    bytes_per_block = max(group.page_size_bytes for group in kv_cache_groups)
    assert bytes_per_block > 0
    alignments = [
        spec.page_size_bytes
        for group in kv_cache_groups
        if issubclass(group.manager_cls, HiSparseHotManager)
        for spec in group.kv_cache_specs.values()
    ]
    alignments.extend(
        spec.get_block_stride_alignment()
        for group in kv_cache_groups
        for spec in group.kv_cache_specs.values()
    )
    return round_up(bytes_per_block, math.lcm(*alignments))


def validate_kv_cache_layout(
    layout: KVCacheLayout,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """Validate that the resolved layout can express this model's packing.

    The layout was chosen once in the engine core from the backends' supported
    sets; a backend whose model packs pages side by side (e.g. the DeepSeek-V4
    indexer) declares block-outermost layouts there, so an inexpressible
    layout reaching this point is an error.
    """
    page_sizes = {
        spec.page_size_bytes
        for group in kv_cache_groups
        for spec in group.kv_cache_specs.values()
    }
    if len(page_sizes) == 1:
        # A rectangular layer dim exists; every layout can express it.
        return

    # Mixed page sizes pack pages side by side within a block, which needs each page
    # to be one contiguous chunk inside its block (a block-compact layout) and, with
    # multiple KV cache groups, the layer dim inside the block dim.
    if not layout.is_block_compact or (
        len(kv_cache_groups) > 1 and layout.is_layer_compact
    ):
        raise ValueError(
            f"KV cache layout {layout.name} cannot express this model's "
            f"mixed page sizes ({sorted(page_sizes)}); a backend should "
            "declare block-outermost supported layouts (e.g. BLHNC), or "
            "set VLLM_KV_CACHE_LAYOUT=BLHNC."
        )


def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Generate the KV cache configuration from the KV cache groups and spec
    of each layer.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_groups: The KV cache groups
        available_memory: Memory available for KV cache in bytes
    Returns:
        The generated KVCacheConfig

    """
    if len(kv_cache_groups) == 0:
        # Attention free models do not have KV cache.
        # Return num_blocks=1 as BlockPool always needs a null_block.
        return KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )

    if vllm_config.attention_config.hisparse_config is not None:
        host_budget = get_hisparse_host_pool_bytes(vllm_config)
        return get_hisparse_kv_cache_config(
            vllm_config, kv_cache_groups, available_memory, host_budget
        )

    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    validate_kv_cache_layout(layout, kv_cache_groups)
    bytes_per_block = _get_kv_cache_bytes_per_block(kv_cache_groups, layout)
    interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None

    num_blocks = available_memory // bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    size = bytes_per_block * num_blocks

    # Groups alias from byte 0. Spec regions are laid out differently:
    #
    # block-outer (the same packing repeats for every block):
    # group 0: | blk 0 [ A | B  | pad ] | blk 1 [ A | B  | pad ] | ...
    # group 1: | blk 0 [  C  |    D   ] | blk 1 [  C  |    D   ] | ...
    #          |<--- bytes_per_block -->|
    #
    # layer-outer (only supported for uniform page sizes or single-group models):
    # group 0: | A [ blk 0 | blk 1 | ... ] | B [ blk 0 | blk 1 | ... ] |
    # group 1: | C [ blk 0 | blk 1 | ... ] | D [ blk 0 | blk 1 | ... ] |

    kv_cache_tensors = []
    for group in kv_cache_groups:
        layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
        for layer_name, spec in group.kv_cache_specs.items():
            layers_by_spec[spec].append(layer_name)

        byte_offset = 0
        for spec, layer_names in layers_by_spec.items():
            block_stride = interleaved_block_stride
            padded_page = _get_layer_block_stride(spec)
            if block_stride is None and padded_page != spec.page_size_bytes:
                assert layout.is_block_compact, f"{layout.name} cannot pad pages"
                block_stride = padded_page
            layer_stride, block_stride, _, _, _ = compute_layout_strides(
                spec,
                num_blocks,
                len(layer_names),
                layout,
                fixed_strides=(None, block_stride, None, None, None),
            )
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=size,
                    layers=layer_names,
                    layer_stride=layer_stride,
                    block_stride=block_stride,
                    offset=byte_offset,
                )
            )
            # Block-outer layers share each block; layer-outer ones own regions.
            byte_offset += len(layer_names) * (
                spec.page_size_bytes if layout.is_block_outermost else layer_stride
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )


def _is_local_attention(layer_config: KVCacheLayerConfig) -> bool:
    return issubclass(
        layer_config.manager_cls, (SlidingWindowManager, ChunkedLocalAttentionManager)
    )


def _promote_local_kv_cache_specs(
    layer_configs: dict[str, KVCacheLayerConfig],
) -> dict[str, KVCacheLayerConfig]:
    """Use full-attention allocation for local-attention layers.

    The returned configs affect KV cache management only. Attention modules
    keep their original sliding-window or chunked-local compute behavior.
    """
    promoted_specs = layer_configs.copy()

    if can_share_block_table(promoted_specs):
        return promoted_specs

    full_block_sizes = {
        lc.spec.block_size
        for lc in promoted_specs.values()
        if lc.manager_cls.is_full_attention
    }
    has_local_attention = any(_is_local_attention(lc) for lc in promoted_specs.values())
    full_attention_block_size = (
        next(iter(full_block_sizes)) if len(full_block_sizes) == 1 else None
    )

    def promoted_page_size_padded(spec: AttentionSpec, block_size: int) -> int | None:
        if spec.page_size_padded is None:
            return None
        unpadded_page_size = (
            spec.unpadded_page_size_bytes * block_size // spec.block_size
        )
        return max(spec.page_size_padded, unpadded_page_size)

    if full_block_sizes and has_local_attention:
        for layer_name, layer_config in layer_configs.items():
            if not _is_local_attention(layer_config):
                continue
            spec = layer_config.spec
            assert isinstance(spec, AttentionSpec)
            block_size = full_attention_block_size or spec.block_size
            # Promoted layers allocate blocks for all tokens and never free
            # below the window.
            promoted_specs[layer_name] = FullAttentionManager.layer_config(
                replace(
                    spec,
                    block_size=block_size,
                    page_size_padded=promoted_page_size_padded(spec, block_size),
                )
            )

    if not can_share_block_table(promoted_specs):
        raise ValueError("Failed to promote local KV cache specs to one unified type.")

    return promoted_specs


def _try_get_full_allocation_fallback_groups(
    layer_configs: dict[str, KVCacheLayerConfig],
) -> list[KVCacheGroupSpec] | None:
    """Try a supported full-allocation fallback for local-attention layers."""
    if any(isinstance(lc.spec, HiddenStateCacheSpec) for lc in layer_configs.values()):
        return None
    has_mla = False
    has_regular_swa = False
    for lc in layer_configs.values():
        is_mla = isinstance(lc.spec, MLAAttentionSpec)
        is_swa = issubclass(lc.manager_cls, SlidingWindowManager)
        if issubclass(lc.manager_cls, ChunkedLocalAttentionManager) or (
            is_swa and is_mla
        ):
            return None
        has_mla |= is_mla
        has_regular_swa |= is_swa
    if not (has_mla and has_regular_swa):
        return None

    try:
        promoted_specs = _promote_local_kv_cache_specs(layer_configs)
    except ValueError:
        return None
    if not can_share_block_table(promoted_specs):
        return None
    logger.warning(
        "KV cache page sizes cannot be unified; treating sliding-window "
        "layers as full attention for cache allocation. Sliding-window "
        "attention compute is unchanged."
    )
    return [KVCacheGroupSpec.from_layers(promoted_specs)]


def unify_hybrid_kv_cache_specs(layer_configs: dict[str, KVCacheLayerConfig]):
    """This function tries to convert the KV cache layers to one policy if the
    model is a hybrid model with multiple types of KV cache. It gives all
    local-attention layers the full attention policy if both are present.

    Args:
        layer_configs: The KV cache config of each attention layer in the model

    """
    groups: defaultdict[bool, dict[str, KVCacheLayerConfig]] = defaultdict(dict)
    for name, layer_config in layer_configs.items():
        spec = layer_config.spec
        replicated = isinstance(spec, AttentionSpec) and not spec.dcp_sharded
        groups[replicated][name] = layer_config
    for specs in groups.values():
        promoted_specs = _promote_local_kv_cache_specs(specs)
        if promoted_specs == specs:
            continue
        logger.warning(
            "Hybrid KV cache manager is disabled for this hybrid model, "
            "This means we do not enable any optimizations for saving KV cache "
            "memory (e.g., dropping the KV cache outside the sliding window). "
            "The compute of layers like sliding window is still saved."
        )
        layer_configs.update(promoted_specs)


def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")

    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d

    best_d = min_d
    best_pad: int | None = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d

    return best_d


def _get_packed_kv_cache_groups(
    vllm_config: VllmConfig,
    layer_configs: dict[str, KVCacheLayerConfig],
) -> list[KVCacheGroupSpec] | None:
    """Group mixed-page-size layers for contiguous block-outermost packing.

    Greedily buckets layers that can share a block table. Buckets with equal layer
    counts per page size are treated as a repeating layer pattern (one layer
    per page size) and split into groups covering the same number of pattern
    repeats (picked by ``_approximate_gcd`` to minimize padding), so all
    groups pack into the same per-block layout. Mamba buckets are additionally
    split to fit the block the attention buckets already need.
    Returns None when the layout is not block-outermost or all layers already
    share one page size.
    """
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    page_sizes = {lc.spec.page_size_bytes for lc in layer_configs.values()}
    if not layout.is_block_outermost or len(page_sizes) <= 1:
        return None

    bucketed = []
    for bucket in _bucket_layers(layer_configs):
        bucket_group = KVCacheGroupSpec.from_layers(bucket)
        page_size_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, layer_config in bucket.items():
            page_size_layers[layer_config.spec.page_size_bytes].append(layer_name)
        # Only 1:1 patterns (one layer of each page size per repeat) are
        # supported; counts sharing a gcd > 1 (e.g. 2:1) could in principle
        # repeat too, but such buckets are emitted whole instead.
        balanced = len(set(map(len, page_size_layers.values()))) == 1
        bucketed.append((bucket_group, page_size_layers, balanced))

    def max_layers_per_page_size(group: KVCacheGroupSpec) -> int:
        """Repeats of the layer pattern, for a balanced bucket."""
        pages = Counter(spec.page_size_bytes for spec in group.kv_cache_specs.values())
        return pages.most_common(1)[0][1]

    # Balanced buckets that mix page sizes must stay whole, so the largest one
    # sets a floor on the repeats per group; larger single-size buckets are
    # split down toward it. No such bucket means nothing needs packing.
    min_repeats_per_group = max(
        (
            max_layers_per_page_size(spec)
            for spec, page_size_layers, balanced in bucketed
            if balanced and len(page_size_layers) > 1
        ),
        default=0,
    )
    repeats_per_group = (
        _approximate_gcd(
            [
                max_layers_per_page_size(spec)
                for spec, _, balanced in bucketed
                if balanced
            ],
            lower_bound=min_repeats_per_group,
        )
        if min_repeats_per_group
        else None
    )

    def num_groups_for(spec: KVCacheGroupSpec, balanced: bool) -> int:
        if balanced and repeats_per_group is not None:
            return cdiv(max_layers_per_page_size(spec), repeats_per_group)
        return 1

    def widest_group_bytes(page_size_layers: dict[int, list[str]], n: int) -> int:
        """Page bytes of the largest of the n groups a bucket splits into."""
        return sum(
            cdiv(len(names), n) * page for page, names in page_size_layers.items()
        )

    # Bytes a block must hold however the state buckets end up split: mamba,
    # circular-buffer and unsplit sliding-window buckets can go down to one
    # state per group, every other bucket's split is fixed by the repeat pattern.
    def is_state_bucket(spec: KVCacheGroupSpec) -> bool:
        if issubclass(spec.manager_cls, (MambaManager, CircularBufferManager)):
            return True
        return repeats_per_group is None and issubclass(
            spec.manager_cls, SlidingWindowManager
        )

    anchor_bytes = max(
        (
            widest_group_bytes(
                page_size_layers,
                len(spec.kv_cache_specs)
                if is_state_bucket(spec)
                else num_groups_for(spec, balanced),
            )
            for spec, page_size_layers, balanced in bucketed
        ),
        default=0,
    )

    groups = []
    for spec, page_size_layers, balanced in bucketed:
        num_groups = num_groups_for(spec, balanced)
        # Cap a state group at the states a block already fits rather than let
        # it widen the block.
        if anchor_bytes and is_state_bucket(spec):
            state_page = next(iter(spec.kv_cache_specs.values())).page_size_bytes
            states_per_block = max(anchor_bytes // state_page, 1)
            num_groups = max(
                num_groups, cdiv(len(spec.kv_cache_specs), states_per_block)
            )
        if num_groups == 1:
            groups.append(spec)
            continue

        pattern_repeats = list(zip(*page_size_layers.values()))
        for i in range(num_groups):
            group_layer_names = [
                name for repeat in pattern_repeats[i::num_groups] for name in repeat
            ]
            groups.append(
                KVCacheGroupSpec.from_layers(
                    {name: layer_configs[name] for name in group_layer_names}
                )
            )

    return groups


def _warn_if_unannotated_eagle_mamba(
    spec_config: SpeculativeConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> None:
    """Warn when no KV cache group could be identified as the draft model's.

    Args:
        spec_config: Config supplying the speculative method.
        kv_cache_groups: Groups as they will be handed to consumers.

    """
    if any(group.is_eagle_group for group in kv_cache_groups):
        return
    mamba_groups = [
        idx
        for idx, group in enumerate(kv_cache_groups)
        if issubclass(group.manager_cls, MambaManager)
    ]
    if not mamba_groups:
        return
    logger.warning(
        "Speculative decoding (method=%s) is enabled but no KV cache group "
        "could be identified as the draft model's.",
        spec_config.method,
    )


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _ensure_min_page_size(
    groups: list[KVCacheGroupSpec],
    common_page: int,
    hidden_specs: dict[str, HiddenStateCacheSpec],
) -> tuple[list[KVCacheGroupSpec], int]:
    """Scale up group block sizes so the common page is at least as large as
    the biggest hidden-state per-token cost.

    This protects the hidden state extraction feature, where we store hidden states in
    the KV cache, and return them to clients using a custom hidden state connector.

    Returns the (possibly scaled) groups and updated common page size.
    """
    max_per_token = max(
        spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
        for spec in hidden_specs.values()
    )
    if max_per_token <= common_page:
        return groups, common_page

    scale = cdiv(max_per_token, common_page)
    common_page *= scale
    scaled: list[KVCacheGroupSpec] = []

    def scale_spec(s: KVCacheSpec) -> KVCacheSpec:
        kw: dict[str, Any] = {"block_size": s.block_size * scale}
        if isinstance(s, (AttentionSpec, MambaSpec)) and s.page_size_padded is not None:
            kw["page_size_padded"] = s.page_size_padded * scale
        return replace(s, **kw)

    for g in groups:
        scaled.append(
            replace(
                g,
                kv_cache_specs={n: scale_spec(s) for n, s in g.kv_cache_specs.items()},
                block_size=g.block_size * scale,
            )
        )
    return scaled, common_page


def get_kv_cache_groups(
    vllm_config: VllmConfig,
    layer_configs: dict[str, KVCacheLayerConfig],
) -> list[KVCacheGroupSpec]:
    """Split the layers in the model into groups that share a block table.

    Args:
        vllm_config: The global VllmConfig
        layer_configs: The KV cache config of each attention layer in the model

    Returns:
        The generated KVCacheGroups

    """
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        unify_hybrid_kv_cache_specs(layer_configs)

    if is_kv_cache_type_attention_free(layer_configs):
        # This returns an empty list to allow for the KVCacheManager to handle
        # attention free models.
        return []

    if hisparse_groups := get_hisparse_kv_cache_groups(vllm_config, layer_configs):
        return hisparse_groups

    if can_share_block_table(layer_configs):
        # All layers share a manager policy (e.g., all layers are full
        # attention, or all layers are sliding window attention with the same
        # window size), which is true for most models. Put all layers into one
        # group.
        return [KVCacheGroupSpec.from_layers(layer_configs)]
    # Hidden-state layers use their own block table and must not be absorbed
    # into a compatible attention bucket.
    hidden_specs = {
        k: v.spec
        for k, v in layer_configs.items()
        if isinstance(v.spec, HiddenStateCacheSpec)
    }
    filtered_spec = {
        k: v
        for k, v in layer_configs.items()
        if not isinstance(v.spec, HiddenStateCacheSpec)
    }

    if packed_groups := _get_packed_kv_cache_groups(vllm_config, filtered_spec):
        # Block-outermost blocks are strided by the widest group, so hidden
        # groups need no page alignment.
        packed_groups += [
            KVCacheGroupSpec.from_layers({name: layer_configs[name]})
            for name in hidden_specs
        ]
        return packed_groups

    # Prefer preserving each layer's cache semantics. If physical pages cannot
    # be unified, try a supported allocation-only fallback before failing.
    try:
        filtered_spec = unify_kv_cache_spec_page_size(filtered_spec)
    except NotImplementedError:
        fallback_groups = _try_get_full_allocation_fallback_groups(layer_configs)
        if fallback_groups is None:
            raise
        return fallback_groups
    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec)

    # Add hidden-state layers back with page aligned to the common page.
    if hidden_specs:
        common_page = get_uniform_page_size(
            [spec for g in groups for spec in g.kv_cache_specs.values()]
        )
        # TP may shrink the common page below hidden-state per-token cost.
        groups, common_page = _ensure_min_page_size(groups, common_page, hidden_specs)
        group_block_size = math.gcd(*(g.block_size for g in groups))
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            max_block_size = max(common_page // per_token, 1)
            new_bs = _largest_divisor_at_most(group_block_size, max_block_size)
            wasted_bytes = common_page - new_bs * per_token
            logger.info(
                "Using block size %d for hidden-state cache layer %s; "
                "page alignment wastes %d bytes (%.2f%%) per block",
                new_bs,
                name,
                wasted_bytes,
                wasted_bytes / common_page * 100,
            )
            aligned = replace(spec, block_size=new_bs, page_size_padded=common_page)
            hidden = replace(layer_configs[name], spec=aligned)
            groups.append(KVCacheGroupSpec.from_layers({name: hidden}))

    return groups


def get_kv_cache_groups_from_builder(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    layer_configs: dict[str, KVCacheLayerConfig],
) -> list[KVCacheGroupSpec]:
    """Group the layers with the builder and check the result.

    Core checks every grouping, whether it comes from the default grouping, a
    model's ``SupportsKVCacheGrouping`` or a platform builder.
    """
    groups = builder.get_kv_cache_groups(vllm_config, layer_configs)
    _check_exact_partition(groups, layer_configs)
    return groups


def _check_exact_partition(
    groups: list[KVCacheGroupSpec],
    layer_configs: dict[str, KVCacheLayerConfig],
) -> None:
    """Ensure the groups partition every input layer exactly once."""
    grouped_layers: set[str] = set()
    for group in groups:
        assert not set(group.layer_names) & grouped_layers, (
            f"KV cache group overlaps: {set(group.layer_names) & grouped_layers}."
        )
        grouped_layers |= set(group.layer_names)
    missing = set(layer_configs) - grouped_layers
    assert not missing, f"KV cache groups miss layers: {sorted(missing)}."


def _layer_tp_replicas(spec: KVCacheSpec, tp_size: int, dcp_size: int) -> int:
    if not isinstance(spec, AttentionSpec) or spec.max_tp_shards is None:
        return 1
    if spec.dcp_sharded and dcp_size > 1:
        return 1
    return max(1, tp_size // spec.max_tp_shards)


def kv_cache_groups_tp_replicas(
    groups: list[KVCacheGroupSpec], tp_size: int, dcp_size: int = 1
) -> int:
    """Consecutive TP ranks holding identical KV for every layer."""
    specs = [spec for g in groups for spec in g.kv_cache_specs.values()]
    if not specs:
        return 1
    return reduce(
        math.gcd, (_layer_tp_replicas(s, tp_size, dcp_size) for s in specs), tp_size
    )


def _max_memory_usage_bytes_from_groups(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Calculate maximum memory usage in bytes from KV cache groups.

    This correctly accounts for padding in hybrid models. For example, if a
    model has 8 full attention layers and 9 sliding window layers, they will
    be padded to 9 full + 9 sliding window for uniform group sizes.

    Each group independently claims blocks from the shared pool, so a request consumes
    the sum of the per-group block counts, i.e. ``bytes_per_block * total_blocks``.
    """
    kv_cache_groups = [group for group in kv_cache_groups if not group.host_resident]
    if not kv_cache_groups:
        return 0

    bytes_per_block = _pool_bytes_per_block_from_builder(
        builder, vllm_config, kv_cache_groups
    )
    total_blocks = sum(
        group.max_blocks_per_request(vllm_config) for group in kv_cache_groups
    )

    return bytes_per_block * total_blocks


def _estimate_max_model_len_from_groups(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> int:
    """Binary search for the maximum model length that fits in available memory.
    Returns 0 if even 1 token doesn't fit.
    """
    original_max = vllm_config.model_config.max_model_len
    hisparse_enabled = (
        vllm_config.attention_config.hisparse_config is not None
        and bool(kv_cache_groups)
    )

    def fits(model_len: int) -> bool:
        vllm_config.model_config.max_model_len = model_len
        if hisparse_enabled:
            try:
                num_blocks = available_memory // _pool_bytes_per_block_from_builder(
                    builder, vllm_config, kv_cache_groups
                )
                builder.get_kv_cache_config_from_groups(
                    vllm_config, kv_cache_groups, num_blocks
                )
            except ValueError:
                return False
        return (
            _max_memory_usage_bytes_from_groups(builder, vllm_config, kv_cache_groups)
            <= available_memory
        )

    try:
        left, right = 1, original_max
        if not fits(left):
            return 0
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        vllm_config.model_config.max_model_len = original_max


def _auto_fit_max_model_len(
    builder: KVCacheConfigBuilder,
    vllm_config: VllmConfig,
    projected_groups_per_worker: list[list[KVCacheGroupSpec]],
    available_memory: list[int],
) -> None:
    """When max_model_len is set to -1, this function estimates the largest
    context length that can be supported with the available GPU memory.
    It uses binary search to find the maximum length that fits across all
    workers.

    Args:
        builder: The builder whose hooks drive block accounting.
        vllm_config: The global VllmConfig (will be modified in-place)
        projected_groups_per_worker: KV cache groups projected to each worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    """
    original_max = vllm_config.model_config.max_model_len

    if all(not groups for groups in projected_groups_per_worker):
        # All workers have empty specs (attention-free model)
        logger.info_once(
            "Auto-fit max_model_len: attention-free model, "
            "using derived max_model_len=%d",
            original_max,
        )
        return

    # Find the max_model_len that fits across all workers.
    auto_fit_max = original_max
    limiting_worker_mem = available_memory[0]
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            continue
        worker_max = _estimate_max_model_len_from_groups(
            builder, vllm_config, groups, avail_mem
        )
        if worker_max < auto_fit_max:
            auto_fit_max = worker_max
            limiting_worker_mem = avail_mem

    if auto_fit_max <= 0:
        raise ValueError(
            "Cannot auto-fit max_model_len: not enough GPU memory available "
            "to serve even a single token. Try increasing `gpu_memory_utilization`."
        )

    kv_cache_memory = (
        f"{format_gib(limiting_worker_mem)} GiB {_kv_cache_device_name(vllm_config)}"
    )
    if vllm_config.attention_config.hisparse_config is not None:
        host_pool_gib = format_gib(get_hisparse_host_pool_bytes(vllm_config))
        kv_cache_memory += f" + {host_pool_gib} GiB host"

    if auto_fit_max >= original_max:
        # The model's full context length fits in memory
        logger.info_once(
            "Auto-fit max_model_len: full model context length %d fits in "
            "available KV cache memory (%s)",
            original_max,
            kv_cache_memory,
        )
    else:
        # Need to reduce max_model_len to fit in memory
        vllm_config.model_config.max_model_len = auto_fit_max
        logger.info_once(
            "Auto-fit max_model_len: reduced from %d to %d to fit in "
            "available KV cache memory (%s)",
            original_max,
            auto_fit_max,
            kv_cache_memory,
        )


def _project_kv_cache_groups_to_worker(
    global_kv_cache_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheLayerConfig],
) -> list[KVCacheGroupSpec]:
    """Projects global KV cache groups onto a single worker's assigned layers.

    In pipeline parallelism, each worker only owns a subset of layers. This
    function filters the global groups to include only layers present on the
    given worker. The groups keep their block table properties even when they
    hold no layer of this worker.

    Args:
        global_kv_cache_groups: The global KV cache groups for the whole model.
        worker_spec: The KV cache spec of each layer on this worker.

    Returns:
        The projected KV cache groups containing only this worker's layers.

    """
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_kv_cache_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        if len(worker_layer_names) == len(group.layer_names):
            projected_groups.append(group)
            continue
        projected_groups.append(
            replace(
                group,
                kv_cache_specs={
                    name: group.kv_cache_specs[name] for name in worker_layer_names
                },
            )
        )
    return projected_groups


def get_kv_cache_configs(
    vllm_config: VllmConfig,
    layer_configs: list[dict[str, KVCacheLayerConfig]],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Generates the KV cache configurations for a model.
    Since we use a shared centralized controller for all workers, we need the
    `kv_cache_config` to be consistent across all workers to make sure
    the KV cache allocation can be applied to all workers. However, different
    workers may have different memory available, and different type of layers
    (when pipeline parallel is enabled). To handle the difference between
    workers, the current implementation is:
    1. Merge the layer configs of all workers to get the layers of the whole
       model.
    2. Generate the KV cache groups based on the layer ratio of the whole model.
       This also handles spec unification for hybrid models.
    3. Handle auto-fit max_model_len and memory checks using per-worker
       projected groups to account for PP sharding.
    4. Generate the KV cache configs for each worker based on the KV cache
       grouping strategy. (This is reasonable because the layer ratio of
       different PP stages are similar.)
    5. Change the num_blocks of each worker to the smallest among all workers
       and shrink tensor sizes proportionally to avoid allocating unused memory.

    Args:
        vllm_config: The global VllmConfig
        layer_configs: List of dict[layer_name, KVCacheLayerConfig] for each
            worker.
        available_memory: Memory available for KV cache in bytes for each
            worker.

    Returns:
        The generated KVCacheConfigs for each worker.

    """
    from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder

    builder = get_kv_cache_config_builder(vllm_config)

    # Merge the layer configs of all workers. Different PP stages may have
    # different layer names, and different TP ranks of the same PP stage should
    # have the same layer config.
    merged_layer_configs: dict[str, KVCacheLayerConfig] = {}
    for layer_configs_one_worker in layer_configs:
        for layer_name, layer_config in layer_configs_one_worker.items():
            if layer_name not in merged_layer_configs:
                merged_layer_configs[layer_name] = layer_config
            else:
                assert merged_layer_configs[layer_name] == layer_config, (
                    "The KV cache configs for the same layer are different "
                    "across workers. This is not supported yet."
                )

    # When speculating with more than 1 speculative module (e.g. multi-layered MTP)
    # tag every sliding window with how many extra tokens to retain in the window.
    extra_retained_tokens = max(0, vllm_config.num_prefill_lookahead_tokens - 1)
    for layer_name, layer_config in merged_layer_configs.items():
        config = layer_config.manager_config
        if isinstance(config, SlidingWindowManager.Config):
            merged_layer_configs[layer_name] = replace(
                layer_config,
                manager_config=replace(
                    config, extra_retained_tokens=extra_retained_tokens
                ),
            )

    # Get global KV cache groups. This also handles policy unification for
    # hybrid models when disable_hybrid_kv_cache_manager is enabled.
    # After this call, merged_layer_configs may be modified in-place.
    global_kv_cache_groups = get_kv_cache_groups_from_builder(
        builder, vllm_config, merged_layer_configs
    )
    spec_config = vllm_config.speculative_config
    if spec_config is not None and spec_config.use_eagle_block_drop():
        draft_layer_names = {
            name for name, config in merged_layer_configs.items() if config.is_draft
        }
        for group in global_kv_cache_groups:
            group.is_eagle_group = not draft_layer_names.isdisjoint(group.layer_names)
        _warn_if_unannotated_eagle_mamba(spec_config, global_kv_cache_groups)

    # If original_max_model_len was -1, automatically
    # determine the maximum model length that fits in available GPU memory.
    # We use per-worker projected groups to account for PP sharding.
    projected_groups_per_worker = [
        _project_kv_cache_groups_to_worker(global_kv_cache_groups, worker_spec)
        for worker_spec in layer_configs
    ]

    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`. Reflect
    # that in `available_memory` here so auto-fit, the admission check, and
    # the per-worker config builder all plan against the same effective
    # capacity.
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block_from_builder(
                builder, vllm_config, groups
            )
            logger.info(
                "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
                avail_mem // bytes_per_block,
                override,
            )
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory

    # Reserve the null block BlockPool permanently holds back, so auto-fit and
    # the capacity check both plan against usable blocks. Allocation below
    # still uses the full memory.
    check_memory = [
        (
            avail_mem - _pool_bytes_per_block_from_builder(builder, vllm_config, groups)
            if groups
            else avail_mem
        )
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory)
    ]

    if vllm_config.model_config.original_max_model_len == -1:
        _auto_fit_max_model_len(
            builder, vllm_config, projected_groups_per_worker, check_memory
        )

    # Check if the available memory is enough per worker.
    for groups, avail_mem in zip(projected_groups_per_worker, check_memory):
        if not groups:
            continue
        _check_enough_kv_cache_memory(
            avail_mem,
            partial(_max_memory_usage_bytes_from_groups, builder, vllm_config, groups),
            vllm_config.model_config.max_model_len,
            partial(_estimate_max_model_len_from_groups, builder, vllm_config, groups),
        )

    kv_cache_configs: list[KVCacheConfig] = []
    for projected_groups, layer_configs_one_worker, available_memory_one_worker in zip(
        projected_groups_per_worker, layer_configs, available_memory
    ):
        assert sum(len(group.layer_names) for group in projected_groups) == len(
            layer_configs_one_worker
        ), "Some layers are not assigned to any group."
        bytes_per_block = _pool_bytes_per_block_from_builder(
            builder, vllm_config, projected_groups
        )
        num_blocks = (
            available_memory_one_worker // bytes_per_block if projected_groups else 1
        )
        kv_cache_configs.append(
            builder.get_kv_cache_config_from_groups(
                vllm_config, projected_groups, num_blocks
            )
        )

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for i, kv_cache_config in enumerate(kv_cache_configs):
        if kv_cache_config.num_blocks == min_num_blocks:
            continue
        # Re-plan with exactly the smallest rank's block count, so strides and
        # offsets stay consistent with the shrunken allocation.
        kv_cache_configs[i] = builder.get_kv_cache_config_from_groups(
            vllm_config,
            kv_cache_config.kv_cache_groups,
            min_num_blocks,
        )

    for kv_cache_config in kv_cache_configs:
        kv_cache_config.kv_tp_replicas = kv_cache_groups_tp_replicas(
            kv_cache_config.kv_cache_groups,
            vllm_config.parallel_config.tensor_parallel_size,
            vllm_config.parallel_config.decode_context_parallel_size,
        )

    return kv_cache_configs


class DefaultKVCacheConfigBuilder(KVCacheConfigBuilder):
    """Default grouping and placement, backed by the module-level functions here.

    .. note::
        This class is part of the experimental ``KVCacheConfigBuilder``
        interface and is subject to change.

    Subclasses override only the hooks they need. The planning flow and its
    invariants (``num_gpu_blocks_override``, null-block reservation, auto-fit,
    admission checks, cross-rank convergence) stay in the Core-owned
    :func:`get_kv_cache_configs`.
    """

    def get_kv_cache_groups(
        self,
        vllm_config: VllmConfig,
        layer_configs: dict[str, KVCacheLayerConfig],
    ) -> list[KVCacheGroupSpec]:
        """See ``KVCacheConfigBuilder.get_kv_cache_groups``.

        A model implementing ``SupportsKVCacheGrouping`` owns the grouping
        decision; otherwise the default grouping is used.
        """
        if not vllm_config.model_config.supports_kv_cache_grouping:
            return get_kv_cache_groups(vllm_config, layer_configs)
        if vllm_config.attention_config.hisparse_config is not None:
            raise ValueError(
                "A model-declared get_kv_cache_groups cannot be combined "
                "with HiSparse, which owns its own KV cache grouping. "
                "Remove one of them."
            )
        model_cls, _ = vllm_config.model_config.registry.resolve_model_cls(
            vllm_config.model_config.architectures,
            vllm_config.model_config,
        )
        return model_cls.get_kv_cache_groups(vllm_config, layer_configs)

    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
    ) -> KVCacheConfig:
        """See ``KVCacheConfigBuilder.get_kv_cache_config_from_groups``."""
        if len(kv_cache_groups) == 0:
            # Attention free models do not have KV cache.
            # Return num_blocks=1 as BlockPool always needs a null_block.
            return KVCacheConfig(
                num_blocks=1,
                kv_cache_tensors=[],
                kv_cache_groups=kv_cache_groups,
                prefix_cache_retention_interval=(
                    vllm_config.cache_config.prefix_cache_retention_interval
                ),
            )
        bytes_per_block = _pool_bytes_per_block(vllm_config, kv_cache_groups)
        return get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, num_blocks * bytes_per_block
        )
