# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local KV storage placement, independent of logical block ownership."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class KVCacheBundle:
    """Cache components consumed together, with one persistent owner.

    ``owner=None`` keeps an independent persistent copy on every worker.
    Bundle order is execution order, supplied by the loaded model.
    """

    layers: tuple[str, ...]
    owner: int | None


@dataclass(frozen=True)
class KVCachePlacement:
    """Serializable placement inside one verified KV replica domain."""

    rank: int
    world_size: int
    bundles: tuple[KVCacheBundle, ...]
    alignment: int = 256
    scratch_slots: int = 2

    def __post_init__(self) -> None:
        if not self.bundles:
            raise ValueError("KV placement requires at least one cache bundle.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid KV cache placement rank.")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("KV cache alignment must be a positive power of two.")
        if self.scratch_slots < 2:
            raise ValueError("Layer prefetch requires at least two scratch slots.")
        names = [name for bundle in self.bundles for name in bundle.layers]
        if len(names) != len(set(names)) or any(not b.layers for b in self.bundles):
            raise ValueError("Every KV cache component must occur in one bundle.")
        if any(
            b.owner is not None and not 0 <= b.owner < self.world_size
            for b in self.bundles
        ):
            raise ValueError("KV cache owner is outside its replica domain.")


@dataclass(frozen=True)
class KVCacheBundleRegion:
    bundle: KVCacheBundle
    offset: int
    size: int
    scratch_slot: int | None


@dataclass(frozen=True)
class KVCacheStoragePlan:
    placement: KVCachePlacement
    regions: tuple[KVCacheBundleRegion, ...]
    backing_size: int

    @property
    def allocation_bytes(self) -> int:
        # The allocator may need to advance to the requested base alignment.
        return self.backing_size + self.placement.alignment - 1

    @property
    def persistent_layers(self) -> tuple[str, ...]:
        return tuple(
            name
            for region in self.regions
            if region.scratch_slot is None
            for name in region.bundle.layers
        )


def layer_specs(groups: list[KVCacheGroupSpec]) -> dict[str, KVCacheSpec]:
    return {
        name: (
            group.kv_cache_spec.kv_cache_specs[name]
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else group.kv_cache_spec
        )
        for group in groups
        for name in group.layer_names
    }


def validate_kv_cache_placements(
    worker_specs: list[dict[str, KVCacheSpec]],
    placements: list[KVCachePlacement],
) -> None:
    """Validate collective participants before initializing any transport."""
    if len(worker_specs) != len(placements):
        raise ValueError("Every worker must provide a KV placement.")
    replicas: dict[tuple[str, ...], list[int]] = {}
    for index, (specs, placement) in enumerate(zip(worker_specs, placements)):
        names = {name for bundle in placement.bundles for name in bundle.layers}
        if names != set(specs):
            raise ValueError("KV placement does not cover its worker specs.")
        replicas.setdefault(tuple(sorted(specs)), []).append(index)
    for indices in replicas.values():
        first = placements[indices[0]]
        if sorted(placements[i].rank for i in indices) != list(range(first.world_size)):
            raise ValueError("KV replica domain is incomplete or has duplicate ranks.")
        for i in indices:
            if replace(placements[i], rank=first.rank) != first:
                raise ValueError(
                    "KV replica ranks disagree on bundle ownership/layout."
                )
            if worker_specs[i] != worker_specs[indices[0]]:
                raise ValueError("KV replica ranks have different logical cache specs.")


def build_kv_cache_storage(
    config: KVCacheConfig,
    placement: KVCachePlacement,
    layout: KVCacheLayout,
) -> KVCacheConfig:
    """Materialize stable owner and alternating scratch views for final blocks.

    This function also defines the exact footprint used for capacity planning.
    It does not change logical specs, groups, or block numbering.
    """
    if not layout.is_layer_compact:
        raise ValueError("Layer-sharded KV requires a layer-compact layout.")
    if config.num_blocks < 1:
        raise ValueError("KV storage must include at least the null block.")
    specs = layer_specs(config.kv_cache_groups)
    names = {name for bundle in placement.bundles for name in bundle.layers}
    if names != set(specs):
        raise ValueError("KV placement must cover the complete worker cache spec.")
    if any(group.host_resident for group in config.kv_cache_groups):
        raise ValueError("Layer-sharded KV does not support host-resident groups.")

    alignment = placement.alignment
    component_offsets: list[dict[str, int]] = []
    bundle_sizes = []
    for bundle in placement.bundles:
        offset = 0
        offsets = {}
        for name in bundle.layers:
            spec = specs[name]
            if not spec.has_layer_views:
                raise ValueError(f"KV placement needs a layer view for {name}.")
            offset = round_up(offset, alignment)
            offsets[name] = offset
            offset += spec.page_size_bytes * config.num_blocks
        component_offsets.append(offsets)
        bundle_sizes.append(round_up(offset, alignment))

    scratch_size = max(
        (
            size
            for bundle, size in zip(placement.bundles, bundle_sizes)
            if bundle.owner is not None
        ),
        default=0,
    )
    cursor = scratch_size * placement.scratch_slots
    regions = []
    tensors = []
    target_index = 0
    for bundle, size, offsets in zip(
        placement.bundles, bundle_sizes, component_offsets
    ):
        if bundle.owner in (None, placement.rank):
            base = cursor
            cursor += size
            slot = None
        else:
            slot = target_index % placement.scratch_slots
            base = slot * scratch_size
        if bundle.owner is not None:
            target_index += 1
        regions.append(KVCacheBundleRegion(bundle, base, size, slot))
        for name in bundle.layers:
            layer_stride, block_stride, *_ = compute_layout_strides(
                specs[name], config.num_blocks, 1, layout
            )
            tensors.append(
                KVCacheTensor(
                    size=0,
                    layers=[name],
                    layer_stride=layer_stride,
                    block_stride=block_stride,
                    offset=base + offsets[name],
                )
            )
    for tensor in tensors:
        tensor.size = cursor
    plan = KVCacheStoragePlan(placement, tuple(regions), cursor)
    return replace(config, kv_cache_tensors=tensors, storage_plan=plan)


def fit_kv_cache_storage(
    config: KVCacheConfig,
    placement: KVCachePlacement,
    layout: KVCacheLayout,
    available_bytes: int,
) -> int:
    """Find the largest allocatable block count including alignment padding."""
    if available_bytes < 0:
        raise ValueError("Available KV memory must be nonnegative.")

    def fits(num_blocks: int) -> bool:
        candidate = build_kv_cache_storage(
            replace(config, num_blocks=num_blocks), placement, layout
        )
        assert candidate.storage_plan is not None
        return candidate.storage_plan.allocation_bytes <= available_bytes

    low, high = 0, 1
    while fits(high):
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def plan_layer_sharded_configs(
    vllm_config: "VllmConfig",
    worker_groups: list[list[KVCacheGroupSpec]],
    placements: list[KVCachePlacement],
    available_memory: list[int],
) -> list[KVCacheConfig]:
    """Plan common logical capacity against every worker's physical layout."""
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

    if not len(worker_groups) == len(placements) == len(available_memory):
        raise ValueError("Each worker must provide its KV placement and budget.")
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    configs = []
    capacities = []
    for groups, placement, available in zip(
        worker_groups, placements, available_memory
    ):
        specs = layer_specs(groups)
        if not specs or any(
            not isinstance(s, FullAttentionSpec) for s in specs.values()
        ):
            raise ValueError("KVPP currently requires full-attention cache specs.")
        if len({s.block_size for s in specs.values()}) != 1:
            raise ValueError("KVPP currently requires a common cache block size.")
        config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=groups,
            prefix_cache_retention_interval=(
                vllm_config.cache_config.prefix_cache_retention_interval
            ),
        )
        configs.append(config)
        capacities.append(fit_kv_cache_storage(config, placement, layout, available))
    num_blocks = min(capacities)
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        if override > num_blocks:
            raise ValueError(
                f"KVPP block override {override} exceeds physical capacity "
                f"{num_blocks}."
            )
        num_blocks = override
    if num_blocks < 2:
        raise ValueError(
            "KVPP memory budget cannot hold a usable block and null block."
        )
    configs = [
        build_kv_cache_storage(replace(c, num_blocks=num_blocks), p, layout)
        for c, p in zip(configs, placements)
    ]

    offload_block_size_bytes = max(
        sum(
            layer_specs(c.kv_cache_groups)[name].page_size_bytes
            for name in c.storage_plan.persistent_layers
        )
        for c in configs
        if c.storage_plan is not None
    )
    for config in configs:
        config.offload_block_size_bytes = offload_block_size_bytes

    def fits_length(length: int) -> bool:
        original = vllm_config.model_config.max_model_len
        try:
            vllm_config.model_config.max_model_len = length
            return all(
                get_max_concurrency_for_kv_cache_config(
                    vllm_config, replace(c, num_blocks=num_blocks - 1)
                )
                >= 1
                for c in configs
            )
        finally:
            vllm_config.model_config.max_model_len = original

    model = vllm_config.model_config
    if model.original_max_model_len == -1:
        low, high = 0, model.max_model_len + 1
        while low + 1 < high:
            middle = (low + high) // 2
            if fits_length(middle):
                low = middle
            else:
                high = middle
        if low == 0:
            raise ValueError("KVPP cannot fit even one token at this memory budget.")
        model.max_model_len = low
    elif not fits_length(model.max_model_len):
        raise ValueError(
            "KVPP physical capacity is insufficient for max_model_len; increase "
            "the KV memory budget or reduce max_model_len."
        )
    return configs
