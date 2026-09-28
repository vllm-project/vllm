# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local KV storage placement, independent of logical block ownership."""

from dataclasses import dataclass, replace

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
    scratch_slots: int = 2

    def __post_init__(self) -> None:
        assert self.bundles, "KV placement requires at least one cache bundle."
        assert 0 <= self.rank < self.world_size, "Invalid KV cache placement rank."
        assert self.scratch_slots >= 2, (
            "Layer prefetch requires at least two scratch slots."
        )
        names = [name for bundle in self.bundles for name in bundle.layers]
        assert len(names) == len(set(names)) and all(b.layers for b in self.bundles), (
            "Every KV cache component must occur in one bundle."
        )
        assert all(
            b.owner is None or 0 <= b.owner < self.world_size for b in self.bundles
        ), "KV cache owner is outside its replica domain."


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
    assert names == set(specs), (
        "KV placement must cover the complete worker cache spec."
    )
    if any(group.host_resident for group in config.kv_cache_groups):
        raise ValueError("Layer-sharded KV does not support host-resident groups.")

    component_offsets: list[dict[str, int]] = []
    bundle_sizes = []
    for bundle in placement.bundles:
        offset = 0
        offsets = {}
        for name in bundle.layers:
            spec = specs[name]
            if not spec.has_layer_views:
                raise ValueError(f"KV placement needs a layer view for {name}.")
            offsets[name] = offset
            offset += spec.page_size_bytes * config.num_blocks
        component_offsets.append(offsets)
        bundle_sizes.append(offset)

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
    """Convert the packed bytes per logical block to an allocatable count."""
    if available_bytes < 0:
        raise ValueError("Available KV memory must be nonnegative.")

    single_block = build_kv_cache_storage(
        replace(config, num_blocks=1), placement, layout
    )
    assert single_block.storage_plan is not None
    return available_bytes // single_block.storage_plan.backing_size


def get_layer_sharded_capacity(
    groups: list[KVCacheGroupSpec],
    placement: KVCachePlacement,
    layout: KVCacheLayout,
    available_memory: int,
) -> int:
    """Convert a physical memory budget to the usual logical block capacity."""
    specs = layer_specs(groups)
    if not specs or any(not isinstance(s, FullAttentionSpec) for s in specs.values()):
        raise ValueError("KVPP currently requires full-attention cache specs.")
    if len({s.block_size for s in specs.values()}) != 1:
        raise ValueError("KVPP currently requires a common cache block size.")
    config = KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=groups)
    return fit_kv_cache_storage(config, placement, layout, available_memory)


def set_layer_sharded_offload_block_size(configs: list[KVCacheConfig]) -> None:
    """Use one persistent-block budget for offload across all KV ranks."""
    block_size_bytes = max(
        sum(
            layer_specs(config.kv_cache_groups)[name].page_size_bytes
            for name in config.storage_plan.persistent_layers
        )
        for config in configs
        if config.storage_plan is not None
    )
    for config in configs:
        config.offload_block_size_bytes = block_size_bytes
