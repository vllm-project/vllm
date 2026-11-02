# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated movable-layer cache pages with scheduler block IDs unchanged."""

from collections.abc import Mapping
from typing import Any

import torch

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
    create_kv_cache_views,
)
from vllm.v1.kv_cache_layout import KVCacheLayout


def isolate_pp_cache(
    config: KVCacheConfig,
    caches: dict[str, torch.Tensor],
    components: Mapping[str, tuple[int, str]],
    layout: KVCacheLayout,
    kernel_block_sizes: list[int],
    device: torch.device,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, dict[str, Any]]]:
    """Allocate before binding or inference; never clone a shared pool per layer.

    Each component gets complete, padded pages for every scheduler block ID,
    including cached blocks with no live request. Other layers retain their
    original pooled storage. Allocation failures leave the input views intact.
    """
    if not components:
        raise ValueError("Dynamic PP requires movable cache components")
    plans = []
    identities = set()
    for name, (layer, key) in sorted(components.items()):
        if name not in caches or (layer, key) in identities:
            raise ValueError(f"Missing or duplicate dynamic PP cache component {name}")
        identities.add((layer, key))
        matches = [
            (i, group)
            for i, group in enumerate(config.kv_cache_groups)
            if name in group.layer_names
        ]
        if len(matches) != 1:
            raise ValueError(f"Dynamic PP cache {name} needs one scheduler group")
        group_id, group = matches[0]
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[name]
        if group.host_resident or not spec.has_layer_views or spec.num_heads != 1:
            raise ValueError("Dynamic PP requires device cache pages with layer views")
        size = config.num_blocks * spec.page_size_bytes
        tensor = KVCacheTensor(
            size=size,
            layers=[name],
            layer_stride=size,
            block_stride=spec.page_size_bytes,
        )
        kernel_size = kernel_block_sizes[group_id]
        description = (
            group_id,
            repr(spec),
            config.num_blocks,
            kernel_size,
            layout.name,
            config.hash_block_size,
            config.cache_hit_alignment_tokens,
            config.prefix_cache_retention_interval,
            group.is_eagle_group,
        )
        plans.append((name, layer, key, spec, tensor, kernel_size, description))

    required = sum(plan[4].size for plan in plans)
    if device.type == "cuda":
        free, _ = torch.accelerator.get_memory_info(device)
        if free < required + (1 << 30):
            raise ValueError(
                f"Dynamic PP needs {required} extra cache bytes plus 1 GiB headroom"
            )
    state: dict[int, dict[str, torch.Tensor]] = {}
    descriptions: dict[int, dict[str, Any]] = {}
    views = {}
    for name, layer, key, spec, tensor, kernel_size, description in plans:
        raw = torch.zeros(tensor.size, dtype=torch.int8, device=device)
        view = create_kv_cache_views(
            raw,
            spec,
            config.num_blocks,
            layout,
            tensor,
            kernel_block_size=kernel_size,
        )[0]
        if view.shape != caches[name].shape or view.dtype != caches[name].dtype:
            raise ValueError(f"Dynamic PP cache view changed shape or dtype: {name}")
        views[name] = view
        state.setdefault(layer, {})[key] = raw
        descriptions.setdefault(layer, {})[key] = (
            description,
            tuple(view.shape),
            tuple(view.stride()),
            str(view.dtype),
        )
    caches.update(views)
    return state, descriptions


def prepare_pp_cache(
    config: KVCacheConfig,
    caches: dict[str, torch.Tensor],
    forward_context: Mapping[str, Any],
    layout: KVCacheLayout,
    kernel_block_sizes: list[int],
    device: torch.device,
) -> None:
    components = {
        name: layer.dynamic_pp_cache_component
        for name, layer in forward_context.items()
        if hasattr(layer, "dynamic_pp_cache_component")
    }
    state, descriptions = isolate_pp_cache(
        config, caches, components, layout, kernel_block_sizes, device
    )
    for name, (layer, key) in components.items():
        forward_context[name].dynamic_pp_cache_raw = state[layer][key]
        forward_context[name].dynamic_pp_cache_layout = descriptions[layer][key]
