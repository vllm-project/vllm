# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical-to-kernel block sizing resolved per KV cache group.

Each cache group indexes its own block table in its own block size, so the
number of physical kernel blocks backing one logical block is a per-group
quantity. The Mooncake P2P connector indexes block ids per transfer group and
therefore cannot describe that split with a single scalar.

The helpers here resolve the split the same way the caches were registered
(`vllm.v1.worker.utils.prepare_kernel_block_sizes`) but walk
`KVCacheConfig.transfer_groups`, so the result lines up with the block ids the
connector indexes.
"""

from collections.abc import Sequence
from typing import Any, cast

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig


def get_transfer_group_kernel_block_sizes(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> list[int]:
    """Kernel block size of every transfer group, in `transfer_groups` order.

    Mirrors `vllm.v1.worker.utils.prepare_kernel_block_sizes`, which decided the
    sizes the caches were registered with. Walking `transfer_groups` rather than
    `kv_cache_groups` keeps the result aligned with the block ids the connector
    indexes: transfer groups are a subset of cache groups, so their positions
    differ as soon as one group is not transferable.
    """
    # Imported lazily: `vllm.v1.worker.utils` pulls in the attention layer
    # stack, which imports the connector package back through
    # `v1/attention/backends/utils.py`.
    from vllm.v1.worker.utils import select_common_block_size

    layer_type = cast(type[Any], AttentionLayerBase)
    kernel_block_sizes: list[int] = []
    for group in kv_cache_config.transfer_groups:
        if not group.kv_cache_specs:
            # A pipeline stage that owns none of the group's layers.
            kernel_block_sizes.append(group.block_size)
            continue
        # All layers of a group hold the same kind of cache; pick any.
        spec = next(iter(group.kv_cache_specs.values()))
        if not isinstance(spec, AttentionSpec) or not spec.has_layer_views:
            # Mamba state and other caches are laid out by the spec itself.
            kernel_block_sizes.append(group.block_size)
            continue
        # Block-outermost packing puts other layers' pages between the blocks,
        # so they are never split group-wide.
        if any(
            tensor.block_stride > tensor.layer_stride
            for tensor in kv_cache_config.kv_cache_tensors
            if not set(tensor.layers).isdisjoint(group.layer_names)
        ):
            kernel_block_sizes.append(group.block_size)
            continue

        # One entry per attention group the model runner builds: layers sharing
        # a backend, a spec and a Q-head count share a kernel block size.
        pairs: dict[
            tuple[type[AttentionBackend], AttentionSpec | None, int], list[str]
        ] = {}
        layers = get_layers_from_vllm_config(vllm_config, layer_type, group.layer_names)
        for layer_name, layer in layers.items():
            layer_spec = group.kv_cache_specs[layer_name]
            key = (
                layer.get_attn_backend(),
                layer_spec if isinstance(layer_spec, AttentionSpec) else None,
                getattr(layer, "num_heads", 0),
            )
            pairs.setdefault(key, []).append(layer_name)
        if not pairs:
            # No layer object is registered for this group, e.g. before the
            # model is built or in unit tests. Fall back to the default backend
            # so the group still resolves to a supported kernel block size.
            fallback = get_current_attn_backends(vllm_config, group.layer_names)
            if not fallback:
                kernel_block_sizes.append(group.block_size)
                continue
            kernel_block_sizes.append(
                select_common_block_size(
                    group.block_size, fallback, [None] * len(fallback)
                )
            )
            continue
        kernel_block_sizes.append(
            select_common_block_size(
                group.block_size,
                [backend for backend, _, _ in pairs],
                [layer_spec for _, layer_spec, _ in pairs],
                [names for _, _, names in pairs],
            )
        )
    return kernel_block_sizes


def check_compressed_kernel_split_per_group(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kernel_block_sizes: Sequence[int],
) -> None:
    """A layer-outer layout cannot split compressed pages into kernel blocks.

    Applies the same judgement as
    `vllm.distributed.kv_transfer.kv_connector.utils.check_compressed_kernel_split`,
    but with each group's own kernel block size instead of one shared value.
    """
    if vllm_config.cache_config.get_resolved_kv_cache_layout().is_block_outermost:
        return
    groups = kv_cache_config.transfer_groups
    if len(kernel_block_sizes) != len(groups):
        raise ValueError(
            f"Expected one kernel block size per transfer group, got "
            f"{len(kernel_block_sizes)} for {len(groups)} groups."
        )
    for group, kernel_block_size in zip(groups, kernel_block_sizes, strict=True):
        for spec in group.kv_cache_specs.values():
            if (
                isinstance(spec, AttentionSpec)
                and spec.tokens_per_state > 1
                and spec.block_size % kernel_block_size == 0
                and spec.block_size != kernel_block_size
            ):
                raise NotImplementedError(
                    "Compressed KV caches cannot be split into kernel blocks "
                    "with a layer-outermost KV cache layout."
                )
