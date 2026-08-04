# SPDX-License-Identifier: Apache-2.0
"""Build LMCache engine group infos from vLLM KV cache group metadata."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def _is_sliding_window_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM sliding-window spec.

    Checked by class name so this module stays importable without vLLM.
    Subclasses such as ``SlidingWindowMLASpec`` count.
    """
    return any(cls.__name__ == "SlidingWindowSpec" for cls in type(spec).__mro__)


def _resolve_per_layer_sw_sizes(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[int]:
    """Resolve the sliding window size in tokens for each registered KV tensor.

    Will resolve -1 for non-sliding-window layers.

    Args:
        vllm_groups: vLLM ``KVCacheGroupSpec`` instances.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length ``num_layers`` mapping each registered tensor index
        to its sliding window size in tokens, or ``-1`` for
        non-sliding-window layers.
    """
    per_layer_sw_size = [-1] * num_layers
    for group in vllm_groups:
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        # ``UniformTypeKVCacheSpecs`` carries per-layer specs in
        # ``kv_cache_specs``; other specs apply to all of the group's layers.
        per_layer_specs = getattr(spec, "kv_cache_specs", None)
        for name in group.layer_names:
            layer_spec = per_layer_specs[name] if per_layer_specs else spec
            if _is_sliding_window_spec(layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.sliding_window
    return per_layer_sw_size


def _merge_layer_sw_sizes(per_layer_sw_size: list[int], indices: list[int]) -> int:
    """Merge the per-layer sliding window sizes of one LMCache group.

    Args:
        per_layer_sw_size: Sliding window size per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common sliding window size in tokens, or ``-1`` when the
        layers are not sliding-window attention.

    Raises:
        ValueError: If the layers have different non-negative sliding window sizes.
    """
    sw_sizes = {per_layer_sw_size[idx] for idx in indices}
    if len(sw_sizes) != 1:
        raise ValueError(
            f"Layers with indices {indices} have different sliding window sizes "
            f"{sw_sizes}, but they are in the same group. This should "
            "not happen because vLLM should only group layers with the same "
            "KV cache spec, but got inconsistent metadata or registered tensors."
        )
    return sw_sizes.pop()


def create_engine_group_infos_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
) -> list[EngineGroupInfo]:
    """Build the LMCache engine group infos from vLLM metadata and registered tensors.

    This is the single entry point for the vLLM -> LMCache conversion. It reads
    the vLLM-specific fields (``KVCacheConfig.kv_cache_groups`` and
    ``KVCacheGroupSpec.layer_names`` from the v1 KV cache interface), maps each
    engine KV cache group's layer names to registered tensor indices, then
    splits the layers by physical transfer identity using the real tensors (via
    the shared :func:`lmcache.v1.kv_layer_groups.group_layers_by_identity`).
    vLLM-specific field access is intentionally confined to this function.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` describing the engine KV cache
            groups (or ``None`` / no groups, which yields a single-group spec).
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer-name -> tensor-index mapping; values
            are inspected for physical shape and dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. ``NHD``/``HND`` and compression metadata).

    Returns:
        The list of ``EngineGroupInfo`` in protocol order, i.e. the LMCache group
        order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import (
        normalize_and_discover_per_layer_formats,
    )
    from lmcache.v1.kv_layer_groups import (
        EXCLUDED_ENGINE_GROUP,
        group_layers_by_identity,
    )

    # vLLM-specific field access (confined to this function): map each
    # registered KV tensor to its vLLM engine KV cache group index. vLLM places
    # every registered layer in exactly one group; layers in different groups
    # have disjoint block-id spaces and must not share an LMCache group. ``None``
    # means a single (non-hybrid) group, i.e. every layer shares one block-id
    # space.
    per_layer_discoverable_kv_caches = list(kv_caches.values())
    layer_to_idx = {name: idx for idx, name in enumerate(kv_caches.keys())}
    vllm_groups = (
        getattr(kv_cache_config, "kv_cache_groups", ()) or ()
        if kv_cache_config is not None
        else ()
    )

    layer_index_groups = [
        [layer_to_idx[name] for name in group.layer_names] for group in vllm_groups
    ]
    normalized_kv_caches, engine_kv_formats = normalize_and_discover_per_layer_formats(
        per_layer_discoverable_kv_caches,
        layer_index_groups,
        EngineType.VLLM,
        layout_hints,
    )
    num_layers = len(engine_kv_formats)
    # Layers absent from every engine group's ``layer_names`` are cross-layer
    # KV-sharing layers (e.g. google/gemma-4-E4B-it): vLLM aliases them to a
    # target owner's KV tensor, so the owner's group already covers them. Tag
    # them EXCLUDED_ENGINE_GROUP so they form no group of their own (a
    # wrong-block-size group would corrupt the per-group block-id counts).
    per_layer_group_idx: list[int] | None = None
    group_tokens_per_block: dict[int, int] = {}
    per_layer_sw_size = [-1] * num_layers
    if vllm_groups:
        per_layer_group_idx = [EXCLUDED_ENGINE_GROUP] * num_layers
        for engine_group_id, group in enumerate(vllm_groups):
            # The spec's block_size is the logical tokens covered by one of
            # this group's paged chunks (block IDs); the physical slot count
            # per chunk is discovered later from the registered tensors.
            group_tokens_per_block[engine_group_id] = group.kv_cache_spec.block_size
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
        per_layer_sw_size = _resolve_per_layer_sw_sizes(
            vllm_groups, layer_to_idx, num_layers
        )

    # Within one vLLM engine group, layers can have different hidden dimensions
    # (e.g. a different head count), which require different GPU copy kernels.
    # ``group_layers_by_identity`` splits each engine group further by physical
    # transfer identity (kv_size, num_heads, head_size, block_size, dtype), so
    # every resulting LMCache group can be served by a single copy kernel. It is
    # the shared, engine-neutral primitive the server reuses to reproduce the
    # same grouping from the registered tensors.
    return [
        EngineGroupInfo(
            engine_group_id=identity.engine_group_idx,
            layer_indices=tuple(indices),
            tokens_per_block=group_tokens_per_block.get(identity.engine_group_idx, 0),
            sw_size_tokens=_merge_layer_sw_sizes(per_layer_sw_size, indices),
        )
        for identity, indices in group_layers_by_identity(
            normalized_kv_caches,
            engine_kv_formats,
            per_layer_group_idx,
        )
    ]
