# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build LMCache engine group infos from vLLM KV cache group metadata.

Synced from `lmcache/integration/vllm/kv_cache_groups.py` at LMCache
v0.5.7.post1 plus LMCache/LMCache#6608, for the builtin LMCache MP connector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmcache.v1.gpu_connector.utils import LayoutHints

from lmcache.logging import init_logger
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def _is_attention_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM attention spec.

    Checked by class name so this module stays importable without vLLM.
    `UniformTypeKVCacheSpecs` is unwrapped first (same-typed layers, one
    leaf suffices); it does not derive from `AttentionSpec` itself.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        spec = next(iter(inner.values()))
    return any(cls.__name__ == "AttentionSpec" for cls in type(spec).__mro__)


def is_scratch_spec(spec: Any) -> bool:
    """Return whether vLLM marks the spec as non-prefix-cacheable.

    vLLM never hashes or restores the blocks of such a scratch group, so they
    carry no token range LMCache could store (e.g. the per-request rings
    `CircularBufferSpec` and `KpoolTailSpec`). `prefix_cacheable` takes
    precedence over the older `participates_in_prefix_caching` marker.
    Specs with neither marker are prefix-cacheable.
    `UniformTypeKVCacheSpecs` is unwrapped first.

    Args:
        spec: A vLLM KV cache spec, or a `UniformTypeKVCacheSpecs` container.

    Returns:
        `True` for a scratch spec, `False` for a prefix-cacheable one.

    """
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        spec = next(iter(inner.values()))
    if hasattr(spec, "prefix_cacheable"):
        return not spec.prefix_cacheable
    return not getattr(spec, "participates_in_prefix_caching", True)


def get_tokens_per_block(kv_cache_spec: Any, dcp_size: int) -> int:
    """Global tokens covered by one block id of `kv_cache_spec`.

    Attention blocks span `block_size * dcp_size` tokens under DCP
    (vLLM's `resolve_kv_cache_block_sizes` rule); recurrent state is
    replicated, not sharded, and stays at `block_size`. Scratch specs
    (see `is_scratch_spec`) cover no tokens and yield `0`.
    """
    if is_scratch_spec(kv_cache_spec):
        return 0
    block_size = kv_cache_spec.block_size
    if dcp_size <= 1:
        return block_size
    if _is_attention_spec(kv_cache_spec):
        return block_size * dcp_size
    return block_size


def _is_sliding_window_spec(spec: Any) -> bool:
    """Return whether the KV cache spec is a vLLM sliding-window spec.

    Checked by class name so this module stays importable without vLLM.
    Subclasses such as `SlidingWindowMLASpec` count.
    """
    return any(cls.__name__ == "SlidingWindowSpec" for cls in type(spec).__mro__)


# vLLM's allocation-policy refactor (vllm-project/vllm#61452) replaced
# ``KVCacheGroupSpec.kv_cache_spec`` with ``KVCacheGroup.kv_cache_specs`` plus a
# ``manager_config`` that holds the sliding window and the Mamba cache mode.
# The ``group_*`` helpers below read either form.


def _manager_config(group: Any) -> Any:
    return getattr(group, "manager_config", None)


def _manager_is(group: Any, name: str) -> bool:
    config = _manager_config(group)
    return config is not None and any(
        cls.__name__ == name for cls in config.manager_cls.__mro__
    )


def group_layer_specs(group: Any) -> dict[str, Any]:
    """Return each layer's KV cache spec in a vLLM KV cache group.

    Older vLLM keeps one `kv_cache_spec` per group, possibly a
    `UniformTypeKVCacheSpecs` with per-layer specs inside.
    """
    specs = getattr(group, "kv_cache_specs", None)
    if isinstance(specs, dict):
        return specs
    spec = group.kv_cache_spec
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict) and inner:
        return {name: inner[name] for name in group.layer_names}
    return {name: spec for name in group.layer_names}


def group_specs(group: Any) -> list[Any]:
    """Return the distinct KV cache specs of a vLLM KV cache group.

    On new vLLM this is `distinct_specs`: under pipeline parallelism a group
    projected onto a stage can hold none of its layers in `kv_cache_specs`.
    """
    specs = getattr(group, "kv_cache_specs", None)
    if isinstance(specs, dict):
        return list(group.distinct_specs)
    spec = group.kv_cache_spec
    inner = getattr(spec, "kv_cache_specs", None)
    if isinstance(inner, dict):
        return list(inner.values())
    return [spec]


def group_sliding_window(group: Any, spec: Any) -> int | None:
    """Return the sliding window in tokens of one layer, or `None`."""
    if _manager_config(group) is not None:
        if _manager_is(group, "SlidingWindowManager"):
            return int(_manager_config(group).params.sliding_window)
        return None
    if _is_sliding_window_spec(spec):
        return spec.sliding_window
    return None


def _is_mamba_spec(spec: Any) -> bool:
    return any(cls.__name__ == "MambaSpec" for cls in type(spec).__mro__)


def group_mamba_cache_mode(group: Any, spec: Any) -> str:
    """Return the Mamba cache mode set for one layer, `"none"` if unset."""
    config = _manager_config(group)
    if config is not None:
        return getattr(config.params, "mamba_cache_mode", "none")
    return getattr(spec, "mamba_cache_mode", "none")


def _is_cacheable_mamba(group: Any, spec: Any) -> bool:
    """Return whether the layer is Mamba in a snapshotting cache mode.

    Align mode snapshots only the last block; all mode snapshots every block
    boundary. Either way a restore consumes only the last matched block's
    page, so both behave like a cross-chunk sliding window of one block.
    """
    return _is_mamba_spec(spec) and group_mamba_cache_mode(group, spec) in (
        "align",
        "all",
    )


def is_scratch_group(group: Any) -> bool:
    """Return whether vLLM never prefix-caches the group (see
    `is_scratch_spec`)."""
    if _manager_config(group) is not None:
        return not group.prefix_cacheable
    return is_scratch_spec(group.kv_cache_spec)


def tokens_per_group_block(group: Any, dcp_size: int) -> int:
    """Return the global tokens one block id of the group covers (see
    `get_tokens_per_block`)."""
    if _manager_config(group) is None:
        return get_tokens_per_block(group.kv_cache_spec, dcp_size)
    if is_scratch_group(group):
        return 0
    return group.block_size * (dcp_size if group.dcp_sharded else 1)


def group_prefix_replay_tokens(group: Any) -> int:
    """Return the group's prefix-replay window in tokens, `0` if none."""
    if _manager_config(group) is not None:
        return int(group.prefix_replay_tokens or 0)
    return int(getattr(group.kv_cache_spec, "prefix_replay_tokens", 0) or 0)


def _resolve_per_layer_sw_sizes(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[int]:
    """Resolve the sliding window size in tokens for each registered KV tensor.

    Will resolve -1 for non-sliding-window layers.

    Args:
        vllm_groups: vLLM KV cache groups.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length `num_layers` mapping each registered tensor index
        to its cross-chunk window size in tokens: the sliding window for
        sliding-window attention, the engine-side block size for align-mode
        Mamba/linear layers (a one-block window), or `-1` for full-attention
        layers.

    """
    per_layer_sw_size = [-1] * num_layers
    for group in vllm_groups:
        for name, layer_spec in group_layer_specs(group).items():
            sliding_window = group_sliding_window(group, layer_spec)
            if sliding_window is not None:
                per_layer_sw_size[layer_to_idx[name]] = sliding_window
            elif _is_cacheable_mamba(group, layer_spec):
                per_layer_sw_size[layer_to_idx[name]] = layer_spec.block_size
    return per_layer_sw_size


#: Reserved layer-name prefix for CacheBlend fused-aux page pools:
#: ``cb.aux_pool.<tokens_per_block>[.<label>]`` — the first suffix part is
#: the logical block size; the optional label distinguishes multiple pools
#: sharing one block size (they then share an engine group).
CB_AUX_POOL_LAYER_PREFIX = "cb.aux_pool."


def cb_aux_pool_entries(kv_caches) -> list[tuple[str, int]]:
    """CacheBlend fused-aux pool entries among the registered tensors.

    Presence-gated: any layer name starting with
    `CB_AUX_POOL_LAYER_PREFIX` is a connector-owned aux page pool
    served by a synthetic engine group (one group per block size).

    Args:
        kv_caches: Registered tensors keyed by layer name.

    Returns:
        `(layer_name, tokens_per_block)` per aux pool, in registration
        order; empty for models without one.

    Raises:
        ValueError: If a marker name's block-size suffix does not parse.

    """
    entries: list[tuple[str, int]] = []
    for name in kv_caches:
        if not name.startswith(CB_AUX_POOL_LAYER_PREFIX):
            continue
        suffix = name[len(CB_AUX_POOL_LAYER_PREFIX) :].split(".", 1)[0]
        try:
            tokens_per_block = int(suffix)
        except ValueError as exc:
            raise ValueError(
                f"aux pool layer name {name!r}: block-size suffix "
                f"{suffix!r} is not an integer"
            ) from exc
        if tokens_per_block <= 0:
            raise ValueError(
                f"aux pool layer name {name!r}: tokens_per_block must be "
                f"positive, got {tokens_per_block}"
            )
        entries.append((name, tokens_per_block))
    return entries


def _resolve_per_layer_recurrent(
    vllm_groups: Sequence[Any],
    layer_to_idx: Mapping[str, int],
    num_layers: int,
) -> list[bool]:
    """Resolve whether each registered KV tensor holds recurrent state pages.

    Args:
        vllm_groups: vLLM KV cache groups.
        layer_to_idx: Layer name to registered tensor index mapping.
        num_layers: Number of registered KV tensors.

    Returns:
        A list of length `num_layers`: `True` for Mamba/linear-attention
        layers in a snapshotting cache mode (see `_is_cacheable_mamba`),
        `False` for attention layers.

    """
    per_layer_recurrent = [False] * num_layers
    for group in vllm_groups:
        for name, layer_spec in group_layer_specs(group).items():
            if _is_cacheable_mamba(group, layer_spec):
                per_layer_recurrent[layer_to_idx[name]] = True
    return per_layer_recurrent


def _merge_layer_recurrent(per_layer_recurrent: list[bool], indices: list[int]) -> bool:
    """Merge the per-layer recurrent-state flags of one LMCache group.

    Args:
        per_layer_recurrent: Recurrent-state flag per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common flag.

    Raises:
        ValueError: If the group mixes recurrent and attention layers (vLLM
            groups layers by KV cache spec, so a mix indicates inconsistent
            metadata).

    """
    flags = {per_layer_recurrent[idx] for idx in indices}
    if len(flags) != 1:
        raise ValueError(
            f"Layers with indices {indices} mix recurrent-state and attention "
            "layers in one group. This should not happen because vLLM only "
            "groups layers with the same KV cache spec."
        )
    return flags.pop()


def _merge_layer_sw_sizes(per_layer_sw_size: list[int], indices: list[int]) -> int:
    """Merge the per-layer sliding window sizes of one LMCache group.

    Args:
        per_layer_sw_size: Sliding window size per registered tensor index.
        indices: Registered tensor indices of the group's layers.

    Returns:
        The group's common sliding window size in tokens, or `-1` when the
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
    layout_hints: LayoutHints | None = None,
    dcp_size: int = 1,
) -> list[EngineGroupInfo]:
    """Build the LMCache engine group infos from vLLM metadata and registered tensors.

    This is the single entry point for the vLLM -> LMCache conversion. It reads
    the vLLM-specific fields (`KVCacheConfig.kv_cache_groups` and
    the groups' `layer_names` from the v1 KV cache interface), maps each
    engine KV cache group's layer names to registered tensor indices, then
    splits the layers by physical transfer identity using the real tensors (via
    the shared `lmcache.v1.kv_layer_groups.group_layers_by_identity`).
    vLLM-specific field access is intentionally confined to this function.

    Args:
        kv_cache_config: vLLM `KVCacheConfig` describing the engine KV cache
            groups (or `None` / no groups, which yields a single-group spec).
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer-name -> tensor-index mapping; values
            are inspected for physical shape and dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. `NHD`/`HND` and compression metadata).
        dcp_size: Decode context parallel size.

    Note:
        Under DCP each attention group's `tokens_per_block` is scaled by
        `dcp_size` to stay in the scheduler's coordinate space; its ratio
        to the physical slot count is what sizes each rank's memory object.
        Mamba groups are replicated per rank and stay unscaled. Layers of
        scratch groups (see `is_scratch_spec`) are excluded and never
        form an info.

    Returns:
        The list of `EngineGroupInfo` in protocol order, i.e. the LMCache group
        order used by store/retrieve block IDs.

    """
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

    # Scratch groups skip format discovery: their layers are excluded below and
    # never transferred, so a layout the transfer kernels reject must not fail
    # registration.
    layer_index_groups = [
        [layer_to_idx[name] for name in group.layer_names]
        for group in vllm_groups
        if not is_scratch_group(group)
    ]

    # CacheBlend fused-aux (presence-gated): the pool joins detection as
    # its own group so its rank-3 layout is classified independently.
    aux_entries = cb_aux_pool_entries(kv_caches)
    layer_index_groups += [[layer_to_idx[name]] for name, _ in aux_entries]
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
    # Scratch groups (tokens_per_block 0) are excluded the same way.
    per_layer_group_idx: list[int] | None = None
    group_tokens_per_block: dict[int, int] = {}
    per_layer_sw_size = [-1] * num_layers
    per_layer_recurrent = [False] * num_layers
    if vllm_groups:
        per_layer_group_idx = [EXCLUDED_ENGINE_GROUP] * num_layers
        for engine_group_id, group in enumerate(vllm_groups):
            # The spec's block_size is the logical tokens covered by one of
            # this group's paged chunks (block IDs); the physical slot count
            # per chunk is discovered later from the registered tensors.
            # Under DCP the two diverge (see get_tokens_per_block).
            group_tokens_per_block[engine_group_id] = tokens_per_group_block(
                group, dcp_size
            )
            if group_tokens_per_block[engine_group_id] == 0:
                continue
            for name in group.layer_names:
                per_layer_group_idx[layer_to_idx[name]] = engine_group_id
        per_layer_sw_size = _resolve_per_layer_sw_sizes(
            vllm_groups, layer_to_idx, num_layers
        )
        per_layer_recurrent = _resolve_per_layer_recurrent(
            vllm_groups, layer_to_idx, num_layers
        )

    # Aux pools form synthetic engine groups after the vLLM groups (an
    # unassigned marker layer would fall to EXCLUDED_ENGINE_GROUP and never
    # store), bucketed by tokens_per_block: pools sharing a block size share
    # one engine group — and thereby one kernel group when their tensor
    # identities also match. Tags are 1-based per bucket.
    aux_group_tags: dict[int, int] = {}
    if aux_entries:
        if per_layer_group_idx is None:
            # Non-hybrid engine config: all real layers share group 0.
            per_layer_group_idx = [0] * num_layers
        next_group_id = len(vllm_groups) if vllm_groups else 1
        group_by_tpb: dict[int, int] = {}
        for name, tokens_per_block in aux_entries:
            group_id = group_by_tpb.get(tokens_per_block)
            if group_id is None:
                group_id = next_group_id
                next_group_id += 1
                group_by_tpb[tokens_per_block] = group_id
                group_tokens_per_block[group_id] = tokens_per_block
                aux_group_tags[group_id] = len(group_by_tpb)
            per_layer_group_idx[layer_to_idx[name]] = group_id

    # Within one vLLM engine group, layers can have different hidden dimensions
    # (e.g. a different head count), which require different GPU copy kernels.
    # ``group_layers_by_identity`` splits each engine group further by physical
    # transfer identity (kv_size, num_heads, head_size, block_size,
    # engine_group_idx, dtype, engine_kv_format, block_stride_elems), so every
    # resulting LMCache group can be served by a single copy kernel. It is the
    # shared, engine-neutral primitive the server reuses to reproduce the same
    # grouping from the registered tensors.
    return [
        EngineGroupInfo(
            engine_group_id=identity.engine_group_idx,
            layer_indices=tuple(indices),
            tokens_per_block=group_tokens_per_block.get(identity.engine_group_idx, 0),
            sw_size_tokens=_merge_layer_sw_sizes(per_layer_sw_size, indices),
            # Connector-private pools bucket by tag under
            # --separate-object-groups, after the regular groups.
            extra_object_group_tag=aux_group_tags.get(identity.engine_group_idx, 0),
            recurrent_state=_merge_layer_recurrent(per_layer_recurrent, indices),
        )
        for identity, indices in group_layers_by_identity(
            normalized_kv_caches,
            engine_kv_formats,
            per_layer_group_idx,
        )
    ]
