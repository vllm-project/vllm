# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from enum import Enum

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheSpecKind,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind_for_class,
)


class TransferClass(Enum):
    """How a KV cache spec moves across engines."""

    ATTENTION = "attention"
    SSM = "ssm"
    OTHER = "other"


_KIND_TO_TRANSFER_CLASS: dict[KVCacheSpecKind, TransferClass] = {
    KVCacheSpecKind.FULL_ATTENTION: TransferClass.ATTENTION,
    KVCacheSpecKind.MLA_ATTENTION: TransferClass.ATTENTION,
    KVCacheSpecKind.SLIDING_WINDOW: TransferClass.ATTENTION,
    KVCacheSpecKind.SLIDING_WINDOW_MLA: TransferClass.ATTENTION,
    KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION: TransferClass.ATTENTION,
    KVCacheSpecKind.SINK_FULL_ATTENTION: TransferClass.ATTENTION,
    KVCacheSpecKind.ENCODER_ONLY_ATTENTION: TransferClass.ATTENTION,
    KVCacheSpecKind.CROSS_ATTENTION: TransferClass.ATTENTION,
    KVCacheSpecKind.MAMBA: TransferClass.SSM,
}

_MLA_KINDS = (KVCacheSpecKind.MLA_ATTENTION, KVCacheSpecKind.SLIDING_WINDOW_MLA)


def get_representative_spec(spec: KVCacheSpec) -> KVCacheSpec:
    """Return the spec that stands for a cache group.

    Args:
        spec: a group spec; ``UniformTypeKVCacheSpecs`` wraps one spec per layer.

    Returns:
        The first wrapped spec for a wrapper, ``spec`` itself otherwise.

    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.first_spec
    return spec


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    """Return the concrete spec type behind ``spec``, unwrapping uniform-type groups."""
    return type(get_representative_spec(spec))


def _spec_class(spec: type[KVCacheSpec] | KVCacheSpec) -> type[KVCacheSpec]:
    """Return the spec class a spec or a spec class stands for.

    Args:
        spec: a spec instance, a group spec, or a spec class.

    Returns:
        The class itself for a class argument, the class of the representative
        spec for an instance.

    Raises:
        ValueError: If ``spec`` is the wrapper class, which describes no single
            spec and therefore has no class-level kind.

    """
    if isinstance(spec, type):
        if issubclass(spec, UniformTypeKVCacheSpecs):
            raise ValueError(
                "UniformTypeKVCacheSpecs is a wrapper with no class-level kind; "
                "pass a wrapped spec or a group instance instead."
            )
        return spec
    return type(get_representative_spec(spec))


def transfer_class(spec: type[KVCacheSpec] | KVCacheSpec) -> TransferClass:
    """Return the transfer class of a spec or a spec class.

    The class comes from ``get_kv_cache_spec_kind_for_class`` wherever the kind
    table names the kind. The kinds it reports as ``UNKNOWN`` are decided by the
    spec hierarchy, so a spec the framework has not named yet transfers like its
    base class.

    Args:
        spec: a spec instance, a group spec, or a spec class.

    Returns:
        ``TransferClass.ATTENTION``, ``TransferClass.SSM`` or
        ``TransferClass.OTHER``.

    """
    spec_cls = _spec_class(spec)
    mapped = _KIND_TO_TRANSFER_CLASS.get(get_kv_cache_spec_kind_for_class(spec_cls))
    if mapped is not None:
        return mapped
    if issubclass(spec_cls, AttentionSpec):
        return TransferClass.ATTENTION
    if issubclass(spec_cls, MambaSpec):
        return TransferClass.SSM
    return TransferClass.OTHER


def is_attention_spec(spec: type[KVCacheSpec] | KVCacheSpec) -> bool:
    """Whether a spec or spec class transfers as attention state."""
    return transfer_class(spec) is TransferClass.ATTENTION


def is_ssm_spec(spec: type[KVCacheSpec] | KVCacheSpec) -> bool:
    """Whether a spec or spec class transfers as SSM state."""
    return transfer_class(spec) is TransferClass.SSM


def is_mla_spec(spec: type[KVCacheSpec] | KVCacheSpec) -> bool:
    """Whether a spec or spec class is MLA, including its sliding-window form."""
    spec_cls = _spec_class(spec)
    kind = get_kv_cache_spec_kind_for_class(spec_cls)
    if kind in _MLA_KINDS:
        return True
    if kind is KVCacheSpecKind.UNKNOWN:
        return issubclass(spec_cls, (MLAAttentionSpec, SlidingWindowMLASpec))
    return False


def build_layer_to_spec(kv_cache_config: KVCacheConfig) -> dict[str, KVCacheSpec]:
    """Map every layer name to its own spec, unwrapping uniform-type groups."""
    layer_to_spec: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        if isinstance(group_spec, UniformTypeKVCacheSpecs):
            layer_to_spec.update(
                {
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in group.layer_names
                }
            )
        else:
            layer_to_spec.update(
                {layer_name: group_spec for layer_name in group.layer_names}
            )
    return layer_to_spec
