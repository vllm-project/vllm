# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from enum import Enum

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind_for_class,
    iter_layer_specs,
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


def get_representative_spec(spec: KVCacheSpec) -> KVCacheSpec:
    """Return the spec that stands for a cache group.

    Layers only merge into a group when they share one registered base spec, so
    every wrapped spec answers the group-level questions the same way.

    Args:
        spec: a group spec; ``UniformTypeKVCacheSpecs`` wraps one spec per layer.

    Returns:
        The first wrapped spec for a wrapper, ``spec`` itself otherwise.

    Raises:
        ValueError: If the wrapper carries no layer spec.

    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        if not spec.kv_cache_specs:
            raise ValueError("UniformTypeKVCacheSpecs carries no layer spec")
        return spec.first_spec
    return spec


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    """Return the concrete spec type behind ``spec``, unwrapping uniform-type groups.

    Only the concrete type of the representative spec is returned, which is the
    answer a group-level question needs: layers merge into a group only when
    they share one registered base spec, so any wrapped spec stands for the
    group. Ask a per-layer question through ``iter_layer_specs`` instead.
    """
    return type(get_representative_spec(spec))


def _spec_class(spec: type[KVCacheSpec] | KVCacheSpec) -> type[KVCacheSpec]:
    """Return the spec class a spec or a spec class stands for.

    Args:
        spec: a spec instance or a spec class.

    Returns:
        The class itself for a class argument, the class of the given spec
        otherwise.

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
    return type(spec)


def _class_transfer_class(spec_cls: type[KVCacheSpec]) -> TransferClass:
    """Return the transfer class of one spec class.

    The class comes from ``get_kv_cache_spec_kind_for_class`` wherever the kind
    table names the kind. The kinds it reports as ``UNKNOWN`` are decided by the
    spec hierarchy, so a spec the framework has not named yet transfers like its
    base class.
    """
    mapped = _KIND_TO_TRANSFER_CLASS.get(get_kv_cache_spec_kind_for_class(spec_cls))
    if mapped is not None:
        return mapped
    if issubclass(spec_cls, AttentionSpec):
        return TransferClass.ATTENTION
    if issubclass(spec_cls, MambaSpec):
        return TransferClass.SSM
    return TransferClass.OTHER


def transfer_class(spec: type[KVCacheSpec] | KVCacheSpec) -> TransferClass:
    """Return the transfer class of a spec, a group spec or a spec class.

    A group moves as one unit, so every layer of a ``UniformTypeKVCacheSpecs``
    has to agree and the answer does not depend on which layer comes first.

    Args:
        spec: a spec instance, a group spec, or a spec class.

    Returns:
        ``TransferClass.ATTENTION``, ``TransferClass.SSM`` or
        ``TransferClass.OTHER``.

    Raises:
        ValueError: If ``spec`` is the wrapper class, or a group whose layers do
            not share one transfer class.

    """
    if isinstance(spec, UniformTypeKVCacheSpecs):
        layer_classes = {
            _class_transfer_class(type(layer_spec))
            for layer_spec in iter_layer_specs(spec)
        }
        if len(layer_classes) != 1:
            raise ValueError(
                "A cache group does not move as one transfer class: "
                f"{sorted(cls.value for cls in layer_classes)}"
            )
        return next(iter(layer_classes))
    return _class_transfer_class(_spec_class(spec))


def is_attention_spec(spec: type[KVCacheSpec] | KVCacheSpec) -> bool:
    """Whether a spec, group spec or spec class transfers as attention state."""
    return transfer_class(spec) is TransferClass.ATTENTION


def is_ssm_spec(spec: type[KVCacheSpec] | KVCacheSpec) -> bool:
    """Whether a spec, group spec or spec class transfers as SSM state."""
    return transfer_class(spec) is TransferClass.SSM
