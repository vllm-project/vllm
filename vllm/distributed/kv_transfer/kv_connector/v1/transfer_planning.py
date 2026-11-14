# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Classify KV cache groups by how their state moves across engines."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheGroup,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)


class TransferClass(Enum):
    """How a KV cache spec moves across engines."""

    ATTENTION = "attention"
    SSM = "ssm"
    OTHER = "other"


def get_representative_spec_type(group: KVCacheGroup) -> type[KVCacheSpec]:
    """Return the concrete spec type of the group's first layer.

    Args:
        group: a cache group of the connector's KV cache config.

    Returns:
        The type of the group's representative spec.

    """
    return type(group.distinct_specs[0])


def get_dcp_replicated_group_ids(groups: Sequence[KVCacheGroup]) -> set[int]:
    """Return the MLA groups held whole on every DCP rank.

    A draft running at DCP=1 under a DCP target keeps its MLA cache unsharded,
    so a peer holds it in full. Non-MLA replicated caches keep their split
    transfer path.

    Args:
        groups: the transfer groups of the connector's KV cache config.

    Returns:
        The indices of the groups whose MLA spec is not DCP sharded.

    """
    return {
        i
        for i, group in enumerate(groups)
        if issubclass(get_representative_spec_type(group), MLAAttentionSpec)
        and not group.dcp_sharded
    }


def _spec_class(spec: type[KVCacheSpec] | KVCacheSpec) -> type[KVCacheSpec]:
    """Return the spec class a spec or a spec class stands for.

    Args:
        spec: a spec instance or a spec class.

    Returns:
        The class itself for a class argument, the class of the given spec
        otherwise.

    """
    return spec if isinstance(spec, type) else type(spec)


def _class_transfer_class(spec_cls: type[KVCacheSpec]) -> TransferClass:
    """Return the transfer class of one spec class.

    The spec hierarchy decides: SSM state is a ``MambaSpec``, attention state
    is an ``AttentionSpec``, and any other spec moves as opaque state.

    Args:
        spec_cls: a concrete spec class.

    Returns:
        The matching ``TransferClass``.

    """
    if issubclass(spec_cls, MambaSpec):
        return TransferClass.SSM
    if issubclass(spec_cls, AttentionSpec):
        return TransferClass.ATTENTION
    return TransferClass.OTHER


def transfer_class(
    spec: type[KVCacheSpec] | KVCacheSpec | KVCacheGroup,
) -> TransferClass:
    """Return the transfer class of a spec, a spec class or a cache group.

    A group moves as one unit, so every distinct spec of a ``KVCacheGroup`` has
    to agree.

    Args:
        spec: a spec instance, a spec class, or a cache group.

    Returns:
        ``TransferClass.ATTENTION``, ``TransferClass.SSM`` or
        ``TransferClass.OTHER``.

    Raises:
        ValueError: If a group's layers do not share one transfer class.

    """
    if isinstance(spec, KVCacheGroup):
        layer_classes = {
            _class_transfer_class(type(layer_spec))
            for layer_spec in spec.distinct_specs
        }
        if len(layer_classes) != 1:
            raise ValueError(
                "A cache group does not move as one transfer class: "
                f"{sorted(cls.value for cls in layer_classes)}"
            )
        return next(iter(layer_classes))
    return _class_transfer_class(_spec_class(spec))


def is_attention_spec(spec: type[KVCacheSpec] | KVCacheSpec | KVCacheGroup) -> bool:
    """Whether a spec, spec class or cache group transfers as attention state."""
    return transfer_class(spec) is TransferClass.ATTENTION


def is_ssm_spec(spec: type[KVCacheSpec] | KVCacheSpec | KVCacheGroup) -> bool:
    """Whether a spec, spec class or cache group transfers as SSM state."""
    return transfer_class(spec) is TransferClass.SSM
