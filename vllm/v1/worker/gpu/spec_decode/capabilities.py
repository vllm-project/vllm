# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the target model's layers can verify, chains or trees (see
``round_contract.md``, Startup)."""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

# Their home before resources.py; out-of-tree speculators import them from here.
from vllm.v1.spec_decode.resources import (  # noqa: F401
    PlanKind,
    SpecCapabilities,
    SpecLayerKind,
    SpecLimit,
    validate_effective_resources,
)
from vllm.v1.worker.gpu.spec_decode.kernels import KernelRegistry, SpecLayerSpec

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheGroup
    from vllm.v1.worker.utils import AttentionGroup


def spec_layers(
    attn_groups: Iterable[Iterable["AttentionGroup"]], layers: Mapping[str, Any]
) -> Iterator[tuple["AttentionGroup", str, Any, Any]]:
    """``(group, name, spec, layer)`` for each target layer a round verifies:
    not draft layers, hidden-state caches, or encoder or cross attention."""
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

    skipped_attn_types = (
        AttentionType.ENCODER,
        AttentionType.ENCODER_ONLY,
        AttentionType.ENCODER_DECODER,
    )
    for groups in attn_groups:
        for group in groups:
            spec = group.kv_cache_spec
            for name in group.layer_names:
                layer = layers.get(name)
                # A group's spec can come from a draft layer or a target layer
                # alike, so the layer says whether it is a draft layer.
                if (
                    getattr(layer, "is_draft_layer", False)
                    or isinstance(spec, HiddenStateCacheSpec)
                    or getattr(layer, "attn_type", None) in skipped_attn_types
                ):
                    continue
                yield group, name, spec, layer


def layer_spec_limits(
    attn_groups: Iterable[Iterable["AttentionGroup"]],
    layers: Mapping[str, Any],
    kv_cache_groups: Sequence["KVCacheGroup"],
    kernels: KernelRegistry | None = None,
) -> list[SpecLimit]:
    """The plan kind caps of the target's layers (:func:`spec_layers`), one per
    (layer kind, reason) with its layer count. ``kv_cache_groups`` are indexed
    by each attention group's ``kv_cache_group_id``. ``kernels`` defaults to
    none: then every layer that needs one verifies chains.
    """
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    if kernels is None:
        kernels = KernelRegistry()
    # (plan kind, what the layers are, why) -> number of layers
    num_layers: Counter[tuple[PlanKind, str, str]] = Counter()
    for group, name, spec, layer in spec_layers(attn_groups, layers):
        if isinstance(layer, MambaBase):
            kind = SpecLayerKind("recurrent", type(layer).__name__)
            why = layer.tree_decline_reason()
            layout = "recurrent_state"
        else:
            kv_cache_group = kv_cache_groups[group.kv_cache_group_id]
            kind = attn_layer_kind(kv_cache_group, spec)
            why = group.backend.tree_decline_reason(layer, spec)
            layout = attn_layout(layer, kv_cache_group, spec)
        if why is not None:
            num_layers[("chain", kind.key(), why)] += 1
            continue
        plan_kind, why = kernels.plan_kind(
            SpecLayerSpec(name, kind, layer, layout=layout)
        )
        num_layers[(plan_kind, kind.key(), why)] += 1
    return [
        SpecLimit(plan_kind, f"{what}, {n} layer{'s' * (n != 1)}: {why}")
        for (plan_kind, what, why), n in num_layers.items()
        if plan_kind == "chain"
    ]


def causal_full_attention(layer: Any, spec: Any) -> bool:
    """Whether an attention layer attends to its whole context: a decoder
    ``Attention`` layer without a sliding window, which ``Attention`` gives a
    full attention KV cache manager. Subclasses that pick their own manager
    (chunked, sink, RSWA, ring) are not."""
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backend import AttentionType

    return (
        isinstance(layer, Attention)
        and type(layer).get_kv_cache_config is Attention.get_kv_cache_config
        and getattr(layer, "sliding_window", None) is None
        and getattr(layer, "attn_type", AttentionType.DECODER) == AttentionType.DECODER
    )


def attn_layer_kind(kv_cache_group: "KVCacheGroup", spec: Any) -> SpecLayerKind:
    """The kind of an attention layer with KV cache spec ``spec`` in
    ``kv_cache_group``: the kind of the group's KV cache manager."""
    from vllm.v1.core.single_type_kv_cache_manager import get_kv_cache_kind

    return SpecLayerKind(
        "attn", get_kv_cache_kind(kv_cache_group.manager_config, [spec]).value
    )


def attn_layout(layer: Any, kv_cache_group: "KVCacheGroup", spec: Any) -> str:
    """What committing an attention layer's accepted path moves (``""``: a
    cache not paged by slot, whose model keeps its own commit)."""
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    if not (kv_cache_group.manager_cls.uses_slot_mapping and spec.uses_slot_mapping):
        return ""
    mode = get_kv_quant_mode(str(getattr(layer, "kv_cache_dtype", "auto")))
    if mode == KVQuantMode.NVFP4 or mode.is_per_token_head:
        return "paged_split_scale"
    return "paged_token"
