# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the target model's layers can verify, chains or trees (see
``round_contract.md``, Startup)."""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
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
    from vllm.v1.worker.utils import AttentionGroup


def spec_layers(
    attn_groups: Iterable[Iterable["AttentionGroup"]], layers: Mapping[str, Any]
) -> Iterator[tuple["AttentionGroup", str, Any, Any]]:
    """``(group, name, spec, layer)`` for each target layer a round verifies:
    not draft layers, hidden-state caches, or encoder or cross attention."""
    from vllm.v1.kv_cache_interface import (
        CrossAttentionSpec,
        EncoderOnlyAttentionSpec,
        HiddenStateCacheSpec,
        UniformTypeKVCacheSpecs,
    )

    skipped = (HiddenStateCacheSpec, EncoderOnlyAttentionSpec, CrossAttentionSpec)
    for groups in attn_groups:
        for group in groups:
            for name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                layer = layers.get(name)
                # A group's spec can come from a draft layer or a target layer
                # alike, so the layer says whether it is a draft layer.
                if getattr(layer, "is_draft_layer", False) or isinstance(spec, skipped):
                    continue
                yield group, name, spec, layer


def layer_spec_limits(
    attn_groups: Iterable[Iterable["AttentionGroup"]],
    layers: Mapping[str, Any],
    kernels: KernelRegistry | None = None,
) -> list[SpecLimit]:
    """The plan kind caps of the target's layers (:func:`spec_layers`), one per
    (layer kind, reason) with its layer count. ``kernels`` defaults to none:
    then every layer that needs one verifies chains.
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
            kind = attn_layer_kind(spec)
            why = group.backend.tree_decline_reason(layer, spec)
            layout = attn_layout(layer, spec)
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
    """Whether an attention layer attends to its whole context (no window or chunk)."""
    return attn_layer_kind(spec).variant == "full_attention" and not any(
        getattr(spec, name, None) or getattr(layer, name, None)
        for name in ("sliding_window", "attention_chunk_size", "rswa_window")
    )


def attn_layer_kind(spec: Any) -> SpecLayerKind:
    """The kind of an attention layer with KV cache spec ``spec``."""
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    return SpecLayerKind("attn", get_kv_cache_spec_kind(spec).value)


def attn_layout(layer: Any, spec: Any) -> str:
    """What committing an attention layer's accepted path moves (``""``: a
    cache not paged by slot, whose model keeps its own commit)."""
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    if not getattr(spec, "uses_slot_mapping", True):
        return ""
    mode = get_kv_quant_mode(str(getattr(layer, "kv_cache_dtype", "auto")))
    if mode == KVQuantMode.NVFP4 or mode.is_per_token_head:
        return "paged_split_scale"
    return "paged_token"
