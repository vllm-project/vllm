# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the target model can verify, chains or trees, and the startup rule that
lowers a speculator's resources to it (see ``round_contract.md``, Startup)."""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from vllm.logger import init_logger
from vllm.v1.spec_decode.resources import PlanKind

if TYPE_CHECKING:
    from vllm.v1.spec_decode.resources import SpecResources
    from vllm.v1.worker.gpu.spec_decode.kernels import KernelRegistry
    from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig
    from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


@dataclass(frozen=True)
class SpecLayerKind:
    """What a layer is to speculative decoding, derived from the KV cache spec
    or the layer class (a plugin cannot invent one)."""

    family: Literal["attn", "recurrent", "layout"]
    variant: str
    """attn: the ``KVCacheSpecKind`` value; recurrent: the layer class name;
    layout: ``"paged_token"``, ``"paged_split_scale"`` or ``"recurrent_state"``."""

    def key(self) -> str:
        return f"{self.family}:{self.variant}"


@dataclass(frozen=True)
class SpecLimit:
    """A layer kind or a setting that caps the plan kind, and why."""

    plan_kind: PlanKind
    reason: str


@dataclass(frozen=True)
class SpecCapabilities:
    """What a worker's layers verify; reduced over ranks by the engine core."""

    plan_kind: PlanKind
    """``"chain"`` when any layer or setting caps it, else ``"tree"``."""
    limits: tuple[SpecLimit, ...] = ()
    """Every cap to chains, for the startup log."""

    @classmethod
    def from_limits(cls, limits: Iterable[SpecLimit]) -> "SpecCapabilities":
        limits = tuple(dict.fromkeys(limits))
        return cls(plan_kind="chain" if limits else "tree", limits=limits)


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
    kernels: "KernelRegistry | None" = None,
) -> list[SpecLimit]:
    """The plan kind caps of the target's layers (:func:`spec_layers`), one per
    (layer kind, reason) with its layer count. ``kernels`` defaults to none:
    then every layer that needs one verifies chains.
    """
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.worker.gpu.spec_decode.kernels import KernelRegistry, SpecLayerSpec

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


def resolve_spec_resources(
    speculator_config: "SpeculatorConfig",
    declared: "SpecResources",
    capabilities: Sequence[SpecCapabilities],
) -> "SpecResources":
    """The speculator's declared resources adapted to what every rank verifies
    (``capabilities``, one per worker); logs each lowering."""
    # Pipeline ranks hold different layers: the model verifies the lowest.
    caps = SpecCapabilities.from_limits(
        limit for rank in capabilities for limit in rank.limits
    )
    lowering = caps.limits if declared.plan_kind == "tree" else ()
    for limit in lowering:
        logger.info(
            "spec decode: %s -> %s (%s)",
            declared.plan_kind,
            limit.plan_kind,
            limit.reason,
        )
    effective = speculator_config.adapt_resources(declared, caps)
    validate_effective_resources(declared, effective, caps)
    if effective.max_rows < declared.max_rows:
        logger.info(
            "spec decode: max_rows %d -> %d (adapt_resources)",
            declared.max_rows,
            effective.max_rows,
        )
    return effective


def validate_effective_resources(
    declared: "SpecResources", effective: "SpecResources", caps: SpecCapabilities
) -> None:
    """Check that ``effective`` takes no more than ``declared`` and ``caps``."""
    problems = []
    if effective.plan_kind == "tree" and "chain" in (
        declared.plan_kind,
        caps.plan_kind,
    ):
        problems.append(
            f"plan_kind {effective.plan_kind!r} is above the declared "
            f"{declared.plan_kind!r} or the model's {caps.plan_kind!r}"
        )
    if effective.max_branching > declared.max_branching:
        problems.append(
            f"max_branching {effective.max_branching} is above the declared "
            f"{declared.max_branching}"
        )
    extra_rows = set(effective.capture_rows) - set(declared.capture_rows)
    if extra_rows:
        problems.append(f"capture_rows {sorted(extra_rows)} not declared")
    if problems:
        raise ValueError(
            "SpeculatorConfig.adapt_resources returned resources the model cannot "
            "run: " + "; ".join(problems)
        )
