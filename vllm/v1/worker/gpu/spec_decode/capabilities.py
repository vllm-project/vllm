# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the target model can verify: per-layer spec capabilities.

A speculator declares the plans it wants to verify (``SpecResources.plan_kind``).
Each worker walks its layers after the attention backends are built and reports
the highest plan kind every layer verifies correctly, with a reason for each
layer or setting that lowers it (``SpecCapabilities``). The engine core reduces
the reports of every rank, lets the speculator adapt its resources to them
(``SpeculatorConfig.adapt_resources``) and hands the result to the scheduler and
to every worker. A model whose layers cannot verify trees therefore runs a tree
speculator as a chain; only a layer that cannot verify a chain at all (plan kind
``"none"``) refuses speculative decoding.
"""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.kernels import KernelRegistry
    from vllm.v1.worker.gpu.spec_decode.round import SpecResources
    from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig
    from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

PlanKind = Literal["none", "chain", "tree", "mask"]
"""The plans a model verifies, in increasing order: ``"none"`` (no speculative
decoding), ``"chain"`` (each row follows the one before it), ``"tree"`` (each
row names its parent row and sees its ancestors), ``"mask"`` (a tree whose rows
may also carry their own visibility mask, ``VerifyPlan.mask``)."""

PLAN_KINDS: tuple[PlanKind, ...] = get_args(PlanKind)


def plan_kind_rank(kind: PlanKind) -> int:
    """The position of ``kind`` in :data:`PLAN_KINDS`."""
    return PLAN_KINDS.index(kind)


def min_plan_kind(kinds: Iterable[PlanKind]) -> PlanKind:
    """The lowest of ``kinds``; the highest plan kind when there are none."""
    return min(kinds, key=plan_kind_rank, default=PLAN_KINDS[-1])


@dataclass(frozen=True)
class SpecLayerKind:
    """What a layer is to speculative decoding, derived from the KV cache spec
    or the layer class (a plugin cannot invent one)."""

    family: Literal["attn", "mamba", "layout"]
    variant: str
    """attn: the ``KVCacheSpecKind`` value; mamba: the layer class's
    ``spec_layer_variant``; layout: what a commit moves,
    ``"paged_token"`` (KV of one tensor per layer, any element size),
    ``"paged_split_scale"`` (KV with separate scales: nvfp4, per-token-head) or
    ``"mamba_state"``."""
    state_mode: str = ""
    """mamba and layout ``mamba_state`` only: how drafts keep recurrent state,
    ``"slots"``, ``"replay"`` or ``"recover"``."""

    def key(self) -> str:
        suffix = f"@{self.state_mode}" if self.state_mode else ""
        return f"{self.family}:{self.variant}{suffix}"


@dataclass(frozen=True)
class SpecLimit:
    """A layer kind or a setting that caps the plan kind, and why."""

    plan_kind: PlanKind
    reason: str


@dataclass(frozen=True)
class SpecCapabilities:
    """What a worker's layers verify; reduced over ranks by the engine core."""

    plan_kind: PlanKind
    """The lowest plan kind any layer or setting allows."""
    limits: tuple[SpecLimit, ...] = ()
    """Every cap below the highest plan kind, for the startup log."""

    @classmethod
    def from_limits(cls, limits: Iterable[SpecLimit]) -> "SpecCapabilities":
        limits = tuple(dict.fromkeys(limits))
        return cls(
            plan_kind=min_plan_kind(limit.plan_kind for limit in limits),
            limits=limits,
        )


def reduce_spec_capabilities(
    capabilities: Sequence[SpecCapabilities],
) -> SpecCapabilities:
    """Combine the reports of every rank: pipeline-parallel ranks hold
    different layers, so the model verifies the lowest plan kind of any rank."""
    return SpecCapabilities.from_limits(
        limit for caps in capabilities for limit in caps.limits
    )


def spec_layers(
    attn_groups: Iterable[Iterable["AttentionGroup"]], layers: Mapping[str, Any]
) -> Iterator[tuple["AttentionGroup", str, Any, Any]]:
    """``(group, name, spec, layer)`` for each target layer a speculative round
    verifies: its attention group, KV cache spec and module. Draft layers,
    hidden-state caches and encoder or cross attention (whose rows do not
    attend to each other) are not verified.
    """
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


def mamba_state_mode(cache_config: Any) -> str:
    """How recurrent layers keep the drafts' states."""
    if cache_config.use_replayssm:
        return "replay"
    if cache_config.use_kda_recoverssm:
        return "recover"
    return "slots"


def layer_spec_limits(
    attn_groups: Iterable[Iterable["AttentionGroup"]],
    layers: Mapping[str, Any],
    mamba_state_mode: str,
    kernels: "KernelRegistry | None" = None,
) -> list[SpecLimit]:
    """The plan kind caps of the target's layers, one per (layer kind, reason)
    with its layer count.

    A layer verifies what the spec kernels it needs support for it
    (``KernelRegistry.plan_kind``; ``kernels`` defaults to the reference
    kernels). An attention layer is further capped by what its metadata
    builder's ``spec_verify_limit`` returns for the layer's features. A
    recurrent layer (``MambaBase``) verifies nothing when its class sets
    ``spec_chain_supported = False`` and chains when it declares no
    ``spec_layer_variant``. Only the layers of :func:`spec_layers` count.
    """
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.worker.gpu.spec_decode.kernels import KernelRegistry, SpecLayerSpec

    if kernels is None:
        kernels = KernelRegistry()
    # (plan kind, what the layers are, why) -> number of layers
    num_layers: Counter[tuple[PlanKind, str, str]] = Counter()
    for group, name, spec, layer in spec_layers(attn_groups, layers):
        if isinstance(layer, MambaBase):
            num_layers[_mamba_layer_limit(name, layer, kernels, mamba_state_mode)] += 1
            continue
        kind = attn_layer_kind(spec)
        plan_kind, why = group.metadata_builders[0].spec_verify_limit(
            **attn_layer_features(layer, spec)
        )
        kernel_kind, kernel_why = kernels.plan_kind(
            SpecLayerSpec(name, kind, layer, layout=attn_layout(layer, spec))
        )
        if plan_kind_rank(kernel_kind) < plan_kind_rank(plan_kind):
            plan_kind, why = kernel_kind, kernel_why
        num_layers[(plan_kind, kind.key(), why)] += 1
    return [
        SpecLimit(plan_kind, f"{what}, {n} layer{'s' * (n != 1)}: {why}")
        for (plan_kind, what, why), n in num_layers.items()
        if plan_kind != PLAN_KINDS[-1]
    ]


def _mamba_layer_limit(
    name: str, layer: Any, kernels: "KernelRegistry", state_mode: str
) -> tuple[PlanKind, str, str]:
    from vllm.v1.worker.gpu.spec_decode.kernels import SpecLayerSpec

    layer_cls = type(layer)
    if not layer_cls.spec_chain_supported:
        return ("none", layer_cls.__name__, "decodes one token per request")
    variant = layer_cls.spec_layer_variant
    if variant is None:
        return ("chain", layer_cls.__name__, "declares no spec_layer_variant")
    # The state mode is part of the key: the slots kernels read a state slot
    # per draft row, which replay and recovery do not keep.
    kind = SpecLayerKind("mamba", variant, state_mode)
    plan_kind, why = kernels.plan_kind(
        SpecLayerSpec(name, kind, layer, layout="mamba_state")
    )
    if plan_kind == "tree":
        # A row's state folds in every ancestor along parent, so a recurrent
        # layer cannot hide one: masks lower to trees unless its verify kernel
        # itself verifies masks.
        why += " (a recurrent state folds in every ancestor: no masks)"
    return (plan_kind, kind.key(), why)


def attn_layer_kind(spec: Any) -> SpecLayerKind:
    """The kind of an attention layer with KV cache spec ``spec``."""
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    return SpecLayerKind("attn", get_kv_cache_spec_kind(spec).value)


def attn_layout(layer: Any, spec: Any) -> str:
    """What committing an attention layer's accepted path moves: ``""`` for a
    cache not paged by slot (its model keeps its own commit),
    ``"paged_split_scale"`` for KV quantised with separate scales, else
    ``"paged_token"``."""
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    if not getattr(spec, "uses_slot_mapping", True):
        return ""
    mode = get_kv_quant_mode(str(getattr(layer, "kv_cache_dtype", "auto")))
    if mode == KVQuantMode.NVFP4 or mode.is_per_token_head:
        return "paged_split_scale"
    return "paged_token"


def attn_layer_features(layer: Any, spec: Any) -> dict[str, Any]:
    """The keyword arguments of ``AttentionMetadataBuilder.spec_verify_limit``
    for an attention layer with KV cache spec ``spec``. Features the layer does
    not have are None or False (e.g. a soft cap of 0 is no soft cap)."""
    from vllm.v1.kv_cache_interface import SinkFullAttentionSpec

    impl = getattr(layer, "impl", None)
    return dict(
        kv_cache_dtype=getattr(layer, "kv_cache_dtype", "auto"),
        sliding_window=getattr(spec, "sliding_window", None)
        or getattr(layer, "sliding_window", None),
        has_sinks=bool(getattr(layer, "has_sink", False))
        or isinstance(spec, SinkFullAttentionSpec),
        logits_soft_cap=getattr(impl, "logits_soft_cap", None) or None,
        has_alibi=getattr(impl, "alibi_slopes", None) is not None,
        chunk_size=getattr(spec, "attention_chunk_size", None),
        rswa_window=getattr(spec, "rswa_window", None),
        kv_sharing_target=getattr(layer, "kv_sharing_target_layer_name", None),
    )


def resolve_spec_resources(
    speculator_config: "SpeculatorConfig",
    declared: "SpecResources",
    capabilities: Sequence[SpecCapabilities],
) -> "SpecResources":
    """The resources speculation runs with: the speculator's declaration adapted
    to what every rank verifies (``capabilities``, one per worker).

    Logs one line per cap that lowers the declared plan kind. Raises when a
    layer verifies no plans at all, and when the adapted resources exceed the
    declared ones or the capabilities. A speculator that must not run with a
    lower plan kind raises in its ``adapt_resources``.
    """
    caps = reduce_spec_capabilities(capabilities)
    lowering = [
        limit
        for limit in caps.limits
        if plan_kind_rank(limit.plan_kind) < plan_kind_rank(declared.plan_kind)
    ]
    if caps.plan_kind == "none":
        raise ValueError(
            "Speculative decoding is not supported by this model: "
            + "; ".join(limit.reason for limit in lowering)
        )
    for limit in lowering:
        logger.info(
            "spec decode: %s -> %s (%s)",
            declared.plan_kind,
            limit.plan_kind,
            limit.reason,
        )
    effective = speculator_config.adapt_resources(declared, caps)
    validate_effective_resources(declared, effective, caps)
    return effective


def validate_effective_resources(
    declared: "SpecResources", effective: "SpecResources", caps: SpecCapabilities
) -> None:
    """Check ``SpeculatorConfig.adapt_resources``: memory and graphs are sized
    for ``declared``, so ``effective`` may only take less of them, and its plan
    kind must be one every layer verifies."""
    problems = []
    if plan_kind_rank(effective.plan_kind) > plan_kind_rank(
        min_plan_kind((declared.plan_kind, caps.plan_kind))
    ):
        problems.append(
            f"plan_kind {effective.plan_kind!r} is above the declared "
            f"{declared.plan_kind!r} or the model's {caps.plan_kind!r}"
        )
    if effective.max_candidates > declared.max_candidates:
        problems.append(
            f"max_candidates {effective.max_candidates} > declared "
            f"{declared.max_candidates}"
        )
    extra_shapes = set(effective.capture_shapes) - set(declared.capture_shapes)
    if extra_shapes:
        problems.append(f"capture_shapes {sorted(extra_shapes, key=str)} not declared")
    if problems:
        raise ValueError(
            "SpeculatorConfig.adapt_resources returned resources the model cannot "
            "run: " + "; ".join(problems)
        )
