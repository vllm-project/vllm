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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.round import SpecResources
    from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig
    from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

PlanKind = Literal["none", "chain", "tree"]
"""The plans a model verifies, in increasing order: ``"none"`` (no speculative
decoding), ``"chain"`` (each row follows the one before it), ``"tree"`` (each
row names its parent row)."""

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

    family: Literal["attn", "mamba", "indexer"]
    variant: str
    """attn: the ``KVCacheSpecKind`` value; mamba: the layer class's
    ``spec_layer_variant``; indexer: ``"dsa"``."""
    state_mode: str = ""
    """mamba only: how drafts keep recurrent state, ``"slots"``, ``"replay"``
    or ``"recover"``."""

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
    tree_requires_uniform_rows: bool
    """Whether a layer's builder only takes batches whose requests have the same
    number of query rows. Binds only when the effective plan kind is ``"tree"``
    or higher: a variable-length chain runs on its prefill path."""
    limits: tuple[SpecLimit, ...] = ()
    """Every cap below the highest plan kind, for the startup log."""

    @classmethod
    def from_limits(
        cls, limits: Iterable[SpecLimit], tree_requires_uniform_rows: bool
    ) -> "SpecCapabilities":
        limits = tuple(dict.fromkeys(limits))
        return cls(
            plan_kind=min_plan_kind(limit.plan_kind for limit in limits),
            tree_requires_uniform_rows=tree_requires_uniform_rows,
            limits=limits,
        )


def reduce_spec_capabilities(
    capabilities: Sequence[SpecCapabilities],
) -> SpecCapabilities:
    """Combine the reports of every rank: pipeline-parallel ranks hold
    different layers, so the model verifies the lowest plan kind of any rank."""
    return SpecCapabilities.from_limits(
        (limit for caps in capabilities for limit in caps.limits),
        tree_requires_uniform_rows=any(
            caps.tree_requires_uniform_rows for caps in capabilities
        ),
    )


def layer_spec_limits(
    attn_groups: Iterable[Iterable["AttentionGroup"]],
    layers: Mapping[str, Any],
    mamba_state_mode: str,
    excluded_layer_names: frozenset[str] = frozenset(),
) -> tuple[list[SpecLimit], bool]:
    """The plan kind caps of the target's layers, one per (layer kind, reason)
    with its layer count, and whether a layer's builder needs uniform rows.

    An attention layer verifies what its metadata builder's
    ``spec_verify_plan_kind`` returns for the layer's features. A recurrent layer
    (``MambaBase``) verifies nothing when its class sets
    ``spec_chain_supported = False``, chains when it declares no
    ``spec_layer_variant`` or keeps its state by replay or recovery, and
    otherwise what its builder verifies. Draft layers (and
    ``excluded_layer_names``), hidden-state caches and encoder or cross
    attention (whose rows do not attend to each other) do not count.
    """
    from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.kv_cache_interface import (
        CrossAttentionSpec,
        EncoderOnlyAttentionSpec,
        HiddenStateCacheSpec,
        MLAAttentionSpec,
        SparseCacheRole,
        UniformTypeKVCacheSpecs,
        get_kv_cache_spec_kind,
    )

    # (plan kind, what the layers are, why) -> number of layers
    num_layers: Counter[tuple[PlanKind, str, str]] = Counter()
    tree_requires_uniform_rows = False
    for groups in attn_groups:
        for group in groups:
            builder = group.metadata_builders[0]
            for name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                if (
                    name in excluded_layer_names
                    or getattr(spec, "is_draft", False)
                    or isinstance(
                        spec,
                        (
                            HiddenStateCacheSpec,
                            EncoderOnlyAttentionSpec,
                            CrossAttentionSpec,
                        ),
                    )
                ):
                    continue
                layer = layers.get(name)
                if isinstance(layer, MambaBase):
                    num_layers[
                        _mamba_layer_limit(layer, builder, mamba_state_mode)
                    ] += 1
                    continue
                if (
                    isinstance(spec, MLAAttentionSpec)
                    and spec.cache_role == SparseCacheRole.INDEXER
                ):
                    kind = SpecLayerKind("indexer", "dsa")
                else:
                    kind = SpecLayerKind("attn", get_kv_cache_spec_kind(spec).value)
                plan_kind = builder.spec_verify_plan_kind(
                    **attn_layer_features(layer, spec)
                )
                why = f"{type(builder).__name__} verifies {plan_kind}"
                num_layers[(plan_kind, kind.key(), why)] += 1
                if (
                    getattr(builder, "query_len_support", None)
                    == QueryLenSupport.UNIFORM
                ):
                    tree_requires_uniform_rows = True
    limits = [
        SpecLimit(plan_kind, f"{what}, {n} layer{'s' * (n != 1)}: {why}")
        for (plan_kind, what, why), n in num_layers.items()
        if plan_kind != PLAN_KINDS[-1]
    ]
    return limits, tree_requires_uniform_rows


def _mamba_layer_limit(
    layer: Any, builder: Any, state_mode: str
) -> tuple[PlanKind, str, str]:
    layer_cls = type(layer)
    if not layer_cls.spec_chain_supported:
        return ("none", layer_cls.__name__, "decodes one token per request")
    variant = layer_cls.spec_layer_variant
    if variant is None:
        return ("chain", layer_cls.__name__, "declares no spec_layer_variant")
    what = SpecLayerKind("mamba", variant, state_mode).key()
    if state_mode != "slots":
        # The tree kernels read a state slot per draft row, which replay and
        # recovery do not keep.
        return ("chain", what, "trees need a state slot per draft row")
    plan_kind = builder.spec_verify_plan_kind(**NO_ATTN_FEATURES)
    return (plan_kind, what, f"{type(builder).__name__} verifies {plan_kind}")


NO_ATTN_FEATURES: dict[str, Any] = dict(
    kv_cache_dtype="auto",
    sliding_window=None,
    has_sinks=False,
    logits_soft_cap=None,
    has_alibi=False,
    chunk_size=None,
    rswa_window=None,
    kv_sharing_target=None,
)
"""``spec_verify_plan_kind`` arguments of a layer without attention features."""


def attn_layer_features(layer: Any, spec: Any) -> dict[str, Any]:
    """The keyword arguments of ``AttentionMetadataBuilder.spec_verify_plan_kind``
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
    if (
        plan_kind_rank(effective.plan_kind) >= plan_kind_rank("tree")
        and caps.tree_requires_uniform_rows
        and effective.varlen
    ):
        problems.append(
            "a variable-length capture shape with trees on a builder that needs "
            "uniform rows"
        )
    if problems:
        raise ValueError(
            "SpeculatorConfig.adapt_resources returned resources the model cannot "
            "run: " + "; ".join(problems)
        )
