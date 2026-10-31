# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer spec capabilities: what a model's layers verify, and a tree
speculator downgraded to chains on a model whose layers cannot verify trees.

Each model is its layer mix: the real layer classes (built without weights) and
KV cache specs, behind the metadata builders their backends use. Attention
builders are stand-ins: the stock one verifies chains (the default), and
``_TreeBuilder`` is a tree-capable builder with no window, sink, soft cap,
ALiBi, chunk or RSWA term, as a plugin's tree attention would be.
"""

import importlib
import importlib.util
import logging
from types import SimpleNamespace

import pytest
import torch

from tests.v1.spec_decode.utils import fake_speculative_config
from vllm.config import CacheConfig, ParallelConfig
from vllm.model_executor.layers.attention.mla_attention import QueryLenSupport
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
    OlmoHybridGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
    ShortConvAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    RSWASpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu.spec_decode.capabilities import (
    SpecCapabilities,
    SpecLimit,
    layer_spec_limits,
    reduce_spec_capabilities,
    resolve_spec_resources,
)
from vllm.v1.worker.gpu.spec_decode.commit import KvCompactor
from vllm.v1.worker.gpu.spec_decode.kernels import KernelContext, KernelRegistry
from vllm.v1.worker.gpu.spec_decode.round import PlanShape, SpecResources
from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound, spec_kernel_layers
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

# --- builders ----------------------------------------------------------------


class _StockBuilder(AttentionMetadataBuilder):
    """An attention builder that keeps the default: chains only."""

    def __init__(self):
        pass

    def build(self, *args, **kwargs):
        raise NotImplementedError


class _TreeBuilder(_StockBuilder):
    """A tree attention builder that verifies trees with none of the features."""

    def spec_verify_limit(self, **features):
        plain = not any(
            features[name]
            for name in (
                "sliding_window",
                "has_sinks",
                "logits_soft_cap",
                "has_alibi",
                "chunk_size",
                "rswa_window",
            )
        )
        kind = "tree" if plain else "chain"
        return kind, f"{type(self).__name__} verifies {kind}"


class _UniformMLABuilder(_StockBuilder):
    """An MLA decode builder (FlashMLA-like) that takes uniform rows only."""

    query_len_support = QueryLenSupport.UNIFORM


def _builder(cls):
    return object.__new__(cls)


# --- layers and specs --------------------------------------------------------

_FULL = FullAttentionSpec(
    block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
)
_SWA = SlidingWindowSpec(
    block_size=16,
    num_kv_heads=2,
    head_size=64,
    dtype=torch.bfloat16,
    sliding_window=128,
)
_MAMBA = MambaSpec(block_size=16, shapes=((4, 8),), dtypes=(torch.float32,))


def _attn(**impl):
    """An attention layer: ``kv_cache_dtype``, ``has_sink``, ``sliding_window`` on
    the layer, soft cap and ALiBi on its impl."""
    has_sink = impl.pop("has_sink", False)
    return SimpleNamespace(
        kv_cache_dtype="auto",
        has_sink=has_sink,
        sliding_window=impl.pop("sliding_window", None),
        kv_sharing_target_layer_name=None,
        impl=SimpleNamespace(
            logits_soft_cap=impl.get("logits_soft_cap"),
            alibi_slopes=impl.get("alibi_slopes"),
        ),
    )


def _mamba(layer_cls):
    return object.__new__(layer_cls)


def _group(builder, spec, layers):
    return SimpleNamespace(
        metadata_builders=[_builder(builder)],
        kv_cache_spec=spec,
        layer_names=list(layers),
    )


def _model(*groups):
    """``(attn_groups, layers)`` from ``(builder class, spec, {name: layer})``."""
    layers = {}
    attn_groups = []
    for builder, spec, group_layers in groups:
        attn_groups.append([_group(builder, spec, group_layers)])
        layers.update(group_layers)
    return attn_groups, layers


def _caps(model, state_mode="slots") -> SpecCapabilities:
    attn_groups, layers = model
    return SpecCapabilities.from_limits(
        layer_spec_limits(attn_groups, layers, state_mode)
    )


def _layers(prefix, n, make):
    return {f"{prefix}.{i}": make() for i in range(n)}


def _llama(builder):
    return _model((builder, _FULL, _layers("attn", 4, _attn)))


def _gpt_oss():
    # Alternating sliding-window and full layers, all with attention sinks.
    return _model(
        (_TreeBuilder, _SWA, _layers("swa", 2, lambda: _attn(has_sink=True))),
        (_TreeBuilder, _FULL, _layers("full", 2, lambda: _attn(has_sink=True))),
    )


def _gemma2():
    softcap = lambda: _attn(logits_soft_cap=50.0)  # noqa: E731
    return _model(
        (_TreeBuilder, _SWA, _layers("swa", 2, softcap)),
        (_TreeBuilder, _FULL, _layers("full", 2, softcap)),
    )


def _rswa():
    rswa = RSWASpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        rswa_window=64,
    )
    return _model((_TreeBuilder, rswa, _layers("rswa", 2, _attn)))


def _dsv3():
    mla = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    return _model((_UniformMLABuilder, mla, _layers("mla", 3, _attn)))


def _hybrid(layer_cls, builder, attn_builder=_TreeBuilder):
    return _model(
        (builder, _MAMBA, _layers("linear", 3, lambda: _mamba(layer_cls))),
        (attn_builder, _FULL, _layers("attn", 1, _attn)),
    )


def _import_class(qualname):
    module, name = qualname.rsplit(":", 1)
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as e:
        pytest.skip(f"{qualname} does not import here: {e}")


# --- per-model plan kinds ----------------------------------------------------


def _chain_limits(caps):
    """The limits that cap a model at chains (or less): tree-capable layers
    report a "tree" limit too, as masks lower to trees on them."""
    return [limit for limit in caps.limits if limit.plan_kind != "tree"]


def test_llama_verifies_trees_on_a_tree_builder_and_chains_on_the_stock_one():
    tree = _caps(_llama(_TreeBuilder))
    assert tree.plan_kind == "tree"
    # A tree is a limit now that plans can carry masks.
    assert tree.limits == (
        SpecLimit("tree", "attn:full_attention, 4 layers: _TreeBuilder verifies tree"),
    )
    stock = _caps(_llama(_StockBuilder))
    assert stock.plan_kind == "chain"
    assert stock.limits == (
        SpecLimit(
            "chain", "attn:full_attention, 4 layers: _StockBuilder verifies chain"
        ),
    )


@pytest.mark.parametrize(
    "model, kinds",
    [
        (_gpt_oss, ["attn:sliding_window, 2 layers", "attn:full_attention, 2 layers"]),
        (_gemma2, ["attn:sliding_window, 2 layers", "attn:full_attention, 2 layers"]),
        (_rswa, ["attn:full_attention, 2 layers"]),
    ],
    ids=["gpt-oss", "gemma2", "rswa"],
)
def test_attention_features_without_a_tree_term_verify_chains(model, kinds):
    caps = _caps(model())
    assert caps.plan_kind == "chain"
    assert [limit.reason for limit in caps.limits] == [
        f"{kind}: _TreeBuilder verifies chain" for kind in kinds
    ]


def test_the_builder_sees_each_layers_features():
    seen = []

    class Recording(_StockBuilder):
        def spec_verify_limit(self, **features):
            seen.append(features)
            return "tree", "verifies tree"

    sink_swa = _attn(has_sink=True, logits_soft_cap=30.0, alibi_slopes=torch.ones(2))
    rswa = RSWASpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        rswa_window=64,
    )
    _caps(_model((Recording, _SWA, {"a": sink_swa}), (Recording, rswa, {"b": _attn()})))
    assert seen == [
        dict(
            kv_cache_dtype="auto",
            sliding_window=128,
            has_sinks=True,
            logits_soft_cap=30.0,
            has_alibi=True,
            chunk_size=None,
            rswa_window=None,
            kv_sharing_target=None,
        ),
        dict(
            kv_cache_dtype="auto",
            sliding_window=None,
            has_sinks=False,
            logits_soft_cap=None,
            has_alibi=False,
            chunk_size=None,
            rswa_window=64,
            kv_sharing_target=None,
        ),
    ]


def test_dsv3_mla_verifies_chains():
    caps = _caps(_dsv3())
    assert caps.plan_kind == "chain"
    # A variable-length chain still runs.
    declared = SpecResources(
        max_candidates=4,
        capture_shapes=(PlanShape(4), PlanShape(None)),
    )
    assert resolve_spec_resources(_TreeConfig(), declared, [caps]) == declared


def test_nemotron_h_verifies_trees():
    caps = _caps(_hybrid(MambaMixer2, Mamba2AttentionMetadataBuilder))
    assert caps.plan_kind == "tree", caps.limits


def test_qwen_gdn_verifies_trees_with_state_slots_and_chains_by_replay():
    model = _hybrid(QwenGatedDeltaNetAttention, GDNAttentionMetadataBuilder)
    assert _caps(model).plan_kind == "tree"
    for mode in ("replay", "recover"):
        caps = _caps(model, state_mode=mode)
        assert caps.plan_kind == "chain"
        assert caps.limits[0].reason.startswith(f"mamba:gdn@{mode}, 3 layers")


@pytest.mark.parametrize(
    "layer, builder",
    [
        (OlmoHybridGatedDeltaNetAttention, GDNAttentionMetadataBuilder),
        (KimiGatedDeltaNetAttention, GDNAttentionMetadataBuilder),  # Kimi Linear
        # Qwen4Exp's PLE layer reports SHORT_CONV but runs its own layer.
        (
            "vllm.models.qwen4_exp.nvidia.ple_layer:Qwen4ExpPLELayer",
            PleShortConvAttentionMetadataBuilder,
        ),
        (
            "vllm.models.kimi_k3.nvidia.kda:KimiK3DeltaAttention",
            GDNAttentionMetadataBuilder,
        ),
        (
            "vllm.models.kimi_k3.amd.kda:KimiK3DeltaAttention",
            "vllm.models.kimi_k3.amd.kda_metadata:KimiK3ROCmKDAMetadataBuilder",
        ),
        (ShortConv, ShortConvAttentionMetadataBuilder),
    ],
    ids=[
        "olmo-hybrid",
        "kimi-linear",
        "qwen4exp-ple",
        "kimi-k3-nvidia",
        "kimi-k3-amd",
        "lfm2-short-conv",
    ],
)
def test_recurrent_layers_without_tree_kernels_verify_chains(layer, builder):
    """A layer subclassing a tree-capable one (and a builder subclassing a
    tree-capable builder) runs its own kernels: chains only."""
    if isinstance(layer, str):
        layer = _import_class(layer)
    if isinstance(builder, str):
        builder = _import_class(builder)
    caps = _caps(_hybrid(layer, builder))
    assert caps.plan_kind == "chain"
    [limit] = _chain_limits(caps)
    assert limit.reason in (
        f"{layer.__name__}, 3 layers: declares no spec_layer_variant",
        "mamba:short_conv@slots, 3 layers: no mamba:short_conv@slots/verify kernel",
    )


def test_jamba_mamba1_refuses_spec_decode():
    caps = _caps(_hybrid(MambaMixer, Mamba1AttentionMetadataBuilder))
    assert caps.plan_kind == "none"
    declared = _TreeConfig().resources(8)
    with pytest.raises(ValueError, match="MambaMixer, 3 layers: decodes one token"):
        resolve_spec_resources(_TreeConfig(), declared, [caps])


def test_draft_and_hidden_state_layers_do_not_count():
    hidden = HiddenStateCacheSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    draft = _attn()
    draft.is_draft_layer = True
    model = _model(
        (_TreeBuilder, _FULL, _layers("attn", 2, _attn)),
        (_StockBuilder, hidden, {"hidden": _attn()}),
        (_StockBuilder, _FULL, {"draft": draft}),
    )
    assert _caps(model).plan_kind == "tree"


# --- MambaBase class attributes ----------------------------------------------


class _OwnKernels(QwenGatedDeltaNetAttention):
    """A GDN subclass running kernels of its own."""


class _OwnTreeKernels(_OwnKernels):
    spec_layer_variant = "gdn"


def test_spec_layer_variant_is_not_inherited():
    assert QwenGatedDeltaNetAttention.spec_layer_variant == "gdn"
    assert MambaMixer2.spec_layer_variant == "mamba2"
    assert ShortConv.spec_layer_variant == "short_conv"
    assert _OwnTreeKernels.spec_layer_variant == "gdn"
    for chain_only in (
        MambaBase,
        GatedDeltaNetAttention,
        _OwnKernels,
        OlmoHybridGatedDeltaNetAttention,
        KimiGatedDeltaNetAttention,
        MambaMixer,
    ):
        assert chain_only.spec_layer_variant is None, chain_only


def test_only_mamba1_declares_no_chain():
    assert not MambaMixer.spec_chain_supported
    for cls in (MambaMixer2, QwenGatedDeltaNetAttention, ShortConv, _OwnKernels):
        assert cls.spec_chain_supported, cls


# --- runner: settings lower to chain -----------------------------------------


class _TreeState:
    supports_tree_verify = True


class _ChainState:
    supports_tree_verify = False


def _runner(**overrides):
    attn_groups, layers = _llama(_TreeBuilder)
    runner = SimpleNamespace(
        cache_config=CacheConfig(),
        attn_groups=attn_groups,
        compilation_config=SimpleNamespace(static_forward_context=layers),
        speculator=SimpleNamespace(draft_attn_layer_names=None),
        use_pp=False,
        pcp_manager=None,
        dcp_size=1,
        batch_sharder=None,
        parallel_config=ParallelConfig(),
        model_state=_TreeState(),
    )
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def _resolve(runner) -> SpecCapabilities:
    spec_round = object.__new__(SpecRound)
    spec_round.kernels = KernelRegistry()
    return spec_round.resolve_capabilities(runner)


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"use_pp": True}, "pipeline parallelism (no tree path broadcast)"),
        ({"pcp_manager": object()}, "prefill context parallelism"),
        ({"dcp_size": 2}, "decode context parallelism"),
        ({"batch_sharder": object()}, "batch-sharded sampling"),
        ({"parallel_config": ParallelConfig(enable_dbo=True)}, "micro-batching"),
        ({"model_state": _ChainState()}, "model state _ChainState"),
    ],
)
def test_runner_settings_lower_trees_to_chains(override, reason):
    default = _resolve(_runner())
    assert default.plan_kind == "tree"
    caps = _resolve(_runner(**override))
    assert caps.plan_kind == "chain"
    assert [limit.reason for limit in _chain_limits(caps)] == [
        f"{reason}: keeps verify rows in chain order"
    ]


def test_runner_reads_the_mamba_state_mode():
    attn_groups, layers = _hybrid(
        QwenGatedDeltaNetAttention, GDNAttentionMetadataBuilder
    )
    runner = _runner(
        attn_groups=attn_groups,
        compilation_config=SimpleNamespace(static_forward_context=layers),
        cache_config=CacheConfig(use_replayssm=True),
    )
    caps = _resolve(runner)
    assert caps.plan_kind == "chain"
    assert caps.limits[0].reason.startswith("mamba:gdn@replay")


# --- engine core: reduce, adapt, validate ------------------------------------


def _spec():
    return fake_speculative_config(
        SpeculatorConfig,
        num_speculative_tokens=3,
        enable_adaptive_verification=False,
        num_speculative_tokens_per_batch_size=None,
        uses_dynamic_speculative_decoding=lambda: False,
    )


class _TreeConfig(SpeculatorConfig):
    """A tree speculator: 2-way branching, a uniform and a varlen shape."""

    def __init__(self):
        super().__init__(_spec())

    def resources(self, max_num_seqs):
        return SpecResources(
            max_candidates=4,
            capture_shapes=(PlanShape(4), PlanShape(None)),
            max_branching=2,
        )


def _tree_caps():
    return SpecCapabilities.from_limits([])


def _chain_caps(reason="attn:sliding_window, 2 layers: X verifies chain"):
    return SpecCapabilities.from_limits([SpecLimit("chain", reason)])


def test_tree_speculator_runs_as_a_chain_with_one_log_line_per_reason(caplog):
    config = _TreeConfig()
    declared = config.resources(8)
    assert declared.plan_kind == "tree"
    caps = SpecCapabilities.from_limits(
        [SpecLimit("chain", "a: x"), SpecLimit("chain", "b: y")]
    )
    with caplog.at_level(logging.INFO, logger="vllm"):
        effective = resolve_spec_resources(config, declared, [caps])
    assert effective.plan_kind == "chain"
    assert effective.max_branching == 1
    assert effective.max_candidates == declared.max_candidates
    assert effective.capture_shapes == declared.capture_shapes
    lines = [r.getMessage() for r in caplog.records if "spec decode" in r.getMessage()]
    assert len(lines) == 2
    assert "(a: x)" in lines[0] and "(b: y)" in lines[1]


def test_tree_on_a_model_that_verifies_trees_is_kept():
    config = _TreeConfig()
    declared = config.resources(8)
    assert resolve_spec_resources(config, declared, [_tree_caps()]) is declared


def test_a_chain_speculator_is_not_lowered_nor_logged(caplog):
    config = _spec().speculator_config
    declared = config.resources(8)
    with caplog.at_level(logging.INFO, logger="vllm"):
        assert resolve_spec_resources(config, declared, [_chain_caps()]) == declared
    assert not [r for r in caplog.records if "spec decode" in r.getMessage()]


class _Overreach(_TreeConfig):
    def __init__(self, **changes):
        super().__init__()
        self.changes = changes

    def adapt_resources(self, declared, caps):
        from dataclasses import replace

        return replace(declared, **self.changes)


@pytest.mark.parametrize(
    "changes, caps, problem",
    [
        ({}, _chain_caps(), "plan_kind 'tree' is above"),
        ({"max_branching": 1, "max_candidates": 5}, _chain_caps(), "max_candidates 5"),
        (
            {"max_branching": 1, "capture_shapes": (PlanShape(2),)},
            _chain_caps(),
            "capture_shapes",
        ),
    ],
    ids=["plan-kind", "candidates", "shapes"],
)
def test_adapt_resources_is_validated(changes, caps, problem):
    config = _Overreach(**changes)
    with pytest.raises(ValueError, match=problem):
        resolve_spec_resources(config, config.resources(8), [caps])


def test_pipeline_ranks_reduce_to_the_lowest_plan_kind():
    """PP ranks hold different layers: rank 0 the trees, rank 1 a SWA layer."""
    rank0 = _caps(_llama(_TreeBuilder))
    rank1 = _caps(_model((_UniformMLABuilder, _SWA, {"swa.9": _attn()})))
    caps = reduce_spec_capabilities([rank0, rank1])
    assert caps.plan_kind == "chain"
    assert caps.limits == rank0.limits + rank1.limits
    assert reduce_spec_capabilities([rank0, rank0]).plan_kind == "tree"
    none = _caps(_hybrid(MambaMixer, Mamba1AttentionMetadataBuilder))
    assert reduce_spec_capabilities([rank0, none, rank1]).plan_kind == "none"


class _Executor:
    def __init__(self, reports):
        self.reports = reports
        self.calls = []

    def collective_rpc(self, method, args=()):
        self.calls.append((method, args))
        if method == "get_spec_capabilities":
            return self.reports
        return [None] * len(self.reports)


def _engine(reports):
    from vllm.v1.engine.core import EngineCore

    engine = object.__new__(EngineCore)
    engine.model_executor = _Executor(reports)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(speculator_config=_TreeConfig()),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    return engine, config


def test_engine_core_sets_the_reduced_resources_on_every_worker():
    from vllm.v1.engine.core import EngineCore

    # PP rank 1 holds a layer that verifies chains only.
    engine, config = _engine([_tree_caps(), _chain_caps()])
    EngineCore._set_spec_resources(engine, config)
    assert engine.spec_resources.plan_kind == "chain"
    assert engine.model_executor.calls[-1] == (
        "set_spec_resources",
        (engine.spec_resources,),
    )
    # Workers that do not report (V1 runner, other platforms): declared, no RPC.
    engine, config = _engine([None, None])
    EngineCore._set_spec_resources(engine, config)
    assert engine.spec_resources is None
    assert [method for method, _ in engine.model_executor.calls] == [
        "get_spec_capabilities"
    ]


# --- compactor ---------------------------------------------------------------


def test_compactor_skips_ring_shared_and_hidden_state_caches():
    """The DSv4 ring (not paged by slot) keeps its own chain commit, a layer
    sharing another's KV cache moves with it, and a hidden-state cache is not a
    decoder layer: none is compacted on its own."""

    def spec(cls, **kwargs):
        return cls(
            block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16, **kwargs
        )

    specs = {
        "attn": spec(FullAttentionSpec),
        "ring": spec(CircularBufferSpec),
        "hidden": spec(HiddenStateCacheSpec),
        "shared": spec(FullAttentionSpec),
    }
    runner = SimpleNamespace(device=torch.device("cpu"))
    cache = torch.zeros(4, 2, 16, 1, 64, dtype=torch.bfloat16)
    # Only the paged layer has a cache the compactor can read.
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            "attn": SimpleNamespace(kv_cache=cache),
            "shared": SimpleNamespace(
                kv_cache=cache, kv_sharing_target_layer_name="attn"
            ),
        }
    )
    runner.kv_cache_config = SimpleNamespace(num_blocks=4)
    runner.attn_groups = [
        [
            SimpleNamespace(
                layer_names=[name], kv_cache_spec=group_spec, kv_cache_group_id=gid
            )
        ]
        for gid, (name, group_spec) in enumerate(specs.items())
    ]
    runner.kernel_block_sizes = [16] * 4
    runner.block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(2, 4, dtype=torch.int32)] * 4
    )
    runner.cache_config = CacheConfig()
    layers = spec_kernel_layers(runner)
    assert [(layer.name, layer.layout) for layer in layers] == [
        ("attn", "paged_token"),
        ("ring", ""),
    ]
    registry = KernelRegistry()
    ctx = KernelContext(
        device=runner.device,
        plan_kind="tree",
        max_num_reqs=4,
        max_candidates=4,
        capture_shapes=(PlanShape(4),),
    )
    registry.bind(ctx, layers)
    (commit,) = registry._commits
    assert isinstance(commit, KvCompactor)
    assert len(commit.launches) == 1
