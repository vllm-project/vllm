# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The startup rule and the spec kernel registry: what a model's layers verify,
a tree speculator lowered to chains where they cannot verify trees, and the
kernels providers overlay on the reference.

Each model is its layer mix: the real layer classes (built without weights) and
their KV cache specs. CPU only; kernels are toys.
"""

import importlib
import logging
import uuid
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_layer_module
from tests.v1.spec_decode.utils import fake_speculative_config
from vllm.config import CacheConfig, ParallelConfig
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
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    RSWASpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu.spec_decode.capabilities import (
    SpecCapabilities,
    SpecLayerKind,
    SpecLimit,
    layer_spec_limits,
    reduce_spec_capabilities,
    resolve_spec_resources,
)
from vllm.v1.worker.gpu.spec_decode.commit import KvCompactor
from vllm.v1.worker.gpu.spec_decode.kernels import (
    CommitStep,
    GDNSpecRows,
    KernelContext,
    KernelRegistry,
    SpecKernel,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.round import PlanShape, SpecResources
from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

GDN_VERIFY = ("mamba:gdn@slots", "verify")
SHORT_CONV_VERIFY = ("mamba:short_conv@slots", "verify")
STATE_COMMIT = ("layout:mamba_state@slots", "commit")
KV_COMMIT = ("layout:paged_token", "commit")

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


class _TreeBackend(FlashAttentionBackend):
    """A plugin backend that verifies trees."""

    serves_trees = True


def _group(spec, layers, backend):
    return SimpleNamespace(
        kv_cache_spec=spec, layer_names=list(layers), backend=backend
    )


def _model(*groups, backend=_TreeBackend):
    """``(attn_groups, layers)`` from ``(spec, {name: layer})``."""
    layers = {}
    attn_groups = []
    for spec, group_layers in groups:
        attn_groups.append([_group(spec, group_layers, backend)])
        layers.update(group_layers)
    return attn_groups, layers


def _caps(model, state_mode="slots", kernels=None) -> SpecCapabilities:
    attn_groups, layers = model
    return SpecCapabilities.from_limits(
        layer_spec_limits(attn_groups, layers, state_mode, kernels)
    )


class ToyKernel(SpecKernel):
    """Supports ``kinds`` on layers whose name is not in ``declines``; records
    its binds and commits."""

    def __init__(self, kinds=("chain", "tree"), declines=()):
        self.kinds = frozenset(kinds)
        self.declines = frozenset(declines)
        self.binds: list[list[str]] = []
        self.steps: list[CommitStep] = []

    def supports(self, layer):
        return frozenset() if layer.name in self.declines else self.kinds

    def bind(self, ctx, layers):
        self.binds.append([layer.name for layer in layers])

    def verify(self, rows):
        raise AssertionError("not called on CPU")

    def commit(self, step):
        self.steps.append(step)


class Provider:
    def __init__(self, kernels, name=None):
        self.name = name or f"toy-{uuid.uuid4().hex[:8]}"
        self._kernels = kernels

    def kernels(self):
        return self._kernels


def _provided() -> KernelRegistry:
    """A plugin's recurrent tree verify: verify and state commit kernels."""
    return KernelRegistry(
        [Provider({GDN_VERIFY: ToyKernel(), STATE_COMMIT: ToyKernel()})]
    )


def _layers(prefix, n, make):
    return {f"{prefix}.{i}": make() for i in range(n)}


def _llama(backend=_TreeBackend):
    return _model((_FULL, _layers("attn", 4, _attn)), backend=backend)


def _gpt_oss():
    # Alternating sliding-window and full layers, all with attention sinks.
    return _model(
        (_SWA, _layers("swa", 2, lambda: _attn(has_sink=True))),
        (_FULL, _layers("full", 2, lambda: _attn(has_sink=True))),
    )


def _gemma2():
    softcap = lambda: _attn(logits_soft_cap=50.0)  # noqa: E731
    return _model(
        (_SWA, _layers("swa", 2, softcap)),
        (_FULL, _layers("full", 2, softcap)),
    )


def _rswa():
    rswa = RSWASpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        rswa_window=64,
    )
    return _model((rswa, _layers("rswa", 2, _attn)))


def _dsv3():
    mla = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    return _model((mla, _layers("mla", 3, _attn)))


def _hybrid(layer_cls):
    return _model(
        (_MAMBA, _layers("linear", 3, lambda: _mamba(layer_cls))),
        (_FULL, _layers("attn", 1, _attn)),
    )


def _import_class(qualname):
    module, name = qualname.rsplit(":", 1)
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as e:
        pytest.skip(f"{qualname} does not import here: {e}")


def _runner(**overrides):
    attn_groups, layers = _llama()
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
    )
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def _resolve(runner, kernels=None) -> SpecCapabilities:
    spec_round = object.__new__(SpecRound)
    spec_round.kernels = kernels or KernelRegistry()
    return spec_round.resolve_capabilities(runner)


def _hybrid_runner(layer_cls, **overrides):
    attn_groups, layers = _hybrid(layer_cls)
    return _runner(
        attn_groups=attn_groups,
        compilation_config=SimpleNamespace(static_forward_context=layers),
        **overrides,
    )


# --- the startup rule ---------------------------------------------------------


def _full_attention_features():
    """Sinks, a soft cap and ALiBi on full attention do not lower the plan."""
    layer = _attn(has_sink=True, logits_soft_cap=30.0, alibi_slopes=torch.ones(2))
    return _model((_FULL, {"a": layer}))


def _draft_and_hidden_layers():
    hidden = HiddenStateCacheSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    draft = _attn()
    draft.is_draft_layer = True
    return _model(
        (_FULL, _layers("attn", 2, _attn)),
        (hidden, {"hidden": _attn()}),
        (_SWA, {"draft": draft}),
    )


_SWA_REASON = "attn:sliding_window, 2 layers: not full attention"
_LOWERED = ": keeps verify rows in chain order"


def _gdn_reason(mode):
    return f"mamba:gdn@{mode}, 3 layers: no mamba:gdn@{mode}/verify kernel"


# (model, state mode, with a provider, plan kind, reasons)
_MODEL_RULES: dict[str, tuple[Any, str, bool, str, list[str] | None]] = {
    "llama": (_llama, "slots", False, "tree", []),
    "llama-in-tree-backend": (
        lambda: _llama(FlashAttentionBackend),
        "slots",
        False,
        "chain",
        ["attn:full_attention, 4 layers: backend FLASH_ATTN verifies chains"],
    ),
    "full-attention-features": (_full_attention_features, "slots", False, "tree", []),
    "draft-and-hidden-state": (_draft_and_hidden_layers, "slots", False, "tree", []),
    "gpt-oss": (_gpt_oss, "slots", False, "chain", [_SWA_REASON]),
    "gemma2": (_gemma2, "slots", False, "chain", [_SWA_REASON]),
    "rswa": (
        _rswa,
        "slots",
        False,
        "chain",
        ["attn:full_attention, 2 layers: not full attention"],
    ),
    "dsv3-mla": (
        _dsv3,
        "slots",
        False,
        "chain",
        ["attn:mla_attention, 3 layers: not full attention"],
    ),
    "qwen-gdn": (
        QwenGatedDeltaNetAttention,
        "slots",
        False,
        "chain",
        [_gdn_reason("slots")],
    ),
    "qwen-gdn-provider": (QwenGatedDeltaNetAttention, "slots", True, "tree", []),
    "qwen-gdn-replay": (
        QwenGatedDeltaNetAttention,
        "replay",
        True,
        "chain",
        [_gdn_reason("replay")],
    ),
    "qwen-gdn-recover": (
        QwenGatedDeltaNetAttention,
        "recover",
        True,
        "chain",
        [_gdn_reason("recover")],
    ),
}
# A layer subclassing a tree-capable one runs its own kernels: chains only, with
# or without a provider. Qwen4Exp's PLE layer reports SHORT_CONV but runs its own.
_OWN_KERNEL_LAYERS = {
    "nemotron-h": MambaMixer2,
    "olmo-hybrid": OlmoHybridGatedDeltaNetAttention,
    "kimi-linear": KimiGatedDeltaNetAttention,
    "qwen4exp-ple": "vllm.models.qwen4_exp.nvidia.ple_layer:Qwen4ExpPLELayer",
    "kimi-k3-nvidia": "vllm.models.kimi_k3.nvidia.kda:KimiK3DeltaAttention",
    "kimi-k3-amd": "vllm.models.kimi_k3.amd.kda:KimiK3DeltaAttention",
    "lfm2-short-conv": ShortConv,
    "jamba-mamba1": MambaMixer,
}
for _name, _layer_cls in _OWN_KERNEL_LAYERS.items():
    _MODEL_RULES[_name] = (_layer_cls, "slots", True, "chain", None)


@pytest.mark.parametrize("case", list(_MODEL_RULES))
def test_plan_kind_rules(case):
    """What a model's layers let it verify, and why."""
    model, state_mode, provided, plan_kind, reasons = _MODEL_RULES[case]
    if isinstance(model, str):
        model = _import_class(model)
    if isinstance(model, type):
        layer_cls = model
        model = lambda: _hybrid(layer_cls)  # noqa: E731
        if reasons is None:
            reasons = [
                f"{layer_cls.__name__}, 3 layers: declares no spec_layer_variant"
            ]
    caps = _caps(model(), state_mode, _provided() if provided else None)
    assert caps.plan_kind == plan_kind
    assert [limit.reason for limit in caps.limits] == reasons


@pytest.mark.parametrize(
    "override, reason",
    [
        ({}, None),
        ({"use_pp": True}, "pipeline parallelism (no tree path broadcast)" + _LOWERED),
        ({"pcp_manager": object()}, "prefill context parallelism" + _LOWERED),
        ({"dcp_size": 2}, "decode context parallelism" + _LOWERED),
        ({"batch_sharder": object()}, "batch-sharded sampling" + _LOWERED),
        (
            {"parallel_config": ParallelConfig(enable_dbo=True)},
            "micro-batching" + _LOWERED,
        ),
        ({"cache_config": CacheConfig(use_replayssm=True)}, _gdn_reason("replay")),
    ],
    ids=["default", "pp", "pcp", "dcp", "batch-sharded", "dbo", "replayssm"],
)
def test_runner_settings_lower_trees_to_chains(override, reason):
    """The runner's settings, and the Mamba state mode it reads from the cache
    config, lower a model that verifies trees to chains."""
    if "cache_config" in override:
        runner, kernels = (
            _hybrid_runner(QwenGatedDeltaNetAttention, **override),
            _provided(),
        )
    else:
        runner, kernels = _runner(**override), None
    caps = _resolve(runner, kernels)
    assert caps.plan_kind == ("tree" if reason is None else "chain")
    assert [limit.reason for limit in caps.limits] == (
        [] if reason is None else [reason]
    )


# --- resolving the speculator's resources -------------------------------------


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


def _chain_caps(reason="attn:sliding_window, 2 layers: not full attention"):
    return SpecCapabilities.from_limits([SpecLimit("chain", reason)])


@pytest.mark.parametrize(
    "layer, with_provider",
    [(MambaMixer2, "chain"), (QwenGatedDeltaNetAttention, "tree")],
    ids=["nemotron-h", "qwen-gdn"],
)
def test_tree_speculator_lowers_to_chain_and_logs_once(layer, with_provider, caplog):
    """On a hybrid without a provider the runner lowers a tree to a chain: the
    same candidates and shapes, nothing raises, one log line per reason."""
    runner = _hybrid_runner(layer)
    caps = _resolve(runner)
    config = _TreeConfig()
    declared = config.resources(8)
    assert declared.plan_kind == "tree"
    with caplog.at_level(logging.INFO, logger="vllm"):
        effective = resolve_spec_resources(config, declared, [caps])
    assert effective.plan_kind == "chain"
    assert effective.max_branching == 1
    assert effective.max_candidates == declared.max_candidates
    assert effective.capture_shapes == declared.capture_shapes
    lines = [r.getMessage() for r in caplog.records if "spec decode" in r.getMessage()]
    assert len(lines) == len(caps.limits) == 1
    assert f"({caps.limits[0].reason})" in lines[0]
    assert _resolve(runner, _provided()).plan_kind == with_provider

    two = SpecCapabilities.from_limits(
        [SpecLimit("chain", "a: x"), SpecLimit("chain", "b: y")]
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="vllm"):
        resolve_spec_resources(config, declared, [two])
    lines = [r.getMessage() for r in caplog.records if "spec decode" in r.getMessage()]
    assert len(lines) == 2
    assert "(a: x)" in lines[0] and "(b: y)" in lines[1]


def test_kept_resources(caplog):
    """A tree on a model that verifies trees, a chain speculator on one that
    does not, and a variable-length chain on DSv3 keep what they declared."""
    config = _TreeConfig()
    declared = config.resources(8)
    assert resolve_spec_resources(config, declared, [_tree_caps()]) is declared
    chain_config = _spec().speculator_config
    chain = chain_config.resources(8)
    with caplog.at_level(logging.INFO, logger="vllm"):
        assert resolve_spec_resources(chain_config, chain, [_chain_caps()]) == chain
    assert not [r for r in caplog.records if "spec decode" in r.getMessage()]
    varlen = SpecResources(
        max_candidates=4, capture_shapes=(PlanShape(4), PlanShape(None))
    )
    dsv3 = _caps(_dsv3())
    assert resolve_spec_resources(_TreeConfig(), varlen, [dsv3]) == varlen


class _Overreach(_TreeConfig):
    def __init__(self, **changes):
        super().__init__()
        self.changes = changes

    def adapt_resources(self, declared, caps):
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
    rank0 = _caps(_llama())
    rank1 = _caps(_model((_SWA, {"swa.9": _attn()})))
    caps = reduce_spec_capabilities([rank0, rank1])
    assert caps.plan_kind == "chain"
    assert caps.limits == rank0.limits + rank1.limits
    assert reduce_spec_capabilities([rank0, rank0]).plan_kind == "tree"


# --- the spec kernel registry ------------------------------------------------


def _module():
    return SimpleNamespace(spec_verify_kernel=None)


def _mamba_layer(name, variant="gdn", state_mode="slots"):
    kind = SpecLayerKind("mamba", variant, state_mode)
    return SpecLayerSpec(name, kind, _module(), "mamba_state")


def _attn_layer(name, layout="paged_token"):
    return SpecLayerSpec(
        name, SpecLayerKind("attn", "full_attention"), _module(), layout
    )


SHAPES = (PlanShape(4), PlanShape(None))


def _ctx(plan_kind="tree", shapes=SHAPES):
    return KernelContext(
        device=torch.device("cpu"),
        plan_kind=plan_kind,
        max_num_reqs=3,
        max_candidates=4,
        capture_shapes=tuple(shapes),
    )


def test_registry_lookup(caplog):
    """The reference is the KV commit; providers overlay it key by key, a later
    provider wins a key, each override is logged once, and an unknown op is
    refused."""
    registry = KernelRegistry()
    for key in (GDN_VERIFY, SHORT_CONV_VERIFY, STATE_COMMIT):
        assert registry.get(key) is None
    assert isinstance(registry.get(KV_COMMIT), KvCompactor)
    assert KernelRegistry.required_keys(_mamba_layer("m", "gdn", "replay")) == [
        ("mamba:gdn@replay", "verify"),
        ("layout:mamba_state@replay", "commit"),
    ]
    assert KernelRegistry.required_keys(_attn_layer("a")) == [KV_COMMIT]
    assert KernelRegistry.required_keys(_attn_layer("ring", layout="")) == []

    first, second = ToyKernel(), ToyKernel()
    provider = Provider({GDN_VERIFY: first, SHORT_CONV_VERIFY: ToyKernel()})
    with caplog.at_level(logging.INFO, logger="vllm"):
        partial = KernelRegistry([provider])
        KernelRegistry([provider])
    assert partial.get(GDN_VERIFY) is first
    assert isinstance(partial.get(KV_COMMIT), KvCompactor)
    lines = [r.getMessage() for r in caplog.records if provider.name in r.getMessage()]
    assert len(lines) == 2
    assert "mamba:gdn@slots/verify" in lines[0]
    assert "mamba:short_conv@slots/verify" in lines[1]
    later = KernelRegistry(
        [Provider({GDN_VERIFY: first}), Provider({GDN_VERIFY: second})]
    )
    assert later.get(GDN_VERIFY) is second

    with pytest.raises(ValueError, match="the op must be one of"):
        KernelRegistry([Provider({("mamba:gdn@slots", "prefill"): ToyKernel()})])


@pytest.mark.parametrize(
    "providers, layer, plan_kind, why",
    [
        ({}, _mamba_layer("m", "gdn"), "chain", "no mamba:gdn@slots/verify kernel"),
        ({}, _mamba_layer("m", "short_conv"), "chain", "no mamba:short_conv@slots"),
        ({}, _mamba_layer("m", "gdn", "replay"), "chain", "no mamba:gdn@replay"),
        ({}, _attn_layer("a"), "tree", None),
        # A cache not paged by slot keeps its own commit: nothing to need.
        ({}, _attn_layer("ring", layout=""), "tree", None),
        (
            {},
            _attn_layer("a", layout="paged_split_scale"),
            "chain",
            "no layout:paged_split_scale/commit",
        ),
        (
            {SHORT_CONV_VERIFY: ToyKernel(), STATE_COMMIT: ToyKernel()},
            _mamba_layer("m", "short_conv"),
            "tree",
            None,
        ),
        (
            {GDN_VERIFY: ToyKernel(declines={"m.1"}), STATE_COMMIT: ToyKernel()},
            _mamba_layer("m.0"),
            "tree",
            None,
        ),
        (
            {GDN_VERIFY: ToyKernel(declines={"m.1"}), STATE_COMMIT: ToyKernel()},
            _mamba_layer("m.1"),
            "chain",
            "ToyKernel declines the layer",
        ),
        # Verify takes trees, the state commit chains only: the layer chains.
        (
            {GDN_VERIFY: ToyKernel(), STATE_COMMIT: ToyKernel(kinds={"chain"})},
            _mamba_layer("m"),
            "chain",
            "ToyKernel verifies chain",
        ),
    ],
    ids=[
        "gdn",
        "short-conv",
        "gdn-replay",
        "attn",
        "ring",
        "nvfp4-kv",
        "provided",
        "accepted",
        "declined",
        "lowest-kind",
    ],
)
def test_registry_plan_kind(providers, layer, plan_kind, why):
    """A layer verifies what every kernel it needs supports."""
    registry = KernelRegistry([Provider(providers)] if providers else [])
    kind, reason = registry.plan_kind(layer)
    assert kind == plan_kind
    if why is not None:
        assert why in reason


def test_bind_commit_capture():
    """Bind binds each kernel once to its layers (a chain binds the verify
    kernels only, no spec decode unbinds them, and a layer without one runs its
    own); commit and capture run every bound commit kernel, capture once per
    shape with identity paths."""
    verify, state, kv = ToyKernel(), ToyKernel(), ToyKernel()
    registry = KernelRegistry(
        [Provider({GDN_VERIFY: verify, STATE_COMMIT: state, KV_COMMIT: kv})]
    )
    layers = [
        _mamba_layer("m.0"),
        _mamba_layer("m.1"),
        _attn_layer("a.0"),
        _attn_layer("ring", layout=""),
    ]
    registry.bind(_ctx("chain"), layers)
    assert verify.binds == [["m.0", "m.1"]] and not state.binds and not kv.binds
    registry.capture(_ctx("chain"))
    assert not state.steps and not kv.steps
    registry.bind(_ctx("none"), layers)
    assert layers[0].module.spec_verify_kernel is None

    shapes = (PlanShape(4), PlanShape(2), PlanShape(None))
    ctx = _ctx("tree", shapes)
    verify.binds.clear()
    registry.bind(ctx, layers)
    assert verify.binds == state.binds == [["m.0", "m.1"]]
    assert kv.binds == [["a.0"]]
    assert all(layer.module.spec_verify_kernel is verify for layer in layers[:2])
    step = CommitStep(result=None, num_reqs=2, count=None, seq_lens=None)
    registry.commit(step)
    assert state.steps == [step] and kv.steps == [step]
    state.steps.clear()
    kv.steps.clear()
    runs = []
    registry.capture(ctx, run=lambda fn: runs.append(fn()))
    assert len(runs) == len(shapes)
    for kernel in (state, kv):
        assert [step.count.tolist() for step in kernel.steps] == [
            [4, 4, 4],
            [2, 2, 2],
            [4, 4, 4],  # varlen: up to max_candidates rows
        ]
        for step in kernel.steps:
            assert step.num_reqs == ctx.max_num_reqs
            assert step.result.path.tolist() == [[0, 1, 2, 3]] * 3
            assert torch.equal(step.result.path_len, step.count)

    stale = _mamba_layer("m.0")
    stale.module.spec_verify_kernel = "stale"
    KernelRegistry([Provider({STATE_COMMIT: ToyKernel()})]).bind(_ctx(), [stale])
    assert stale.module.spec_verify_kernel is None


# --- a provider's GDN verify kernel ---------------------------------------------


NUM_TOKENS = 4


class _RecordingKernel(SpecKernel):
    """Writes ``fill`` to every spec row's output and records what it got."""

    def __init__(self, fill: float):
        self.fill = fill
        self.rows: list = []

    def supports(self, layer):
        return frozenset({"chain", "tree"})

    def verify(self, rows):
        self.rows.append(rows)
        rows.out.fill_(self.fill)


class _Calls:
    def __init__(self):
        self.conv = 0
        self.recurrent = 0


def _layer(
    monkeypatch, calls: _Calls, metadata: GDNAttentionMetadata
) -> QwenGatedDeltaNetAttention:
    """A tiny GDN layer whose spec kernels are fakes that count their calls.

    Built with ``__new__``: the real constructor needs a TP group and weights,
    and these tests only exercise how ``_forward_core`` routes the rows.
    """

    def fake_conv(mixed_qkv_spec, *args, **kwargs):
        calls.conv += 1
        return mixed_qkv_spec + 100

    def fake_recurrent(**kwargs):
        calls.recurrent += 1
        return torch.full((1, kwargs["q"].shape[0], 2, 3), -1.0), None

    monkeypatch.setattr(gdn_layer_module, "causal_conv1d_update", fake_conv)
    monkeypatch.setattr(
        gdn_layer_module, "fused_sigmoid_gating_delta_rule_update", fake_recurrent
    )
    monkeypatch.setattr(gdn_layer_module, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(
        gdn_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}),
    )
    layer = QwenGatedDeltaNetAttention.__new__(QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.enable_packed_recurrent_decode = False
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(weight=torch.zeros(5, 1, 4), bias=None)
    layer.kv_cache = (torch.zeros(2, 5, 3), torch.zeros(2, 2, 3, 3))
    layer.A_log = torch.zeros(2)
    layer.dt_bias = torch.ones(2)
    layer.tp_size = 1
    layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim = (
        1,
        2,
        2,
        3,
    )
    layer.rearrange_mixed_qkv = lambda x: (None, None, None) if x is None else (x, x, x)
    layer.qkv_row_views = layer.rearrange_mixed_qkv
    return layer


def _spec_metadata(parents=None) -> GDNAttentionMetadata:
    """One spec-decode request of NUM_TOKENS rows and nothing else."""
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=NUM_TOKENS,
        num_actual_tokens=NUM_TOKENS,
        spec_query_start_loc=torch.tensor([0, NUM_TOKENS], dtype=torch.int32),
        spec_state_indices_tensor=torch.zeros(1, 4, dtype=torch.int32),
        spec_sequence_masks=torch.ones(1, dtype=torch.bool),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        spec_tree_parents=parents,
    )


def test_gdn_runs_the_bound_kernel(monkeypatch):
    """The layer hands its spec rows, as the views they are and with the plan's
    parents, to the bound kernel; plain decode rows next to them run the in-tree
    kernels. The fused MTP decode kernel is off while a kernel is bound."""
    kernel = _RecordingKernel(fill=7.0)
    parents = torch.tensor([-1, 0, 0, 1], dtype=torch.int32)
    metadata = _spec_metadata(parents)
    metadata.num_decodes = metadata.num_decode_tokens = 1
    metadata.num_actual_tokens = NUM_TOKENS + 1
    metadata.spec_token_indx = torch.arange(1, NUM_TOKENS + 1)
    metadata.non_spec_token_indx = torch.tensor([0])
    metadata.non_spec_state_indices_tensor = torch.tensor([1], dtype=torch.int32)
    metadata.non_spec_query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    calls = _Calls()
    layer = _layer(monkeypatch, calls, metadata)
    layer.spec_verify_kernel = kernel
    # Two rows of padding past the batch.
    out = torch.zeros(NUM_TOKENS + 3, 2, 3)
    b, a = torch.zeros(NUM_TOKENS + 3, 4).chunk(2, dim=-1)
    layer._forward_core(
        mixed_qkv=torch.zeros(NUM_TOKENS + 3, 5), b=b, a=a, core_attn_out=out
    )

    (rows,) = kernel.rows
    assert isinstance(rows, GDNSpecRows)
    assert rows.mixed_qkv.shape[0] == NUM_TOKENS and rows.out.shape[0] == NUM_TOKENS
    assert rows.layer_name == "layer"
    assert (rows.num_k_heads, rows.num_v_heads, rows.head_k_dim, rows.head_v_dim) == (
        1,
        2,
        2,
        3,
    )
    assert rows.state_indices is metadata.spec_state_indices_tensor
    assert rows.num_accepted_tokens is metadata.num_accepted_tokens
    assert rows.query_start_loc is metadata.spec_query_start_loc
    assert rows.parents is parents
    assert (calls.conv, calls.recurrent) == (1, 1)  # the decode row
    assert torch.all(out[0] == -1.0)
    assert torch.all(out[1 : NUM_TOKENS + 1] == 7.0)
    assert torch.all(out[NUM_TOKENS + 1 :] == 0)

    state_dtype = next(iter(gdn_layer_module.FUSED_GDN_STATE_DTYPES))
    layer.kv_cache = (torch.zeros(1), torch.zeros(1, dtype=state_dtype))
    layer.gdn_decode_kernel = "cuda"
    layer.num_v_heads = layer.num_k_heads = 2
    monkeypatch.setattr(
        torch.ops._C, "fused_gdn_decode_post_conv_mtp", object(), raising=False
    )
    assert layer._can_use_fused_gdn_mtp_decode(_spec_metadata()) is False
    layer.spec_verify_kernel = None
    assert layer._can_use_fused_gdn_mtp_decode(_spec_metadata()) is True
