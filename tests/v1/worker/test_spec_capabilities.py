# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The startup rule and the spec kernel registry, on real layer classes (CPU)."""

import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.v1.spec_decode.utils import fake_speculative_config
from vllm.config import CacheConfig, CompilationConfig, ParallelConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
    OlmoHybridGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    MambaSpec,
    RSWASpec,
    SlidingWindowSpec,
)
from vllm.v1.spec_decode.resources import (
    SpecCapabilities,
    SpecLayerKind,
    SpecLimit,
    SpecResources,
    adapt_spec_resources,
    resolve_spec_resources,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.capabilities import (
    causal_full_attention,
    layer_spec_limits,
)
from vllm.v1.worker.gpu.spec_decode.kernels import (
    CommitStep,
    KernelContext,
    KernelKey,
    KernelRegistry,
    SpecKernel,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig
from vllm.v1.worker.utils import AttentionGroup

STATE_COMMIT: KernelKey = ("layout:recurrent_state", "commit")
KV_COMMIT: KernelKey = ("layout:paged_token", "commit")


def _spec(cls=FullAttentionSpec, **kwargs):
    return cls(
        block_size=kwargs.pop("block_size", 16),
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
        **kwargs,
    )


_FULL = _spec()
_SWA = _spec(SlidingWindowSpec, sliding_window=128)
_MAMBA = MambaSpec(block_size=16, shapes=((4, 8),), dtypes=(torch.float32,))


def _bare_layer(cls=Attention, **attrs):
    """A layer of ``cls`` built without weights."""
    layer = object.__new__(cls)
    torch.nn.Module.__init__(layer)
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer


class _TreeBackend(FlashAttentionBackend):
    """A plugin backend that verifies trees."""

    @classmethod
    def tree_decline_reason(cls, layer, kv_cache_spec):
        if not causal_full_attention(layer, kv_cache_spec):
            return "not full attention"
        return None


def _model(*groups, backend=_TreeBackend):
    """``(attn_groups, layers)`` from ``(spec, prefix, n, layer class)``."""
    attn_groups, layers = [], {}
    for gid, (spec, prefix, n, cls) in enumerate(groups):
        names = [f"{prefix}.{i}" for i in range(n)]
        attn_groups.append([AttentionGroup(backend, names, spec, gid)])
        layers.update({name: _bare_layer(cls) for name in names})
    return attn_groups, layers


class ToyKernel(SpecKernel):
    """Runs the layers whose name is not in ``declines``; records its binds
    and commits."""

    def __init__(self, declines=()):
        self.declines = frozenset(declines)
        self.binds: list[list[str]] = []
        self.ctxs: list[KernelContext] = []
        self.steps: list[CommitStep] = []

    def supports(self, layer):
        return layer.name not in self.declines

    def bind(self, ctx, layers):
        self.binds.append([layer.name for layer in layers])
        self.ctxs.append(ctx)

    def commit(self, step):
        self.steps.append(step)


def _kv_provided() -> KernelRegistry:
    """A plugin's KV compaction: what every attention layer needs for trees."""
    return KernelRegistry({KV_COMMIT: ToyKernel()})


def _provided() -> KernelRegistry:
    """A plugin's KV and recurrent-state commits."""
    return KernelRegistry({STATE_COMMIT: ToyKernel(), KV_COMMIT: ToyKernel()})


def _hybrid(layer_cls, backend=_TreeBackend):
    return _model(
        (_MAMBA, "linear", 3, layer_cls), (_FULL, "attn", 1, Attention), backend=backend
    )


_LLAMA = ((_FULL, "attn", 4, Attention),)
_IN_TREE = "attn:full_attention, 4 layers: backend FLASH_ATTN verifies chains"


class TreeGDN(QwenGatedDeltaNetAttention):
    """What a plugin swaps in with ``PluggableLayer.register_oot``."""

    def tree_decline_reason(self):
        return None


def _chains(cls, why="verifies chains"):
    return f"recurrent:{cls.__name__}, 3 layers: {why}"


# case: (model, state mode, with plugin kernels, plan kind, reasons)
_MODEL_RULES: dict[str, tuple[Any, str, bool, str, list[str] | None]] = {
    "llama": (lambda: _model(*_LLAMA), "slots", False, "tree", []),
    "llama-in-tree-backend": (
        lambda: _model(*_LLAMA, backend=FlashAttentionBackend),
        "slots",
        False,
        "chain",
        [_IN_TREE],
    ),
    "draft-and-hidden-state": (
        lambda: _model(
            *_LLAMA,
            (_spec(HiddenStateCacheSpec), "hidden", 1, Attention),
            (_SWA, "draft", 1, DraftAttention),
        ),
        "slots",
        False,
        "tree",
        [],
    ),
    "gpt-oss": (
        lambda: _model((_SWA, "swa", 2, Attention), *_LLAMA),
        "slots",
        False,
        "chain",
        ["attn:sliding_window, 2 layers: not full attention"],
    ),
    "rswa": (
        lambda: _model((_spec(RSWASpec, rswa_window=64), "rswa", 2, Attention)),
        "slots",
        False,
        "chain",
        ["attn:full_attention, 2 layers: not full attention"],
    ),
    "qwen-gdn": (QwenGatedDeltaNetAttention, "slots", True, "chain", None),
    "tree-gdn": (TreeGDN, "slots", True, "tree", []),
    "tree-gdn-no-state-commit": (
        TreeGDN,
        "slots",
        False,
        "chain",
        ["recurrent:TreeGDN, 3 layers: no layout:recurrent_state/commit kernel"],
    ),
    "tree-gdn-in-tree-backend": (
        lambda: _hybrid(TreeGDN, FlashAttentionBackend),
        "slots",
        True,
        "chain",
        [_IN_TREE.replace("4 layers", "1 layer")],
    ),
}
# In-tree recurrent layers verify chains, with or without plugin kernels.
for _name, _layer_cls in {
    "nemotron-h": MambaMixer2,
    "olmo-hybrid": OlmoHybridGatedDeltaNetAttention,
}.items():
    _MODEL_RULES[_name] = (_layer_cls, "slots", True, "chain", None)


class DraftAttention(Attention):
    is_draft_layer = True


@pytest.mark.parametrize("case", list(_MODEL_RULES))
def test_plan_kind_rules(case):
    """What a model's layers let it verify, and why."""
    model, state_mode, provided, plan_kind, reasons = _MODEL_RULES[case]
    if isinstance(model, type):
        layer_cls = model
        model = lambda: _hybrid(layer_cls)  # noqa: E731
        if reasons is None:
            reasons = [_chains(layer_cls)]
    attn_groups, layers = model()
    kernels = _provided() if provided else _kv_provided()
    caps = SpecCapabilities.from_limits(layer_spec_limits(attn_groups, layers, kernels))
    assert caps.plan_kind == plan_kind
    assert [limit.reason for limit in caps.limits] == reasons


def test_no_kv_commit_verifies_chains():
    """Without a KV commit kernel, attention layers verify chains:
    vLLM keeps no compaction kernel, so a tree speculator lowers at startup."""
    attn_groups, layers = _model(*_LLAMA)
    caps = SpecCapabilities.from_limits(layer_spec_limits(attn_groups, layers))
    assert caps.plan_kind == "chain"
    assert [limit.reason for limit in caps.limits] == [
        "attn:full_attention, 4 layers: no layout:paged_token/commit kernel"
    ]


_LOWERED = ": keeps verify rows in chain order"


class _Speculator:
    def __init__(self, hooks):
        self.hooks = frozenset(hooks)

    def overridden_hooks(self):
        return self.hooks


def _resolve(**settings):
    """Resolve the capabilities of a runner over a hybrid whose layers all
    verify trees, with ``settings`` on the runner."""
    runner = object.__new__(GPUModelRunner)
    runner.attn_groups, layers = _hybrid(TreeGDN)
    runner.compilation_config = CompilationConfig()
    runner.compilation_config.static_forward_context = layers
    runner.cache_config, runner.parallel_config = CacheConfig(), ParallelConfig()
    runner.use_pp, runner.pcp_manager, runner.dcp_size = False, None, 1
    runner.batch_sharder, runner.adaptive_verification = None, None
    runner.speculator = _Speculator({"plan", "tree_accept"})
    runner.model_state = SimpleNamespace(tree_decline_reason=lambda: None)
    for setting, value in settings.items():
        setattr(runner, setting, value)
    spec_round = object.__new__(SpecRound)
    spec_round.kernels = _provided()
    spec_round.resources = SpecResources(capture_rows=(1, 4), max_branching=2)
    return spec_round.resolve_capabilities(runner)


@pytest.mark.parametrize(
    "setting, value, reason",
    [
        (None, None, None),
        ("use_pp", True, "pipeline parallelism (no tree path broadcast)" + _LOWERED),
        ("pcp_manager", object(), "prefill context parallelism" + _LOWERED),
        ("dcp_size", 2, "decode context parallelism" + _LOWERED),
        (
            "parallel_config",
            ParallelConfig(enable_dbo=True),
            "micro-batching" + _LOWERED,
        ),
        (
            "model_state",
            SimpleNamespace(tree_decline_reason=lambda: "no tree resume"),
            "model state: no tree resume",
        ),
        # vLLM has no tree walk: a speculator chooses its accepted paths.
        (
            "speculator",
            _Speculator({"plan"}),
            "a speculator without tree_accept" + _LOWERED,
        ),
    ],
)
def test_runner_settings_lower_trees_to_chains(setting, value, reason):
    """The runner's settings and its model state lower a model that verifies
    trees to chains."""
    caps = _resolve(**({} if setting is None else {setting: value}))

    assert caps.plan_kind == ("tree" if reason is None else "chain")
    assert [limit.reason for limit in caps.limits] == ([reason] if reason else [])


def test_adaptive_verification_refuses_plan_hooks():
    """Adaptive verification sets each request's verified drafts itself, so a
    speculator that plans cannot run with it."""
    with pytest.raises(ValueError, match="enable_adaptive_verification=false"):
        _resolve(adaptive_verification=object())


def _speculative_config():
    return fake_speculative_config(
        SpeculatorConfig,
        num_speculative_tokens=3,
        enable_adaptive_verification=False,
        num_speculative_tokens_per_batch_size=None,
        uses_dynamic_speculative_decoding=lambda: False,
    )


class _TreeConfig(SpeculatorConfig):
    """A tree speculator, 2-way branching; ``changes`` are what its
    ``adapt_resources`` returns over the declared resources."""

    def __init__(self, **changes):
        super().__init__(_speculative_config())
        self.changes = changes

    def resources(self, max_num_seqs):
        return SpecResources(capture_rows=(1, 4), max_branching=2)

    def adapt_resources(self, declared, caps):
        if not self.changes:
            return super().adapt_resources(declared, caps)
        return replace(declared, **self.changes)


_TREE = SpecCapabilities.from_limits([])
_CHAIN = SpecCapabilities.from_limits(
    [SpecLimit("chain", "a: x"), SpecLimit("chain", "b: y")]
)


@pytest.mark.parametrize(
    "config, caps, want, logged",
    [
        # Lowered: the same candidates and shapes, one log line per reason.
        (_TreeConfig(), _CHAIN, "lowered", ["(a: x)", "(b: y)"]),
        (_TreeConfig(), _TREE, "declared", []),
        (_speculative_config().speculator_config, _CHAIN, "declared", []),
        (_TreeConfig(max_branching=2), _CHAIN, "plan_kind 'tree' is above", []),
        (_TreeConfig(max_branching=3), _TREE, "max_branching 3 is above", []),
        # Dropping the widest graph verifies fewer rows, and says so.
        (
            _TreeConfig(max_branching=1, capture_rows=(1,)),
            _CHAIN,
            "lowered",
            ["(a: x)", "(b: y)", "max_rows 4 -> 1"],
        ),
        (
            _TreeConfig(max_branching=1, capture_rows=(5,)),
            _CHAIN,
            r"capture_rows \[5\]",
            [],
        ),
    ],
    ids=[
        "lowered",
        "tree-kept",
        "chain-kept",
        "bad-kind",
        "bad-branching",
        "narrowed",
        "bad-rows",
    ],
)
def test_resolve_spec_resources(config, caps, want, logged, caplog):
    declared = config.resources(8)
    if want not in ("lowered", "declared"):
        with pytest.raises(ValueError, match=want):
            resolve_spec_resources(config, declared, [caps])
        return
    with caplog.at_level(logging.INFO, logger="vllm"):
        effective = resolve_spec_resources(config, declared, [caps])
    if want == "declared":
        assert effective == declared
    else:
        assert effective == replace(declared, **{"max_branching": 1, **config.changes})
        assert effective.plan_kind == "chain"
    lines = [r.getMessage() for r in caplog.records if "spec decode" in r.getMessage()]
    assert len(lines) == len(logged)
    assert all(why in line for why, line in zip(logged, lines))


@pytest.mark.parametrize("tree", [False, True], ids=["chain", "tree"])
def test_adapts_only_branching_resources(tree):
    """A chain runs as declared with no handshake (each worker checks it
    against its layers); a branching speculator's resources are adapted to the
    workers' layers, sent to them, and kept for the scheduler."""
    config = _TreeConfig() if tree else _speculative_config().speculator_config
    calls = []

    def collective_rpc(method, args=()):
        calls.append(method)
        return [_CHAIN] if method == "get_spec_capabilities" else None

    effective = adapt_spec_resources(
        SimpleNamespace(
            speculative_config=SimpleNamespace(speculator_config=config),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
        ),
        collective_rpc,
    )
    if not tree:
        assert calls == [] and effective is None
        return
    assert calls == ["get_spec_capabilities", "set_spec_resources"]
    assert effective == replace(config.resources(8), max_branching=1)


def test_pipeline_ranks_reduce_to_the_lowest_plan_kind():
    """PP ranks hold different layers: rank 0 the trees, rank 1 a SWA layer."""
    config = _TreeConfig()
    declared = config.resources(8)
    assert (
        resolve_spec_resources(config, declared, [_TREE, _CHAIN]).plan_kind == "chain"
    )
    assert resolve_spec_resources(config, declared, [_TREE, _TREE]) == declared


def _module():
    return torch.nn.Module()


def _recurrent_layer(name):
    kind = SpecLayerKind("recurrent", "TreeGDN")
    return SpecLayerSpec(name, kind, _module(), "recurrent_state")


_TABLE = torch.zeros(3, 4, dtype=torch.int32)


def _attn_layer(name, layout="paged_token", cache=None):
    module = _module()
    module.kv_cache = torch.zeros(8, 2, 4, 16) if cache is None else cache
    return SpecLayerSpec(
        name,
        SpecLayerKind("attn", "full_attention"),
        module,
        layout,
        block_table=_TABLE,
        block_size=4,
        num_blocks=module.kv_cache.shape[0],
    )


SHAPES = (4,)


def _ctx(plan_kind="tree", shapes=SHAPES):
    return KernelContext(
        device=torch.device("cpu"),
        plan_kind=plan_kind,
        max_num_reqs=3,
        max_rows=4,
        capture_rows=tuple(shapes),
    )


@pytest.mark.parametrize(
    "kernels, layer, plan_kind, why",
    [
        pytest.param(
            {},
            _recurrent_layer("m"),
            "chain",
            "no layout:recurrent_state/commit",
            id="recurrent",
        ),
        # No KV commit kernel: the layer verifies chains.
        pytest.param(
            {}, _attn_layer("a"), "chain", "no layout:paged_token/commit", id="attn"
        ),
        pytest.param(
            {KV_COMMIT: ToyKernel()}, _attn_layer("a"), "tree", None, id="attn-kv"
        ),
        # A cache not paged by slot keeps its own commit: nothing to need.
        pytest.param({}, _attn_layer("ring", layout=""), "tree", None, id="ring"),
        pytest.param(
            {},
            _attn_layer("a", layout="paged_split_scale"),
            "chain",
            "no layout:paged_split_scale/commit",
            id="nvfp4-kv",
        ),
        pytest.param(
            {STATE_COMMIT: ToyKernel()},
            _recurrent_layer("m"),
            "tree",
            None,
            id="provided",
        ),
        pytest.param(
            {STATE_COMMIT: ToyKernel(declines={"m.1"})},
            _recurrent_layer("m.1"),
            "chain",
            "ToyKernel declines the layer",
            id="declined",
        ),
    ],
)
def test_registry_plan_kind(kernels, layer, plan_kind, why):
    """A layer verifies what every kernel it needs supports."""
    registry = KernelRegistry(kernels)
    kind, reason = registry.plan_kind(layer)
    assert kind == plan_kind
    if why is not None:
        assert why in reason


def test_bind_and_commit():
    """Bind binds each commit kernel once to its layers, with the plan's
    parents; a chain binds none; commit runs every bound kernel."""
    state, kv = ToyKernel(), ToyKernel()
    registry = KernelRegistry({STATE_COMMIT: state, KV_COMMIT: kv})
    layers = [
        _recurrent_layer("m.0"),
        _recurrent_layer("m.1"),
        _attn_layer("a.0"),
        _attn_layer("ring", layout=""),
    ]
    registry.bind(_ctx("chain"), layers)
    assert not state.binds and not kv.binds

    parents = torch.full((8,), -1, dtype=torch.int32)
    registry.bind(replace(_ctx("tree"), tree_parents=parents), layers)
    assert state.binds == [["m.0", "m.1"]]
    assert kv.binds == [["a.0"]]
    assert [len(c.kv_layouts) for c in kv.ctxs] == [1]
    assert not any(c.kv_layouts for c in state.ctxs)
    assert all(c.tree_parents is parents for c in state.ctxs + kv.ctxs)
    step = CommitStep(result=None, num_reqs=2, rows=None, seq_lens=None)
    registry.commit(step)
    assert state.steps == [step] and kv.steps == [step]


def _kv_layouts(*layers):
    kernel = ToyKernel()
    KernelRegistry({KV_COMMIT: kernel}).bind(_ctx(), list(layers))
    return kernel.ctxs[0].kv_layouts


def test_paged_kv_layout_groups_and_sharing():
    """Layers whose cache is shared (KV sharing) appear once; layers of
    different block tables or shapes get a layout each."""
    shared = torch.zeros(8, 2, 4, 16)
    a, b = _attn_layer("a", cache=shared), _attn_layer("b", cache=shared)
    c = _attn_layer("c")
    other_table = replace(
        _attn_layer("d"), block_table=torch.zeros(3, 4, dtype=torch.int32)
    )
    mla = _attn_layer("mla", cache=torch.zeros(8, 4, 24))
    layouts = _kv_layouts(a, b, c, other_table, mla)
    assert [len(layout.caches) for layout in layouts] == [2, 1, 1]
    assert layouts[0].caches[0] is shared


@pytest.mark.parametrize(
    "layer, match",
    [
        pytest.param(
            _attn_layer("a", cache=torch.zeros(8, 2, 4, 2, 16)), "no paged KV", id="5d"
        ),
        pytest.param(
            replace(_attn_layer("a"), num_blocks=7), "no paged KV", id="num-blocks"
        ),
        pytest.param(
            _attn_layer("a", cache=torch.zeros(8, 2, 16, 4).transpose(2, 3)),
            "strided",
            id="strided-dim",
        ),
        pytest.param(
            replace(_attn_layer("a"), block_table=None), "int32", id="no-table"
        ),
    ],
)
def test_paged_kv_layout_refuses_at_bind(layer, match):
    """A cache the descriptor cannot describe fails at bind, before any step."""
    with pytest.raises(ValueError, match=match):
        _kv_layouts(layer)


@pytest.mark.parametrize("plan_kind", ["chain", "tree"])
def test_bind_one_kernel_for_two_commits(plan_kind):
    """A kernel given for both the KV and the recurrent-state commit is bound
    once, to every layer, and commits above a chain."""
    kernel = ToyKernel()
    registry = KernelRegistry({STATE_COMMIT: kernel, KV_COMMIT: kernel})
    registry.bind(_ctx(plan_kind), [_recurrent_layer("m.0"), _attn_layer("a.0")])
    assert kernel.binds == ([["m.0", "a.0"]] if plan_kind == "tree" else [])
    step = CommitStep(result=None, num_reqs=2, rows=None, seq_lens=None)
    registry.commit(step)
    assert kernel.steps == ([step] if plan_kind == "tree" else [])
