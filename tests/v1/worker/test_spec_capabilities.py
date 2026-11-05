# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The startup rule and the spec kernel registry, on real layer classes (CPU)."""

import logging
import uuid
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_layer_module
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
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    MambaSpec,
    RSWASpec,
    SlidingWindowSpec,
)
from vllm.v1.spec_decode.resources import SpecResources, adapt_spec_resources
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.capabilities import (
    SpecCapabilities,
    SpecLayerKind,
    SpecLimit,
    layer_spec_limits,
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
from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig
from vllm.v1.worker.utils import AttentionGroup

GDN_VERIFY = ("mamba:gdn@slots", "verify")
SHORT_CONV_VERIFY = ("mamba:short_conv@slots", "verify")
STATE_COMMIT = ("layout:mamba_state@slots", "commit")
KV_COMMIT = ("layout:paged_token", "commit")


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

    serves_trees = True


def _model(*groups, backend=_TreeBackend):
    """``(attn_groups, layers)`` from ``(spec, prefix, n, layer class)``."""
    attn_groups, layers = [], {}
    for gid, (spec, prefix, n, cls) in enumerate(groups):
        names = [f"{prefix}.{i}" for i in range(n)]
        attn_groups.append([AttentionGroup(backend, names, spec, gid)])
        layers.update({name: _bare_layer(cls) for name in names})
    return attn_groups, layers


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


def _hybrid(layer_cls, backend=_TreeBackend):
    return _model(
        (_MAMBA, "linear", 3, layer_cls), (_FULL, "attn", 1, Attention), backend=backend
    )


_LLAMA = ((_FULL, "attn", 4, Attention),)
_IN_TREE = "attn:full_attention, 4 layers: backend FLASH_ATTN verifies chains"


def _gdn_reason(mode):
    return f"mamba:gdn@{mode}, 3 layers: no mamba:gdn@{mode}/verify kernel"


# case: (model, state mode, with a provider, plan kind, reasons)
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
    "qwen-gdn": (QwenGatedDeltaNetAttention, "slots", False, "chain", None),
    "qwen-gdn-provider": (QwenGatedDeltaNetAttention, "slots", True, "tree", []),
    "qwen-gdn-provider-in-tree-backend": (
        lambda: _hybrid(QwenGatedDeltaNetAttention, FlashAttentionBackend),
        "slots",
        True,
        "chain",
        [_IN_TREE.replace("4 layers", "1 layer")],
    ),
    "qwen-gdn-replay": (QwenGatedDeltaNetAttention, "replay", True, "chain", None),
}
# A layer subclassing a tree-capable one (olmo) runs its own kernels: chains
# only, with or without a provider.
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
        if reasons is None and layer_cls is QwenGatedDeltaNetAttention:
            reasons = [_gdn_reason(state_mode)]
        elif reasons is None:
            reasons = [
                f"{layer_cls.__name__}, 3 layers: declares no spec_layer_variant"
            ]
    attn_groups, layers = model()
    kernels = _provided() if provided else None
    caps = SpecCapabilities.from_limits(
        layer_spec_limits(attn_groups, layers, state_mode, kernels)
    )
    assert caps.plan_kind == plan_kind
    assert [limit.reason for limit in caps.limits] == reasons


_LOWERED = ": keeps verify rows in chain order"


def _resolve(**settings):
    """Resolve the capabilities of a runner over a hybrid whose layers all
    verify trees, with ``settings`` on the runner."""
    runner = object.__new__(GPUModelRunner)
    runner.attn_groups, layers = _hybrid(QwenGatedDeltaNetAttention)
    runner.compilation_config = CompilationConfig()
    runner.compilation_config.static_forward_context = layers
    runner.cache_config, runner.parallel_config = CacheConfig(), ParallelConfig()
    runner.use_pp, runner.pcp_manager, runner.dcp_size = False, None, 1
    runner.batch_sharder, runner.adaptive_verification = None, None
    runner.speculator_plans = True
    for setting, value in settings.items():
        setattr(runner, setting, value)
    spec_round = object.__new__(SpecRound)
    spec_round.kernels = _provided()
    return spec_round.resolve_capabilities(runner)


@pytest.mark.parametrize(
    "setting, value, reason",
    [
        (None, None, None),
        ("use_pp", True, "pipeline parallelism (no tree path broadcast)" + _LOWERED),
        (
            "parallel_config",
            ParallelConfig(enable_dbo=True),
            "micro-batching" + _LOWERED,
        ),
        ("cache_config", CacheConfig(use_replayssm=True), _gdn_reason("replay")),
    ],
)
def test_runner_settings_lower_trees_to_chains(setting, value, reason):
    """The runner's settings, and the Mamba state mode it reads from the cache
    config, lower a model that verifies trees to chains."""
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
    module = torch.nn.Module()
    module.spec_verify_kernel = None
    return module


def _mamba_layer(name, variant="gdn", state_mode="slots"):
    kind = SpecLayerKind("mamba", variant, state_mode)
    return SpecLayerSpec(name, kind, _module(), "mamba_state")


def _attn_layer(name, layout="paged_token"):
    return SpecLayerSpec(
        name, SpecLayerKind("attn", "full_attention"), _module(), layout
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


def test_registry_lookup(caplog):
    """The reference is the KV commit; providers overlay it key by key, a later
    provider wins a key, each override is logged once, and an unknown op is
    refused."""
    registry = KernelRegistry()
    for key in (GDN_VERIFY, SHORT_CONV_VERIFY, STATE_COMMIT):
        assert registry.get(key) is None
    assert isinstance(registry.get(KV_COMMIT), KvCompactor)

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
        pytest.param(
            {}, _mamba_layer("m"), "chain", "no mamba:gdn@slots/verify", id="gdn"
        ),
        pytest.param({}, _attn_layer("a"), "tree", None, id="attn"),
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
            {SHORT_CONV_VERIFY: ToyKernel(), STATE_COMMIT: ToyKernel()},
            _mamba_layer("m", "short_conv"),
            "tree",
            None,
            id="provided",
        ),
        pytest.param(
            {GDN_VERIFY: ToyKernel(declines={"m.1"}), STATE_COMMIT: ToyKernel()},
            _mamba_layer("m.1"),
            "chain",
            "ToyKernel declines the layer",
            id="declined",
        ),
        # Verify takes trees, the state commit chains only: the layer chains.
        pytest.param(
            {GDN_VERIFY: ToyKernel(), STATE_COMMIT: ToyKernel(kinds={"chain"})},
            _mamba_layer("m"),
            "chain",
            "ToyKernel verifies chain",
            id="lowest-kind",
        ),
        # The layer runs its verify kernel on chains too.
        pytest.param(
            {GDN_VERIFY: ToyKernel(kinds={"tree"}), STATE_COMMIT: ToyKernel()},
            _mamba_layer("m"),
            "chain",
            "ToyKernel does not verify chains",
            id="tree-only-verify",
        ),
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

    shapes = (4, 2)
    ctx = _ctx("tree", shapes)
    verify.binds.clear()
    registry.bind(ctx, layers)
    assert verify.binds == state.binds == [["m.0", "m.1"]]
    assert kv.binds == [["a.0"]]
    assert all(layer.module.spec_verify_kernel is verify for layer in layers[:2])
    step = CommitStep(result=None, num_reqs=2, rows=None, seq_lens=None)
    registry.commit(step)
    assert state.steps == [step] and kv.steps == [step]
    state.steps.clear()
    kv.steps.clear()
    runs = []
    registry.capture(ctx, run=lambda fn: runs.append(fn()))
    assert len(runs) == len(shapes)
    for kernel in (state, kv):
        assert [step.rows.tolist() for step in kernel.steps] == [
            [4, 4, 4],
            [2, 2, 2],
        ]
        for step in kernel.steps:
            assert step.num_reqs == ctx.max_num_reqs
            assert step.result.path.tolist() == [[0, 1, 2, 3]] * 3
            assert torch.equal(step.result.path_len, step.rows)

    stale = _mamba_layer("m.0")
    stale.module.spec_verify_kernel = "stale"
    KernelRegistry([Provider({STATE_COMMIT: ToyKernel()})]).bind(_ctx(), [stale])
    assert stale.module.spec_verify_kernel is None


@pytest.mark.parametrize("plan_kind", ["chain", "tree"])
def test_bind_one_kernel_for_verify_and_commit(plan_kind):
    """A kernel a provider gives for both a recurrent layer's verify and its
    state commit is bound once to each layer, and commits above a chain."""
    kernel = ToyKernel()
    registry = KernelRegistry([Provider({GDN_VERIFY: kernel, STATE_COMMIT: kernel})])
    layers = [_mamba_layer("m.0"), _mamba_layer("m.1")]
    registry.bind(_ctx(plan_kind), layers)
    assert kernel.binds == [["m.0", "m.1"]]
    assert all(layer.module.spec_verify_kernel is kernel for layer in layers)
    step = CommitStep(result=None, num_reqs=2, rows=None, seq_lens=None)
    registry.commit(step)
    assert kernel.steps == ([step] if plan_kind == "tree" else [])


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
        self.b: list[torch.Tensor] = []


_HEADS = dict(num_k_heads=1, num_v_heads=2, head_k_dim=2, head_v_dim=3)


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
        calls.b.append(kwargs["b"])
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
    layer.conv1d = torch.nn.Conv1d(5, 5, 4, groups=5, bias=False)
    layer.kv_cache = (torch.zeros(2, 5, 3), torch.zeros(2, 2, 3, 3))
    layer.A_log = torch.zeros(2)
    layer.dt_bias = torch.ones(2)
    layer.tp_size = 1
    for name, value in _HEADS.items():
        setattr(layer, name, value)
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
    assert {name: getattr(rows, name) for name in _HEADS} == _HEADS
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


def test_gdn_decode_keeps_b_and_a_strided(monkeypatch):
    """Without spec rows the layer hands ``b`` and ``a`` to the kernels as the
    strided views of ``ba`` they are, as on main."""
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        non_spec_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.zeros(2, dtype=torch.int32),
    )
    calls = _Calls()
    layer = _layer(monkeypatch, calls, metadata)
    b, a = torch.zeros(2, 4).chunk(2, dim=-1)
    layer._forward_core(
        mixed_qkv=torch.zeros(2, 5), b=b, a=a, core_attn_out=torch.zeros(2, 2, 3)
    )
    assert [x.data_ptr() for x in calls.b] == [b.data_ptr()]
    assert not calls.b[0].is_contiguous()
