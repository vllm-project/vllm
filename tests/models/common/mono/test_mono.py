# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MonoKernel library on CPU, with a fake spec and kernel: refusals, the
capture sizes, routing, cache re-binding, parameter names, fail-stop, the
counters and the JIT warm-up."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.models.common.mono.kit import (
    FakeKernel,
    failed_words,
    fake_spec,
    fake_vllm_config,
)
from vllm.config import CUDAGraphMode
from vllm.models.common.mono import failstop, guards, layer, routing
from vllm.models.common.mono.spec import Feature, MonoStep, MonoUnsupported


class _Layer(nn.Module):
    def __init__(self, i: int):
        super().__init__()
        self.layer_idx = i
        self.weight = nn.Parameter(torch.full((4,), float(i)))

    def forward(self, positions, hidden_states, residual):
        return hidden_states - 1, residual


class _Model(nn.Module):
    def __init__(self, n: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(i) for i in range(n))
        self.start_layer, self.end_layer = 0, n
        self.config = SimpleNamespace()


@pytest.fixture
def one_rank(monkeypatch):
    """A single-rank TP group: TP agreement is a no-op."""
    import vllm.distributed

    tp = SimpleNamespace(rank_in_group=0, cpu_group=None)
    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: tp)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    failstop.STEPS.clear()
    failstop.INCIDENTS.clear()


def _forward_context(monkeypatch, mode="FULL", md=None):
    fc = SimpleNamespace(
        cudagraph_runtime_mode=getattr(CUDAGraphMode, mode),
        attn_metadata={} if md is None else md,
    )
    monkeypatch.setattr(routing, "get_forward_context", lambda: fc)
    monkeypatch.setattr(layer, "get_forward_context", lambda: fc)
    return fc


def _mono(spec=None, model=None, **cfg):
    return routing.Mono(spec or fake_spec(), model or _Model(), fake_vllm_config(**cfg))


# ---------------------------------------------------------------- refusals


def test_supported_config_is_accepted():
    assert guards.refuse_config(fake_spec(), fake_vllm_config()) == []


def test_every_refusal_is_reported_at_once():
    spec = fake_spec(refuse_config=lambda c: ["spec's own"])
    reasons = guards.refuse_config(
        spec,
        fake_vllm_config(
            **{
                "parallel_config.pipeline_parallel_size": 2,
                "parallel_config.data_parallel_size": 2,
                "parallel_config.enable_expert_parallel": True,
                "parallel_config.enable_eplb": True,
                "parallel_config.decode_context_parallel_size": 2,
                "parallel_config.prefill_context_parallel_size": 2,
                "lora_config": object(),
                "kv_transfer_config": object(),
                "aux_output_config.enable_return_routed_experts": True,
                "model_config.enable_sleep_mode": True,
                "offload_config.uva.cpu_offload_gb": 1,
                "weight_transfer_config": object(),
                "use_v2_model_runner": False,
                "compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE,
            }
        ),
    )
    # 12 unsupported features, the runner, the graph mode, the spec's own
    assert len(reasons) == 15, reasons
    assert reasons[-1] == "spec's own"


def test_a_supported_feature_is_not_refused():
    spec = fake_spec(
        supports=frozenset({Feature.SPECULATIVE_DECODING, Feature.DATA_PARALLEL})
    )
    cfg = fake_vllm_config(**{"parallel_config.data_parallel_size": 2})
    assert guards.refuse_config(spec, cfg) == []


def test_configure_raises_with_every_reason(monkeypatch):
    spec = fake_spec(refuse_config=lambda c: ["a", "b"])
    monkeypatch.setattr(guards, "load_spec", lambda path: spec)
    with pytest.raises(MonoUnsupported, match="a; b"):
        guards.configure(fake_vllm_config(), "x:SPEC")


def test_capture_sizes_up_to_the_widest_are_the_widths(monkeypatch):
    """Configure sets them rather than refusing; larger sizes stay."""
    monkeypatch.setattr(guards, "load_spec", lambda path: fake_spec())
    cfg = fake_vllm_config()
    guards.configure(cfg, "x:SPEC")
    assert cfg.compilation_config.cudagraph_capture_sizes == [4, *range(8, 17)]


# ---------------------------------------------------------------- routing


@pytest.mark.parametrize(
    "mode, tokens, eligible, why",
    [
        ("FULL", 4, None, None),
        ("NONE", 8, None, None),
        ("PIECEWISE", 4, None, "piecewise"),
        ("FULL", 6, None, "width"),
        ("NONE", 4, "not pure decode", "not pure decode"),
    ],
)
def test_routing_runs_pure_decode_widths_only(
    monkeypatch, one_rank, mode, tokens, eligible, why
):
    _forward_context(monkeypatch, mode)
    mono = _mono(fake_spec(eligible=lambda ctx, step: eligible))
    assert mono.begin_step(torch.zeros(tokens)) == why
    assert (mono.step is None) == (why is not None)


def test_every_rank_decides_alike(monkeypatch, one_rank):
    """The decision reads only the batch and the attention metadata."""
    md = {"attn": SimpleNamespace(num_prefills=0)}
    _forward_context(monkeypatch, "FULL", md)
    spec = fake_spec(
        eligible=lambda ctx, step: (
            "prefill" if step.attn_metadata["attn"].num_prefills else None
        )
    )
    ranks = [_mono(spec) for _ in range(8)]
    for n in (4, 6, 8):
        assert len({m.begin_step(torch.zeros(n)) for m in ranks}) == 1


def test_forward_layer_decides_once_a_forward(monkeypatch, one_rank):
    fc = _forward_context(monkeypatch, "FULL")
    calls: list[MonoStep] = []

    def eligible(ctx, step):
        calls.append(step)
        return None

    spec = fake_spec(eligible=eligible)
    model = _Model()
    mono = _mono(spec, model)
    h, r = torch.zeros(4), torch.zeros(4)
    pos = torch.zeros(4)
    out = [
        layer.forward_layer(mono, i, model.layers[i], pos, h, r)[0] for i in range(4)
    ]
    assert len(calls) == 1
    # layers 1, 2 have kernels (+1), layers 0, 3 run vLLM's (-1)
    assert [o[0].item() for o in out] == [-1.0, 1.0, 1.0, -1.0]
    fc2 = SimpleNamespace(**vars(fc))  # a new forward decides again
    monkeypatch.setattr(layer, "get_forward_context", lambda: fc2)
    monkeypatch.setattr(routing, "get_forward_context", lambda: fc2)
    layer.forward_layer(mono, 1, model.layers[1], pos, h, r)
    assert len(calls) == 2


def test_caches_are_rebound_when_they_move(monkeypatch, one_rank):
    _forward_context(monkeypatch, "FULL")
    binds: list[dict] = []

    def bind(ctx, caches):
        binds.append(caches)
        return []

    spec = fake_spec(bind=bind, caches=True)
    mono = _mono(spec)
    sfc = mono.ctx.vllm_config.compilation_config.static_forward_context
    for i in (1, 2):
        sfc[f"layers.{i}.attn"] = SimpleNamespace(kv_cache=torch.zeros(4))
    mono.begin_step(torch.zeros(4))
    mono.begin_step(torch.zeros(4))
    assert len(binds) == 1
    sfc["layers.1.attn"].kv_cache = torch.zeros(8)  # a profiling cache replaced
    mono.begin_step(torch.zeros(4))
    assert len(binds) == 2


def test_a_cache_refusal_raises(monkeypatch, one_rank):
    _forward_context(monkeypatch, "FULL")
    mono = _mono(fake_spec(bind=lambda ctx, caches: ["bad layout"], caches=True))
    sfc = mono.ctx.vllm_config.compilation_config.static_forward_context
    for i in (1, 2):
        sfc[f"layers.{i}.attn"] = SimpleNamespace(kv_cache=torch.zeros(4))
    with pytest.raises(MonoUnsupported, match="bad layout"):
        mono.begin_step(torch.zeros(4))


def test_a_model_refusal_raises(one_rank):
    with pytest.raises(MonoUnsupported, match="views"):
        _mono(fake_spec(refuse_model=lambda m: ["views"]))


def test_parameter_names_and_values_do_not_change(monkeypatch, one_rank):
    """Attach leaves the module tree, names and loaded values untouched."""
    model = _Model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(layer, "load_spec", lambda path: fake_spec())
    cfg = fake_vllm_config()
    model.mono = layer.attach(model, cfg, "x:SPEC")
    assert model.mono is not None
    after = model.state_dict()
    assert list(after) == list(before)
    assert all(torch.equal(after[k], v) for k, v in before.items())
    cfg.kernel_config.monokernel = False
    assert layer.attach(_Model(), cfg, "x:SPEC") is None


# ---------------------------------------------------------------- fail-stop


def test_a_failed_step_raises():
    failstop.check(failed_words(8, []))
    with pytest.raises(RuntimeError, match=r"TP ranks \[3\]"):
        failstop.check(failed_words(8, [3]))


def test_a_failed_step_is_counted_by_rank():
    failstop.INCIDENTS.clear()
    with pytest.raises(RuntimeError):
        failstop.check(failed_words(8, [2, 5]))
    assert failstop.drain() == {"incidents:2": 1, "incidents:5": 1}


def test_steps_are_counted_by_path(monkeypatch, one_rank):
    """A FULL replay runs no Python: its step counts the decision its graph
    baked in at capture."""
    _forward_context(monkeypatch, "FULL")
    mono = _mono(fake_spec(eligible=lambda ctx, step: None))
    mono.begin_step(torch.zeros(4))  # captured: mono
    mono.begin_step(torch.zeros(6))  # captured: width
    mono.account(4, CUDAGraphMode.FULL)
    mono.account(4, CUDAGraphMode.FULL)
    mono.account(6, CUDAGraphMode.FULL)
    _forward_context(monkeypatch, "NONE")
    mono.begin_step(torch.zeros(5))
    mono.account(5, CUDAGraphMode.NONE)
    assert failstop.drain() == {
        "steps:fake:mono": 2,
        "steps:fake:width": 2,
    }
    assert failstop.drain() is None


def test_a_capture_bakes_its_decision_in_any_runtime_mode(monkeypatch, one_rank):
    """Model Runner V2 captures FULL graphs with the runtime mode NONE."""
    _forward_context(monkeypatch, "NONE")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    mono = _mono(fake_spec(eligible=lambda ctx, step: None))
    mono.begin_step(torch.zeros(4))
    mono.account(4, CUDAGraphMode.FULL)
    assert failstop.drain() == {"steps:fake:mono": 1}


def test_every_width_is_registered_for_jit_warmup(monkeypatch, one_rank):
    registered = []
    monkeypatch.setattr(
        routing.MonoWarmup, "register_warmup", lambda self: registered.append(self)
    )
    mono = _mono()
    (warmup,) = registered
    warmup.warmup()
    compiled = {k for kern in mono.kernels.values() for k in kern.compiled}
    assert compiled == {("fake", 4), ("fake", 8)}


def test_fake_kernel_health_reaches_the_runner(one_rank):
    words = failed_words(8, [1])
    mono = _mono(fake_spec(words=words))
    assert isinstance(next(iter(mono.kernels.values())), FakeKernel)
    assert layer.health(mono) is words
    assert layer.health(None) is None
    with pytest.raises(RuntimeError):
        failstop.check(layer.health(mono))
