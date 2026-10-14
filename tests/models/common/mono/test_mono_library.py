# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the shared MonoKernel runtime, with the kit's fake spec."""

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.mono import Feature, failstop, guards, layer
from vllm.models.common.mono.layer import MonoModel

from . import kit


def test_supported_config_passes():
    assert guards.refuse_config(kit.make_vllm_config(), kit.make_spec(), 256) == []


def test_every_reason_is_reported_at_once():
    cfg = kit.make_vllm_config(
        parallel=dict(pipeline_parallel_size=2, enable_eplb=True),
        compilation=dict(cudagraph_capture_sizes=[1, 2, 3, 4, 8]),
        lora_config=object(),
        use_v2_model_runner=False,
    )
    spec = kit.make_spec(config_reasons=("spec says no",))
    reasons = guards.refuse_config(cfg, spec, compute_units=128)
    for expected in (
        "needs Model Runner V2",
        "pipeline parallel",
        "EPLB",
        "LoRA",
        "capture sizes [3]",
        "needs 256 compute units",
        "spec says no",
    ):
        assert any(expected in r for r in reasons), (expected, reasons)


def test_features_are_default_deny():
    cfg = kit.make_vllm_config(lora_config=object())
    assert "LoRA" in guards.refuse_config(cfg, kit.make_spec(), 256)
    supported = kit.make_spec(supports=frozenset({Feature.LORA}))
    assert guards.refuse_config(cfg, supported, 256) == []


@pytest.mark.parametrize(
    "compilation, ok",
    [
        (dict(cudagraph_mode=CUDAGraphMode.PIECEWISE), False),
        (dict(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY), True),
        (dict(cudagraph_mode=CUDAGraphMode.NONE), True),
    ],
)
def test_graph_modes(compilation, ok):
    cfg = kit.make_vllm_config(compilation=compilation)
    assert (guards.refuse_config(cfg, kit.make_spec(), 256) == []) is ok


def test_configure_raises_with_every_reason(monkeypatch):
    monkeypatch.setattr(guards, "enabled", lambda vllm_config: True)
    spec = kit.make_spec(config_reasons=("a", "b"))
    monkeypatch.setattr(guards, "load_spec", lambda path: spec)
    with pytest.raises(ValueError, match="a; b"):
        guards.configure(kit.make_vllm_config(), "x:y")


def _mono(monkeypatch, spec=None, log=None, rank=0):
    log = [] if log is None else log
    spec = spec or kit.make_spec(log=log)
    monkeypatch.setattr("vllm.distributed.get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 4
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tp_group",
        lambda: type("G", (), {"cpu_group": None})(),
    )
    model = kit.FakeModel()
    return MonoModel(model, kit.make_vllm_config(), spec), model, log


def _run(monkeypatch, mono, model, fc, tokens):
    monkeypatch.setattr(layer, "get_forward_context", lambda: fc)
    hidden, residual = torch.zeros(tokens, 4), None
    positions = torch.zeros(tokens, dtype=torch.long)
    for i, lay in enumerate(model.layers):
        hidden, residual = layer.forward_layer(
            mono, i, lay, positions, hidden, residual
        )
    return hidden, residual


@pytest.mark.parametrize("tokens, width", [(1, 2), (2, 2), (4, 4), (8, 8)])
def test_decode_step_runs_the_kernel_layers(monkeypatch, tokens, width):
    mono, model, log = _mono(monkeypatch)
    names = [model.layers[i].attn_name for i in (1, 2, 3)]
    fc = kit.forward_context(kit.decode_metadata(names))
    hidden, residual = _run(monkeypatch, mono, model, fc, tokens)
    assert model.layers[0].calls == 1
    assert all(model.layers[i].calls == 0 for i in (1, 2, 3))
    assert mono.step is not None and mono.step.width == width
    assert [k.launches for k in mono.kernels[width].values()] == [
        [(i, tokens, width)] for i in (1, 2, 3)
    ]
    assert torch.equal(hidden, torch.full((tokens, 4), 4.0))
    assert ("pack",) in log and sum(e[0] == "build" for e in log) == 9


@pytest.mark.parametrize(
    "case",
    [
        dict(tokens=3),
        dict(tokens=16),
        dict(metadata=dict(num_prefills=1)),
        dict(metadata=dict(max_query_len=2)),
        dict(mode=CUDAGraphMode.PIECEWISE),
        dict(no_metadata=True),
        dict(micro_batched=True),
    ],
)
def test_other_steps_run_vllm_layers(monkeypatch, case):
    mono, model, _ = _mono(monkeypatch)
    names = [model.layers[i].attn_name for i in (1, 2, 3)]
    md = kit.decode_metadata(names, **case.get("metadata", {}))
    if case.get("no_metadata"):
        md = None
    if case.get("micro_batched"):
        md = [md, md]
    fc = kit.forward_context(md, case.get("mode", CUDAGraphMode.FULL))
    _run(monkeypatch, mono, model, fc, case.get("tokens", 2))
    assert mono.step is None and not mono.kernels
    assert all(lay.calls == 1 for lay in model.layers)


def test_spec_eligibility_can_decline(monkeypatch):
    mono, model, _ = _mono(monkeypatch, kit.make_spec(eligible_reason="long context"))
    names = [model.layers[i].attn_name for i in (1, 2, 3)]
    _run(monkeypatch, mono, model, kit.forward_context(kit.decode_metadata(names)), 2)
    assert mono.step is None
    assert failstop.counters["vllm_path:long context"] >= 1


def test_decision_matches_on_every_rank(monkeypatch):
    per_rank = []
    for rank in range(4):
        mono, model, _ = _mono(monkeypatch, rank=rank)
        names = [model.layers[i].attn_name for i in (1, 2, 3)]
        steps = [
            (kit.decode_metadata(names), 4),
            (kit.decode_metadata(names, num_prefills=1), 4),
            (kit.decode_metadata(names), 3),
        ]
        decisions = []
        for md, tokens in steps:
            _run(monkeypatch, mono, model, kit.forward_context(md), tokens)
            decisions.append(mono.step is not None)
        per_rank.append(decisions)
    assert per_rank == [[True, False, False]] * 4


def test_moved_caches_are_rebound_and_rechecked(monkeypatch):
    log = []
    mono, model, _ = _mono(monkeypatch, log=log)
    names = [model.layers[i].attn_name for i in (1, 2, 3)]
    md = kit.decode_metadata(names)
    for _ in range(2):
        _run(monkeypatch, mono, model, kit.forward_context(md), 2)
    assert sum(e[0] == "bind" for e in log) == 1
    model.layers[2].cache = torch.zeros(16, 4)
    _run(monkeypatch, mono, model, kit.forward_context(md), 2)
    assert sum(e[0] == "bind" for e in log) == 2
    model.layers[3].cache = torch.zeros(8, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="3.kv: dtype torch.float16"):
        _run(monkeypatch, mono, model, kit.forward_context(md), 2)


def test_caches_index_into_tuples_and_skip_other_layers(monkeypatch):
    from dataclasses import replace

    from vllm.models.common.mono import CacheReq

    log = []
    req = CacheReq("pair[1]", (torch.float32,), when=lambda layer: layer.idx != 2)
    spec = replace(kit.make_spec(log=log), caches={"rec": req})
    mono, model, _ = _mono(monkeypatch, spec)
    for i in (1, 3):
        model.layers[i].pair = (torch.zeros(2), torch.zeros(3))
    names = [model.layers[i].attn_name for i in (1, 2, 3)]
    _run(monkeypatch, mono, model, kit.forward_context(kit.decode_metadata(names)), 2)
    assert ("bind", ("1.rec", "3.rec")) in log
    assert mono.router.caches()["3.rec"].numel() == 3


def test_bind_reasons_raise(monkeypatch):
    mono, model, _ = _mono(monkeypatch, kit.make_spec(bind_reasons=("bad layout",)))
    names = [model.layers[i].attn_name for i in (1, 2, 3)]
    with pytest.raises(ValueError, match="bad layout"):
        _run(
            monkeypatch, mono, model, kit.forward_context(kit.decode_metadata(names)), 2
        )


def test_model_reasons_raise_all_at_once(monkeypatch):
    with pytest.raises(ValueError, match="x; y"):
        _mono(monkeypatch, kit.make_spec(model_reasons=("x", "y")))


def test_switch_off_is_the_vllm_layer():
    model = kit.FakeModel()
    hidden, residual = torch.zeros(2, 4), None
    for i, lay in enumerate(model.layers):
        hidden, residual = layer.forward_layer(None, i, lay, None, hidden, residual)
    assert all(lay.calls == 1 for lay in model.layers)
    assert layer.health(None) is None


def _watch(monkeypatch, rank, unsafe=False, flags=True):
    monkeypatch.setenv("VLLM_MONO_UNSAFE_WARN", "1" if unsafe else "0")
    words = [torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)]
    f = torch.zeros(4, dtype=torch.int32) if flags else None
    return failstop.HealthWatch("fake", words, kit.STAGES, rank, f), words, f


def test_clean_words_pass(monkeypatch):
    watch, _, _ = _watch(monkeypatch, 0)
    watch.check([[0, 0], [0, 0], [0, 0, 0, 0]])
    assert watch.health().tolist() == [0] * 8


def test_output_rank_raises_and_schedules_exit(monkeypatch):
    armed = []
    monkeypatch.setattr(
        failstop.faulthandler, "dump_traceback_later", lambda *a, **k: armed.append(k)
    )
    watch, words, _ = _watch(monkeypatch, 0)
    with pytest.raises(
        RuntimeError, match=r"expired waits in ffn; ranks flagging an expiry: \[2\]"
    ):
        watch.check([[0, 0], [0, 1], [0, 0, 1, 0]])
    assert armed == [{"exit": True}]


def test_other_ranks_exit_their_worker(monkeypatch):
    def exit_(code):
        raise SystemExit(code)

    monkeypatch.setattr(failstop.os, "_exit", exit_)
    watch, _, _ = _watch(monkeypatch, 2, flags=False)
    with pytest.raises(SystemExit) as exited:
        watch.check([[1, 0], [0, 0]])
    assert exited.value.code == failstop.WORKER_EXIT_CODE


def test_unsafe_warn_logs_and_clears(monkeypatch):
    watch, words, flags = _watch(monkeypatch, 0, unsafe=True)
    words[0][1] = 1
    flags[3] = 1
    watch.check([[0, 1], [0, 0], [0, 0, 0, 1]])
    assert not words[0].any() and not flags.any()


def test_fault_spec(monkeypatch):
    from vllm.models.common.mono import abi

    monkeypatch.delenv(abi.FAULT_INJECT_ENV, raising=False)
    assert abi.fault_spec() is None
    monkeypatch.setenv(abi.FAULT_INJECT_ENV, "rank=1,step=5,us=2000")
    assert abi.fault_spec() == abi.FaultSpec(1, 5, 2000)
    monkeypatch.setenv(abi.FAULT_INJECT_ENV, "rank=1")
    with pytest.raises(ValueError, match="rank=R,step=N,us=D"):
        abi.fault_spec()
