# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the GLM-5.3-Flash specialized decode: config refusals, the per-step
plan, cache binding and the fail-stop. The layer builders are the kit's fakes."""

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.glm5next.amd.specialized_decode.runtime import (
    decode,
    failstop,
    spec,
)
from vllm.models.glm5next.amd.specialized_decode.runtime.decode import (
    SpecializedDecode,
)

from . import kit

STAGES = ("a", "b")
PLATFORM = ("needs ROCm", "needs gfx950")


def _reasons(cfg) -> list[str]:
    return [r for r in spec.refusal_reasons(cfg) if r not in PLATFORM]


def test_supported_config_passes():
    assert _reasons(kit.make_vllm_config()) == []


def test_every_reason_is_reported_at_once():
    cfg = kit.make_vllm_config(
        parallel=dict(
            pipeline_parallel_size=2, enable_eplb=True, tensor_parallel_size=16
        ),
        compilation=dict(cudagraph_capture_sizes=[1, 2, 3, 4, 8]),
        cache=dict(cache_dtype="fp8_e5m2", mamba_ssm_cache_dtype="bfloat16"),
        hf=dict(hidden_size=2048, hc_mult=2),
        lora_config=object(),
        speculative_config=object(),
        use_v2_model_runner=False,
    )
    reasons = _reasons(cfg)
    for expected in (
        "needs Model Runner V2",
        "pipeline parallel",
        "EPLB",
        "LoRA",
        "speculative decoding",
        "capture sizes [3]",
        "needs hidden_size 4096",
        "needs hc_mult 4",
        "tensor parallel size",
        "fp8_e5m2 KV cache",
        "bfloat16 KDA recurrent state",
    ):
        assert any(expected in r for r in reasons), (expected, reasons)


@pytest.mark.parametrize("cache", ["auto", "bfloat16", "fp8", "fp8_e4m3", "mxfp4_mla"])
def test_bf16_and_e4m3_kv_caches_pass(cache):
    assert _reasons(kit.make_vllm_config(cache=dict(cache_dtype=cache))) == []


@pytest.mark.parametrize(
    "compilation, ok",
    [
        (dict(cudagraph_mode=CUDAGraphMode.PIECEWISE), False),
        (dict(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY), True),
        (dict(cudagraph_mode=CUDAGraphMode.NONE), True),
    ],
)
def test_graph_modes(compilation, ok):
    assert (_reasons(kit.make_vllm_config(compilation=compilation)) == []) is ok


def test_switch_off_is_none_and_on_raises_with_every_reason():
    off = kit.make_vllm_config()
    off.kernel_config.monokernel = False
    assert spec.specialized_decode_config(off) is None
    cfg = kit.make_vllm_config(lora_config=object(), use_v2_model_runner=False)
    with pytest.raises(ValueError, match="needs Model Runner V2.*LoRA"):
        spec.specialized_decode_config(cfg)


def _decode(monkeypatch, log=None, rank=0, **fakes):
    log = [] if log is None else log
    kit.patch_layers(monkeypatch, log, rank=rank, **fakes)
    backbone = kit.FakeBackbone()
    d = SpecializedDecode(backbone, spec.DecodeConfig(tp_size=4))
    d.bind_caches()
    return d, backbone, log


def _plan(monkeypatch, d, fc, tokens):
    monkeypatch.setattr(decode, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode, "get_forward_context", lambda: fc)
    return d.plan(torch.zeros(tokens, dtype=torch.long), None)


def _names(backbone):
    return [backbone.layers[i].attn_name for i in kit.KERNEL_LAYERS]


@pytest.mark.parametrize("tokens, width", [(1, 2), (2, 2), (4, 4), (8, 8)])
def test_decode_step_runs_the_kernel_layers(monkeypatch, tokens, width):
    d, backbone, log = _decode(monkeypatch)
    fc = kit.forward_context(kit.decode_metadata(_names(backbone)))
    step = _plan(monkeypatch, d, fc, tokens)
    assert step is not None and step.step.width == width
    hidden, _ = backbone.forward_layers(step, tokens)
    assert backbone.layers[0].calls == backbone.layers[4].calls == 1
    assert all(backbone.layers[i].calls == 0 for i in kit.KERNEL_LAYERS)
    assert [k.launches for k in d.kernels[width].values()] == [
        [(i, tokens, width)] for i in kit.KERNEL_LAYERS
    ]
    assert torch.equal(hidden, torch.full((tokens, 4), 5.0))
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
    d, backbone, _ = _decode(monkeypatch)
    md = kit.decode_metadata(_names(backbone), **case.get("metadata", {}))
    if case.get("no_metadata"):
        md = None
    if case.get("micro_batched"):
        md = [md, md]
    fc = kit.forward_context(md, case.get("mode", CUDAGraphMode.FULL))
    tokens = case.get("tokens", 2)
    assert _plan(monkeypatch, d, fc, tokens) is None
    backbone.forward_layers(None, tokens)
    assert all(layer.calls == 1 for layer in backbone.layers)


def test_pipeline_inputs_run_vllm_layers(monkeypatch):
    d, backbone, _ = _decode(monkeypatch)
    monkeypatch.setattr(decode, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        decode,
        "get_forward_context",
        lambda: kit.forward_context(kit.decode_metadata(_names(backbone))),
    )
    assert d.plan(torch.zeros(2, dtype=torch.long), object()) is None


def test_layer_eligibility_can_decline(monkeypatch):
    d, backbone, _ = _decode(monkeypatch, eligible_reason="long context")
    fc = kit.forward_context(kit.decode_metadata(_names(backbone)))
    assert _plan(monkeypatch, d, fc, 2) is None
    assert failstop.counters["vllm_path:long context"] >= 1


def test_decision_matches_on_every_rank(monkeypatch):
    per_rank = []
    for rank in range(4):
        d, backbone, _ = _decode(monkeypatch, rank=rank)
        names = _names(backbone)
        steps = [
            (kit.decode_metadata(names), 4),
            (kit.decode_metadata(names, num_prefills=1), 4),
            (kit.decode_metadata(names), 3),
        ]
        per_rank.append(
            [
                _plan(monkeypatch, d, kit.forward_context(md), n) is not None
                for md, n in steps
            ]
        )
    assert per_rank == [[True, False, False]] * 4


def test_moved_caches_are_rebound_and_rechecked(monkeypatch):
    d, backbone, log = _decode(monkeypatch)
    fc = kit.forward_context(kit.decode_metadata(_names(backbone)))
    for _ in range(2):
        _plan(monkeypatch, d, fc, 2)
    d.bind_caches()
    assert sum(e[0] == "bind" for e in log) == 1
    backbone.layers[2].cache = torch.zeros(16, 4)
    _plan(monkeypatch, d, fc, 2)
    assert sum(e[0] == "bind" for e in log) == 2
    backbone.layers[3].cache = torch.zeros(8, 4, dtype=torch.float16)
    with pytest.raises(ValueError, match="3.kv: dtype torch.float16"):
        _plan(monkeypatch, d, fc, 2)


def test_caches_index_into_tuples_and_skip_other_layers(monkeypatch):
    from vllm.models.glm5next.amd.specialized_decode.runtime import layers

    log: list = []
    kit.patch_layers(monkeypatch, log)
    when = lambda layer: layer.idx != 2  # noqa: E731
    monkeypatch.setattr(layers, "CACHES", {"rec": ("pair[1]", (torch.float32,), when)})
    backbone = kit.FakeBackbone()
    for i in (1, 3):
        backbone.layers[i].pair = (torch.zeros(2), torch.zeros(3))
    d = SpecializedDecode(backbone, spec.DecodeConfig(tp_size=4))
    d.bind_caches()
    assert ("bind", ("1.rec", "3.rec")) in log
    assert d.caches()["3.rec"].numel() == 3


def test_bind_reasons_raise(monkeypatch):
    with pytest.raises(ValueError, match="bad layout"):
        _decode(monkeypatch, bind_reasons=("bad layout",))


def test_model_reasons_raise_all_at_once(monkeypatch):
    with pytest.raises(ValueError, match="x; y"):
        _decode(monkeypatch, model_reasons=("x", "y"))


def test_health_words_cover_every_stage(monkeypatch):
    d, _, _ = _decode(monkeypatch)
    assert d.health().tolist() == [0] * len(spec.STAGES)


def _watch(monkeypatch, rank, unsafe=False, flags=True):
    monkeypatch.setenv("VLLM_MONO_UNSAFE_WARN", "1" if unsafe else "0")
    words = [torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)]
    f = torch.zeros(4, dtype=torch.int32) if flags else None
    return failstop.HealthWatch("fake", words, STAGES, rank, f), words, f


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
        RuntimeError, match=r"expired waits in b; ranks flagging an expiry: \[2\]"
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
    from vllm.models.glm5next.amd.specialized_decode.mono import abi

    monkeypatch.delenv(abi.FAULT_INJECT_ENV, raising=False)
    assert abi.fault_spec() is None
    monkeypatch.setenv(abi.FAULT_INJECT_ENV, "rank=1,step=5,us=2000")
    assert abi.fault_spec() == abi.FaultSpec(1, 5, 2000)
    monkeypatch.setenv(abi.FAULT_INJECT_ENV, "rank=1")
    with pytest.raises(ValueError, match="rank=R,step=N,us=D"):
        abi.fault_spec()
