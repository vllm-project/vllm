# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the specialized decode's config check (runtime/__init__.py), executor
lifecycle (runtime/decode.py), build / KV-bind guards (runtime/guards.py) and the
amd/model.py + amd/model_state.py hooks: the decode_backend switch, every unsupported()
reason (reported together), the DecodeConfig options, FULL-graph capture sizes, the
RoPE / KV / MLA-cache checks, the output-rank fail-stop latch + exit watchdog and the
poll-error watch."""

import subprocess
import sys
import time
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

from vllm.config.compilation import CompilationMode
from vllm.models.deepseek_v32.amd.specialized_decode import runtime as R
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import DecodeConfig
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import decode as D
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import guards as G


@pytest.fixture
def gfx950(monkeypatch):
    """The platform probe passes (the CPU test host is not a gfx950)."""
    monkeypatch.setattr(R, "_platform_reason", lambda: None)


def _set(vc, path: str, value):
    """Set one (dotted) VllmConfig field in place; returns vc."""
    *parents, name = path.split(".")
    obj = vc
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, name, value)
    return vc


# (VllmConfig field, unsupported value, reason fragment)
REASONS = [
    ("model_config.hf_config.model_type", "deepseek_v32", "model_type deepseek_v32"),
    ("parallel_config.tensor_parallel_size", 4, "TP 4"),
    ("parallel_config.pipeline_parallel_size", 2, "pipeline parallelism"),
    ("parallel_config.data_parallel_size", 2, "data parallelism"),
    ("parallel_config.enable_expert_parallel", True, "expert parallelism"),
    ("parallel_config.decode_context_parallel_size", 2, "decode context"),
    ("parallel_config.prefill_context_parallel_size", 2, "prefill context"),
    ("parallel_config.use_ubatching", True, "DBO"),
    ("speculative_config", object(), "speculative decoding"),
    ("model_config.enable_sleep_mode", True, "sleep mode"),
    ("lora_config", object(), "LoRA"),
    ("kv_transfer_config", object(), "KV transfer"),
    ("aux_output_config.enable_return_routed_experts", True, "routed-expert capture"),
    ("model_config.dtype", torch.float16, "dtype torch.float16"),
    ("cache_config.cache_dtype", "fp8", "kv_cache_dtype fp8"),
    ("use_v2_model_runner", False, "V1 model runner"),
    ("compilation_config.mode", CompilationMode.STOCK_TORCH_COMPILE, "torch.compile"),
]


@pytest.mark.usefixtures("gfx950")
@pytest.mark.parametrize("path, value, frag", REASONS, ids=[r[0] for r in REASONS])
def test_unsupported_each_reason(make_vc, path, value, frag):
    """Each unsupported deployment yields its reason, and only that one; startup
    raises with it."""
    vc = _set(make_vc(), path, value)
    reasons = R.unsupported(vc)
    assert len(reasons) == 1 and frag in reasons[0], reasons
    with pytest.raises(ValueError, match="does not support"):
        R.specialized_decode_config(vc)


@pytest.mark.usefixtures("gfx950")
def test_unsupported_fused_shared_experts(make_vc, monkeypatch):
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(
        rocm_aiter_ops, "is_fusion_moe_shared_experts_enabled", lambda: True
    )
    (why,) = R.unsupported(make_vc())
    assert "fused shared experts" in why


def test_unsupported_platform_and_config(make_vc, monkeypatch):
    """The platform probe (CPU host: not gfx950), step_sync under FULL graphs and
    malformed options are reasons too."""
    assert any("gfx950" in r for r in R.unsupported(make_vc()))
    monkeypatch.setattr(R, "_platform_reason", lambda: None)
    (why,) = R.unsupported(make_vc(graphs="full", options={"step_sync": True}))
    assert "step_sync" in why
    for options, frag in (
        ({"no_such_field": 1}, "no_such_field"),
        ({"indexer_mode": "x"}, "indexer_mode"),
        ({"failstop": "1"}, "failstop"),
    ):
        (why,) = R.unsupported(make_vc(options=options))
        assert '"specialized_decode"' in why and frag in why, why


def test_all_reasons_reported(make_vc):
    """Startup lists every reason, not only the first."""
    vc = make_vc(tp=4, pp=2, kv="fp8")
    _set(_set(vc, "lora_config", object()), "use_v2_model_runner", False)
    with pytest.raises(ValueError) as e:
        R.specialized_decode_config(vc)
    for frag in ("TP 4", "pipeline", "kv_cache_dtype", "LoRA", "V1 model runner"):
        assert frag in str(e.value), (frag, str(e.value))
    assert "gfx950" in str(e.value)  # the platform probe of the CPU host


@pytest.mark.usefixtures("gfx950")
def test_supported_config(make_vc, monkeypatch):
    """No reason: a DecodeConfig for the graph mode."""
    assert R.unsupported(make_vc()) == []
    dc = R.specialized_decode_config(make_vc())
    assert isinstance(dc, DecodeConfig)
    assert dc.batch_sizes == (1, 2, 4, 5, 6, 8)
    assert dc.step_sync
    assert dc.max_model_len == 4096
    dc = R.specialized_decode_config(make_vc(graphs="full"))
    assert dc.step_sync is False


@pytest.mark.parametrize("backend", [None, "auto"])
def test_switch_off_runs_no_checks(make_vc, monkeypatch, backend):
    """decode_backend unset / auto: None, unsupported() never runs."""
    monkeypatch.setattr(R, "_reasons_and_config", lambda vc: pytest.fail("checks ran"))
    assert R.specialized_decode_config(make_vc(backend=backend, tp=4)) is None
    vc = make_vc()
    vc.additional_config = None  # VllmConfig's default may be a non-dict
    assert R.decode_backend(vc) == "auto"


def test_bad_backend(make_vc):
    with pytest.raises(ValueError, match="'auto' or 'latency_specialized'"):
        R.specialized_decode_config(make_vc(backend="fast"))


def test_decode_config_options(make_vc):
    """additional_config["specialized_decode"] overrides the defaults."""
    options = {"indexer_mode": "attn"}
    assert R.decode_config(make_vc(options=options)).indexer_mode == "attn"
    # the RoPE table length is raised to vLLM's max_model_len
    cfg = R.decode_config(make_vc(mml=8000, options={"max_model_len": 1024}))
    assert cfg.max_model_len == 8000


def test_moe_refusal():
    """The kernel reads the MoE weights in place: a fused shared expert or another
    MoE backend is refused when the executor is built."""
    mlp = lambda be="AITER_MXFP4_MXFP4", sh=True: NS(
        shared_experts=object() if sh else None,
        experts=NS(routed_experts=NS(quant_method=NS(mxfp4_backend=NS(value=be)))),
    )
    model = lambda m: NS(layers={L: NS(mlp=m) for L in range(3, 78)})
    cfg = DecodeConfig()
    assert D.moe_refusal(model(mlp()), cfg) is None
    assert "AITER_MXFP4_BF16" in D.moe_refusal(model(mlp("AITER_MXFP4_BF16")), cfg)
    assert "shared" in D.moe_refusal(model(mlp(sh=False)), cfg)
    with pytest.raises(ValueError, match="shared-expert"):
        D.SpecializedDecode(model(mlp(sh=False)), cfg)


def fake_model(rope_rows=8192):
    rope = NS(cos_sin_cache=torch.zeros(rope_rows, 64))
    return NS(layers={L: NS(self_attn=NS(rotary_emb=rope)) for L in range(78)})


def test_check_rope_table():
    G.check_rope_table(fake_model(), DecodeConfig(max_model_len=8192))
    with pytest.raises(RuntimeError, match="cos_sin_cache"):
        G.check_rope_table(fake_model(rope_rows=4096), DecodeConfig(max_model_len=8000))


def test_piecewise_graphs_only(monkeypatch):
    from vllm import envs

    def vc(name, full):
        return NS(
            compilation_config=NS(
                cudagraph_mode=NS(name=name, has_full_cudagraphs=lambda: full)
            )
        )

    env = envs.environment_variables
    monkeypatch.setitem(env, "VLLM_USE_BREAKABLE_CUDAGRAPH", lambda: True)
    assert R.piecewise_graphs_only(vc("PIECEWISE", False))
    assert not R.piecewise_graphs_only(vc("NONE", False))
    assert not R.piecewise_graphs_only(vc("FULL", True))
    monkeypatch.setitem(env, "VLLM_USE_BREAKABLE_CUDAGRAPH", lambda: False)
    assert not R.piecewise_graphs_only(vc("PIECEWISE", False))


def test_output_rank_failstop_latch(failstop_latch):
    """The output rank latches its first fail-stop and arms the exit watchdog once."""
    for msg in ("first", "second"):
        with pytest.raises(RuntimeError, match=f"{msg} -> fail-stop"):
            G.fail_stop(msg, rank=0)
    assert G._FAILED["msg"] == "first -> fail-stop" and len(failstop_latch) == 1


def test_exit_watchdog_exits_worker():
    """The watchdog ends a process whose main thread is stuck (a sleep standing in for a
    device sync waiting on exited peers) and dumps its stack."""
    code = (
        "import time\n"
        "from vllm.models.deepseek_v32.amd.specialized_decode.runtime.guards import "
        "_arm_exit_watchdog\n"
        "_arm_exit_watchdog(0.5)\n"
        "time.sleep(60)\n"
    )
    t0 = time.monotonic()
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=50
    )
    assert p.returncode != 0 and time.monotonic() - t0 < 50
    assert "time.sleep" in p.stderr or "<string>" in p.stderr, p.stderr[-2000:]


def test_poll_watch(monkeypatch):
    """The executor owns a PollErrorWatch advanced by after_step (from compute_logits);
    a tripped watch raises; failstop="off" builds none."""

    class FakeWatch:
        def __init__(self, mono):
            self.mono, self.n, self.trip = mono, 0, False

        def after_step(self):
            self.n += 1
            if self.trip:
                raise RuntimeError("expired kernel polls -> fail-stop")

    monkeypatch.setattr(G, "PollErrorWatch", FakeWatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for mode in ("raise", "off"):
        obj = D.SpecializedDecode.__new__(D.SpecializedDecode)
        obj.mono, obj.cfg = object(), DecodeConfig(failstop=mode)
        obj.watch = obj._poll_watch()
        obj.after_step()
        if mode == "off":
            assert obj.watch is None
            continue
        assert obj.watch.mono is obj.mono and obj.watch.n == 1
        obj.watch.trip = True
        with pytest.raises(RuntimeError, match="fail-stop"):
            obj.after_step()


def _rec(log: list, item, ret=None):
    log.append(item)
    return ret


def _bare_lm():
    from vllm.models.deepseek_v32.amd import model as M

    lm = M.DeepseekV32ForCausalLM.__new__(M.DeepseekV32ForCausalLM)
    torch.nn.Module.__init__(lm)
    lm.specialized_decode_config = lm.specialized_decode = None
    return lm


def test_causal_lm_hooks(monkeypatch):
    """amd/model.py: the config check runs before the modules are built,
    compute_logits advances the watch, a weight reload raises once the executor
    exists, and forward plans the step and hands it to the backbone."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
    from vllm.models.deepseek_v32.amd import model as M

    order: list = []
    monkeypatch.setattr(
        M, "specialized_decode_config", lambda vc: _rec(order, "check", "cfg")
    )
    monkeypatch.setattr(
        DeepseekV2ForCausalLM,
        "__init__",
        lambda self, **kw: torch.nn.Module.__init__(self) or _rec(order, "modules"),
    )
    lm = M.DeepseekV32ForCausalLM(vllm_config=object())
    assert order == ["check", "modules"]
    assert lm.specialized_decode_config == "cfg" and lm.specialized_decode is None

    lm = _bare_lm()
    calls = []
    lm.lm_head, lm.logits_processor = None, lambda head, h: h * 2
    lm.compute_logits(torch.ones(2))
    lm.process_weights_after_loading()  # first load: nothing to do
    lm.specialized_decode = NS(
        after_step=lambda: calls.append("after_step"),
        plan=lambda p, it: _rec(calls, ("plan", it), "step"),
    )
    assert torch.equal(lm.compute_logits(torch.ones(2)), torch.full((2,), 2.0))
    assert calls == ["after_step"]
    with pytest.raises(RuntimeError, match="reloading"):
        lm.process_weights_after_loading()

    seen: list = []
    lm.model = lambda *a, step: _rec(seen, step, "h")
    assert lm.forward(None, torch.zeros(2)) == "h" and seen == ["step"]
    assert calls[-1] == ("plan", None)
    lm.specialized_decode = None
    lm.forward(None, torch.zeros(2))
    assert seen == ["step", None]


def test_layer_loop(monkeypatch):
    """DeepseekV32Model.forward: Step.layer(i)'s kernel replaces layer i; every layer
    is vLLM's without a step."""
    from vllm.models.deepseek_v32.amd import model as M

    monkeypatch.setattr(
        M, "get_pp_group", lambda: NS(is_first_rank=True, is_last_rank=True)
    )
    monkeypatch.setattr(M, "fused_allreduce_rms_norm", lambda h, r, norm: (h + r, None))
    calls: list = []

    def layer(i):
        return lambda p, h, r: _rec(calls, i, (h + 1, h if r is None else r + h))

    def kernel(i):
        return lambda p, h, r: _rec(calls, f"k{i}", (h + 1, r + h))

    me = NS(
        layers=[layer(i) for i in range(6)],
        start_layer=0,
        end_layer=6,
        aux_hidden_state_layers=(),
        embed_input_ids=lambda ids: ids.float(),
        norm=None,
    )
    step = D.Step(2, 2, {3: kernel(3), 4: kernel(4)})
    for st, want in ((None, [0, 1, 2, 3, 4, 5]), (step, [0, 1, 2, "k3", "k4", 5])):
        calls.clear()
        out = M.DeepseekV32Model.forward(me, torch.tensor([1.0, 2.0]), None, step=st)
        assert calls == want
        h, r = torch.tensor([1.0, 2.0]), None
        for _ in range(6):
            h, r = h + 1, (h if r is None else r + h)
        assert torch.equal(out, h + r)


def test_model_state(monkeypatch):
    """resolve_model_state_cls picks DeepseekV32ModelState; it builds the executor only
    when the switch selected it, and prepares it at every KV bind."""
    from vllm.models.deepseek_v32.amd.model_state import DeepseekV32ModelState
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    lm = _bare_lm()
    assert resolve_model_state_cls(None, lm) is DeepseekV32ModelState
    calls = []
    monkeypatch.setattr(
        DefaultModelState,
        "__init__",
        lambda self, vc, model, ec, dev: setattr(self, "model", model),
    )
    lm.build_specialized_decode = lambda: calls.append("build")
    DeepseekV32ModelState(None, lm, None, None).initialize_kv_cache(None, None)
    assert calls == []
    lm.specialized_decode_config = "cfg"
    ms = DeepseekV32ModelState(None, lm, None, None)
    lm.specialized_decode = NS(prepare_kv=lambda: calls.append("prepare_kv"))
    ms.initialize_kv_cache(None, None)
    ms.initialize_kv_cache(None, None)
    assert calls == ["build", "prepare_kv", "prepare_kv"]


def _quark(attention):
    """A QuarkConfig like the GLM-5.2 MXFP4 checkpoints: MXFP4 global, attention
    excluded ("bf16") or given ``attention`` (a weight / input spec pair)."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    mx = dict(
        dtype="fp4",
        qscheme="per_group",
        group_size=32,
        is_dynamic=False,
        scale_format="e8m0",
        ch_axis=-1,
    )
    cfg: dict[str, Any] = dict(
        global_quant_config=dict(weight=mx, input_tensors=dict(mx, is_dynamic=True)),
        layer_type_quant_config={},
        layer_quant_config={},
        exclude=["lm_head"],
    )
    names = R.ATTENTION_LINEARS
    if attention == "bf16":
        cfg["exclude"] += [  # explicit names, as the checkpoints list them
            f"model.layers.{L}.self_attn.{p}" for L in range(78) for p in names
        ]
    else:
        w, x = attention
        cfg["layer_quant_config"] = {
            f"*self_attn.{p}": dict(weight=w, input_tensors=x, output_tensors=None)
            for p in names
        }
    return QuarkConfig(quant_config=cfg)


_FP8_W = dict(dtype="fp8_e4m3", qscheme="per_channel", ch_axis=0, is_dynamic=False)
_FP8_X = dict(dtype="fp8_e4m3", qscheme="per_channel", ch_axis=1, is_dynamic=True)


@pytest.mark.usefixtures("gfx950")
def test_attention_quant_selection(make_vc):
    """BF16 (excluded) and Quark PTPC FP8 attention are served as checkpointed;
    block / per-tensor FP8 and non-Quark quantization are refused."""
    for ok in ("bf16", (_FP8_W, _FP8_X)):
        vc = _set(make_vc(), "quant_config", _quark(ok))
        assert R.unsupported(vc) == [], ok
    block = (
        dict(_FP8_W, qscheme="per_block", block_size=[128, 128], symmetric=True),
        dict(_FP8_X, qscheme="per_group", group_size=128, symmetric=True),
    )
    tensor = (dict(_FP8_W, qscheme="per_tensor"), dict(_FP8_X, qscheme="per_tensor"))
    for bad in (block, tensor):
        reasons = R.unsupported(_set(make_vc(), "quant_config", _quark(bad)))
        assert reasons and all("per-channel FP8 with per-token" in r for r in reasons)
        assert any("q_b_proj" in r and "layers 3..77" in r for r in reasons), reasons
    fp8 = NS(get_name=lambda: "fp8")  # e.g. zai-org/GLM-5.2-FP8 (block FP8)
    (why,) = R.unsupported(_set(make_vc(), "quant_config", fp8))
    assert "quantization fp8" in why


def _quark_split(fp8):
    """A QuarkConfig with Quark PTPC FP8 on the attention linears ``fp8`` gives as
    {layer: names} and BF16 (excluded) on all others."""
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    cfg = _quark("bf16").quant_config
    cfg["exclude"] = ["lm_head"] + [
        f"model.layers.{L}.self_attn.{p}"
        for L in range(78)
        for p in R.ATTENTION_LINEARS
        if p not in fp8.get(L, ())
    ]
    cfg["layer_quant_config"] = {
        f"*self_attn.{p}": dict(
            weight=_FP8_W, input_tensors=_FP8_X, output_tensors=None
        )
        for p in R.ATTENTION_LINEARS
    }
    return QuarkConfig(quant_config=cfg)


@pytest.mark.usefixtures("gfx950")
def test_attention_quant_partial_or_mixed(make_vc):
    """Per-layer partial FP8 attention and FP8 / BF16 mixed across mono layers are
    refused at startup; an FP8 or BF16 indexer wq_b alone is either path."""
    every = R.ATTENTION_LINEARS
    no_wq_b = tuple(p for p in every if p != "indexer.wq_b")
    for fp8 in (
        {L: every for L in range(78)},
        {L: no_wq_b for L in range(78)},
        {L: ("indexer.wq_b",) for L in range(78)},
    ):
        vc = _set(make_vc(), "quant_config", _quark_split(fp8))
        assert R.unsupported(vc) == [], fp8[3]
    partial = {L: every for L in range(78)} | {7: ("q_b_proj",)}
    (why,) = R.unsupported(_set(make_vc(), "quant_config", _quark_split(partial)))
    assert "partial FP8 attention on layers 7..7" in why and "'q_b_proj'" in why
    mixed = {L: every for L in range(3, 40)}
    (why,) = R.unsupported(_set(make_vc(), "quant_config", _quark_split(mixed)))
    assert "mixed attention precision" in why and "3..39" in why
