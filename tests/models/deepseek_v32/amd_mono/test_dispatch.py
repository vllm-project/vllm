# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the model-integrated dispatch (mono/dispatch.py) and the install guards
(mono/guards.py): env gating, refusal reasons, config from the env, checkpoint
resolution, FULL-graph capture sizes vs kernel widths, the KV / RoPE / MLA-cache install
checks, the output-rank fail-stop latch + exit watchdog, the poll-error watch and the
amd/model.py hooks."""

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.common.mono import register_mono_layer_op
from vllm.models.deepseek_v32.amd.mono import dispatch as D
from vllm.models.deepseek_v32.amd.mono import envs as E
from vllm.models.deepseek_v32.amd.mono import guards as G
from vllm.models.deepseek_v32.amd.mono.live import LiveConfig
from vllm.models.deepseek_v32.amd.mono.spec import GLM5_MONO


@pytest.mark.parametrize(
    "kw, frag",
    [
        (dict(model_type="deepseek_v32"), "model_type"),
        (dict(tp=4), "tensor parallel size 4"),
        (dict(pp=2), "pipeline parallelism"),
        (dict(ep=True), "expert parallelism"),
        (dict(spec=object()), "speculative decoding"),
        (dict(kv="fp8"), "kv cache dtype"),
        ("lora_config", "LoRA"),
        ("kv_transfer_config", "KV transfer"),
        ("routed", "routed-expert"),
        ("dtype", "dtype"),
    ],
)
def test_refusal(make_vc, no_platform_check, kw, frag):
    """The spec is the whole deployment gate, and it reports every reason at once."""
    vc = make_vc(**kw) if isinstance(kw, dict) else make_vc()
    if kw == "routed":
        vc.aux_output_config.enable_return_routed_experts = True
    elif kw == "dtype":
        vc.model_config.dtype = torch.float16
    elif isinstance(kw, str):
        setattr(vc, kw, object())
    why = GLM5_MONO.refuse(vc)
    assert why and any(frag in w for w in why), why
    assert not GLM5_MONO.refuse(make_vc())


def test_env_off_and_on(make_vc, no_platform_check, monkeypatch):
    """Off: create builds nothing. On: a refusal or a second creation (weight reload:
    the kernel holds a private weight copy) raises."""
    assert D.Glm5MonoDecode.create(make_vc(), model=object()) is None
    monkeypatch.setenv(E.ENABLE, "1")
    with pytest.raises(ValueError, match="tensor parallel size 4"):
        D.Glm5MonoDecode.create(make_vc(tp=4), model=object())
    model = object()
    register_mono_layer_op(object(), id(model))
    with pytest.raises(ValueError, match="already built"):
        D.Glm5MonoDecode.create(make_vc(), model=model)


@pytest.mark.parametrize(
    "val, want",
    [("1", True), ("true", True), ("TRUE", True), ("0", False), ("yes", False)],
)
def test_env_parsing(monkeypatch, val, want):
    from vllm import envs

    monkeypatch.setenv(E.ENABLE, val)
    assert envs.VLLM_ROCM_USE_GLM5_MONOKERNEL is want


def test_config_from_env(make_vc, ckpt, monkeypatch):
    monkeypatch.setenv(E.CONFIG, '{"sizes": [1, 2, 4, 8], "indexer_mode": "x"}')
    cfg = D.config_from_env(make_vc())
    assert cfg.sizes == (1, 2, 4, 8) and cfg.ckpt == ckpt and cfg.max_model_len == 4096
    assert cfg.indexer_mode == "x" and cfg.step_sync is True  # eager
    # graphs: no per-step host sync inside the capture
    assert (
        D.config_from_env(make_vc(graphs="full", sizes=(1, 2, 4, 8))).step_sync is False
    )
    monkeypatch.setenv(E.CONFIG, json.dumps(dict(ckpt="/explicit")))
    assert D.config_from_env(make_vc(model="/no/such")).ckpt == "/explicit"


def test_ckpt_resolution(make_vc, ckpt, tmp_path):
    """A local dir as is; an HF repo id -> the cached snapshot (no download); clear
    errors naming the fix otherwise."""
    assert D.resolve_ckpt_dir(make_vc()) == ckpt
    cache = tmp_path / "hub"
    repo = cache / "models--zai-org--GLM-5.2-test"
    snap = repo / "snapshots" / "0123abcd"
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("0123abcd")
    (snap / D.CKPT_INDEX).write_text("{}")
    got = D.resolve_ckpt_dir(make_vc(model="zai-org/GLM-5.2-test", download_dir=cache))
    assert os.path.realpath(got) == os.path.realpath(snap)
    (tmp_path / "empty").mkdir()
    for kw, frag in (
        (dict(model="zai-org/not-cached", download_dir=cache), "ckpt"),
        (dict(model="/no/such/dir"), "ckpt"),
        (dict(model=str(tmp_path / "empty")), D.CKPT_INDEX),
        (dict(load_format="dummy"), "dummy"),
    ):
        with pytest.raises(RuntimeError, match=frag) as e:
            D.resolve_ckpt_dir(make_vc(**kw))
        assert E.CONFIG in str(e.value)


def test_width_mismatch(make_vc, no_platform_check, monkeypatch):
    """Switch on + FULL-graph capture sizes != kernel widths -> a refusal naming the
    sizes to use; matching sizes (or eager) -> built."""

    class Fake(D.Glm5MonoDecode):
        def build(self):
            self.cfg = D.config_from_env(self.vllm_config, self.over)

    monkeypatch.setenv(E.ENABLE, "1")
    for sizes in ((1, 2, 4, 8, 16, 32), (1, 2, 4, 8), (1, 2, 4, 5, 6, 8, 16)):
        with pytest.raises(ValueError, match=r"capture_sizes=\[1, 2, 4, 5, 6, 8\]"):
            Fake.create(make_vc(graphs="full", sizes=sizes), model=object())
    # capture sizes that are all kernel widths: the message also offers matching sizes
    with pytest.raises(ValueError, match=r'"sizes": \[1, 2, 4, 8\]'):
        Fake.create(make_vc(graphs="full", sizes=(1, 2, 4, 8)), model=object())
    for over, sizes in (
        (None, (1, 2, 4, 5, 6, 8)),
        ('{"sizes": [1, 2, 4, 8]}', (1, 2, 4, 8)),
    ):
        if over:
            monkeypatch.setenv(E.CONFIG, over)
        obj = Fake.create(make_vc(graphs="full", sizes=sizes), model=object())
        assert obj.cfg.sizes == sizes and obj.cfg.step_sync is False
    assert Fake.create(make_vc(), model=object()) is not None  # eager


class FakeKV:
    def __init__(self, nbytes):
        self.n = nbytes // 2

    def numel(self):
        return self.n

    def element_size(self):
        return 2


def fake_model(kvs=None, kv_bytes=151 << 20, rope_rows=8192):
    rope = NS(cos_sin_cache=torch.zeros(rope_rows, 64))
    kvs = kvs or {L: FakeKV(kv_bytes) for L in range(78)}
    layers = {
        L: NS(self_attn=NS(kv_cache=kv, rotary_emb=rope)) for L, kv in kvs.items()
    }
    return NS(model=NS(layers=layers))


@pytest.mark.parametrize(
    "model_kw, vc_kw, ok",
    [
        (dict(), dict(), True),
        (dict(), dict(graphs=CUDAGraphMode.NONE, sizes=()), True),
        (dict(rope_rows=4096), dict(mml=8000), False),
        (dict(kv_bytes=5 << 30), dict(), False),
        (dict(), dict(sizes=(1, 2, 4, 8)), False),
        (dict(), dict(sizes=(1, 2, 4, 5, 6, 8, 16)), False),
    ],
)
def test_check_before_install(make_vc, model_kw, vc_kw, ok):
    """RoPE table length, KV cache < 4 GiB per layer, FULL-graph capture sizes ==
    kernel widths (none for eager); the RoPE length is raised to max_model_len."""
    vc_kw = dict(dict(graphs=CUDAGraphMode.FULL_DECODE_ONLY), **vc_kw)
    cfg = LiveConfig(ckpt="x", max_model_len=1024)
    if ok:
        G.check_before_install(fake_model(**model_kw), cfg, make_vc(**vc_kw))
        assert cfg.max_model_len == 4096
    else:
        with pytest.raises(RuntimeError):
            G.check_before_install(fake_model(**model_kw), cfg, make_vc(**vc_kw))


def test_cache_contract():
    cfg = NS(layers=[3, 4, 5])
    good = {L: torch.zeros(4, 16, 576, dtype=torch.bfloat16) for L in cfg.layers}
    G.check_cache_contract(fake_model(good), cfg)
    for frag, kv in {
        "dtype": torch.zeros(4, 16, 576, dtype=torch.float16),
        "not contiguous": torch.zeros(4, 576, 16, dtype=torch.bfloat16).transpose(1, 2),
        "row width": torch.zeros(4, 16, 512, dtype=torch.bfloat16),
        "share one MLA cache": good[3],
    }.items():
        with pytest.raises(RuntimeError, match=frag):
            G.check_cache_contract(fake_model({**good, 5: kv}), cfg)
    G.check_cache_contract(fake_model(), cfg)  # size-only stand-ins are skipped


def test_live_config_defaults():
    """The validated optimizations are on; the debug / safety extras are off."""
    c = LiveConfig(ckpt="x")
    assert c.early_cache_checks and c.indexer_trim and c.poll_early_out and c.enabled
    assert c.attention_weight == "fp8_block128" and c.indexer_mode == "attn"
    assert c.fused_indexer is None  # AUTO: on when supported
    assert c.index_q_fp8
    assert c.fused_select_radix11 and c.fused_index_proj_spread
    assert c.cache_hoist and c.split_keys64 and c.step_sync and c.check_every == 1
    assert not c.device_nonfinite and not c.failstop_nonfinite


def test_step_sync_rejected_under_full_graphs(make_vc, no_platform_check, monkeypatch):
    """step_sync syncs and all-reduces, so forcing it on under FULL graphs is refused;
    it defaults off there."""
    monkeypatch.setenv(E.ENABLE, "1")
    assert D.config_from_env(make_vc(graphs="full")).step_sync is False
    monkeypatch.setenv(E.CONFIG, '{"step_sync": true}')
    with pytest.raises(ValueError, match="step_sync"):
        D.Glm5MonoDecode.create(make_vc(graphs="full"), model=object())


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
    assert G.piecewise_graphs_only(vc("PIECEWISE", False))
    assert not G.piecewise_graphs_only(vc("NONE", False))
    assert not G.piecewise_graphs_only(vc("FULL", True))
    monkeypatch.setitem(env, "VLLM_USE_BREAKABLE_CUDAGRAPH", lambda: False)
    assert not G.piecewise_graphs_only(vc("PIECEWISE", False))


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
        "from vllm.models.deepseek_v32.amd.mono.guards import _arm_exit_watchdog\n"
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
    """The dispatch owns a PollErrorWatch advanced by after_step (from compute_logits);
    a tripped watch raises; MONO_LIVE_FAILSTOP=0 builds none."""

    class FakeWatch:
        def __init__(self, lv):
            self.lv, self.n, self.trip = lv, 0, False

        def after_step(self):
            self.n += 1
            if self.trip:
                raise RuntimeError("mono live: expired kernel polls -> fail-stop")

    monkeypatch.setattr(G, "PollErrorWatch", FakeWatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for mode in ("1", "0"):
        monkeypatch.setenv(E.FAILSTOP, mode)
        obj = D.Glm5MonoDecode.__new__(D.Glm5MonoDecode)
        obj.lv = object()
        obj.watch = obj._poll_watch()
        obj.after_step()
        if mode == "0":
            assert obj.watch is None
            continue
        assert obj.watch.lv is obj.lv and obj.watch.n == 1
        obj.watch.trip = True
        with pytest.raises(RuntimeError, match="fail-stop"):
            obj.after_step()


def test_model_hooks(monkeypatch):
    """amd/model.py: compute_logits advances the watch, process_weights_after_loading
    creates the dispatch only when the switch is on, and forward routes the mono layers
    through it (all layers are vLLM's while it is None)."""
    from vllm.models.deepseek_v32.amd import model as M

    cls = M.DeepseekV32ForCausalLM
    lm = cls.__new__(cls)
    torch.nn.Module.__init__(lm)
    calls = []
    lm.model = NS(mono=NS(after_step=lambda: calls.append("after_step")))
    lm.lm_head, lm.logits_processor = None, lambda head, h: h * 2
    assert torch.equal(lm.compute_logits(torch.ones(2)), torch.full((2,), 2.0))
    lm.model.mono = None
    lm.compute_logits(torch.ones(2))
    assert calls == ["after_step"]

    created = []

    def create(cls, vc, model):
        created.append((vc, model))
        return "mono"

    monkeypatch.setattr(D.Glm5MonoDecode, "create", classmethod(create))
    lm.vllm_config, lm.model = object(), NS(mono=None)
    lm.process_weights_after_loading()
    assert lm.model.mono is None and not created
    monkeypatch.setenv(E.ENABLE, "true")
    lm.process_weights_after_loading()
    assert lm.model.mono == "mono" and created == [(lm.vllm_config, lm)]

    monkeypatch.setattr(
        M, "get_pp_group", lambda: NS(is_first_rank=True, is_last_rank=True)
    )
    monkeypatch.setattr(M, "fused_allreduce_rms_norm", lambda h, r, norm: (h + r, None))
    calls.clear()

    def layer(i):
        def run(p, h, r):
            calls.append(i)
            return h + 1, h if r is None else r + h

        return run

    def mono_layer(layer, p, h, r):
        calls.append("mono")
        return layer(p, h, r)

    for mono in (None, NS(layers={3, 4}, forward_layer=mono_layer)):
        me = NS(
            layers=[layer(i) for i in range(6)],
            start_layer=0,
            end_layer=6,
            aux_hidden_state_layers=(),
            embed_input_ids=lambda ids: ids.float(),
            norm=None,
            mono=mono,
        )
        calls.clear()
        out = M.DeepseekV32Model.forward(me, torch.tensor([1.0, 2.0]), None)
        want = (
            [0, 1, 2, 3, 4, 5] if mono is None else [0, 1, 2, "mono", 3, "mono", 4, 5]
        )
        assert calls == want
        h, r = torch.tensor([1.0, 2.0]), None
        for _ in range(6):
            h, r = h + 1, (h if r is None else r + h)
        assert torch.equal(out, h + r)
