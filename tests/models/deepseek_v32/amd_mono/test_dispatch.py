# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the model-integrated dispatch path (mono/dispatch.py):

* env gating (off -> ``maybe_create`` returns None and builds nothing), refusal
  reasons, config from ``VLLM_ROCM_GLM5_MONOKERNEL_CONFIG``;
* env parsing without / with registration in ``vllm.envs``;
* checkpoint directory resolution (local dir, cached HF snapshot, errors);
* FULL-graph capture sizes vs kernel widths (switch on + mismatch raises);
* the poll-error fail-stop watch and the compute_logits / post-load model hooks;
* with the existing-file hook applied (amd/model.py), env OFF leaves
  ``DeepseekV32Model.forward`` unchanged (skipped otherwise).
"""

import ast
import json
import os
from itertools import islice
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.deepseek_v32.amd.mono import dispatch as D
from vllm.models.deepseek_v32.amd.mono import envs as E

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
TMP = ""
LOCAL_CKPT = ""


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    """A local checkpoint dir with the safetensors index; the mono env vars unset."""
    global TMP, LOCAL_CKPT
    TMP = str(tmp_path)
    LOCAL_CKPT = os.path.join(TMP, "local_ckpt")
    os.makedirs(LOCAL_CKPT)
    with open(os.path.join(LOCAL_CKPT, D.CKPT_INDEX), "w") as f:
        f.write(json.dumps(dict(weight_map={})))
    for name in (E.ENABLE, E.CONFIG, E.FAILSTOP):
        monkeypatch.setenv(name, "")  # records the original value: restored at teardown
        monkeypatch.delenv(name)
    yield
    D._ACTIVE["obj"] = None


def vc(**kw):
    model = kw.pop("model", None)
    d = dict(model_type="glm_moe_dsa", tp=8, pp=1, dp=1, ep=False, spec=None, kv="auto")
    d.update(kw)
    return NS(
        model_config=NS(
            hf_config=NS(model_type=d["model_type"]),
            model=model or LOCAL_CKPT,
            max_model_len=4096,
        ),
        parallel_config=NS(
            tensor_parallel_size=d["tp"],
            pipeline_parallel_size=d["pp"],
            data_parallel_size=d["dp"],
            enable_expert_parallel=d["ep"],
            decode_context_parallel_size=1,
        ),
        speculative_config=d["spec"],
        cache_config=NS(cache_dtype=d["kv"]),
    )


def test_env_gating_and_refusals():
    os.environ.pop("VLLM_ROCM_USE_GLM5_MONOKERNEL", None)
    assert E.enabled() is False
    assert (
        D.Glm5MonoDecode.maybe_create(vc(), object()) is None
        and D._ACTIVE["obj"] is None
    )
    for kw, frag in (
        (dict(model_type="deepseek_v32"), "model_type"),
        (dict(tp=4), "TP 4"),
        (dict(pp=2), "PP"),
        (dict(ep=True), "expert"),
        (dict(spec=object()), "speculative"),
        (dict(kv="fp8"), "kv_cache"),
    ):
        why = D.refusal(vc(**kw))
        assert why is not None and frag in why, (kw, why)
    for attr, val, frag in (
        ("lora_config", object(), "LoRA"),
        ("kv_transfer_config", object(), "KV transfer"),
        ("aux_output_config", NS(enable_return_routed_experts=True), "routed"),
    ):
        cfg = vc()
        setattr(cfg, attr, val)
        why = D.refusal(cfg)
        assert why is not None and frag in why, (attr, why)
    cfg = vc()
    cfg.model_config.dtype = torch.float16
    assert "dtype" in D.refusal(cfg)
    os.environ["VLLM_ROCM_GLM5_MONOKERNEL_CONFIG"] = (
        '{"sizes": [1, 2, 4, 8], "indexer_mode": "indexer_only"}'
    )
    cfg = D.config_from_env(vc())
    assert (
        cfg.sizes == (1, 2, 4, 8)
        and cfg.ckpt == LOCAL_CKPT
        and cfg.max_model_len == 4096
    )
    assert cfg.indexer_mode == "indexer_only"
    assert cfg.step_sync is True  # no compilation_config -> eager
    full = vc()
    full.compilation_config = NS(cudagraph_mode=NS(has_full_cudagraphs=lambda: True))
    # graphs: no per-step host sync inside the capture
    assert D.config_from_env(full).step_sync is False
    os.environ.pop("VLLM_ROCM_GLM5_MONOKERNEL_CONFIG")


def test_env_parsing_without_registration(monkeypatch):
    """Unregistered in vllm.envs (envs' module __getattr__ raises AttributeError for
    unknown names): os.environ is parsed the vLLM ROCm way. Registered: vllm.envs
    wins."""
    from vllm import envs

    monkeypatch.delitem(envs.environment_variables, E.ENABLE, raising=False)
    for val, want in (
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" True ", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("yes", False),
        (None, False),
    ):
        if val is None:
            monkeypatch.delenv(E.ENABLE, raising=False)
        else:
            monkeypatch.setenv(E.ENABLE, val)
        assert E.enabled() is want, (val, E.enabled())
    monkeypatch.setenv(E.ENABLE, "0")
    monkeypatch.setitem(envs.environment_variables, E.ENABLE, lambda: True)
    assert E.enabled() is True
    monkeypatch.setitem(envs.environment_variables, E.ENABLE, lambda: False)
    monkeypatch.setenv(E.ENABLE, "1")
    assert E.enabled() is False


def test_ckpt_resolution():
    """model_config.model may be an HF repo id -> the cached snapshot vLLM
    loaded from."""
    # local directory: as is
    assert D.resolve_ckpt_dir(vc()) == LOCAL_CKPT
    # HF repo id: the snapshot of the revision under load_config.download_dir (hub cache
    # layout), no download
    cache = os.path.join(TMP, "hub")
    repo = os.path.join(cache, "models--zai-org--GLM-5.2-test")
    snap = os.path.join(repo, "snapshots", "0123abcd")
    os.makedirs(snap)
    os.makedirs(os.path.join(repo, "refs"))
    with open(os.path.join(repo, "refs", "main"), "w") as f:
        f.write("0123abcd")
    with open(os.path.join(snap, D.CKPT_INDEX), "w") as f:
        f.write("{}")
    v = vc(model="zai-org/GLM-5.2-test")
    v.model_config.revision = None
    v.load_config = NS(download_dir=cache, load_format="auto")
    got = D.resolve_ckpt_dir(v)
    assert os.path.realpath(got) == os.path.realpath(snap), got
    os.environ["VLLM_ROCM_GLM5_MONOKERNEL_CONFIG"] = "{}"
    assert os.path.realpath(D.config_from_env(v).ckpt) == os.path.realpath(snap)
    # an explicit ckpt wins (no resolution)
    os.environ["VLLM_ROCM_GLM5_MONOKERNEL_CONFIG"] = json.dumps(dict(ckpt="/explicit"))
    assert D.config_from_env(v).ckpt == "/explicit"
    os.environ.pop("VLLM_ROCM_GLM5_MONOKERNEL_CONFIG")
    # not cached / unknown repo -> clear error naming the fix
    for bad in (vc(model="zai-org/not-cached"), vc(model="/no/such/dir")):
        bad.load_config = NS(download_dir=cache, load_format="auto")
        try:
            D.resolve_ckpt_dir(bad)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "VLLM_ROCM_GLM5_MONOKERNEL_CONFIG" in str(e) and "ckpt" in str(e), e
    # directory without the safetensors index
    empty = os.path.join(TMP, "empty")
    os.makedirs(empty)
    try:
        D.resolve_ckpt_dir(vc(model=empty))
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert D.CKPT_INDEX in str(e)
    # dummy weights: nothing to read
    dm = vc()
    dm.load_config = NS(download_dir=None, load_format="dummy")
    try:
        D.resolve_ckpt_dir(dm)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "dummy" in str(e)


def test_width_mismatch_refusal(monkeypatch):
    """Switch on + FULL-graph capture sizes != kernel widths -> RuntimeError naming the
    capture sizes to use; matching sizes -> created."""

    class Fake(D.Glm5MonoDecode):
        def __init__(self, vllm_config, causal_lm, cfg=None):
            self.cfg = cfg

    def full(sizes):
        v = vc()
        v.compilation_config = NS(
            cudagraph_mode=NS(has_full_cudagraphs=lambda: True),
            cudagraph_capture_sizes=list(sizes),
        )
        return v

    monkeypatch.setattr(D, "refusal", lambda v: None)  # the platform probe needs gfx950
    monkeypatch.setenv(E.ENABLE, "1")
    for sizes in ((1, 2, 4, 8, 16, 32), (1, 2, 4, 8), (1, 2, 4, 5, 6, 8, 16)):
        with pytest.raises(RuntimeError) as e:
            Fake.maybe_create(full(sizes), object())
        assert "cudagraph_capture_sizes=[1, 2, 4, 5, 6, 8]" in str(e.value), sizes
        assert D._ACTIVE["obj"] is None
    # capture sizes that are all kernel widths: the message also offers matching sizes
    with pytest.raises(RuntimeError, match=r'"sizes": \[1, 2, 4, 8\]'):
        Fake.maybe_create(full((1, 2, 4, 8)), object())
    # matching sizes (default widths, or "sizes" set to the capture sizes) -> created
    obj = Fake.maybe_create(full((1, 2, 4, 5, 6, 8)), object())
    assert obj is not None and obj.cfg.sizes == (1, 2, 4, 5, 6, 8)
    assert obj.cfg.step_sync is False
    D._ACTIVE["obj"] = None
    monkeypatch.setenv(E.CONFIG, '{"sizes": [1, 2, 4, 8]}')
    obj = Fake.maybe_create(full((1, 2, 4, 8)), object())
    assert obj is not None and obj.cfg.sizes == (1, 2, 4, 8)
    D._ACTIVE["obj"] = None
    # eager (no FULL graphs): no width constraint
    monkeypatch.delenv(E.CONFIG)
    assert Fake.maybe_create(vc(), object()) is not None
    D._ACTIVE["obj"] = None


def test_refusal_raises_when_enabled(monkeypatch):
    monkeypatch.setenv(E.ENABLE, "1")
    with pytest.raises(RuntimeError, match="TP 4"):
        D.Glm5MonoDecode.maybe_create(vc(tp=4), object())


def test_poll_watch(monkeypatch):
    """The dispatch path owns a PollErrorWatch advanced by after_step (called from
    compute_logits); a tripped watch raises (fail-stop); MONO_LIVE_FAILSTOP=0 builds
    none."""
    from vllm.models.deepseek_v32.amd.mono import guards as G

    class FakeWatch:
        def __init__(self, lv):
            self.lv, self.n, self.trip = lv, 0, False

        def after_step(self):
            self.n += 1
            if self.trip:
                raise RuntimeError("mono live: expired kernel polls -> fail-stop")

    monkeypatch.setattr(G, "PollErrorWatch", FakeWatch)
    # not capturing (also on a CPU-only torch build, where the query raises)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    obj = D.Glm5MonoDecode.__new__(D.Glm5MonoDecode)
    obj.lv, obj.watch = object(), None
    obj._install_poll_watch()
    assert isinstance(obj.watch, FakeWatch) and obj.watch.lv is obj.lv
    for _ in range(3):
        obj.after_step()
    assert obj.watch.n == 3
    obj.watch.trip = True
    with pytest.raises(RuntimeError, match="fail-stop"):
        obj.after_step()
    monkeypatch.setenv(E.FAILSTOP, "0")
    obj2 = D.Glm5MonoDecode.__new__(D.Glm5MonoDecode)
    obj2.lv, obj2.watch = object(), None
    obj2._install_poll_watch()
    assert obj2.watch is None
    obj2.after_step()  # no watch: no-op


def test_model_hooks(monkeypatch):
    """With the existing-file hook applied: compute_logits advances the watch, and
    process_weights_after_loading creates the dispatch only when the switch is on."""
    amd_model = pytest.importorskip("vllm.models.deepseek_v32.amd.model")
    cls = amd_model.DeepseekV32ForCausalLM
    if "compute_logits" not in cls.__dict__:
        pytest.skip("amd/model.py has no mono hook (existing-file commit not applied)")
    lm = cls.__new__(cls)
    torch.nn.Module.__init__(lm)
    calls = []
    lm.model = NS(mono=NS(after_step=lambda: calls.append("after_step")))
    lm.lm_head = None
    lm.logits_processor = lambda head, h: h * 2
    out = lm.compute_logits(torch.ones(2))
    assert calls == ["after_step"] and torch.equal(out, torch.full((2,), 2.0))
    lm.model.mono = None
    assert torch.equal(lm.compute_logits(torch.ones(2)), torch.full((2,), 2.0))

    created = []

    def maybe_create(c, vllm_config, causal_lm):
        created.append((vllm_config, causal_lm))
        return "mono"

    monkeypatch.setattr(D.Glm5MonoDecode, "maybe_create", classmethod(maybe_create))
    lm.vllm_config = object()
    lm.model = NS(mono=None)
    monkeypatch.setenv(E.ENABLE, "0")
    lm.process_weights_after_loading()
    assert lm.model.mono is None and not created
    monkeypatch.setenv(E.ENABLE, "true")
    lm.process_weights_after_loading()
    assert lm.model.mono == "mono" and created == [(lm.vllm_config, lm)]


def _func(src, cls, name):
    tree = ast.parse(src)
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    f = next(
        (n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name), None
    )
    return None if f is None else ast.get_source_segment(src, f)


def test_model_forward_env_off():
    with open(os.path.join(ROOT, "vllm/models/deepseek_v32/amd/model.py")) as f:
        src = f.read()
    fwd = _func(src, "DeepseekV32Model", "forward")
    if "self.mono" not in fwd:
        pytest.skip("amd/model.py has no mono hook (existing-file commit not applied)")
    calls = []

    class Layer:
        def __init__(self, i):
            self.i = i

        def __call__(self, positions, h, r):
            calls.append(self.i)
            return h + 1, (h if r is None else r + h)

    class IT(dict):
        pass

    ns = dict(
        islice=islice,
        torch=torch,
        IntermediateTensors=IT,
        get_pp_group=lambda: NS(is_first_rank=True, is_last_rank=True),
        fused_allreduce_rms_norm=lambda h, r, norm: (h + r, None),
    )
    exec(
        compile(
            "def _f" + fwd.strip()[len("def forward") :].replace("self,", "self,", 1),
            "model.py",
            "exec",
        ),
        ns,
    )
    fwd_fn = ns["_f"]
    me = NS(
        layers=[Layer(i) for i in range(6)],
        start_layer=0,
        end_layer=6,
        aux_hidden_state_layers=(),
        embed_input_ids=lambda ids: ids.float(),
        norm=None,
        mono=None,
    )
    out = fwd_fn(me, torch.tensor([1.0, 2.0]), torch.tensor([0, 1]))
    assert calls == list(range(6))
    ref_h, ref_r = torch.tensor([1.0, 2.0]), None
    for _ in range(6):
        ref_h, ref_r = ref_h + 1, (ref_h if ref_r is None else ref_r + ref_h)
    assert torch.equal(out, ref_h + ref_r)
    pwal = _func(src, "DeepseekV32ForCausalLM", "process_weights_after_loading")
    assert pwal is not None and "VLLM_ROCM_USE_GLM5_MONOKERNEL" in pwal
    tree = ast.parse(pwal.strip())
    first_if = next(n for n in ast.walk(tree) if isinstance(n, ast.If))
    assert "VLLM_ROCM_USE_GLM5_MONOKERNEL" in ast.unparse(first_if.test)
    assert "mono" in ast.unparse(first_if.body[0]) or any(
        "mono" in ast.unparse(b) for b in first_if.body
    )


def test_second_create_raises(monkeypatch):
    """The kernel holds a private weight copy: a second creation (weight reload) must
    fail loudly instead of running stale weights."""
    monkeypatch.setenv(E.ENABLE, "1")
    D._ACTIVE["obj"] = object()
    with pytest.raises(RuntimeError, match="already created"):
        D.Glm5MonoDecode.maybe_create(vc(), object())


def test_step_sync_rejected_under_full_graphs():
    from vllm.models.deepseek_v32.amd.mono import live as LV

    full = vc()
    full.compilation_config = NS(cudagraph_mode=NS(has_full_cudagraphs=lambda: True))
    with pytest.raises(ValueError, match="step_sync"):
        LV.MonoLive(None, LV.LiveConfig(ckpt="x", step_sync=True), full)


def test_piecewise_graphs_only(monkeypatch):
    from vllm import envs
    from vllm.models.deepseek_v32.amd.mono import guards as G

    def mode(name, full):
        return NS(cudagraph_mode=NS(name=name, has_full_cudagraphs=lambda: full))

    monkeypatch.setitem(
        envs.environment_variables, "VLLM_USE_BREAKABLE_CUDAGRAPH", lambda: True
    )
    assert G.piecewise_graphs_only(NS(compilation_config=mode("PIECEWISE", False)))
    assert not G.piecewise_graphs_only(NS(compilation_config=mode("NONE", False)))
    assert not G.piecewise_graphs_only(NS(compilation_config=mode("FULL", True)))
    monkeypatch.setitem(
        envs.environment_variables, "VLLM_USE_BREAKABLE_CUDAGRAPH", lambda: False
    )
    assert not G.piecewise_graphs_only(NS(compilation_config=mode("PIECEWISE", False)))
