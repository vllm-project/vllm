# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter routes (`akr_draft.setup`) for both Kimi-K3.1 drafts: the
preview's MLA draft (`K3DSparkForCausalLM`) and the release's GQA sliding-window
draft (`Qwen3DSparkForCausalLM`), on draft shells with tiny random weights (the
kernels themselves are checked in tests/kernels/models/amd/): which pieces each
draft takes, the modules they read, the lm_head hook, and the release's context
norm moving out of `precompute_and_store_context_kv`.

Run: python -m pytest tests/models/kimi_k3/akr/test_akr_draft_routes.py
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.amd import akr_config, akr_nonlayer
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="amd_kernels routes are for ROCm"
)

H, NUM_AUX = 64, 5


@pytest.fixture(autouse=True)
def _env(monkeypatch, default_vllm_config, dist_init):
    monkeypatch.setattr(akr_config, "_routes", frozenset(akr_config.ROUTES))
    for n in akr_nonlayer.SWITCHES:
        monkeypatch.delenv(f"VLLM_KIMI_AKR_{n}", raising=False)
    torch.set_default_device("cuda")
    yield default_vllm_config
    torch.set_default_device(None)


class _Logits:
    def __call__(self, lm_head, h):
        return torch.full((h.shape[0], 3), -1.0)


def _release(eps: float = 1e-5):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

    m = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(m)
    m.use_aux_hidden_state = True
    m.fc = ReplicatedLinear(
        NUM_AUX * H, H, bias=False, params_dtype=torch.bfloat16, return_bias=False
    )
    m.fc.weight.data.normal_(0, 0.05)
    m.hidden_norm = RMSNorm(H, eps=eps).to(torch.bfloat16)
    m.hidden_norm.weight.data.uniform_(0.5, 1.5)
    m._hidden_norm_weight = m.hidden_norm.weight.data
    m._rms_norm_eps = 1e-5
    layer = nn.Module()
    layer.mlp = Qwen2MLP(H, 2 * H, "silu").to(torch.bfloat16)
    m.layers = nn.ModuleList([layer])
    model = Qwen3DSparkForCausalLM.__new__(Qwen3DSparkForCausalLM)  # type: ignore[type-abstract]
    nn.Module.__init__(model)
    model.model = m
    model.lm_head = nn.Identity()
    model.logits_processor = _Logits()
    return model


def _preview():
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

    layer = nn.Module()
    layer.aiter_ar = False
    m = nn.Module()
    m.context_proj = ReplicatedLinear(
        NUM_AUX * H, H, bias=False, params_dtype=torch.bfloat16, return_bias=False
    )
    m.context_norm = RMSNorm(H, eps=1e-5).to(torch.bfloat16)
    m.layers = nn.ModuleList([layer])
    model = K3DSparkForCausalLM.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = m
    model.lm_head = nn.Identity()
    model.logits_processor = _Logits()
    model._akr_lmhead = False
    return model


def _spec(model, cfg, mla: bool):
    return SimpleNamespace(
        model=model,
        vllm_config=cfg,
        device=torch.device("cuda"),
        max_num_reqs=8,
        num_speculative_steps=4,
        _aux_staging=object(),
        _num_graph_context_tokens=lambda n: n * 5,
        draft_model_config=SimpleNamespace(use_mla=mla),
    )


class _KimiTarget:
    pass


_KimiTarget.__module__ = "vllm.models.kimi_k3.amd.model"


def _record(monkeypatch):
    from vllm.models.kimi_k3.amd import (
        akr_draft,
        akr_draft_gqa,
        akr_draft_mla,
        akr_lmhead,
        akr_sampling,
    )

    calls: dict[str, list] = {}

    def rec(name, ret=None):
        def f(*a, **k):
            calls.setdefault(name, []).append(a)
            return ret

        return f

    monkeypatch.setattr(akr_lmhead, "setup", rec("lmhead", True))
    monkeypatch.setattr(akr_draft, "_context_proj", rec("ctxproj"))
    monkeypatch.setattr(akr_draft, "_mlp_reason", rec("mlp_reason"))
    monkeypatch.setattr(akr_draft, "_install_mlp", rec("dmlp", 1))
    monkeypatch.setattr(akr_draft, "_dar", rec("dar"))
    monkeypatch.setattr(akr_sampling, "setup_speculator", rec("markov"))
    monkeypatch.setattr(akr_draft_mla, "setup", rec("dmla"))
    monkeypatch.setattr(akr_draft_gqa, "setup", rec("gqa"))
    return calls


@pytest.mark.parametrize("draft", ["preview", "release"])
def test_setup_routes_per_draft(monkeypatch, _env, draft):
    from vllm.models.kimi_k3.amd import akr_draft, akr_lmhead

    calls = _record(monkeypatch)
    model = _preview() if draft == "preview" else _release()
    spec = _spec(model, _env, mla=draft == "preview")
    akr_draft.setup(spec, _KimiTarget())
    assert calls["lmhead"][0][:2] == ("draft", model.lm_head)
    assert calls["ctxproj"][0] == (spec, model)
    assert calls["dmlp"][0] == (spec, model)
    assert calls["markov"][0] == (spec,)
    assert calls["dar"][0] == (spec, model)  # the preview's layers keep their own
    # the dense MLA decode is the MLA draft's only, the GQA attention inputs the GQA's
    assert ("dmla" in calls) == (draft == "preview")
    assert calls.get("gqa") == (None if draft == "preview" else [(spec, model)])
    h = torch.zeros(2, H, dtype=torch.bfloat16)
    if draft == "preview":
        assert model._akr_lmhead is True  # read by K3DSparkForCausalLM.compute_logits
        assert "compute_draft_logits" not in vars(model)
    else:
        sentinel = torch.ones(2, 3)
        monkeypatch.setattr(akr_lmhead, "logits", lambda *a: sentinel)
        assert model.compute_draft_logits(h) is sentinel
        monkeypatch.setattr(akr_lmhead, "logits", lambda *a: None)
        assert bool((model.compute_draft_logits(h) == -1).all())  # stock


def test_setup_needs_a_kimi_target(monkeypatch, _env):
    from vllm.models.kimi_k3.amd import akr_draft

    calls = _record(monkeypatch)
    spec = _spec(_release(), _env, mla=False)
    akr_draft.setup(spec, nn.Linear(2, 2))
    assert calls == {} and spec._akr_ctx is None and spec._akr_markov is None


def test_context_layout_and_reasons(_env):
    from vllm.models.kimi_k3.amd import akr_draft

    rel, pre = _release(), _preview()
    proj, norm, in_pre = akr_draft._ctx_modules(rel)
    assert (proj, norm, in_pre) == (rel.model.fc, rel.model.hidden_norm, True)
    proj, norm, in_pre = akr_draft._ctx_modules(pre)
    assert (proj, norm, in_pre) == (
        pre.model.context_proj,
        pre.model.context_norm,
        False,
    )
    for model in (rel, pre):
        assert akr_draft._ctx_reason(_spec(model, _env, False), model) is None
    spec = _spec(rel, _env, False)
    spec.max_num_reqs = 48  # c48, 4 speculative tokens: 240 graph context rows
    assert akr_draft._ctx_reason(spec, rel) is None
    spec.max_num_reqs = 52
    assert "260 graph context rows" in (akr_draft._ctx_reason(spec, rel) or "")
    spec._aux_staging = None
    assert "FULL" in (akr_draft._ctx_reason(spec, rel) or "")
    # the precompute norms with the q_norm eps: it must be hidden_norm's
    odd = _release()
    odd.model._rms_norm_eps = 1e-6
    assert "precompute" in str(akr_draft._ctx_reason(_spec(odd, _env, False), odd))
    assert "eps" in str(
        akr_draft._ctx_reason(
            _spec(_release(eps=1e-6), _env, False), _release(eps=1e-6)
        )
    )

    # VLLM_K31_BEAT_DRAFTFC's TP-split fc (latency_specialized) wins over CTXPROJ
    class TPShardedFC(nn.Module):
        pass

    rel.model.fc = TPShardedFC()
    assert "TPShardedFC" in str(akr_draft._ctx_reason(_spec(rel, _env, False), rel))
    other = SimpleNamespace(model=nn.Module())
    assert "no context projection" in str(
        akr_draft._ctx_reason(_spec(other, _env, False), other)
    )


def _ctx_shell(model, combine=None):
    """A `ContextProj` without its communicator (the 8-rank test runs the kernel):
    the norm bookkeeping and the row split only. Its eager call computes what the
    kernel does (the projection and its norm) with `combine` (the model's
    `combine_hidden_states` by default) and records its row ranges."""
    from vllm.models.kimi_k3.amd import akr_draft

    combine = combine or model.combine_hidden_states

    class Shell(akr_draft.ContextProj):
        def __init__(self):
            self.calls: list[tuple[int, int]] = []

        def combine_eager(self, aux, start, end, out):
            self.calls.append((start, end))
            x = torch.cat([h[start:end] for h in aux], dim=-1)
            out.copy_(self.post_norm(combine(x)))

    ctx = Shell()
    _, _, ctx.norm_in_precompute = akr_draft._ctx_modules(model)
    ctx.post_norm = (
        model.model._normalize_context_states
        if ctx.norm_in_precompute
        else akr_draft._identity
    )
    ctx._model = model.model
    ctx.num_aux = NUM_AUX
    return ctx


def test_release_context_norm_moves_to_the_store(_env):
    """The release stores normed rows; the precompute's norm becomes the identity, so
    the context K/V it projects are the stock ones."""
    from vllm.models.kimi_k3.amd import akr_draft

    model = _release()
    m = model.model
    L, nkv, hd = 2, 1, 16
    m._fused_kv_weight = (torch.randn(L * 2 * nkv * hd, H) * 0.1).to(torch.bfloat16)
    m._fused_kv_bias = None
    m._context_kv_proj = None
    rows = akr_draft.CTX_MAX_ROWS + 8  # two kernel calls
    aux = [torch.randn(rows, H).to(torch.bfloat16) for _ in range(NUM_AUX)]
    raw = model.combine_hidden_states(torch.cat(aux, dim=-1))
    ref = m._project_context_kv_token_major(raw, rows, L, nkv, hd)

    ctx = _ctx_shell(model)
    ctx.install()
    assert m._normalize_context_states(raw) is raw
    spec = SimpleNamespace(
        _akr_ctx=ctx,
        _aux_staging=None,
        model=model,
        hidden_states=torch.zeros(rows + 4, H, dtype=torch.bfloat16),
    )
    assert akr_draft.store_hidden_states(spec, rows, None, aux, 0)
    assert ctx.calls == [(0, akr_draft.CTX_MAX_ROWS), (akr_draft.CTX_MAX_ROWS, rows)]
    got = m._project_context_kv_token_major(spec.hidden_states[:rows], rows, L, nkv, hd)
    assert torch.equal(got, ref)
    # no aux hidden states: the target's last hidden states, normed the same way
    last = torch.randn(rows, H).to(torch.bfloat16)
    assert akr_draft.store_hidden_states(spec, 6, last, None, 0)
    assert torch.equal(
        spec.hidden_states[:6], type(m)._normalize_context_states(m, last[:6])
    )


def test_preview_store_runs_every_row_on_the_kernel(_env):
    """Past CTX_MAX_ROWS the preview's rows still take the packed projection (the
    replicated weight is released), in CTX_MAX_ROWS-row calls, with no norm after it
    (the preview's combine norms)."""
    from vllm.models.kimi_k3.amd import akr_draft

    model = _preview()
    ctx = _ctx_shell(model, combine=lambda x: x[:, :H] * 2)
    ctx.install()  # nothing to move: the preview's combine norms
    rows = akr_draft.CTX_MAX_ROWS + 1
    aux = [torch.randn(rows, H).to(torch.bfloat16) for _ in range(NUM_AUX)]
    spec = SimpleNamespace(
        _akr_ctx=ctx,
        _aux_staging=None,
        model=model,
        hidden_states=torch.zeros(rows, H, dtype=torch.bfloat16),
    )
    assert akr_draft.store_hidden_states(spec, rows, None, aux, 0)
    assert ctx.calls == [(0, akr_draft.CTX_MAX_ROWS), (akr_draft.CTX_MAX_ROWS, rows)]
    assert torch.equal(spec.hidden_states, aux[0] * 2)


def test_release_mlp_reason_takes_a_reducing_mlp(_env, monkeypatch):
    """The release's `Qwen2MLP` all-reduces in its down projection: the route keeps
    that all-reduce after the kernel instead of declining."""
    pytest.importorskip("amd_kernels")
    from vllm.model_executor.models.qwen2 import Qwen2MLP
    from vllm.models.kimi_k3.amd import akr_draft

    if not akr_config._available():
        pytest.skip("amd_kernels / gfx950 not available")
    model = _release()
    model.model.layers[0].mlp = Qwen2MLP(7168, 1792, "silu").to(torch.bfloat16)
    assert model.model.layers[0].mlp.down_proj.reduce_results
    spec = _spec(model, _env, False)
    assert akr_draft._mlp_reason(spec, model) is None
    spec.max_num_reqs = 48  # 192 draft rows at c48
    assert akr_draft._mlp_reason(spec, model) is None
    spec.max_num_reqs = 65
    assert "260" in (akr_draft._mlp_reason(spec, model) or "")
    # the rows are the query rows: 1 + k a request for a fill-in block
    spec.max_num_reqs, spec.num_query_per_req = 48, 5
    assert akr_draft._mlp_reason(spec, model) is None
    spec.max_num_reqs = 52
    assert "260" in (akr_draft._mlp_reason(spec, model) or "")


def test_release_ar_reason(_env):
    """The release's all-reduce + RMSNorm pairs (`DAR`) engage only on an uncompiled
    backbone with a TP all-reduce (the 8-rank kernel check: draft_ar_norm_ranks.py)."""
    from vllm.models.kimi_k3.amd import akr_draft

    model = _release()
    spec = _spec(model, _env, False)
    assert akr_draft._ar_reason(spec, model) == "TP 1 (no all-reduce)"
    model.model.do_not_compile = False
    assert "compiled" in (akr_draft._ar_reason(spec, model) or "")
    akr_draft._dar(spec, model)
    assert not any(isinstance(n, akr_draft.AkrArNorm) for n in model.modules())
