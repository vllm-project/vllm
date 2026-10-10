# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which Kimi-K3 MoE calls take the mono MoE launch (CPU, no AITER needed).

The runner's predicates are called on a stand-in object; the kernel module is
replaced by a fake that records what it is asked about.
"""

import sys
import types

import pytest
import torch

from vllm.models.kimi_k3.amd import latent_moe_runner as lmr

R = lmr.ROCmLatentMoERunner
RUNNER = "vllm.models.kimi_k3.amd.mono.runner"
HIDDEN, SH_HIDDEN, NE = 3584, 7168, 896


class _Router:
    top_k = 16
    capture_fn = None


class _Experts:
    w13_weight = torch.empty(NE, 768, 1792, dtype=torch.uint8)


def _stub(layer_ok=True):
    s = types.SimpleNamespace(router=_Router(), routed_experts=_Experts())
    s._mono_layer_ok = layer_ok
    s._shared_mlp_weights = (
        torch.empty(1536, SH_HIDDEN, dtype=torch.bfloat16),
        torch.empty(SH_HIDDEN, 768, dtype=torch.bfloat16),
        4.0,
        25.0,
    )
    return s


@pytest.fixture
def fake_runner(monkeypatch):
    asked = []
    mod = types.ModuleType(RUNNER)

    def supported(x, ne, topk, inter, shared_x, w_gu, w_dn):
        asked.append((x.shape[0], ne, topk, inter, tuple(w_dn.shape)))
        return x.shape[0] <= 16

    mod.supported = supported  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RUNNER, mod)
    return asked


def _call(s, m, x_dtype=torch.bfloat16, logits_dtype=torch.float32, shared=True):
    x = torch.empty(m, HIDDEN, dtype=x_dtype)
    logits = torch.empty(m, NE, dtype=logits_dtype)
    shared_x = torch.empty(m, SH_HIDDEN, dtype=torch.bfloat16) if shared else None
    return R._use_mono(s, x, logits, shared_x)


def test_off_without_switch(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_MONO_DECODE", raising=False)
    assert R._mono_layer_ok.func(_stub()) is False


def test_off_on_other_platforms(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", "1")
    monkeypatch.setattr(lmr.current_platform, "is_rocm", lambda: False)
    assert R._mono_layer_ok.func(_stub()) is False


def test_small_decode_batches(fake_runner):
    s = _stub()
    assert _call(s, 1)
    assert _call(s, 16)
    assert not _call(s, 17)
    assert [a[0] for a in fake_runner] == [1, 16, 17]
    assert fake_runner[0][1:] == (NE, 16, 384, (SH_HIDDEN, 768))


def test_declined(fake_runner):
    assert not _call(_stub(), 4, shared=False)
    assert not _call(_stub(), 4, x_dtype=torch.float16)
    assert not _call(_stub(), 4, logits_dtype=torch.bfloat16)
    assert not _call(_stub(layer_ok=False), 4)
    s = _stub()
    s.router.capture_fn = lambda *a: None
    assert not _call(s, 4)
    assert fake_runner == []
