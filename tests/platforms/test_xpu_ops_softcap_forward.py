# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""`softcap` / `alibi_slopes` forwarding in `vllm._xpu_ops.xpu_ops`.

The XPU wrapper accepted both parameters in its signature but dropped them
at the call into `vllm_xpu_kernels.flash_attn_interface`, so models that
require attention logit soft-capping (Gemma-2 and friends) ran uncapped on
Intel GPUs with no warning. These tests pin the forwarding. They run on CPU:
the kernel module is replaced by a recording stub.
"""

import importlib
import sys
import types

import pytest
import torch

# vllm._xpu_ops binds `flash_attn_varlen_func` once at first import, so the
# recorder must be a single module-level object shared by every test.
_CALLS: list[dict] = []


def _fake_flash_attn_varlen_func(**kwargs):
    _CALLS.append(kwargs)
    return "kernel-output"


@pytest.fixture()
def xpu_ops(monkeypatch):
    _CALLS.clear()
    fake_pkg = types.ModuleType("vllm_xpu_kernels")
    fake_pkg.__path__ = []
    fake_fa = types.ModuleType("vllm_xpu_kernels.flash_attn_interface")
    fake_fa.flash_attn_varlen_func = _fake_flash_attn_varlen_func
    fake_rotary = types.ModuleType("vllm_xpu_kernels.rotary")
    fake_rotary.apply_rotary_emb = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", fake_pkg)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.flash_attn_interface", fake_fa)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.rotary", fake_rotary)
    mod = importlib.import_module("vllm._xpu_ops")
    return mod.xpu_ops


def _call(xpu_ops_module, **kwargs):
    cu = torch.tensor([0, 4], dtype=torch.int32)
    inputs = dict(
        q=torch.empty(4, 8, 64, dtype=torch.bfloat16),
        k=torch.empty(4, 8, 64, dtype=torch.bfloat16),
        v=torch.empty(4, 8, 64, dtype=torch.bfloat16),
        cu_seqlens_q=cu,
        max_seqlen_q=4,
        max_seqlen_k=4,
        cu_seqlens_k=cu,
        causal=True,
    )
    inputs.update(kwargs)
    return xpu_ops_module.flash_attn_varlen_func(**inputs)


def test_softcap_and_alibi_are_forwarded(xpu_ops):
    slopes = torch.tensor([0.5, 0.25], dtype=torch.float32)

    out = _call(xpu_ops, softcap=30.0, alibi_slopes=slopes)

    assert out == "kernel-output"
    assert _CALLS[-1]["softcap"] == 30.0
    assert _CALLS[-1]["alibi_slopes"] is slopes


def test_default_call_forwards_neutral_values(xpu_ops):
    _call(xpu_ops)

    assert _CALLS[-1]["softcap"] == 0.0
    assert _CALLS[-1]["alibi_slopes"] is None
