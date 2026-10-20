# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated RMS norm with one weight shared by all groups of a row.

The Qwen GDN spec-decode path normalizes ``[tokens, heads, head_dim]`` with a
``head_dim`` weight and an output gate that is a strided view of the input
projection. Flattening the gate per head copied it; with a shared weight the
kernel takes one row per token and reads the gate in place.

The kernel test needs Triton; the call-site test runs anywhere.
"""

import types

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.third_party.flash_linear_attention.ops import layernorm_guard
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd
from vllm.triton_utils import HAS_TRITON

ON_GPU = torch.cuda.is_available()
DEVICE = "cuda" if ON_GPU else "cpu"
needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs Triton")

HEADS, HEAD_DIM = 4, 32
# [q, k, v, z] per token: the gate z is the last HEADS * HEAD_DIM columns.
QKV_WIDTH = 3 * HEADS * HEAD_DIM


@pytest.fixture(autouse=True)
def _cpu_compute_units(monkeypatch):
    if not ON_GPU:
        monkeypatch.setattr(layernorm_guard, "num_compute_units", lambda *_: 148)


def _inputs(num_tokens: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, HEADS, HEAD_DIM, generator=gen)
    proj = torch.randn(num_tokens, QKV_WIDTH + HEADS * HEAD_DIM + 8, generator=gen)
    weight = torch.randn(HEAD_DIM, generator=gen)
    bias = torch.randn(HEAD_DIM, generator=gen)
    to = dict(device=DEVICE, dtype=torch.bfloat16)
    proj = proj.to(**to)
    gate = proj[:, QKV_WIDTH : QKV_WIDTH + HEADS * HEAD_DIM]
    return (
        x.to(**to),
        gate.view(num_tokens, HEADS, HEAD_DIM),
        weight.to(**to),
        bias.to(**to),
    )


def _per_head_rows(x, gate, weight, bias, norm_before_gate, is_rms_norm):
    """Reference: one row per (token, head), contiguous gate copy."""
    out, _, _ = layer_norm_fwd(
        x.reshape(-1, HEAD_DIM),
        weight,
        bias,
        1e-6,
        z=gate.reshape(-1, HEAD_DIM),
        group_size=HEAD_DIM,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )
    return out.view_as(x)


@needs_triton
@pytest.mark.parametrize("num_tokens", [1, 3, 16])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("is_rms_norm", [True, False])
@torch.inference_mode()
def test_shared_weight_matches_per_head_rows(num_tokens, norm_before_gate, is_rms_norm):
    x, gate, weight, bias = _inputs(num_tokens)
    if is_rms_norm:
        bias = None
    gate_2d = gate.flatten(-2)
    assert gate_2d.data_ptr() == gate.data_ptr()  # a view, not a copy
    out, mean, rstd = layer_norm_fwd(
        x.view(num_tokens, -1),
        weight,
        bias,
        1e-6,
        z=gate_2d,
        group_size=HEAD_DIM,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )
    assert rstd.shape == (HEADS * num_tokens,)
    ref = _per_head_rows(x, gate, weight, bias, norm_before_gate, is_rms_norm)
    torch.testing.assert_close(out.view_as(x), ref, atol=0, rtol=0)


def _norm(group_size=None):
    return types.SimpleNamespace(
        weight=torch.ones(HEAD_DIM),
        bias=None,
        eps=1e-6,
        group_size=group_size,
        norm_before_gate=True,
        activation="silu",
    )


def _capture_layer_norm(monkeypatch):
    calls = []

    def fake(x, weight, bias, eps, *, z, out, group_size, **kw):
        calls.append(dict(x=x, weight=weight, z=z, out=out, group_size=group_size))

    monkeypatch.setattr(layernorm_guard, "layer_norm_fwd", fake)
    return calls


@pytest.mark.parametrize("num_tokens", [1, 5])
def test_gdn_gated_norm_reads_the_gate_in_place(monkeypatch, num_tokens):
    calls = _capture_layer_norm(monkeypatch)
    proj = torch.randn(num_tokens, QKV_WIDTH + HEADS * HEAD_DIM)
    gate = proj[:, QKV_WIDTH:].view(num_tokens, HEADS, HEAD_DIM)
    x = torch.randn(num_tokens, HEADS, HEAD_DIM)
    layer = types.SimpleNamespace(norm=_norm())
    QwenGatedDeltaNetAttention._rms_norm_gated_cuda(layer, x, gate, x)
    (call,) = calls
    assert call["z"].data_ptr() == gate.data_ptr()
    assert call["z"].shape == (num_tokens, HEADS * HEAD_DIM)
    if num_tokens > 1:
        assert call["z"].stride(0) == proj.stride(0)
    assert call["x"].shape == call["out"].shape == (num_tokens, HEADS * HEAD_DIM)
    assert call["weight"].shape == (HEAD_DIM,)
    assert call["group_size"] == HEAD_DIM
