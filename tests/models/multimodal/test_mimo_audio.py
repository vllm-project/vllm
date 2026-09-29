# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiMo audio encoder attention."""

import sys
from types import ModuleType

import pytest
import torch

pytestmark = pytest.mark.cpu_test


def test_audio_attention_uses_platform_flash_attention(monkeypatch):
    from vllm.model_executor.models.mimo_audio import AudioEncoderAttention

    calls = {}

    def flash_attn_varlen_func(q, k, v, **kwargs):
        calls["q"] = q
        calls["k"] = k
        calls["v"] = v
        calls["kwargs"] = kwargs
        return q

    fa_utils = ModuleType("vllm.v1.attention.backends.fa_utils")
    fa_utils.__dict__["flash_attn_varlen_func"] = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.fa_utils", fa_utils)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)

    hidden_states = torch.randn(5, 8)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    attn = AudioEncoderAttention(
        embed_dim=8, num_heads=2, window_size=(3, 4), causal=True
    )

    output = attn(hidden_states, cu_seqlens, max_seqlen=3)

    assert output.shape == hidden_states.shape
    assert calls["q"].shape == (5, 2, 4)
    assert calls["k"].shape == (5, 2, 4)
    assert calls["v"].shape == (5, 2, 4)
    assert calls["kwargs"] == {
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "max_seqlen_q": 3,
        "max_seqlen_k": 3,
        "causal": True,
        "window_size": [3, 4],
    }
