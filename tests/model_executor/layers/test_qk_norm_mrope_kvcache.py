# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention import qk_norm_mrope_kvcache as fusion


def _call_fusion(q_out: torch.Tensor) -> None:
    fusion.fused_qk_norm_mrope_and_unified_kv_cache_update_impl(
        qkv=torch.zeros(2, 16),
        q_out=q_out,
        positions=torch.arange(2),
        cos_sin_cache=torch.zeros(16, 4),
        q_weight=torch.ones(4),
        k_weight=torch.ones(4),
        num_heads_q=2,
        num_heads_k=1,
        head_size=4,
        rms_norm_eps=1e-6,
        is_neox=True,
        is_interleaved=False,
        mrope_section=[1, 1, 0],
        layer_name="model.layers.0.self_attn.attn",
    )


def test_profiling_pass_initializes_query(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        fusion,
        "get_attention_context",
        lambda _: (None, None, None, None),
    )
    q_out = torch.full((2, 8), torch.nan)

    _call_fusion(q_out)

    torch.testing.assert_close(q_out, torch.zeros_like(q_out))


def test_attention_context_failure_propagates(monkeypatch: pytest.MonkeyPatch):
    def fail_context(_):
        raise RuntimeError("missing attention context")

    monkeypatch.setattr(fusion, "get_attention_context", fail_context)

    with pytest.raises(RuntimeError, match="missing attention context"):
        _call_fusion(torch.empty(2, 8))


def test_runtime_uses_strided_kv_views(monkeypatch: pytest.MonkeyPatch):
    kv_cache = torch.zeros(2, 2, 4, 1, 4)
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    assert not key_cache.is_contiguous()
    assert not value_cache.is_contiguous()

    impl = SimpleNamespace(
        kv_cache_dtype="auto",
        _split_kv_cache=lambda _: (key_cache, value_cache),
    )
    attn_layer = SimpleNamespace(
        impl=impl,
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    monkeypatch.setattr(
        fusion,
        "get_attention_context",
        lambda _: (None, attn_layer, kv_cache, torch.arange(2)),
    )

    recorded = {}

    def record_call(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(
        fusion.rocm_aiter_ops,
        "do_qk_norm_mrope_kvcache_update",
        staticmethod(record_call),
    )
    fusion._COS_SIN_CACHE.clear()

    _call_fusion(torch.empty(2, 8))

    assert recorded["key_cache"] is key_cache
    assert recorded["value_cache"] is value_cache
    assert recorded["positions"].shape == (3, 2)
