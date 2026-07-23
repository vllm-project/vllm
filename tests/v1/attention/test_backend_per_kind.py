# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for attention configuration and per-KV-group backend selection."""

from types import SimpleNamespace

import pytest

from vllm.config.attention import AttentionConfig, HiSparseConfig
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.hisparse import ResolvedHiSparseConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_spec_kind
from vllm.v1.kv_cache_interface import KVCacheSpecKind


@pytest.mark.parametrize(
    "signals,expected",
    [
        (dict(use_mla=False, has_sliding_window=False), "full"),
        (dict(use_mla=True, has_sliding_window=False), "mla"),
        (dict(use_mla=True, has_sliding_window=True), "sw_mla"),
        (dict(use_mla=False, has_sliding_window=True), "sw"),
    ],
)
def test_get_attn_spec_kind_decoder(signals, expected):
    kind_by_name = {
        "full": KVCacheSpecKind.FULL_ATTENTION,
        "mla": KVCacheSpecKind.MLA_ATTENTION,
        "sw_mla": KVCacheSpecKind.SLIDING_WINDOW_MLA,
        "sw": KVCacheSpecKind.SLIDING_WINDOW,
    }
    kind = get_attn_spec_kind(attn_type=AttentionType.DECODER, **signals)
    assert kind is kind_by_name[expected]


@pytest.mark.parametrize(
    "attn_type,expected",
    [
        (AttentionType.ENCODER_ONLY, KVCacheSpecKind.ENCODER_ONLY_ATTENTION),
        (AttentionType.ENCODER_DECODER, KVCacheSpecKind.CROSS_ATTENTION),
    ],
)
def test_get_attn_spec_kind_attn_type(attn_type, expected):
    kind = get_attn_spec_kind(
        use_mla=False,
        has_sliding_window=False,
        attn_type=attn_type,
    )
    assert kind is expected


def test_backend_per_kind_parses_strings():
    cfg = AttentionConfig(
        backend_per_kind={
            "mla_attention": "FLASHINFER_MLA",
            "sliding_window_mla": "triton_mla",  # case-insensitive
        }
    )
    assert cfg.backend_per_kind["mla_attention"] is AttentionBackendEnum.FLASHINFER_MLA
    assert cfg.backend_per_kind["sliding_window_mla"] is AttentionBackendEnum.TRITON_MLA


def test_backend_per_kind_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown KV cache group kind"):
        AttentionConfig(backend_per_kind={"not_a_kind": "TRITON_MLA"})


def test_backend_per_kind_defaults_empty():
    assert AttentionConfig().backend_per_kind == {}


def test_hisparse_config_parses_mapping():
    config = AttentionConfig(
        hisparse_config={"host_pool_gib": 1.0}  # type: ignore[arg-type]
    )
    assert config.hisparse_config == HiSparseConfig(host_pool_gib=1.0)


def test_hisparse_config_resolves_model_constraints():
    vllm_config = SimpleNamespace(
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(host_pool_gib=1.0)
        )
    )
    resolved = ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k=128)
    assert resolved is not None and resolved.device_buffer_size == 256

    vllm_config.attention_config.hisparse_config = HiSparseConfig(
        host_pool_gib=1.0, device_buffer_size=127
    )
    with pytest.raises(ValueError, match="at least the model's index_topk"):
        ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k=128)
