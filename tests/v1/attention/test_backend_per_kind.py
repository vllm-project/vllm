# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for attention configuration and per-KV-group backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.attention import AttentionConfig, HiSparseConfig
from vllm.model_executor.layers.attention.attention import (
    Attention,
    _largest_kernel_block_within,
)
from vllm.v1.attention.backend import AttentionType, MultipleOf
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_spec_kind
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.hisparse.runtime import ResolvedHiSparseConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpecKind,
    SlidingWindowSpec,
    get_kv_quant_mode,
)


@pytest.mark.parametrize(
    "supported_sizes,expected",
    [
        ([MultipleOf(16)], 1536),
        ([16, 32], 32),
        ([2048], 2048),
        ([MultipleOf(2048)], 2048),
    ],
)
def test_largest_kernel_block_within(supported_sizes, expected):
    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return supported_sizes

    assert _largest_kernel_block_within(Backend, 1024, 1024 * 1536, 2048) == expected


@pytest.mark.parametrize(
    "supported_sizes,divisor_of,expected",
    [
        # FlashInfer on sm_120 next to a 1648-token hybrid (GDN) block: neither
        # 64 nor 32 divides 1648, so only 16 lets ``unify`` scale the page.
        ([16, 32, 64], 1648, 16),
        ([16, 32, 64], 1536, 64),
        ([MultipleOf(16)], 1648, 1648),
        # No supported block divides it: keep the largest fitting block.
        ([MultipleOf(32)], 1648, 1632),
        ([2048], 1648, 2048),
    ],
)
def test_largest_kernel_block_within_divisor_of(supported_sizes, divisor_of, expected):
    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return supported_sizes

    per_token = 1024
    got = _largest_kernel_block_within(
        Backend, per_token, per_token * divisor_of, divisor_of, divisor_of=divisor_of
    )
    assert got == expected


@pytest.mark.parametrize("primary_block_size", [1648, 1536])
def test_sliding_window_spec_unifies_without_padding(primary_block_size):
    # A SW draft layer next to a hybrid primary block, on a backend limited to
    # small kernel blocks (FlashInfer on sm_120).
    backend = SimpleNamespace(
        is_mla=lambda: False,
        customize_spec=lambda spec: spec,
        get_supported_kernel_block_sizes=lambda kv_cache_spec=None: [16, 32, 64],
    )
    layer = SimpleNamespace(
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="fp8",
        kv_cache_torch_dtype=torch.float8_e4m3fn,
        sliding_window=2048,
        attn_backend=backend,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=primary_block_size, skip_page_size_padded=None
        )
    )
    sw_spec = Attention.get_kv_cache_spec(layer, vllm_config)
    assert isinstance(sw_spec, SlidingWindowSpec)
    assert primary_block_size % sw_spec.block_size == 0

    primary_spec = FullAttentionSpec(
        block_size=primary_block_size,
        num_kv_heads=4,
        head_size=256,
        dtype=torch.float8_e4m3fn,
        kv_quant_mode=get_kv_quant_mode("fp8"),
    )
    unified = unify_kv_cache_spec_page_size({"sw": sw_spec, "full": primary_spec})
    assert unified["sw"].block_size == primary_block_size
    assert unified["sw"].page_size_padded is None


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


def test_hisparse_device_buffer_size_boundaries():
    vllm_config = SimpleNamespace(
        attention_config=AttentionConfig(hisparse_config=HiSparseConfig()),
        speculative_config=None,
    )
    resolved = ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k=128)
    assert resolved is not None
    assert resolved.device_buffer_size == 256

    vllm_config.attention_config.hisparse_config = HiSparseConfig(
        device_buffer_size=127
    )
    with pytest.raises(ValueError, match="expected at least 128"):
        ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k=128)

    vllm_config.attention_config.hisparse_config = HiSparseConfig(
        device_buffer_size=32768
    )
    resolved = ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k=128)
    assert resolved is not None
    assert resolved.device_buffer_size == 32768

    vllm_config.attention_config.hisparse_config = HiSparseConfig(
        device_buffer_size=32769
    )
    with pytest.raises(ValueError, match="int16 slot-index limit"):
        ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k=128)


def test_hisparse_device_buffer_covers_speculative_window():
    vllm_config = SimpleNamespace(
        attention_config=AttentionConfig(hisparse_config=HiSparseConfig()),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            parallel_drafting=False,
        ),
    )

    resolved = ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k=128)
    assert resolved is not None
    assert resolved.device_buffer_size == 5 * 128

    vllm_config.attention_config.hisparse_config = HiSparseConfig(
        device_buffer_size=4 * 128 - 1
    )
    with pytest.raises(ValueError, match="expected at least 512"):
        ResolvedHiSparseConfig.from_vllm_config(vllm_config, model_top_k=128)
