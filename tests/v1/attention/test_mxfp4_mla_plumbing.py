# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the mxfp4_mla dtype and spec plumbing.

CPU-only: these check config wiring and page-size arithmetic, not kernels.
"""

from __future__ import annotations

import types

import pytest
import torch

from vllm.v1.attention.ops.mxfp4_mla import row_bytes

LATENT = 512


def test_dtype_is_a_registered_cache_dtype():
    from vllm.config.cache import CacheDType

    assert "mxfp4_mla" in CacheDType.__args__


def test_dtype_maps_to_uint8():
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

    assert STR_DTYPE_TO_TORCH_DTYPE["mxfp4_mla"] is torch.uint8


def test_backend_accepts_the_dtype():
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert "mxfp4_mla" in ROCMAiterMLASparseBackend.supported_kv_cache_dtypes
    # The dtypes it already served must not have been disturbed.
    for kept in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        assert kept in ROCMAiterMLASparseBackend.supported_kv_cache_dtypes


def _layer_spec(kv_cache_dtype: str, block_size: int = 64):
    """The spec the production MLA layer publishes, for a GLM-5.3 main layer."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    layer = types.SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        head_size=LATENT,
        sliding_window=None,
        indexer=None,
        non_causal_multi_token_decode=False,
        attn_backend=types.SimpleNamespace(get_name=lambda: "ROCM_AITER_MLA_SPARSE"),
        _uses_flat_kv_cache=lambda: False,
    )
    config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(block_size=block_size),
        model_config=types.SimpleNamespace(dtype=torch.bfloat16),
    )
    return MLAAttention.get_kv_cache_spec(layer, config)


def test_layer_publishes_a_272_byte_row():
    spec = _layer_spec("mxfp4_mla")
    assert row_bytes(LATENT) == 272
    assert spec.dtype is torch.uint8
    assert spec.state_content_bytes == 272
    assert spec.page_size_bytes == 64 * 272


def test_bf16_page_size_is_unchanged():
    spec = _layer_spec("auto")
    assert spec.state_content_bytes is None
    assert spec.page_size_bytes == 64 * LATENT * 2


@pytest.mark.parametrize("latent,expected", [(512, 272), (256, 136), (1024, 544)])
def test_row_bytes_scales_with_latent(latent: int, expected: int):
    assert row_bytes(latent) == expected
    # Always 4.25 bits per value, the OCP MXFP4 ratio.
    assert row_bytes(latent) * 8 / latent == 4.25


def test_block_count_scales_by_3_76x_at_a_fixed_memory_budget():
    """The tokens-per-GiB claim, through the allocator's own page-size read."""
    from vllm.v1.core.kv_cache_utils import get_uniform_page_size

    block, layers = 64, 11  # 11 DSA layers in GLM-5.3-Flash
    available = 64 * 1024**3

    def tokens(kv_cache_dtype: str) -> int:
        page = get_uniform_page_size([_layer_spec(kv_cache_dtype, block)])
        return (available // page // layers) * block

    ratio = tokens("mxfp4_mla") / tokens("auto")
    assert 3.7 < ratio < 3.8, ratio


@pytest.mark.parametrize("kv_cache_dtype", ["mxfp4_mla", "auto"])
def test_hybrid_block_size_matches_the_layer_page(monkeypatch, kv_cache_dtype):
    """The platform sizes the hybrid block from the same row the layer uses.

    GLM-5.3-Flash at TP=4 has a 1085440-byte mamba state; the glm5_next KV
    grouping rejects the config if that exceeds the real MLA page.
    """
    import contextlib

    import vllm.config.vllm as vllm_config_mod
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms.interface import Platform

    mamba_page = 1085440
    model_cls = types.SimpleNamespace(
        get_mamba_specs_from_config=lambda _: [
            types.SimpleNamespace(page_size_bytes=mamba_page)
        ]
    )
    monkeypatch.setattr(
        ModelRegistry, "resolve_model_cls", lambda *a, **k: (model_cls, None)
    )
    monkeypatch.setattr(
        vllm_config_mod, "set_current_vllm_config", lambda _: contextlib.nullcontext()
    )
    cache = types.SimpleNamespace(
        cache_dtype=kv_cache_dtype,
        block_size=64,
        mamba_block_size=None,
        user_specified_mamba_block_size=False,
        mamba_cache_mode="none",
        mamba_page_size_padded=None,
        kv_cache_dtype_skip_layers=None,
    )
    config = types.SimpleNamespace(
        cache_config=cache,
        parallel_config=None,
        model_config=types.SimpleNamespace(
            dtype=torch.bfloat16,
            use_mla=True,
            architecture="GlmMoeDsaForCausalLM",
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: LATENT,
        ),
    )
    backend = types.SimpleNamespace(get_supported_kernel_block_sizes=lambda: [64])

    Platform._align_hybrid_block_size(config, backend)

    layer_page = _layer_spec(kv_cache_dtype, cache.block_size).page_size_bytes
    assert layer_page >= mamba_page
    assert cache.mamba_page_size_padded == layer_page


def test_write_hook_rejects_a_rope_bearing_latent():
    """mxfp4_mla targets NoPE models; a non-empty k_pe would be silently lost."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseImpl,
    )

    with pytest.raises(AssertionError, match="rope-free latent"):
        ROCMAiterMLASparseImpl.do_kv_cache_update(
            None,
            kv_c_normed=torch.zeros(1, 512),
            k_pe=torch.zeros(1, 1, 64),
            kv_cache=torch.zeros(1, 272, dtype=torch.uint8),
            slot_mapping=torch.zeros(1, dtype=torch.int64),
            kv_cache_dtype="mxfp4_mla",
            k_scale=torch.ones(1),
        )


def test_engine_config_validation_accepts_the_dtype():
    """``--kv-cache-dtype mxfp4_mla`` must survive CacheConfig validation."""
    from pydantic import ValidationError

    from vllm.config.cache import CacheConfig

    assert CacheConfig(block_size=64, cache_dtype="mxfp4_mla").cache_dtype == (
        "mxfp4_mla"
    )
    for kept in ("auto", "fp8_e4m3"):
        assert CacheConfig(block_size=64, cache_dtype=kept).cache_dtype == kept
    with pytest.raises(ValidationError):
        CacheConfig(block_size=64, cache_dtype="bogus_dtype")


def test_mxfp4_is_not_a_quantized_kv_mode():
    """No KVQuantMode, so the fp8 query-quant and fp8 cache-view paths skip it."""
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    assert get_kv_quant_mode("mxfp4_mla") == KVQuantMode.NONE
