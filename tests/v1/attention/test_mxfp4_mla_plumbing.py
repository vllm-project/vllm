# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mxfp4_mla dtype selection, page sizing, and routing to the kernels that can
read a packed cache."""

import contextlib
import types
from typing import Any

import pytest
import torch

LATENT = 512
ROW = 272


def _layer_spec(block_size: int = 64):
    """The spec the MLA layer publishes for a GLM-5.3 main layer."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    layer = types.SimpleNamespace(
        kv_cache_dtype="mxfp4_mla",
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


def test_dtype_selects_a_uint8_272_byte_row():
    from vllm.config.cache import CacheConfig
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert CacheConfig(block_size=64, cache_dtype="mxfp4_mla").cache_dtype == (
        "mxfp4_mla"
    )
    assert STR_DTYPE_TO_TORCH_DTYPE["mxfp4_mla"] is torch.uint8
    assert "mxfp4_mla" in ROCMAiterMLASparseBackend.supported_kv_cache_dtypes

    spec = _layer_spec()
    assert spec.dtype is torch.uint8
    assert spec.page_size_bytes == 64 * ROW


def test_hybrid_block_size_matches_the_layer_page(monkeypatch):
    """The platform sizes the hybrid block from the same row the layer uses.

    GLM-5.3-Flash at TP=4 has a 1085440-byte mamba state; the glm5_next KV
    grouping rejects the config if that exceeds the real MLA page.
    """
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
        cache_dtype="mxfp4_mla",
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
    monkeypatch.setattr(
        Platform, "_find_non_ssm_backends", classmethod(lambda cls, _: [backend])
    )

    Platform._align_hybrid_block_size(config, backend)

    layer_page = _layer_spec(cache.block_size).page_size_bytes
    assert layer_page >= mamba_page
    assert cache.mamba_page_size_padded == layer_page


def test_decode_and_prefill_take_the_triton_route():
    """Only the Triton kernels read a packed cache; AITER's would misread it."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        _use_rocm_sparse_triton,
    )

    decode = dict(num_prefills=0, num_decodes=8, num_decode_tokens=8, max_query_len=1)
    prefill = dict(
        num_prefills=2, num_decodes=0, num_decode_tokens=0, max_query_len=128
    )
    for step in (decode, prefill):
        assert _use_rocm_sparse_triton(
            kv_cache_dtype="mxfp4_mla", head_size=LATENT, kv_lora_rank=LATENT, **step
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="_forward_mla imports vllm.platforms.rocm, which queries the GPU",
)
def test_long_decode_takes_split_k_with_the_packed_cache(monkeypatch):
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod

    def no_single_pass(**kwargs):
        raise AssertionError("expected the split-K decode")

    captured: dict[str, Any] = {}
    monkeypatch.setattr(sparse_mod, "rocm_sparse_decode_bf16_num_splits", lambda *a: 4)
    monkeypatch.setattr(sparse_mod, "rocm_sparse_attn_decode_bf16", captured.update)
    monkeypatch.setattr(sparse_mod, "rocm_sparse_attn_prefill", no_single_pass)

    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = 16
    impl.kv_lora_rank = LATENT
    impl.kv_cache_dtype = "mxfp4_mla"
    impl.scale = LATENT**-0.5
    impl.sinks = None
    metadata = types.SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=0,
        num_decodes=2,
        num_decode_tokens=2,
        max_query_len=1,
        max_seq_len=8192,
        topk_tokens=2048,
        paged_kv_indices=torch.zeros(4, dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    q = torch.zeros(2, 16, LATENT, dtype=torch.bfloat16)
    cache = torch.zeros(4, 1, ROW, dtype=torch.uint8)

    impl._forward_mla(types.SimpleNamespace(), q, cache, metadata)

    assert captured["kv_cache_dtype"] == "mxfp4_mla"
    assert captured["kv"].shape == (4, 1, ROW)
    assert captured["num_splits"] == 4
