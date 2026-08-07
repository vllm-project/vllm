# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends import b12x_attn
from vllm.v1.attention.backends.b12x_attn import (
    B12XPagedAttentionBackend,
    B12XPagedAttentionImpl,
    B12XPagedMetadataBuilder,
    _kv_page_size,
    _max_page_table_width,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import set_kv_cache_layout


def test_b12x_attention_backend_is_registered() -> None:
    assert AttentionBackendEnum.B12X_ATTN.get_class() is B12XPagedAttentionBackend


def test_b12x_attention_advertises_only_supported_contract() -> None:
    assert B12XPagedAttentionBackend.get_supported_kernel_block_sizes() == [64, 128]
    assert B12XPagedAttentionBackend.get_supported_head_sizes() == [64, 128, 192, 256]
    assert B12XPagedAttentionBackend.supports_sliding_window()
    assert B12XPagedAttentionBackend.supports_sink()
    assert not B12XPagedAttentionBackend.supports_non_causal()
    assert B12XPagedAttentionBackend.supports_compute_capability(
        DeviceCapability(12, 0)
    )
    assert B12XPagedAttentionBackend.supports_compute_capability(
        DeviceCapability(12, 1)
    )
    assert not B12XPagedAttentionBackend.supports_compute_capability(
        DeviceCapability(12, 2)
    )
    assert not B12XPagedAttentionBackend.supports_compute_capability(
        DeviceCapability(10, 0)
    )
    assert {"fp8", "fp8_e4m3"}.issubset(
        B12XPagedAttentionBackend.supported_kv_cache_dtypes
    )
    assert B12XPagedAttentionBackend.supports_dtype(torch.bfloat16)
    assert not B12XPagedAttentionBackend.supports_dtype(torch.float16)


def test_b12x_attention_rejects_fp16_queries_with_fp8_kv(
    default_vllm_config,
) -> None:
    reason = B12XPagedAttentionBackend.supports_combination(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="fp8_e4m3",
        block_size=128,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=DeviceCapability(12, 0),
    )

    assert reason == "B12X_ATTN currently requires bfloat16 queries"


def test_b12x_attention_rejects_float16_kv_dtype(
    default_vllm_config,
) -> None:
    reason = B12XPagedAttentionBackend.supports_combination(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="float16",
        block_size=128,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=DeviceCapability(12, 0),
    )

    assert reason == "B12X_ATTN does not support float16 KV cache"


def test_b12x_attention_uses_two_plane_nhd_cache() -> None:
    assert B12XPagedAttentionBackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        cache_dtype_str="fp8_e4m3",
    ) == (3, 2, 128, 4, 128)
    set_kv_cache_layout("NHD")
    try:
        assert B12XPagedAttentionBackend.get_kv_cache_stride_order() == (
            0,
            1,
            2,
            3,
            4,
        )
        assert B12XPagedAttentionBackend.get_kv_cache_stride_order(True) == (
            1,
            0,
            2,
            3,
            4,
            5,
        )
    finally:
        set_kv_cache_layout(None)
    assert B12XPagedAttentionBackend.get_required_kv_cache_layout() == "NHD"


def test_b12x_attention_rejects_unsupported_page_size() -> None:
    with pytest.raises(ValueError, match="block_size"):
        B12XPagedAttentionBackend.get_kv_cache_shape(3, 32, 4, 128)


def test_b12x_attention_uses_uniform_batch_graphs() -> None:
    assert (
        B12XPagedMetadataBuilder._cudagraph_support is AttentionCGSupport.UNIFORM_BATCH
    )


def test_b12x_attention_hybrid_cache_capacity_includes_expansion() -> None:
    assert _max_page_table_width(4096, 128, 4096, False) == 32
    assert _max_page_table_width(4096, 128, 4096, True) == 64


def test_b12x_attention_runtime_page_size_comes_from_cache() -> None:
    key_cache = torch.empty((3, 64, 4, 128), device="meta")
    value_cache = torch.empty_like(key_cache)

    assert _kv_page_size(key_cache, value_cache) == 64
    with pytest.raises(ValueError, match="matching K/V page sizes"):
        _kv_page_size(key_cache, torch.empty((3, 128, 4, 128), device="meta"))


def test_b12x_attention_lazily_prepares_decode_bucket(monkeypatch) -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    plan = SimpleNamespace(layout=SimpleNamespace(nbytes=96))
    created: list[tuple[int, int]] = []

    def create_plan(page_size: int, batch_size: int) -> SimpleNamespace:
        created.append((page_size, batch_size))
        return plan

    impl._decode_plans = {}
    impl._create_decode_plan = create_plan
    impl._scratch_nbytes = 128
    impl._extend_plans = {}
    impl._verify_q_per_req = 0
    metadata = SimpleNamespace(max_query_len=1)
    monkeypatch.setattr(b12x_attn, "_capture_alloc_forbidden", lambda: False)

    assert impl._select_plan(metadata, 7, 7, 7, 64) is plan
    assert impl._select_plan(metadata, 7, 7, 7, 64) is plan
    assert created == [(64, 7)]


def test_b12x_attention_fp8_descales_follow_request_batch() -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    impl.kv_cache_dtype = "fp8_e4m3"
    layer = SimpleNamespace(
        _k_scale=torch.tensor(2.0),
        _v_scale=torch.tensor([3.0, 4.0, 5.0]),
    )

    k_descale, v_descale = impl._prepare_fp8_descales(
        layer, num_reqs=2, device=torch.device("cpu")
    )

    torch.testing.assert_close(k_descale, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(v_descale, torch.tensor([3.0, 4.0]))
    assert k_descale.stride() == (0,)
    assert v_descale.stride() == (1,)


def test_b12x_attention_sinks_refresh_in_place_after_reload() -> None:
    impl = object.__new__(B12XPagedAttentionImpl)
    impl._sinks_cache = {}
    source = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    sinks = impl._prepare_sinks(source, torch.device("cpu"))
    assert sinks is not None
    sinks_ptr = sinks.data_ptr()
    source.copy_(torch.tensor([3.0, 4.0], dtype=torch.bfloat16))
    refreshed = impl._prepare_sinks(source, torch.device("cpu"))

    assert refreshed is not None
    assert refreshed.data_ptr() == sinks_ptr
    torch.testing.assert_close(refreshed, source.float())
