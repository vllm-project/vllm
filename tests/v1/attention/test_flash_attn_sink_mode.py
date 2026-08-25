# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends import flash_attn


@pytest.mark.parametrize(
    ("requested_mode", "fused_sink_supported", "expected"),
    [
        ("auto", True, "fused"),
        ("auto", False, "unfused"),
        ("fused", True, "fused"),
        ("unfused", True, "unfused"),
        ("unfused", False, "unfused"),
    ],
)
def test_select_fa3_sink_path(requested_mode, fused_sink_supported, expected):
    assert (
        flash_attn._select_fa3_sink_path(requested_mode, fused_sink_supported)
        == expected
    )


def test_select_fa3_sink_path_fails_when_fused_is_unsupported():
    with pytest.raises(RuntimeError, match="fused FA3 sink path requires"):
        flash_attn._select_fa3_sink_path("fused", False)


def test_select_fa3_sink_path_rejects_unknown_mode():
    with pytest.raises(ValueError, match="fa3_sink_mode must be one of"):
        flash_attn._select_fa3_sink_path("unknown", True)


@pytest.mark.parametrize(
    ("version", "num_sink_tokens", "has_alibi", "kv_cache_dtype", "expected"),
    [
        (3, 1, False, "auto", True),
        (2, 1, False, "auto", False),
        (3, 0, False, "auto", False),
        (3, 9, False, "auto", False),
        (3, 1, True, "auto", False),
        (3, 1, False, "fp8", False),
    ],
)
def test_fused_sink_support_predicate(
    version, num_sink_tokens, has_alibi, kv_cache_dtype, expected
):
    impl = flash_attn.FlashAttentionImpl.__new__(flash_attn.FlashAttentionImpl)
    impl.vllm_flash_attn_version = version
    impl.alibi_slopes = object() if has_alibi else None
    impl.kv_cache_dtype = kv_cache_dtype

    assert impl._fused_sink_supported(num_sink_tokens) is expected


def test_populate_sinks_kv_fails_early_for_unsupported_fused_mode():
    impl = flash_attn.FlashAttentionImpl.__new__(flash_attn.FlashAttentionImpl)
    impl.fa3_sink_mode = "fused"
    impl.vllm_flash_attn_version = 2
    impl.alibi_slopes = None
    impl.kv_cache_dtype = "auto"
    impl.num_kv_heads = 2
    impl.head_size = 4
    sinks = torch.zeros((1, 1, 2, 4))

    with pytest.raises(RuntimeError, match="fused FA3 sink path requires"):
        impl.populate_sinks_kv(sinks, sinks)
