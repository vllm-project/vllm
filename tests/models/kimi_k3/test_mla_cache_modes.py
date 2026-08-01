# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.kimi_k3.nvidia import mla


@pytest.mark.parametrize("use_rope", [False, True])
def test_fp8_cache_keeps_bf16_query_for_backend_without_quant_input(
    monkeypatch: pytest.MonkeyPatch, use_rope: bool
) -> None:
    impl = SimpleNamespace(
        supports_quant_query_input=False,
        do_kv_cache_update=Mock(),
    )
    layer = SimpleNamespace(
        impl=impl,
        kv_cache_dtype="fp8",
        kv_cache=torch.empty(2, 1, 576, dtype=torch.float8_e4m3fn),
        _k_scale=torch.ones(1),
        rotary_emb=SimpleNamespace(is_neox_style=False) if use_rope else None,
    )
    ql_nope = torch.randn(2, 1, 512, dtype=torch.bfloat16)
    q_pe = torch.randn(2, 1, 64, dtype=torch.bfloat16)
    kv_c = torch.randn(2, 512, dtype=torch.bfloat16)
    k_pe = torch.randn(2, 1, 64, dtype=torch.bfloat16)
    rotated_q_pe = torch.randn(q_pe.shape, dtype=torch.float32)
    rotated_k_pe = torch.randn(k_pe.shape, dtype=torch.float32)
    rotary_emb = Mock(return_value=(rotated_q_pe, rotated_k_pe))
    layer.rotary_emb = rotary_emb if use_rope else None
    slots = torch.arange(2)
    positions = torch.arange(2) if use_rope else None
    cos_sin = torch.randn(4, 64) if use_rope else None
    monkeypatch.setattr(
        mla,
        "fused_mla_decode_q_concat_kv_cache_insert",
        Mock(side_effect=AssertionError("quantized-query epilogue must not run")),
    )

    result = mla.MultiHeadLatentAttention._decode_concat_cache(
        layer, ql_nope, q_pe, kv_c, k_pe, positions, cos_sin, slots
    )

    expected_q_pe = rotated_q_pe.to(q_pe.dtype) if use_rope else q_pe
    torch.testing.assert_close(result, torch.cat((ql_nope, expected_q_pe), dim=-1))
    if use_rope:
        rotary_emb.assert_called_once_with(positions, q_pe, k_pe)
        impl.do_kv_cache_update.assert_called_once()
        cache_args = impl.do_kv_cache_update.call_args.args
        assert cache_args[0] is kv_c
        torch.testing.assert_close(cache_args[1], rotated_k_pe.to(kv_c.dtype))
        assert cache_args[2] is layer.kv_cache
        assert cache_args[3] is slots
        assert cache_args[4] == "fp8"
        assert cache_args[5] is layer._k_scale
    else:
        rotary_emb.assert_not_called()
        impl.do_kv_cache_update.assert_called_once_with(
            kv_c, k_pe, layer.kv_cache, slots, "fp8", layer._k_scale
        )


def test_fp8_cache_preserves_quantized_query_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = SimpleNamespace(
        supports_quant_query_input=True,
        do_kv_cache_update=Mock(),
    )
    cache = torch.empty(2, 1, 576, dtype=torch.float8_e4m3fn)
    q_scale_inv = torch.ones(1)
    cache_scale_inv = torch.ones(1)
    layer = SimpleNamespace(
        impl=impl,
        kv_cache_dtype="fp8",
        kv_cache=cache,
        _q_scale_inv=q_scale_inv,
        _k_scale_inv=cache_scale_inv,
    )
    ql_nope = torch.randn(2, 1, 512, dtype=torch.bfloat16)
    q_pe = torch.randn(2, 1, 64, dtype=torch.bfloat16)
    kv_c = torch.randn(2, 512, dtype=torch.bfloat16)
    k_pe = torch.randn(2, 1, 64, dtype=torch.bfloat16)
    slots = torch.arange(2)
    expected = torch.empty(2, 1, 576, dtype=torch.float8_e4m3fn)
    fused_insert = Mock(return_value=expected)
    monkeypatch.setattr(
        mla,
        "fused_mla_decode_q_concat_kv_cache_insert",
        fused_insert,
    )

    result = mla.MultiHeadLatentAttention._decode_concat_cache(
        layer, ql_nope, q_pe, kv_c, k_pe, None, None, slots
    )

    assert result is expected
    fused_insert.assert_called_once_with(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        cache,
        slots,
        q_scale_inv=q_scale_inv,
        cache_scale_inv=cache_scale_inv,
        positions=None,
        cos_sin_cache=None,
    )
    impl.do_kv_cache_update.assert_not_called()
