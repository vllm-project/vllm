# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.v1.attention.backend import PrequantizedQKV
from vllm.v1.attention.backends import rocm_aiter_fa


def _make_tensors():
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 1, 8)
    value = torch.randn(2, 1, 8)
    return query, key, value


def _call_aiter_flash_attention(**kwargs):
    query, key, value = _make_tensors()
    return rocm_aiter_ops.flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 2], dtype=torch.int32),
        max_seqlen_q=2,
        max_seqlen_k=2,
        **kwargs,
    )


def test_aiter_flash_attention_omits_unused_descales(monkeypatch):
    recorded_kwargs = None
    sentinel = object()

    def fake_flash_attention(**kwargs):
        nonlocal recorded_kwargs
        recorded_kwargs = kwargs
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(flash_attn_varlen_func=fake_flash_attention),
    )

    assert _call_aiter_flash_attention() is sentinel
    assert recorded_kwargs is not None
    assert "q_descale" not in recorded_kwargs
    assert "k_descale" not in recorded_kwargs
    assert "v_descale" not in recorded_kwargs


def test_aiter_flash_attention_forwards_complete_descales(monkeypatch):
    recorded_kwargs = None
    descales = tuple(torch.randn(1, 1) for _ in range(3))

    def fake_flash_attention(**kwargs):
        nonlocal recorded_kwargs
        recorded_kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(flash_attn_varlen_func=fake_flash_attention),
    )

    _call_aiter_flash_attention(
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
    )

    assert recorded_kwargs is not None
    assert recorded_kwargs["q_descale"] is descales[0]
    assert recorded_kwargs["k_descale"] is descales[1]
    assert recorded_kwargs["v_descale"] is descales[2]


def test_aiter_flash_attention_rejects_partial_descales(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(flash_attn_varlen_func=lambda **kwargs: None),
    )

    with pytest.raises(ValueError, match="requires q_descale, k_descale"):
        _call_aiter_flash_attention(q_descale=torch.randn(1, 1))


def test_aiter_backend_uses_prequantized_qkv_for_prefill(monkeypatch):
    impl = object.__new__(rocm_aiter_fa.AiterFlashAttentionImpl)
    impl.head_size = 8
    impl.kv_cache_dtype = "auto"
    impl.scale = 1.0
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.sinks = None
    impl.logits_soft_cap = 0.0

    query, key, value = _make_tensors()
    prequantized = PrequantizedQKV(
        query=torch.randn_like(query),
        key=torch.randn_like(key),
        value=torch.randn_like(value),
        query_descale=torch.randn(1, 1),
        key_descale=torch.randn(1, 1),
        value_descale=torch.randn(1, 1),
    )
    output = torch.empty_like(query)
    metadata = rocm_aiter_fa.AiterFlashAttentionMetadata(
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        max_seq_len=2,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        causal=True,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_extends=0,
        num_extend_tokens=0,
        decode_metadata=None,
        prefill_metadata=rocm_aiter_fa.AiterFlashAttentionPrefillMetadata(
            max_query_len=2,
            max_seq_len=2,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        ),
        extend_metadata=None,
        use_cascade=False,
        k_scale=None,
        v_scale=None,
    )
    recorded_kwargs = None

    def fake_flash_attention(**kwargs):
        nonlocal recorded_kwargs
        recorded_kwargs = kwargs
        kwargs["out"].zero_()

    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops,
        "flash_attn_varlen_func",
        fake_flash_attention,
    )

    impl._forward(
        SimpleNamespace(),
        query,
        key,
        value,
        torch.empty((1, 1, 1, 16)),
        metadata,
        output,
        output_scale=None,
        output_block_scale=None,
        prequantized_qkv=prequantized,
    )

    assert recorded_kwargs is not None
    assert recorded_kwargs["q"].data_ptr() == prequantized.query.data_ptr()
    assert recorded_kwargs["k"].data_ptr() == prequantized.key.data_ptr()
    assert recorded_kwargs["v"].data_ptr() == prequantized.value.data_ptr()
    assert (
        recorded_kwargs["q_descale"].data_ptr() == prequantized.query_descale.data_ptr()
    )
    assert (
        recorded_kwargs["k_descale"].data_ptr() == prequantized.key_descale.data_ptr()
    )
    assert (
        recorded_kwargs["v_descale"].data_ptr() == prequantized.value_descale.data_ptr()
    )


def test_aiter_backend_directly_gathers_fp8_context(monkeypatch):
    impl = object.__new__(rocm_aiter_fa.AiterFlashAttentionImpl)
    impl.num_kv_heads = 1
    impl.kv_cache_dtype = "fp8"
    impl.scale = 1.0
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.sinks = None

    query, key, value = _make_tensors()
    fp8_dtype = torch.float8_e4m3fn
    prequantized = PrequantizedQKV(
        query=query.to(fp8_dtype),
        key=key.to(fp8_dtype),
        value=value.to(fp8_dtype),
        query_descale=torch.randn(1, 1),
        key_descale=torch.randn(1, 1),
        value_descale=torch.randn(1, 1),
    )
    workspace = torch.empty((2, 4, 1, 8), dtype=fp8_dtype)
    chunk_metadata = rocm_aiter_fa.AiterChunkContextMetadata(
        workspace=workspace,
        cu_seq_lens_chunk=torch.tensor([[0, 3]], dtype=torch.int32),
        chunk_starts=torch.tensor([[0]], dtype=torch.int32),
        token_to_batch=torch.zeros((1, 3), dtype=torch.int32),
        max_seq_lens=[3],
        num_chunks=1,
        total_token_per_batch=[3],
        swa_metadata=None,
    )
    metadata = SimpleNamespace(
        extend_metadata=rocm_aiter_fa.AiterFlashAttentionChunkPrefillMetadata(
            max_query_len=2,
            max_seq_len=5,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            chunk_context_metadata=chunk_metadata,
        )
    )
    gather_kwargs = None
    attention_calls = []

    def fake_gather(**kwargs):
        nonlocal gather_kwargs
        gather_kwargs = kwargs

    def fake_flash_attention(**kwargs):
        attention_calls.append(kwargs)
        return torch.zeros_like(query), torch.zeros((1, 4))

    monkeypatch.setattr(
        rocm_aiter_fa,
        "cp_mha_gather_cache",
        fake_gather,
        raising=False,
    )
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops,
        "flash_attn_varlen_func",
        fake_flash_attention,
    )
    monkeypatch.setattr(rocm_aiter_fa, "merge_attn_states", lambda **kwargs: None)
    monkeypatch.setattr(
        rocm_aiter_fa.current_platform,
        "fp8_dtype",
        lambda: fp8_dtype,
    )

    k_scale = torch.tensor([0.25])
    v_scale = torch.tensor([0.5])
    impl.extend_forward(
        attn_metadata=metadata,
        query=query,
        key=key,
        value=value,
        key_cache=torch.empty((1, 1, 1, 16)),
        value_cache=torch.empty((1, 1, 1, 16)),
        output=torch.empty_like(query),
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        max_seqlen_q=2,
        min_seqlen_q=1,
        max_seqlen_k=5,
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        k_scale=k_scale,
        v_scale=v_scale,
        prequantized_qkv=prequantized,
    )

    assert gather_kwargs is not None
    assert gather_kwargs["dequant"] is False
    assert len(attention_calls) == 2
    context_call = attention_calls[1]
    assert context_call["q"].data_ptr() == prequantized.query.data_ptr()
    assert context_call["k"].data_ptr() == workspace[0].data_ptr()
    assert context_call["v"].data_ptr() == workspace[1].data_ptr()
    assert context_call["q_descale"] is prequantized.query_descale
    torch.testing.assert_close(context_call["k_descale"], k_scale.reshape(1, 1))
    torch.testing.assert_close(context_call["v_descale"], v_scale.reshape(1, 1))
