# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.model_executor.layers import (
    fused_qk_norm_rope_gate_fp8_quant as prep_module,
)
from vllm.model_executor.layers.attention import attention as attention_module
from vllm.platforms import current_platform
from vllm.v1.attention.backend import PrequantizedQKV
from vllm.v1.attention.backends import rocm_aiter_fa

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm AITER attention backend"
)


def _make_tensors(tokens=2, num_heads=4, num_kv_heads=1, head_size=8):
    query = torch.randn(tokens, num_heads, head_size)
    key = torch.randn(tokens, num_kv_heads, head_size)
    value = torch.randn(tokens, num_kv_heads, head_size)
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


@pytest.mark.parametrize(
    "seq_lens",
    [(1,), (1, 1, 1), (2, 3, 2), (1, 129, 5, 64)],
    ids=["one-decode", "decodes", "mixed", "mixed-chunk"],
)
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_size"),
    [(8, 1, 256), (16, 2, 256)],
    ids=["tp2", "tp1"],
)
def test_slice_prequantized_qkv_uses_mixed_batch_offsets(
    seq_lens, num_heads, num_kv_heads, head_size
):
    num_tokens = sum(seq_lens)
    num_seqs = len(seq_lens)
    starts = [0]
    for seq_len in seq_lens:
        starts.append(starts[-1] + seq_len)
    query = torch.arange(num_tokens * num_heads * head_size).reshape(
        num_tokens, num_heads, head_size
    )
    key = torch.arange(num_tokens * num_kv_heads * head_size).reshape(
        num_tokens, num_kv_heads, head_size
    )
    value = key + 1000
    query_descale = torch.arange(num_seqs * num_kv_heads).reshape(
        num_seqs, num_kv_heads
    )
    key_descale = query_descale + 10
    value_descale = query_descale + 20
    prequantized = PrequantizedQKV(
        query, key, value, query_descale, key_descale, value_descale
    )

    # Every single sequence, and the span from the second sequence to the end.
    spans = [(i, i + 1) for i in range(num_seqs)] + [(min(1, num_seqs - 1), num_seqs)]
    for first, last in spans:
        tokens = slice(starts[first], starts[last])
        seqs = slice(first, last)
        sliced = rocm_aiter_fa._slice_prequantized_qkv(prequantized, tokens, seqs)
        torch.testing.assert_close(sliced.query, query[tokens])
        torch.testing.assert_close(sliced.key, key[tokens])
        torch.testing.assert_close(sliced.value, value[tokens])
        torch.testing.assert_close(sliced.query_descale, query_descale[seqs])
        torch.testing.assert_close(sliced.key_descale, key_descale[seqs])
        torch.testing.assert_close(sliced.value_descale, value_descale[seqs])


def test_aiter_flash_attention_passes_no_descales_by_default(monkeypatch):
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
    assert recorded_kwargs["q_descale"] is None
    assert recorded_kwargs["k_descale"] is None
    assert recorded_kwargs["v_descale"] is None


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


@pytest.mark.parametrize(
    ("q_lens", "k_lens", "num_heads", "num_kv_heads"),
    [
        pytest.param((5,), (5,), 4, 1, id="one-sequence-one-head"),
        pytest.param((5,), (5,), 8, 2, id="one-sequence-two-heads"),
        pytest.param((3, 1, 7), (3, 1, 7), 4, 1, id="three-sequences"),
        pytest.param((3, 4), (6, 0), 16, 2, id="chunk-without-context"),
    ],
)
def test_grouped_fp8_attention_calls_once_per_sequence_and_kv_head(
    monkeypatch, q_lens, k_lens, num_heads, num_kv_heads
):
    # The AITER FP8 kernel reads only descale[0, 0], so each call must hold one
    # sequence, one KV head and that group's own descales.
    head_size = 8
    num_seqs = len(q_lens)
    q_starts = [0, *torch.tensor(q_lens).cumsum(0).tolist()]
    k_starts = [0, *torch.tensor(k_lens).cumsum(0).tolist()]
    fp8 = torch.float8_e4m3fn
    q = torch.randn(q_starts[-1], num_heads, head_size).to(fp8)
    k = torch.randn(k_starts[-1], num_kv_heads, head_size).to(fp8)
    q_descale = torch.rand(num_seqs, num_kv_heads) + 1
    k_descale = torch.rand(num_seqs, num_kv_heads) + 2
    v_descale = torch.rand(num_seqs, num_kv_heads) + 3
    group = num_heads // num_kv_heads
    calls = []

    def fake_flash_attention(**kwargs):
        calls.append(kwargs)
        # Echo the query so the assembled output shows where each call landed.
        return kwargs["q"].to(torch.bfloat16), kwargs["q"][..., 0].float().T

    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "flash_attn_varlen_func", fake_flash_attention
    )
    out, lse = rocm_aiter_fa._grouped_fp8_varlen_attention(
        q=q,
        k=k,
        v=k,
        cu_seqlens_q=torch.tensor(q_starts, dtype=torch.int32),
        cu_seqlens_k=torch.tensor(k_starts, dtype=torch.int32),
        q_starts=q_starts,
        k_starts=k_starts,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(k_lens),
        min_seqlen_q=1,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        out_dtype=torch.bfloat16,
        softmax_scale=1.0,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        sink_ptr=None,
        return_lse=True,
    )

    live = [i for i in range(num_seqs) if k_lens[i] > 0]
    assert len(calls) == len(live) * num_kv_heads
    expected_groups = iter((i, g) for i in live for g in range(num_kv_heads))
    for call in calls:
        seq, kv_head = next(expected_groups)
        assert call["q"].shape[1] == group and call["k"].shape[1] == 1
        assert call["q"].shape[0] == q_lens[seq] and call["k"].shape[0] == k_lens[seq]
        for name, descale in (
            ("q_descale", q_descale),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
        ):
            assert call[name].numel() == 1
            assert call[name].item() == descale[seq, kv_head].item()
    for seq in range(num_seqs):
        rows = slice(q_starts[seq], q_starts[seq + 1])
        if k_lens[seq] > 0:
            torch.testing.assert_close(out[rows], q[rows].to(torch.bfloat16))
            assert torch.isfinite(lse[:, rows]).all()
        else:
            assert not out[rows].any()
            assert torch.isneginf(lse[:, rows]).all()


def test_aiter_backend_ignores_prequantized_qkv_for_pure_decode(monkeypatch):
    impl = object.__new__(rocm_aiter_fa.AiterFlashAttentionImpl)
    impl.head_size = 64
    impl.num_heads = 4
    impl.num_kv_heads = 1
    impl.kv_cache_dtype = "auto"
    impl.scale = 1.0
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.sinks = None
    impl.logits_soft_cap = 0.0
    impl.kv_sharing_target_layer_name = None

    query = torch.randn(1, 4, 64)
    key = torch.randn(1, 1, 64)
    value = torch.randn_like(key)
    prequantized = PrequantizedQKV(
        query=torch.full_like(query, float("nan")),
        key=torch.full_like(key, float("nan")),
        value=torch.full_like(value, float("nan")),
        query_descale=torch.full((1, 1), float("nan")),
        key_descale=torch.full((1, 1), float("nan")),
        value_descale=torch.full((1, 1), float("nan")),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        causal=True,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        num_extends=0,
        num_extend_tokens=0,
        decode_metadata=SimpleNamespace(max_query_len=1, uniform_query_len=1),
        prefill_metadata=None,
        extend_metadata=None,
        k_scale=None,
        v_scale=None,
        kv_sharing_metadata=None,
    )
    recorded_query = None

    def fake_paged_attention(*args, **kwargs):
        nonlocal recorded_query
        recorded_query = args[2]

    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops,
        "is_shuffle_kv_cache_enabled",
        lambda: False,
    )
    # aiter registers its ops on torch.ops.aiter at import; patching the
    # namespace first makes that registration fail and poisons the process.
    import aiter  # noqa: F401

    monkeypatch.setattr(torch.ops.aiter, "paged_attention_v1", fake_paged_attention)

    impl._forward(
        SimpleNamespace(_k_scale=torch.ones(1), _v_scale=torch.ones(1)),
        query,
        key,
        value,
        torch.empty((1, 1, 1, 128)),
        metadata,
        torch.empty_like(query),
        output_scale=None,
        output_block_scale=None,
        prequantized_qkv=prequantized,
    )

    assert recorded_query is not None
    assert recorded_query.data_ptr() == query.data_ptr()
    assert torch.isfinite(recorded_query).all()


@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_aiter_backend_gathers_context_for_fp8_attention(monkeypatch, num_kv_heads):
    impl = object.__new__(rocm_aiter_fa.AiterFlashAttentionImpl)
    impl.num_kv_heads = num_kv_heads
    impl.kv_cache_dtype = "fp8"
    impl.scale = 1.0
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.sinks = None

    query, key, value = _make_tensors(num_kv_heads=num_kv_heads)
    fp8_dtype = torch.float8_e4m3fn
    prequantized = PrequantizedQKV(
        query=query.to(fp8_dtype),
        key=key.to(fp8_dtype),
        value=value.to(fp8_dtype),
        query_descale=torch.randn(1, num_kv_heads),
        key_descale=torch.randn(1, num_kv_heads),
        value_descale=torch.randn(1, num_kv_heads),
    )
    workspace = torch.empty((2, 4, num_kv_heads, 8), dtype=fp8_dtype)
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
        q = kwargs["q"]
        return torch.zeros(q.shape, dtype=query.dtype), torch.zeros(
            q.shape[1], q.shape[0]
        )

    monkeypatch.setattr(rocm_aiter_fa, "cp_mha_gather_cache", fake_gather)
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

    k_scale = torch.arange(1, num_kv_heads + 1, dtype=torch.float32) * 0.25
    v_scale = torch.arange(1, num_kv_heads + 1, dtype=torch.float32) * 0.5
    impl.extend_forward(
        attn_metadata=metadata,
        query=query,
        key=key,
        value=value,
        key_cache=torch.empty((1, 1, num_kv_heads, 16), dtype=fp8_dtype),
        value_cache=torch.empty((1, 1, num_kv_heads, 16), dtype=fp8_dtype),
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
    # One causal and one context call per KV head: the kernel reads one descale.
    assert len(attention_calls) == 2 * num_kv_heads
    for kv_head, context_call in enumerate(attention_calls[num_kv_heads:]):
        group = query.shape[1] // num_kv_heads
        rows = slice(kv_head * group, (kv_head + 1) * group)
        assert context_call["q"].data_ptr() == prequantized.query[:, rows].data_ptr()
        assert (
            context_call["k"].data_ptr()
            == workspace[0][:3, kv_head : kv_head + 1].data_ptr()
        )
        assert (
            context_call["v"].data_ptr()
            == workspace[1][:3, kv_head : kv_head + 1].data_ptr()
        )
        assert (
            context_call["q_descale"].item()
            == prequantized.query_descale[0, kv_head].item()
        )
        assert context_call["k_descale"].item() == k_scale[kv_head].item()
        assert context_call["v_descale"].item() == v_scale[kv_head].item()


@pytest.mark.parametrize(
    ("cdna_version", "kv_cache_dtype", "expected"),
    [
        (4, "fp8", True),
        (4, "auto", False),
        (4, "bfloat16", False),
        (3, "fp8", False),
        (0, "fp8", False),
    ],
    ids=["cdna4-fp8", "cdna4-auto", "cdna4-bf16", "cdna3-fp8", "non-cdna-fp8"],
)
def test_aiter_backend_prequantized_qkv_cache_support(
    monkeypatch, cdna_version, kv_cache_dtype, expected
):
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(get_cdna_version=lambda: cdna_version),
    )

    impl = rocm_aiter_fa.AiterFlashAttentionImpl(
        num_heads=16,
        head_size=256,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
    )

    assert impl.supports_prequantized_qkv_input is expected


def _layer(supports: bool, receives: bool):
    return SimpleNamespace(
        impl=SimpleNamespace(supports_prequantized_qkv_input=supports),
        receives_prequantized_qkv=receives,
    )


@pytest.mark.parametrize(
    ("layers", "cache_dtype", "expected"),
    [
        ([_layer(True, True), _layer(True, True)], "fp8", True),
        # A backend-capable layer that the model never feeds prequantized
        # Q/K/V must keep the BF16 context workspace.
        ([_layer(True, False)], "fp8", False),
        ([_layer(True, True), _layer(True, False)], "fp8", False),
        ([_layer(False, True)], "fp8", False),
        ([_layer(True, True)], "auto", False),
        ([], "fp8", False),
    ],
)
def test_direct_fp8_context_gather_needs_prequantized_producers(
    monkeypatch, layers, cache_dtype, expected
):
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: False
    )
    assert rocm_aiter_fa._uses_direct_fp8_context_gather(layers, cache_dtype) is (
        expected
    )


def test_direct_fp8_context_gather_is_off_for_shuffle_layout(monkeypatch):
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: True
    )
    assert not rocm_aiter_fa._uses_direct_fp8_context_gather(
        [_layer(True, True)], "fp8"
    )


def test_fp8_context_workspace_requires_prequantized_qkv(monkeypatch):
    impl = object.__new__(rocm_aiter_fa.AiterFlashAttentionImpl)
    impl.num_kv_heads = 1
    impl.kv_cache_dtype = "fp8"
    impl.scale = 1.0
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.sinks = None
    fp8_dtype = torch.float8_e4m3fn
    monkeypatch.setattr(rocm_aiter_fa.current_platform, "fp8_dtype", lambda: fp8_dtype)
    chunk_metadata = rocm_aiter_fa.AiterChunkContextMetadata(
        workspace=torch.empty((2, 4, 1, 8), dtype=fp8_dtype),
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
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops,
        "flash_attn_varlen_func",
        lambda **kwargs: (torch.zeros(2, 4, 8), torch.zeros((1, 4))),
    )
    query, key, value = _make_tensors()
    with pytest.raises(ValueError, match="FP8 context workspace"):
        impl.extend_forward(
            attn_metadata=metadata,
            query=query,
            key=key,
            value=value,
            key_cache=torch.empty((1, 1, 1, 16), dtype=fp8_dtype),
            value_cache=torch.empty((1, 1, 1, 16), dtype=fp8_dtype),
            output=torch.empty_like(query),
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            max_seqlen_q=2,
            min_seqlen_q=1,
            max_seqlen_k=5,
            block_table=torch.zeros((1, 1), dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
            k_scale=torch.ones(1),
            v_scale=torch.ones(1),
        )


# ---------------------------------------------------------------------------
# Numerics on a gfx950 card with the real AITER kernels. Nothing above runs a
# kernel, so a wrong descale would pass every test there.
# ---------------------------------------------------------------------------


def _on_cdna4_with_aiter() -> bool:
    if not (
        current_platform.is_rocm() and IS_AITER_FOUND and torch.cuda.is_available()
    ):
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


requires_cdna4_aiter = pytest.mark.skipif(
    not _on_cdna4_with_aiter(), reason="needs a CDNA4 (gfx950) GPU with AITER"
)

HEAD_DIM = 256
ROTARY_DIM = 64
EPS = 1.0e-6
# (query heads, KV heads) per rank for the target models at TP1 to TP8.
RANK_HEADS = pytest.mark.parametrize(
    ("num_heads", "num_kv_heads"),
    [(32, 2), (16, 2), (16, 1), (8, 1), (4, 1), (2, 1)],
    ids=lambda n: str(n),
)
# (sequence lengths, leading decode sequences, padded token rows). Decodes come
# first in a batch, as the scheduler orders them.
BATCH_LAYOUTS = pytest.mark.parametrize(
    ("seq_lens", "num_decodes", "padding"),
    [
        pytest.param((513,), 0, 0, id="one-prefill"),
        pytest.param((300, 129, 2, 1000), 0, 0, id="several-prefills"),
        pytest.param((1, 1, 1, 257, 64), 3, 0, id="decodes-then-prefills"),
        pytest.param((5, 4, 6, 3) * 64, 0, 0, id="256-sequences"),
        pytest.param((200, 100), 0, 13, id="padded-rows"),
    ],
)


def _run_producer(monkeypatch, seq_lens, num_decodes, padding, heads, kv_heads):
    """Real fused Q/K norm + RoPE + gate + FP8 quant, plus its BF16 inputs."""
    device = "cuda"
    torch.manual_seed(0)
    tokens = sum(seq_lens)
    rows = tokens + padding
    # Different magnitude per sequence, so per-sequence descales differ.
    magnitude = torch.cat(
        [torch.full((n,), 1.0 + 2.0 * (i % 3)) for i, n in enumerate(seq_lens)]
        + [torch.ones(padding)]
    ).to(device)
    q_gate = (torch.randn(rows, heads * 2 * HEAD_DIM, device=device) * 2).bfloat16()
    key = (torch.randn(rows, kv_heads * HEAD_DIM, device=device) * 2).bfloat16()
    value = (
        torch.randn(rows, kv_heads * HEAD_DIM, device=device) * magnitude[:, None]
    ).bfloat16()
    query_norm_weight = (torch.randn(HEAD_DIM, device=device) * 0.1).bfloat16()
    key_norm_weight = (torch.randn(HEAD_DIM, device=device) * 0.1).bfloat16()
    cos_sin_cache = torch.randn(4096, ROTARY_DIM, device=device).bfloat16()
    positions = torch.cat([torch.arange(n) for n in seq_lens]).to(device)
    positions = torch.cat(
        [positions, torch.zeros(padding, dtype=torch.int64, device=device)]
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    metadata = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=cu_seqlens,
        num_decodes=num_decodes,
        num_decode_tokens=sum(seq_lens[:num_decodes]),
        num_extends=0,
        num_prefills=len(seq_lens) - num_decodes,
    )
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (metadata, None, None, None),
    )
    outputs = prep_module.fused_qk_norm_rope_gate_fp8_quant(
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
        "layer",
        EPS,
        heads,
        kv_heads,
        HEAD_DIM,
        ROTARY_DIM,
    )
    inputs = (
        q_gate,
        key,
        value,
        query_norm_weight,
        key_norm_weight,
        cos_sin_cache,
        positions,
    )
    return inputs, outputs, tokens, metadata


def _native_qk_gate(inputs, heads, kv_heads):
    """Gemma RMSNorm (1 + w) and NeoX RoPE in fp32, plus the gate."""
    q_gate, key, _, query_weight, key_weight, cos_sin_cache, positions = inputs
    rows = q_gate.shape[0]
    q_and_gate = q_gate.view(rows, heads, 2, HEAD_DIM).float()
    query, gate = q_and_gate[:, :, 0], q_and_gate[:, :, 1]
    key = key.view(rows, kv_heads, HEAD_DIM).float()

    def norm(x, weight):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + EPS) * (1.0 + weight.float())

    cos, sin = cos_sin_cache[positions].float().chunk(2, dim=-1)

    def rope(x):
        x1, x2 = x[..., : ROTARY_DIM // 2], x[..., ROTARY_DIM // 2 : ROTARY_DIM]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, x[..., ROTARY_DIM:]), -1)

    return rope(norm(query, query_weight)), rope(norm(key, key_weight)), gate


def _dequantize(fp8, descale, first_sequence, starts, kv_heads):
    """fp8 (tokens, heads, D) with one descale per (sequence, KV head)."""
    values = fp8.float()
    group = fp8.shape[1] // kv_heads
    for seq in range(len(starts) - 1):
        rows = slice(starts[seq], starts[seq + 1])
        for kv_head in range(kv_heads):
            heads = slice(kv_head * group, (kv_head + 1) * group)
            values[rows, heads] *= descale[first_sequence + seq, kv_head]
    return values


@requires_cdna4_aiter
@RANK_HEADS
@BATCH_LAYOUTS
def test_producer_matches_native_norm_rope_and_quantizes_per_sequence(
    monkeypatch, num_heads, num_kv_heads, seq_lens, num_decodes, padding
):
    inputs, outputs, tokens, _ = _run_producer(
        monkeypatch, seq_lens, num_decodes, padding, num_heads, num_kv_heads
    )
    query, key, gate, query_fp8, key_fp8, value_fp8 = outputs[:6]
    query_descale, key_descale, value_descale = outputs[6:]
    ref_query, ref_key, ref_gate = _native_qk_gate(inputs, num_heads, num_kv_heads)

    # The kernel rounds to BF16 and rotates in a different order than the native
    # reference, so a few near-cancelling elements differ by about one BF16 ulp.
    torch.testing.assert_close(
        query[:tokens].view(tokens, num_heads, HEAD_DIM).float(),
        ref_query[:tokens],
        rtol=2e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        key[:tokens].view(tokens, num_kv_heads, HEAD_DIM).float(),
        ref_key[:tokens],
        rtol=2e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        gate[:tokens].view(tokens, num_heads, HEAD_DIM).float(),
        ref_gate[:tokens],
        rtol=2e-2,
        atol=5e-2,
    )

    # Only the prefill suffix is quantized; its dequantized values match the
    # floating-point reference to FP8 precision (relative to the sequence's
    # largest magnitude).
    starts = [0, *torch.tensor(seq_lens).cumsum(0).tolist()]
    decode_tokens = starts[num_decodes]
    suffix = [start - decode_tokens for start in starts[num_decodes:]]
    value = inputs[2].view(-1, num_kv_heads, HEAD_DIM).float()
    for fp8, descale, reference in (
        (query_fp8, query_descale, ref_query),
        (key_fp8, key_descale, ref_key),
        (value_fp8, value_descale, value),
    ):
        dequantized = _dequantize(
            fp8[decode_tokens:tokens], descale, num_decodes, suffix, num_kv_heads
        )
        for seq in range(len(suffix) - 1):
            rows = slice(suffix[seq], suffix[seq + 1])
            expected = reference[decode_tokens:tokens][rows]
            error = (dequantized[rows] - expected).abs().max()
            assert error <= 0.08 * expected.abs().max()


def _reference_attention(q, k, v, starts, num_heads, num_kv_heads):
    outputs = []
    for seq in range(len(starts) - 1):
        rows = slice(starts[seq], starts[seq + 1])
        out = torch.nn.functional.scaled_dot_product_attention(
            q[rows].permute(1, 0, 2),
            k[rows].permute(1, 0, 2).repeat_interleave(num_heads // num_kv_heads, 0),
            v[rows].permute(1, 0, 2).repeat_interleave(num_heads // num_kv_heads, 0),
            is_causal=True,
        )
        outputs.append(out.permute(1, 0, 2))
    return torch.cat(outputs)


@requires_cdna4_aiter
@RANK_HEADS
@BATCH_LAYOUTS
def test_fp8_prefill_attention_matches_dequantized_reference(
    monkeypatch, num_heads, num_kv_heads, seq_lens, num_decodes, padding
):
    """The backend's grouped FP8 call, sliced as the backend slices, against
    attention on the same dequantized values."""
    _, outputs, tokens, _ = _run_producer(
        monkeypatch, seq_lens, num_decodes, padding, num_heads, num_kv_heads
    )
    query_fp8, key_fp8, value_fp8 = (o[:tokens] for o in outputs[3:6])
    query_descale, key_descale, value_descale = outputs[6:]
    starts = [0, *torch.tensor(seq_lens).cumsum(0).tolist()]
    decode_tokens = starts[num_decodes]
    prefill = [start - decode_tokens for start in starts[num_decodes:]]
    prequantized = rocm_aiter_fa._slice_prequantized_qkv(
        PrequantizedQKV(
            query_fp8,
            key_fp8,
            value_fp8,
            query_descale,
            key_descale,
            value_descale,
        ),
        slice(decode_tokens, None),
        slice(num_decodes, len(seq_lens)),
    )
    cu_seqlens = torch.tensor(prefill, dtype=torch.int32, device="cuda")
    output = torch.empty(
        tokens - decode_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )

    rocm_aiter_fa._grouped_fp8_varlen_attention(
        q=prequantized.query,
        k=prequantized.key,
        v=prequantized.value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        q_starts=prefill,
        k_starts=prefill,
        max_seqlen_q=max(seq_lens[num_decodes:]),
        max_seqlen_k=max(seq_lens[num_decodes:]),
        min_seqlen_q=1,
        q_descale=prequantized.query_descale,
        k_descale=prequantized.key_descale,
        v_descale=prequantized.value_descale,
        out_dtype=torch.bfloat16,
        softmax_scale=HEAD_DIM**-0.5,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        sink_ptr=None,
        out=output,
    )

    expected = _reference_attention(
        _dequantize(
            prequantized.query, query_descale, num_decodes, prefill, num_kv_heads
        ),
        _dequantize(prequantized.key, key_descale, num_decodes, prefill, num_kv_heads),
        _dequantize(
            prequantized.value, value_descale, num_decodes, prefill, num_kv_heads
        ),
        prefill,
        num_heads,
        num_kv_heads,
    )
    relative_error = (output.float() - expected).norm() / expected.norm()
    assert relative_error < 0.03
