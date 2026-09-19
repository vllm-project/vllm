# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    _use_rocm_sparse_triton,
    fit_kpool_indices_to_aiter,
)
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_ops
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _can_use_aiter_sparse_mla_fwd,
    _sparse_kv_row_offset,
    _validate_dsv4_sparse_dims,
    _validate_sparse_dims,
    rocm_sparse_attn_prefill,
)


@triton.jit
def _store_sparse_kv_row_offset_kernel(slot_ptr, output_ptr, stride: tl.constexpr):
    slot = tl.load(slot_ptr)
    tl.store(output_ptr, _sparse_kv_row_offset(slot, stride))


def test_fit_kpool_indices_preserves_tail_and_best_history():
    token_indices = torch.tensor(
        [
            [10, 9, 8, 7, 6, 5, 100, 101],
            [10, 9, 8, -1, -1, -1, 100, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=6)

    assert fitted.tolist() == [
        [10, 9, 8, 7, 100, 101],
        [10, 9, 8, 100, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ]


def test_fit_kpool_indices_exact_width_is_noop():
    token_indices = torch.tensor([[3, 2, 1, -1]], dtype=torch.int32)

    fitted = fit_kpool_indices_to_aiter(token_indices, topk_tokens=4)

    assert fitted.data_ptr() == token_indices.data_ptr()


def test_fit_kpool_indices_rejects_narrow_input():
    with pytest.raises(ValueError, match="at least topk_tokens"):
        fit_kpool_indices_to_aiter(
            torch.zeros((1, 3), dtype=torch.int32), topk_tokens=4
        )


@pytest.mark.parametrize(
    (
        "kv_cache_dtype",
        "head_size",
        "num_prefills",
        "num_decodes",
        "num_decode_tokens",
        "max_query_len",
        "expected",
    ),
    [
        ("auto", 512, 1, 0, 0, 32, True),
        ("auto", 512, 1, 2, 2, 32, True),
        ("auto", 512, 0, 2, 2, 1, True),
        ("fp8", 512, 1, 0, 0, 32, False),
        ("auto", 576, 1, 0, 0, 32, False),
        ("auto", 512, 0, 2, 4, 2, True),
        ("auto", 512, 0, 2, 12, 6, True),
        ("auto", 512, 0, 0, 0, 0, False),
    ],
)
def test_rocm_sparse_triton_route(
    kv_cache_dtype,
    head_size,
    num_prefills,
    num_decodes,
    num_decode_tokens,
    max_query_len,
    expected,
):
    """Validate Triton routing for prefill, decode, and MTP verification."""
    assert (
        _use_rocm_sparse_triton(
            kv_cache_dtype=kv_cache_dtype,
            head_size=head_size,
            kv_lora_rank=512,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            max_query_len=max_query_len,
        )
        is expected
    )


@pytest.mark.parametrize("num_heads", [8, 12])
def test_rocm_sparse_triton_route_preserves_padded_sinks(monkeypatch, num_heads):
    captured = {}

    def fake_rocm_sparse_attn_prefill(**kwargs):
        output = kwargs["output"]
        captured["attn_sink"] = kwargs["attn_sink"]
        output.copy_(
            captured["attn_sink"].to(output.dtype).view(1, -1, 1).expand_as(output)
        )

    monkeypatch.setattr(
        sparse_mod, "rocm_sparse_attn_prefill", fake_rocm_sparse_attn_prefill
    )

    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = 512
    impl.kv_cache_dtype = "auto"
    impl.scale = 512**-0.5
    impl.sinks = torch.arange(num_heads, dtype=torch.float32)

    q = torch.zeros(2, 16, 512, dtype=torch.bfloat16)
    kv = torch.zeros(4, 1, 512, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        max_query_len=2,
        paged_kv_indices=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(3, dtype=torch.int32),
    )

    output, lse = impl._forward_mla(SimpleNamespace(), q, kv, metadata)

    if num_heads == 8:
        expected_sinks = impl.sinks.repeat_interleave(2)
    else:
        expected_sinks = torch.cat((impl.sinks, impl.sinks[:4]))
    torch.testing.assert_close(captured["attn_sink"], expected_sinks)
    assert output.shape == (2, num_heads, 512)
    torch.testing.assert_close(
        output[:, :, 0].float(),
        impl.sinks.expand(2, -1),
    )
    assert lse is None


def test_rocm_sparse_attention_accepts_glm_nope_dimensions():
    _validate_sparse_dims(512, 512, 0, "test")


def test_rocm_sparse_attention_rejects_inconsistent_dimensions():
    with pytest.raises(AssertionError, match="expected head_dim"):
        _validate_sparse_dims(511, 512, 0, "test")


def test_dsv4_sparse_attention_keeps_layout_constraint():
    _validate_dsv4_sparse_dims(512, 448, 64, "test")
    with pytest.raises(AssertionError, match="expects 448 NoPE dims"):
        _validate_dsv4_sparse_dims(512, 512, 0, "test")


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize(
    ("q_dtype", "kv_dtype", "rope_head_dim", "on_gfx950", "expected"),
    [
        (torch.bfloat16, torch.bfloat16, 0, True, True),
        (torch.bfloat16, torch.bfloat16, 64, True, False),
        (torch.bfloat16, torch.bfloat16, 0, False, False),
        (torch.float16, torch.float16, 0, True, False),
    ],
)
def test_aiter_gluon_sparse_mla_gate(
    q_dtype, kv_dtype, rope_head_dim, on_gfx950, expected
):
    head_dim = 512 + rope_head_dim
    q = torch.zeros(4, 16, head_dim, dtype=q_dtype, device="cuda")
    kv = torch.zeros(32, head_dim, dtype=kv_dtype, device="cuda")
    output = torch.zeros(4, 16, head_dim, dtype=q_dtype, device="cuda")

    assert (
        _can_use_aiter_sparse_mla_fwd(q, kv, output, rope_head_dim, on_gfx950)
        is expected
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
@pytest.mark.parametrize("rope_head_dim", [0, 64])
def test_sparse_prefill_hands_rope_free_work_to_aiter_gluon(monkeypatch, rope_head_dim):
    captured: dict[str, Any] = {}

    def fake_sparse_mla_fwd(q, kv_buffer, kv_indptr, kv_indices, softmax_scale, **kw):
        captured.update(
            kv_lora_rank=kw["kv_lora_rank"],
            qk_rope_head_dim=kw["qk_rope_head_dim"],
            has_invalid=kw["has_invalid"],
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
        )
        kw["out"].fill_(1.0)
        return kw["out"], None

    gate = sparse_ops._can_use_aiter_sparse_mla_fwd
    monkeypatch.setattr(
        sparse_ops,
        "_can_use_aiter_sparse_mla_fwd",
        lambda *args: gate(*args, on_gfx950=True),
    )
    monkeypatch.setattr(
        sparse_ops, "_get_aiter_sparse_mla_fwd", lambda: fake_sparse_mla_fwd
    )

    head_dim = 512 + rope_head_dim
    q = torch.zeros(3, 16, head_dim, dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros(8, 1, head_dim, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(3, 16, 512, dtype=torch.bfloat16, device="cuda")
    ragged_indices = torch.tensor([0, 1, -1, 2, 3], dtype=torch.int32, device="cuda")
    ragged_indptr = torch.tensor([0, 1, 3, 5], dtype=torch.int32, device="cuda")

    rocm_sparse_attn_prefill(
        q=q,
        kv=kv,
        indices=None,
        topk_length=None,
        scale=head_dim**-0.5,
        head_dim=head_dim,
        nope_head_dim=512,
        rope_head_dim=rope_head_dim,
        attn_sink=None,
        output=output,
        ragged_indices=ragged_indices,
        ragged_indptr=ragged_indptr,
    )

    if rope_head_dim:
        assert not captured
        return
    assert captured["kv_lora_rank"] == head_dim
    assert captured["qk_rope_head_dim"] == 0
    assert captured["has_invalid"]
    torch.testing.assert_close(captured["kv_indices"], ragged_indices)
    torch.testing.assert_close(captured["kv_indptr"], ragged_indptr)
    assert output.eq(1).all()


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_sparse_prefill_kv_row_offset_does_not_overflow_int32():
    # GLM's 640-token pages cross the signed-int32 address boundary at block
    # 6554 for a 512-element KV row. The production kernel must promote the
    # slot before multiplying by the row stride.
    slot = torch.tensor([6554 * 640], dtype=torch.int32, device="cuda")
    output = torch.empty(1, dtype=torch.int64, device="cuda")

    _store_sparse_kv_row_offset_kernel[(1,)](slot, output, stride=512)

    assert output.item() == 6554 * 640 * 512
