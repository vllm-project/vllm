# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse_mod
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    _use_rocm_sparse_triton,
    fit_kpool_indices_to_aiter,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _sparse_kv_row_offset,
    _validate_dsv4_sparse_dims,
    _validate_sparse_dims,
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
        ("fp8", 512, 1, 0, 0, 32, True),
        ("fp8_e4m3", 512, 0, 2, 2, 1, True),
        ("auto", 576, 1, 0, 0, 32, False),
        ("auto", 512, 0, 2, 4, 2, False),
        ("fp8", 576, 1, 0, 0, 32, False),
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
def test_sparse_prefill_kv_row_offset_does_not_overflow_int32():
    # GLM's 640-token pages cross the signed-int32 address boundary at block
    # 6554 for a 512-element KV row. The production kernel must promote the
    # slot before multiplying by the row stride.
    slot = torch.tensor([6554 * 640], dtype=torch.int32, device="cuda")
    output = torch.empty(1, dtype=torch.int64, device="cuda")

    _store_sparse_kv_row_offset_kernel[(1,)](slot, output, stride=512)

    assert output.item() == 6554 * 640 * 512


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_sparse_prefill_fp8_nope_matches_bf16_reference():
    """FP8 NoPE KV must dequant in-kernel, not route to the 576-wide asm kernel.

    The AITER mla_a8w8 HSACO reads 576 B/row. A 512-wide GLM cache would
    overrun. This kernel is the replacement path: same ragged gather, FP8
    values, and a tensor scale.
    """
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _rocm_sparse_attn_prefill_ragged_triton,
    )

    torch.manual_seed(0)
    device = "cuda"
    num_queries, num_heads, head_dim = 4, 16, 512
    num_kv = 32
    kv_scale = 1.5
    q = torch.randn(
        num_queries, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    kv_f32 = torch.randn(num_kv, head_dim, dtype=torch.float32, device=device)
    # Must be the platform's cache dtype: gfx950 stores OCP e4m3, gfx942 fnuz.
    kv_fp8 = (kv_f32 / kv_scale).to(current_platform.fp8_dtype())
    kv_bf16 = (kv_fp8.float() * kv_scale).to(torch.bfloat16)
    indices = torch.arange(num_queries * 8, device=device, dtype=torch.int32) % num_kv
    indptr = torch.arange(0, num_queries * 8 + 1, 8, device=device, dtype=torch.int32)

    out_fp8 = _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv_fp8,
        indices=indices,
        indptr=indptr,
        scale=head_dim**-0.5,
        attn_sink=None,
        nope_head_dim=head_dim,
        rope_head_dim=0,
        kv_scale=kv_scale,
    )
    out_bf16 = _rocm_sparse_attn_prefill_ragged_triton(
        q=q,
        kv=kv_bf16,
        indices=indices,
        indptr=indptr,
        scale=head_dim**-0.5,
        attn_sink=None,
        nope_head_dim=head_dim,
        rope_head_dim=0,
        kv_scale=1.0,
    )
    # Both sides consume identical FP8 values, so the only spread is bf16
    # rounding of the dequantized product. A dropped or doubled scale moves
    # the result by ~50%, far outside this band.
    torch.testing.assert_close(out_fp8, out_bf16, rtol=5e-3, atol=5e-3)


def _nope_impl_for_asm_guard(num_decode_tokens, num_decodes, max_query_len):
    impl = object.__new__(sparse_mod.ROCMAiterMLASparseImpl)
    impl.num_heads = 16
    impl.kv_lora_rank = 512
    impl.kv_cache_dtype = "fp8_e4m3"
    impl.scale = 512**-0.5
    impl.sinks = None
    q = torch.zeros(num_decode_tokens, 16, 512, dtype=torch.bfloat16)
    kv = torch.zeros(4, 1, 512, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        attn_out_dtype=torch.bfloat16,
        num_prefills=0,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        max_query_len=max_query_len,
        paged_kv_indices=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(num_decode_tokens + 1, dtype=torch.int32),
    )
    return impl, q, kv, metadata


def test_nope_rows_never_reach_aiter_asm_kernels():
    """Speculative decode widens the decode step past the Triton gate.

    AITER's precompiled kernels assume a 576-byte row, so a 512-byte NoPE row
    must raise rather than fault the GPU.
    """
    impl, q, kv, metadata = _nope_impl_for_asm_guard(
        num_decode_tokens=4, num_decodes=2, max_query_len=2
    )

    with pytest.raises(NotImplementedError, match="576-byte KV row"):
        impl._forward_mla(SimpleNamespace(), q, kv, metadata)
