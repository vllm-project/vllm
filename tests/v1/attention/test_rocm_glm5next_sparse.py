# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
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
        ("fp8", 512, 1, 0, 0, 32, False),
        ("auto", 576, 1, 0, 0, 32, False),
        ("auto", 512, 0, 2, 4, 2, False),
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


def _build_indexer_cache(num_blocks: int, head_dim: int, device: str, seed: int = 0):
    """A block_size=1 indexer KV cache plus the float K it encodes.

    Per block the layout is ``[block_size*head_dim fp8 | block_size*4 f32 scale]``,
    which at block_size=1 is the NORMAL (pos-major) layout both the Triton kernel
    and ``fp8_paged_mqa_logits_torch`` read -- see ``_indexer_cache_layout``.
    """
    fp8_dtype = current_platform.fp8_dtype()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(num_blocks, head_dim, generator=gen, dtype=torch.float32)
    scale = k.abs().amax(dim=-1, keepdim=True) / torch.finfo(fp8_dtype).max
    scale = scale.clamp(min=1e-6)
    k_q = (k / scale).to(fp8_dtype)

    cache = torch.empty(num_blocks, 1, 1, head_dim + 4, dtype=torch.uint8)
    cache[..., :head_dim] = k_q.view(torch.uint8).view(num_blocks, 1, 1, head_dim)
    cache[..., head_dim:] = (
        scale.contiguous().view(torch.uint8).view(num_blocks, 1, 1, 4)
    )
    # the values the cache actually encodes, after the fp8 round trip
    k_eff = k_q.to(torch.float32) * scale
    return cache.to(device), k_eff.to(device)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_paged_mqa_logits_triton_matches_torch_reference():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_torch,
        fp8_paged_mqa_logits_triton,
    )

    device, head_dim, heads = "cuda", 128, 4
    batch, num_blocks, max_model_len = 2, 64, 32
    cache, _ = _build_indexer_cache(num_blocks, head_dim, device)
    gen = torch.Generator(device="cpu").manual_seed(1)
    # q is fp8 here because that is what production passes (``q_fp8``); the
    # kernel's tl.dot requires both operands in the same fp8 type.
    q = (
        torch.randn(batch, 1, heads, head_dim, generator=gen)
        .clamp(-8.0, 8.0)
        .to(current_platform.fp8_dtype())
        .to(device)
    )
    weights = torch.rand(batch, heads, generator=gen).to(device)
    context_lens = torch.tensor([13, 32], dtype=torch.int32, device=device)
    block_tables = (
        torch.arange(batch * max_model_len, dtype=torch.int32, device=device)
        .remainder(num_blocks)
        .view(batch, max_model_len)
    )

    expected = fp8_paged_mqa_logits_torch(
        q, cache, weights, context_lens, block_tables, max_model_len
    )
    actual = fp8_paged_mqa_logits_triton(
        q, cache, weights, context_lens, block_tables, max_model_len
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm required")
def test_paged_mqa_logits_triton_masks_each_query_row_to_its_own_context_len():
    # Under MTP the indexer hands the kernel per-query context lengths as
    # [batch, next_n] -- seq_lens[b, j] already folds in the speculative offset.
    # A kernel that reads that as 1-D scores every row against a single length,
    # which silently yields the wrong top-k rather than failing; that is what
    # corrupted GLM-5.1 decode. Each row must be finite exactly below its own
    # length and -inf at or beyond it.
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        fp8_paged_mqa_logits_triton,
    )

    device, head_dim, heads = "cuda", 128, 4
    batch, next_n, num_blocks, max_model_len = 2, 2, 64, 32
    cache, _ = _build_indexer_cache(num_blocks, head_dim, device, seed=2)
    gen = torch.Generator(device="cpu").manual_seed(3)
    q = (
        torch.randn(batch, next_n, heads, head_dim, generator=gen)
        .clamp(-8.0, 8.0)
        .to(current_platform.fp8_dtype())
        .to(device)
    )
    weights = torch.rand(batch * next_n, heads, generator=gen).to(device)
    # deliberately different per row, and different between the two requests
    context_lens = torch.tensor([[9, 10], [30, 31]], dtype=torch.int32, device=device)
    block_tables = (
        torch.arange(batch * max_model_len, dtype=torch.int32, device=device)
        .remainder(num_blocks)
        .view(batch, max_model_len)
    )

    logits = fp8_paged_mqa_logits_triton(
        q, cache, weights, context_lens, block_tables, max_model_len
    )

    assert logits.shape == (batch * next_n, max_model_len)
    finite = torch.isfinite(logits)
    for row, ctx in enumerate(context_lens.flatten().tolist()):
        assert finite[row, :ctx].all(), f"row {row} has a gap below ctx={ctx}"
        assert not finite[row, ctx:].any(), f"row {row} scored past ctx={ctx}"
