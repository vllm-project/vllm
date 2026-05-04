# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the sparse MLA backends and utilities."""

import math
from types import MethodType, SimpleNamespace

import pytest
import torch

import vllm.utils.deep_gemm as deep_gemm_utils
from tests.v1.attention.test_mla_backends import (
    BATCH_SPECS,
    BatchSpec,
    MockSparseMLAAttentionLayer,
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.utils import (
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm import _custom_ops as ops
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform

# TODO: Integrate ROCMAiterMLASparseBackend for ROCm.
# The ROCm sparse MLA backend (rocm_aiter_mla_sparse.py) has a compatible
# forward_mqa interface but needs validation on ROCm hardware.
if not current_platform.is_cuda():
    pytest.skip(
        "Sparse MLA backend tests currently only support CUDA. "
        "ROCm support requires integrating ROCMAiterMLASparseBackend.",
        allow_module_level=True,
    )

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseBackend,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.indexer import (
    sparse_indexer_max_logits_bytes,
    split_indexer_prefill_chunks,
)
from vllm.v1.attention.backends.utils import split_prefill_chunks
from vllm.v1.attention.ops import flashmla
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
)

SPARSE_BACKEND_BATCH_SPECS = {
    name: BATCH_SPECS[name]
    for name in [
        "mixed_small",
        "mixed_medium",
        "small_prefill",
        "medium_prefill",
        "single_prefill",
    ]
}

SPARSE_BACKEND_BATCH_SPECS["large_q_prefill"] = BatchSpec(
    seq_lens=[1024] * 2, query_lens=[256] * 2
)
SPARSE_BACKEND_BATCH_SPECS["large_q_pure_prefill"] = BatchSpec(
    seq_lens=[256] * 2, query_lens=[256] * 2
)

DEVICE_TYPE = current_platform.device_type


def _make_packed_fp8_indexer_cache(
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
) -> torch.Tensor:
    num_blocks, block_size, num_kv_heads, head_dim = kv_fp8.shape
    assert num_kv_heads == 1
    kv_scale_bytes = kv_scale.contiguous().view(torch.uint8).reshape(
        num_blocks, block_size, num_kv_heads, -1
    )
    scale_bytes = kv_scale_bytes.shape[-1]
    fused_kv = torch.empty(
        num_blocks,
        block_size,
        head_dim + scale_bytes,
        device=kv_fp8.device,
        dtype=torch.uint8,
    )
    fused_kv_blocks = fused_kv.view(num_blocks, -1)
    value_end = block_size * head_dim
    scale_end = value_end + block_size * scale_bytes
    fused_kv_blocks[:, :value_end] = kv_fp8.view(torch.uint8).reshape(
        num_blocks, -1
    )
    fused_kv_blocks[:, value_end:scale_end] = kv_scale_bytes.reshape(
        num_blocks, -1
    )
    return fused_kv


def test_sm120_fp8_mqa_logits_chunk_sizes_cap_large_scores():
    assert deep_gemm_utils._fp8_mqa_logits_head_chunk_size(128, 128, 32) == 8
    assert deep_gemm_utils._fp8_mqa_logits_head_chunk_size(8192, 8192, 32) == 1
    assert deep_gemm_utils._fp8_mqa_logits_k_chunk_size(128, 128, 8) == 128
    assert deep_gemm_utils._fp8_mqa_logits_k_chunk_size(8192, 8192, 1) == 2048


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_tf32_hc_prenorm_gemm_fallback_matches_split_abi(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(0)
    num_tokens, out_features, hidden_size = 7, 12, 64
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(out_features, hidden_size, device="cuda", dtype=torch.float32)

    out = torch.empty(num_tokens, out_features, device="cuda", dtype=torch.float32)
    sqrsum = torch.empty(num_tokens, device="cuda", dtype=torch.float32)
    deep_gemm_utils._tf32_hc_prenorm_gemm_torch(x, fn, out, sqrsum, num_split=1)

    expected_out = x.float() @ fn.T
    expected_sqrsum = x.float().square().sum(dim=-1)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(sqrsum, expected_sqrsum, rtol=0, atol=0)

    split_out = torch.empty(3, num_tokens, out_features, device="cuda")
    split_sqrsum = torch.empty(3, num_tokens, device="cuda")
    deep_gemm_utils._tf32_hc_prenorm_gemm_torch(
        x, fn, split_out, split_sqrsum, num_split=3
    )
    torch.testing.assert_close(split_out.sum(dim=0), expected_out, rtol=0, atol=0)
    torch.testing.assert_close(split_sqrsum.sum(dim=0), expected_sqrsum, rtol=0, atol=0)

    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm_utils, "_tf32_hc_prenorm_gemm_impl", None)
    wrapper_out = torch.empty_like(split_out)
    wrapper_sqrsum = torch.empty_like(split_sqrsum)
    deep_gemm_utils.tf32_hc_prenorm_gemm(
        x, fn, wrapper_out, wrapper_sqrsum, num_split=3
    )
    torch.testing.assert_close(
        wrapper_out.sum(dim=0), expected_out, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        wrapper_sqrsum.sum(dim=0), expected_sqrsum, rtol=1e-4, atol=1e-4
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_fp8_paged_mqa_logits_fallback_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(1)
    batch_size, next_n, num_heads, head_dim = 2, 2, 4, 32
    block_size, max_model_len, num_blocks = 4, 12, 4

    q = torch.randn(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv = torch.randn(
        num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_scale = kv.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4) / 448.0
    kv_fp8 = (kv * kv_scale.reciprocal()).to(torch.float8_e4m3fn)
    fused_kv = _make_packed_fp8_indexer_cache(kv_fp8, kv_scale)

    weights = torch.randn(
        batch_size * next_n, num_heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.tensor([[3, 6], [7, 11]], device="cuda", dtype=torch.int32)
    block_tables = torch.tensor(
        [[0, 1, 2], [1, 2, 3]], device="cuda", dtype=torch.int32
    )
    expected = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    kv_dequant = kv_fp8.float() * kv_scale
    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            for token_idx in range(int(context_lens[batch_idx, next_idx].item())):
                block = int(block_tables[batch_idx, token_idx // block_size].item())
                offset = token_idx % block_size
                score = (
                    q_fp8[batch_idx, next_idx].float() * kv_dequant[block, offset, 0]
                ).sum(dim=1)
                expected[row, token_idx] = (score.relu() * weights[row]).sum()

    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm_utils, "_fp8_fp4_paged_mqa_logits_impl", None)

    def fail_torch_path(*args, **kwargs):
        raise AssertionError("torch paged fallback should not be used")

    monkeypatch.setattr(deep_gemm_utils, "_fp8_paged_mqa_logits_torch", fail_torch_path)
    actual = deep_gemm_utils.fp8_fp4_paged_mqa_logits(
        (q_fp8.contiguous(), None),
        fused_kv,
        weights,
        context_lens,
        block_tables,
        schedule_metadata=torch.empty(0, device="cuda", dtype=torch.int32),
        max_model_len=max_model_len,
        clean_logits=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)

    from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
        fp8_paged_mqa_logits_triton,
    )

    triton_actual = fp8_paged_mqa_logits_triton(
        q_fp8.contiguous(), fused_kv, weights, context_lens, block_tables, max_model_len
    )
    assert torch.equal(torch.isneginf(triton_actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    assert (triton_actual[finite] - expected[finite]).abs().max() < 2e-2


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_fp8_paged_mqa_rowwise_logits_matches_reference():
    torch.manual_seed(11)
    batch_size, next_n, num_heads, head_dim = 2, 1, 8, 64
    block_size, max_model_len, num_blocks = 4, 18, 8

    q = torch.randn(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8 = q.to(torch.float8_e4m3fn).contiguous()
    kv = torch.randn(
        num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_scale = kv.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4) / 448.0
    kv_fp8 = (kv * kv_scale.reciprocal()).to(torch.float8_e4m3fn)
    fused_kv = _make_packed_fp8_indexer_cache(kv_fp8, kv_scale)

    weights = torch.randn(
        batch_size * next_n, num_heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.tensor([[7], [17]], device="cuda", dtype=torch.int32)
    block_tables = (
        torch.arange(
            batch_size * cdiv(max_model_len, block_size),
            device="cuda",
            dtype=torch.int32,
        ).reshape(batch_size, -1)
        % num_blocks
    )

    from vllm.model_executor.layers.deepseek_v4_triton_kernels import (
        fp8_paged_mqa_logits_rowwise_triton,
    )

    actual = fp8_paged_mqa_logits_rowwise_triton(
        q_fp8, fused_kv, weights, context_lens, block_tables, max_model_len
    )
    expected = deep_gemm_utils._fp8_paged_mqa_logits_torch(
        (q_fp8, None), fused_kv, weights, context_lens, block_tables, max_model_len
    )

    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    assert (actual[finite] - expected[finite]).abs().max() < 2e-2


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_fp8_paged_mqa_topk_indices_streams_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(3)
    batch_size, next_n, num_heads, head_dim = 2, 2, 8, 32
    block_size, max_model_len, num_blocks = 4, 20, 8
    topk_tokens = 5
    monkeypatch.setattr(
        deep_gemm_utils,
        "_SM120_PAGED_MQA_TOPK_CHUNK_SIZE",
        7,
    )
    monkeypatch.setattr(
        torch,
        "cat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paged MQA top-k should reuse candidate buffers")
        ),
    )

    q = torch.randn(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv = torch.randn(
        num_blocks, block_size, 1, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_scale = kv.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4) / 448.0
    kv_fp8 = (kv * kv_scale.reciprocal()).to(torch.float8_e4m3fn)
    fused_kv = _make_packed_fp8_indexer_cache(kv_fp8, kv_scale)

    weights = torch.randn(
        batch_size * next_n, num_heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.tensor([[3, 11], [17, 20]], device="cuda", dtype=torch.int32)
    block_tables = (
        torch.arange(
            batch_size * cdiv(max_model_len, block_size),
            device="cuda",
            dtype=torch.int32,
        ).reshape(batch_size, -1)
        % num_blocks
    )
    topk_indices = torch.empty(
        batch_size * next_n, topk_tokens, device="cuda", dtype=torch.int32
    )

    assert deep_gemm_utils.fp8_fp4_paged_mqa_topk_indices(
        (q_fp8.contiguous(), None),
        fused_kv,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        topk_indices,
    )

    logits = deep_gemm_utils._fp8_paged_mqa_logits_torch(
        (q_fp8.contiguous(), None),
        fused_kv,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )
    expected = torch.full_like(topk_indices, -1)
    flat_context_lens = context_lens.reshape(-1)
    for row in range(batch_size * next_n):
        valid_count = int(flat_context_lens[row].item())
        row_topk = min(topk_tokens, valid_count)
        if row_topk > 0:
            expected[row, :row_topk] = (
                logits[row].topk(row_topk).indices.to(torch.int32)
            )

    for row in range(batch_size * next_n):
        row_topk = min(topk_tokens, int(flat_context_lens[row].item()))
        assert set(topk_indices[row, :row_topk].tolist()) == set(
            expected[row, :row_topk].tolist()
        )
        assert torch.all(topk_indices[row, row_topk:] == -1)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_fp8_mqa_logits_torch_path_streams_head_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(0)
    seq_len, seq_len_kv, num_heads, head_dim = 9, 17, 32, 32
    monkeypatch.setattr(
        deep_gemm_utils,
        "_SM120_MQA_LOGITS_MAX_SCORE_BYTES",
        seq_len * 5 * 4,
    )

    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device="cuda", dtype=torch.float32)
    cu_seqlen_ks = torch.arange(seq_len, device="cuda", dtype=torch.int32) % 3
    cu_seqlen_ke = torch.minimum(
        torch.arange(seq_len, device="cuda", dtype=torch.int32) + 4,
        torch.full((seq_len,), seq_len_kv, device="cuda", dtype=torch.int32),
    )

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_amax = kv.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    kv_scale = (kv_amax / 448.0).squeeze(1).contiguous()
    kv_fp8 = (kv * (1.0 / kv_scale[:, None])).to(torch.float8_e4m3fn)

    logits = deep_gemm_utils._fp8_mqa_logits_torch(
        (q_fp8, None),
        (kv_fp8, kv_scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=True,
    )

    kv_dequant = kv_fp8.float() * kv_scale[:, None]
    score = torch.einsum("mhd,nd->hmn", q_fp8.float(), kv_dequant)
    ref_logits = (score.relu() * weights.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
    offsets = torch.arange(seq_len_kv, device="cuda")
    valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
        offsets[None, :] < cu_seqlen_ke[:, None]
    )
    ref_logits = ref_logits.masked_fill(~valid, float("-inf"))

    assert torch.equal(torch.isneginf(logits), torch.isneginf(ref_logits))
    finite = torch.isfinite(ref_logits)
    assert (logits[finite] - ref_logits[finite]).abs().max() < 1e-4


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_fp8_mqa_logits_wrapper_uses_triton_when_deepgemm_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(2)
    seq_len, seq_len_kv, num_heads, head_dim = 5, 13, 8, 32

    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device="cuda", dtype=torch.float32)
    cu_seqlen_ks = torch.arange(seq_len, device="cuda", dtype=torch.int32) % 3
    cu_seqlen_ke = torch.minimum(
        cu_seqlen_ks + 6,
        torch.full((seq_len,), seq_len_kv, device="cuda", dtype=torch.int32),
    )

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_amax = kv.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    kv_scale = (kv_amax / 448.0).squeeze(1).contiguous()
    kv_fp8 = (kv * (1.0 / kv_scale[:, None])).to(torch.float8_e4m3fn)

    kv_dequant = kv_fp8.float() * kv_scale[:, None]
    score = torch.einsum("mhd,nd->hmn", q_fp8.float(), kv_dequant)
    expected = (score.relu() * weights.transpose(0, 1).unsqueeze(-1)).sum(dim=0)
    offsets = torch.arange(seq_len_kv, device="cuda")
    valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
        offsets[None, :] < cu_seqlen_ke[:, None]
    )
    expected = expected.masked_fill(~valid, float("-inf"))

    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm_utils, "_fp8_fp4_mqa_logits_impl", None)

    def fail_torch_path(*args, **kwargs):
        raise AssertionError("torch fallback should not be used")

    monkeypatch.setattr(deep_gemm_utils, "_fp8_mqa_logits_torch", fail_torch_path)
    actual = deep_gemm_utils.fp8_fp4_mqa_logits(
        (q_fp8, None),
        (kv_fp8, kv_scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=True,
    )

    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    assert (actual[finite] - expected[finite]).abs().max() < 2e-2


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_fp8_mqa_logits_topk_streams_k_chunks(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(1)
    seq_len, seq_len_kv, num_heads, head_dim = 11, 23, 16, 32
    topk_tokens = 5
    monkeypatch.setattr(
        deep_gemm_utils,
        "_SM120_MQA_LOGITS_MAX_SCORE_BYTES",
        seq_len * 5 * 4,
    )
    monkeypatch.setattr(
        torch,
        "cat",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("MQA top-k should reuse candidate buffers")
        ),
    )

    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device="cuda", dtype=torch.float32)
    cu_seqlen_ks = torch.arange(seq_len, device="cuda", dtype=torch.int32) % 4
    valid_lens = torch.arange(seq_len, device="cuda", dtype=torch.int32) % 7
    cu_seqlen_ke = torch.minimum(
        cu_seqlen_ks + valid_lens,
        torch.full((seq_len,), seq_len_kv, device="cuda", dtype=torch.int32),
    )

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_amax = kv.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    kv_scale = (kv_amax / 448.0).squeeze(1).contiguous()
    kv_fp8 = (kv * (1.0 / kv_scale[:, None])).to(torch.float8_e4m3fn)

    topk_indices = deep_gemm_utils._fp8_mqa_logits_topk_torch(
        (q_fp8, None),
        (kv_fp8, kv_scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_tokens,
    )

    logits = deep_gemm_utils._fp8_mqa_logits_torch(
        (q_fp8, None),
        (kv_fp8, kv_scale),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=True,
    )
    expected = torch.full_like(topk_indices, -1)
    for row in range(seq_len):
        valid_count = int((cu_seqlen_ke[row] - cu_seqlen_ks[row]).item())
        row_topk = min(topk_tokens, valid_count)
        if row_topk > 0:
            expected[row, :row_topk] = (
                logits[row].topk(row_topk).indices.to(torch.int32)
            )

    for row in range(seq_len):
        valid_count = int((cu_seqlen_ke[row] - cu_seqlen_ks[row]).item())
        row_topk = min(topk_tokens, valid_count)
        assert set(topk_indices[row, :row_topk].tolist()) == set(
            expected[row, :row_topk].tolist()
        )
        assert torch.all(topk_indices[row, row_topk:] == -1)


def _float_to_e8m0_truncate(f: float) -> float:
    """Simulate SM100's float -> e8m0 -> bf16 scale conversion.
    e8m0 format only stores the exponent (power of 2).
    cudaRoundZero truncates toward zero, meaning we round down to the
    nearest power of 2.
    """
    if f <= 0:
        return 0.0
    # e8m0 = floor(log2(f)), then 2^(e8m0)
    # This is equivalent to truncating to the nearest power of 2 below f
    exp = math.floor(math.log2(f))
    return 2.0**exp


def _dequantize_fp8_ds_mla_entry(
    cache_slice: torch.Tensor,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: torch.dtype,
    simulate_sm100_e8m0_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize a single fp8_ds_mla cache entry back to latent + rope.

    Args:
        simulate_sm100_e8m0_scales: If True, simulate the SM100 kernel's
            float -> e8m0 -> bf16 scale conversion path.
    """

    # The first kv_lora_rank bytes store FP8 latent values with one scale per
    # 128 element tile written as float32 right after the latent payload.
    scales = cache_slice.view(torch.float32)[kv_lora_rank // 4 : kv_lora_rank // 4 + 4]
    latent = torch.empty(kv_lora_rank, dtype=torch.float16, device=cache_slice.device)
    for tile_idx in range(4):
        tile_start = tile_idx * 128
        tile_end = tile_start + 128
        scale_val = float(scales[tile_idx].item())
        if simulate_sm100_e8m0_scales:
            # Simulate the lossy float -> e8m0 -> bf16 conversion
            scale_val = _float_to_e8m0_truncate(scale_val)
        ops.convert_fp8(
            latent[tile_start:tile_end],
            cache_slice[tile_start:tile_end],
            scale_val,
            kv_dtype="fp8",
        )
    latent = latent.to(dtype)

    rope_offset = kv_lora_rank // 2 + 8
    rope_vals = cache_slice.view(dtype)[rope_offset : rope_offset + rope_dim]
    return latent, rope_vals.clone()


def _quantize_dequantize_fp8_ds_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    block_size: int,
    scale: torch.Tensor,
    simulate_sm100_e8m0_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-trip kv_c/k_pe though the fp8_ds_mla cache layout.

    Args:
        simulate_sm100_e8m0_scales: If True, simulate the SM100 kernel's
            float -> e8m0 -> bf16 scale conversion in dequantization.
    """

    if kv_c.numel() == 0:
        return kv_c.clone(), k_pe.clone()

    kv_lora_rank = kv_c.shape[-1]
    rope_dim = k_pe.shape[-1]
    num_tokens = kv_c.shape[0]
    num_blocks = max(1, math.ceil(num_tokens / block_size))
    entry_size = kv_lora_rank + 4 * 4 + 2 * rope_dim

    tmp_cache = torch.zeros(
        num_blocks, block_size, entry_size, dtype=torch.uint8, device=kv_c.device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=kv_c.device)

    ops.concat_and_cache_mla(
        kv_c, k_pe, tmp_cache, slot_mapping, kv_cache_dtype="fp8_ds_mla", scale=scale
    )

    dequant_kv_c = torch.empty_like(kv_c)
    dequant_k_pe = torch.empty_like(k_pe)

    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        cache_slice = tmp_cache[block_idx, block_offset]
        latent, rope_vals = _dequantize_fp8_ds_mla_entry(
            cache_slice,
            kv_lora_rank,
            rope_dim,
            kv_c.dtype,
            simulate_sm100_e8m0_scales=simulate_sm100_e8m0_scales,
        )
        dequant_kv_c[token_idx] = latent
        dequant_k_pe[token_idx] = rope_vals

    return dequant_kv_c, dequant_k_pe


@pytest.mark.parametrize(
    "backend_cls",
    [FlashMLASparseBackend, FlashInferMLASparseBackend],
    ids=["FlashMLA", "FlashInfer"],
)
@pytest.mark.parametrize("batch_name", list(SPARSE_BACKEND_BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8", "fp8_ds_mla"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize(("q_scale", "k_scale"), [(1.0, 1.0), (2.0, 3.0)])
def test_sparse_backend_decode_correctness(
    default_vllm_config,
    dist_init,
    backend_cls,
    batch_name,
    kv_cache_dtype,
    tensor_parallel_size,
    block_size,
    workspace_init,
    q_scale: float,
    k_scale: float,
):
    if kv_cache_dtype not in backend_cls.supported_kv_cache_dtypes:
        pytest.skip(f"{backend_cls.get_name()} does not support {kv_cache_dtype}")

    if (
        backend_cls == FlashMLASparseBackend
        and kv_cache_dtype.startswith("fp8")
        and kv_cache_dtype != "fp8_ds_mla"
    ):
        pytest.skip(
            "FlashMLA Sparse Attention backend fp8 only supports "
            "fp8_ds_mla kv-cache dtype"
        )

    supported_block_sizes = backend_cls.get_supported_kernel_block_sizes()
    if block_size not in supported_block_sizes:
        pytest.skip(
            f"{backend_cls.get_name()} does not support block_size={block_size}"
        )

    if backend_cls == FlashMLASparseBackend:
        ok, reason = flashmla.is_flashmla_sparse_supported()
        if not ok:
            pytest.skip(reason)
    elif backend_cls == FlashInferMLASparseBackend:
        capability = current_platform.get_device_capability()
        if capability is None or not backend_cls.supports_compute_capability(
            capability
        ):
            pytest.skip(
                "FlashInferMLASparseBackend does not support "
                f"{capability} on this platform"
            )

    batch_spec = SPARSE_BACKEND_BATCH_SPECS[batch_name]
    use_fp8_ds_mla_quantization = kv_cache_dtype == "fp8_ds_mla"

    device = torch.device(DEVICE_TYPE)
    dtype = torch.bfloat16

    # Model hyper-parameters (kept intentionally small for the unit test)
    total_num_heads = 128
    # Compute per-rank heads for simulated TP
    num_heads = max(1, total_num_heads // tensor_parallel_size)

    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    head_size = kv_lora_rank + qk_rope_head_dim
    topk_tokens = 128

    max_seqlen = max(batch_spec.seq_lens)
    total_cache_tokens = sum(batch_spec.seq_lens)

    # Note: We use TP=1 to avoid multi-GPU requirements in CI.
    # The test simulates head partitioning via mocked methods below.
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        tensor_parallel_size=1,
        max_model_len=max_seqlen,
        num_gpu_blocks=max(2048, cdiv(total_cache_tokens, block_size) + 1),
        block_size=block_size,
        hf_config_override={
            "index_topk": topk_tokens,
            "attn_module_list_cfg": [{"topk_tokens": topk_tokens}],
        },
    )
    model_config = vllm_config.model_config
    model_config.hf_text_config = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        model_type="deepseek_v2",
    )
    model_config.dtype = dtype
    model_config.get_num_attention_heads = MethodType(
        lambda self, parallel_config: num_heads,
        model_config,
    )
    model_config.get_num_kv_heads = MethodType(
        lambda self, parallel_config: 1, model_config
    )
    model_config.get_head_size = MethodType(lambda self: head_size, model_config)
    model_config.get_sliding_window = MethodType(lambda self: None, model_config)

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    torch.manual_seed(0)

    scale = 1.0 / math.sqrt(head_size)

    # Shared MLA projection weights to keep reference and backend in sync
    W_UK = torch.rand(
        kv_lora_rank, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )
    W_UV = torch.rand(kv_lora_rank, num_heads, v_head_dim, dtype=dtype, device=device)

    # Build synthetic decode-only workload
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    # Pre-compute positions and sparse indices for all tokens.
    # We need these BEFORE computing the reference to use sparse attention masks.
    total_query_tokens = sum(query_lens)
    positions = []
    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len
        for q_idx in range(q_len):
            positions.append(ctx_len + q_idx)

    # Create sparse indices with UNIQUE per-token offsets to catch bugs where
    # the kernel uses wrong indices for some tokens (e.g., due to incorrect
    # tensor shapes like [1, num_tokens, ...] instead of [num_tokens, 1, ...]).
    # Also include -1 masked indices to verify the kernel handles them correctly.
    sparse_indices = torch.empty(
        total_query_tokens, topk_tokens, dtype=torch.int32, device=device
    )
    for tok_idx in range(total_query_tokens):
        max_valid_idx = positions[tok_idx]
        offset = tok_idx * 7  # Prime number for varied offsets
        # Use only half the topk indices as valid, mask the rest with -1
        # This tests that the kernel correctly ignores -1 indices
        num_valid = min(topk_tokens // 2, max_valid_idx + 1)
        if num_valid > 0:
            valid_range = torch.arange(num_valid, device=device, dtype=torch.int32)
            tok_indices = (valid_range + offset) % (max_valid_idx + 1)
            # Pad with -1 for the remaining positions
            tok_indices = torch.cat(
                [
                    tok_indices,
                    torch.full(
                        (topk_tokens - num_valid,), -1, device=device, dtype=torch.int32
                    ),
                ]
            )
        else:
            tok_indices = torch.full(
                (topk_tokens,), -1, device=device, dtype=torch.int32
            )
            tok_indices[0] = 0  # At least one valid index
        sparse_indices[tok_idx] = tok_indices

    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    kv_c_contexts, k_pe_contexts = [], []
    reference_outputs = []

    kv_cache_scale = torch.tensor(k_scale, dtype=torch.float32, device=device)
    global_token_idx = 0

    for i in range(batch_spec.batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        ctx_len = s_len - q_len

        q_c = torch.rand(
            q_len,
            num_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )
        kv_c_full = torch.rand(s_len, kv_lora_rank, dtype=dtype, device=device)
        k_pe_full = torch.rand(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        if use_fp8_ds_mla_quantization:
            is_sm100 = torch.cuda.get_device_capability()[0] >= 10
            kv_c_full, k_pe_squeezed = _quantize_dequantize_fp8_ds_mla(
                kv_c_full,
                k_pe_full.squeeze(1),
                block_size=block_size,
                scale=kv_cache_scale,
                simulate_sm100_e8m0_scales=is_sm100,
            )
            k_pe_full = k_pe_squeezed.unsqueeze(1)

        q_nope, q_pe = q_c.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)

        k_mqa = torch.cat([kv_c_full, k_pe_full.squeeze(1)], dim=-1)
        v_mqa = kv_c_full

        # Compute sparse SDPA reference per query token using its sparse indices
        for q_idx in range(q_len):
            tok_sparse_idx = sparse_indices[global_token_idx]
            valid_mask = tok_sparse_idx >= 0
            valid_indices = tok_sparse_idx[valid_mask].long()

            q_tok = q_mqa[q_idx : q_idx + 1]  # [1, num_heads, head_dim]
            k_sparse = k_mqa[valid_indices]  # [num_valid, head_dim]
            v_sparse = v_mqa[valid_indices]  # [num_valid, kv_lora_rank]

            k_sparse = k_sparse.unsqueeze(1).expand(-1, num_heads, -1)
            v_sparse = v_sparse.unsqueeze(1).expand(-1, num_heads, -1)

            # SDPA: [1, num_heads, 1, head_dim] x [1, num_heads, num_valid, head_dim]
            q_sdpa_in = q_tok.unsqueeze(0).transpose(1, 2)
            k_sdpa_in = k_sparse.unsqueeze(0).transpose(1, 2)
            v_sdpa_in = v_sparse.unsqueeze(0).transpose(1, 2)

            sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa_in, k_sdpa_in, v_sdpa_in, scale=scale
            )
            sdpa_out = sdpa_out.transpose(1, 2).squeeze(
                0
            )  # [1, num_heads, kv_lora_rank]

            sdpa_out = torch.einsum("qnl,lnv->qnv", sdpa_out, W_UV)
            reference_outputs.append(sdpa_out.flatten(start_dim=-2))

            global_token_idx += 1

        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[ctx_len:])
        all_k_pe_vllm.append(k_pe_full[ctx_len:])
        kv_c_contexts.append(kv_c_full[: ctx_len + 1])
        k_pe_contexts.append(k_pe_full[: ctx_len + 1])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)
    sdpa_reference = torch.cat(reference_outputs, dim=0)

    vllm_config.cache_config.cache_dtype = kv_cache_dtype
    vllm_config.model_config.hf_config.index_topk = topk_tokens

    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        vllm_config.cache_config.block_size,
        device,
        arange_block_indices=True,
    )

    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=vllm_config.cache_config.block_size,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype,
        scale=kv_cache_scale,
    )

    builder_cls = backend_cls.get_builder_cls()
    builder = builder_cls(kv_cache_spec, ["placeholder"], vllm_config, device)
    metadata = builder.build(
        common_prefix_len=0, common_attn_metadata=common_attn_metadata
    )

    # Use the pre-computed sparse_indices for the mock indexer
    mock_indexer = SimpleNamespace(topk_indices_buffer=sparse_indices)

    kv_b_proj_weight = torch.cat([W_UK, W_UV], dim=-1)
    kv_b_proj_weight = kv_b_proj_weight.view(
        kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)
    )

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank,
        output_size=num_heads * (qk_nope_head_dim + v_head_dim),
        bias=False,
    ).to(device=device, dtype=dtype)
    mock_kv_b_proj.weight = torch.nn.Parameter(kv_b_proj_weight.T.contiguous())

    impl_cls = backend_cls.get_impl_cls()
    with set_current_vllm_config(vllm_config):
        impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_b_proj=mock_kv_b_proj,
            indexer=mock_indexer,
        )

        impl.process_weights_after_loading(dtype)

        # Create mock sparse MLA layer with weight matrices
        mock_layer = MockSparseMLAAttentionLayer(
            impl=impl,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            device=device,
            W_UK=W_UK,
            W_UV=W_UV,
            q_scale=q_scale,
            k_scale=k_scale,
        )

    out_buffer = torch.empty(
        metadata.num_actual_tokens, num_heads * v_head_dim, dtype=dtype, device=device
    )

    with torch.inference_mode():
        backend_output = mock_layer.forward_impl(
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            metadata,
            out_buffer,
        )

    assert backend_output.shape == sdpa_reference.shape
    assert backend_output.dtype == sdpa_reference.dtype
    assert torch.isfinite(backend_output).all()

    # FP8 quantization introduces some error, but should be within reasonable bounds
    # BF16 (auto) should be very accurate, FP8 allows slightly more tolerance
    if kv_cache_dtype.startswith("fp8"):
        torch.testing.assert_close(
            backend_output, sdpa_reference, rtol=0.065, atol=0.05
        )
    else:
        torch.testing.assert_close(backend_output, sdpa_reference, rtol=0.01, atol=0.01)


def _triton_convert_reference_impl(
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    num_topk_tokens: int,
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation for triton_convert_req_index_to_global_index."""
    num_tokens = req_ids.shape[0]
    max_blocks_per_req = block_table.shape[1]
    result = torch.empty(
        num_tokens, num_topk_tokens, dtype=torch.int32, device=req_ids.device
    )

    for token_id in range(num_tokens):
        req_id = req_ids[token_id].item()

        # Determine if this token uses workspace or paged cache
        use_prefill_workspace = False
        workspace_start = 0
        if HAS_PREFILL_WORKSPACE and prefill_workspace_request_ids is not None:
            assert prefill_workspace_starts is not None
            prefill_req_id = prefill_workspace_request_ids[token_id].item()
            if prefill_req_id >= 0:
                use_prefill_workspace = True
                workspace_start = prefill_workspace_starts[prefill_req_id].item()

        for idx_id in range(num_topk_tokens):
            token_idx = token_indices[token_id, idx_id].item()

            if token_idx == -1:
                result[token_id, idx_id] = -1
            elif use_prefill_workspace:
                # Prefill + using prefill workspace: map to workspace offset
                result[token_id, idx_id] = workspace_start + token_idx
            else:
                # Decode: map to paged cache
                block_id = token_idx // block_size
                if block_id >= max_blocks_per_req:
                    result[token_id, idx_id] = -1
                else:
                    block_num = block_table[req_id, block_id].item()
                    offset = token_idx % block_size
                    result[token_id, idx_id] = block_num * block_size + offset

    return result


@pytest.mark.parametrize("block_size", [16, 64, 128])
@pytest.mark.parametrize("num_topk_tokens", [128, 256, 512])
@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_req_index_to_global_index_decode_only(
    block_size, num_topk_tokens
):
    device = torch.device(DEVICE_TYPE)
    num_tokens = 8
    num_requests = 4
    max_blocks_per_req = 10

    req_id = torch.randint(
        0, num_requests, (num_tokens,), dtype=torch.int32, device=device
    )
    block_table = torch.randint(
        0, 100, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )

    token_indices = torch.randint(
        0,
        block_size * max_blocks_per_req,
        (num_tokens, num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    # Set some to -1 to test masking
    token_indices[0, :10] = -1
    token_indices[3, 50:60] = -1

    # Set some to out of bounds
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    token_indices[6, 150:160] = max_blocks_per_req * block_size

    result = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
    )

    reference_result = _triton_convert_reference_impl(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
    )

    torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.skipif(
    torch.cuda.get_device_capability() < (9, 0),
    reason="FlashMLASparseBackend requires CUDA 9.0 or higher",
)
def test_triton_convert_req_index_to_global_index_with_prefill_workspace(block_size):
    device = torch.device(DEVICE_TYPE)
    num_requests = 4
    max_blocks_per_req = 8
    num_topk_tokens = 128

    # First 6 tokens are decode (reqs 0, 1), last 6 are prefill (reqs 2, 3)
    req_id = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.int32, device=device
    )
    prefill_workspace_request_ids = torch.tensor(
        [-1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=torch.int32, device=device
    )

    # Workspace starts for the 2 prefill reqs: req 2 starts at 0, req 3 starts at 100
    prefill_workspace_starts = torch.tensor([0, 100], dtype=torch.int32, device=device)

    block_table = torch.randint(
        0, 50, (num_requests, max_blocks_per_req), dtype=torch.int32, device=device
    )
    token_indices = torch.randint(
        0,
        block_size * max_blocks_per_req,
        (req_id.shape[0], num_topk_tokens),
        dtype=torch.int32,
        device=device,
    )

    # Set some to -1 to test masking
    token_indices[0, :10] = -1
    token_indices[3, 50:60] = -1

    # Set some to out of bounds
    token_indices[2, 100:110] = max_blocks_per_req * block_size
    token_indices[6, 150:160] = max_blocks_per_req * block_size

    result = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
    )

    reference_result = _triton_convert_reference_impl(
        req_id,
        block_table,
        token_indices,
        block_size,
        num_topk_tokens,
        HAS_PREFILL_WORKSPACE=True,
        prefill_workspace_request_ids=prefill_workspace_request_ids,
        prefill_workspace_starts=prefill_workspace_starts,
    )

    torch.testing.assert_close(result, reference_result, rtol=0, atol=0)


@pytest.mark.parametrize(
    "seq_lens,max_buf,expected",
    [
        # Basic split: totals per chunk ≤ max_buf
        (torch.tensor([2, 3, 4, 2]), 5, [(0, 2), (2, 3), (3, 4)]),
        # Exact fits should split between items when adding the next would overflow
        (torch.tensor([5, 5, 5]), 5, [(0, 1), (1, 2), (2, 3)]),
        # All requests fit in a single chunk
        (torch.tensor([1, 1, 1]), 10, [(0, 3)]),
        # Large buffer
        (torch.tensor([4, 4, 4]), 100, [(0, 3)]),
    ],
)
def test_split_prefill_chunks(seq_lens, max_buf, expected):
    out = split_prefill_chunks(seq_lens, max_buf)
    assert out == expected


@pytest.mark.parametrize(
    "seq_lens,query_lens,workspace_size,max_logits_bytes,expected",
    [
        # Logits constraint triggers split (M*N exceeds budget)
        # req0: M=10, N=100 -> 1000 elems (4000 bytes) - fits in 5000
        # req1: adding M=10, N=100 -> new_M=20, new_N=200 -> 4000 elems > 1250
        (
            torch.tensor([100, 100, 100]),
            torch.tensor([10, 10, 10]),
            1000,  # workspace allows all
            5000,  # 1250 float32 elems -> forces split
            [
                (slice(0, 1), slice(0, 10)),
                (slice(1, 2), slice(0, 10)),
                (slice(2, 3), slice(0, 10)),
            ],
        ),
        # Both constraints satisfied - all fit in one chunk
        (
            torch.tensor([10, 10, 10]),
            torch.tensor([5, 5, 5]),
            100,
            10000,  # 2500 elems, M*N = 15*30 = 450 < 2500
            [(slice(0, 3), slice(0, 15))],
        ),
        # Workspace constraint triggers first
        (
            torch.tensor([50, 50, 50]),
            torch.tensor([1, 1, 1]),
            50,  # workspace only fits one at a time
            1000000,  # logits budget is huge
            [
                (slice(0, 1), slice(0, 1)),
                (slice(1, 2), slice(0, 1)),
                (slice(2, 3), slice(0, 1)),
            ],
        ),
        # Greedy filling: first two fit, third doesn't
        # req0: M=5, N=10 -> 50 elems
        # req0+1: M=10, N=20 -> 200 elems <= 250
        # req0+1+2: M=15, N=30 -> 450 elems > 250
        (
            torch.tensor([10, 10, 10]),
            torch.tensor([5, 5, 5]),
            100,
            1000,  # 250 elems
            [(slice(0, 2), slice(0, 10)), (slice(2, 3), slice(0, 5))],
        ),
    ],
)
def test_split_indexer_prefill_chunks(
    seq_lens, query_lens, workspace_size, max_logits_bytes, expected
):
    out = split_indexer_prefill_chunks(
        seq_lens,
        query_lens,
        workspace_size,
        max_logits_bytes,
    )
    assert out == expected


def test_sparse_indexer_max_logits_bytes_uses_sm12x_safe_default(monkeypatch):
    monkeypatch.delenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", raising=False)

    assert sparse_indexer_max_logits_bytes(is_sm12x=True) == 256 * 1024 * 1024
    assert sparse_indexer_max_logits_bytes(is_sm12x=False) == 512 * 1024 * 1024


def test_sparse_indexer_max_logits_bytes_honors_env_override(monkeypatch):
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "384")

    assert sparse_indexer_max_logits_bytes(is_sm12x=True) == 384 * 1024 * 1024
    assert sparse_indexer_max_logits_bytes(is_sm12x=False) == 384 * 1024 * 1024


def test_compute_global_topk_indices_supports_in_place_output():
    device = torch.device(DEVICE_TYPE)
    block_size = 4
    topk_indices = torch.tensor(
        [[0, 3, 4, -1], [2, 5, -1, -1], [1, 7, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    token_to_req = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    block_table = torch.tensor(
        [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device=device
    )
    is_valid = torch.tensor([True, True, False], device=device)

    expected_indices = torch.tensor(
        [
            [40, 43, 44, -1],
            [82, 85, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    expected_lens = torch.tensor([3, 2, 0], dtype=torch.int32, device=device)

    out, lens = compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        block_table,
        block_size,
        is_valid,
    )
    torch.testing.assert_close(out, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(lens, expected_lens, rtol=0, atol=0)

    in_place = topk_indices.clone()
    provided_lens = torch.empty(3, dtype=torch.int32, device=device)
    out, lens = compute_global_topk_indices_and_lens(
        in_place,
        token_to_req,
        block_table,
        block_size,
        is_valid,
        global_topk_indices=in_place,
        topk_lens=provided_lens,
    )
    assert out is in_place
    assert lens is provided_lens
    torch.testing.assert_close(in_place, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(provided_lens, expected_lens, rtol=0, atol=0)


def test_combine_topk_swa_indices_supports_workspace_outputs():
    device = torch.device(DEVICE_TYPE)
    num_tokens = 6
    topk = 4
    window_size = 8
    topk_indices = (
        torch.arange(num_tokens * topk, dtype=torch.int32, device=device)
        .reshape(num_tokens, topk)
        .remainder(5)
    )
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([20], dtype=torch.int32, device=device)
    gather_lens = torch.tensor([8], dtype=torch.int32, device=device)

    expected_indices, expected_lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        4,
        topk,
        16,
        12,
    )
    workspace_indices = torch.empty_like(expected_indices)
    workspace_lens = torch.empty_like(expected_lens)
    actual_indices, actual_lens = combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        4,
        topk,
        16,
        12,
        combined_indices=workspace_indices,
        combined_lens=workspace_lens,
    )

    assert actual_indices.data_ptr() == workspace_indices.data_ptr()
    assert actual_lens.data_ptr() == workspace_lens.data_ptr()
    torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(actual_lens, expected_lens, rtol=0, atol=0)


def test_split_indexer_prefill_chunks_single_request_overflow():
    """Test that single request exceeding budget is sub-chunked on query dim."""
    seq_lens = torch.tensor([1000, 50])
    query_lens = torch.tensor([100, 5])

    out = split_indexer_prefill_chunks(seq_lens, query_lens, 2000, 1000)
    # max_logits_elems = 250, N=1000 -> max_q = 1 -> 100 query sub-chunks
    expected = [(slice(0, 1), slice(i, i + 1)) for i in range(100)]
    # req1: M=5, N=50 -> 250 elems fits budget
    expected.append((slice(1, 2), slice(0, 5)))
    assert out == expected


def test_triton_convert_returns_valid_counts():
    """Test that return_valid_counts correctly counts non-negative indices."""
    device = torch.device(DEVICE_TYPE)
    num_tokens = 8
    num_requests = 2
    max_blocks_per_req = 10
    block_size = 64
    num_topk_tokens = 128

    req_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)
    block_table = torch.arange(
        num_requests * max_blocks_per_req, dtype=torch.int32, device=device
    ).view(num_requests, max_blocks_per_req)

    # Create token indices with varying numbers of valid entries
    # Token 0: 64 valid, 64 invalid (-1)
    # Token 1: 32 valid, 96 invalid
    # Token 2: 128 valid (all)
    # Token 3: 1 valid, 127 invalid
    # etc.
    token_indices = torch.full(
        (num_tokens, num_topk_tokens), -1, dtype=torch.int32, device=device
    )
    expected_valid = []
    for i in range(num_tokens):
        num_valid = [64, 32, 128, 1, 64, 32, 128, 1][i]
        token_indices[i, :num_valid] = torch.arange(
            num_valid, dtype=torch.int32, device=device
        ) % (block_size * max_blocks_per_req)
        expected_valid.append(num_valid)

    expected_valid_tensor = torch.tensor(
        expected_valid, dtype=torch.int32, device=device
    )

    # Test with return_valid_counts=True
    result, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        return_valid_counts=True,
    )

    torch.testing.assert_close(valid_counts, expected_valid_tensor, rtol=0, atol=0)

    # Test that return_valid_counts=False returns only the indices
    result_only = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=block_size,
        NUM_TOPK_TOKENS=num_topk_tokens,
        return_valid_counts=False,
    )
    assert isinstance(result_only, torch.Tensor)
    torch.testing.assert_close(result_only, result, rtol=0, atol=0)
