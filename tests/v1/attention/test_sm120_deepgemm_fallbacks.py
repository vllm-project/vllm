# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.utils.deep_gemm as deep_gemm_utils
from vllm.model_executor.layers.sparse_attn_indexer import (
    _decode_logits_width,
    _decode_topk_logits_width,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.deepseek_v4_ops import sm12x_deep_gemm_fallbacks


def _make_indexer_kv_cache(
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
) -> torch.Tensor:
    num_blocks, block_size, num_kv_heads, head_dim = kv_fp8.shape
    assert num_kv_heads == 1
    fused_kv = torch.empty(
        num_blocks,
        block_size,
        1,
        head_dim + torch.float32.itemsize,
        device=kv_fp8.device,
        dtype=torch.uint8,
    )
    block_stride = fused_kv.stride(0)
    kv_values = torch.as_strided(
        fused_kv,
        size=kv_fp8.shape,
        stride=(block_stride, head_dim, head_dim, 1),
    )
    kv_scales = torch.as_strided(
        fused_kv,
        size=(num_blocks, block_size, 1, torch.float32.itemsize),
        stride=(block_stride, torch.float32.itemsize, torch.float32.itemsize, 1),
        storage_offset=block_size * head_dim,
    )
    kv_values.copy_(kv_fp8.view(torch.uint8))
    kv_scales.copy_(kv_scale.contiguous().view(torch.uint8))
    return fused_kv


def _reference_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    batch_size, next_n, _, _ = q_fp8.shape
    _, block_size, _, _ = kv_fp8.shape
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    q = q_fp8.float()
    kv = kv_fp8.float() * kv_scale.float()
    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = min(
                int(context_lens[batch_idx, next_idx].item()),
                max_model_len,
            )
            for token_idx in range(context_len):
                block_idx = block_tables[batch_idx, token_idx // block_size]
                block_offset = token_idx % block_size
                k = kv[block_idx, block_offset, 0]
                scores = (q[batch_idx, next_idx] * k).sum(dim=-1).relu()
                logits[row, token_idx] = (scores * weights[row]).sum()
    return logits


def test_decode_logits_width_uses_active_context_bound():
    assert _decode_logits_width(262144, 1024) == 1024
    assert _decode_logits_width(4096, 8192) == 4096
    assert _decode_logits_width(4096, 0) == 4096
    assert _decode_logits_width(0, 1024) == 0


def test_decode_topk_logits_width_keeps_topk_kernel_width():
    assert _decode_topk_logits_width(262144, 1024, 512) == 1024
    assert _decode_topk_logits_width(262144, 128, 512) == 512
    assert _decode_topk_logits_width(300, 128, 512) == 300
    assert _decode_topk_logits_width(0, 128, 512) == 0


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="SM120 only"
)
def test_sm120_paged_mqa_direct_topk_matches_truncated_decode_width(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(7)
    batch_size, next_n, num_heads, head_dim = 2, 2, 8, 32
    block_size, max_model_len, num_blocks = 4, 64, 16
    active_max_len = 13
    topk_tokens = 6
    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        sm12x_deep_gemm_fallbacks,
        "_SM120_PAGED_MQA_TOPK_CHUNK_SIZE",
        7,
    )

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
    fused_kv = _make_indexer_kv_cache(kv_fp8, kv_scale)

    weights = torch.randn(
        batch_size * next_n, num_heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.tensor(
        [[7, active_max_len], [9, 12]], device="cuda", dtype=torch.int32
    )
    block_tables = (
        torch.arange(
            batch_size * cdiv(max_model_len, block_size),
            device="cuda",
            dtype=torch.int32,
        ).reshape(batch_size, -1)
        % num_blocks
    )

    full_width_topk = torch.empty(
        batch_size * next_n, topk_tokens, device="cuda", dtype=torch.int32
    )
    truncated_width_topk = torch.empty_like(full_width_topk)

    assert deep_gemm_utils.fp8_fp4_paged_mqa_topk_indices(
        (q_fp8, None),
        fused_kv,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        full_width_topk,
    )
    assert deep_gemm_utils.fp8_fp4_paged_mqa_topk_indices(
        (q_fp8, None),
        fused_kv,
        weights,
        context_lens,
        block_tables,
        active_max_len,
        truncated_width_topk,
    )

    reference_logits = _reference_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_tables,
        active_max_len,
    )
    expected_topk = torch.topk(reference_logits, topk_tokens, dim=1).indices.to(
        torch.int32
    )

    torch.testing.assert_close(truncated_width_topk, full_width_topk, rtol=0, atol=0)
    torch.testing.assert_close(truncated_width_topk, expected_topk, rtol=0, atol=0)
