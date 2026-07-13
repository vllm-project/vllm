# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.sparse_indexer_sm120 import (
    fp8_mqa_logits_sm120,
    fp8_paged_mqa_logits_sm120,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference_logits(q, k, k_scale, weights):
    scores = torch.einsum("mhd,nd->mnh", q.float(), k.float()).relu_()
    return (scores * weights.float().unsqueeze(1)).sum(dim=-1) * k_scale.float()


def test_fp8_mqa_logits_sm120_matches_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    fp8_dtype = torch.float8_e4m3fn
    q = torch.randn(5, 4, 128, device=device).to(fp8_dtype)
    k = torch.randn(11, 128, device=device).to(fp8_dtype)
    k_scale = torch.rand(11, device=device) + 0.5
    weights = torch.randn(5, 4, device=device)
    starts = torch.tensor([0, 0, 1, 2, 4], dtype=torch.int32, device=device)
    ends = torch.tensor([1, 3, 5, 8, 11], dtype=torch.int32, device=device)

    actual = fp8_mqa_logits_sm120(
        q, k, k_scale, weights, starts, ends, clean_logits=True
    )
    expected = _reference_logits(q, k, k_scale, weights)
    positions = torch.arange(k.shape[0], device=device).unsqueeze(0)
    valid = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    expected.masked_fill_(~valid, float("-inf"))

    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.02)


def test_fp8_paged_mqa_logits_sm120_matches_reference():
    torch.manual_seed(1)
    device = torch.device("cuda")
    fp8_dtype = torch.float8_e4m3fn
    batch_size, next_n, num_heads, head_dim = 2, 2, 4, 128
    block_size, num_blocks, max_model_len = 64, 4, 128

    values = torch.randn(num_blocks, block_size, head_dim, device=device).to(fp8_dtype)
    scales = torch.rand(num_blocks, block_size, device=device) + 0.5
    cache_flat = torch.empty(
        num_blocks,
        block_size * (head_dim + 4),
        dtype=torch.uint8,
        device=device,
    )
    value_bytes = block_size * head_dim
    cache_flat[:, :value_bytes].copy_(values.view(torch.uint8).view(num_blocks, -1))
    cache_flat[:, value_bytes:].copy_(
        scales.contiguous().view(torch.uint8).view(num_blocks, -1)
    )
    cache = cache_flat.view(num_blocks, block_size, 1, head_dim + 4)

    q = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(
        fp8_dtype
    )
    weights = torch.randn(batch_size * next_n, num_heads, device=device)
    context_lens = torch.tensor([[63, 64], [97, 128]], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device=device)

    actual = fp8_paged_mqa_logits_sm120(
        q,
        cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        clean_logits=True,
    )

    expected_rows = []
    weights = weights.view(batch_size, next_n, num_heads)
    for batch in range(batch_size):
        pages = block_tables[batch].long()
        k = values[pages].reshape(max_model_len, head_dim)
        k_scale = scales[pages].reshape(max_model_len)
        expected_rows.append(_reference_logits(q[batch], k, k_scale, weights[batch]))
    expected = torch.cat(expected_rows)
    positions = torch.arange(max_model_len, device=device).unsqueeze(0)
    expected.masked_fill_(~(positions < context_lens.reshape(-1, 1)), float("-inf"))

    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.02)
