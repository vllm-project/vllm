# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_indexed_d512_split_sparse_mla_attention,
    accumulate_indexed_sparse_mla_attention_chunk,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexed_d512_split_sparse_mla_matches_indexed_accumulate():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(17)
    num_tokens = 64
    num_heads = 8
    head_dim = 512
    num_candidates = 640
    kv_tokens = 4096
    scale = head_dim**-0.5

    q = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv = torch.randn(kv_tokens, head_dim, device=device, dtype=torch.bfloat16)
    indices = torch.randint(
        0,
        kv_tokens,
        (num_tokens, num_candidates),
        device=device,
        dtype=torch.int32,
    )
    lens = torch.randint(
        num_candidates // 2,
        num_candidates + 1,
        (num_tokens,),
        device=device,
        dtype=torch.int32,
    )

    current_max = torch.full(
        (num_tokens, num_heads),
        -float("inf"),
        device=device,
        dtype=torch.float32,
    )
    current_denom = torch.zeros_like(current_max)
    current_acc = torch.zeros(
        num_tokens, num_heads, head_dim, device=device, dtype=torch.float32
    )
    split_max = torch.empty_like(current_max)
    split_denom = torch.empty_like(current_denom)
    split_acc = torch.empty_like(current_acc)
    split_scores = torch.empty(
        num_tokens,
        num_heads,
        num_candidates,
        device=device,
        dtype=torch.float32,
    )

    accumulate_indexed_sparse_mla_attention_chunk(
        q=q,
        kv_flat=kv,
        indices=indices,
        lens=lens,
        scale=scale,
        max_score=current_max,
        denom=current_denom,
        acc=current_acc,
    )
    accumulate_indexed_d512_split_sparse_mla_attention(
        q=q,
        kv_flat=kv,
        indices=indices,
        lens=lens,
        scale=scale,
        max_score=split_max,
        denom=split_denom,
        acc=split_acc,
        scores=split_scores,
    )
    torch.cuda.synchronize()

    current = current_acc / current_denom[:, :, None]
    split = split_acc / split_denom[:, :, None]
    torch.testing.assert_close(split_max, current_max, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(split_denom, current_denom, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(split, current, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexed_d512_split_sparse_mla_matches_c128_combined_width():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(23)
    num_tokens = 64
    num_heads = 8
    head_dim = 512
    num_candidates = 1152
    kv_tokens = 4096
    scale = head_dim**-0.5

    q = torch.randn(
        num_tokens,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv = torch.randn(kv_tokens, head_dim, device=device, dtype=torch.bfloat16)
    indices = torch.randint(
        0,
        kv_tokens,
        (num_tokens, num_candidates),
        device=device,
        dtype=torch.int32,
    )
    lens = torch.randint(
        128,
        1097,
        (num_tokens,),
        device=device,
        dtype=torch.int32,
    )

    current_max = torch.full(
        (num_tokens, num_heads),
        -float("inf"),
        device=device,
        dtype=torch.float32,
    )
    current_denom = torch.zeros_like(current_max)
    current_acc = torch.zeros(
        num_tokens, num_heads, head_dim, device=device, dtype=torch.float32
    )
    split_max = torch.empty_like(current_max)
    split_denom = torch.empty_like(current_denom)
    split_acc = torch.empty_like(current_acc)
    split_scores = torch.empty(
        num_tokens,
        num_heads,
        num_candidates,
        device=device,
        dtype=torch.float32,
    )

    accumulate_indexed_sparse_mla_attention_chunk(
        q=q,
        kv_flat=kv,
        indices=indices,
        lens=lens,
        scale=scale,
        max_score=current_max,
        denom=current_denom,
        acc=current_acc,
    )
    accumulate_indexed_d512_split_sparse_mla_attention(
        q=q,
        kv_flat=kv,
        indices=indices,
        lens=lens,
        scale=scale,
        max_score=split_max,
        denom=split_denom,
        acc=split_acc,
        scores=split_scores,
    )
    torch.cuda.synchronize()

    current = current_acc / current_denom[:, :, None]
    split = split_acc / split_denom[:, :, None]
    torch.testing.assert_close(split_max, current_max, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(split_denom, current_denom, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(split, current, atol=2e-3, rtol=2e-3)
