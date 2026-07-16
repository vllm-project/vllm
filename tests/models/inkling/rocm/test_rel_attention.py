# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Inkling's ROCm relative-bias paged attention."""

import pytest
import torch

from vllm.models.inkling.amd.ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_rel_attention,
)
from vllm.platforms import current_platform

HEAD_DIM = 128
PAGE_SIZE = 16
DTYPE = torch.bfloat16


def _reference(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    block_table: torch.Tensor,
    q_lens: list[int],
    kv_lens: list[int],
    rel_extent: int,
    window_left: int | None,
) -> torch.Tensor:
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    gqa_group = num_heads // num_kv_heads
    out: list[torch.Tensor] = []
    q_start = 0

    for req, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        page_ids = block_table[req, :num_pages].long()
        k = key_cache[page_ids].reshape(-1, num_kv_heads, HEAD_DIM)[:kv_len]
        v = value_cache[page_ids].reshape(-1, num_kv_heads, HEAD_DIM)[:kv_len]
        k = k.float().repeat_interleave(gqa_group, dim=1)
        v = v.float().repeat_interleave(gqa_group, dim=1)

        q_req = q[q_start : q_start + q_len].float()
        rel_req = rel_logits[q_start : q_start + q_len].float()
        scores = torch.einsum("qhd,khd->hqk", q_req, k) / HEAD_DIM

        q_pos = torch.arange(q_len, device=q.device)[:, None] + kv_len - q_len
        k_pos = torch.arange(kv_len, device=q.device)[None, :]
        distance = q_pos - k_pos
        rel_idx = distance.clamp(0, rel_extent - 1)
        bias = rel_req.permute(1, 0, 2).gather(
            2, rel_idx[None].expand(num_heads, -1, -1)
        )
        in_rel_extent = (distance >= 0) & (distance < rel_extent)
        scores += torch.where(in_rel_extent[None], bias, 0.0)

        masked = distance < 0
        if window_left is not None:
            masked |= distance > window_left
        scores.masked_fill_(masked[None], float("-inf"))
        out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v))
        q_start += q_len

    return torch.cat(out).to(DTYPE)


def test_query_length_bucket():
    assert bucket_max_seqlen_q(1) == 1
    assert bucket_max_seqlen_q(17) == 32
    assert bucket_max_seqlen_q(33) == 64


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize("window_left", [None, 15])
@torch.inference_mode()
def test_ragged_multi_page_relative_attention(window_left: int | None):
    """Covers full/chunked prefill, decode, GQA, and the local window."""
    torch.manual_seed(19 + int(window_left is not None))
    device = "cuda"
    q_lens = [17, 1]
    kv_lens = [35, 80]
    num_heads = 8
    num_kv_heads = 2
    rel_extent = 16
    max_pages = max((length + PAGE_SIZE - 1) // PAGE_SIZE for length in kv_lens)

    q = torch.randn(sum(q_lens), num_heads, HEAD_DIM, device=device)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)
    key_cache = torch.randn(
        1 + len(q_lens) * max_pages,
        PAGE_SIZE,
        num_kv_heads,
        HEAD_DIM,
        device=device,
    )
    key_cache = torch.nn.functional.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn_like(key_cache)
    rel_logits = torch.randn(
        sum(q_lens), num_heads, rel_extent, device=device, dtype=DTYPE
    )

    block_table = torch.stack(
        [
            torch.arange(1 + req * max_pages, 1 + (req + 1) * max_pages)
            for req in range(len(q_lens))
        ]
    ).to(device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    window_size = (-1, -1) if window_left is None else (window_left, 0)

    preallocated = torch.empty_like(q)
    actual = inkling_fa4_rel_attention(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=bucket_max_seqlen_q(max(q_lens)),
        softmax_scale=1.0 / HEAD_DIM,
        causal=True,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        out=preallocated,
    )
    assert actual.data_ptr() == preallocated.data_ptr()

    expected = _reference(
        q,
        key_cache,
        value_cache,
        rel_logits,
        block_table,
        q_lens,
        kv_lens,
        rel_extent,
        window_left,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)
