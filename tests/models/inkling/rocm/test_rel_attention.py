# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Inkling's ROCm relative-bias paged attention."""

import pytest
import torch

from vllm.models.inkling.amd.ops.fa4_rel_attention import (
    bucket_max_seqlen_q,
    inkling_fa4_rel_attention,
    use_gfx950_gluon_decode,
    use_gfx950_gluon_extend,
)
from vllm.models.inkling.amd.ops.rel_attention_decode import (
    decode_split_count,
    use_split_kv_decode,
)
from vllm.platforms import current_platform

HEAD_DIM = 128
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
        page_size = key_cache.shape[1]
        num_pages = (kv_len + page_size - 1) // page_size
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


def test_split_kv_dispatch_thresholds():
    assert not use_split_kv_decode(
        max_query_len=4,
        max_kv_len=65536,
        page_size=128,
        window_left=-1,
    )
    assert not use_split_kv_decode(
        max_query_len=1,
        max_kv_len=4096,
        page_size=16,
        window_left=-1,
    )
    assert use_split_kv_decode(
        max_query_len=1,
        max_kv_len=8192,
        page_size=16,
        window_left=-1,
    )
    assert use_split_kv_decode(
        max_query_len=1,
        max_kv_len=512,
        page_size=128,
        window_left=511,
    )
    assert decode_split_count(512, 511) == 4
    assert decode_split_count(65536, -1) == 32


def test_gfx950_gluon_dispatch_is_strict(monkeypatch: pytest.MonkeyPatch):
    import vllm.models.inkling.amd.ops.fa4_rel_attention as rel_attention

    monkeypatch.setattr(rel_attention, "on_gfx950", lambda: True)
    assert use_gfx950_gluon_decode(max_query_len=1, page_size=128, head_dim=128)
    assert not use_gfx950_gluon_decode(max_query_len=1, page_size=16, head_dim=128)
    assert not use_gfx950_gluon_decode(max_query_len=4, page_size=128, head_dim=128)
    assert not use_gfx950_gluon_decode(max_query_len=1, page_size=128, head_dim=256)

    monkeypatch.setattr(rel_attention, "on_gfx950", lambda: False)
    assert not use_gfx950_gluon_decode(max_query_len=1, page_size=128, head_dim=128)

    monkeypatch.setattr(rel_attention, "on_gfx950", lambda: True)
    assert use_gfx950_gluon_extend(
        max_query_len=4,
        max_kv_len=8192,
        page_size=128,
        head_dim=128,
        window_left=-1,
    )
    assert not use_gfx950_gluon_extend(
        max_query_len=4,
        max_kv_len=8192,
        page_size=128,
        head_dim=128,
        window_left=511,
    )
    assert not use_gfx950_gluon_extend(
        max_query_len=4,
        max_kv_len=8192,
        page_size=16,
        head_dim=128,
        window_left=-1,
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize(
    ("page_size", "window_left", "max_kv_len"),
    [
        (16, None, None),
        (16, 15, None),
        (128, None, 8192),
    ],
)
@torch.inference_mode()
def test_ragged_multi_page_relative_attention(
    page_size: int,
    window_left: int | None,
    max_kv_len: int | None,
):
    """Covers full/chunked prefill, decode, GQA, and the local window."""
    torch.manual_seed(19 + int(window_left is not None))
    device = "cuda"
    q_lens = [17, 1]
    kv_lens = [35, 80]
    num_heads = 8
    num_kv_heads = 2
    rel_extent = 16
    max_pages = max((length + page_size - 1) // page_size for length in kv_lens)

    q = torch.randn(sum(q_lens), num_heads, HEAD_DIM, device=device)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)
    key_cache = torch.randn(
        1 + len(q_lens) * max_pages,
        page_size,
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
        max_kv_len=max_kv_len,
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


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize(
    ("page_size", "kv_lens", "num_kv_heads", "rel_extent", "window_left"),
    [
        (16, [257, 513], 1, 64, None),
        (128, [257, 512], 2, 512, 511),
    ],
)
@torch.inference_mode()
def test_split_kv_decode_matches_reference(
    page_size: int,
    kv_lens: list[int],
    num_kv_heads: int,
    rel_extent: int,
    window_left: int | None,
):
    """Covers page-16 split-KV and page-128 gfx950 Gluon decode."""
    torch.manual_seed(41 + page_size)
    device = "cuda"
    batch_size = len(kv_lens)
    q_lens = [1] * batch_size
    num_heads = 8
    max_pages = max((length + page_size - 1) // page_size for length in kv_lens)

    q = torch.randn(batch_size, num_heads, HEAD_DIM, device=device)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)
    key_cache = torch.randn(
        batch_size * max_pages,
        page_size,
        num_kv_heads,
        HEAD_DIM,
        device=device,
    )
    key_cache = torch.nn.functional.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn_like(key_cache)
    rel_logits = torch.randn(
        batch_size,
        num_heads,
        rel_extent,
        device=device,
        dtype=DTYPE,
    )
    block_table = torch.arange(
        batch_size * max_pages,
        device=device,
        dtype=torch.int32,
    ).view(batch_size, max_pages)
    cache_seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.arange(
        batch_size + 1,
        device=device,
        dtype=torch.int32,
    )
    window_size = (-1, -1) if window_left is None else (window_left, 0)
    max_kv_len = 8192 if page_size == 16 else max(kv_lens)

    actual = inkling_fa4_rel_attention(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=1.0 / HEAD_DIM,
        causal=True,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        max_kv_len=max_kv_len,
    )
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


@pytest.mark.skipif(not current_platform.is_rocm(), reason="requires ROCm")
@pytest.mark.parametrize(
    (
        "q_len",
        "kv_len",
        "num_kv_heads",
        "rel_extent",
        "window_left",
        "gluon_enabled",
    ),
    [
        (1, 513, 2, 512, 511, True),
        (1, 513, 2, 512, 511, False),
        (4, 8192, 1, 1024, None, True),
    ],
)
@torch.inference_mode()
def test_gfx950_page128_packed_kv_views_match_reference(
    monkeypatch: pytest.MonkeyPatch,
    q_len: int,
    kv_len: int,
    num_kv_heads: int,
    rel_extent: int,
    window_left: int | None,
    gluon_enabled: bool,
):
    """Cover both Gluon and split-KV against vLLM's packed KV allocation."""
    monkeypatch.setenv(
        "INKLING_GFX950_GLUON",
        "1" if gluon_enabled else "0",
    )
    torch.manual_seed(71 + q_len)
    device = "cuda"
    page_size = 128
    batch_size = 2
    num_heads = 8
    q_lens = [q_len] * batch_size
    kv_lens = [kv_len] * batch_size
    pages_per_req = (kv_len + page_size - 1) // page_size

    q = torch.randn(
        batch_size * q_len,
        num_heads,
        HEAD_DIM,
        device=device,
        dtype=DTYPE,
    )
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)

    # FlashAttentionBackend allocates logical [block, head, page, 2 * dim].
    # Inkling transposes to [block, page, head, 2 * dim] and splits K/V,
    # producing non-contiguous views whose page/head strides include both.
    packed_kv = torch.randn(
        batch_size * pages_per_req,
        num_kv_heads,
        page_size,
        2 * HEAD_DIM,
        device=device,
        dtype=DTYPE,
    )
    key_cache, value_cache = packed_kv.transpose(1, 2).split(HEAD_DIM, dim=-1)
    assert not key_cache.is_contiguous()
    assert key_cache.stride() == value_cache.stride()

    rel_logits = torch.randn(
        batch_size * q_len,
        num_heads,
        rel_extent,
        device=device,
        dtype=DTYPE,
    )
    block_table = torch.arange(
        batch_size * pages_per_req,
        device=device,
        dtype=torch.int32,
    ).view(batch_size, pages_per_req)
    cache_seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.arange(
        0,
        batch_size * q_len + 1,
        q_len,
        device=device,
        dtype=torch.int32,
    )
    window_size = (-1, -1) if window_left is None else (window_left, 0)

    actual = inkling_fa4_rel_attention(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        softmax_scale=1.0 / HEAD_DIM,
        causal=True,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        max_kv_len=kv_len,
    )
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
