# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLM blocks as varlen segments (doc_range_segments.py), against a
dense reference of the Triton mask `causal OR same block`."""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.doc_range_segments import (
    build_doc_range_segments,
    split_doc_range_segments,
)

F, T = False, True


@pytest.mark.parametrize(
    "ranges,q_lens,seq_lens,expected",
    [
        # A decode, a PrefixLM prefill, an empty row and a causal prefill.
        (
            {1: [(0, 6), (7, 9)]},
            [1, 10, 0, 5],
            [30, 10, 0, 12],
            ([0, 1, 8, 11, 16], [0, 1, 1, 3], [-1, 7, 10, -1], [F, F, F, T]),
        ),
        # 4 cached tokens: queries are positions [4, 10).
        ({0: [(0, 6), (7, 9)]}, [6], [10], ([0, 3, 6], [0, 0], [7, 10], [F, F])),
        # A step that ends at the prefix.
        ({0: [(0, 6), (7, 9)]}, [7], [7], ([0, 7], [0], [7], [F])),
    ],
    ids=["mixed_batch", "cache_hit", "ends_at_prefix"],
)
def test_split(ranges, q_lens, seq_lens, expected):
    qsl = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    out = split_doc_range_segments(
        ranges, qsl, np.asarray(seq_lens, dtype=np.int32), len(q_lens)
    )
    assert tuple(a.tolist() for a in out) == expected


def _reference(q, k, v, q_pos, ranges, scale):
    """fp32 attention of one request with the Triton mm_prefix mask."""
    kp = torch.arange(k.shape[0], device=q.device)
    mask = kp[None, :] <= q_pos[:, None]
    for s, t in ranges:
        qi = (q_pos >= s) & (q_pos <= t)
        ki = (kp >= s) & (kp <= t)
        mask |= qi[:, None] & ki[None, :]
    group = q.shape[1] // k.shape[1]
    kf = k.float().repeat_interleave(group, dim=1)
    vf = v.float().repeat_interleave(group, dim=1)
    s_ = torch.einsum("qhd,khd->hqk", q.float(), kf) * scale
    s_ = s_.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", s_.softmax(-1), vf)


def _fa4(q, k_cache, v_cache, seg, scale):
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    dynamic = seg.causal if isinstance(seg.causal, torch.Tensor) else None
    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=seg.cu_seqlens_q,
        max_seqlen_q=seg.max_seqlen_q,
        seqused_k=seg.seqused_k,
        max_seqlen_k=seg.max_seqlen_k,
        softmax_scale=scale,
        causal=dynamic is not None or seg.causal is True,
        block_table=seg.block_table,
        dynamic_causal=dynamic,
        fa_version=4,
    )
    return out


def _trtllm(q, k_cache, v_cache, seg, scale):
    """As FlashInferImpl runs them: one call over all segments, and for a
    mixed batch a causal and a dense one, picked per token."""
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    page = k_cache.shape[1]
    cum_pages = torch.zeros(
        seg.seqused_k.shape[0] + 1, dtype=torch.int32, device=q.device
    )
    torch.cumsum((seg.seqused_k + page - 1) // page, dim=0, out=cum_pages[1:])
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    # HND: (pages, kv heads, page, head dim).
    kv_cache = tuple(c.transpose(1, 2).contiguous() for c in (k_cache, v_cache))

    def run(causal: bool) -> torch.Tensor:
        return trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=seg.block_table,
            seq_lens=seg.seqused_k,
            max_q_len=seg.max_seqlen_q,
            max_kv_len=seg.max_seqlen_k,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            batch_size=seg.seqused_k.shape[0],
            cum_seq_lens_q=seg.cu_seqlens_q,
            cum_seq_lens_kv=cum_pages,
            causal=causal,
        )

    if not isinstance(seg.causal, torch.Tensor):
        return run(seg.causal)
    q_seg = seg.cu_seqlens_q[1:] - seg.cu_seqlens_q[:-1]
    token_causal = torch.repeat_interleave(seg.causal.bool(), q_seg)
    return torch.where(token_causal[:, None, None], run(True), run(False))


# (seq_len, query_len, ranges): prefill, cache hit, decode, causal prefill, and
# a step that ends at the prefix.
REQS = [
    (700, 700, [(0, 499), (500, 699)]),
    (900, 500, [(0, 599), (600, 899)]),
    (333, 1, []),
    (257, 257, []),
    (400, 400, [(0, 399), (400, 599)]),
]


@pytest.mark.parametrize("ranged_only", [False, True])
def test_segments_match_reference(ranged_only):
    cuda = current_platform.is_cuda()
    if cuda and current_platform.is_device_capability_family(90):
        attention = _fa4
    elif cuda and current_platform.is_device_capability_family(100):
        attention = _trtllm
    else:
        pytest.skip("doc range segments run on SM90 (FA4) and SM100 (trtllm-gen)")
    torch.manual_seed(0)
    dev = "cuda"
    page, num_heads, num_kv_heads, head_dim = 64, 8, 2, 512
    reqs = [r for r in REQS if r[2] or not ranged_only]
    pages = [(s + page - 1) // page for s, _, _ in reqs]
    num_pages = sum(pages) + 4
    # NHD: (pages, page, kv heads, head dim).
    shape = (num_pages, page, num_kv_heads, head_dim)
    k_cache = torch.randn(shape, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.zeros(len(reqs), max(pages), dtype=torch.int32, device=dev)
    perm = torch.randperm(num_pages, device=dev).to(torch.int32)
    nxt = 0
    for i, n in enumerate(pages):
        block_table[i, :n] = perm[nxt : nxt + n]
        nxt += n
    q_lens = [ql for _, ql, _ in reqs]
    qsl = torch.tensor(np.concatenate([[0], np.cumsum(q_lens)]), dtype=torch.int32)
    seq_lens_cpu = torch.tensor([s for s, _, _ in reqs], dtype=torch.int32)
    q = torch.randn(int(qsl[-1]), num_heads, head_dim, device=dev).bfloat16()
    ranges = {i: r for i, (_, _, r) in enumerate(reqs) if r}
    seg = build_doc_range_segments(
        ranges, qsl, seq_lens_cpu, seq_lens_cpu.to(dev), block_table, len(reqs)
    )
    if ranged_only:
        assert seg.causal is False
    else:
        assert isinstance(seg.causal, torch.Tensor)
    scale = head_dim**-0.5
    out = attention(q, k_cache, v_cache, seg, scale)
    for i, (s, ql, r) in enumerate(reqs):
        blocks = block_table[i, : pages[i]].long()
        k = k_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:s]
        v = v_cache[blocks].reshape(-1, num_kv_heads, head_dim)[:s]
        q_pos = torch.arange(s - ql, s, device=dev)
        ref = _reference(q[qsl[i] : qsl[i + 1]], k, v, q_pos, r, scale)
        torch.testing.assert_close(
            out[qsl[i] : qsl[i + 1]].float(), ref, atol=2e-2, rtol=2e-2
        )
