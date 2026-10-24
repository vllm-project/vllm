# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged Step-5 CSA kernels vs PyTorch references on scrambled block tables.

Each test mixes requests with their own block-table rows, and prefill, chunked
prefill and decode rows, since a page- or request-addressing bug reads wrong
tokens without changing any shape.
"""

import pytest
import torch

from vllm.models.step5.common.ops.csa_paged import sparse_attention_paged
from vllm.models.step5.nvidia.ops.sparse_decode_attention import (
    sparse_decode_attention,
)
from vllm.models.step5.nvidia.ops.sparse_indexer import select_regions
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Step5 CSA kernels are the ROCm path"
)

DEVICE = "cuda"
REGION = 8
PROXY = 256


def _paged(
    seqs: list[torch.Tensor], rows_per_page: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter each request's rows into shared pages behind its own row of a
    shuffled ``[requests, pages]`` block table."""
    pages = [-(-x.shape[0] // rows_per_page) for x in seqs]
    perm = torch.randperm(sum(pages) + 3, device=DEVICE).int()
    table = torch.zeros(len(seqs), max(pages), dtype=torch.int32, device=DEVICE)
    cache = torch.zeros(
        (sum(pages) + 3, rows_per_page, *seqs[0].shape[1:]),
        dtype=seqs[0].dtype,
        device=DEVICE,
    )
    start = 0
    for r, (x, n) in enumerate(zip(seqs, pages)):
        table[r, :n] = perm[start : start + n]
        start += n
        idx = torch.arange(x.shape[0], device=DEVICE)
        cache[table[r, idx // rows_per_page].long(), idx % rows_per_page] = x
    return cache, table


def ref_select(scores: torch.Tensor, positions: torch.Tensor, topk: int):
    tokens, groups, _ = scores.shape
    out = torch.full((tokens, groups, topk + 1), -1, dtype=torch.int32)
    for t in range(tokens):
        own = int(positions[t]) // REGION
        for g in range(groups):
            order = torch.argsort(-scores[t, g, :own].cpu(), stable=True)[:topk]
            row = torch.cat([torch.sort(order).values, torch.tensor([own])])
            out[t, g, : row.numel()] = row.to(torch.int32)
    return out


def ref_attention(q, k, v, regions, req, positions, scale):
    tokens, heads, _ = q.shape
    groups = k.shape[2]
    per_group = heads // groups
    out = torch.zeros(q.shape, dtype=torch.float32)
    for t in range(tokens):
        kr, vr = k[int(req[t])], v[int(req[t])]
        for g in range(groups):
            ids = regions[t, g]
            ids = ids[ids >= 0].long().cpu()
            cols = (ids[:, None] * REGION + torch.arange(REGION)).flatten()
            cols = cols[cols <= int(positions[t])]
            hs = slice(g * per_group, (g + 1) * per_group)
            p = torch.softmax(
                q[t, hs].float().cpu() @ kr[cols, g].float().cpu().T * scale, -1
            )
            out[t, hs] = p @ vr[cols, g].float().cpu()
    return out


def test_select_regions_breaks_ties_by_region_id():
    """Post-ReLU scores are often exactly zero; ties must pick lower ids."""
    torch.manual_seed(0)
    scores = torch.randn(48, 2, 40, device=DEVICE)
    scores[scores.abs() < 0.5] = 0.0
    positions = torch.randint(0, 40 * REGION, (48,), device=DEVICE)
    own = (positions // REGION).repeat_interleave(2)
    got = select_regions(scores.flatten(0, 1), own, 6, "per_row").view(48, 2, -1)
    want = ref_select(scores, positions, 6)
    for g, w in zip(got.cpu(), want):
        for gg, ww in zip(g, w):
            assert sorted(gg[gg >= 0].tolist()) == sorted(ww[ww >= 0].tolist())


FP8_KV = [None, torch.float8_e4m3fn, torch.float8_e4m3fnuz]


def _quant_kv(k, v, kv_dtype):
    """KV as stored (FP8 with per-tensor scales, or BF16) and as the reference
    sees it."""
    if kv_dtype is None:
        return k, v, k, v, None, None
    k_s, v_s = 0.5, 2.0
    fmax = torch.finfo(kv_dtype).max
    kq = (k.float() / k_s).clamp(-fmax, fmax).to(kv_dtype)
    vq = (v.float() / v_s).clamp(-fmax, fmax).to(kv_dtype)
    scales = [torch.tensor([x], dtype=torch.float32, device=DEVICE) for x in (k_s, v_s)]
    return kq, vq, kq.float() * k_s, vq.float() * v_s, *scales


@pytest.mark.parametrize("kv_dtype", FP8_KV)
@pytest.mark.parametrize("topk", [6, 512])
def test_sparse_attention_paged(topk: int, kv_dtype):
    """topk=512 covers the whole history, so it also checks dense equivalence."""
    torch.manual_seed(0)
    seq_len, heads, groups, dim = 203, 16, 4, 192
    shape = (2, seq_len, groups, dim)
    k = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    kq, vq, k, v, k_scale, v_scale = _quant_kv(k, v, kv_dtype)
    pos = torch.cat([torch.arange(seq_len), torch.tensor([202])]).int().to(DEVICE)
    req = torch.tensor([0] * seq_len + [1], dtype=torch.int32, device=DEVICE)
    q = torch.randn(len(pos), heads, dim, dtype=torch.bfloat16, device=DEVICE)
    scores = torch.randn(len(pos), groups, seq_len // REGION + 1, device=DEVICE)
    regions = ref_select(scores, pos, topk).to(DEVICE)

    kv_cache, bt = _paged([torch.stack([kq[r], vq[r]], 1) for r in range(2)], 64)
    got = torch.empty_like(q)
    sparse_attention_paged(
        q,
        kv_cache[:, :, 0],
        kv_cache[:, :, 1],
        regions,
        bt,
        req,
        pos,
        got,
        region_size=REGION,
        scale=dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    want = ref_attention(q, k, v, regions, req, pos, dim**-0.5)
    torch.testing.assert_close(got.float().cpu(), want, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("kv_dtype", FP8_KV)
def test_sparse_decode_attention(kv_dtype):
    """One decode row per request through the split-K kernel the model uses."""
    torch.manual_seed(0)
    seq_len, heads, groups, dim, topk = 203, 16, 4, 192, 6
    shape = (2, seq_len, groups, dim)
    k = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    kq, vq, k, v, k_scale, v_scale = _quant_kv(k, v, kv_dtype)
    pos = torch.tensor([seq_len - 1, seq_len - 9], dtype=torch.int32, device=DEVICE)
    req = torch.arange(2, dtype=torch.int32, device=DEVICE)
    q = torch.randn(2, heads, dim, dtype=torch.bfloat16, device=DEVICE)
    scores = torch.randn(2, groups, seq_len // REGION + 1, device=DEVICE)
    regions = ref_select(scores, pos, topk).to(DEVICE)

    kv_cache, bt = _paged([torch.cat([kq[r], vq[r]], -1) for r in range(2)], 16)
    got = torch.empty_like(q)
    sparse_decode_attention(
        q.view(2 * groups, -1, dim),
        kv_cache.transpose(1, 2).contiguous(),
        regions.view(2 * groups, -1).long(),
        bt,
        pos.long().repeat_interleave(groups),
        torch.ones(2 * groups, dtype=torch.bool, device=DEVICE),
        got.view(2 * groups, -1, dim),
        scale=dim**-0.5,
        splits=4,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    want = ref_attention(q, k, v, regions, req, pos, dim**-0.5)
    torch.testing.assert_close(got.float().cpu(), want, atol=2e-2, rtol=2e-2)


def test_summaries_on_coarse_summary_blocks():
    """ROCm keeps summaries in their own KV group, whose block (384 tokens,
    48 regions) differs from the main 16-token block; summaries and decode
    scores must match the main-block layout."""
    from vllm.models.step5.nvidia.ops.sparse_indexer_scores import (
        decode_summary_scores,
    )
    from vllm.models.step5.nvidia.ops.summary_update import (
        build_prefill_groups,
        update_prefill,
    )

    torch.manual_seed(0)
    seq, capacity = 1001, 12
    regions = seq // REGION
    key = torch.randn((seq, 1, PROXY), device=DEVICE).to(torch.bfloat16)
    z = torch.randn((seq, 1, PROXY), device=DEVICE).to(torch.bfloat16)
    positions = torch.arange(seq, device=DEVICE)
    query = torch.randn((2, 1, 8, PROXY), device=DEVICE).to(torch.float8_e4m3fn)
    weights = torch.rand((2, 1, 8), device=DEVICE)

    def run(block: int):
        blocks = -(-seq // block)
        table = torch.randperm(blocks + 2, device=DEVICE)[:blocks].int()
        mean = torch.zeros(
            (blocks + 2, block // REGION, PROXY), dtype=torch.uint8, device=DEVICE
        )
        rows = torch.zeros((capacity, 2 * PROXY), dtype=torch.bfloat16, device=DEVICE)
        slots = table[positions // block].long() * block + positions % block
        groups = build_prefill_groups(
            slots,
            positions,
            torch.zeros_like(positions),
            torch.tensor([0, seq], device=DEVICE),
        )
        keep = positions >= seq - capacity
        update_prefill(
            key,
            z,
            groups,
            positions,
            torch.zeros_like(positions),
            torch.where(keep, positions % capacity, -1),
            mean,
            rows,
            capacity,
        )
        r = torch.arange(regions, device=DEVICE)
        per_region = mean[table[r // mean.shape[1]].long(), r % mean.shape[1]]
        scores, meta = decode_summary_scores(
            query,
            weights,
            mean,
            table[None].expand(2, -1).contiguous(),
            slots[-1:].expand(2).contiguous(),
            torch.tensor([seq, seq - 9], device=DEVICE),
            regions,
            num_sms=4,
        )
        return per_region, scores, meta.lengths

    fine, fine_scores, lengths = run(16)
    coarse, coarse_scores, _ = run(384)
    assert fine.any() and torch.equal(fine, coarse)
    for row, length in enumerate(lengths.tolist()):
        assert torch.equal(fine_scores[row, :length], coarse_scores[row, :length])
