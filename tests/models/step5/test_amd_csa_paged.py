# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged Step-5 CSA kernels vs PyTorch references on scrambled block tables.

Each test mixes requests with their own block-table rows, and prefill, chunked
prefill and decode rows, since a page- or request-addressing bug reads wrong
tokens without changing any shape.
"""

import pytest
import torch

from vllm.models.step5.common.ops.csa import round_e4m3
from vllm.models.step5.common.ops.csa_paged import (
    compress_regions_paged,
    indexer_scores_paged,
    sparse_attention_paged,
)
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


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    return round_e4m3(x).float()


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


def ref_compress(k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    regions = k.shape[0] // REGION
    k = k[: regions * REGION].float().view(regions, REGION, -1)
    z = z[: regions * REGION].float().view(regions, REGION, -1)
    w = torch.exp(z - z.amax(dim=1, keepdim=True))
    return _e4m3((w * k).sum(1) / w.sum(1))


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


def test_round_e4m3_saturates():
    x = torch.tensor([447.0, 500.0, -1e4], device=DEVICE)
    assert round_e4m3(x).tolist() == [448.0, 448.0, -448.0]


def test_compress_regions_paged():
    torch.manual_seed(0)
    lens = [64, 203]
    ks = [torch.randn(n, PROXY, dtype=torch.bfloat16, device=DEVICE) for n in lens]
    zs = [torch.randn(n, PROXY, dtype=torch.bfloat16, device=DEVICE) * 4 for n in lens]
    zs[1][:, :128] += 200
    index_cache, index_bt = _paged([torch.cat(kz, -1) for kz in zip(ks, zs)], 64)
    regions = [n // REGION for n in lens]
    summary_cache, summary_bt = _paged(
        [torch.zeros(n, PROXY, dtype=torch.bfloat16, device=DEVICE) for n in regions],
        4,
    )
    work = [(req, r) for req, n in enumerate(regions) for r in range(n)]
    work = torch.tensor(work + [(-1, -1)] * 3, dtype=torch.int32, device=DEVICE)
    compress_regions_paged(
        index_cache, summary_cache, index_bt, summary_bt, work, region_size=REGION
    )
    for req, n in enumerate(regions):
        r = torch.arange(n, device=DEVICE)
        got = summary_cache[summary_bt[req, r // 4].long(), r % 4].float()
        # Within one E4M3 step: exp and reduction order may differ by an ulp.
        torch.testing.assert_close(
            got, ref_compress(ks[req], zs[req]), rtol=0.125, atol=0
        )


def test_indexer_scores_paged():
    torch.manual_seed(0)
    groups, regions = 4, 25
    summaries = [
        _e4m3(torch.randn(regions, PROXY, device=DEVICE)).to(torch.bfloat16)
        for _ in range(2)
    ]
    summary_cache, summary_bt = _paged(summaries, 8)
    pos = torch.cat([torch.arange(200), torch.arange(150, 154)]).int().to(DEVICE)
    req = torch.tensor([0] * 200 + [1] * 4, dtype=torch.int32, device=DEVICE)
    q = torch.randn(len(pos), groups, 4, PROXY, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(len(pos), groups, 4, device=DEVICE)
    got = indexer_scores_paged(
        q,
        w,
        summary_cache,
        summary_bt,
        req,
        pos,
        max_regions=regions,
        region_size=REGION,
    )
    history = torch.stack(summaries)[req.long()].float()
    dots = torch.einsum("tghd,trd->tghr", _e4m3(q), history).clamp_min(0)
    want = (dots * w.unsqueeze(-1)).sum(2)
    visible = torch.arange(regions, device=DEVICE) < (pos // REGION)[:, None]
    visible = visible[:, None, :].expand_as(got)
    torch.testing.assert_close(got[visible], want[visible], atol=1e-3, rtol=1e-4)
    assert torch.isneginf(got[~visible]).all()


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


@pytest.mark.parametrize("topk", [6, 512])
def test_sparse_attention_paged(topk: int):
    """topk=512 covers the whole history, so it also checks dense equivalence."""
    torch.manual_seed(0)
    seq_len, heads, groups, dim = 203, 16, 4, 192
    shape = (2, seq_len, groups, dim)
    k = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    pos = torch.cat([torch.arange(seq_len), torch.tensor([202])]).int().to(DEVICE)
    req = torch.tensor([0] * seq_len + [1], dtype=torch.int32, device=DEVICE)
    q = torch.randn(len(pos), heads, dim, dtype=torch.bfloat16, device=DEVICE)
    scores = torch.randn(len(pos), groups, seq_len // REGION + 1, device=DEVICE)
    regions = ref_select(scores, pos, topk).to(DEVICE)

    kv_cache, bt = _paged([torch.stack([k[r], v[r]], 1) for r in range(2)], 64)
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
    )
    want = ref_attention(q, k, v, regions, req, pos, dim**-0.5)
    torch.testing.assert_close(got.float().cpu(), want, atol=2e-2, rtol=2e-2)


def test_sparse_decode_attention():
    """One decode row per request through the split-K kernel the model uses."""
    torch.manual_seed(0)
    seq_len, heads, groups, dim, topk = 203, 16, 4, 192, 6
    shape = (2, seq_len, groups, dim)
    k = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    pos = torch.tensor([seq_len - 1, seq_len - 9], dtype=torch.int32, device=DEVICE)
    req = torch.arange(2, dtype=torch.int32, device=DEVICE)
    q = torch.randn(2, heads, dim, dtype=torch.bfloat16, device=DEVICE)
    scores = torch.randn(2, groups, seq_len // REGION + 1, device=DEVICE)
    regions = ref_select(scores, pos, topk).to(DEVICE)

    kv_cache, bt = _paged([torch.cat([k[r], v[r]], -1) for r in range(2)], 16)
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
    )
    want = ref_attention(q, k, v, regions, req, pos, dim**-0.5)
    torch.testing.assert_close(got.float().cpu(), want, atol=2e-2, rtol=2e-2)
