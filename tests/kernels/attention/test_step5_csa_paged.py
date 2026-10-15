# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged Step-5 CSA kernels vs PyTorch references on scrambled block tables.

Selection is discrete, so rounding and tie-breaks must match exactly, and a
page-addressing bug reads wrong tokens without changing any shape.
"""

import pytest
import torch

from vllm.models.step5.common.ops.csa import select_regions
from vllm.models.step5.common.ops.csa_paged import (
    compress_regions_paged,
    indexer_scores_paged,
    sparse_attention_paged,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Triton kernels need a GPU"
)

DEVICE = "cuda"
REGION = 8
PROXY = 256


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float8_e4m3fn).float()


def _paged(rows: torch.Tensor, rows_per_page: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter rows into ``[pages, rows_per_page, ...]`` behind a shuffled table."""
    pages = -(-rows.shape[0] // rows_per_page)
    table = torch.randperm(pages + 3, device=DEVICE)[:pages].int()
    cache = torch.zeros(
        (pages + 3, rows_per_page, *rows.shape[1:]), dtype=rows.dtype, device=DEVICE
    )
    idx = torch.arange(rows.shape[0], device=DEVICE)
    cache[table[idx // rows_per_page].long(), idx % rows_per_page] = rows
    return cache, table.view(1, -1)


def ref_compress(k: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    regions = k.shape[0] // REGION
    k = k[: regions * REGION].float().view(regions, REGION, -1)
    z = z[: regions * REGION].float().view(regions, REGION, -1)
    w = torch.exp(z - z.amax(dim=(1, 2), keepdim=True))
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


def ref_attention(q, k, v, regions, positions, scale):
    tokens, heads, _ = q.shape
    groups = k.shape[1]
    per_group = heads // groups
    out = torch.zeros(q.shape, dtype=torch.float32)
    for t in range(tokens):
        for g in range(groups):
            ids = regions[t, g]
            ids = ids[ids >= 0].long().cpu()
            cols = (ids[:, None] * REGION + torch.arange(REGION)).flatten()
            cols = cols[cols <= int(positions[t])]
            hs = slice(g * per_group, (g + 1) * per_group)
            p = torch.softmax(
                q[t, hs].float().cpu() @ k[cols, g].float().cpu().T * scale, -1
            )
            out[t, hs] = p @ v[cols, g].float().cpu()
    return out


@pytest.mark.parametrize("seq_len", [64, 200])
def test_compress_regions_paged(seq_len: int):
    torch.manual_seed(0)
    k = torch.randn(seq_len, PROXY, dtype=torch.bfloat16, device=DEVICE)
    z = torch.randn(seq_len, PROXY, dtype=torch.bfloat16, device=DEVICE) * 4
    index_cache, index_bt = _paged(torch.cat([k, z], -1), 64)
    regions = seq_len // REGION
    summary_cache, summary_bt = _paged(
        torch.zeros(regions, PROXY, dtype=torch.bfloat16, device=DEVICE), 4
    )
    # Trailing -1 rows are padding and must be skipped.
    work = torch.full((regions + 3, 2), -1, dtype=torch.int32, device=DEVICE)
    work[:regions, 0] = 0
    work[:regions, 1] = torch.arange(regions, device=DEVICE)
    compress_regions_paged(
        index_cache, summary_cache, index_bt, summary_bt, work, region_size=REGION
    )
    r = torch.arange(regions, device=DEVICE)
    got = summary_cache[summary_bt[0, r // 4].long(), r % 4]
    torch.testing.assert_close(got.float(), ref_compress(k, z), atol=0, rtol=0)


def test_indexer_scores_paged():
    torch.manual_seed(0)
    seq_len, groups = 200, 4
    regions = seq_len // REGION
    summaries = _e4m3(torch.randn(regions, PROXY, device=DEVICE)).to(torch.bfloat16)
    summary_cache, summary_bt = _paged(summaries, 8)
    q = torch.randn(seq_len, groups, 4, PROXY, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(seq_len, groups, 4, device=DEVICE)
    pos = torch.arange(seq_len, dtype=torch.int32, device=DEVICE)
    req = torch.zeros(seq_len, dtype=torch.int32, device=DEVICE)
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
    dots = torch.einsum("tghd,rd->tghr", _e4m3(q), summaries.float()).clamp_min(0)
    want = (dots * w.unsqueeze(-1)).sum(2)
    visible = torch.arange(regions, device=DEVICE)[None, None, :] < (
        pos // REGION
    ).view(-1, 1, 1)
    visible = visible.expand_as(got)
    torch.testing.assert_close(got[visible], want[visible], atol=1e-3, rtol=1e-4)
    assert torch.isneginf(got[~visible]).all()


def test_select_regions_breaks_ties_by_region_id():
    """Post-ReLU scores are often exactly zero; ties must pick lower ids."""
    torch.manual_seed(0)
    scores = torch.randn(48, 2, 40, device=DEVICE)
    scores[scores.abs() < 0.5] = 0.0
    positions = torch.randint(0, 40 * REGION, (48,), device=DEVICE)
    got = select_regions(scores, positions, REGION, topk=6)
    assert torch.equal(got.cpu(), ref_select(scores, positions, 6))


@pytest.mark.parametrize("topk", [6, 512])
def test_sparse_attention_paged(topk: int):
    """topk=512 covers the whole history, so it also checks dense equivalence."""
    torch.manual_seed(0)
    seq_len, heads, groups, dim = 200, 16, 4, 192
    q = torch.randn(seq_len, heads, dim, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(seq_len, groups, dim, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(seq_len, groups, dim, dtype=torch.bfloat16, device=DEVICE)
    pos = torch.arange(seq_len, dtype=torch.int32, device=DEVICE)
    scores = torch.randn(seq_len, groups, seq_len // REGION + 1, device=DEVICE)
    regions = select_regions(scores, pos, REGION, topk)

    kv_cache, bt = _paged(torch.stack([k, v], 1), 64)
    got = torch.empty_like(q)
    req = torch.zeros(seq_len, dtype=torch.int32, device=DEVICE)
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
    want = ref_attention(q, k, v, regions, pos, dim**-0.5)
    torch.testing.assert_close(got.float().cpu(), want, atol=2e-2, rtol=2e-2)
