# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx950 gluon paged MQA logits kernel against AITER's on flattened
MTP-verify rows."""

import functools

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only", allow_module_level=True)

from vllm.platforms.rocm import get_cdna_version

if get_cdna_version() != 4:
    pytest.skip("gfx950-only", allow_module_level=True)

aiter_cache = pytest.importorskip("aiter.ops.cache")
aiter_pa_mqa_logits = pytest.importorskip("aiter.ops.triton.pa_mqa_logits")

from vllm.v1.attention.ops import rocm_aiter_mla_sparse  # noqa: E402
from vllm.v1.attention.ops import rocm_gluon_paged_mqa_logits as gluon  # noqa: E402
from vllm.v1.worker.workspace import (  # noqa: E402
    init_workspace_manager,
    reset_workspace_manager,
)

HEADS, DIM, PAGE, POOL_TOKENS = 32, 128, 64, 1 << 19
DEVICE = "cuda"


@functools.cache
def make_pool(page):
    """Preshuffled ue8m0 index-K pool, written by AITER's cache writer."""
    torch.manual_seed(0)
    k = torch.randn(POOL_TOKENS, DIM, device=DEVICE, dtype=torch.bfloat16)
    pool = torch.zeros(
        POOL_TOKENS // page, page, 1, DIM + 4, dtype=torch.uint8, device=DEVICE
    )
    aiter_cache.indexer_k_quant_and_cache(
        k,
        pool.view(-1, page, DIM + 4).view(torch.float8_e4m3fn),
        torch.arange(POOL_TOKENS, device=DEVICE),
        DIM,
        "ue8m0",
        preshuffle=True,
    )
    return pool


@pytest.fixture(scope="module")
def kv_cache():
    return make_pool(PAGE)


def make_rows(lens, rows_per_request=6, share=True, pad_rows=0, page=PAGE, slack=0):
    """`rows_per_request` rows per request with lengths L+1..L+r; with `share`
    a request's rows use one page table. Zero-length `pad_rows` follow. With
    `slack`, the page table is a column slice of a wider one."""
    torch.manual_seed(len(lens))
    num_pages = POOL_TOKENS // page
    rows = len(lens) * rows_per_request + pad_rows
    width = -(-(max(lens) + rows_per_request) // page)
    pt = torch.zeros(rows, width + slack, dtype=torch.int32)
    seq = torch.zeros(rows, dtype=torch.int32)
    for i, length in enumerate(lens):
        pages = torch.randperm(num_pages)[: width + slack].int()
        for j in range(rows_per_request):
            r = i * rows_per_request + j
            pt[r] = pages if share else torch.randperm(num_pages)[: width + slack].int()
            seq[r] = length + j + 1
    q = torch.randn(rows, 1, HEADS, DIM).clamp(-4, 4).to(torch.float8_e4m3fn)
    w = torch.randn(rows, HEADS) * HEADS**-0.5
    return q.to(DEVICE), w.to(DEVICE), seq.to(DEVICE), pt.to(DEVICE)[:, :width]


def aiter_logits(kv_cache, q, w, seq, pt):
    page = kv_cache.shape[1]
    pt = pt.contiguous()
    out = torch.full((q.shape[0], pt.shape[1] * page), float("-inf"), device=DEVICE)
    aiter_pa_mqa_logits.deepgemm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        w,
        out,
        seq,
        pt,
        out.shape[1],
        Preshuffle=True,
        KVBlockSize=page,
        ChunkK=256,
        WavePerEU=2,
    )
    return out


def assert_logits_close(got, ref, seq):
    for r, n in enumerate(seq.tolist()):
        torch.testing.assert_close(got[r, :n], ref[r, :n], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "lens,rows_per_request,share,pad_rows,page,slack",
    [
        ([100_000] * 8, 6, True, 0, 64, 0),
        ([3000, 50_000, 200_000, 9000, 130_000, 70_000], 6, True, 0, 64, 0),
        ([500, 1500, 2500, 4000], 6, True, 0, 64, 0),
        ([30_000 + 2999 * i for i in range(36)], 6, True, 0, 64, 0),  # > 32 groups
        ([100_000] * 42, 6, True, 0, 64, 0),
        ([2000 + 97 * i for i in range(256)], 6, True, 0, 64, 0),  # 256 groups
        ([20_000 + 1999 * i for i in range(64)], 2, True, 0, 64, 0),
        ([20_000 + 1999 * i for i in range(32)], 4, True, 0, 64, 0),
        ([40_000 + 999 * i for i in range(16)], 3, True, 0, 64, 0),  # odd group
        ([40_000 + 999 * i for i in range(16)], 5, False, 0, 64, 0),
        ([3000, 90_000, 20_000, 60_000] * 2, 6, True, 0, 16, 0),
        ([3000, 90_000, 20_000, 60_000] * 2, 3, True, 6, 32, 0),
        ([60_000] * 8, 6, True, 0, 64, 5),  # non-contiguous page table
        ([60_000] * 8, 6, False, 0, 64, 0),  # paired rows with different tables
        ([50_000, 80_000] * 4, 6, True, 18, 64, 0),  # CUDA-graph padding
    ],
)
def test_matches_aiter(lens, rows_per_request, share, pad_rows, page, slack):
    kv_cache = make_pool(page)
    q, w, seq, pt = make_rows(lens, rows_per_request, share, pad_rows, page, slack)
    assert gluon.supported(
        q, kv_cache, seq, pt, preshuffle=True, rows_per_request=rows_per_request
    )
    got = torch.full((q.shape[0], pt.shape[1] * page), float("-inf"), device=DEVICE)
    gluon.gluon_fp8_paged_mqa_logits(
        q, kv_cache, w, got, seq, pt, got.shape[1], rows_per_request
    )
    assert_logits_close(got, aiter_logits(kv_cache, q, w, seq, pt), seq)


def test_supported(kv_cache):
    q, _, seq, pt = make_rows([1000] * 257)

    def supported(rows, rows_per_request=6, preshuffle=True, cache=kv_cache):
        return gluon.supported(
            q[:rows],
            cache,
            seq[:rows],
            pt[:rows],
            preshuffle=preshuffle,
            rows_per_request=rows_per_request,
        )

    assert supported(12) and supported(1536) and supported(12, rows_per_request=3)
    assert supported(512, rows_per_request=2)
    assert not supported(514, rows_per_request=2)  # > 256 requests
    assert not supported(1540, rows_per_request=7)  # > MAX_ROWS
    assert not supported(13)  # not a whole number of requests
    assert not supported(12, rows_per_request=1)  # plain decode
    assert not supported(12, rows_per_request=0)  # non-uniform decode
    assert not supported(12, preshuffle=False)
    assert supported(12, cache=kv_cache.view(-1, 16, 1, DIM + 4))
    assert not supported(12, cache=kv_cache.view(-1, 128, 1, DIM + 4))


@pytest.mark.parametrize("rows_per_request", [6, 0])
def test_dispatch(kv_cache, monkeypatch, rows_per_request):
    """`rocm_fp8_paged_mqa_logits` uses gluon only for uniform verify rows."""
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", classmethod(lambda cls: True))
    calls = []
    real = gluon.gluon_fp8_paged_mqa_logits

    def spy(*args):
        calls.append(args[-1])
        return real(*args)

    monkeypatch.setattr(gluon, "gluon_fp8_paged_mqa_logits", spy)
    q, w, seq, pt = make_rows([70_000, 90_000] * 3)
    init_workspace_manager(torch.device(DEVICE))
    try:
        got = rocm_aiter_mla_sparse.rocm_fp8_paged_mqa_logits(
            q,
            kv_cache,
            w,
            seq,
            pt,
            None,
            max_model_len=pt.shape[1] * PAGE,
            rows_per_request=rows_per_request,
        ).clone()
    finally:
        reset_workspace_manager()
    assert calls == ([6] if rows_per_request else [])
    assert_logits_close(got, aiter_logits(kv_cache, q, w, seq, pt), seq)
