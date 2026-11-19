# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx950 gluon paged MQA logits kernel against AITER's, on the flattened
DSA decode / MTP-verify rows: same logits to fp32 rounding (the head sum is
reordered) and the same top-2048 selection."""

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

HEADS, DIM, PAGE, TOPK = 32, 128, 64, 2048
NUM_PAGES = 8192
FP8 = torch.float8_e4m3fn
DEVICE = "cuda"


@pytest.fixture(scope="module")
def kv_cache():
    """Index-K pool written by the production writer (preshuffled, ue8m0)."""
    g = torch.Generator(device=DEVICE).manual_seed(0)
    k = torch.randn(NUM_PAGES * PAGE, DIM, device=DEVICE, generator=g).to(
        torch.bfloat16
    )
    pool = torch.zeros(NUM_PAGES, PAGE * (DIM + 4), dtype=torch.uint8, device=DEVICE)
    aiter_cache.indexer_k_quant_and_cache(
        k,
        pool.view(-1, PAGE, DIM + 4).view(FP8),
        torch.arange(NUM_PAGES * PAGE, device=DEVICE),
        DIM,
        "ue8m0",
        preshuffle=True,
    )
    return pool.view(NUM_PAGES, PAGE, 1, DIM + 4)


def _case(lens, rows_per_request, share, width, pad_rows=0):
    """Rows grouped `rows_per_request` per request with lengths L+1..L+r, as
    the flattened decode path lays them out; with `share` the rows of a
    request use one page table (MTP verify). `pad_rows` zero-length rows
    follow, like CUDA-graph padding."""
    g = torch.Generator(device="cpu").manual_seed(sum(lens) + width)
    rows = len(lens) * rows_per_request + pad_rows
    pt = torch.zeros(rows, width, dtype=torch.int32)
    seq = torch.zeros(rows, dtype=torch.int32)
    for i, length in enumerate(lens):
        npg = -(-(length + rows_per_request) // PAGE)
        base = torch.randperm(NUM_PAGES, generator=g)[:npg].int()
        for j in range(rows_per_request):
            r = i * rows_per_request + j
            pt[r, :npg] = (
                base if share else torch.randperm(NUM_PAGES, generator=g)[:npg].int()
            )
            seq[r] = length + j + 1
    q = torch.randn(rows, 1, HEADS, DIM, generator=g).clamp_(-4, 4).to(FP8)
    w = torch.randn(rows, HEADS, generator=g) * HEADS**-0.5
    return q.to(DEVICE), w.to(DEVICE), seq.to(DEVICE), pt.to(DEVICE)


def _aiter_logits(kv_cache, q, w, seq, pt):
    out = torch.full((q.shape[0], pt.shape[1] * PAGE), float("-inf"), device=DEVICE)
    aiter_pa_mqa_logits.deepgemm_fp8_paged_mqa_logits(
        q,
        kv_cache,
        w,
        out,
        seq,
        pt,
        out.shape[1],
        Preshuffle=True,
        KVBlockSize=PAGE,
        ChunkK=256,
        WavePerEU=2,
    )
    torch.accelerator.synchronize()
    return out


def _gluon_logits(kv_cache, q, w, seq, pt, rows_per_request):
    out = torch.full((q.shape[0], pt.shape[1] * PAGE), float("-inf"), device=DEVICE)
    gluon.gluon_fp8_paged_mqa_logits(
        q, kv_cache, w, out, seq, pt, out.shape[1], rows_per_request
    )
    torch.accelerator.synchronize()
    return out


def _assert_close_with_same_topk(got, ref, seq):
    for r in range(seq.shape[0]):
        n = int(seq[r])
        torch.testing.assert_close(
            got[r, :n], ref[r, :n], rtol=1e-5, atol=1e-5, msg=f"row {r}"
        )
        if n > TOPK:
            # Logits differ only by rounding, so the top-k selections may only
            # differ among elements within that rounding of the k-th score.
            eps = 1e-4
            kth = torch.topk(ref[r, :n], TOPK).values[-1]
            sel = torch.zeros(n, dtype=torch.bool, device=DEVICE)
            sel[torch.topk(got[r, :n], TOPK).indices] = True
            assert bool(sel[ref[r, :n] > kth + eps].all()), (
                f"row {r}: a clear top-k member was not selected"
            )
            assert bool((ref[r, :n][sel] >= kth - eps).all()), (
                f"row {r}: selected an element clearly below the k-th"
            )


def _assert_matches_aiter(
    kv_cache, lens, rows_per_request=6, share=True, width=4096, pad_rows=0
):
    q, w, seq, pt = _case(lens, rows_per_request, share, width, pad_rows)
    assert gluon.supported(
        q, kv_cache, seq, pt, preshuffle=True, rows_per_request=rows_per_request
    )
    ref = _aiter_logits(kv_cache, q, w, seq, pt)
    got = _gluon_logits(kv_cache, q, w, seq, pt, rows_per_request)
    _assert_close_with_same_topk(got, ref, seq)


@pytest.mark.parametrize(
    "lens",
    [
        [100_000] * 8,
        [3000, 50_000, 120_000, 200_000, 9000, 60_000, 130_000, 70_000],
        [250_000] * 14,
        [40_000] * 21,
        [500, 1500, 2500, 4000],
        [30_000 + 2999 * i for i in range(36)],
        [100_000] * 42,
    ],
)
def test_verify_rows_shared_tables(kv_cache, lens):
    _assert_matches_aiter(kv_cache, lens)


def test_rows_without_shared_tables(kv_cache):
    """Paired rows whose page tables differ: the kernel must score them
    separately."""
    _assert_matches_aiter(kv_cache, [60_000] * 8, share=False)


@pytest.mark.parametrize(
    "rows_per_request,requests", [(2, 32), (2, 64), (4, 12), (4, 32), (4, 64)]
)
def test_other_verify_group_sizes(kv_cache, rows_per_request, requests):
    """MTP with fewer draft tokens."""
    _assert_matches_aiter(
        kv_cache,
        [20_000 + 1999 * i for i in range(requests)],
        rows_per_request,
    )


@pytest.mark.parametrize("pad_rows", [6, 18])
def test_padded_rows(kv_cache, pad_rows):
    """Zero-length rows, like CUDA-graph padding, are skipped."""
    _assert_matches_aiter(kv_cache, [50_000, 80_000] * 4, pad_rows=pad_rows)


def test_supported_bounds(kv_cache):
    q, _, seq, pt = _case([1000] * 66, 6, True, 64)

    def supported(rows, preshuffle=True, rows_per_request=6, width=64):
        return gluon.supported(
            q[:rows],
            kv_cache,
            seq[:rows],
            pt[:rows, :width],
            preshuffle=preshuffle,
            rows_per_request=rows_per_request,
        )

    assert supported(12)
    assert supported(126)
    assert supported(216)
    assert supported(252)
    assert not supported(258)
    assert supported(128, rows_per_request=2)
    assert not supported(130, rows_per_request=2)
    assert not supported(12, preshuffle=False)
    assert not supported(13)
    assert not supported(12, rows_per_request=0)
    assert not supported(12, rows_per_request=1)
    assert not supported(12, rows_per_request=3)
    assert not supported(12, width=32)
    # Native MTP decode: several queries per request share one row.
    q2 = q[:12].view(6, 2, HEADS, DIM)
    assert not gluon.supported(
        q2, kv_cache, seq[:6], pt[:6], preshuffle=True, rows_per_request=2
    )


@pytest.mark.parametrize("rows_per_request", [6, 0])
def test_dispatch(kv_cache, monkeypatch, rows_per_request):
    """`rocm_fp8_paged_mqa_logits` takes the gluon kernel for flattened MTP
    verify rows and AITER's otherwise."""
    from vllm._aiter_ops import rocm_aiter_ops

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", classmethod(lambda cls: True))
    calls = []
    real = gluon.gluon_fp8_paged_mqa_logits

    def spy(*args, **kwargs):
        calls.append(args[7])
        return real(*args, **kwargs)

    monkeypatch.setattr(gluon, "gluon_fp8_paged_mqa_logits", spy)
    q, w, seq, pt = _case([70_000, 90_000] * 3, 6, True, 2048)
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
    ref = _aiter_logits(kv_cache, q, w, seq, pt)
    _assert_close_with_same_topk(got, ref, seq)
