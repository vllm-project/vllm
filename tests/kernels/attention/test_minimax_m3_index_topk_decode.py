# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 index top-k (FlyDSL) against AITER's ``pa_sparse_block_topk``
block ids, bit for bit, on realistic and adversarial decode and prefill rows."""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm (gfx950) FlyDSL kernels", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 instructions", allow_module_level=True)

from vllm.models.minimax_m3.amd.ops.index_topk_decode.host import (  # noqa: E402
    minimax_m3_index_topk_decode_indices,
    minimax_m3_index_topk_prefill_indices,
    supports_index_topk,
)

BLOCK = 128


def _width(max_seq_len: int) -> int:
    n = math.ceil(math.ceil(max_seq_len / BLOCK) / 64)
    return (1 << (n - 1).bit_length()) * 64


def _scores(kind, heads, rows, width, nblk, gen):
    s = torch.randn(heads, rows, width, generator=gen)
    if kind == "real":  # a few outliers, the init block 1e30, the last two 1e29
        s = torch.where(torch.rand(s.shape, generator=gen) < 4.0 / width, s + 6, s)
        for r in range(rows):
            b = min(int(nblk[r]), width)
            if b > 0:
                s[:, r, max(b - 2, 0) : b] = 1e29
                s[:, r, 0] = 1e30
    elif kind == "ties":
        s = torch.randint(-3, 4, s.shape, generator=gen).float()
    elif kind == "equal":
        s = torch.full_like(s, 0.5)
    elif kind == "neginf":
        s = torch.full_like(s, float("-inf"))
    elif kind == "nan":
        s[torch.rand(s.shape, generator=gen) < 0.3] = float("nan")
        neg_nan = torch.tensor([-4194303], dtype=torch.int32).view(torch.float32)
        s[torch.rand(s.shape, generator=gen) < 0.1] = neg_nan.item()
    elif kind == "zeros":  # +0 / -0 / denormals
        vals = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 1e-45, -1e-45])
        s = vals[torch.randint(0, 6, s.shape, generator=gen)]
    elif kind == "sentinels":
        s = torch.where(torch.rand(s.shape, generator=gen) < 0.2, 1e29, s)
        s = torch.where(torch.rand(s.shape, generator=gen) < 0.05, 1e30, s)
    elif kind == "inf":
        r = torch.rand(s.shape, generator=gen)
        s = torch.where(r < 0.1, float("inf"), s)
        s = torch.where((r > 0.1) & (r < 0.2), float("-inf"), s)
    elif kind == "rising":  # increasing with position
        s = torch.arange(width, dtype=torch.float32).expand(heads, rows, width).clone()
    # blocks past a row's length hold garbage the kernel must not read as scores
    tail = torch.randn(s.shape, generator=gen) * 1e6
    tail[torch.rand(s.shape, generator=gen) < 0.1] = float("nan")
    live = torch.arange(width)[None, None, :] < nblk[None, :, None]
    return torch.where(live, s, tail)


def _topk_out(heads, rows, pad):
    """``topk_idx`` and the tensor compared: with ``pad`` a view (non-zero row
    start, larger head stride) of a larger persistent buffer, compared whole
    (nothing outside the view is written)."""
    if not pad:
        t = torch.full((heads, rows, 16), -7, dtype=torch.int32, device="cuda")
        return t, t
    full = torch.full((heads, rows + 37, 16), -7, dtype=torch.int32, device="cuda")
    return full[:, 5 : 5 + rows], full


def _aiter_topk(score, bt, seq, msl, nkv, pad, **kw):
    from aiter.ops.msa_attention import pa_sparse_block_topk

    heads, rows, _ = score.shape
    topk, compared = _topk_out(heads, rows, pad)
    sbt = torch.empty(rows * nkv, 128, dtype=torch.int32, device="cuda")
    sctx = torch.empty(rows * nkv, dtype=torch.int32, device="cuda")
    pa_sparse_block_topk(
        score, topk, bt, seq, sbt, sctx, max_seq_len=msl, block_size=BLOCK,
        num_kv_heads=nkv, pages_per_block=8, **kw,
    )  # fmt: skip
    return compared


SHAPES = [
    # reqs, query_len, max_seq_len, (lo, hi) seq_len, num_kv_heads, index heads
    (40, 4, 1 << 20, (30000, 210000), 1, 1),  # the QPS decode batch
    (8, 4, 262144, (1, 262144), 1, 1),
    (5, 4, 1 << 20, (1 << 19, 1 << 20), 1, 1),  # 4 windows of 2048 blocks
    (16, 1, 1 << 20, (2040 * BLOCK, 2100 * BLOCK), 1, 1),  # the window edge
    (12, 1, 4000, (1, 4000), 1, 1),
    (4, 2, 300000, (1, 300000), 2, 2),
    (7, 4, 20000, (20000, 40000), 1, 1),  # rows past max_seq_len
    (256, 4, 1 << 20, (1, 3000), 1, 1),  # 1024 short rows
    (32, 6, 204800, (190000, 204800), 4, 4),  # 6 rows, 4 heads
    # rows past max_seq_len inside the padded width (2048 blocks): the window
    # sized by max_seq_len must go on to the next one, not cut them
    (8, 1, 200000, (200000, 262000), 4, 1),
    (4, 2, 0, (0, 0), 1, 1),  # a zero max_seq_len (dummy batch)
    (1, 1, 1, (0, 1), 1, 1),
]
KINDS = [
    "real",
    "ties",
    "equal",
    "neginf",
    "nan",
    "zeros",
    "sentinels",
    "inf",
    "rising",
]


def _check_decode(kind, shape, pad=False):
    reqs, ql, msl, (lo, hi), nkv, heads = shape
    gen = torch.Generator().manual_seed(100 * KINDS.index(kind) + SHAPES.index(shape))
    seq = torch.randint(lo, hi + 1, (reqs,), generator=gen)
    if reqs > 1:
        seq[-1] = 0  # a cudagraph padding request
    tok = torch.arange(reqs * ql) % ql
    kv = (seq.repeat_interleave(ql) - ql + tok + 1).clamp(min=0)
    score = _scores(kind, heads, reqs * ql, _width(msl), (kv + BLOCK - 1) // BLOCK, gen)
    # the table covers every row's blocks (rows may run past max_seq_len)
    cols = max(1, math.ceil(max(msl, int(seq.max())) / BLOCK))
    bt = torch.randint(0, 1 << 22, (reqs, cols), generator=gen).int().cuda()
    score, seq = score.cuda(), seq.int().cuda()
    ref = _aiter_topk(score, bt, seq, msl, nkv, pad, query_len=ql)
    topk, got = _topk_out(heads, reqs * ql, pad)
    minimax_m3_index_topk_decode_indices(
        score, topk, seq, max_seq_len=msl, query_len=ql
    )
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_index_topk_decode_matches_aiter(kind, shape):
    _check_decode(kind, shape)


@pytest.mark.parametrize("kind", ["real", "nan", "ties"])
@pytest.mark.parametrize("shape", [SHAPES[0], SHAPES[5]])
def test_index_topk_decode_padded_output(kind, shape):
    _check_decode(kind, shape, pad=True)


def test_index_topk_decode_all_padding():
    """A captured dummy batch: every request padding (seq_len 0)."""
    reqs, ql, msl = 48, 4, 1 << 20
    score = torch.randn(1, reqs * ql, _width(msl), device="cuda")
    bt = torch.zeros(reqs, msl // BLOCK, dtype=torch.int32, device="cuda")
    seq = torch.zeros(reqs, dtype=torch.int32, device="cuda")
    ref = _aiter_topk(score, bt, seq, msl, 1, False, query_len=ql)
    topk, _ = _topk_out(1, reqs * ql, False)
    minimax_m3_index_topk_decode_indices(
        score, topk, seq, max_seq_len=msl, query_len=ql
    )
    torch.testing.assert_close(topk, ref, rtol=0, atol=0)


PREFILL_SHAPES = [
    # (q_len, seq_len) per request, max_seq_len, num_kv_heads, index heads
    ([(1000, 1000)], 1000, 1, 1),  # a fresh prompt: every row diagonal
    ([(700, 30000), (1500, 1500)], 30000, 4, 4),  # cached prefix + fresh
    ([(2000, 9000), (513, 513), (3000, 20000)], 20000, 4, 1),  # ragged
    ([(300, 262144), (64, 200000)], 262144, 4, 4),  # one window, 2048 blocks
    ([(129, 2049 * BLOCK), (40, 1 << 20)], 1 << 20, 2, 2),  # multi-window rows
    ([(1, 1), (5, 5), (128, 128), (129, 129)], 4096, 1, 1),  # tiny rows
    # 3K new on a 200K prefix beside a fresh 3K prompt
    ([(3072, 203072), (3072, 3072), (3072, 207872)], 207872, 4, 1),
    ([(64, 262000), (3072, 203072)], 200000, 4, 1),  # rows past max_seq_len
]


def _check_prefill(kind, shape, pad=False):
    reqs, msl, nkv, heads = shape
    gen = torch.Generator().manual_seed(
        100 * KINDS.index(kind) + PREFILL_SHAPES.index(shape)
    )
    # every request's chunk ends its sequence: row positions seq - q .. seq - 1
    pos = torch.cat([torch.arange(s - q, s) for q, s in reqs])
    rid = torch.cat([torch.full((q,), r) for r, (q, _) in enumerate(reqs)])
    nvp = pos // BLOCK + 1
    seq = torch.tensor([s for _, s in reqs])
    score = _scores(kind, heads, pos.numel(), _width(msl), nvp, gen).cuda()
    cols = max(1, math.ceil(max(msl, int(seq.max())) / BLOCK))
    bt = torch.randint(0, 1 << 22, (len(reqs), cols), generator=gen).int().cuda()
    nvp, rid, kvl = (x.int().cuda() for x in (nvp, rid, pos + 1))
    ref = _aiter_topk(
        score, bt, seq.int().cuda(), msl, nkv, pad,
        num_valid_pages=nvp, row_req_id=rid, kv_lens=kvl,
    )  # fmt: skip
    topk, got = _topk_out(heads, pos.numel(), pad)
    minimax_m3_index_topk_prefill_indices(
        score, topk, max_seq_len=msl, num_valid_pages=nvp
    )
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", PREFILL_SHAPES)
def test_index_topk_prefill_matches_aiter(kind, shape):
    _check_prefill(kind, shape)


@pytest.mark.parametrize("kind", ["real", "nan", "ties"])
@pytest.mark.parametrize("shape", [PREFILL_SHAPES[2], PREFILL_SHAPES[4]])
def test_index_topk_prefill_padded_output(kind, shape):
    _check_prefill(kind, shape, pad=True)


def test_supports_index_topk():
    assert supports_index_topk(16, BLOCK)
    assert not supports_index_topk(8, BLOCK) and not supports_index_topk(16, 64)
