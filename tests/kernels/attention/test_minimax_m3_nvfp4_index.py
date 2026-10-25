# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 NVFP4 index scorers (FlyDSL) against AITER's FP8 scorers and an
fp32 reference over the same e4m3 keys; the context-parallel push scorer
against the single-head scorer, bit for bit, on one GPU."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm (gfx950) FlyDSL kernels", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 instructions", allow_module_level=True)

from tests.kernels.attention.minimax_m3_nvfp4_ref import dequant  # noqa: E402
from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode import (  # noqa: E402
    host as decode_host,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode import (  # noqa: E402
    kernel_cp_push as cp,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill import (  # noqa: E402
    host as prefill_host,
)

minimax_m3_nvfp4_index_score_decode = decode_host.minimax_m3_nvfp4_index_score_decode
minimax_m3_nvfp4_index_score_prefill = prefill_host.minimax_m3_nvfp4_index_score_prefill

BLOCK = 128
D = 128
SPECIAL_LENS = [0, 1, 2, 3, 127, 128, 129, 1024]  # padding, short and edge rows


def _index_cache(num_blocks: int, page_step: int = 1) -> torch.Tensor:
    """Random NVFP4 index pages ``page_step`` pages apart (a view of a larger
    cache)."""
    n = num_blocks * page_step
    cache = torch.randint(0, 256, (n, BLOCK * 72), device="cuda").to(torch.uint8)
    s = torch.rand(n, BLOCK * 8, device="cuda") * 1.75 + 0.25
    cache[:, BLOCK * 64 :] = s.to(torch.float8_e4m3fn).view(torch.uint8)
    return cache.view(n, BLOCK, 72)[::page_step]


def _fp8_keys(cache: torch.Tensor) -> torch.Tensor:
    """Every page's keys as e4m3_rn_satfinite(e2m1 x scale), [nb, 128, 128]."""
    flat = cache.reshape(cache.shape[0], -1)
    keys = dequant(
        flat[:, : BLOCK * 64].view(-1, BLOCK, 64),
        flat[:, BLOCK * 64 :].view(-1, BLOCK, 8),
    )
    return keys.clamp(-448, 448).to(torch.float8_e4m3fn)


def _block_table(seq_lens, share=False):
    """Shuffled pages per request; ``share``: some requests start with another
    request's pages (a shared prefix)."""
    widths = [(s + BLOCK - 1) // BLOCK for s in seq_lens]
    num_blocks = sum(widths) + 2
    perm = torch.randperm(num_blocks, device="cuda", dtype=torch.int32)
    table = torch.zeros(len(seq_lens), max(widths), dtype=torch.int32, device="cuda")
    start = 0
    for r, w in enumerate(widths):
        table[r, :w] = perm[start : start + w]
        start += w
    for r in range(len(seq_lens) if share else 0):
        src = 0 if r % 2 == 0 else 3 * (r // 6)
        n = min(widths[r], widths[src], 10 if r % 2 == 0 else 6)
        if r != src:
            table[r, :n] = table[src, :n]
    return table, num_blocks


def _lens(num_reqs: int, hi: int) -> list[int]:
    lens = torch.randint(1, hi, (num_reqs,)).tolist()
    for i, v in enumerate(SPECIAL_LENS[: num_reqs // 2]):
        lens[2 * i + 1] = v
    return lens


def _ref_block_scores(q_rows, keys, kv_lens, init_blocks, local_blocks):
    """[H, R, nb] per-block max of q . k over each row's visible tokens, -inf
    where none is visible, AITER's init (1e30) / local (1e29) sentinels."""
    nb = keys.shape[0] // BLOCK
    dots = torch.einsum("rhd,td->hrt", q_rows.float(), keys)
    tok = torch.arange(nb * BLOCK, device=keys.device)
    dots = dots.masked_fill(tok[None, None, :] >= kv_lens[None, :, None], -torch.inf)
    mx = dots.view(dots.shape[0], dots.shape[1], nb, BLOCK).amax(-1)
    rb = (kv_lens.clamp(min=0) + BLOCK - 1) // BLOCK  # visible blocks per row
    blk = torch.arange(nb, device=keys.device)[None, :]
    vis = blk < rb[:, None]
    init = vis & (blk < init_blocks)
    local = vis & (blk >= (rb - local_blocks).clamp(min=0)[:, None])
    mx = torch.where(init[None], torch.full_like(mx, 1e30), mx)
    return torch.where(local[None], torch.full_like(mx, 1e29), mx)


def _assert_scores(got, want, msg=None):
    """Sentinels and -inf identical; finite scores close."""
    exact = ~torch.isfinite(want) | (want >= 1e28)
    assert torch.equal(got[exact], want[exact]), msg
    torch.testing.assert_close(got[~exact], want[~exact], rtol=2e-4, atol=1e-3, msg=msg)


def _check_decode(actual, q, keys, table, lens, query_len, init_blocks, local_blocks):
    """Each request's rows against the reference; nothing past its blocks
    written."""
    for r, s in enumerate(lens):
        nb = (s + BLOCK - 1) // BLOCK
        sl = slice(r * query_len, (r + 1) * query_len)
        assert actual[:, sl, nb:].isnan().all(), (r, s)
        if nb == 0:
            continue
        kv_lens = (s - query_len + 1 + torch.arange(query_len, device="cuda")).clamp(
            min=0
        )
        page_keys = keys[table[r, :nb].long()].reshape(-1, D)
        want = _ref_block_scores(q[sl], page_keys, kv_lens, init_blocks, local_blocks)
        _assert_scores(actual[:, sl, :nb], want, (r, s))


@pytest.mark.parametrize(
    ("num_heads", "query_len"),
    [(1, 1), (1, 4), (1, 16), (2, 4), (4, 1), (4, 4), (4, 5), (4, 8)],
)
@pytest.mark.parametrize("page_step", [1, 2])
def test_index_decode_matches_aiter_fp8(num_heads, query_len, page_step):
    """Every block a row can see against AITER's FP8 decode scorer."""
    from aiter.ops.msa_attention import pa_sparse_block_score_decode

    torch.manual_seed(0)
    lens = [5, 300, 2100, 9000]
    table, num_blocks = _block_table(lens)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cache = _index_cache(num_blocks, page_step)
    rows = len(lens) * query_len
    q = (torch.randn(rows, num_heads, D, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    width = (max(lens) + BLOCK - 1) // BLOCK
    args = (table, seq_lens, 1, 2, query_len, max(lens))
    actual = torch.full((num_heads, rows, width), float("nan"), device="cuda")
    minimax_m3_nvfp4_index_score_decode(q, cache, actual, *args)
    expected = torch.full_like(actual, float("nan"))
    pages = _fp8_keys(cache)
    for h0 in range(0, num_heads, 2):  # AITER builds one- and two-head scorers
        pa_sparse_block_score_decode(
            q[:, h0 : h0 + 2].contiguous(), pages, expected[h0 : h0 + 2], *args
        )
    torch.accelerator.synchronize()
    for r, s in enumerate(lens):
        for t in range(query_len):
            row = r * query_len + t
            seen = (max(s - query_len + t + 1, 0) + BLOCK - 1) // BLOCK
            got, want = actual[:, row, :seen], expected[:, row, :seen]
            finite = want < 1e28
            assert torch.equal(got[~finite], want[~finite])
            torch.testing.assert_close(got[finite], want[finite], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    ("num_heads", "query_len"), [(1, 1), (1, 4), (2, 3), (4, 1), (4, 4), (4, 6), (4, 8)]
)
@pytest.mark.parametrize("num_reqs", [1, 40, 130])
@pytest.mark.parametrize(("init_blocks", "local_blocks"), [(1, 2), (0, 1)])
@pytest.mark.parametrize("buffered", [False, True])
def test_index_decode_matches_reference(
    num_heads, query_len, num_reqs, init_blocks, local_blocks, buffered, monkeypatch
):
    """Request chunks, shared prefixes, padding requests, sequences shorter
    than the query; ``buffered`` forces the LDS-buffered variant."""
    if buffered:
        monkeypatch.setattr(decode_host, "BUFFER_MIN_BYTES", 0)
        monkeypatch.setattr(decode_host, "UNITS_PER_SLOT", 1)
    torch.manual_seed(num_reqs)
    lens = _lens(num_reqs, 6000)
    table, num_blocks = _block_table(lens, share=True)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cache = _index_cache(num_blocks, 2 if num_reqs == 40 else 1)
    rows = num_reqs * query_len
    q = (torch.randn(rows, num_heads, D, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    width = (max(lens) + BLOCK - 1) // BLOCK + 2
    actual = torch.full((num_heads, rows, width), float("nan"), device="cuda")
    minimax_m3_nvfp4_index_score_decode(
        q,
        cache,
        actual,
        table,
        seq_lens,
        init_blocks,
        local_blocks,
        query_len,
        max(lens),
    )
    keys = _fp8_keys(cache).float()
    torch.accelerator.synchronize()
    _check_decode(actual, q, keys, table, lens, query_len, init_blocks, local_blocks)


@pytest.mark.parametrize(("num_heads", "query_len"), [(1, 4), (4, 4)])
def test_index_decode_graph_replays_new_lengths(num_heads, query_len):
    """Captured once at a large max_seq_len, replayed with new lengths and
    queries."""
    torch.manual_seed(5)
    sets = [
        [140000, 6, 70001, 129, 7, 5000, 0, 128],
        [6, 127, 2048, 140000, 40000, 300, 129, 0],
    ]
    table, num_blocks = _block_table([max(x) for x in zip(*sets)])
    cache = _index_cache(num_blocks)
    seq_lens = torch.tensor(sets[0], dtype=torch.int32, device="cuda")
    rows = len(sets[0]) * query_len
    q = torch.zeros(rows, num_heads, D, device="cuda", dtype=torch.float8_e4m3fn)
    max_seq_len = 1 << 18
    out = torch.empty(num_heads, rows, max_seq_len // BLOCK, device="cuda")
    args = (q, cache, out, table, seq_lens, 1, 2, query_len, max_seq_len)
    minimax_m3_nvfp4_index_score_decode(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        minimax_m3_nvfp4_index_score_decode(*args)
    keys = _fp8_keys(cache).float()
    for lens in sets + sets[:1]:
        seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        q.copy_((torch.randn(q.shape, device="cuda") * 0.5).to(q.dtype))
        out.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        _check_decode(out, q, keys, table, lens, query_len, 1, 2)


def _check_cp_push(lens, page_step, init_blocks, local_blocks, grid, query_len=4):
    """The context-parallel scorer on one GPU, the other ranks' index-q rows
    and flags pre-set locally: per rank, the single-head scorer's bits on its
    blocks, -inf past the length inside its last run, nothing elsewhere; the
    four ranks cover every block once."""
    num_reqs = len(lens)
    table, num_blocks = _block_table(lens, share=True)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cache = _index_cache(num_blocks, page_step)
    heads = cp.HEADS
    rows = num_reqs * query_len
    q = (torch.randn(rows, heads, D, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    max_seq_len = max(lens)
    width = max(64, ((max_seq_len + BLOCK - 1) // BLOCK + 63) // 64 * 64)
    expected = torch.full((heads, rows, width), float("nan"), device="cuda")
    for h in range(heads):
        decode_host.minimax_m3_nvfp4_index_score_decode(
            q[:, h : h + 1], cache, expected[h : h + 1], table, seq_lens,
            init_blocks, local_blocks, query_len, max_seq_len,
        )  # fmt: skip
    blocks = torch.arange(width, device="cuda")
    groups = query_len // cp.QL
    tile_lens = (
        seq_lens[:, None]
        - torch.arange(groups - 1, -1, -1, device="cuda")[None, :] * cp.QL
    ).clamp_min(0)
    nblk = (
        ((tile_lens + BLOCK - 1) // BLOCK).flatten().repeat_interleave(cp.QL)[:, None]
    )
    seen = torch.zeros(heads, rows, width, dtype=torch.int32, device="cuda")
    for rank in range(cp.WORLD):
        counts = torch.zeros(cp.CNT_WORDS, dtype=torch.int32, device="cuda")
        flags = torch.zeros(cp.FLAG_BYTES // 4, dtype=torch.int32, device="cuda")
        flags[cp.SC_FLAGS : cp.SC_FLAGS + cp.WORLD] = 1 << 30
        flags[cp.SC_FLAGS + rank] = 0
        iq_buf = torch.full((heads, rows, D), 0x7F, dtype=torch.uint8, device="cuda")
        iq_buf = iq_buf.view(torch.float8_e4m3fn)
        for p in range(heads):
            if p != rank:
                iq_buf[p] = q[:, p]
                w = cp.SC_IQ + p * cp.SC_MAX_REQS
                flags[w : w + cp.SC_MAX_REQS] = 1 << 30
        outs = []
        for _ in range(2):
            actual = torch.full_like(expected, float("nan"))
            dests = [actual[h].data_ptr() for h in range(heads)]
            decode_host.minimax_m3_nvfp4_index_score_push_cp(
                q[:, rank].contiguous(), cache, dests, width, rows * width * 4,
                table, seq_lens, init_blocks, local_blocks, rank, counts,
                [flags.data_ptr()] * 4, [iq_buf.data_ptr()] * 4, rows, grid=grid,
                query_len=query_len,
            )  # fmt: skip
            outs.append(actual)
        torch.accelerator.synchronize()
        actual = outs[0]
        assert torch.equal(actual.view(torch.int32), outs[1].view(torch.int32))
        mine = (blocks // cp.CP_CH) % cp.WORLD == rank
        in_runs = mine & (blocks // cp.CP_CH * cp.CP_CH < nblk)
        inside = in_runs & (blocks < nblk)
        assert actual[:, ~in_runs].isnan().all(), rank
        assert (actual[:, in_runs & ~inside] == float("-inf")).all(), rank
        assert torch.equal(
            actual[:, inside].view(torch.int32), expected[:, inside].view(torch.int32)
        ), rank
        seen += inside.int()
    assert torch.equal(seen, (blocks < nblk)[None].expand_as(seen).int())


@pytest.mark.parametrize("num_reqs", [1, 5, 17, 45])
@pytest.mark.parametrize("page_step", [1, 2])
@pytest.mark.parametrize(("init_blocks", "local_blocks"), [(1, 2), (0, 1)])
@pytest.mark.parametrize("grid", [3, 768])
@pytest.mark.parametrize("query_len", [4, 8])
def test_index_decode_cp_push_matches_single_head(
    num_reqs, page_step, init_blocks, local_blocks, grid, query_len
):
    """Grid 3: many runs and requests per workgroup."""
    torch.manual_seed(num_reqs)
    _check_cp_push(
        _lens(num_reqs, 9000), page_step, init_blocks, local_blocks, grid, query_len
    )


@pytest.mark.parametrize("grid", [3, 768])
@pytest.mark.parametrize("query_len", [4, 8])
def test_index_decode_cp_push_all_padding(grid, query_len):
    """Padding requests only (graph capture): nothing written."""
    _check_cp_push([0] * 5, 1, 1, 2, grid, query_len)


def test_index_decode_cp_push_eight_row_request_capacity():
    """Eight-row requests fill all 256 query-flag slots, including ragged tails."""
    _check_cp_push([129, 2049, 4095, 0] * 32, 2, 1, 2, 768, 8)


def _prefill_case(q_lens, seq_lens_list, num_heads, page_step=1):
    torch.manual_seed(0)
    table, num_blocks = _block_table(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device="cuda").cumsum(0)
    cache = _index_cache(num_blocks, page_step)
    q = torch.randn(sum(q_lens), num_heads, D, device="cuda") * 0.5
    return table, seq_lens, cu, cache, q.to(torch.float8_e4m3fn)


def _causal_blocks(q_len, s, num_heads):
    """[H, q_len, nb] blocks each row of a chunk ending its sequence sees."""
    nb = (s + BLOCK - 1) // BLOCK
    row_pos = s - q_len + torch.arange(q_len, device="cuda")
    seen = torch.arange(nb, device="cuda")[None, :] <= (row_pos // BLOCK)[:, None]
    return row_pos, seen[None].expand(num_heads, -1, -1)


@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize(
    ("q_lens", "seq_lens_list"),
    [
        ([129, 300, 5], [129, 2500, 2049]),  # 128-row tiles, ragged
        ([1000, 700], [4300, 700]),  # 256-row tiles, unaligned prefix
    ],
)
@pytest.mark.parametrize("page_step", [1, 2])
def test_index_prefill_matches_aiter_fp8(num_heads, q_lens, seq_lens_list, page_step):
    """Every block a row can see against AITER's FP8 prefill scorer."""
    from aiter.ops.msa_attention import pa_sparse_block_score_prefill

    table, seq_lens, cu, cache, q = _prefill_case(
        q_lens, seq_lens_list, num_heads, page_step
    )
    msl = max(seq_lens_list)
    args = (table, cu, seq_lens, 1, 2, max(q_lens), msl)
    width = (msl + BLOCK - 1) // BLOCK
    actual = torch.full((num_heads, sum(q_lens), width), float("nan"), device="cuda")
    minimax_m3_nvfp4_index_score_prefill(q, cache, actual, *args)
    expected = torch.full_like(actual, float("nan"))
    pages = _fp8_keys(cache)
    for h in range(num_heads):
        pa_sparse_block_score_prefill(
            q[:, h : h + 1].contiguous(), pages, expected[h : h + 1], *args
        )
    torch.accelerator.synchronize()
    q_start = 0
    for q_len, s in zip(q_lens, seq_lens_list):
        _, seen = _causal_blocks(q_len, s, num_heads)
        rows = slice(q_start, q_start + q_len)
        got = actual[:, rows, : seen.shape[2]][seen]
        want = expected[:, rows, : seen.shape[2]][seen]
        finite = want < 1e28
        assert torch.equal(got[~finite], want[~finite])
        torch.testing.assert_close(got[finite], want[finite], rtol=1e-5, atol=1e-4)
        q_start += q_len


@pytest.mark.parametrize(
    ("q_lens", "seq_lens_list"),
    [
        ([4096], [4096]),  # fresh prompt: every tile diagonal
        ([2000, 513, 3000], [9000, 513, 20000]),  # ragged, unaligned prefixes
        ([700, 1500], [30000, 1500]),  # long cached prefix + fresh
    ],
)
@pytest.mark.parametrize(("init_blocks", "local_blocks"), [(0, 1), (2, 2)])
@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_index_prefill_matches_reference(
    q_lens, seq_lens_list, init_blocks, local_blocks, num_heads, monkeypatch
):
    """Every kernel the host can pick (``kernel_shared.py`` with 128- and
    256-row tiles, ``kernel.py``) bit for bit alike, and against an fp32
    reference."""
    table, seq_lens, cu, cache, q = _prefill_case(
        q_lens, seq_lens_list, num_heads, page_step=2 if num_heads == 2 else 1
    )
    width = (max(seq_lens_list) + BLOCK - 1) // BLOCK + 3
    args = (
        table,
        cu,
        seq_lens,
        init_blocks,
        local_blocks,
        max(q_lens),
        max(seq_lens_list),
    )

    def run(shared, mq):
        monkeypatch.setattr(prefill_host, "PIN_SHARED", shared)
        monkeypatch.setattr(prefill_host, "PIN_MQ", mq)
        out = torch.full((num_heads, sum(q_lens), width), float("nan"), device="cuda")
        minimax_m3_nvfp4_index_score_prefill(q, cache, out, *args)
        return out

    variants = [(True, 1), (True, 2), (False, None)]
    outs = [run(*v) for v in variants]
    keys = _fp8_keys(cache).float()
    torch.accelerator.synchronize()
    q_start = 0
    for req, (q_len, s) in enumerate(zip(q_lens, seq_lens_list)):
        row_pos, seen = _causal_blocks(q_len, s, num_heads)
        nb = seen.shape[2]
        rows = slice(q_start, q_start + q_len)
        for v, out in zip(variants[1:], outs[1:]):
            got, ref = out[:, rows, :nb][seen], outs[0][:, rows, :nb][seen]
            assert torch.equal(got, ref), (v, q_len, s)
        page_keys = keys[table[req, :nb].long()].reshape(-1, D)
        for r0 in range(0, q_len, 512):  # [H, rows, tokens] dots in slices
            r1 = min(q_len, r0 + 512)
            want = _ref_block_scores(
                q[q_start + r0 : q_start + r1], page_keys, row_pos[r0:r1] + 1,
                init_blocks, local_blocks,
            )  # fmt: skip
            got = outs[0][:, q_start + r0 : q_start + r1, :nb]
            sl = seen[:, r0:r1]
            _assert_scores(got[sl], want[sl], (q_len, s))
        q_start += q_len
