# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 NVFP4 sparse decode (FlyDSL) against fp64 attention over
K / V = e4m3_rn_satfinite(e2m1 x block scale) x global scale."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm (gfx950) FlyDSL kernel", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("gfx950 instructions", allow_module_level=True)

from tests.kernels.attention.minimax_m3_nvfp4_ref import (  # noqa: E402
    dequant,
    head_offsets,
    insert_nvfp4_kv,
    nvfp4_kv_cache_token_major,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode.host import (  # noqa: E402
    minimax_m3_nvfp4_sparse_decode_fly,
)

BLOCK = 128
D = 128
GROUP = 16  # query heads per KV head
TOPK = 16
SM_SCALE = D**-0.5


def _nvfp4_cache(num_blocks, scales, num_kv_heads=1):
    """Every slot of the cache filled with random K / V."""
    k_scale, v_scale = scales
    cache = torch.zeros(
        num_blocks, 2 * num_kv_heads, BLOCK, 72, dtype=torch.uint8, device="cuda"
    )
    slots = torch.arange(num_blocks * BLOCK, device="cuda")
    shape = (slots.numel(), num_kv_heads, D)
    k = (torch.randn(shape, device="cuda") * 2).bfloat16()
    v = (torch.randn(shape, device="cuda") * 2).bfloat16()
    insert_nvfp4_kv(k, v, cache, slots, k_scale, v_scale)
    return cache


def _block_table(seq_lens):
    widths = [(s + BLOCK - 1) // BLOCK for s in seq_lens]
    num_blocks = sum(widths) + 3
    perm = torch.randperm(num_blocks, device="cuda", dtype=torch.int32)
    table = torch.zeros(len(seq_lens), max(widths), dtype=torch.int32, device="cuda")
    start = 0
    for r, w in enumerate(widths):
        table[r, :w] = perm[start : start + w]
        start += w
    return table, num_blocks


def _scales():
    return (
        torch.tensor(0.5, dtype=torch.float32, device="cuda"),
        torch.tensor(2.0, dtype=torch.float32, device="cuda"),
    )


def _selections(seq_lens, qlen, num_kv_heads=1, seed=0):
    """[Hkv, rows, 16] picks, mostly shared within a request, with -1 padding
    and an out-of-range entry; head h's picks rotated by h."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    topk = torch.full(
        (1, len(seq_lens) * qlen, TOPK), -1, dtype=torch.int32, device="cuda"
    )
    for r, s in enumerate(seq_lens):
        nb = (s + BLOCK - 1) // BLOCK
        base = torch.randperm(nb, device="cuda", generator=g).to(torch.int32)
        for q in range(qlen):
            cur = max(s - qlen + q, 0) // BLOCK
            older = base[(base != cur) & (base <= cur)]
            pick = torch.cat([older[:10], older[10 + 3 * q : 13 + 3 * q]])
            sel = torch.cat([torch.tensor([cur], device="cuda"), pick])
            sel = torch.cat([sel, torch.tensor([nb + 5], device="cuda")])[:TOPK]
            topk[0, r * qlen + q, : sel.numel()] = sel.to(torch.int32)
    return torch.cat([topk.roll(h, -1) for h in range(num_kv_heads)])


def _reference(q, cache, topk, table, seq_lens, scales, qlen):
    """fp64 attention over the dequantized K / V; rows that see nothing give 0."""
    k_data, k_sf, v_data, v_sf = nvfp4_kv_cache_token_major(cache)
    k_all, v_all = (
        dequant(data, sf).clamp(-448, 448).to(torch.float8_e4m3fn).double() * s.item()
        for data, sf, s in ((k_data, k_sf, scales[0]), (v_data, v_sf, scales[1]))
    )
    hkv, rows, _ = topk.shape
    req = torch.arange(rows, device="cuda") // qlen
    pos = seq_lens.long()[req] - qlen + torch.arange(rows, device="cuda") % qlen
    offs = torch.arange(BLOCK, device="cuda")
    out = torch.zeros(q.shape, dtype=torch.float64, device="cuda")
    for h in range(hkv):
        blk = topk[h, :rows].long()
        ok = (pos[:, None] >= 0) & (blk >= 0) & (blk <= (pos // BLOCK)[:, None])
        blk = torch.where(ok, blk, 0)
        tok = blk[..., None] * BLOCK + offs  # [rows, TOPK, BLOCK]
        ok = (ok[..., None] & (tok <= pos[:, None, None])).flatten(1)
        page = table[req[:, None], blk].long()[..., None].expand_as(tok)
        k = k_all[page, h, tok % BLOCK].flatten(1, 2)
        v = v_all[page, h, tok % BLOCK].flatten(1, 2).masked_fill(~ok[..., None], 0)
        heads = slice(h * GROUP, (h + 1) * GROUP)
        s = torch.einsum("nhd,ntd->nht", q[:, heads].double(), k) * SM_SCALE
        s = s.masked_fill(~ok[:, None, :], float("-inf"))
        p = torch.softmax(s, -1).nan_to_num(0.0)
        out[:, heads] = torch.einsum("nht,ntd->nhd", p, v)
    return out


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def _decode(q, cache, topk, table, seq_lens, qlen, out=None, **kw):
    k_scale, v_scale = _scales()
    out = torch.full_like(q, float("nan")) if out is None else out
    minimax_m3_nvfp4_sparse_decode_fly(
        q, cache, topk, table, seq_lens, SM_SCALE, out, k_scale=k_scale,
        v_scale=v_scale, decode_query_len=qlen, **kw,
    )  # fmt: skip
    return out


@pytest.mark.parametrize("decode_query_len", [1, 4, 8])
@pytest.mark.parametrize("splits", [None, 1, 2, 4, 8])
@pytest.mark.parametrize("random_bytes", [False, True])
@pytest.mark.parametrize("xcd_count", [0, 8])
def test_decode_matches_reference(decode_query_len, splits, random_bytes, xcd_count):
    """Realistic and random-byte caches: close to fp64, bitwise repeatable."""
    torch.manual_seed(0)
    lens = [9, 300, 2100, 20000, 131, 4096]
    table, num_blocks = _block_table(lens)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = _selections(lens, decode_query_len)
    rows = topk.shape[1]
    if random_bytes:
        cache = torch.randint(
            0, 256, (num_blocks, 2, BLOCK, 72), dtype=torch.uint8, device="cuda"
        )
        for side in range(2):
            sf = cache.view(num_blocks, -1)[:, side * 9216 + 8192 : (side + 1) * 9216]
            sf.masked_fill_((sf & 0x7F) == 0x7F, 0x38)  # no NaN scales
        q = (torch.randn(rows, GROUP, D, device="cuda") * 0.01).bfloat16()
    else:
        cache = _nvfp4_cache(num_blocks, _scales())
        q = torch.randn(rows, GROUP, D, device="cuda").bfloat16()
    kw = dict(splits=splits, xcd_count=xcd_count)
    out = _decode(q, cache, topk, table, seq_lens, decode_query_len, **kw)
    again = _decode(q, cache, topk, table, seq_lens, decode_query_len, **kw)
    ref = _reference(q, cache, topk, table, seq_lens, _scales(), decode_query_len)
    assert torch.isfinite(out).all()
    assert torch.equal(out, again)
    assert _rel(out, ref) < 5e-3


@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("decode_query_len", [1, 4, 6])
@pytest.mark.parametrize("splits", [1, 4])
def test_decode_padding_rows_and_kv_heads(num_kv_heads, decode_query_len, splits):
    """Padding requests (seq_len 0) give zero rows; nothing past the rows is
    written."""
    torch.manual_seed(1)
    lens = [700, 0, 3000, 0, 0, 129]
    table, num_blocks = _block_table([max(s, 1) for s in lens])
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = _selections([max(s, 1) for s in lens], decode_query_len, num_kv_heads)
    cache = _nvfp4_cache(num_blocks, _scales(), num_kv_heads)
    rows = topk.shape[1]
    q = torch.randn(rows, GROUP * num_kv_heads, D, device="cuda").bfloat16()
    out = torch.full((rows + 3, *q.shape[1:]), float("nan"), device="cuda").bfloat16()
    _decode(
        q, cache, topk, table, seq_lens, decode_query_len, out[:rows], splits=splits
    )
    ref = _reference(q, cache, topk, table, seq_lens, _scales(), decode_query_len)
    real = (seq_lens.repeat_interleave(decode_query_len) > 0).cpu()
    assert _rel(out[:rows][real], ref[real]) < 5e-3
    assert (out[:rows][~real] == 0).all()
    assert out[rows:].isnan().all()
    zeros = torch.zeros_like(seq_lens)
    pad = _decode(q, cache, topk, table, zeros, decode_query_len, splits=splits)
    assert (pad == 0).all()


@pytest.mark.parametrize(
    "num_kv_heads,qlen,splits,xcd_count",
    [(1, 1, 1, 0), (1, 4, 4, 8), (4, 6, 1, 8), (4, 8, 4, 0)],
)
def test_garbage_past_the_length_is_ignored(num_kv_heads, qlen, splits, xcd_count):
    """NaN scale bytes past each request's length leave the output unchanged."""
    torch.manual_seed(42)
    lens = [131, 4097, 2100, 777, 65]
    table, blocks = _block_table(lens)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = _selections(lens, qlen, num_kv_heads)
    cache = _nvfp4_cache(blocks, _scales(), num_kv_heads)
    q = torch.randn(len(lens) * qlen, num_kv_heads * GROUP, D, device="cuda").bfloat16()
    kw = dict(splits=splits, xcd_count=xcd_count)
    out = _decode(q, cache, topk, table, seq_lens, qlen, **kw)
    dirty = cache.clone().view(blocks, -1)
    for side in range(2):
        start = side * num_kv_heads * 9216 + num_kv_heads * 8192
        region = dirty[:, start : start + num_kv_heads * 1024]
        region = region.view(blocks, num_kv_heads, 1024)
        _, _, offsets = head_offsets(side, cache.device)
        for req, length in enumerate(lens):
            page = table[req, length // BLOCK]
            region[page, :, offsets[length % BLOCK :].reshape(-1)] = 0x7F
    dirty_out = _decode(q, dirty.view_as(cache), topk, table, seq_lens, qlen, **kw)
    assert torch.isfinite(out).all()
    assert torch.equal(out, dirty_out)


@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("decode_query_len", [1, 4])
def test_decode_block_outer_pages_bitwise(num_kv_heads, decode_query_len):
    """A page inside a wider block (stride(0) past one page): the dense
    cache's output."""
    torch.manual_seed(2)
    lens = [300, 2100, 20000, 131]
    table, num_blocks = _block_table(lens)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = _selections(lens, decode_query_len, num_kv_heads)
    cache = _nvfp4_cache(num_blocks, _scales(), num_kv_heads)
    page = cache[0].numel()
    wide = torch.full((num_blocks, 3 * page), 0x7F, dtype=torch.uint8, device="cuda")
    wide_cache = wide[:, page : 2 * page].view(cache.shape)
    wide_cache.copy_(cache)
    q = torch.randn(topk.shape[1], GROUP * num_kv_heads, D, device="cuda").bfloat16()
    outs = [
        _decode(q, c, topk, table, seq_lens, decode_query_len)
        for c in (cache, wide_cache)
    ]
    assert torch.isfinite(outs[0]).all()
    assert torch.equal(outs[0], outs[1])


def test_xcd_splits_table():
    """Splits of the XCD-aware grid: 4 up to 8 requests (one per XCD) and for
    17-24, 2 for 9-16, the cost model beyond and off the tuned shape."""
    from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode import host

    got = [host._xcd_splits(n, 8, 16, 8, 1) for n in (1, 5, 8, 9, 16, 17, 24, 25)]
    assert got == [4, 4, 4, 2, 2, 4, 4, None]
    assert host._xcd_splits(5, 8, 16, 4, 1) is None  # DSpark 3: not tuned
    assert host._xcd_splits(5, 0, 16, 8, 1) is None  # no XCD-aware grid
    assert host._xcd_splits(5, 8, 32, 8, 1) is None  # other top-k width
    assert host._xcd_splits(5, 8, 16, 8, 2) is None  # TP2: over-splits
    assert host._xcd_splits(5, 8, 16, 8, 4) is None  # TP1: over-splits


@pytest.mark.parametrize("num_reqs", [1, 3, 5, 8, 9, 12, 16, 17, 24, 40])
@pytest.mark.parametrize("ctx", [30000, 500000])
def test_default_splits_match_unsplit(num_reqs, ctx):
    """The default split count agrees with the unsplit kernel (the summation
    order is the only difference) and repeats bitwise, at DSpark 7 batch sizes."""
    torch.manual_seed(num_reqs)
    lens = torch.randint(ctx * 6 // 10, ctx * 14 // 10 + 1, (num_reqs,)).tolist()
    table, num_blocks = _block_table(lens)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = _selections(lens, 8)
    cache = torch.randint(
        0, 256, (num_blocks, 2, BLOCK, 72), dtype=torch.uint8, device="cuda"
    )
    for side in range(2):
        sf = cache.view(num_blocks, -1)[:, side * 9216 + 8192 : (side + 1) * 9216]
        sf.masked_fill_((sf & 0x7F) == 0x7F, 0x38)  # no NaN scales
    q = (torch.randn(num_reqs * 8, GROUP, D, device="cuda") * 0.05).bfloat16()
    out = _decode(q, cache, topk, table, seq_lens, 8)
    unsplit = _decode(q, cache, topk, table, seq_lens, 8, splits=1)
    assert torch.isfinite(out).all()
    assert _rel(out, unsplit) < 3e-3
    assert torch.equal(out, _decode(q, cache, topk, table, seq_lens, 8))


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("num_reqs", [1, 5, 8, 9, 16, 17, 24, 40])
def test_default_splits_are_chosen_per_xcd(monkeypatch, num_kv_heads, num_reqs):
    """The default split count is the XCD table at one KV head per rank and the
    cost model otherwise, counted over every request including padding
    (seq_len 0), and agrees with the unsplit kernel."""
    from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode import host

    torch.manual_seed(num_reqs)
    lens = [0 if r % 3 == 2 else 20000 + 700 * r for r in range(num_reqs)]
    live = [max(s, 1) for s in lens]
    table, num_blocks = _block_table(live)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = _selections(live, 8, num_kv_heads)
    cache = _nvfp4_cache(num_blocks, _scales(), num_kv_heads)
    q = (
        torch.randn(num_reqs * 8, GROUP * num_kv_heads, D, device="cuda") * 0.05
    ).bfloat16()

    chosen = []
    real = host._get_kernel

    def record(*args):
        chosen.append(args)
        return real(*args)

    monkeypatch.setattr(host, "_get_kernel", record)
    out = _decode(q, cache, topk, table, seq_lens, 8)

    xcds = host._xcd_count(q.device.index)
    assert xcds == 8
    rows = num_reqs * 8
    expected = host._xcd_splits(num_reqs, xcds, 16, 8, num_kv_heads)
    if expected is None:
        expected = host.decode_splits(rows * num_kv_heads, 16, host.cu_count(q.device))
    grid_x = -(-rows // (xcds * 8)) * xcds * 8
    if grid_x * num_kv_heads > host.MAX_SPLIT_ROWS:
        expected = 1
    assert chosen == [(expected, 16, xcds)]
    if num_kv_heads == 1:
        table_splits = {1: 4, 5: 4, 8: 4, 9: 2, 16: 2, 17: 4, 24: 4}
        assert expected == table_splits.get(num_reqs, expected)

    unsplit = _decode(q, cache, topk, table, seq_lens, 8, splits=1)
    real_rows = (seq_lens.repeat_interleave(8) > 0).cpu()
    assert torch.isfinite(out).all()
    assert (out[~real_rows] == 0).all()
    assert _rel(out[real_rows], unsplit[real_rows]) < 3e-3
