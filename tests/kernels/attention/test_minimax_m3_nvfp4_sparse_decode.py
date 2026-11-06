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


def _quantized_decode(q, cache, topk, table, seq_lens, qlen, extra_rows=0, **kw):
    """The kernel's MXFP8 output (values, scales) of the rows plus ``extra_rows``
    sentinel rows that must stay untouched."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
    )

    k_scale, v_scale = _scales()
    rows, heads, dim = q.shape
    out_q = torch.full(
        (rows + extra_rows, heads * dim), 0x5A, dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e4m3fn)
    out_s = torch.full(
        (rows + extra_rows, heads * dim // MXFP8_BLOCK_SIZE),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    minimax_m3_nvfp4_sparse_decode_fly(
        q, cache, topk, table, seq_lens, SM_SCALE, None, k_scale=k_scale,
        v_scale=v_scale, decode_query_len=qlen, out_q=out_q, out_scale=out_s, **kw,
    )  # fmt: skip
    return out_q, out_s


def _assert_mxfp8_equals_quantized_bf16(q, cache, topk, table, seq_lens, qlen, **kw):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    rows = q.shape[0]
    out_bf16 = _decode(q, cache, topk, table, seq_lens, qlen, **kw)
    assert torch.isfinite(out_bf16).all()
    ref_q, ref_s = mxfp8_e4m3_quantize(out_bf16.view(rows, -1))
    got_q, got_s = _quantized_decode(
        q, cache, topk, table, seq_lens, qlen, extra_rows=3, **kw
    )
    assert torch.equal(got_q[:rows].view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(got_s[:rows], ref_s)
    # Nothing past the rows is written.
    assert (got_q[rows:].view(torch.uint8) == 0x5A).all()
    assert (got_s[rows:] == 0xA5).all()


def _assert_multi_request_mxfp8(num_reqs, qlen, ctx, **kw):
    torch.manual_seed(num_reqs * 7 + qlen)
    nb = (ctx + BLOCK - 1) // BLOCK
    # Every request reads the same pages (read-only): a small cache at any context.
    cache = torch.randint(
        0, 256, (nb + 3, 2, BLOCK, 72), dtype=torch.uint8, device="cuda"
    )
    for side in range(2):
        sf = cache.view(nb + 3, -1)[:, side * 9216 + 8192 : (side + 1) * 9216]
        sf.masked_fill_((sf & 0x7F) == 0x7F, 0x38)  # no NaN scales
    lens = [ctx - 97 * r for r in range(num_reqs)]
    table = torch.arange(nb, dtype=torch.int32, device="cuda").repeat(num_reqs, 1)
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = _selections(lens, qlen)
    q = (torch.randn(num_reqs * qlen, GROUP, D, device="cuda") * 0.02).bfloat16()
    _assert_mxfp8_equals_quantized_bf16(q, cache, topk, table, seq_lens, qlen, **kw)


@pytest.mark.parametrize("ctx", [30_000, 150_000, 500_000])
@pytest.mark.parametrize("qlen", [4, 8])
@pytest.mark.parametrize("num_reqs", [1, 2, 3, 5, 8, 12, 16, 24])
def test_decode_mxfp8_output_matches_quantized_bf16(num_reqs, qlen, ctx):
    """The MXFP8 epilogue equals ``mxfp8_e4m3_quantize`` of the same kernel's
    bf16 output, bit for bit, at the default split choice for each batch."""
    _assert_multi_request_mxfp8(num_reqs, qlen, ctx)


@pytest.mark.parametrize("splits", [2, 4])
@pytest.mark.parametrize("num_reqs", [1, 5, 8, 9, 16, 17, 24])
def test_decode_mxfp8_output_multi_request_xcd_splits(num_reqs, splits):
    """DSpark 7 rows (8 per request) on the 8-XCD grid, explicit split counts
    across the request ranges a split heuristic distinguishes."""
    _assert_multi_request_mxfp8(num_reqs, 8, 150_000, splits=splits, xcd_count=8)


@pytest.mark.parametrize("splits", [1, 2, 4, 8])
@pytest.mark.parametrize("xcd_count", [0, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("qlen", [1, 4, 8])
def test_decode_mxfp8_output_splits_and_heads(qlen, num_kv_heads, xcd_count, splits):
    """Every split count (the last workgroup quantizes the merged O), XCD grid
    and KV-head count; padding requests (seq_len 0) give zero values and scale 0,
    like the quantized bf16 zeros."""
    torch.manual_seed(3)
    lens = [700, 0, 3000, 20000, 0, 129]
    table, num_blocks = _block_table([max(s, 1) for s in lens])
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = _selections([max(s, 1) for s in lens], qlen, num_kv_heads)
    cache = _nvfp4_cache(num_blocks, _scales(), num_kv_heads)
    # Magnitudes over many binades, so block scales cover a wide E8M0 range.
    mag = torch.pow(2.0, torch.randint(-6, 7, (topk.shape[1], GROUP * num_kv_heads, 1)))
    q = (torch.randn(topk.shape[1], GROUP * num_kv_heads, D) * mag).cuda().bfloat16()
    _assert_mxfp8_equals_quantized_bf16(
        q, cache, topk, table, seq_lens, qlen, splits=splits, xcd_count=xcd_count
    )


@pytest.mark.parametrize(
    "mode,prefills,decodes,fused",
    [
        ("NONE", 0, 8, True),  # decode-only, eager
        ("FULL", 0, 8, True),  # decode-only, full graph
        ("NONE", 2, 8, False),  # mixed batch
        ("NONE", 3, 0, False),  # prefill-only
        ("PIECEWISE", 0, 8, False),  # captured piecewise graph
        ("NONE", 0, 0, False),  # no sparse decode work
    ],
)
def test_o_proj_mxfp8_handoff_is_decode_only(
    monkeypatch, mode, prefills, decodes, fused
):
    """Only a decode-only step outside a piecewise graph hands MXFP8 to o_proj;
    mixed, prefill and piecewise steps keep the bf16 output and the linear's
    own quant."""
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode
    from vllm.models.minimax_m3.amd import model
    from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata

    md = object.__new__(MiniMaxM3SparseMetadata)
    md.num_prefills, md.num_decodes = prefills, decodes
    ctx = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode[mode], attn_metadata={"layer": md}
    )
    monkeypatch.setattr(model, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(model, "_can_fuse_input_norm_mxfp8", lambda _: True)
    layer = SimpleNamespace(
        _o_proj_mxfp8=None,
        impl=SimpleNamespace(forward_decode_mxfp8=None),
        o_proj=None,
        layer_name="layer",
    )
    emits = model.MiniMaxM3SparseAttention._emits_o_proj_mxfp8
    assert emits(layer) is fused
    assert layer._o_proj_mxfp8 is True  # static, cached across steps

    ctx.attn_metadata = {}  # no sparse metadata for this layer
    assert emits(layer) is False


def test_o_proj_mxfp8_handoff_needs_kernel_and_mxfp8_o_proj(monkeypatch):
    """The cached decision is False without a fp8_out kernel on the impl or an
    o_proj that takes MXFP8 input, whatever the batch."""
    from types import SimpleNamespace

    from vllm.config import CUDAGraphMode
    from vllm.models.minimax_m3.amd import model
    from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata

    md = object.__new__(MiniMaxM3SparseMetadata)
    md.num_prefills, md.num_decodes = 0, 8
    ctx = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE, attn_metadata={"layer": md}
    )
    monkeypatch.setattr(model, "get_forward_context", lambda: ctx)
    emits = model.MiniMaxM3SparseAttention._emits_o_proj_mxfp8

    monkeypatch.setattr(model, "_can_fuse_input_norm_mxfp8", lambda _: False)
    no_mxfp8 = SimpleNamespace(
        _o_proj_mxfp8=None,
        impl=SimpleNamespace(forward_decode_mxfp8=None),
        o_proj=None,
        layer_name="layer",
    )
    assert emits(no_mxfp8) is False

    monkeypatch.setattr(model, "_can_fuse_input_norm_mxfp8", lambda _: True)
    no_kernel = SimpleNamespace(
        _o_proj_mxfp8=None, impl=SimpleNamespace(), o_proj=None, layer_name="layer"
    )
    assert emits(no_kernel) is False


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


def _default_split_case(num_kv_heads, num_reqs):
    """A decode batch with padding requests (seq_len 0) for the default-splits tests."""
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
    return q, cache, topk, table, seq_lens


def _record_kernels(monkeypatch, host):
    """The (splits, topk, xcds) of every kernel the default path compiles."""
    chosen = []
    real = host._get_kernel

    def record(*args):
        chosen.append(args)
        return real(*args)

    monkeypatch.setattr(host, "_get_kernel", record)
    return chosen


@pytest.mark.parametrize(
    "queried, gfx950, expected",
    [(8, True, 8), (4, True, 0), (2, True, 0), (0, True, 0), (8, False, 0)],
)
def test_xcd_grid_needs_the_tuned_topology(monkeypatch, queried, gfx950, expected):
    """The XCD-aware grid is on only for 8 XCDs on gfx950: a DPX/QPX partition
    of the same GPU reports fewer and keeps the grid off."""
    import vllm.platforms.rocm as rocm
    from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode import host

    monkeypatch.setattr(host, "_query_xcd_count", lambda device_index: queried)
    monkeypatch.setattr(rocm, "on_gfx950", lambda: gfx950)
    host._xcd_count.cache_clear()
    try:
        assert host._xcd_count(0) == expected
    finally:
        host._xcd_count.cache_clear()


@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_reqs", [1, 9, 40])
def test_default_splits_without_the_xcd_grid(monkeypatch, num_kv_heads, num_reqs):
    """On a device without the 8-XCD topology (e.g. a DPX partition) the default
    path uses the cost model on a plain grid and agrees with the unsplit kernel."""
    from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode import host

    monkeypatch.setattr(host, "_xcd_count", lambda device_index: 0)
    q, cache, topk, table, seq_lens = _default_split_case(num_kv_heads, num_reqs)
    chosen = _record_kernels(monkeypatch, host)
    out = _decode(q, cache, topk, table, seq_lens, 8)

    rows = num_reqs * 8
    expected = host.decode_splits(rows * num_kv_heads, 16, host.cu_count(q.device))
    if rows * num_kv_heads > host.MAX_SPLIT_ROWS:
        expected = 1
    assert chosen == [(expected, 16, 0)]
    unsplit = _decode(q, cache, topk, table, seq_lens, 8, splits=1)
    real_rows = (seq_lens.repeat_interleave(8) > 0).cpu()
    assert torch.isfinite(out).all()
    assert (out[~real_rows] == 0).all()
    assert _rel(out[real_rows], unsplit[real_rows]) < 3e-3


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
@pytest.mark.parametrize("num_reqs", [1, 5, 8, 9, 16, 17, 24, 40])
def test_default_splits_are_chosen_per_xcd(monkeypatch, num_kv_heads, num_reqs):
    """The default split count is the XCD table at one KV head per rank and the
    cost model otherwise, counted over every request including padding
    (seq_len 0), and agrees with the unsplit kernel."""
    from vllm.models.minimax_m3.amd.ops.nvfp4_sparse_decode import host

    # The table is tuned for the full 8-XCD device; partitioned (DPX/QPX)
    # devices report fewer XCDs and run the default path without the XCD grid.
    if host._xcd_count(torch.accelerator.current_device_index()) != 8:
        pytest.skip("needs an 8-XCD device")
    q, cache, topk, table, seq_lens = _default_split_case(num_kv_heads, num_reqs)
    chosen = _record_kernels(monkeypatch, host)
    out = _decode(q, cache, topk, table, seq_lens, 8)

    xcds = host._xcd_count(q.device.index)
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
