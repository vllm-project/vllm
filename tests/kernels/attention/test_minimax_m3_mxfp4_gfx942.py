# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MXFP4 K/V cache on ROCm gfx942: the Triton page writer and the
sparse attend that decodes the pages in registers."""

import importlib.util

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm (gfx942) kernels", allow_module_level=True)

from vllm.platforms.rocm import on_gfx942  # noqa: E402

if not on_gfx942():
    pytest.skip("gfx942 path", allow_module_level=True)

from vllm.models.minimax_m3.amd.ops.mxfp4_attn_gfx942 import (  # noqa: E402
    mxfp4_sparse_attend_gfx942,
    store_kv_mxfp4_gfx942,
)
from vllm.models.minimax_m3.amd.ops.mxfp4_pages import (  # noqa: E402
    PAGE_BYTES,
    mxfp4_page_views,
)

BLOCK = 128
D = 128
TOPK = 16
GROUP = 16  # query heads per KV head
SCALE = D**-0.5
SENTINEL = 0xA5
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

requires_aiter = pytest.mark.skipif(
    importlib.util.find_spec("aiter") is None, reason="needs aiter"
)


def reference_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 codes ``[..., 128]`` and e8m0 ``[..., 4]`` of rows ``[..., 128]``:
    e8m0 = ceil(log2(amax / 6)), e2m1 rounded to nearest even, saturating at 6,
    with the sign kept on zero as AITER's packers do."""
    x = x.float().unflatten(-1, (4, 32))
    amax = x.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    e = torch.ceil(torch.log2(amax / 6.0))
    y = x / torch.exp2(e)[..., None]
    grid = torch.tensor(E2M1, device=x.device)
    dist = (y.abs().clamp(max=6.0)[..., None] - grid).abs()
    best = dist.min(-1, keepdim=True).values
    even = (torch.arange(8, device=x.device) % 2) == 0
    tie = (dist == best).sum(-1) > 1
    code = torch.where(tie, ((dist == best) & even).int().argmax(-1), dist.argmin(-1))
    code = code | (y < 0).int() << 3
    return code.flatten(-2), (e + 127).to(torch.uint8)


def dequantize_pages(pages: torch.Tensor) -> torch.Tensor:
    """float32 rows ``[..., 128, 128]`` of uint8 pages ``[..., 8704]``."""
    data, scale = mxfp4_page_views(pages)
    d = data.int()
    code = torch.stack((d & 15, d >> 4), dim=-1).flatten(-2)
    grid = torch.tensor(E2M1, device=pages.device)
    value = grid[code & 7] * torch.where(code & 8 != 0, -1.0, 1.0)
    step = torch.exp2(scale.float() - 127.0)[..., None]
    return (value.unflatten(-1, (4, 32)) * step).flatten(-2)


@pytest.mark.parametrize("num_tokens", [1, 17, 1000])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_store_kv_bit_exact(num_tokens, num_kv_heads, dtype):
    """Every written byte is the reference MXFP4 of the row; other slots and
    padding (slot -1) tokens are left alone."""
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    H = num_kv_heads
    k = torch.randn(num_tokens, H, D, device="cuda", generator=g).to(dtype)
    v = (torch.randn(num_tokens, H, D, device="cuda", generator=g) * 3).to(dtype)
    k[0, 0, :32] = 0  # an all-zero group
    k[-1, H - 1] *= 1e-3
    blocks = (num_tokens + BLOCK - 1) // BLOCK + 2
    cache = torch.full(
        (blocks, 2 * H, BLOCK, 68), SENTINEL, dtype=torch.uint8, device="cuda"
    )
    slots = torch.randperm(blocks * BLOCK, device="cuda", generator=g)[:num_tokens]
    if num_tokens > 7:
        slots[::7] = -1
    store_kv_mxfp4_gfx942(k, v, cache, slots)

    ok = slots >= 0
    data, scale = mxfp4_page_views(cache.view(blocks, 2 * H, PAGE_BYTES))
    blk, tok = slots[ok] // BLOCK, slots[ok] % BLOCK
    for side, src in ((0, k), (1, v)):
        code, e = reference_quant(src[ok])
        heads = slice(side * H, (side + 1) * H)
        want = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
        torch.testing.assert_close(data[blk, heads, tok], want, rtol=0, atol=0)
        torch.testing.assert_close(scale[blk, heads, tok], e, rtol=0, atol=0)
    written = torch.zeros(blocks * BLOCK, dtype=torch.bool, device="cuda")
    written[slots[ok]] = True
    untouched = data.permute(0, 2, 1, 3).reshape(blocks * BLOCK, -1)[~written]
    assert (untouched == SENTINEL).all()


def _case(rows, H, causal_lens, num_pages, seed):
    """Random pages, a compacted per-(row, head) top-16 table and queries."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cache = torch.randint(
        0,
        256,
        (num_pages, 2 * H, BLOCK, 68),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    _, sc = mxfp4_page_views(cache.view(num_pages, 2 * H, PAGE_BYTES))
    sc.copy_(
        torch.randint(118, 130, sc.shape, dtype=torch.uint8, device="cuda", generator=g)
    )
    table = torch.zeros(rows * H, TOPK, dtype=torch.int32, device="cuda")
    ctx = torch.zeros(rows * H, dtype=torch.int32, device="cuda")
    for r in range(rows):
        causal = causal_lens[r]
        nb = -(-causal // BLOCK)
        for h in range(H):
            sel = torch.randperm(num_pages, device="cuda", generator=g)[: min(TOPK, nb)]
            table[r * H + h, : len(sel)] = sel.int()
            # The last selected block is the causal tail.
            tail = causal - (nb - 1) * BLOCK
            ctx[r * H + h] = (len(sel) - 1) * BLOCK + tail
    q = torch.randn(
        rows, H * GROUP, D, device="cuda", generator=g, dtype=torch.bfloat16
    )
    return cache, table, ctx, q


def _reference(q, cache, table, ctx, H):
    pages = cache.view(cache.shape[0], 2 * H, PAGE_BYTES)
    out = torch.empty(q.shape, dtype=torch.float32, device="cuda")
    for r in range(q.shape[0]):
        for h in range(H):
            n = int(ctx[r * H + h])
            blocks = table[r * H + h, : -(-n // BLOCK)].long()
            k = dequantize_pages(pages[blocks, h]).reshape(-1, D)[:n]
            v = dequantize_pages(pages[blocks, H + h]).reshape(-1, D)[:n]
            qh = q[r, h * GROUP : (h + 1) * GROUP].float()
            out[r, h * GROUP : (h + 1) * GROUP] = (
                torch.softmax(qh @ k.T * SCALE, -1) @ v
            )
    return out


@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("causal_lens", [[1, 129, 300], [2048, 5000, 64, 7]])
@pytest.mark.parametrize("num_splits", [None, 1, 3, 16])
def test_sparse_attend(num_kv_heads, causal_lens, num_splits):
    """The attend is fp32 attention over the exactly dequantized pages, to the
    bf16 rounding of its P and its output."""
    H = num_kv_heads
    rows = len(causal_lens)
    cache, table, ctx, q = _case(rows, H, causal_lens, 64, seed=rows * H)
    out = torch.empty_like(q)
    mxfp4_sparse_attend_gfx942(
        q,
        cache,
        table,
        ctx,
        out,
        SCALE,
        num_kv_heads=H,
        max_blocks=TOPK,
        num_splits=num_splits,
    )
    ref = _reference(q, cache, table, ctx, H)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max())


def test_sparse_attend_page16_table():
    """The page-16 table AITER's top-k emits (entry ``(block * 8 + j) * H + h``
    at column ``8 * i``) is read in place."""
    H = 2
    causal_lens = [1000, 3000]
    rows = len(causal_lens)
    cache, table, ctx, q = _case(rows, H, causal_lens, 48, seed=7)
    page16 = torch.zeros(rows * H, TOPK * 8, dtype=torch.int32, device="cuda")
    j = torch.arange(8, device="cuda", dtype=torch.int32)
    for row in range(rows * H):
        h = row % H
        page16[row] = ((table[row, :, None] * 8 + j) * H + h).flatten()
    out = torch.empty_like(q)
    mxfp4_sparse_attend_gfx942(
        q,
        cache,
        page16,
        ctx,
        out,
        SCALE,
        num_kv_heads=H,
        max_blocks=TOPK,
        bt_col_stride=8,
        page_div=8 * H,
    )
    ref = _reference(q, cache, table, ctx, H)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max())


@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_sparse_attend_prefill_dequant_once(num_kv_heads):
    """Dequantizing every request block once to bf16 first gives the same
    attention as decoding per row (the dequantization is exact)."""
    H = num_kv_heads
    causal_lens = [300, 2048, 5000, 129]
    rows = len(causal_lens)
    num_pages = 64
    cache, table, ctx, q = _case(rows, H, causal_lens, num_pages, seed=11)
    # Two requests that between them own every page.
    block_table = torch.randperm(num_pages, device="cuda").int().view(2, 32)
    seq_lens = torch.full((2,), 32 * BLOCK, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    mxfp4_sparse_attend_gfx942(
        q,
        cache,
        table,
        ctx,
        out,
        SCALE,
        num_kv_heads=H,
        max_blocks=TOPK,
        prefill_block_table=block_table,
        prefill_seq_lens=seq_lens,
        prefill_max_seq_len=32 * BLOCK,
    )
    ref = _reference(q, cache, table, ctx, H)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max())


@requires_aiter
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_sparse_attend_prefill_gluon(num_kv_heads):
    """The prefill attend (blocks dequantized once to bf16 page-16 pages, then
    AITER's sparse paged attention) matches fp32 attention over the pages."""
    from vllm.models.minimax_m3.amd.ops.mxfp4_attn_gfx942 import (
        mxfp4_sparse_attend_prefill_gfx942,
    )

    H = num_kv_heads
    causal_lens = [300, 2048, 5000, 129]
    rows = len(causal_lens)
    num_pages = 64
    cache, table, ctx, q = _case(rows, H, causal_lens, num_pages, seed=13)
    block_table = torch.randperm(num_pages, device="cuda").int().view(2, 32)
    seq_lens = torch.full((2,), 32 * BLOCK, dtype=torch.int32, device="cuda")
    j = torch.arange(8, device="cuda", dtype=torch.int32)
    page16 = torch.zeros(rows * H, TOPK * 8, dtype=torch.int32, device="cuda")
    for row in range(rows * H):
        page16[row] = ((table[row, :, None] * 8 + j) * H + row % H).flatten()
    out = torch.empty_like(q)
    mxfp4_sparse_attend_prefill_gfx942(
        q,
        cache,
        page16,
        ctx,
        out,
        SCALE,
        num_kv_heads=H,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=32 * BLOCK,
    )
    ref = _reference(q, cache, table, ctx, H)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max())


@requires_aiter
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_prefill_fp8_transcode(num_kv_heads):
    """The fp8 transcode is exact (per-token power-of-two scales, every chunk
    within 9 binades of the token's largest) and AITER's fp8 sparse paged
    attention over it tracks fp32 attention over the MXFP4 pages to fp8
    compute (fp8 Q and P) accuracy."""
    from vllm.models.minimax_m3.amd.ops.mxfp4_attn_gfx942 import (
        Mxfp4PrefillScratch,
        mxfp4_sparse_attend_prefill_fp8_gfx942,
    )

    H = num_kv_heads
    causal_lens = [300, 2048, 5000, 129]
    rows = len(causal_lens)
    num_pages = 64
    cache, table, ctx, q = _case(rows, H, causal_lens, num_pages, seed=17)
    block_table = torch.randperm(num_pages, device="cuda").int().view(2, 32)
    seq_lens = torch.full((2,), 32 * BLOCK, dtype=torch.int32, device="cuda")
    j = torch.arange(8, device="cuda", dtype=torch.int32)
    page16 = torch.zeros(rows * H, TOPK * 8, dtype=torch.int32, device="cuda")
    for row in range(rows * H):
        page16[row] = ((table[row, :, None] * 8 + j) * H + row % H).flatten()
    scratch = Mxfp4PrefillScratch(num_pages, H, torch.device("cuda"))
    out = torch.empty_like(q)
    assert mxfp4_sparse_attend_prefill_fp8_gfx942(
        q,
        cache,
        page16,
        ctx,
        out,
        SCALE,
        num_kv_heads=H,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=32 * BLOCK,
        scratch=scratch,
    )
    ref = _reference(q, cache, table, ctx, H)
    err = (out.float() - ref).norm() / ref.norm()
    assert err < 0.08, err
    # Too small a scratch declines rather than overrunning.
    small = Mxfp4PrefillScratch(8, H, torch.device("cuda"))
    assert not mxfp4_sparse_attend_prefill_fp8_gfx942(
        q,
        cache,
        page16,
        ctx,
        out,
        SCALE,
        num_kv_heads=H,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=32 * BLOCK,
        scratch=small,
    )


@pytest.mark.parametrize(
    "tokens,hq,hk,hi", [(1, 16, 1, 1), (4, 16, 1, 1), (8, 32, 2, 2)]
)
def test_gfx942_qkv_index_store_matches_split_path(tokens, hq, hk, hi):
    """One-launch decode write path == main norm/RoPE + store_kv_mxfp4 +
    index_norm_rope_store (bit-identical q, index q, K/V and index caches)."""
    from vllm.models.minimax_m3.amd.ops import mxfp4_gfx942 as G
    from vllm.models.minimax_m3.amd.ops.index_norm_rope_store_gfx942 import (
        index_norm_rope_store_gfx942,
    )
    from vllm.models.minimax_m3.amd.ops.main_norm_rope_gfx942 import (
        main_norm_rope_gfx942,
    )
    from vllm.models.minimax_m3.amd.ops.mxfp4_attn_gfx942 import store_kv_mxfp4_gfx942
    from vllm.models.minimax_m3.amd.ops.qkv_index_store_gfx942 import (
        qkv_index_store_gfx942,
    )

    torch.manual_seed(tokens)
    d = 128
    dev = "cuda"
    qkv = (torch.randn(tokens, (hq + 2 * hk + hi + 1) * d, device=dev) * 3).to(
        torch.bfloat16
    )
    pos = torch.randint(0, 500000, (tokens,), device=dev)
    w = [(torch.randn(d, device=dev) * 0.1).to(torch.bfloat16) for _ in range(4)]
    cs = torch.randn(524288, 64, device=dev).to(torch.bfloat16)
    nb = 8
    ic = [
        torch.zeros(nb, G.MANAGER_BLOCK_BYTES, dtype=torch.uint8, device=dev)
        for _ in range(2)
    ]
    kc = [
        torch.zeros(nb, 2 * hk, 128, 68, dtype=torch.uint8, device=dev)
        for _ in range(2)
    ]
    islots = torch.randperm(nb * 128, device=dev)[:tokens].to(torch.int64)
    kslots = torch.randperm(nb * 128, device=dev)[:tokens].to(torch.int64)
    vs = (hq + hk) * d
    q1, k1 = main_norm_rope_gfx942(
        qkv,
        pos,
        w[0],
        w[1],
        cs,
        num_heads=hq,
        num_kv_heads=hk,
        head_dim=d,
        rotary_dim=64,
        eps=1e-6,
    )
    store_kv_mxfp4_gfx942(
        k1, qkv[:, vs : vs + hk * d].view(tokens, hk, d), kc[0], kslots
    )
    iq1 = index_norm_rope_store_gfx942(
        qkv,
        pos,
        w[2],
        w[3],
        cs,
        ic[0],
        islots,
        q_start=(hq + 2 * hk) * d,
        num_heads=hi,
        head_dim=d,
        rotary_dim=64,
        eps=1e-6,
    )
    q2, iq2 = qkv_index_store_gfx942(
        qkv,
        pos,
        w[0],
        w[1],
        w[2],
        w[3],
        cs,
        ic[1],
        islots,
        kc[1],
        kslots,
        num_heads=hq,
        num_kv_heads=hk,
        num_index_heads=hi,
        head_dim=d,
        rotary_dim=64,
        eps=1e-6,
    )
    assert torch.equal(q1, q2)
    assert torch.equal(iq1, iq2)
    assert torch.equal(kc[0], kc[1])
    assert torch.equal(ic[0], ic[1])


def test_decode_falls_back_to_triton_without_aiter_gluon(monkeypatch):
    """An AITER without the gluon MXFP4 decode selects the Triton attend."""
    import sys

    from vllm.models.minimax_m3.amd import sparse_attention_mxfp4 as S

    monkeypatch.setitem(
        sys.modules, "aiter.ops.triton.gluon.pa_decode_mxfp4_gluon", None
    )
    S._aiter_mxfp4_decode.cache_clear()
    try:
        assert S._aiter_mxfp4_decode() is None
    finally:
        S._aiter_mxfp4_decode.cache_clear()
