# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 NVFP4 sparse prefill (FlyDSL) against fp64 attention over the
exactly dequantized K / V, and bitwise stable across runs and call splits."""

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
    insert_nvfp4_kv,
    nvfp4_kv_cache_token_major,
)
from tests.kernels.mfma_spinner import mfma_spinner  # noqa: E402
from vllm.models.minimax_m3.amd.ops.nvfp4_prefill.host import (  # noqa: E402
    minimax_m3_nvfp4_sparse_prefill,
    supports_nvfp4_sparse_prefill,
)

BLOCK = 128
D = 128
GROUP = 16  # query heads per KV head
TOPK = 16
SM_SCALE = D**-0.5


def _case(q_lens, ctx_lens, hkv, k_scale, v_scale, seed=0):
    """The last q_lens[i] of ctx_lens[i] tokens are queries; shuffled pages;
    each token picks its own block and 15 random earlier ones."""
    gen = torch.Generator().manual_seed(seed)
    nbs = [(c + BLOCK - 1) // BLOCK for c in ctx_lens]
    n_pages = sum(nbs) + 2
    cache = torch.zeros(n_pages, 2 * hkv, BLOCK, 72, dtype=torch.uint8, device="cuda")
    slots = torch.arange(n_pages * BLOCK, device="cuda")
    k = (torch.randn(slots.numel(), hkv, D, generator=gen) * 2).bfloat16().cuda()
    v = (torch.randn(slots.numel(), hkv, D, generator=gen) * 2).bfloat16().cuda()
    insert_nvfp4_kv(k, v, cache, slots, k_scale, v_scale)
    perm = torch.randperm(n_pages, generator=gen).int()
    table = torch.zeros(len(q_lens), max(nbs), dtype=torch.int32)
    off = 0
    for r, nb in enumerate(nbs):
        table[r, :nb] = perm[off : off + nb]
        off += nb
    pos = torch.cat([torch.arange(c - q, c) for q, c in zip(q_lens, ctx_lens)]).int()
    req = torch.cat([torch.full((q,), r) for r, q in enumerate(q_lens)]).int()
    rows = []
    for _ in range(hkv):
        for p in pos.tolist():
            cur = p // BLOCK
            sel = torch.cat(
                [torch.tensor([cur]), torch.randperm(cur, generator=gen)[: TOPK - 1]]
            )
            rows.append(torch.cat([sel, torch.full((TOPK - sel.numel(),), -1)]))
    topk = torch.stack(rows).int().view(hkv, pos.numel(), TOPK)
    q = (torch.randn(pos.numel(), hkv * GROUP, D, generator=gen) * 2).bfloat16()
    return cache, table.cuda(), topk.cuda(), q.cuda(), pos.cuda(), req.cuda()


def _prefill(q, cache, topk, table, req, pos, hkv, k_scale=None, v_scale=None):
    out = torch.empty_like(q)
    minimax_m3_nvfp4_sparse_prefill(
        q, cache, topk, table, req, pos, hkv, SM_SCALE, out, k_scale=k_scale,
        v_scale=v_scale,
    )  # fmt: skip
    return out


def _fp64(q, cache, topk, table, req, pos, hkv, k_scale, v_scale, rows):
    """fp64 attention of the token rows ``rows`` over the dequantized K / V."""
    k_data, k_sf, v_data, v_sf = nvfp4_kv_cache_token_major(cache)
    kv = [
        (dequant(data, sf) * s).to(torch.bfloat16).double()  # [pages, hkv, 128, 128]
        for data, sf, s in ((k_data, k_sf, k_scale), (v_data, v_sf, v_scale))
    ]
    ref = torch.empty(len(rows), hkv * GROUP, D, dtype=torch.float64, device="cuda")
    for i, r in enumerate(rows):
        for h in range(hkv):
            blocks = [b for b in topk[h, r].tolist() if b >= 0]
            pages = [int(table[int(req[r]), b]) for b in blocks]
            kk = torch.cat([kv[0][p, h] for p in pages])
            vv = torch.cat([kv[1][p, h] for p in pages])
            key_pos = torch.cat(
                [torch.arange(b * BLOCK, (b + 1) * BLOCK) for b in blocks]
            ).cuda()
            s = (q[r, h * GROUP : (h + 1) * GROUP].double() @ kk.T) * SM_SCALE
            s[:, key_pos > int(pos[r])] = float("-inf")
            ref[i, h * GROUP : (h + 1) * GROUP] = torch.softmax(s, -1) @ vv
    return ref


def _rel(out, ref, rows):
    o = out[rows].double()
    return float((o - ref).norm() / ref.norm())


MAX_REL = 3e-3  # fp16 P and the bf16 output: ~2e-3


@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 2.0)])
def test_matches_fp64(scales):
    k_scale, v_scale = (torch.tensor(s, device="cuda") for s in scales)
    cache, table, topk, q, pos, req = _case([2048], [2048], 1, k_scale, v_scale)
    out = _prefill(q, cache, topk, table, req, pos, 1, k_scale, v_scale)
    assert torch.isfinite(out).all()
    rows = list(range(0, 2048, 37)) + list(range(2040, 2048))
    ref = _fp64(q, cache, topk, table, req, pos, 1, k_scale, v_scale, rows)
    assert _rel(out, ref, rows) < MAX_REL


@pytest.mark.parametrize("hkv", [1, 2, 4])
def test_batch_with_prefix_and_tiles_across_requests(hkv):
    one = torch.tensor(1.0, device="cuda")
    q_lens, ctx_lens = [1027, 514, 3, 262], [3000, 514, 1100, 900]
    cache, table, topk, q, pos, req = _case(q_lens, ctx_lens, hkv, one, one, seed=1)
    out = _prefill(q, cache, topk, table, req, pos, hkv, one, one)
    assert torch.isfinite(out).all()
    rows = [0, 1, 2, 3, 1026, 1027, 1028, 1540, 1541, 1542, 1543, 1544, 1805]
    ref = _fp64(q, cache, topk, table, req, pos, hkv, one, one, rows)
    assert _rel(out, ref, rows) < MAX_REL


@pytest.mark.parametrize("n_tok", [1, 3, 5])
def test_few_tokens(n_tok):
    one = torch.tensor(1.0, device="cuda")
    cache, table, topk, q, pos, req = _case([n_tok], [700], 1, one, one, seed=2)
    out = _prefill(q, cache, topk, table, req, pos, 1)  # unit scales by default
    rows = list(range(n_tok))
    ref = _fp64(q, cache, topk, table, req, pos, 1, one, one, rows)
    assert _rel(out, ref, rows) < MAX_REL


@pytest.mark.parametrize("hkv", [1, 4])
def test_block_outer_pages_bitwise(hkv):
    """A page inside a wider block (stride(0) past one page): the dense
    cache's output."""
    one = torch.tensor(1.0, device="cuda")
    q_lens, ctx_lens = [1027, 3, 262], [3000, 1100, 900]
    cache, table, topk, q, pos, req = _case(q_lens, ctx_lens, hkv, one, one, seed=3)
    page = cache[0].numel()
    wide = torch.full(
        (cache.shape[0], 3 * page), 0x7F, dtype=torch.uint8, device="cuda"
    )
    wide = wide[:, page : 2 * page].view(cache.shape)
    wide.copy_(cache)
    out = _prefill(q, cache, topk, table, req, pos, hkv)
    assert torch.isfinite(out).all()
    assert torch.equal(out, _prefill(q, wide, topk, table, req, pos, hkv))


def _shared_picks(pos, req, hkv, seed):
    """Indexer-like top-k: the own block, then a per-request base of 15
    earlier blocks, each swapped for a random one with probability 0.1."""
    gen = torch.Generator().manual_seed(seed)
    first_blk: dict[int, int] = {}
    for p, r in zip(pos.tolist(), req.tolist()):
        first_blk.setdefault(r, p // BLOCK)
    base = {
        r: torch.randperm(max(b, TOPK), generator=gen)[: TOPK - 1].tolist()
        for r, b in first_blk.items()
    }
    rows = []
    for _ in range(hkv):
        for p, r in zip(pos.tolist(), req.tolist()):
            cur = p // BLOCK
            sel = [cur]
            for b in base[r]:
                if torch.rand(1, generator=gen).item() < 0.1:
                    b = int(torch.randint(0, max(cur, 1), (1,), generator=gen))
                if b < cur and b not in sel:
                    sel.append(b)
            rows.append(sel + [-1] * (TOPK - len(sel)))
    return torch.tensor(rows, dtype=torch.int32).view(hkv, pos.numel(), TOPK).cuda()


# Grids above / below the CU count: the two- and one-round combine.
SHARED_CASES = [
    (
        [1501, 6, 2050],
        [9000, 2100, 6200],
        [0, 1, 2, 3, 1499, 1500, 1501, 1502, 1506, 3556],
    ),  # fmt: skip
    (
        [201, 6, 150],
        [5000, 2100, 3000],
        [0, 1, 2, 3, 199, 200, 201, 202, 206, 207, 356],
    ),  # fmt: skip
]


@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("case", SHARED_CASES)
def test_shared_picks_deterministic(hkv, case):
    """Pages shared within a tile, tiles across requests: matches fp64,
    bitwise repeatable."""
    one = torch.tensor(1.0, device="cuda")
    q_lens, ctx_lens, rows = case
    cache, table, _, q, pos, req = _case(q_lens, ctx_lens, hkv, one, one, seed=3)
    topk = _shared_picks(pos, req, hkv, seed=4)
    out = _prefill(q, cache, topk, table, req, pos, hkv)
    ref = _fp64(q, cache, topk, table, req, pos, hkv, one, one, rows)
    assert _rel(out, ref, rows) < MAX_REL
    assert torch.equal(out, _prefill(q, cache, topk, table, req, pos, hkv))


def test_fresh_prompt_calls_bitwise():
    """A fresh 4096-token prompt in one call equals the same rows in calls of
    1024 / 2048 / 1024 tokens, bitwise (DBO microbatches)."""
    one = torch.tensor(1.0, device="cuda")
    n = 4096
    cache, table, _, q, pos, req = _case([n], [n], 1, one, one, seed=5)
    topk = _shared_picks(pos, req, 1, seed=6)
    assert (topk[0, : 15 * BLOCK] == -1).any(1).all()

    def run(a, b):
        return _prefill(q[a:b], cache, topk[:, a:b], table, req[a:b], pos[a:b], 1)

    full = run(0, n)
    assert torch.equal(full, torch.cat([run(0, 1024), run(1024, 3072), run(3072, n)]))


@pytest.mark.parametrize("n_tok, ctx", [(1024, 31024), (4096, 64096)])
def test_bitwise_next_to_mfma_waves(n_tok, ctx):
    """Another kernel's MFMA waves on every CU leave the output unchanged
    (v_pk_mul_f32 gives wrong products there on MI355X)."""
    one = torch.tensor(1.0, device="cuda")
    cache, table, topk, q, pos, req = _case([n_tok], [ctx], 1, one, one, seed=7)

    def run(stream):
        with torch.cuda.stream(stream):
            return _prefill(q, cache, topk, table, req, pos, 1)

    # A high-priority queue, so the spinner never waits behind the attention.
    main, side = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
    ref = run(main)
    torch.accelerator.synchronize()
    spin = mfma_spinner(torch.cuda.get_device_properties(0).multi_processor_count)
    for _ in range(8):
        spin(200_000, side)  # 2 ms
        out = run(main)
        torch.accelerator.synchronize()
        assert torch.equal(out, ref)


def test_supported_shapes():
    """16 query heads per KV head, 16 selected 128-token blocks only."""
    assert supports_nvfp4_sparse_prefill(16, 1, D, TOPK, BLOCK)
    assert supports_nvfp4_sparse_prefill(32, 2, D, TOPK, BLOCK)
    assert not supports_nvfp4_sparse_prefill(8, 1, D, TOPK, BLOCK)
    assert not supports_nvfp4_sparse_prefill(16, 1, D, 8, BLOCK)
    assert not supports_nvfp4_sparse_prefill(16, 1, 64, TOPK, BLOCK)
