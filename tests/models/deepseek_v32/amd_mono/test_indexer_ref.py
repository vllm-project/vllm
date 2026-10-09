# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the golden decode-indexer model (indexer_ref.py) on synthetic data with
GLM-5.2 shapes (hidden 6144, q_lora 2048, 32 index heads x 128, rope 64 interleaved,
block 16, top-k 2048), and of the kernel-side address / ordering models built on it."""

import itertools
import random

import torch

from tests.models.deepseek_v32.amd_mono import indexer_ref as R

g = torch.Generator().manual_seed(0)
WAVES, LANES, THREADS, TOPK, MAXS = 8, 64, 512, 2048, 4096
ITEMS = MAXS // THREADS


def test_cache_layout():
    """value_offset == the Triton writer's formula; write/read round trip; scale tail
    position."""
    for o in range(16):
        d = torch.arange(128)
        # kernels.py:98-103 (TILE 16)
        triton = o // 16 * 16 * 128 + o % 16 * 16 + d // 16 * 16 * 16 + d % 16
        assert torch.equal(R.value_offset(o, d), triton)
    offs = torch.stack([R.value_offset(o, torch.arange(128)) for o in range(16)])
    # a permutation of the value region
    assert offs.unique().numel() == 16 * 128 and int(offs.max()) == 2047
    cache = torch.zeros(5, 16, 132, dtype=torch.uint8)
    ks = []
    for slot in (0, 7, 16 * 3 + 15):
        k = (torch.randn(128, generator=g) * 3).to(R.FP8)
        R.cache_write(cache, slot, k, 0.125)
        ks.append((slot, k))
    for slot, k in ks:
        v, s = R.cache_read(cache, slot // 16)
        assert (
            torch.equal(v[slot % 16].view(torch.uint8), k.view(torch.uint8))
            and float(s[slot % 16]) == 0.125
        )


def test_ue8m0_and_rope():
    v = torch.randn(7, 32, 128, generator=g) * torch.logspace(-3, 3, 7)[:, None, None]
    q, s = R.ue8m0_fp8(v)
    assert torch.equal(torch.exp2(torch.log2(s).round()), s)  # power of two
    deq = q.float() * s[..., None]
    assert (
        float(q.float().abs().max()) <= 448
        and float(((deq - v).abs() / v.abs().amax(-1, keepdim=True)).max()) < 0.07
    )
    # RoPE: rotation preserves pair norms; pos 0 (cos 1, sin 0) is the identity
    cs = torch.cat([torch.ones(1, 32), torch.zeros(1, 32)], 1)
    x = torch.randn(3, 128, generator=g)
    assert torch.equal(R.rope_interleave(x, cs, torch.zeros(3, dtype=torch.long)), x)
    ang = torch.rand(10, 32, generator=g) * 6
    cs2 = torch.cat([ang.cos(), ang.sin()], 1)
    y = R.rope_interleave(x, cs2, torch.tensor([3, 4, 9]))
    n0 = x[:, :64].view(3, 32, 2).norm(dim=-1)
    n1 = y[:, :64].view(3, 32, 2).norm(dim=-1)
    assert torch.allclose(n0, n1, atol=1e-5) and torch.equal(y[:, 64:], x[:, 64:])


def test_topk_and_csr():
    # short rows: identity in order, -1 tail
    logits = torch.randn(3, 6000, generator=g)
    sl = torch.tensor([5, 2048, 6000])
    top, info = R.topk_ref(logits, sl)
    assert (
        top[0, :5].tolist() == list(range(5))
        and int(top[0, 5]) == -1
        and top[1].tolist() == list(range(2048))
    )
    assert set(top[2].tolist()) == set(
        torch.topk(logits[2, :6000], 2048).indices.tolist()
    )
    # an arrival-order permutation of the same set is accepted, a swapped-in non-member
    # is not
    perm = top.clone()
    perm[2] = perm[2][torch.randperm(2048, generator=g)]
    assert all(R.topk_equal_mod_ties(perm, info))
    bad = perm.clone()
    bad[2, 0] = int((set(range(6000)) - set(perm[2].tolist())).pop())
    assert R.topk_equal_mod_ties(bad, info) == [True, True, False]
    # threshold ties: any choice among them is valid
    lt = torch.zeros(1, 3000)
    lt[0, :2000] = 1.0
    t2, i2 = R.topk_ref(lt, torch.tensor([3000]))
    assert i2[0]["need"] == 48 and len(i2[0]["ties"]) == 1000
    alt = t2.clone()
    alt[0, 2000:] = torch.arange(2900, 2948, dtype=torch.int32)
    assert all(R.topk_equal_mod_ties(alt, i2))
    # CSR: physical slots, 0 for -1 entries, row lengths min(seq_len, 2048)
    bt = torch.tensor([[7, 3, 0, 0], [2, 9, 4, 0]], dtype=torch.int32)
    tk = torch.full((2, 2048), -1, dtype=torch.int32)
    tk[0, :20] = torch.arange(20, dtype=torch.int32)
    tk[1, :3] = torch.tensor([0, 17, 40], dtype=torch.int32)
    indptr, ind = R.csr(tk, torch.tensor([20, 3000]), bt)
    assert indptr.tolist() == [0, 20, 2068]
    assert ind[:20].tolist() == [7 * 16 + i for i in range(16)] + [
        3 * 16 + i for i in range(4)
    ]
    assert (
        ind[20:23].tolist() == [2 * 16, 9 * 16 + 1, 4 * 16 + 8]
        and int(ind[23:].abs().sum()) == 0
    )


def test_decode_step_glm_shapes():
    """Whole step with GLM-5.2 shapes; logits cross-checked by an independent dense
    recomputation."""
    H, QL = 6144, 2048
    T = 3
    W = dict(
        w_qa=torch.randn(QL + 576, H, generator=g) / H**0.5,
        w_qa_norm=1 + 0.1 * torch.randn(QL, generator=g),
        w_qb=torch.randn(32 * 128, QL, generator=g) / QL**0.5,
        w_wk=torch.randn(160, H, generator=g) / H**0.5,
        k_norm_w=1 + 0.1 * torch.randn(128, generator=g),
        k_norm_b=0.1 * torch.randn(128, generator=g),
    )
    seq = torch.tensor([37, 2048, 3100])
    nblk = 340
    perm = torch.randperm(nblk, generator=g)
    bt = torch.zeros(T, 200, dtype=torch.int32)
    off = 0
    for r in range(T):
        nb = (int(seq[r]) + 15) // 16
        bt[r, :nb] = perm[off : off + nb].int()
        off += nb
    cache = torch.zeros(nblk, 16, 132, dtype=torch.uint8)
    # history keys: random FP8 rows with power-of-2 scales at every position < seq - 1
    for r in range(T):
        for p in range(int(seq[r]) - 1):
            R.cache_write(
                cache,
                int(bt[r, p // 16]) * 16 + p % 16,
                (torch.randn(128, generator=g)).to(R.FP8),
                2.0 ** int(torch.randint(-6, -2, (1,), generator=g)),
            )
    pos = seq - 1
    slots = torch.tensor(
        [int(bt[r, int(pos[r]) // 16]) * 16 + int(pos[r]) % 16 for r in range(T)]
    )
    ang = torch.outer(
        torch.arange(4096).float(), 1.0 / (8e6 ** (torch.arange(0, 64, 2).float() / 64))
    )
    cos_sin = torch.cat([ang.cos(), ang.sin()], 1)
    h = torch.randn(T, H, generator=g).to(torch.bfloat16)
    out = R.decode_step(W, h, pos, slots, seq, bt, cache, cos_sin)
    # the current token's index-K is in the cache before scoring and is always selected
    for r in range(T):
        v, s = R.cache_read(cache, int(slots[r]) // 16)
        assert torch.equal(
            v[int(slots[r]) % 16].view(torch.uint8), out["k_fp8"][r].view(torch.uint8)
        )
    # independent dense logits
    for r in range(T):
        L = int(seq[r])
        rows = [R.cache_read(cache, int(bt[r, p // 16])) for p in range(L)]
        K = torch.stack([rows[p][0][p % 16].float() for p in range(L)])
        S = torch.stack([rows[p][1][p % 16] for p in range(L)])
        dense = (
            torch.relu(torch.einsum("hd,td->th", out["iq_fp8"][r].float(), K))
            * out["weights"][r]
        ).sum(1) * S
        assert torch.allclose(dense, out["logits"][r, :L], rtol=1e-5, atol=1e-6)
    assert out["topk"][0, :37].tolist() == list(range(37)) and out["topk"][
        1
    ].tolist() == list(range(2048))
    assert all(R.topk_equal_mod_ties(out["topk"], out["topk_info"]))
    assert out["indptr"].tolist() == [0, 37, 37 + 2048, 37 + 2 * 2048]


def test_pack_index_weights():
    """Fused-indexer weights: FP8 block-128 index projections (rounding only) + the
    BF16 weights_proj."""
    from vllm.models.deepseek_v32.amd.mono.ckpt_weights import pack_index_weights
    from vllm.models.deepseek_v32.amd.mono.fp8_attention import dequant_fp8_block

    bf = torch.bfloat16
    wk, wq = (
        torch.randn(128, 6144, generator=g).to(bf),
        torch.randn(4096, 2048, generator=g).to(bf),
    )
    wp = torch.randn(32, 6144, generator=g).to(bf)
    t = pack_index_weights(wk, wq, wp, torch.ones(128), torch.zeros(128))
    assert t["s_index_k"].shape == (1, 48) and t["s_index_q"].shape == (32, 16)
    assert t["w_index_w"].dtype == bf and t["g_index_k"].dtype == torch.float32
    for (w, s), ref in (
        ((t["w_index_q"], t["s_index_q"]), wq),
        ((t["w_index_k"], t["s_index_k"]), wk),
    ):
        d = dequant_fp8_block(w, s, 128)
        assert float((d - ref.float()).norm() / ref.float().norm()) < 0.05


def test_fused_kernel_addressing():
    """The fused indexer's (kernel/glm/kernel.py) index-cache addressing ==
    vLLM's SHUFFLE writer (deepseek_v32/common/kernels.py _fp8_quant_and_cache_write,
    BLOCK_TILE = HEAD_TILE = 16, block 16): cache-stage dword stores (even lane l: dims
    2l..2l+3), score-stage 8-byte key loads, fp32 scale words."""
    HD, BS, STRIDE = 128, 16, 16 * 132  # uint8 cache.stride(0)

    def vllm_off(slot, d):  # byte offset of value d of the token at slot
        return slot // BS * STRIDE + int(R.value_offset(slot % BS, torch.tensor(d)))

    for slot in (0, 5, 15, 16, 37, 16 * 99 + 15):
        blk, off = slot // 16, slot % 16
        written = {}
        # cache stage: byte = blk*stride + off*16 + (i0//16)*256 + i0%16, i0 = 2*lane
        for lane in range(0, 64, 2):
            i0 = lane * 2
            byte = blk * STRIDE + off * 16 + (i0 // 16) * 256 + i0 % 16
            assert byte % 4 == 0
            # fp8_pack4(q0, q1, n0, n1): dims i0, i0+1 (own), i0+2, i0+3 (lane+1)
            for j in range(4):
                written[i0 + j] = byte + j
        assert sorted(written) == list(range(HD))
        assert all(written[d] == vllm_off(slot, d) for d in range(HD))
        # score stage: k_base = blk*stride + (key%16)*16; k = k32*32 + (lane//16)*8; 8
        # bytes at k_base+(k//16)*256+k%16
        k_base = blk * STRIDE + off * 16
        for k32 in range(HD // 32):
            for lq in range(4):
                k = k32 * 32 + lq * 8
                b0 = k_base + (k // 16) * 256 + k % 16
                assert b0 % 4 == 0 and [b0 + j for j in range(8)] == [
                    vllm_off(slot, k + j) for j in range(8)
                ]
        # scale: kernel (blk*stride + 16*128 + off*4) // 4 (f32 word) == vLLM
        # scale_byte_off // 4
        assert blk * STRIDE + 16 * HD + off * 4 == blk * STRIDE + BS * HD + off * 4


def test_ue8m0_exponent_bits():
    """Kernel _ue8m0_scale (exponent bits + mantissa-nonzero carry) ==
    2^ceil(log2(fp32(max(amax,1e-4)/448))) exactly, incl. exact powers of two and their
    neighbours; inverse is the exact reciprocal."""
    import struct

    def f32(x):
        return struct.unpack("f", struct.pack("f", x))[0]

    def kernel(amax):
        # div_rn: correctly rounded fp32 quotient
        x = f32(max(f32(amax), f32(1e-4)) / 448.0)
        bits = struct.unpack("I", struct.pack("f", x))[0]
        e = (bits >> 23) & 255
        e += 1 if bits & 0x7FFFFF else 0
        return struct.unpack("f", struct.pack("I", e << 23))[0], struct.unpack(
            "f", struct.pack("I", (254 - e) << 23)
        )[0]

    vals = [0.0, 1e-9, 1e-4, 0.0448, 448.0, 448.0 * 2**-10, 1.0, 3.0, 1e4]
    for k in range(-20, 20):
        p = 448.0 * 2.0**k
        vals += [p, f32(p * (1 + 2**-23)), f32(p * (1 - 2**-24))]
    vals += torch.rand(2000, generator=g).mul(50).exp2().mul(1e-6).tolist()
    for a in vals:
        sc, inv = kernel(a)
        t = torch.tensor([[a, -a]], dtype=torch.float32)
        ex = float(R.ue8m0_fp8(t, exact=True)[1][0])
        assert sc == ex, (a, sc, ex)
        assert sc * inv == 1.0
        assert f32(a) / sc <= 448.0, a  # in range: no E4M3 overflow
        lg = float(R.ue8m0_fp8(t)[1][0])  # vLLM-style fp32 log2
        # only in the fp32-log2 rounding window just above a power of two: 2x, and lg
        # overflows 448
        if lg != sc:
            assert sc == 2 * lg and f32(a) / lg > 448.0, (a, sc, lg)


def test_rowpar_layernorm_order():
    """index_cache_rowpar: LayerNorm statistics as a 64-lane xor-butterfly
    wave sum of per-lane pairs (k[2l] + k[2l+1]) in fp32, then the kernel's affine /
    RoPE / ue8m0 FP8 -> vs indexer_ref's golden (torch sums): every difference is a
    1-code E4M3 neighbour, <= 4 per row (the GPU check's bound), scale
    identical. (FMA contraction on GPU moves single roundings the same way.)"""

    def wave_sum(v):  # v [rows, 64] fp32, butterfly lane ^ 32, ^ 16, ... ^ 1
        for off in (32, 16, 8, 4, 2, 1):
            idx = torch.arange(64) ^ off
            v = v + v[:, idx]
        return v[:, 0]

    rows = 3000
    k = (
        (
            torch.randn(rows, 128, generator=g)
            * torch.rand(rows, 1, generator=g).mul(4).exp2()
        )
        .to(torch.bfloat16)
        .float()
    )
    gw = 1 + 0.1 * torch.randn(128, generator=g)
    bw = 0.1 * torch.randn(128, generator=g)
    cs = (
        torch.cat(
            [torch.cos(torch.arange(32.0) * 0.37), torch.sin(torch.arange(32.0) * 0.37)]
        )
        .to(torch.bfloat16)
        .float()[None]
    )
    pair = k[:, 0::2] + k[:, 1::2]
    mean = wave_sum(pair) * (1.0 / 128)
    d = k - mean[:, None]
    dd = d[:, 0::2] * d[:, 0::2] + d[:, 1::2] * d[:, 1::2]
    rstd = torch.rsqrt(wave_sum(dd) * (1.0 / 128) + 1e-6)
    v = d * rstd[:, None] * gw + bw
    v = R.rope_interleave(v, cs.expand(1, 64), torch.zeros(rows, dtype=torch.long))
    kq, ks = R.ue8m0_fp8(v, exact=True)
    gold = R.rope_interleave(
        R.layer_norm(k, gw, bw, 1e-6),
        cs.expand(1, 64),
        torch.zeros(rows, dtype=torch.long),
    )
    gq, gs = R.ue8m0_fp8(gold, exact=True)
    a, b = kq.view(torch.uint8).int(), gq.view(torch.uint8).int()
    diff = a != b
    near = ((a & 0x80) == (b & 0x80)) & (((a & 0x7F) - (b & 0x7F)).abs() == 1)
    assert bool((near | ~diff).all()), "a non-neighbour FP8 difference"
    per_row = diff.sum(1)
    assert int(per_row.max()) <= 4, int(per_row.max())
    scale_diff = int((ks != gs).sum())
    assert scale_diff <= rows // 1000


def test_batched_score_task():
    """index_score_batched: emulate the batched 64-key score task
    (kernel/glm/kernel.py score_task_batched) -- RoPE'd + per-head ue8m0 FP8 index q
    staged as bf16 pairs, per-lane key words from the SHUFFLE cache, the current key
    (position L-1) taken from its mailbox words via new_key.select(...) while the cache
    still holds STALE bytes for that slot (no index_ready wait), C[head, key] of mfma
    16x16x32 bf16, relu x bf16 head weight x q scale, key scale (mailbox for the current
    key) x 128^-.5 x 32^-.5 -> == indexer_ref.paged_logits on the cache with the current
    key written."""
    HD, NH = 128, 32
    L, nb_tot = 2100, 200
    cache = torch.zeros(nb_tot, 16, 132, dtype=torch.uint8)
    flat = cache.view(nb_tot, -1)
    flat[:, :2048] = (
        (torch.randn(nb_tot, 2048, generator=g) * 3).to(R.FP8).view(torch.uint8)
    )
    flat[:, 2048:] = (
        torch.exp2(torch.randint(-8, -2, (nb_tot, 16), generator=g).float())
        .contiguous()
        .view(torch.uint8)
    )
    bt = torch.randperm(nb_tot, generator=g)[: (L + 15) // 16].int()[None]
    cur = L - 1
    slot = int(bt[0, cur // 16]) * 16 + cur % 16
    # this launch's current key (mailbox values)
    knew = (torch.randn(HD, generator=g) * 3).to(R.FP8)
    ksnew = 2.0**-5
    # what the score task may see: the cache stage has not written the slot yet
    stale = cache.clone()
    R.cache_write(cache, slot, knew, ksnew)  # what the golden scores
    cos_sin = torch.cat(
        [
            torch.cos(torch.arange(4096.0)[:, None] * torch.arange(32.0)[None] * 1e-3),
            torch.sin(torch.arange(4096.0)[:, None] * torch.arange(32.0)[None] * 1e-3),
        ],
        1,
    )
    cos_sin = cos_sin.to(torch.bfloat16).float()
    q_raw = (torch.randn(NH, HD, generator=g)).to(torch.bfloat16).float()
    w_raw = torch.randn(NH, generator=g).to(torch.bfloat16).float()
    q = R.rope_interleave(q_raw[None], cos_sin, torch.tensor([cur]))[0]
    qf, s_q = R.ue8m0_fp8(q, exact=True)
    w_eff = w_raw * s_q  # keys[h] * keys[32 + h]
    qb = qf.float().to(torch.bfloat16).float()  # staged bf16 (exact)
    sflat = stale.view(nb_tot, -1)
    got = torch.zeros(L)
    for tile in range((L + 63) // 64):
        C = torch.zeros(NH, 64)
        for kg in range(4):
            for j in range(16):
                key = tile * 64 + kg * 16 + j
                safe = min(key, L - 1)
                blk, off = int(bt[0, safe // 16]), safe % 16
                vo = R.value_offset(off, torch.arange(HD))
                kc = sflat[blk, vo].view(R.FP8).float()
                # new_key.select(mailbox, cache)
                kv = knew.float() if safe == cur else kc
                C[:, kg * 16 + j] = qb @ kv
        part = (torch.relu(C) * w_eff[:, None]).sum(0)
        for c in range(64):
            key = tile * 64 + c
            if key < L:
                blk, off = int(bt[0, key // 16]), key % 16
                ksc = (
                    sflat[blk, 2048 + 4 * off : 2048 + 4 * off + 4]
                    .contiguous()
                    .view(torch.float32)[0]
                )
                ksc = torch.tensor(ksnew) if key == cur else ksc
                got[key] = part[c] * ksc * (128**-0.5) * (32**-0.5)
    ref = R.paged_logits(
        qf[None],
        (w_eff * (128**-0.5) * (32**-0.5))[None],
        cache,
        bt,
        torch.tensor([L]),
        L,
    )[0]
    rel = float((got - ref).norm() / ref.norm())
    assert rel < 1e-6, rel
    # the stale slot really differs: scoring it from the cache would be wrong
    assert not torch.equal(stale.view(-1), cache.view(-1))


def test_select_radix11():
    """select_radix11: the 11/11/10-bit radix (descending bins, per-thread
    bin runs, running count from the top) finds the same threshold key and the same
    number of threshold ties to take as the exact k-th-largest -> the compaction
    (unchanged) emits topk_ref's canonical row. Random, heavy ties, -0.0 / +0.0,
    all-equal and L just above 2048."""
    K = 2048

    def radix11(keys, k):
        prefix, remain = 0, k
        for shift, nbits, hi in ((21, 11, 32), (10, 11, 21), (0, 10, 10)):
            nb = 1 << nbits
            hist = [0] * nb
            for key in keys:
                if hi == 32 or (key >> hi) == (prefix >> hi):
                    hist[(key >> shift) & (nb - 1)] += 1
            run = 0
            # descending digits = the threads' bin runs in order
            for d in range(nb - 1, -1, -1):
                if run < remain <= run + hist[d]:
                    prefix |= d << shift
                    remain -= run
                    break
                run += hist[d]
        return prefix, remain  # threshold key, ties to take

    cases = []
    for L in (2049, 2100, 3000, 4096):
        cases.append(torch.randn(L, generator=g))
        cases.append(torch.randint(-3, 4, (L,), generator=g).float())
        z = torch.where(
            torch.rand(L, generator=g) < 0.5, torch.tensor(0.0), torch.tensor(-0.0)
        )
        z[::5] = torch.randn(len(z[::5]), generator=g)
        cases.append(z)
        cases.append(torch.full((L,), 1.5))
    for x in cases:
        L = x.numel()
        keys = R.order_key(x).tolist()
        thr, rem = radix11(keys, K)
        srt = sorted(keys, reverse=True)
        assert thr == srt[K - 1], (L, thr, srt[K - 1])
        n_gt = sum(k_ > thr for k_ in keys)
        assert rem == K - n_gt, (L, rem, K - n_gt)
        # the unchanged compaction: above-threshold ascending, then the lowest-position
        # `rem` ties
        above = [i for i, k_ in enumerate(keys) if k_ > thr]
        ties = [i for i, k_ in enumerate(keys) if k_ == thr]
        sel = above + ties[:rem]
        want, _ = R.topk_ref(x[None], torch.tensor([L]))
        assert sel == want[0].tolist()


def _lds_select(
    scores: torch.Tensor, barrier_after_digit: bool, order
) -> tuple[list, list]:
    """-> (CSR positions written in [0, topk) order, out-of-range offsets written).
    ``order``: wave order within every barrier phase."""
    L = scores.numel()
    key = R.order_key(scores).tolist() + [0] * (MAXS - L)
    thr_true = sorted(key[:L], reverse=True)[TOPK - 1]
    # final select_digit's hit thread
    keys = {256: thr_true, 257: TOPK - sum(k > thr_true for k in key[:L])}
    wave_state: list[dict] = [dict() for _ in range(WAVES)]
    out, oob = {}, []

    # phase A (after select_digit's last barrier): every wave reads (prefix, remain) =
    # keys[256], keys[257]
    def a_read(w):
        wave_state[w]["thr"] = keys[256]

    # phase B: item flags + gt scan_flags -> lane 63 of wave w stores keys[256 + w] (its
    # gt count)
    def b_write(w):
        thr = wave_state[w]["thr"]
        cnt = []
        for lane in range(LANES):
            t = w * LANES + lane
            items = [t * ITEMS + j for j in range(ITEMS)]
            cnt.append(sum((i < L) and key[i] > thr for i in items))
        wave_state[w]["cnt"] = cnt
        keys[256 + w] = sum(cnt)

    # phase C (after scan barrier): offsets from keys[256..263], gt stores
    def c_store(w):
        thr = wave_state[w]["thr"]
        before = sum(keys[256 + v] for v in range(w))
        run_ = before
        for lane in range(LANES):
            t = w * LANES + lane
            for j in range(ITEMS):
                i = t * ITEMS + j
                if i < L and key[i] > thr:
                    if run_ < TOPK:
                        out[run_] = i
                    else:
                        oob.append(run_)
                    run_ += 1

    phases = (
        [[a_read], [b_write], [c_store]]
        if barrier_after_digit
        else [[a_read, b_write], [c_store]]
    )
    for ph in phases:
        for w in order:
            for op in ph:
                op(w)
    return [out.get(k, -1) for k in range(TOPK)], oob


def test_barrier_after_select_digit_is_schedule_independent():
    """CPU model of the fused paged indexer's select-stage LDS protocol
    (last radix digit through the gt / eq compaction scans): the 8 waves run one at a
    time between barriers in an adversarial order (a legal GPU interleaving). Without
    the barrier after select_digit some schedules corrupt the CSR; with it every
    schedule gives the canonical result."""
    g = torch.Generator().manual_seed(0)
    L = 3500
    scores = torch.randn(L, generator=g)
    k = R.order_key(scores)
    thr = torch.sort(k, descending=True).values[TOPK - 1]
    want = torch.nonzero(k > thr).view(-1).tolist()  # canonical gt part (ascending)
    orders = [list(range(WAVES)), list(reversed(range(WAVES)))]
    rnd = random.Random(1)
    for _ in range(30):
        o = list(range(WAVES))
        rnd.shuffle(o)
        orders.append(o)
    # without the barrier: the in-order schedule (lagging waves read after wave 0 / 1
    # stored their counts) breaks
    bad = 0
    for o in orders:
        got, oob = _lds_select(scores, False, o)
        if got[: len(want)] != want or oob:
            bad += 1
    assert bad > 0, "the model should expose the race without the barrier"
    # with the barrier: every schedule gives the canonical result
    for o in orders + [
        list(p) for p in itertools.islice(itertools.permutations(range(WAVES)), 200)
    ]:
        got, oob = _lds_select(scores, True, o)
        assert got[: len(want)] == want and not oob, o
