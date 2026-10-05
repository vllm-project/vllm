# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the golden decode-indexer model (indexer_ref.py) on synthetic data with
GLM-5.2 shapes (hidden 6144, q_lora 2048, 32 index heads x 128, rope 64 interleaved,
block 16, top-k 2048), and of the kernel-side address / ordering models built on it."""

import torch

from tests.models.deepseek_v32.amd_mono import indexer_ref as R

g = torch.Generator().manual_seed(0)


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


def test_fused_golden_weights():
    """Kernel-format weights (FP8 block-128 index projections + FP8 qkv_a) -> golden
    decode_step weights."""
    from vllm.models.deepseek_v32.amd.mono import index_weights as IW
    from vllm.models.deepseek_v32.amd.mono.fp8_attention import quant_fp8_block

    bf = torch.bfloat16
    wk, wq = (
        torch.randn(128, 6144, generator=g).to(bf),
        torch.randn(4096, 2048, generator=g).to(bf),
    )
    wp = torch.randn(32, 6144, generator=g).to(bf)
    t = IW.pack_index_weights(wk, wq, wp, torch.ones(128), torch.zeros(128))
    assert (
        t["s_index_k"].shape == (1, 48)
        and t["s_index_q"].shape == (32, 16)
        and t["w_index_w"].dtype == bf
    )
    t["w_qkv_a"], t["s_qkv_a"] = quant_fp8_block(
        torch.randn(2624, 6144, generator=g).to(bf), 128, 128
    )
    t["g_q"] = torch.ones(2048)
    W = R.fused_golden_weights(t)
    assert (
        W["w_qa"].shape == (2624, 6144)
        and W["w_qb"].shape == (4096, 2048)
        and W["w_wk"].shape == (160, 6144)
    )
    rel = float((W["w_qb"] - wq.float()).norm() / wq.float().norm())
    assert rel < 0.05, rel  # FP8 block-128 rounding only


def test_fused_kernel_addressing():
    """The fused indexer's (kernel/glm/kernel.py, index_paged) index-cache addressing ==
    vLLM's SHUFFLE writer (deepseek_v32/common/kernels.py _fp8_quant_and_cache_write,
    BLOCK_TILE = HEAD_TILE = 16, block 16): cache-stage dword stores (even lane l: dims
    2l..2l+3), score-stage 8-byte key loads, fp32 scale words."""
    HD, BS, STRIDE = 128, 16, 16 * 132  # uint8 cache.stride(0)

    def vllm_off(slot, d):  # byte offset of value d of the token at slot
        blk, off = slot // BS, slot % BS
        return (
            blk * STRIDE
            + off // 16 * 16 * HD
            + off % 16 * 16
            + d // 16 * 16 * 16
            + d % 16
        )

    for slot in (0, 5, 15, 16, 37, 16 * 99 + 15):
        blk, off = slot // 16, slot % 16
        written = {}
        for lane in range(
            0,
            64,
            2,
            # cache stage: byte = blk*stride + off*16 + (i0//16)*256 + i0%16,
            # i0 = 2*lane
        ):
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
    window = 0
    for a in vals:
        sc, inv = kernel(a)
        t = torch.tensor([[a, -a]], dtype=torch.float32)
        ex = float(R.ue8m0_fp8(t, exact=True)[1][0])
        assert sc == ex, (a, sc, ex)
        assert sc * inv == 1.0
        assert f32(a) / sc <= 448.0, a  # in range: no E4M3 overflow
        lg = float(R.ue8m0_fp8(t)[1][0])  # vLLM-style fp32 log2
        if (
            lg != sc
            # only the fp32-log2 rounding window just above a power of two: 2x, and lg
            # overflows 448
        ):
            assert sc == 2 * lg and f32(a) / lg > 448.0, (a, sc, lg)
            window += 1


def test_fused_select_order():
    """The kernel's paged select (radix threshold, thread-major compaction: every key
    above the threshold in ascending position, then the lowest-position threshold ties)
    == indexer_ref.topk_ref canonical order, incl. -0.0 / +0.0 and heavy ties; identity
    for L <= 2048; CSR via the block table == indexer_ref.csr."""
    THREADS, MAXS, K = 512, 4096, 2048
    for L, mode in (
        (1, "rand"),
        (2048, "rand"),
        (2049, "rand"),
        (3001, "ties"),
        (4096, "zeros"),
        (2500, "rand"),
    ):
        if mode == "rand":
            x = torch.randn(L, generator=g)
        elif mode == "ties":
            x = torch.randint(-3, 4, (L,), generator=g).float()
        else:
            x = torch.where(
                torch.rand(L, generator=g) < 0.5, torch.tensor(0.0), torch.tensor(-0.0)
            )
            x[::7] = torch.randn(len(x[::7]), generator=g)
        key = R.order_key(x)
        if L <= K:
            sel = list(range(L))
        else:
            thr = int(torch.sort(key, descending=True).values[K - 1])
            items = MAXS // THREADS
            gt = [
                i
                for t in range(THREADS)
                for i in range(t * items, (t + 1) * items)
                if i < L and int(key[i]) > thr
            ]
            eq = [
                i
                for t in range(THREADS)
                for i in range(t * items, (t + 1) * items)
                if i < L and int(key[i]) == thr
            ]
            sel = gt + eq[: K - len(gt)]
        want, _ = R.topk_ref(x[None], torch.tensor([L]))
        assert sel == want[0, : min(L, K)].tolist(), (L, mode)
        bt = torch.randperm(400, generator=g)[: (L + 15) // 16 + 1].int()[None]
        slots = [int(bt[0, i // 16]) * 16 + i % 16 for i in sel]
        _, csr = R.csr(want, torch.tensor([L]), bt)
        assert csr.tolist() == slots


def test_rowpar_layernorm_order():
    """index_cache_rowpar: LayerNorm statistics as a 64-lane xor-butterfly
    wave sum of per-lane pairs (k[2l] + k[2l+1]) in fp32, then the kernel's affine /
    RoPE / ue8m0 FP8 -> vs indexer_ref's golden (torch sums): every difference is a
    1-code E4M3 neighbour, <= 4 per row (the GPU check's bound), scale
    identical. (FMA contraction on GPU moves single roundings the same way.)"""

    # v [rows, 64] fp32, butterfly lane ^ 32, ^ 16, ... ^ 1 (kernel wave_sum)
    def wave_sum(
        v,
    ):
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
