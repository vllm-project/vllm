# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""The indexer's top-512 per decode row, straight into
K1's fixed-stride lists. Replaces vLLM's -1 fill, its top-k (vLLM's kernel
below 128 rows, aiter's above), the consumers' candidate mask (an identity
while a row has ≤ 2048 candidate blocks) and our translate.

One workgroup of T = 64·NW threads per row; the row's n = lens[row] logits
(fp32, n ≤ 4·NL·T) sit in VGPRs, element e = 4·(k·T + tid) + j for load k < NL,
j < 4. The result is the exact top-K (K = 512) under the order (value desc,
position asc), values compared as orderable u32 keys (so -0 < +0):

1. n ≤ K: every position is selected.
2. Else, digits: ⌊(vmax - v)·2047 / (vmax - vmin)⌋ clamped to 1..2047 per
   element (an affine map, then one rounding: monotone in v). One LDS
   histogram (ds_add_u32, 2048 bins), a block scan finds the threshold bin B:
   A elements lie in better bins, A < K ≤ A + c_B.
3. If c_B > 64 (a dense bin; ties), refine: the better bins become selected,
   the worse ones dead, and the live ones get new digits from their keys
   ((kmax - key) >> s, s so that the live key range fits 2048 bins), or from
   their positions once the keys are all equal. Repeat.
4. c_B ≤ 64: collect the bin's elements into an LDS list; wave 0 ranks them
   (64-bit composite key, position ascending on ties) while the other waves
   write the output.
5. Output: the elements better than B in position order (a packed block scan
   of per-thread counts gives every one its slot), then the bin's winners at
   A + rank. The copy-out maps each local position through the block-table
   row (prefetched into LDS): tk[row·512 + i] = bt[loc >> bs_log2]·bs + loc %
   bs, -1 past the count; out_local`[row][i]` = loc, -1 past it (vLLM's top-k
   buffer); tk_lens[row] = valid ? count : 0.

Element state d (a byte offset into the histogram, so ds_add needs no
address math): selected 4·lane, live 4·(64 + digit), dead 4·(2113 + lane).
Selected / dead are spread over the lanes: 64 lanes adding to one LDS word
cost 128 clk per instruction (measured), and a row's dead tail (positions
≥ n) is up to 2k elements.

Kernargs: see KA. Grid = rows, block = T.
"""

import regex as re

from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source

K = 512
PFLAG = "nt"  # the merge's poll loads: skip vL1, read the XCD's L2 (a row's CTAs share an XCD; sc1 loads
# miss the producers' plain stores while those sit dirty in L2)
CAP = 64  # the collected bin's capacity (one wave ranks it)
DIG0 = 64  # live digit d sits at 4·(DIG0 + d)
DEAD0 = DIG0 + 2049  # dead states 4·(DEAD0 + lane)

KA = KArgs(
    ("logits", "ptr"),
    ("lens", "ptr"),
    ("valid", "ptr"),
    ("req", "ptr"),
    ("bt", "ptr"),
    ("out_local", "ptr"),
    ("tk", "ptr"),
    ("tk_lens", "ptr"),
    ("lstride", "u32"),
    ("bt_stride", "u32"),
    ("local_stride", "u32"),
    ("bs_log2", "u32"),
)

# LDS map (bytes)
HIST_OFF = 12  # state d counts at HIST_OFF + d: digit 1 at byte 272 (b128-aligned)
HIST_END = 8736  # ≥ HIST_OFF + 4·(DEAD0 + 64), a multiple of 16
RED_MM = HIST_END  # 16 × 8: max / min
RED_S1 = RED_MM + 128  # 16 × 8 dwords: scans
RED_S2 = RED_S1 + 512
RES = RED_S2 + 512  # B4, A, c_B
CNT = RES + 16  # collect counter
RED_K = CNT + 16  # 2 × 16 × 8: slow-path key / position max, min
CAND = RED_K + 256  # CAP × (value, position)
OUT = CAND + 8 * CAP  # K local positions in output order, then 64 dummy slots
BT = OUT + 4 * (K + 64)  # the block-table row
assert HIST_END >= HIST_OFF + 4 * (DEAD0 + 64) and HIST_END % 16 == 0 and CAND % 8 == 0


def source(
    nw: int = 8,
    nl: int = 5,
    name: str | None = None,
    bt_max: int = 1024,
    row: str = "s2",
    lg_lds: int | None = None,
    lg_sel: str | None = None,
    lg_flag: str = "",
    body_only: bool = False,
    pre: bool = False,
    pfx: str = "",
    cand_out: bool = False,
    cand_in: str | None = None,
    bt_n: str | None = None,
    bt_off: int = BT,
    exit_to: str | None = None,
    filt: bool = False,
    caps: int = 0,
    fallback: str = "",
    cand_range: str | None = None,
    tag: str = "",
    front: dict | None = None,
    pos_lds: int | None = None,
    tau_slot: bool = False,
    k: int = K,
    blk_out: bool = False,
    remap: int | None = None,
):
    """For embedding (idx_logits.py): row = the SGPR holding the row (kept intact); lg_lds = the LDS byte offset
    of the row's logits (element e at lg_lds + 4e), used when lg_sel (an SGPR) is 0 or always if lg_sel is
    None; lg_flag = cache flag of the global logits loads (e.g. "nt"); body_only: return the code lines (they
    end the program) instead of a kernel; pre: the caller has loaded the kernargs into s[4:23] and n, req and
    the valid word into s25, s26, s27.
    Two-level top-k (idx_logits wide rows): pfx makes the labels unique (two bodies in one kernel); exit_to:
    branch there instead of s_endpgm. cand_out: no block table; the copy-out writes the selected elements'
    values (from LDS lg_lds) to s[14:15] + 4i and their positions s22 + e to s[16:17] + 4i, i < min(n, K).
    cand_in = an SGPR pair holding a candidates row: 8 B slots (value, position | tag << 24), polled with sc1
    loads until every slot < cand_range carries s[tag] (= tag << 24): the producers store them without a
    release; slots holding NaN are dead, and n (s25) is the row's real n (count and n ≤ K logic). The
    positions stay in VGPRs; ties at the cut are ranked by them, so the slots may come in any order.
    bt_n = the SGPR with the row's real n (for the block-table row, else n); bt_off = the block-table row's
    LDS offset.
    caps (with cand_out): the output is a segment of caps slots at s[14:15] + 8 i, (value, position | s[tag])
    for i < count as above, the rest (NaN, 0xffffff | s[tag]) (the merge reads every segment: NaN = no
    element). cand_range (with cand_in) = an SGPR with the number of slots to read.
    tau_slot (cand_out): wave 0 also stores (τ key, tag) at s[16:17] (τ = the filter's, 0 for the exact body).
    front = dict(nseg=SGPR with S, cmax, clds, to_c, to_b) (with cand_in): only the merge's front end: poll the S
    τ slots (cand_in + 8 (range + s)) until all carry the tag, τ_row = their max (a lower bound of the row's
    K-th key), poll the candidates, compact the live ones with key ≥ τ_row into LDS (values at clds, positions
    at clds + 4 cmax), n (s25) = their count; → to_c if n ≤ cmax, else to_b (the full merge, which re-polls).
    pos_lds (with lg_lds): the elements' positions are at pos_lds + 4e (a compacted candidate list).
    k: the top-k's K (the plain path; filt / cand_in / front stay at K = 512). blk_out: the copy-out only writes
    the positions to out_local`[row][i]` (-1 past the count): no block table, tk or tk_lens (the candidate
    source's top-2048 blocks, its row = the block scores, n = the block count); the bt_stride kernarg = the live
    rows: rows past it (graph padding) output -1 only (their stale or zero scores: an all-equal row is the radix
    select's worst case, every ds_add on one LDS word).
    filt (needs cand_out, caps, lg_lds): no exact top-K; the caller has filled the histogram (digit
    2048 - (key >> 21) per live element, below); the threshold bin of that and, if more than caps elements
    are at or above it, of a second level (digit 2048 - ((key >> 10) & 2047) inside it) gives a key τ with
    K ≤ #{key ≥ τ} ≤ caps; those elements are the candidates. More than caps at both levels (ties) →
    branch to `fallback` (the exact body)."""
    T = 64 * nw
    EPT = 4 * nl
    KT = k  # (k is a loop variable below)
    assert remap is None or cand_out
    # remap (cand_out; idx_logits gather): element e's position = 8 · LDS[remap + 16 (e >> 3)] + (e & 7) (the
    # slice's block table) instead of s22 + e
    assert KT == K or not (filt or cand_in or front or cand_out), KT
    assert not blk_out or not (cand_out or cand_in or front or filt)
    out_end = OUT + 4 * (KT + 64)  # OUT: k positions, then 64 dummy slots
    bt_off = max(bt_off, out_end) if bt_off == BT else bt_off
    assert nw in (4, 8, 16), nw
    G = 2048 // T  # bins per thread in the scan
    name = name or (
        f"idx_topk_w{nw}_l{nl}"
        + (f"_k{KT}" if KT != K else "")
        + ("_blk" if blk_out else "")
    )
    ka = KA
    lds = (
        BT
        if cand_out
        else max(out_end, BT)
        if blk_out
        else max(BT, bt_off + 4 * bt_max)
    )
    assert not filt or (cand_out and lg_lds is not None and caps and fallback)
    assert not cand_out or (caps >= K and tag)
    assert not cand_in or (tag and cand_range)
    assert not front or cand_in
    FB = (
        10 if KT < 1024 else 12
    )  # packed output counts: fields of FB bits (a load's count ≤ k)
    FPD = 32 // FB
    NQ = -(-nl // FPD)
    assert nl <= 18 and NQ <= 6 and KT < 4096

    # ---- registers: v0 tid, v1 lane, v2 4·tid, v3 one, v4 zero, v5 16·tid, v6 nan, v7 dead state, v8 4·tid,
    # v9 selected state, v10 OUT dummy slot, v12..15 zeros; VAL, ST, then temps
    VAL = 16
    ST = VAL + EPT
    POS = ST + EPT  # cand_in / pos_lds: the elements' positions
    has_pos = bool(cand_in) or pos_lds is not None
    TMP = POS + (EPT if has_pos and not front else 0)
    t = [TMP + i for i in range(48)]
    nv = TMP + 48
    c = []
    A = c.append
    lbl = [0]

    def L(p="L"):
        lbl[0] += 1
        return f".{p}{lbl[0]}"

    def nop(n=1):
        A(f"  s_nop {n}")

    # SGPRs: s[4:19] 8 pointers, s20 lstride, s21 bt_stride, s22 local_stride, s23 bs_log2,
    # s24 row·4, s25 n, s26 req, s27 valid word, s28 wave, s[32:35] logits rsrc, s[36:39] bt rsrc,
    # s40 B4, s41 A (this round), s42 c_B, s43 need, s44 A total, s45 / s46 reduction results, s47 cnt,
    # s48.. temps
    S_N, S_REQ, S_W = 25, 26, 28
    S_B4, S_A, S_CB, S_NEED, S_AT, S_CNT = 40, 41, 42, 43, 44, 47

    def wave_reduce(ops, regs, tmps, steps=6):
        """ops[i] (e.g. max_f32) over the lanes of regs[i]; every lane gets the result (fixed order). steps < 6:
        only the first ones (e.g. 3 = within 8 consecutive lanes)."""
        ctls = ("quad_perm:[1,0,3,2]", "quad_perm:[2,3,0,1]", "row_ror:4", "row_ror:8")
        for ctl in ctls[:steps]:
            for r, tp in zip(regs, tmps):
                A(f"  v_mov_b32 v{tp}, v{r}")
            nop(1)
            for op, r, tp in zip(ops, regs, tmps):
                A(f"  v_{op}_dpp v{r}, v{tp}, v{r} {ctl} row_mask:0xf bank_mask:0xf")
        for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32")[
            : max(0, steps - 4)
        ]:
            for r, tp in zip(regs, tmps):
                A(f"  v_mov_b32 v{tp}, v{r}")
            nop(1)
            for r, tp in zip(regs, tmps):
                A(f"  {sw} v{r}, v{tp}")
            nop(4)
            for op, r, tp in zip(ops, regs, tmps):
                A(f"  v_{op} v{r}, v{r}, v{tp}")

    def wave_scan(regs):
        """Inclusive u32 scan over the 64 lanes (lane order)."""
        for ctl in (
            "row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:0",
            "row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:0",
            "row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0",
            "row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0",
            "row_bcast:15 row_mask:0xa bank_mask:0xf",
            "row_bcast:31 row_mask:0xc bank_mask:0xf",
        ):
            nop(1)
            for r in regs:
                A(f"  v_add_u32_dpp v{r}, v{r}, v{r} {ctl}")
        nop(1)

    def block_scan(regs, red, tmps, totals):
        """Inclusive u32 scan over the T threads (tid order) of each reg; totals[i] (SGPR) gets reg i's block
        total. tmps: 2·len(regs) VGPRs. Uses LDS red[0 : 4·len(regs)·NW]; has one barrier."""
        n = len(regs)
        wave_scan(regs)
        A("  s_lshl_b64 s[50:51], 1, 63")
        A("  s_mov_b64 exec, s[50:51]")
        A(f"  v_mov_b32 v{t[47]}, s{S_W}")
        A(f"  v_mul_u32_u24 v{t[47]}, {4 * n}, v{t[47]}")
        for i, r in enumerate(regs):
            A(f"  ds_write_b32 v{t[47]}, v{r} offset:{red + 4 * i}")
        A("  s_mov_b64 exec, -1")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        A(f"  v_mul_u32_u24 v{t[47]}, {4 * n}, v1")
        A(f"  v_cmp_gt_u32 vcc, {nw}, v1")
        for i in range(n):
            A(f"  v_mov_b32 v{tmps[i]}, 0")
        A("  s_and_saveexec_b64 s[50:51], vcc")
        for i in range(n):
            A(f"  ds_read_b32 v{tmps[i]}, v{t[47]} offset:{red + 4 * i}")
        A("  s_mov_b64 exec, s[50:51]")
        A("  s_waitcnt lgkmcnt(0)")
        for i in range(n):
            A(f"  v_mov_b32 v{tmps[n + i]}, v{tmps[i]}")
        wave_scan(tmps[:n])
        for i, r in enumerate(regs):
            A(f"  v_readlane_b32 s52, v{tmps[i]}, s{S_W}")
            A(f"  v_readlane_b32 s53, v{tmps[n + i]}, s{S_W}")
            A(f"  v_readlane_b32 s{totals[i]}, v{tmps[i]}, {nw - 1}")
            A("  s_sub_u32 s52, s52, s53")
            A(f"  v_add_u32 v{r}, s52, v{r}")

    def cross_wave(ops, regs, region, tmps):
        """Regs (already wave-reduced, uniform per wave) → block result in every lane: wave w's value at
        region + 8 w (+4), then a reduce over the NW values (replicated mod NW across the lanes)."""
        A(f"  v_mov_b32 v{t[44]}, s{S_W}")
        A(f"  v_lshlrev_b32 v{t[44]}, 3, v{t[44]}")
        A("  s_mov_b64 exec, 1")
        for i, r in enumerate(regs):
            A(f"  ds_write_b32 v{t[44]}, v{r} offset:{region + 4 * i}")
        A("  s_mov_b64 exec, -1")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        A(f"  v_and_b32 v{t[44]}, {nw - 1}, v1")
        A(f"  v_lshlrev_b32 v{t[44]}, 3, v{t[44]}")
        for i, r in enumerate(regs):
            A(f"  ds_read_b32 v{r}, v{t[44]} offset:{region + 4 * i}")
        A("  s_waitcnt lgkmcnt(0)")
        wave_reduce(ops, regs, tmps, steps=nw.bit_length() - 1)

    def remap_pos(r, tmp):
        """v[r] = element e → its position through the remap table (waits for the LDS)."""
        A(f"  v_lshrrev_b32 v{tmp}, 3, v{r}")
        A(f"  v_lshlrev_b32 v{tmp}, 4, v{tmp}")
        A(f"  v_add_u32 v{tmp}, {remap}, v{tmp}")
        A(f"  ds_read_b32 v{tmp}, v{tmp}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_and_b32 v{r}, 7, v{r}")
        A(f"  v_lshl_or_b32 v{r}, v{tmp}, 3, v{r}")

    def pos_of(k, j, dst):
        if has_pos:
            A(f"  v_and_b32 v{dst}, 0xffffff, v{POS + 4 * k + j}")
        else:
            A(f"  v_add_u32 v{dst}, {4 * k * T + j}, v2")

    real_pos = pos_of

    def key_of(v, dst, tmp):
        """Dst ← orderable u32 key of the f32 in v."""
        A(f"  v_ashrrev_i32 v{tmp}, 31, v{v}")
        A(f"  v_or_b32 v{tmp}, 0x80000000, v{tmp}")
        A(f"  v_xor_b32 v{dst}, v{tmp}, v{v}")

    def zero_hist():
        for r in range(12, 16):
            A(f"  v_mov_b32 v{r}, 0")
        for r in range(-(-HIST_END // (16 * T))):
            lo = 16 * T * r
            if lo + 16 * T <= HIST_END:
                A(f"  ds_write_b128 v5, v[12:15] offset:{lo}")
            else:
                A(f"  v_cmp_gt_u32 vcc, {(HIST_END - lo) // 16}, v0")
                A("  s_and_saveexec_b64 s[50:51], vcc")
                A(f"  ds_write_b128 v5, v[12:15] offset:{lo}")
                A("  s_mov_b64 exec, s[50:51]")

    def histogram():
        for i in range(EPT):
            A(f"  ds_add_u32 v{ST + i}, v3 offset:{HIST_OFF}")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")

    def find_bin():
        """Digits 1..2048 → the threshold bin for s[need]: S_B4 = its state value, S_A = count in better bins,
        S_CB = c_B."""
        h = t[0:G]
        base = HIST_OFF + 4 * (DIG0 + 1)
        A(f"  v_lshlrev_b32 v{t[40]}, {2 + (G.bit_length() - 1)}, v0")
        if G >= 4:
            for q in range(G // 4):
                A(
                    f"  ds_read_b128 v[{h[4 * q]}:{h[4 * q] + 3}], v{t[40]} offset:{base + 16 * q}"
                )
        else:
            A(f"  ds_read_b64 v[{h[0]}:{h[1]}], v{t[40]} offset:{base}")
        A("  s_waitcnt lgkmcnt(0)")
        ct = t[8]
        A(f"  v_add_u32 v{ct}, v{h[0]}, v{h[1]}")
        for g in range(2, G):
            A(f"  v_add_u32 v{ct}, v{ct}, v{h[g]}")
        A(f"  v_mov_b32 v{t[9]}, v{ct}")
        block_scan([ct], RED_S1, [t[10], t[11]], [54])
        cum, bB, bA, bC = t[12], t[13], t[14], t[15]
        A(f"  v_sub_u32 v{cum}, v{ct}, v{t[9]}")
        A(f"  v_mov_b32 v{bB}, 0")
        A(f"  v_lshlrev_b32 v{t[17]}, {G.bit_length() - 1}, v0")
        for g in range(G):
            A(f"  v_add_u32 v{t[16]}, v{cum}, v{h[g]}")
            A(f"  v_cmp_gt_u32 s[56:57], s{S_NEED}, v{cum}")  # cum < need
            A(f"  v_cmp_le_u32 s[58:59], s{S_NEED}, v{t[16]}")  # need ≤ cum + h
            A("  s_and_b64 s[56:57], s[56:57], s[58:59]")
            A(f"  v_add_u32 v{t[18]}, {1 + g}, v{t[17]}")
            A(f"  v_cndmask_b32_e64 v{bB}, v{bB}, v{t[18]}, s[56:57]")
            A(f"  v_cndmask_b32_e64 v{bA}, v{bA}, v{cum}, s[56:57]")
            A(f"  v_cndmask_b32_e64 v{bC}, v{bC}, v{h[g]}, s[56:57]")
            A(f"  v_mov_b32 v{cum}, v{t[16]}")
        A(f"  v_cmp_ne_u32 vcc, 0, v{bB}")
        A("  s_and_saveexec_b64 s[50:51], vcc")
        A(f"  v_add_lshl_u32 v{bB}, v{bB}, {DIG0}, 2")
        A(f"  ds_write_b32 v4, v{bB} offset:{RES}")
        A(f"  ds_write_b32 v4, v{bA} offset:{RES + 4}")
        A(f"  ds_write_b32 v4, v{bC} offset:{RES + 8}")
        A("  s_mov_b64 exec, s[50:51]")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        A(f"  ds_read_b32 v{t[20]}, v4 offset:{RES}")
        A(f"  ds_read_b32 v{t[21]}, v4 offset:{RES + 4}")
        A(f"  ds_read_b32 v{t[22]}, v4 offset:{RES + 8}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_readfirstlane_b32 s{S_B4}, v{t[20]}")
        A(f"  v_readfirstlane_b32 s{S_A}, v{t[21]}")
        A(f"  v_readfirstlane_b32 s{S_CB}, v{t[22]}")

    def filter_body():
        """filt: τ from the caller's histogram (+ a second level), then the elements with key ≥ τ (ST holds the
        keys, 0 = dead) → LDS staging at lg_lds (values) / lg_lds + 4 caps (positions) → the segment."""
        tau = 60  # SGPR: τ
        A(f"  s_mov_b32 s{tau}, 1")
        A(f"  s_cmp_le_u32 s{S_N}, {K}")
        A("  s_cbranch_scc1 .Lfcol")  # n ≤ K: every live element
        A(f"  s_mov_b32 s{S_NEED}, {K}")
        find_bin()
        # bin b = 2048 - digit: keys [b << 21, (b + 1) << 21)
        A(f"  s_lshr_b32 s61, s{S_B4}, 2")
        A(f"  s_sub_u32 s61, {2048 + DIG0}, s61")
        A(f"  s_lshl_b32 s{tau}, s61, 21")
        A(f"  s_add_u32 s52, s{S_A}, s{S_CB}")
        A(f"  s_cmp_le_u32 s52, {caps}")
        A("  s_cbranch_scc1 .Lfcol")
        A("  s_cmp_eq_u32 s61, 0")  # (bin 0 holds the dead keys too: never for logits)
        A("  s_cbranch_scc1 @@FB@@")
        # level 2: digits 2048 - ((key >> 10) & 2047) of the keys in bin b
        A(f"  s_mov_b32 s{S_AT}, s{S_A}")
        A(f"  s_sub_u32 s{S_NEED}, {K}, s{S_A}")
        zero_hist()
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        HB = HIST_OFF + 4 * (DIG0 + 2048)
        for i in range(EPT):
            A(f"  v_lshrrev_b32 v{t[4]}, 21, v{ST + i}")
            A(f"  v_cmp_eq_u32 vcc, s61, v{t[4]}")
            A(f"  v_bfe_u32 v{t[4]}, v{ST + i}, 10, 11")
            A(f"  v_lshlrev_b32 v{t[4]}, 2, v{t[4]}")
            A(f"  v_sub_u32 v{t[4]}, {HB}, v{t[4]}")
            A("  s_and_saveexec_b64 s[50:51], vcc")
            A(f"  ds_add_u32 v{t[4]}, v3")
            A("  s_mov_b64 exec, s[50:51]")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        find_bin()
        A(f"  s_lshr_b32 s52, s{S_B4}, 2")
        A(f"  s_sub_u32 s52, {2048 + DIG0}, s52")
        A("  s_lshl_b32 s52, s52, 10")
        A(f"  s_or_b32 s{tau}, s{tau}, s52")
        A(f"  s_add_u32 s52, s{S_AT}, s{S_A}")
        A(f"  s_add_u32 s52, s52, s{S_CB}")
        A(f"  s_cmp_le_u32 s52, {caps}")
        A("  s_cbranch_scc0 @@FB@@")
        # collect: per-thread counts → block scan → slots; survivors → LDS staging
        A(".Lfcol:")
        cnt = t[10]
        A(f"  v_mov_b32 v{cnt}, 0")
        for i in range(EPT):
            A(f"  v_cmp_le_u32 vcc, s{tau}, v{ST + i}")
            A(f"  v_addc_co_u32 v{cnt}, vcc, 0, v{cnt}, vcc")
        A(f"  v_mov_b32 v{t[11]}, v{cnt}")
        block_scan(
            [cnt], RED_S1, [t[12], t[13]], [S_CNT]
        )  # its barrier: every wave has read the logits
        A(f"  v_sub_u32 v{cnt}, v{cnt}, v{t[11]}")  # exclusive
        A(f"  v_lshlrev_b32 v{cnt}, 2, v{cnt}")
        for k in range(nl):
            for j in range(4):
                i = 4 * k + j
                A(f"  v_cmp_le_u32 vcc, s{tau}, v{ST + i}")
                A("  s_and_saveexec_b64 s[50:51], vcc")
                pos_of(k, j, t[5])
                if remap is None:
                    A(f"  v_add_u32 v{t[5]}, s22, v{t[5]}")
                else:
                    remap_pos(t[5], t[6])
                A(f"  ds_write_b32 v{cnt}, v{VAL + i} offset:{lg_lds}")
                A(f"  ds_write_b32 v{cnt}, v{t[5]} offset:{lg_lds + 4 * caps}")
                A(f"  v_add_u32 v{cnt}, 4, v{cnt}")
                A("  s_mov_b64 exec, s[50:51]")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        segment(
            lambda sl, dv, dp: (
                A(f"  v_lshlrev_b32 v{t[3]}, 2, v{sl}"),
                A(f"  ds_read_b32 v{dv}, v{t[3]} offset:{lg_lds}"),
                A(f"  ds_read_b32 v{dp}, v{t[3]} offset:{lg_lds + 4 * caps}"),
                A("  s_waitcnt lgkmcnt(0)"),
            )
        )
        write_tau("s60")
        A("  s_branch @@EXIT@@")

    def write_tau(src):
        """tau_slot: wave 0 lane 0 stores (τ, tag) at s[16:17]."""
        if not tau_slot:
            return
        skip = L("Ltau")
        pa = t[6] if t[6] % 2 == 0 else t[7]
        A("  s_cmp_eq_u32 s28, 0")
        A(f"  s_cbranch_scc0 {skip}")
        A("  s_mov_b64 exec, 1")
        A(f"  v_mov_b32 v{pa}, {src}")
        A(f"  v_mov_b32 v{pa + 1}, {tag}")
        A(f"  global_store_dwordx2 v4, v[{pa}:{pa + 1}], s[16:17]")
        A("  s_mov_b64 exec, -1")
        A(f"{skip}:")

    def segment(fetch):
        """The caps-slot output segment: slot sl < s[S_CNT] → fetch(sl, value VGPR, position VGPR), stored with
        the tag as one 8 B slot at s[14:15] + 8 sl; the rest (NaN, 0xffffff | tag)."""
        dv, dp = (
            (t[6], t[7]) if t[6] % 2 == 0 else (t[7], t[8])
        )  # an aligned pair (dwordx2)
        for r in range(-(-caps // T)):
            sl_ = t[0]
            A(f"  v_add_u32 v{sl_}, {r * T}, v0")
            if (r + 1) * T > caps:
                A(f"  v_cmp_gt_u32 vcc, {caps}, v{sl_}")
                A("  s_and_saveexec_b64 s[52:53], vcc")
            A(f"  v_mov_b32 v{dv}, v6")
            A(f"  v_mov_b32 v{dp}, 0xffffff")
            A(f"  v_cmp_gt_u32 vcc, s{S_CNT}, v{sl_}")
            A("  s_and_saveexec_b64 s[50:51], vcc")
            fetch(sl_, dv, dp)
            A("  s_mov_b64 exec, s[50:51]")
            A(f"  v_lshlrev_b32 v{t[3]}, 3, v{sl_}")
            A(f"  v_or_b32 v{dp}, {tag}, v{dp}")
            A(f"  global_store_dwordx2 v{t[3]}, v[{dv}:{dp}], s[14:15]")
            if (r + 1) * T > caps:
                A("  s_mov_b64 exec, s[52:53]")

    def red2_u32(vmax_r, vmin_r, region):
        """Block max / min (u32) → s45 / s46."""
        wave_reduce(["max_u32", "min_u32"], [vmax_r, vmin_r], [t[30], t[31]])
        cross_wave(["max_u32", "min_u32"], [vmax_r, vmin_r], region, [t[30], t[31]])
        A(f"  v_readfirstlane_b32 s45, v{vmax_r}")
        A(f"  v_readfirstlane_b32 s46, v{vmin_r}")

    # ---------------------------------------------------------------- prologue
    if not pre:
        A("  s_load_dwordx16 s[4:19], s[0:1], 0")
        A(f"  s_load_dwordx4 s[20:23], s[0:1], {KA.off['lstride']:#x}")
    A("  v_and_b32 v1, 63, v0")
    A("  v_lshlrev_b32 v2, 2, v0")
    A("  v_mov_b32 v3, 1")
    A("  v_mov_b32 v4, 0")
    A("  v_lshlrev_b32 v5, 4, v0")
    A("  v_mov_b32 v6, 0x7fc00000")
    A("  v_mov_b32 v8, v2")
    A("  v_lshlrev_b32 v9, 2, v1")  # selected: 4·lane
    A(f"  v_add_u32 v7, {4 * DEAD0}, v9")  # dead: 4·(DEAD0 + lane)
    A(f"  v_add_u32 v10, {4 * KT}, v9")  # OUT dummy slot
    A("  v_readfirstlane_b32 s28, v0")
    A("  s_lshr_b32 s28, s28, 6")  # wave
    A(f"  s_lshl_b32 s24, {row}, 2")
    A(f"  s_and_b32 s29, {row}, -4")
    if not pre:
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  s_load_dword s{S_N}, s[6:7], s24")
        A(f"  s_load_dword s{S_REQ}, s[10:11], s24")
        A("  s_load_dword s27, s[8:9], s29")
    # the logits row: no need to know n (num_records = the row; positions ≥ n are masked below)
    lgl = L("Llg")
    lgd = L("Llgd")

    def poll_slots(dst):
        """Poll the (value, position | tag) slots into v[dst(k) ...] until every live one (< range) carries this
        call's tag (element e = 4 (kT + tid) + j: value dst(k) + 2j, position | tag dst(k) + 2j + 1)."""
        poll, pdone = L("Lpoll"), L("Lpold")
        A("  v_lshlrev_b32 v13, 5, v0")
        A(f"{poll}:")
        for k in range(nl):
            A(f"  v_add_u32 v12, {32 * k * T}, v13")
            A(f"  global_load_dwordx4 v[{dst(k)}:{dst(k) + 3}], v12, {cand_in} {PFLAG}")
            A(
                f"  global_load_dwordx4 v[{dst(k) + 4}:{dst(k) + 7}], v12, {cand_in} offset:16 {PFLAG}"
            )
        A("  s_waitcnt vmcnt(0)")
        A("  s_mov_b64 s[56:57], 0")
        for k in range(nl):
            for j in range(4):
                A(f"  s_sub_u32 s49, {cand_range}, {4 * k * T + j}")
                A("  v_cmp_gt_i32 vcc, s49, v2")
                A(f"  v_and_b32 v14, 0xff000000, v{dst(k) + 2 * j + 1}")
                A(f"  v_cmp_ne_u32 s[58:59], {tag}, v14")
                A("  s_and_b64 s[58:59], s[58:59], vcc")
                A("  s_or_b64 s[56:57], s[56:57], s[58:59]")
        A("  s_cmp_eq_u64 s[56:57], 0")
        A(f"  s_cbranch_scc1 {pdone}")
        A("  s_sleep 4")
        A(f"  s_branch {poll}")
        A(f"{pdone}:")

    def front_body():
        """The merge's front end (see `front`)."""
        f = front
        # the S τ slots: every wave polls them (8 B per lane, lanes < S), backing off
        lo, hi = re.findall(r"\d+", cand_in)
        A(f"  s_lshl_b32 s52, {cand_range}, 3")
        A(f"  s_add_u32 s52, s{lo}, s52")
        A(f"  s_addc_u32 s53, s{hi}, 0")
        A("  v_lshlrev_b32 v12, 3, v1")
        pa = t[0] if t[0] % 2 == 0 else t[1]
        probe, pdone = L("Lprobe"), L("Lprobd")
        A(f"{probe}:")
        A(f"  v_mov_b32 v{pa}, 0")
        A(f"  v_mov_b32 v{pa + 1}, {tag}")
        A(f"  v_cmp_gt_u32 vcc, {f['nseg']}, v1")
        A("  s_and_saveexec_b64 s[50:51], vcc")
        A(f"  global_load_dwordx2 v[{pa}:{pa + 1}], v12, s[52:53] {PFLAG}")
        A("  s_waitcnt vmcnt(0)")
        A("  s_mov_b64 exec, s[50:51]")
        A(f"  v_and_b32 v{pa + 1}, 0xff000000, v{pa + 1}")
        A(f"  v_cmp_ne_u32 vcc, {tag}, v{pa + 1}")
        A(f"  s_cbranch_vccz {pdone}")
        A("  s_sleep 8")
        A(f"  s_branch {probe}")
        A(f"{pdone}:")
        wave_reduce(["max_u32"], [pa], [t[3]])
        A(f"  v_readfirstlane_b32 s60, v{pa}")  # τ_row
        dst = lambda k: VAL + 8 * k  # (VAL and ST are free in the front end)
        poll_slots(dst)
        # live: e < range, not NaN, key ≥ τ_row → count, slots, LDS
        cnt, own = t[2], t[3]

        def live(k, j):
            v = dst(k) + 2 * j
            A(f"  s_sub_u32 s49, {cand_range}, {4 * k * T + j}")
            A("  v_cmp_gt_i32 vcc, s49, v2")
            A(f"  v_cmp_o_f32 s[56:57], v{v}, v{v}")
            A("  s_and_b64 vcc, vcc, s[56:57]")
            key_of(v, t[4], t[5])
            A(f"  v_cmp_le_u32 s[56:57], s60, v{t[4]}")
            A("  s_and_b64 vcc, vcc, s[56:57]")
            return v

        A(f"  v_mov_b32 v{cnt}, 0")
        for k in range(nl):
            for j in range(4):
                live(k, j)
                A(f"  v_addc_co_u32 v{cnt}, vcc, 0, v{cnt}, vcc")
        A(f"  v_mov_b32 v{own}, v{cnt}")
        block_scan([cnt], RED_S1, [t[6], t[7]], [S_CNT])
        A(f"  v_sub_u32 v{cnt}, v{cnt}, v{own}")
        A(f"  v_lshlrev_b32 v{cnt}, 2, v{cnt}")
        for k in range(nl):
            for j in range(4):
                v = live(k, j)
                A("  s_and_saveexec_b64 s[50:51], vcc")
                A(f"  v_and_b32 v{t[5]}, 0xffffff, v{v + 1}")
                A(f"  ds_write_b32 v{cnt}, v{v} offset:{f['clds']}")
                A(f"  ds_write_b32 v{cnt}, v{t[5]} offset:{f['clds'] + 4 * f['cmax']}")
                A(f"  v_add_u32 v{cnt}, 4, v{cnt}")
                A("  s_mov_b64 exec, s[50:51]")
        A(f"  s_mov_b32 s{S_N}, s{S_CNT}")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        A(f"  s_cmp_le_u32 s{S_CNT}, {f['cmax']}")
        A("  s_cbranch_scc1 @@FB@@")  # → to_c
        A("  s_branch @@EXIT@@")  # → to_b

    if front:
        front_body()
        return _finish(
            c, [], pfx, front["to_b"], nv, lds, EPT * T, bt_max, front["to_c"]
        )
    if cand_in and nl <= 5:
        poll_slots(lambda k: t[8 + 8 * k])
        for k in range(nl):
            for j in range(4):
                A(f"  v_mov_b32 v{VAL + 4 * k + j}, v{t[8 + 8 * k] + 2 * j}")
                A(f"  v_mov_b32 v{POS + 4 * k + j}, v{t[8 + 8 * k] + 2 * j + 1}")
    elif cand_in:
        # more slots than the temps hold: poll into VAL / ST (8 per load, below POS = VAL + 8 nl), then move in
        # place in (k, j) order: every source still to be read sits at or above the current destination
        poll_slots(lambda k: VAL + 8 * k)
        for k in range(nl):
            for j in range(4):
                A(f"  v_mov_b32 v{POS + 4 * k + j}, v{VAL + 8 * k + 2 * j + 1}")
                if 4 * k + j != 8 * k + 2 * j:
                    A(f"  v_mov_b32 v{VAL + 4 * k + j}, v{VAL + 8 * k + 2 * j}")
    elif lg_lds is not None:
        assert lg_lds % 16 == 0 and 16 * (nl - 1) * T < 65536
        if lg_sel is not None:
            A(f"  s_cmp_eq_u32 {lg_sel}, 0")
            A(f"  s_cbranch_scc0 {lgl}")
        A(f"  v_add_u32 v{t[46]}, {lg_lds}, v5")
        for k in range(nl):
            A(
                f"  ds_read_b128 v[{VAL + 4 * k}:{VAL + 4 * k + 3}], v{t[46]} offset:{16 * k * T}"
            )
        if pos_lds is not None:
            A(f"  v_add_u32 v{t[45]}, {pos_lds}, v5")
            for k in range(nl):
                A(
                    f"  ds_read_b128 v[{POS + 4 * k}:{POS + 4 * k + 3}], v{t[45]} offset:{16 * k * T}"
                )
        A(f"  s_branch {lgd}")
    A(f"{lgl}:")
    if (lg_lds is None or lg_sel is not None) and not cand_in:
        A(f"  s_mul_i32 s30, {row}, s20")
        A(f"  s_mul_hi_u32 s31, {row}, s20")
        A("  s_add_u32 s32, s4, s30")
        A("  s_addc_u32 s33, s5, s31")
        A("  s_and_b32 s33, s33, 0xffff")
        A("  s_mov_b32 s34, s20")
        A("  s_mov_b32 s35, 0x00020000")
        fl = f" {lg_flag}" if lg_flag else ""
        for k in range(nl):
            A(f"  s_mov_b32 s{48 + (k & 7)}, {16 * k * T}")
            A(
                f"  buffer_load_dwordx4 v[{VAL + 4 * k}:{VAL + 4 * k + 3}], v5, s[32:35], s{48 + (k & 7)} offen{fl}"
            )
    A(f"{lgd}:")
    if not filt:
        zero_hist()
    A(f"  ds_write_b32 v4, v4 offset:{CNT}")
    A("  s_waitcnt lgkmcnt(0)")
    if blk_out:  # rows ≥ bt_stride (the live rows) select nothing
        A(f"  s_cmp_ge_u32 {row}, s21")
        A(f"  s_cselect_b32 s{S_N}, 0, s{S_N}")
    # block-table row → LDS (nbt = ceil(n / bs) entries)
    nbtl = 0 if cand_out or blk_out else -(-bt_max // T)
    if nbtl:
        A(f"  s_mul_i32 s30, s{S_REQ}, s21")
        A(f"  s_mul_hi_u32 s31, s{S_REQ}, s21")
        A("  s_add_u32 s36, s12, s30")
        A("  s_addc_u32 s37, s13, s31")
        A("  s_and_b32 s37, s37, 0xffff")
        A("  s_lshl_b32 s30, 1, s23")
        A(f"  s_add_u32 s30, {bt_n or f's{S_N}'}, s30")
        A("  s_sub_u32 s30, s30, 1")
        A("  s_lshr_b32 s30, s30, s23")
        A("  s_lshl_b32 s38, s30, 2")  # num_records = nbt·4
        A("  s_mov_b32 s39, 0x00020000")
    for r in range(nbtl):
        if 4 * T * r < 4096:
            A(f"  buffer_load_dword v{t[r]}, v8, s[36:39], 0 offen offset:{4 * T * r}")
        else:
            A(f"  s_mov_b32 s49, {4 * T * r}")
            A(f"  buffer_load_dword v{t[r]}, v8, s[36:39], s49 offen")
    # valid flag: byte (row & 3) of the word
    A(f"  s_and_b32 s29, {row}, 3")
    A("  s_lshl_b32 s29, s29, 3")
    A("  s_lshr_b32 s27, s27, s29")
    A("  s_and_b32 s27, s27, 0xff")
    A(f"  s_min_u32 s{S_CNT}, s{S_N}, {KT}")
    A(f"  s_mov_b32 s{S_AT}, 0")
    A(f"  s_mov_b32 s{S_NEED}, {KT}")
    A(f"  s_mov_b32 s{S_CB}, 0")
    A("  s_waitcnt vmcnt(0)")
    for r in range(nbtl):
        A(f"  ds_write_b32 v8, v{t[r]} offset:{bt_off + 4 * T * r}")
    # validity: element e = 4 tid + 4kT + j < n; dead → NaN value, dead state (filt: the key, dead → 0)
    for k in range(nl):
        for j in range(4):
            i = 4 * k + j
            A(f"  s_sub_u32 s49, {cand_range or f's{S_N}'}, {4 * k * T + j}")
            A("  v_cmp_gt_i32 vcc, s49, v2")
            if cand_range:
                A(f"  v_cmp_o_f32 s[56:57], v{VAL + i}, v{VAL + i}")
                A("  s_and_b64 vcc, vcc, s[56:57]")
            if filt:
                key_of(VAL + i, ST + i, t[0])
                A(f"  v_cndmask_b32_e64 v{ST + i}, 0, v{ST + i}, vcc")
                continue
            A(f"  v_cndmask_b32_e64 v{VAL + i}, v6, v{VAL + i}, vcc")
            A(f"  v_cndmask_b32_e64 v{ST + i}, v7, v9, vcc")
    if filt:
        filter_body()
        return _finish(c, [], pfx, exit_to, nv, lds, EPT * T, bt_max, fallback)
    A(f"  s_cmp_le_u32 s{S_N}, {KT}")
    A("  s_cbranch_scc0 .Lbig")
    # n ≤ K: every live element is selected (state 4·lane < 256 = B4)
    A(f"  s_mov_b32 s{S_B4}, 256")
    A(f"  s_mov_b32 s{S_AT}, s{S_N}")
    A("  s_branch .Lout")

    # ---------------------------------------------------------------- n > K: value digits
    A(".Lbig:")
    mx, mn = t[0], t[1]
    if blk_out:
        # the block scores hold one +inf (the newest block, pinned): the range spans the finite ones (v_max_f32
        # skips NaN), so the digits keep their resolution and +inf lands in digit 1 (fma → -inf, clamped)
        A(f"  v_mov_b32 v{t[2]}, 0x7f800000")
        A(f"  v_mov_b32 v{mx}, v6")
        for i in range(EPT):
            A(f"  v_cmp_neq_f32 vcc, v{t[2]}, v{VAL + i}")
            A(f"  v_cndmask_b32 v{t[3]}, v6, v{VAL + i}, vcc")
            A(f"  v_max_f32 v{mx}, v{mx}, v{t[3]}")
        A(f"  v_min_f32 v{mn}, v{VAL}, v{VAL + 1}")
        for i in range(2, EPT):
            A(f"  v_min_f32 v{mn}, v{mn}, v{VAL + i}")
    else:
        A(f"  v_max_f32 v{mx}, v{VAL}, v{VAL + 1}")
        A(f"  v_min_f32 v{mn}, v{VAL}, v{VAL + 1}")
        for i in range(2, EPT):
            A(f"  v_max_f32 v{mx}, v{mx}, v{VAL + i}")
            A(f"  v_min_f32 v{mn}, v{mn}, v{VAL + i}")
    wave_reduce(["max_f32", "min_f32"], [mx, mn], [t[2], t[3]])
    cross_wave(["max_f32", "min_f32"], [mx, mn], RED_MM, [t[2], t[3]])
    # s = 2047 / (vmax - vmin) (0 if equal); negs = -s, c1 = vmax·s
    ng, c1 = t[5], t[6]
    A(f"  v_sub_f32 v{t[4]}, v{mx}, v{mn}")
    A(f"  v_rcp_f32 v{ng}, v{t[4]}")
    nop(0)
    A(f"  v_mul_f32 v{ng}, 0x44ffe000, v{ng}")  # 2047.0
    A(f"  v_cmp_lt_f32 vcc, 0, v{t[4]}")
    A(f"  v_cndmask_b32_e64 v{ng}, 0, v{ng}, vcc")
    A(f"  v_mul_f32 v{c1}, v{mx}, v{ng}")
    A(f"  v_xor_b32 v{ng}, 0x80000000, v{ng}")
    A(f"  v_mov_b32 v{t[7]}, 0x44ffe000")  # 2047.0
    A(f"  s_mov_b32 s49, {DIG0}")
    for i in range(EPT):
        r = ST + i
        A(f"  v_fma_f32 v{t[8]}, v{VAL + i}, v{ng}, v{c1}")
        A(
            f"  v_min_f32 v{t[8]}, v{t[8]}, v{t[7]}"
        )  # NaN → 2047 (the state below keeps it dead)
        A(f"  v_max_f32 v{t[8]}, 1.0, v{t[8]}")
        A(f"  v_cvt_u32_f32 v{t[8]}, v{t[8]}")
        A(f"  v_add_lshl_u32 v{t[8]}, v{t[8]}, s49, 2")
        A(
            f"  v_max_u32 v{r}, v{r}, v{t[8]}"
        )  # live: 4·lane < digit state; dead stays dead
    histogram()
    A(".Lround:")
    find_bin()
    A(f"  s_cmp_gt_u32 s{S_CB}, {CAP}")
    A("  s_cbranch_scc0 .Lcollect")

    # ---------------------------------------------------------------- refine (dense bin)
    A(f"  s_add_u32 s{S_AT}, s{S_AT}, s{S_A}")
    A(f"  s_sub_u32 s{S_NEED}, s{S_NEED}, s{S_A}")
    km, kn = t[0], t[1]
    A(f"  v_mov_b32 v{km}, 0")
    A(f"  v_mov_b32 v{kn}, -1")
    for i in range(EPT):
        r = ST + i
        A(f"  v_cmp_gt_u32 vcc, s{S_B4}, v{r}")
        A(f"  v_cndmask_b32_e64 v{r}, v{r}, v9, vcc")  # better → selected
        A(f"  v_cmp_lt_u32 vcc, s{S_B4}, v{r}")
        A(f"  v_cndmask_b32_e64 v{r}, v{r}, v7, vcc")  # worse → dead
        A(f"  v_cmp_eq_u32 s[56:57], s{S_B4}, v{r}")
        key_of(VAL + i, t[2], t[3])
        A(f"  v_cndmask_b32_e64 v{t[4]}, 0, v{t[2]}, s[56:57]")
        A(f"  v_cndmask_b32_e64 v{t[5]}, -1, v{t[2]}, s[56:57]")
        A(f"  v_max_u32 v{km}, v{km}, v{t[4]}")
        A(f"  v_min_u32 v{kn}, v{kn}, v{t[5]}")
    red2_u32(km, kn, RED_K)
    A(f"  s_mov_b32 s55, {DIG0 + 1}")
    A("  s_cmp_eq_u32 s45, s46")
    A("  s_cbranch_scc1 .Lbypos")
    # key digits: ((kmax - key) >> sh) + 1, sh = max(0, bitlen(kmax - kmin) - 11)
    A("  s_sub_u32 s52, s45, s46")
    A("  s_flbit_i32_b32 s53, s52")
    A("  s_sub_u32 s53, 21, s53")
    A("  s_max_i32 s53, s53, 0")
    for i in range(EPT):
        r = ST + i
        A(f"  v_cmp_eq_u32 s[56:57], s{S_B4}, v{r}")
        key_of(VAL + i, t[2], t[3])
        A(f"  v_sub_u32 v{t[2]}, s45, v{t[2]}")
        A(f"  v_lshrrev_b32 v{t[2]}, s53, v{t[2]}")
        A(f"  v_add_lshl_u32 v{t[2]}, v{t[2]}, s55, 2")
        A(f"  v_cndmask_b32_e64 v{r}, v{r}, v{t[2]}, s[56:57]")
    A("  s_branch .Lrehist")
    A(".Lbypos:")
    pm, pn = t[0], t[1]
    A(f"  v_mov_b32 v{pm}, 0")
    A(f"  v_mov_b32 v{pn}, -1")
    for k in range(nl):
        for j in range(4):
            r = ST + 4 * k + j
            A(f"  v_cmp_eq_u32 s[56:57], s{S_B4}, v{r}")
            real_pos(k, j, t[2])
            A(f"  v_cndmask_b32_e64 v{t[4]}, 0, v{t[2]}, s[56:57]")
            A(f"  v_cndmask_b32_e64 v{t[5]}, -1, v{t[2]}, s[56:57]")
            A(f"  v_max_u32 v{pm}, v{pm}, v{t[4]}")
            A(f"  v_min_u32 v{pn}, v{pn}, v{t[5]}")
    red2_u32(pm, pn, RED_K + 128)
    A("  s_sub_u32 s52, s45, s46")
    A("  s_flbit_i32_b32 s53, s52")
    A("  s_sub_u32 s53, 21, s53")
    A("  s_max_i32 s53, s53, 0")
    for k in range(nl):
        for j in range(4):
            r = ST + 4 * k + j
            A(f"  v_cmp_eq_u32 s[56:57], s{S_B4}, v{r}")
            real_pos(k, j, t[2])
            A(f"  v_subrev_u32 v{t[2]}, s46, v{t[2]}")
            A(f"  v_lshrrev_b32 v{t[2]}, s53, v{t[2]}")
            A(f"  v_add_lshl_u32 v{t[2]}, v{t[2]}, s55, 2")
            A(f"  v_cndmask_b32_e64 v{r}, v{r}, v{t[2]}, s[56:57]")
    A(".Lrehist:")
    zero_hist()
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    histogram()
    A("  s_branch .Lround")

    # ---------------------------------------------------------------- collect the threshold bin
    A(".Lcollect:")
    A(f"  s_add_u32 s{S_AT}, s{S_AT}, s{S_A}")
    A(f"  s_sub_u32 s{S_NEED}, s{S_NEED}, s{S_A}")
    ool = []
    for k in range(nl):
        for j in range(4):
            i = 4 * k + j
            back = L("Lcb")
            go = L("Lcg")
            A(f"  v_cmp_eq_u32 vcc, s{S_B4}, v{ST + i}")
            A(f"  s_cbranch_vccnz {go}")
            A(f"{back}:")
            ool += [
                f"{go}:",
                "  s_and_saveexec_b64 s[50:51], vcc",
                f"  ds_add_rtn_u32 v{t[0]}, v4, v3 offset:{CNT}",
                f"  v_and_b32 v{t[1]}, 0xffffff, v{POS + i}"
                if has_pos
                else f"  v_add_u32 v{t[1]}, {4 * k * T + j}, v2",
                "  s_waitcnt lgkmcnt(0)",
                f"  v_lshlrev_b32 v{t[0]}, 3, v{t[0]}",
                f"  ds_write_b32 v{t[0]}, v{VAL + i} offset:{CAND}",
                f"  ds_write_b32 v{t[0]}, v{t[1]} offset:{CAND + 4}",
                "  s_mov_b64 exec, s[50:51]",
                f"  s_branch {back}",
            ]

    # ---------------------------------------------------------------- output: states < B4 in position order
    A(".Lout:")
    cnts = t[0:nl]
    P = t[18 : 18 + NQ]
    for k in range(nl):
        A(f"  v_mov_b32 v{cnts[k]}, 0")
        for j in range(4):
            A(f"  v_cmp_gt_u32 vcc, s{S_B4}, v{ST + 4 * k + j}")
            A(f"  v_addc_co_u32 v{cnts[k]}, vcc, 0, v{cnts[k]}, vcc")
    for q in range(NQ):
        ks = list(range(FPD * q, min(nl, FPD * q + FPD)))
        A(f"  v_mov_b32 v{P[q]}, v{cnts[ks[0]]}")
        for f, kk in enumerate(ks[1:], 1):
            A(f"  v_lshl_or_b32 v{P[q]}, v{cnts[kk]}, {FB * f}, v{P[q]}")
    own = t[24 : 24 + NQ]
    for q in range(NQ):
        A(f"  v_mov_b32 v{own[q]}, v{P[q]}")
    tot = [58 + q for q in range(NQ)]
    block_scan(
        P, RED_S2, t[30 : 30 + 2 * NQ], tot
    )  # its barrier also publishes CAND and the BT row
    for q in range(NQ):
        A(f"  v_sub_u32 v{P[q]}, v{P[q]}, v{own[q]}")  # exclusive
    # slot = Σ_{k' < k} S_k' + field_k(P) + running; unselected elements write a per-lane dummy slot
    A("  s_mov_b32 s52, 0")
    sl, rn, ad = t[42], t[43], t[44]
    for k in range(nl):
        q, f = divmod(k, FPD)
        A(f"  v_bfe_u32 v{sl}, v{P[q]}, {FB * f}, {FB}")
        A(f"  v_add_u32 v{sl}, s52, v{sl}")
        for j in range(4):
            A(f"  v_cmp_gt_u32 vcc, s{S_B4}, v{ST + 4 * k + j}")
            A(f"  v_lshlrev_b32 v{ad}, 2, v{sl}")
            pos_of(k, j, rn)
            A(f"  v_cndmask_b32_e64 v{ad}, v10, v{ad}, vcc")
            A(f"  v_addc_co_u32 v{sl}, vcc, 0, v{sl}, vcc")
            A(f"  ds_write_b32 v{ad}, v{rn} offset:{OUT}")
        A(f"  s_bfe_u32 s53, s{tot[q]}, {(FB * f) | (FB << 16)}")
        A("  s_add_u32 s52, s52, s53")
    # wave 0 ranks the c_B candidates (CAND is visible since the scan's barrier); winners → OUT[A + rank]
    A(f"  s_cmp_eq_u32 s{S_CB}, 0")
    A("  s_cbranch_scc1 .Lranked")
    A("  s_cmp_eq_u32 s28, 0")
    A("  s_cbranch_scc0 .Lranked")
    kv, kp = t[0], t[1]  # composite: hi = key, lo = ~pos
    A(f"  v_mov_b32 v{t[2]}, 0")
    A(f"  v_mov_b32 v{t[3]}, 0")
    A(f"  v_cmp_gt_u32 vcc, s{S_CB}, v1")
    A("  s_and_saveexec_b64 s[50:51], vcc")
    A(f"  v_lshlrev_b32 v{t[4]}, 3, v1")
    A(f"  ds_read_b64 v[{t[2]}:{t[3]}], v{t[4]} offset:{CAND}")
    A("  s_mov_b64 exec, s[50:51]")
    A("  s_waitcnt lgkmcnt(0)")
    key_of(t[2], kv, t[5])
    A(f"  v_not_b32 v{kp}, v{t[3]}")
    A(f"  v_mov_b32 v{t[6]}, v{kp}")
    A(f"  v_mov_b32 v{t[7]}, v{kv}")
    rk = t[8]
    A(f"  v_mov_b32 v{rk}, 0")
    nop(1)
    for j in range(CAP):
        A(f"  s_cmp_le_u32 s{S_CB}, {j}")
        A("  s_cbranch_scc1 .Lrankd")
        A(f"  v_readlane_b32 s52, v{t[6]}, {j}")
        A(f"  v_readlane_b32 s53, v{t[7]}, {j}")
        A(f"  v_cmp_lt_u64 vcc, v[{t[6]}:{t[7]}], s[52:53]")
        A(f"  v_addc_co_u32 v{rk}, vcc, 0, v{rk}, vcc")
    A(".Lrankd:")
    A(f"  v_cmp_gt_u32 s[56:57], s{S_CB}, v1")
    A(f"  v_cmp_gt_u32 vcc, s{S_NEED}, v{rk}")
    A("  s_and_b64 vcc, vcc, s[56:57]")
    A("  s_and_saveexec_b64 s[50:51], vcc")
    A(f"  v_add_u32 v{rk}, s{S_AT}, v{rk}")
    A(f"  v_lshlrev_b32 v{rk}, 2, v{rk}")
    A(f"  ds_write_b32 v{rk}, v{t[3]} offset:{OUT}")
    A("  s_mov_b64 exec, s[50:51]")
    A(".Lranked:")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    # ---------------------------------------------------------------- copy-out
    if cand_out:
        # slot i < cnt: value LDS[lg_lds + 4 e], position s22 + e (e = OUT[i]); the rest of the segment NaN / -1
        def fetch(sl, dv, dp):
            A(f"  v_lshlrev_b32 v{t[3]}, 2, v{sl}")
            A(f"  ds_read_b32 v{dp}, v{t[3]} offset:{OUT}")
            A("  s_waitcnt lgkmcnt(0)")
            A(f"  v_lshlrev_b32 v{t[4]}, 2, v{dp}")
            A(f"  ds_read_b32 v{dv}, v{t[4]} offset:{lg_lds}")
            if remap is None:
                A(f"  v_add_u32 v{dp}, s22, v{dp}")
                A("  s_waitcnt lgkmcnt(0)")
            else:
                remap_pos(dp, t[5])

        segment(fetch)
        write_tau("0")
        A("  s_branch @@EXIT@@")
        return _finish(c, ool, pfx, exit_to, nv, lds, EPT * T, bt_max, fallback)
    A(f"  s_mul_i32 s52, {row}, s22")  # out_local row
    A(f"  s_mul_hi_u32 s53, {row}, s22")
    A("  s_add_u32 s14, s14, s52")
    A("  s_addc_u32 s15, s15, s53")
    if blk_out:  # positions only, -1 past the count
        for r in range(-(-KT // T)):
            sl_ = t[0]
            A(f"  v_add_u32 v{sl_}, {r * T}, v0")
            A(f"  v_mov_b32 v{t[1]}, -1")
            A(f"  v_cmp_gt_u32 vcc, s{S_CNT}, v{sl_}")
            A("  s_and_saveexec_b64 s[50:51], vcc")
            A(f"  v_lshlrev_b32 v{t[3]}, 2, v{sl_}")
            A(f"  ds_read_b32 v{t[1]}, v{t[3]} offset:{OUT}")
            A("  s_waitcnt lgkmcnt(0)")
            A("  s_mov_b64 exec, s[50:51]")
            A(f"  v_lshlrev_b32 v{t[3]}, 2, v{sl_}")
            if (r + 1) * T > KT:
                A(f"  v_cmp_gt_u32 vcc, {KT}, v{sl_}")
                A("  s_and_saveexec_b64 s[50:51], vcc")
            A(f"  global_store_dword v{t[3]}, v{t[1]}, s[14:15]")
            A("  s_mov_b64 exec, -1")
        if exit_to:
            A("  s_branch @@EXIT@@")
        else:
            A("  s_endpgm")
        if body_only:
            return _finish(c, ool, pfx, exit_to, nv, lds, EPT * T, bt_max, fallback)
        c += ool
        src = kernel_source(name, c, ka, lds=lds, vgprs=nv, agprs=0, sgprs=80, block=T)
        return dict(name=name, src=src, ka=ka, block=T, wmax=EPT * T, bt_max=bt_max)
    A(f"  s_lshl_b32 s52, {row}, {2 + 9}")  # tk row (512 dwords)
    A(f"  s_lshr_b32 s53, {row}, {32 - 11}")
    A("  s_add_u32 s16, s16, s52")
    A("  s_addc_u32 s17, s17, s53")
    A("  s_lshl_b32 s54, 1, s23")
    A("  s_sub_u32 s54, s54, 1")  # bs - 1
    for r in range(-(-KT // T)):
        sl_ = t[0]
        A(f"  v_add_u32 v{sl_}, {r * T}, v0")
        A(f"  v_mov_b32 v{t[1]}, -1")
        A(f"  v_mov_b32 v{t[2]}, -1")
        A(f"  v_cmp_gt_u32 vcc, s{S_CNT}, v{sl_}")
        A("  s_and_saveexec_b64 s[50:51], vcc")
        A(f"  v_lshlrev_b32 v{t[3]}, 2, v{sl_}")
        A(f"  ds_read_b32 v{t[1]}, v{t[3]} offset:{OUT}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_lshrrev_b32 v{t[4]}, s23, v{t[1]}")
        A(f"  v_lshlrev_b32 v{t[4]}, 2, v{t[4]}")
        A(f"  ds_read_b32 v{t[4]}, v{t[4]} offset:{bt_off}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_lshlrev_b32 v{t[4]}, s23, v{t[4]}")
        A(f"  v_and_b32 v{t[5]}, s54, v{t[1]}")
        A(f"  v_or_b32 v{t[2]}, v{t[4]}, v{t[5]}")
        A("  s_mov_b64 exec, s[50:51]")
        A(f"  v_lshlrev_b32 v{t[3]}, 2, v{sl_}")
        A(f"  global_store_dword v{t[3]}, v{t[1]}, s[14:15]")
        A(f"  global_store_dword v{t[3]}, v{t[2]}, s[16:17]")
    A("  v_cmp_eq_u32 vcc, 0, v0")
    A("  s_and_saveexec_b64 s[50:51], vcc")
    A("  s_cmp_eq_u32 s27, 0")
    A(f"  s_cselect_b32 s52, 0, s{S_CNT}")
    A("  v_mov_b32 v12, s52")
    A("  v_mov_b32 v13, s24")
    A("  global_store_dword v13, v12, s[18:19]")
    if exit_to:
        A("  s_mov_b64 exec, -1")
        A("  s_branch @@EXIT@@")
    else:
        A("  s_endpgm")
    if body_only:
        return _finish(c, ool, pfx, exit_to, nv, lds, EPT * T, bt_max, fallback)
    c += ool
    src = kernel_source(name, c, ka, lds=lds, vgprs=nv, agprs=0, sgprs=80, block=T)
    return dict(name=name, src=src, ka=ka, block=T, wmax=EPT * T, bt_max=bt_max)


def _finish(c, ool, pfx, exit_to, nv, lds, wmax, bt_max, fallback=""):
    """An embeddable body: labels prefixed with pfx, @@EXIT@@ → exit_to, @@FB@@ → fallback."""
    lines = [re.sub(r"\.L", f".L{pfx}", x) for x in c + ool] if pfx else c + ool
    lines = [
        x.replace("@@EXIT@@", exit_to or "").replace("@@FB@@", fallback) for x in lines
    ]
    return dict(lines=lines, vgprs=nv, lds=lds, wmax=wmax, bt_max=bt_max)
