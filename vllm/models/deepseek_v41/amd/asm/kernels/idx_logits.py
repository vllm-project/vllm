# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""The indexer's MXFP4 logits and its top-512 in one
launch. Replaces aiter's ``paged_mxfp4_mqa_logits`` (decode, next_n 1) plus
e1's idx_topk launch:

    logits[row, t] = Σ_h w[row, h] · relu(q[row, h] · k[t]),  t < n = lens[row]

q: 32 heads × 128 dims MXFP4 per row; k: aiter's preshuffled MXFP4 pages
(``cache_format``: E tokens per page, n_per_tile 32, d_per_tile 16, two scale
lanes): a page holds E·64 value bytes, then E·4 scale bytes. Group g of 32
tokens: value chunk c (dims 32c .. 32c + 31) of token τ at 2048 g + 512 c + 16 τ,
scale s at E·64 + 128 g + 64 (s % 2) + 2 τ + s / 2.

Grid: 8·S·ceil(rows / 8) CTAs of 8 waves, S = 2^slog splits per row. CTA b:
x = b % 8, j = b / 8, row = x + 8 (j >> slog), split = j % S, so a row's
splits run on one XCD (the hand-off stays in its L2, as A3's combine). The
wide split kernel (no hand-off) maps slice-major instead: 8 j + x = S row +
split, slice s of every row on XCD s % 8. The
row's G = ceil(n / 32) groups are cut into S slices of ceil(G / S); wave w
takes the slice's groups w, w + 8, ...

Per group (one wave): 2 dwordx4 + 1 ushort per lane (the B operands and their
scales come straight from the page: lane l = token l % 32, dims 64 kb +
32 (l / 32)), two ``v_mfma_scale_f32_32x32x64_f8f6f4`` (A = q: lane l = head
l % 32), D = 32 heads × 32 tokens; the fold relu → ·w → Σ over the lane's 16
heads, + the other half-wave's 16 (permlane32 swap), → LDS (S = 1) or the
global row (S > 1). D groups of loads in flight per wave (a ring), two
accumulators (the fold of group i − 1 runs behind the MFMAs of group i). The
K loads are ``nt`` (read once; at M 64 the loop streams 6.1 TB/s, plain 4.7).
Before the first K load: n / req / valid (SMEM) and the row's K block table
(→ LDS, one entry per thread) are issued together with q / w.

Then S = 1: the folds wrote the row's logits to LDS; the top-512 (idx_topk's
code, its loads from LDS) runs in the same CTA. S > 1: the folds store the
logits straight to the global row (vmcnt is in order for loads and stores
mixed, so the ring's wait counts just include them); after a barrier wave 0
takes a ticket, ``global_atomic_inc`` on cnt[row] (one counter per 128 B line:
same-line atomics serialise) wrapping at S − 1 (graph-replay safe); the last
arriver runs the top-512 on the row (``nt`` loads: same XCD, so L2 is
coherent). A ticket per wave costs ~8 µs at M 64 (8× the same-line atomics).

Kernargs: idx_topk's (same offsets; ``logits`` = the global scratch rows for
S > 1, lstride a multiple of 128 B covering 32 G columns) + KA below. kbt =
the indexer K block table (row-indexed, ≤ 512 pages per row), page_stride in
bytes (< 2^32).
"""

from vllm.models.deepseek_v41.amd.asm.kernels import idx_topk
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source
from vllm.utils.math_utils import cdiv

KA = KArgs(
    *idx_topk.KA.args,
    ("q", "ptr"),
    ("qs", "ptr"),
    ("w", "ptr"),
    ("kv", "ptr"),
    ("kbt", "ptr"),
    ("cnt", "ptr"),
    ("rows", "u32"),
    ("kbt_stride", "u32"),
    ("page_stride", "u32"),
    ("e_log2", "u32"),
    ("slog", "u32"),
)
KA_S = KArgs(
    *KA.args, ("bs", "ptr"), ("bs_stride", "u32")
)  # wide + block scores (the candidate source)
KA_G = KArgs(
    *KA.args, ("cand", "ptr"), ("cand_stride", "u32")
)  # wide + gather (the candidate consumers)
assert (
    KA_G.off["cand"] == KA_S.off["bs"]
    and KA_G.off["cand_stride"] == KA_S.off["bs_stride"]
)
NCAND, CBLK = 2048, 8  # gather: candidate blocks per row, positions per block
assert KA.off["q"] == 0x50 and KA.off["rows"] == 0x80 and KA.off["slog"] == 0x90
T = 512

# VGPRs of the logits phase (v0 = tid stays)
V_L, V_16L, V_2L, V_4L, V_PG = 1, 2, 3, 4, 5
QA = (8, 12)  # q, K blocks 0 / 1 of the lane's head
QS = 16  # q scales, byte 0 = block 0, byte 2 = block 1
W = 20  # 16 weights: acc r ↔ head 8 (r / 4) + 4 (l / 32) + r % 4
ACC = (36, 52)
RING = 68  # D slots of 12: 8 value VGPRs, 1 scale

# SGPRs: s[4:23] idx_topk's kernargs, s25 / s26 / s27 n, req, valid word (loaded here for the top-k);
# s[84:100] ours; s72 row (kept for the top-k)
S_ROW, S_SPL, S_G, S_GLO, S_GHI, S_W, S_NG, S_I, S_PREV, S_GLW, S_GMASK = (
    72,
    73,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
)
S_N, S_REQ, S_VW = 25, 26, 27
S_SLOG, S_CNTC = 100, 101
S_SM1, S_GSH, S_E64, S_NG1, S_LGW = 58, 38, 39, 56, 57
S_PG0 = 57  # wide: the slice's first page (prologue only; S_LGW later)
S_NT = 74  # wide: the row's n
S_NS = 62  # wide: the slice's live count (s62 / 63 only serve the global path)
S_TAG = 63  # wide: this call's tag << 24 (cnt[row] + 1, mod 256)
S_SH = 44  # wide, prologue → the loop: the slice's first page is shared (1) or not
S_CAPS = 77  # wide, in the merge: S · CAPS slots (S_GHI is free by then)
S_REQ2, S_VW2 = (
    80,
    81,
)  # wide: req / valid word kept for the merge (S_I / S_PREV are free by then)


def plan(width: int, m: int):
    """(nl, lg_cap, lds_s1, slog) for rows of ≤ width columns at m rows: S = 4 / 2 / 1 at m ≤ 64 / ≤ 128 / more
    (S ≥ 2 if the row can't stay in LDS, more if a wave's page ids would not fit one VGPR); lg_cap depends on the
    width only, so one kernel serves every m."""
    nl = -(-width // 2048)
    G = -(-width // 32)
    lds_s1 = nl <= 5  # the top-k's LDS reads must sit below 64 KB
    slog = 2 if m <= 64 else 1 if m <= 128 else 0
    if not lds_s1:
        slog = max(slog, 1)
    while cdiv(cdiv(G, 1 << slog), 8) > 64:
        slog += 1
    lg_cap = 64 * nl if lds_s1 else cdiv(cdiv(G, 2), 16) * 16
    assert -(-G // (1 << slog)) <= lg_cap
    return nl, lg_cap, lds_s1, slog


CAPS = 640  # wide: candidate slots per slice (≥ 512: the exact fallback's)


CMAX = 4096  # wide: the compacted merge's room (candidates with key ≥ τ_row)


def row_slots(nlf: int, slog: int) -> int:
    """8 B slots per candidates row: S segments of CAPS, then S τ slots, ≥ the full merge's 2048 nlf loads."""
    S = 1 << slog
    return max(2048 * nlf, -(-(S * CAPS + S) // 16) * 16)


SLOG_MAX = 5  # wide: ≤ 32 CTAs per row (the merge's τ slots: one per lane)


def plan_wide(width: int, m: int):
    """(nl, lg_cap, nlf, slog) for the wide-row kernel: S = 2^slog CTAs per row, each slice ≤ 512 groups (its
    logits in LDS, ≤ 16384 columns), S ≥ 512 / m CTAs per row at small m (≤ 2^SLOG_MAX); nl = the per-slice
    top-k's loads, nlf = the merge's (S · CAPS candidates). A kernel built for m = 256 (the fewest, widest
    slices) serves every m (its slices only get shorter)."""
    G = -(-width // 32)
    slog = min(SLOG_MAX, max(0, (-(-512 // m) - 1).bit_length()))
    while -(-G // (1 << slog)) > 512:
        slog += 1
    assert slog <= SLOG_MAX, (width, slog)
    per = -(-G // (1 << slog))
    nl = -(-per // 64)
    lg_cap = 64 * nl
    nlf = -(-(1 << slog) * CAPS // 2048)
    return nl, lg_cap, nlf, slog


def plan_gather(m: int, slog: int | None = None):
    """(nl, lg_cap, nlf, slog) for the gather mode (the candidate consumers: ≤ NCAND · CBLK live columns per
    row): S = 4 CTAs per row at m ≤ 64, else 2 (the best at 200k with plain K loads)."""
    G = NCAND * CBLK // 32
    if slog is None:
        slog = 2 if m <= 64 else 1
    per = -(-G // (1 << slog))
    nl = -(-per // 64)
    return nl, 64 * nl, -(-(1 << slog) * CAPS // 2048), slog


def source(
    nl: int = 5,
    lg_cap: int = 320,
    lds_s1: bool = True,
    bt_max: int = 1024,
    name: str | None = None,
    wide: bool = False,
    nlf: int = 2,
    kshared: str | None = None,
    split: bool = False,
    bsc: bool = False,
    gather: bool = False,
):
    """nl: the top-k's loads per thread (width ≤ 2048 nl); lg_cap: groups per CTA slice (≥ ceil(G / S));
    lds_s1: S = 1 allowed (the row's logits stay in LDS; needs lg_cap·128 ≥ 8192 nl). K loads nt; S > 1: one
    ticket per CTA (sc0 atomic: a row's CTAs share an XCD) on cnt[row] at byte row << 7 (one counter per
    128 B line, as same-line atomics serialise).
    wide: rows wider than one CTA's top-k (module doc); every slice's logits stay in LDS (lg_cap groups),
    nl = the slice top-k's loads per thread, nlf = the merge's (≥ S · CAPS / 2048), bt_max ≥ the merge's
    block-table entries per row; kshared: None = per slice, plain K loads when its first page is also row ^ 1's
    or row ^ 2's (a shared prefix: the page stays in L2 / MALL for them), else nt; or a fixed flag ("none" =
    plain); split: no merge here (every CTA leaves after its candidates; merge_source's kernel follows);
    split maps CTAs slice-major (module doc; rows that share pages, spec decode's draft rows or a shared prefix,
    read each slice on one XCD; the row-major map was ~1-3 µs faster at M ≤ 128 with distinct contexts, as the
    merge's CTA for row r then ran on the XCD that wrote its slots, and up to 2x slower otherwise).
    bsc (wide split): also the block scores of the candidate source (vLLM's ``block_scores``): the max over
    every 8 positions (-inf past n), the row's newest block +inf, to bs + row · bs_stride + 4 b (fp32). After
    the loop, from the slice's logits in LDS (no stores in the loop: the ring's in-order vmcnt waits would
    wait for them); they drain while the filter runs.
    gather (wide split, the candidate consumers): the row's columns are its candidate blocks
    (cand + row · cand_stride: NCAND block ids of CBLK positions, any order, -1 = none): the CTA compacts the
    valid ones (the partial newest block last, so the live columns are a prefix: 8 · full + its live part),
    builds its slice's per-block address table in LDS (value base, scale delta, block id), loads each lane's
    token from its own block (a 32-column group = 4 blocks), and the filter maps columns back to positions
    (the merge and its lists see positions). Rows of n = lens[row] context positions as usual.
    """
    D = U = (
        4  # groups of loads in flight per wave (even: the peeled ring and the loop agree)
    )
    if wide:
        lds_s1 = True
    assert not bsc or (wide and split)
    assert not gather or (wide and not bsc)
    name = name or (
        f"idx_logits_l{nl}_g{lg_cap}_d{D}"
        + ("" if lds_s1 else "_g")
        + (f"_w{nlf}_b{bt_max}" if wide else "")
        + (f"_k{kshared.replace(' ', '')}" if kshared is not None else "")
        + ("_split" if split else "")
        + "_nt"
        + ("_bs" if bsc else "")
        + ("_ga" if gather else "")
    )
    kf = " nt"
    HB = idx_topk.HIST_OFF + 4 * (
        idx_topk.DIG0 + 2048
    )  # wide: digit 2048 - b at HB - 4 b
    ka = (KA_S if bsc else KA_G if gather else KA) if wide else KA
    LG = -(-(idx_topk.BT + (0 if wide else 4 * bt_max)) // 128) * 128
    lg_size = lg_cap * 128
    if lds_s1:
        assert lg_size >= 16 * nl * T, (lg_cap, nl)
    if wide and not split and not gather:
        assert lg_size >= 4 * bt_max + 8 * CMAX, (
            lg_cap,
            bt_max,
        )  # the merge reuses the logits' LDS
    # below the logits, so the DS offsets stay < 64 KB (the logits region can pass it: 528 groups at 16k+)
    DUMMY = LG  # the first fold's target
    FLAG = DUMMY + 128  # the 8 waves' tickets (wide: the merger's hand-off)
    KP = FLAG + 128  # the row's (wide: the slice's) K block table (T entries)
    LG = KP + 4 * T
    lds = LG + lg_size
    TBL = lds  # gather: the slice's blocks, 16 B each (value base lo / hi, scale delta, id)
    if gather:
        assert lg_size >= 4 * NCAND, (
            lg_cap
        )  # the compacted candidate list sits in the logits region
        lds += 16 * 4 * lg_cap
        if not split:  # the fused merge: BT + the compacted candidates from LG
            lds = max(lds, LG + 4 * bt_max + 8 * CMAX)
    TV = RING + 12 * D  # 4 temps (wide: TV + 3 = 1, TV + 4 for the prologue)
    nv = TV + 5 if wide else TV + 4
    G_LJ, G_LCV, G_LCS = (
        5,
        6,
        7,
    )  # gather: lane constants (V_PG and the q address temps are free by then)
    if gather:
        nv = (
            TV + 10
        )  # issue(): TV, TV + 1 value address, TV + 4..7 the table entry, TV + 8 / 9 scales
    c = []
    A = c.append
    lbl = [0]

    def L(p="L"):
        lbl[0] += 1
        return f".{p}{lbl[0]}"

    def nop(n=1):
        A(f"  s_nop {n}")

    kfc = [kf]  # the K loads' cache flag (loop variants)

    def issue(slot, tgt):
        """Loads of group tgt (an SGPR: the wave's group index) into ring slot `slot`."""
        B = RING + 12 * slot
        if gather:
            # lane l: block (l % 32) / 8 of the group, from the table at TBL + 64 (w + 8 i) + 16 ((l % 32) / 8)
            # (s30 = TBL + 64 w); value base + 512 (l / 32) + 16 (l % 8), scales base + delta + 64 (l / 32) + 2 (l % 8)
            A(f"  s_lshl_b32 s43, s{tgt}, 9")
            A("  s_add_u32 s43, s43, s30")
            A(f"  v_add_u32 v{TV + 4}, s43, v{G_LJ}")
            A(f"  ds_read_b128 v[{TV + 4}:{TV + 7}], v{TV + 4}")
            A("  s_waitcnt lgkmcnt(0)")
            A(f"  v_add_co_u32 v{TV}, vcc, v{G_LCV}, v{TV + 4}")
            A(f"  v_addc_co_u32 v{TV + 1}, vcc, 0, v{TV + 5}, vcc")
            A(f"  v_add_u32 v{TV + 6}, v{TV + 6}, v{G_LCS}")
            A(f"  v_add_co_u32 v{TV + 8}, vcc, v{TV + 4}, v{TV + 6}")
            A(f"  v_addc_co_u32 v{TV + 9}, vcc, 0, v{TV + 5}, vcc")
            f = f" {kfc[0].strip()}" if kfc[0].strip() else ""
            A(f"  global_load_dwordx4 v[{B}:{B + 3}], v[{TV}:{TV + 1}], off{f}")
            A(
                f"  global_load_dwordx4 v[{B + 4}:{B + 7}], v[{TV}:{TV + 1}], off offset:1024{f}"
            )
            A(f"  global_load_ushort v{B + 8}, v[{TV + 8}:{TV + 9}], off{f}")
            return
        nop(3)  # SALU write → readlane lane select
        A(f"  v_readlane_b32 s42, v{V_PG}, s{tgt}")
        A(f"  s_lshl_b32 s43, s{tgt}, 3")
        A(f"  s_add_u32 s43, s43, s{S_GLW}")  # g
        A(f"  s_and_b32 s44, s43, s{S_GMASK}")  # group in page
        A("  s_lshl_b32 s45, s44, 11")
        A("  s_lshl_b32 s46, s44, 7")
        A(f"  s_add_u32 s46, s46, s{S_E64}")
        A("  s_mul_i32 s48, s42, s98")
        A("  s_mul_hi_u32 s49, s42, s98")
        A("  s_add_u32 s48, s48, s90")
        A("  s_addc_u32 s49, s49, s91")
        A("  s_add_u32 s50, s48, s45")
        A("  s_addc_u32 s51, s49, 0")
        A("  s_add_u32 s52, s48, s46")
        A("  s_addc_u32 s53, s49, 0")
        f = kfc[0]
        A(f"  global_load_dwordx4 v[{B}:{B + 3}], v{V_16L}, s[50:51]{f}")
        A(
            f"  global_load_dwordx4 v[{B + 4}:{B + 7}], v{V_16L}, s[50:51] offset:1024{f}"
        )
        A(f"  global_load_ushort v{B + 8}, v{V_2L}, s[52:53]{f}")

    def mfma(acc, slot):
        B = RING + 12 * slot
        A(
            f"  v_mfma_scale_f32_32x32x64_f8f6f4 v[{acc}:{acc + 15}], v[{QA[0]}:{QA[0] + 3}], v[{B}:{B + 3}], 0, "
            f"v{QS}, v{B + 8} op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4"
        )
        A(
            f"  v_mfma_scale_f32_32x32x64_f8f6f4 v[{acc}:{acc + 15}], v[{QA[1]}:{QA[1] + 3}], v[{B + 4}:{B + 7}], "
            f"v[{acc}:{acc + 15}], v{QS}, v{B + 8} op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4"
        )

    def fold(acc, glb=False):
        """Σ_r relu(acc r)·w r, + the other half-wave's → LDS at s[prev] + 4 l (lanes 0..31), or (glb) the
        global logits row (base s[62:63]) at that byte offset."""
        for r in range(16):
            A(f"  v_max_f32 v{acc + r}, 0, v{acc + r}")
        A(f"  v_mul_f32 v{TV}, v{acc}, v{W}")
        for r in range(1, 16):
            A(f"  v_fmac_f32 v{TV}, v{acc + r}, v{W + r}")
        A(f"  v_mov_b32 v{TV + 1}, v{TV}")
        nop(1)
        A(f"  v_permlane32_swap_b32 v{TV}, v{TV + 1}")
        nop(4)
        A(f"  v_add_f32 v{TV}, v{TV}, v{TV + 1}")
        A(f"  v_add_u32 v{TV + 2}, s{S_PREV}, v{V_4L}")
        A("  s_mov_b32 exec_hi, 0")
        if glb:
            A(f"  global_store_dword v{TV + 2}, v{TV}, s[62:63]")
        else:
            A(f"  ds_write_b32 v{TV + 2}, v{TV}")
        if wide:
            # histogram digit 2048 - (key >> 21) of the live tokens (lane < n_s - first token; the first fold
            # is of a dummy, s[prev] < LG)
            A(f"  s_sub_i32 s41, s{S_PREV}, {LG}")
            A("  s_ashr_i32 s41, s41, 2")
            A(f"  s_sub_i32 s61, s{S_NS}, s41")
            A("  s_cmp_lt_i32 s41, 0")
            A("  s_cselect_b32 s61, 0, s61")
            A(f"  v_ashrrev_i32 v{TV + 1}, 31, v{TV}")
            A(f"  v_or_b32 v{TV + 1}, 0x80000000, v{TV + 1}")
            A(f"  v_xor_b32 v{TV + 1}, v{TV + 1}, v{TV}")
            A(f"  v_lshrrev_b32 v{TV + 1}, 21, v{TV + 1}")
            A(f"  v_lshlrev_b32 v{TV + 1}, 2, v{TV + 1}")
            A(f"  v_sub_u32 v{TV + 1}, {HB}, v{TV + 1}")
            A(f"  v_cmp_gt_i32 vcc, s61, v{V_L}")
            A("  s_and_b32 exec_lo, exec_lo, vcc_lo")
            A(f"  ds_add_u32 v{TV + 1}, v{TV + 3}")
            A("  s_mov_b32 exec_lo, -1")
        A("  s_mov_b32 exec_hi, -1")

    def step(i, glb, peel, sfx=""):
        """Group i (slot i % D, accumulator i % 2): wait for its loads, MFMAs, fold of group i - 1, loads of
        group i + D. glb: one store per step before the loads, so the wait counts differ (in-order vmcnt)."""
        r, a = i % D, i % 2
        if not glb:
            cnt = 3 * (D - 1)
        elif peel:
            cnt = (
                3 * (D - 1) + i
            )  # prefetch loads i+1..D-1, then i steps of store + 3 loads
        else:
            cnt = 4 * (D - 1)
        A(f"  s_waitcnt vmcnt({cnt})")
        mfma(ACC[a], r)
        fold(ACC[1 - a], glb)  # group i - 1 (i = 0: a dummy slot)
        A(f"  s_lshl_b32 s{S_PREV}, s{S_I}, 10")
        A(f"  s_add_u32 s{S_PREV}, s{S_PREV}, s{S_LGW}")  # base + 128 (w + 8 i)
        A(f"  s_add_u32 s40, s{S_I}, {D}")
        A(f"  s_min_u32 s40, s40, s{S_NG1}")
        issue(r, 40)
        A(f"  s_add_u32 s{S_I}, s{S_I}, 1")
        A(f"  s_cmp_ge_u32 s{S_I}, s{S_NG}")
        A(f"  s_cbranch_scc1 .Ltail{'g' if glb else 'l'}{sfx}{a}")

    def loop(glb, sfx="", kfl=None):
        """Prefetch, the groups, the tails → .Lldone (LDS) / .Lgdone (global). sfx / kfl: a variant's label
        suffix and K load flag (default nt)."""
        kfc[0] = kf if kfl is None else kfl
        m = "g" if glb else "l"
        done = f".L{m}done"
        if glb:
            A(
                f"  s_lshl_b32 s{S_LGW}, s{S_GLW}, 7"
            )  # 128 (g_lo + w): the row offset of the first group
            A(
                f"  s_mov_b32 s{S_PREV}, s{S_LGW}"
            )  # the dummy fold rewrites it before the real one
        else:
            A(f"  s_lshl_b32 s{S_LGW}, s{S_W}, 7")
            A(f"  s_add_u32 s{S_LGW}, s{S_LGW}, {LG}")
            A(f"  s_mov_b32 s{S_PREV}, {DUMMY}")
        A(f"  s_mov_b32 s{S_I}, 0")
        A(f"  s_cmp_eq_u32 s{S_NG}, 0")
        A(f"  s_cbranch_scc1 {done}")
        for k in range(D):
            A(f"  s_min_u32 s40, {k}, s{S_NG1}")
            issue(k, 40)
        if glb:
            for i in range(D):
                step(i, True, True, sfx)
        A(f".Lloop{m}{sfx}:")
        for u in range(U):
            step(u, glb, False, sfx)
        A(f"  s_branch .Lloop{m}{sfx}")
        for a in range(2):
            A(f".Ltail{m}{sfx}{a}:")
            for _ in range(3):
                nop(7)  # MFMA → VALU read of its result
            fold(ACC[a], glb)
            A(f"  s_branch {done}")

    def ticket_cta():
        """One ticket per CTA (wave 0, after a barrier) on cnt[row], wrapping at S - 1 (s[S_SM1]); every CTA but
        the last arriver → .Lend."""
        A("  s_barrier")
        A(f"  s_cmp_eq_u32 s{S_W}, 0")
        A("  s_cbranch_scc0 .Ltw")
        A(f"  v_mov_b32 v18, s{S_SM1}")
        A(f"  s_lshl_b32 s41, s{S_ROW}, 7")
        A("  v_mov_b32 v19, s41")
        A("  s_mov_b64 exec, 1")
        A("  global_atomic_inc v20, v19, v18, s[94:95] sc0")
        A("  s_waitcnt vmcnt(0)")
        A(f"  v_mov_b32 v21, {FLAG}")
        A("  ds_write_b32 v21, v20")
        A("  s_mov_b64 exec, -1")
        A("  s_waitcnt lgkmcnt(0)")
        A(".Ltw:")
        A("  s_barrier")
        A(f"  v_mov_b32 v21, {FLAG}")
        A("  ds_read_b32 v20, v21")
        A("  s_waitcnt lgkmcnt(0)")
        A("  v_readfirstlane_b32 s40, v20")
        A(f"  s_cmp_eq_u32 s40, s{S_SM1}")
        A("  s_cbranch_scc0 .Lend")

    def block_scores():
        """The slice's block scores from its logits in LDS: thread t reads elements e = 4 t + 2048 k (one
        conflict-free b128), masks e ≥ n_s to -inf, max of 4, then of the lane pair (DPP): even lane 2 m holds
        block m + 256 k; the row's newest block → +inf; stores from the even lanes of live blocks. The ring's
        VGPRs are free here."""
        R, VM, VE, V16, VB, VO, VINF, VA = (
            RING,
            RING + 4,
            RING + 5,
            RING + 6,
            RING + 7,
            RING + 8,
            RING + 9,
            RING + 10,
        )
        VNINF = RING + 11
        done = L("Lbs")
        A(f"  s_lshl_b32 s30, s{S_GLO}, 5")
        A(f"  s_sub_i32 s30, s{S_NT}, s30")
        A("  s_sub_i32 s30, s30, 1")
        A(
            "  s_ashr_i32 s30, s30, 3"
        )  # the newest block, slice-local (out of range: none)
        A(f"  v_lshlrev_b32 v{VE}, 2, v0")  # e = 4 t
        A(f"  v_lshlrev_b32 v{V16}, 4, v0")
        A(f"  v_lshrrev_b32 v{VB}, 1, v0")  # block t / 2 (even lanes)
        A(f"  v_lshlrev_b32 v{VO}, 2, v{VB}")
        A(f"  v_mov_b32 v{VINF}, 0x7f800000")
        A(f"  v_mov_b32 v{VNINF}, 0xff800000")  # (v_cndmask with vcc: no literal)
        A("  s_mov_b32 s34, 0x55555555")
        A("  s_mov_b32 s35, s34")
        for k in range(lg_cap // 64):
            A(
                f"  s_sub_i32 s32, s{S_NS}, {2048 * k}"
            )  # live elements from this pass's first
            A("  s_cmp_le_i32 s32, 0")
            A(f"  s_cbranch_scc1 {done}")
            A(f"  v_add_u32 v{VA}, {LG + 8192 * k}, v{V16}")
            A(f"  ds_read_b128 v[{R}:{R + 3}], v{VA}")
            A("  s_waitcnt lgkmcnt(0)")
            for j in range(4):
                if j:
                    A(f"  s_sub_i32 s33, s32, {j}")
                A(f"  v_cmp_gt_i32 vcc, s{33 if j else 32}, v{VE}")
                A(f"  v_cndmask_b32 v{R + j}, v{VNINF}, v{R + j}, vcc")
            A(f"  v_max_f32 v{VM}, v{R}, v{R + 1}")
            A(f"  v_max_f32 v{VM}, v{VM}, v{R + 2}")
            A(f"  v_max_f32 v{VM}, v{VM}, v{R + 3}")
            nop(1)
            A(
                f"  v_max_f32_dpp v{VM}, v{VM}, v{VM} quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
            )
            A(f"  s_sub_i32 s33, s30, {256 * k}")
            A(f"  v_cmp_eq_u32 vcc, s33, v{VB}")
            A(f"  v_cndmask_b32 v{VM}, v{VM}, v{VINF}, vcc")
            A(f"  v_cmp_gt_i32 vcc, s32, v{VE}")  # the block's first element is live
            A("  s_and_b64 exec, vcc, s[34:35]")
            A(f"  v_add_u32 v{VA}, {1024 * k}, v{VO}")
            A(f"  global_store_dword v{VA}, v{VM}, s[54:55]")
            A("  s_mov_b64 exec, -1")
        A(f"{done}:")

    def compact():
        """gather: the row's NCAND candidate ids (thread t: 4 t .. 4 t + 3) → the full blocks (8 b + 8 ≤ n) in
        LDS at LG + 4 i, i < F (a block scan), the partial one (8 b < n < 8 b + 8) → s33 (-1: none), F → s32,
        n (s25) → the live columns 8 F + (n - 8 · partial), the row's n → S_NT (the merge's). Ring VGPRs as temps
        (no loads in flight there)."""
        C, F4, CNT, INC, POS, TA, TT = (
            RING,
            RING + 4,
            RING + 8,
            RING + 9,
            RING + 10,
            RING + 11,
            RING + 12,
        )
        PART = FLAG + 32
        A(f"  s_mov_b32 s{S_NT}, s{S_N}")
        A(f"  v_lshlrev_b32 v{TA}, 4, v0")
        A(f"  global_load_dwordx4 v[{C}:{C + 3}], v{TA}, s[54:55]")
        A(
            "  v_cmp_eq_u32 vcc, 0, v0"
        )  # the partial slot ← -1 (before the scan's barrier)
        A("  s_and_saveexec_b64 s[40:41], vcc")
        A(f"  v_mov_b32 v{TA}, -1")
        A(f"  v_mov_b32 v{POS}, {PART}")
        A(f"  ds_write_b32 v{POS}, v{TA}")
        A("  s_mov_b64 exec, s[40:41]")
        A("  s_waitcnt vmcnt(0)")
        A(f"  s_sub_i32 s41, s{S_N}, 8")
        A(f"  v_mov_b32 v{CNT}, 0")
        for i in range(4):
            A(f"  v_lshlrev_b32 v{TA}, 3, v{C + i}")
            A(f"  v_cmp_le_i32 s[42:43], 0, v{C + i}")
            A(f"  v_cmp_ge_i32 vcc, s41, v{TA}")  # 8 b ≤ n - 8
            A("  s_and_b64 vcc, vcc, s[42:43]")
            A(f"  v_cndmask_b32_e64 v{F4 + i}, 0, 1, vcc")
            A(f"  v_add_u32 v{CNT}, v{CNT}, v{F4 + i}")
        A(f"  v_mov_b32 v{INC}, v{CNT}")
        for ctl in (
            "row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:0",
            "row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:0",
            "row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:0",
            "row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:0",
            "row_bcast:15 row_mask:0xa bank_mask:0xf",
            "row_bcast:31 row_mask:0xc bank_mask:0xf",
        ):
            nop(1)
            A(f"  v_add_u32_dpp v{INC}, v{INC}, v{INC} {ctl}")
        nop(1)
        A(f"  v_readlane_b32 s42, v{INC}, 63")  # the wave's total → FLAG + 4 w
        A(f"  s_lshl_b32 s43, s{S_W}, 2")
        A(f"  s_add_u32 s43, s43, {FLAG}")
        A(f"  v_mov_b32 v{TA}, s42")
        A(f"  v_mov_b32 v{POS}, s43")
        A("  s_mov_b64 exec, 1")
        A(f"  ds_write_b32 v{POS}, v{TA}")
        A("  s_mov_b64 exec, -1")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        A(f"  v_mov_b32 v{TA}, {FLAG}")
        A(f"  ds_read_b128 v[{TT}:{TT + 3}], v{TA}")
        A(f"  ds_read_b128 v[{TT + 4}:{TT + 7}], v{TA} offset:16")
        A("  s_waitcnt lgkmcnt(0)")
        for k in range(8):
            A(f"  v_readfirstlane_b32 s{44 + k}, v{TT + k}")
        A("  s_mov_b32 s52, 0")  # the waves before this one
        A("  s_mov_b32 s32, 0")  # F
        for k in range(8):
            A(f"  s_add_u32 s32, s32, s{44 + k}")
            A(f"  s_cmp_gt_u32 s{S_W}, {k}")
            A(f"  s_cselect_b32 s42, s{44 + k}, 0")
            A("  s_add_u32 s52, s52, s42")
        A(f"  v_sub_u32 v{POS}, v{INC}, v{CNT}")
        A(f"  v_add_u32 v{POS}, s52, v{POS}")
        for i in range(4):
            A(f"  v_cmp_ne_u32 vcc, 0, v{F4 + i}")
            A("  s_and_saveexec_b64 s[40:41], vcc")
            A(f"  v_lshlrev_b32 v{TA}, 2, v{POS}")
            A(f"  ds_write_b32 v{TA}, v{C + i} offset:{LG}")
            A("  s_mov_b64 exec, s[40:41]")
            A(f"  v_add_u32 v{POS}, v{POS}, v{F4 + i}")
            # partial: b ≥ 0, 8 b < n, not full
            A(f"  v_lshlrev_b32 v{TA}, 3, v{C + i}")
            A(f"  v_cmp_gt_i32 vcc, s{S_N}, v{TA}")
            A(f"  v_cmp_le_i32 s[42:43], 0, v{C + i}")
            A("  s_and_b64 vcc, vcc, s[42:43]")
            A(f"  v_cmp_eq_u32 s[42:43], 0, v{F4 + i}")
            A("  s_and_b64 vcc, vcc, s[42:43]")
            A("  s_and_saveexec_b64 s[40:41], vcc")
            A(f"  v_mov_b32 v{TA}, {PART}")
            A(f"  ds_write_b32 v{TA}, v{C + i}")
            A("  s_mov_b64 exec, s[40:41]")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        A(f"  v_mov_b32 v{TA}, {PART}")
        A(f"  ds_read_b32 v{TA}, v{TA}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_readfirstlane_b32 s33, v{TA}")
        A("  s_lshl_b32 s42, s32, 3")
        A("  s_lshl_b32 s43, s33, 3")
        A(f"  s_sub_i32 s43, s{S_N}, s43")
        A("  s_cmp_lt_i32 s33, 0")
        A("  s_cselect_b32 s43, 0, s43")
        A(f"  s_add_u32 s{S_N}, s42, s43")

    def table():
        """gather: the slice's 4 · lg_cap table entries (thread t: k = t + 512 r): compacted index i = 4 g_lo + k,
        block b = CL[i] (i < F), else the partial one (or block 0: a valid dummy past the live columns);
        TBL + 16 k ← (A lo, A hi, delta, b): A = kv + page · page_stride + 2048 gg + 16 t0, the scales at
        A + delta = page + 64 E + 128 gg + 2 t0 (in-page token 8 b % E = 32 gg + t0)."""
        K_, IDX, TA, TB, P0, PG, IP, GG, T0, AP = (RING + i for i in range(10))
        Q = RING + 12  # (A lo, A hi, delta, b): one b128
        A("  s_mov_b32 s48, s36")  # the row's K block table (rsrc)
        A("  s_mov_b32 s49, s37")
        A("  s_mov_b32 s50, s97")
        A("  s_mov_b32 s51, 0x00020000")
        A(f"  s_lshl_b32 s40, s{S_GLO}, 2")
        A("  s_max_i32 s41, s33, 0")
        A("  s_lshl_b32 s42, 1, s99")
        A("  s_sub_u32 s42, s42, 1")  # E - 1
        for r in range(-(-4 * lg_cap // T)):
            A(f"  v_add_u32 v{K_}, {r * T}, v0")
            A(f"  v_add_u32 v{IDX}, s40, v{K_}")
            A(f"  v_lshlrev_b32 v{TA}, 2, v{IDX}")
            A(f"  ds_read_b32 v{Q + 3}, v{TA} offset:{LG}")
            A(f"  v_mov_b32 v{TB}, s41")
            A("  s_waitcnt lgkmcnt(0)")
            A(f"  v_cmp_gt_u32 vcc, s32, v{IDX}")
            A(f"  v_cndmask_b32 v{Q + 3}, v{TB}, v{Q + 3}, vcc")
            A(f"  v_lshlrev_b32 v{P0}, 3, v{Q + 3}")
            A(f"  v_lshrrev_b32 v{TA}, s99, v{P0}")
            A(f"  v_lshlrev_b32 v{TA}, 2, v{TA}")
            A(f"  buffer_load_dword v{PG}, v{TA}, s[48:51], 0 offen")
            A(f"  v_and_b32 v{IP}, s42, v{P0}")
            A(f"  v_lshrrev_b32 v{GG}, 5, v{IP}")
            A(f"  v_and_b32 v{T0}, 31, v{IP}")
            A(f"  v_lshlrev_b32 v{AP}, 11, v{GG}")
            A(f"  v_lshl_add_u32 v{AP}, v{T0}, 4, v{AP}")
            A(f"  v_lshlrev_b32 v{Q + 2}, 7, v{GG}")
            A(f"  v_lshl_add_u32 v{Q + 2}, v{T0}, 1, v{Q + 2}")
            A(f"  v_add_u32 v{Q + 2}, s{S_E64}, v{Q + 2}")
            A(f"  v_sub_u32 v{Q + 2}, v{Q + 2}, v{AP}")
            A(f"  v_add_co_u32 v{Q}, vcc, s90, v{AP}")
            A(f"  v_mov_b32 v{Q + 1}, s91")
            A(f"  v_addc_co_u32 v{Q + 1}, vcc, 0, v{Q + 1}, vcc")
            A("  s_waitcnt vmcnt(0)")
            A(f"  v_mad_u64_u32 v[{Q}:{Q + 1}], s[44:45], v{PG}, s98, v[{Q}:{Q + 1}]")
            A(f"  v_lshlrev_b32 v{TA}, 4, v{K_}")
            A(f"  v_add_u32 v{TA}, {TBL}, v{TA}")
            A(f"  ds_write_b128 v{TA}, v[{Q}:{Q + 3}]")

    # ---------------------------------------------------------------- prologue
    A("  s_load_dwordx16 s[4:19], s[0:1], 0")
    A(f"  s_load_dwordx4 s[20:23], s[0:1], {idx_topk.KA.off['lstride']:#x}")
    A(f"  s_load_dwordx16 s[84:99], s[0:1], {KA.off['q']:#x}")
    A(f"  s_load_dword s{S_SLOG}, s[0:1], {KA.off['slog']:#x}")
    if bsc or gather:  # s[54:55] / s31 (free until the top-k): bs / cand, stride
        A(f"  s_load_dwordx2 s[54:55], s[0:1], {KA_S.off['bs']:#x}")
        A(f"  s_load_dword s31, s[0:1], {KA_S.off['bs_stride']:#x}")
    A(f"  v_and_b32 v{V_L}, 63, v0")
    A(f"  v_lshlrev_b32 v{V_16L}, 4, v{V_L}")
    A(f"  v_lshlrev_b32 v{V_2L}, 1, v{V_L}")
    A(f"  v_lshlrev_b32 v{V_4L}, 2, v{V_L}")
    A(f"  v_readfirstlane_b32 s{S_W}, v0")
    A(f"  s_lshr_b32 s{S_W}, s{S_W}, 6")
    # q at 64 (l % 32) + 16 (l / 32), its scales at 4 (l % 32), w at 16 (l / 32) + 32 j
    A(f"  v_and_b32 v6, 31, v{V_L}")
    A("  v_lshlrev_b32 v7, 2, v6")
    A("  v_lshlrev_b32 v6, 6, v6")
    A(f"  v_lshrrev_b32 v17, 5, v{V_L}")
    A("  v_lshlrev_b32 v17, 4, v17")
    A("  v_add_u32 v6, v6, v17")
    A("  s_waitcnt lgkmcnt(0)")
    # CTA → row, split
    A("  s_and_b32 s28, s2, 7")
    A("  s_lshr_b32 s29, s2, 3")
    A(f"  s_lshl_b32 s{S_SM1}, 1, s{S_SLOG}")
    A(f"  s_sub_u32 s{S_SM1}, s{S_SM1}, 1")  # S - 1
    if wide and split:
        # slice-major: p = 8 j + x = S row + split, so slice s of every row runs on XCD s % 8, the rows in order:
        # rows that share pages (a request's draft rows, a shared prefix) read each slice's pages on one XCD, close
        # together in time (L2), and each XCD only reads its slices' pages
        A("  s_lshl_b32 s24, s29, 3")
        A("  s_add_u32 s24, s24, s28")
        A(f"  s_and_b32 s{S_SPL}, s24, s{S_SM1}")
        A(f"  s_lshr_b32 s{S_ROW}, s24, s{S_SLOG}")
    else:
        A(f"  s_lshr_b32 s30, s29, s{S_SLOG}")
        A("  s_lshl_b32 s30, s30, 3")
        A(f"  s_add_u32 s{S_ROW}, s28, s30")
        A(f"  s_and_b32 s{S_SPL}, s29, s{S_SM1}")
    A(f"  s_cmp_ge_u32 s{S_ROW}, s96")
    A("  s_cbranch_scc1 .Lend")
    if bsc or gather:  # s[54:55] = the row's block scores / candidates
        A(f"  s_mul_i32 s30, s{S_ROW}, s31")
        A(f"  s_mul_hi_u32 s31, s{S_ROW}, s31")
        A("  s_add_u32 s54, s54, s30")
        A("  s_addc_u32 s55, s55, s31")
    A(f"  s_lshl_b32 s24, s{S_ROW}, 2")
    A(f"  s_and_b32 s59, s{S_ROW}, -4")
    A(f"  s_load_dword s{S_N}, s[6:7], s24")
    A(f"  s_load_dword s{S_REQ}, s[10:11], s24")
    A(f"  s_load_dword s{S_VW}, s[8:9], s59")
    # the row's K block table → LDS KP (one entry per thread; num_records = the row)
    A(f"  s_mul_i32 s36, s{S_ROW}, s97")
    A(f"  s_mul_hi_u32 s37, s{S_ROW}, s97")
    A("  s_add_u32 s36, s92, s36")
    A("  s_addc_u32 s37, s93, s37")
    A("  s_and_b32 s37, s37, 0xffff")
    A("  s_mov_b32 s38, s97")
    A("  s_mov_b32 s39, 0x00020000")
    A("  v_lshlrev_b32 v19, 2, v0")
    if not wide:
        A("  buffer_load_dword v18, v19, s[36:39], 0 offen")
    # the row's q, scales, weights
    A(f"  s_lshl_b32 s28, s{S_ROW}, 11")
    A(f"  s_lshr_b32 s29, s{S_ROW}, 21")
    A("  s_add_u32 s30, s84, s28")
    A("  s_addc_u32 s31, s85, s29")
    A(f"  s_lshl_b32 s28, s{S_ROW}, 7")
    A(f"  s_lshr_b32 s29, s{S_ROW}, 25")
    A("  s_add_u32 s32, s86, s28")
    A("  s_addc_u32 s33, s87, s29")
    A("  s_add_u32 s34, s88, s28")
    A("  s_addc_u32 s35, s89, s29")
    A(f"  global_load_dwordx4 v[{QA[0]}:{QA[0] + 3}], v6, s[30:31]")
    A(f"  global_load_dwordx4 v[{QA[1]}:{QA[1] + 3}], v6, s[30:31] offset:32")
    A(f"  global_load_dword v{QS}, v7, s[32:33]")
    for j in range(4):
        A(
            f"  global_load_dwordx4 v[{W + 4 * j}:{W + 4 * j + 3}], v17, s[34:35] offset:{32 * j}"
        )
    # page geometry
    A(f"  s_sub_u32 s{S_GSH}, s99, 5")  # groups per page = 2^gsh
    A(f"  s_lshl_b32 s{S_GMASK}, 1, s{S_GSH}")
    A(f"  s_sub_u32 s{S_GMASK}, s{S_GMASK}, 1")
    A("  s_add_u32 s28, s99, 6")
    A(f"  s_lshl_b32 s{S_E64}, 1, s28")  # E·64: the scales
    A("  s_waitcnt lgkmcnt(0)")
    if gather:
        compact()
    A(f"  s_add_u32 s{S_G}, s{S_N}, 31")
    A(f"  s_lshr_b32 s{S_G}, s{S_G}, 5")
    A(f"  s_add_u32 s60, s{S_G}, s{S_SM1}")
    A(f"  s_lshr_b32 s60, s60, s{S_SLOG}")  # per = ceil(G / S)
    A(f"  s_mul_i32 s{S_GLO}, s{S_SPL}, s60")
    A(f"  s_add_u32 s{S_GHI}, s{S_GLO}, s60")
    A(f"  s_min_u32 s{S_GHI}, s{S_GHI}, s{S_G}")
    A(f"  s_sub_u32 s{S_CNTC}, s{S_GHI}, s{S_GLO}")
    A(f"  s_max_i32 s{S_CNTC}, s{S_CNTC}, 0")  # groups in this slice
    A(f"  s_sub_u32 s61, s{S_CNTC}, s{S_W}")
    A("  s_add_u32 s61, s61, 7")
    A(f"  s_lshr_b32 s{S_NG}, s61, 3")
    A(f"  s_cmp_gt_u32 s{S_CNTC}, s{S_W}")
    A(f"  s_cselect_b32 s{S_NG}, s{S_NG}, 0")  # this wave's groups
    A(f"  s_add_u32 s{S_GLW}, s{S_GLO}, s{S_W}")
    A(f"  s_sub_u32 s{S_NG1}, s{S_NG}, 1")
    if bsc:  # + the slice's first block (4 per group)
        A(f"  s_lshl_b32 s33, s{S_GLO}, 4")
        A("  s_add_u32 s54, s54, s33")
        A("  s_addc_u32 s55, s55, 0")
    if gather:
        table()
        A(f"  s_lshl_b32 s40, s{S_ROW}, 7")
        A(f"  s_load_dword s{S_TAG}, s[94:95], s40 glc")
    elif wide:
        # the slice's K block-table entries from page0 = g_lo >> gsh (KP index = page - page0)
        A(f"  s_lshr_b32 s{S_PG0}, s{S_GLO}, s{S_GSH}")
        A(f"  s_lshl_b32 s40, s{S_PG0}, 2")  # (s38 / s39 hold the page geometry by now)
        A("  s_add_u32 s48, s36, s40")
        A("  s_addc_u32 s49, s37, 0")
        A("  s_and_b32 s49, s49, 0xffff")
        A("  s_sub_i32 s50, s97, s40")
        A("  s_max_i32 s50, s50, 0")
        A("  s_mov_b32 s51, 0x00020000")
        A("  buffer_load_dword v18, v19, s[48:51], 0 offen")
        # shared prefix? this slice's first K page vs rows row ^ 1 and row ^ 2 at the same index (plain loads
        # keep a page in L2 / MALL for the rows that share it; nt streams faster when nobody does)
        A(f"  s_mul_i32 s41, s{S_ROW}, s97")
        A("  s_add_u32 s41, s41, s40")
        A(f"  s_load_dword s{S_SH}, s[92:93], s41")
        for q, r in ((1, 45), (2, 46)):
            A(f"  s_xor_b32 s41, s{S_ROW}, {q}")
            A("  s_cmp_lt_u32 s41, s96")
            A(
                f"  s_cselect_b32 s41, s41, s{S_ROW}"
            )  # past the rows: compare with itself (masked below)
            A("  s_mul_i32 s41, s41, s97")
            A("  s_add_u32 s41, s41, s40")
            A(f"  s_load_dword s{r}, s[92:93], s41")
        A(f"  s_lshl_b32 s40, s{S_ROW}, 7")
        A(f"  s_load_dword s{S_TAG}, s[94:95], s40 glc")
    if wide:
        # the slice's live count n_s = clamp(n - 32 g_lo, 0, 32 groups)
        if not gather:  # (gather: compact() kept the row's n there)
            A(f"  s_mov_b32 s{S_NT}, s{S_N}")
        A(f"  s_lshl_b32 s40, s{S_GLO}, 5")
        A(f"  s_sub_i32 s40, s{S_N}, s40")
        A(f"  s_lshl_b32 s41, s{S_CNTC}, 5")
        A("  s_min_i32 s40, s40, s41")
        A(f"  s_max_i32 s{S_NS}, s40, 0")
        # the coarse histogram (idx_topk's digit layout) → 0
        for r in range(4):
            A(f"  v_mov_b32 v{TV + r}, 0")
        A(f"  v_lshlrev_b32 v{TV + 4}, 4, v0")
        hz = idx_topk.HIST_END
        for lo in range(0, hz, 16 * T):
            if lo + 16 * T > hz:
                A(f"  v_cmp_gt_u32 vcc, {(hz - lo) // 16}, v0")
                A("  s_and_saveexec_b64 s[40:41], vcc")
            A(f"  ds_write_b128 v{TV + 4}, v[{TV}:{TV + 3}] offset:{lo}")
            if lo + 16 * T > hz:
                A("  s_mov_b64 exec, s[40:41]")
    A("  s_waitcnt vmcnt(0)")
    if not gather:
        A(f"  ds_write_b32 v19, v18 offset:{KP}")
    A("  s_waitcnt lgkmcnt(0)")
    if wide and not gather:
        A("  s_mov_b32 s47, 0")
        for q, r in ((1, 45), (2, 46)):
            A(f"  s_xor_b32 s41, s{S_ROW}, {q}")
            A("  s_cmp_lt_u32 s41, s96")
            A("  s_cselect_b32 s41, 1, 0")
            A(f"  s_cmp_eq_u32 s{S_SH}, s{r}")
            A("  s_cselect_b32 s42, 1, 0")
            A("  s_and_b32 s41, s41, s42")
            A("  s_or_b32 s47, s47, s41")
        A(f"  s_mov_b32 s{S_SH}, s47")
    A("  s_barrier")
    if gather:
        # the lane constants; s30 = the wave's first table entry
        A(f"  v_and_b32 v{G_LJ}, 48, v{V_2L}")  # 16 ((l % 32) / 8)
        A(f"  v_lshrrev_b32 v{G_LCS}, 5, v{V_L}")
        A(f"  v_lshlrev_b32 v{G_LCV}, 9, v{G_LCS}")
        A(f"  v_lshlrev_b32 v{G_LCS}, 6, v{G_LCS}")
        A(f"  v_and_b32 v{TV}, 0x70, v{V_16L}")
        A(f"  v_add_u32 v{G_LCV}, v{G_LCV}, v{TV}")  # 512 (l / 32) + 16 (l % 8)
        A(f"  v_and_b32 v{TV}, 14, v{V_2L}")
        A(f"  v_add_u32 v{G_LCS}, v{G_LCS}, v{TV}")  # 64 (l / 32) + 2 (l % 8)
        A(f"  s_lshl_b32 s30, s{S_W}, 6")
        A(f"  s_add_u32 s30, s30, {TBL}")
        A(f"  s_mov_b32 s{S_SH}, 0")
    if not gather:
        # page ids: lane i ← KP[(g_lo + w + 8 i) >> gsh], i < ng
        A(f"  v_lshlrev_b32 v18, 3, v{V_L}")
        A(f"  v_add_u32 v18, s{S_GLW}, v18")
        A(f"  v_lshrrev_b32 v18, s{S_GSH}, v18")
        if wide:
            A(f"  v_subrev_u32 v18, s{S_PG0}, v18")
        A("  v_lshlrev_b32 v18, 2, v18")
        A(f"  v_mov_b32 v{V_PG}, 0")
        A(f"  v_cmp_gt_u32 vcc, s{S_NG}, v{V_L}")
        A("  s_and_saveexec_b64 s[40:41], vcc")
        A(f"  ds_read_b32 v{V_PG}, v18 offset:{KP}")
        A("  s_mov_b64 exec, s[40:41]")
        A("  s_waitcnt lgkmcnt(0)")
    A(
        f"  v_lshrrev_b32 v{TV}, 1, v17"
    )  # 8 (l / 32): lanes 32..63 take scale bytes 1 / 3
    A(f"  v_lshrrev_b32 v{QS}, v{TV}, v{QS}")
    if wide:
        A(f"  v_mov_b32 v{TV + 3}, 1")
        A(f"  s_add_u32 s{S_TAG}, s{S_TAG}, 1")
        A(f"  s_lshl_b32 s{S_TAG}, s{S_TAG}, 24")
        if kshared is None:
            A(f"  s_cmp_eq_u32 s{S_SH}, 0")
            A("  s_cbranch_scc1 .Lknt")
            loop(False, "p", "")  # shared pages: plain loads
            A(".Lknt:")
            loop(False)
        else:
            loop(False, "", kshared if kshared != "none" else "")
        A(".Lldone:")
        A("  s_waitcnt vmcnt(0) lgkmcnt(0)")
        A("  s_barrier")
        if bsc:
            block_scores()
        # the slice's candidates → segment `split` (CAPS 8 B slots) of the row of `logits`
        A(f"  s_mov_b32 s{S_N}, s{S_NS}")
        A(f"  s_lshl_b32 s22, s{S_GLO}, 5")  # the slice's first position
        A(f"  s_mov_b32 s{S_REQ2}, s{S_REQ}")
        A(f"  s_mov_b32 s{S_VW2}, s{S_VW}")
        A(f"  s_mul_i32 s42, s{S_ROW}, s20")
        A(f"  s_mul_hi_u32 s43, s{S_ROW}, s20")
        A(f"  s_mul_i32 s40, s{S_SPL}, {8 * CAPS}")
        A("  s_add_u32 s42, s42, s40")
        A("  s_addc_u32 s43, s43, 0")
        A("  s_add_u32 s14, s4, s42")
        A("  s_addc_u32 s15, s5, s43")
        # the τ slot: row + 8 (S CAPS + split)
        A(f"  s_mul_i32 s42, s{S_ROW}, s20")
        A(f"  s_mul_hi_u32 s43, s{S_ROW}, s20")
        A(f"  s_lshl_b32 s40, {CAPS}, s{S_SLOG}")
        A(f"  s_add_u32 s40, s40, s{S_SPL}")
        A("  s_lshl_b32 s40, s40, 3")
        A("  s_add_u32 s42, s42, s40")
        A("  s_addc_u32 s43, s43, 0")
        A("  s_add_u32 s16, s4, s42")
        A("  s_addc_u32 s17, s5, s43")
        tkf = idx_topk.source(
            8,
            nl,
            row=f"s{S_ROW}",
            lg_lds=LG,
            body_only=True,
            pre=True,
            pfx="f",
            cand_out=True,
            filt=True,
            caps=CAPS,
            fallback=".Lfallback",
            exit_to=".Lwt",
            tau_slot=True,
            remap=TBL + 12 if gather else None,
            tag=f"s{S_TAG}",
        )
        c.extend(tkf["lines"])
        # (the exact fallback sits after the end: a far jump into cold code costs an L2 round trip, ~9k clk
        # with the memory system saturated, so the hot path stays contiguous)
        tka = idx_topk.source(
            8,
            nl,
            row=f"s{S_ROW}",
            lg_lds=LG,
            body_only=True,
            pre=True,
            pfx="a",
            cand_out=True,
            caps=CAPS,
            exit_to=".Lwt",
            tag=f"s{S_TAG}",
            tau_slot=True,
            remap=TBL + 12 if gather else None,
        )
        A(".Lwt:")
        # the merger = the row's last split (dispatched last on its XCD: every CTA it waits for is resident or
        # done); the others leave with their stores in flight
        A(f"  s_lshl_b32 s{S_SM1}, 1, s{S_SLOG}")
        A(f"  s_sub_u32 s{S_SM1}, s{S_SM1}, 1")
        if split:  # no merge here: merge_source's kernel
            A("  s_branch .Lend")
            body = c + [".Lend:", "  s_endpgm", ".Lfallback:"] + tka["lines"]
            nv = max(nv, tkf["vgprs"], tka["vgprs"])
            assert max(tkf["lds"], tka["lds"]) <= DUMMY
            src = kernel_source(
                name, body, ka, lds=lds, vgprs=nv, agprs=0, sgprs=102, block=T
            )
            return dict(
                name=name,
                src=src,
                ka=ka,
                block=T,
                lg_cap=lg_cap,
                lds_s1=True,
                bt_max=bt_max,
                wide=True,
                nlf=nlf,
            )
        A(f"  s_cmp_eq_u32 s{S_SPL}, s{S_SM1}")
        A("  s_cbranch_scc0 .Lend")
        # the merge: the top-512 of the S · CAPS slots (NaN = empty), n = the row's
        A(f"  s_mov_b32 s{S_N}, s{S_NT}")
        A(f"  s_lshl_b32 s{S_CAPS}, {CAPS}, s{S_SLOG}")
        A("  s_load_dwordx16 s[4:19], s[0:1], 0")
        A(f"  s_load_dwordx4 s[20:23], s[0:1], {idx_topk.KA.off['lstride']:#x}")
        A(f"  s_mov_b32 s{S_REQ}, s{S_REQ2}")
        A(f"  s_mov_b32 s{S_VW}, s{S_VW2}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  s_mul_i32 s42, s{S_ROW}, s20")
        A(f"  s_mul_hi_u32 s43, s{S_ROW}, s20")
        A("  s_add_u32 s84, s4, s42")
        A("  s_addc_u32 s85, s5, s43")
        A(f"  s_lshl_b32 s76, 1, s{S_SLOG}")  # S (S_GLO is free by now)
        CL = LG + 4 * bt_max  # the compacted candidates (BT sits at LG)
        assert lds >= CL + 8 * CMAX, (CL, lds)
        tkm = idx_topk.source(
            8,
            nlf,
            bt_max=bt_max,
            row=f"s{S_ROW}",
            body_only=True,
            pre=True,
            pfx="m",
            cand_in="s[84:85]",
            cand_range=f"s{S_CAPS}",
            tag=f"s{S_TAG}",
            front=dict(nseg="s76", cmax=CMAX, clds=CL, to_c=".Lmc", to_b=".Lmb"),
        )
        c.extend(tkm["lines"])
        A(".Lmc:")
        tkc = idx_topk.source(
            8,
            CMAX // 2048,
            bt_max=bt_max,
            row=f"s{S_ROW}",
            lg_lds=CL,
            pos_lds=CL + 4 * CMAX,
            body_only=True,
            pre=True,
            pfx="c",
            bt_n=f"s{S_NT}",
            bt_off=LG,
            exit_to=".Ltagw",
        )
        c.extend(tkc["lines"])
        A(".Lmb:")
        tkb = idx_topk.source(
            8,
            nlf,
            bt_max=bt_max,
            row=f"s{S_ROW}",
            lg_flag="nt",
            body_only=True,
            pre=True,
            pfx="b",
            cand_in="s[84:85]",
            bt_n=f"s{S_NT}",
            bt_off=LG,
            cand_range=f"s{S_CAPS}",
            tag=f"s{S_TAG}",
            exit_to=".Ltagw",
        )
        assert max(tkf["lds"], tka["lds"]) <= DUMMY and tkb["lds"] <= lds, (
            tkb["lds"],
            lds,
        )
        tagw = [
            ".Ltagw:",  # the row's tag for the next call
            f"  s_cmp_eq_u32 s{S_W}, 0",
            "  s_cbranch_scc0 .Lend",
            f"  s_lshr_b32 s{S_TAG}, s{S_TAG}, 24",
            f"  s_lshl_b32 s40, s{S_ROW}, 7",
            "  s_mov_b64 exec, 1",
            f"  v_mov_b32 v0, s{S_TAG}",
            "  v_mov_b32 v1, s40",
            "  global_store_dword v1, v0, s[94:95] sc0 sc1",
        ]
        body = (
            c
            + tkb["lines"]
            + tagw
            + [".Lend:", "  s_endpgm", ".Lfallback:"]
            + tka["lines"]
        )
        nv = max(
            nv, tkf["vgprs"], tka["vgprs"], tkb["vgprs"], tkm["vgprs"], tkc["vgprs"]
        )
        src = kernel_source(
            name, body, ka, lds=lds, vgprs=nv, agprs=0, sgprs=102, block=T
        )
        return dict(
            name=name,
            src=src,
            ka=ka,
            block=T,
            wmax=tkb["wmax"],
            lg_cap=lg_cap,
            lds_s1=True,
            bt_max=bt_max,
            wide=True,
            nlf=nlf,
        )
    if lds_s1:  # S = 1: the logits stay in LDS
        A(f"  s_cmp_eq_u32 s{S_SLOG}, 0")
        A("  s_cbranch_scc0 .Lglb")
        loop(False)
        A(".Lldone:")
        A("  s_waitcnt vmcnt(0) lgkmcnt(0)")
        A("  s_barrier")
        A("  s_branch .Ltopk")
    # ---------------------------------------------------------------- S > 1: logits → the global row, ticket
    A(".Lglb:")
    A(f"  s_mul_i32 s62, s{S_ROW}, s20")
    A(f"  s_mul_hi_u32 s63, s{S_ROW}, s20")
    A("  s_add_u32 s62, s4, s62")
    A("  s_addc_u32 s63, s5, s63")
    loop(True)
    A(".Lgdone:")
    A("  s_waitcnt vmcnt(0)")  # this wave's stores are in L2
    ticket_cta()
    # ---------------------------------------------------------------- the top-512 (idx_topk)
    A(".Ltopk:")
    tk = idx_topk.source(
        8,
        nl,
        bt_max=bt_max,
        row=f"s{S_ROW}",
        lg_lds=LG if lds_s1 else None,
        lg_sel=f"s{S_SLOG}" if lds_s1 else None,
        lg_flag="nt",
        body_only=True,
        pre=True,
    )
    assert tk["lds"] <= DUMMY
    body = c + tk["lines"] + [".Lend:", "  s_endpgm"]
    nv = max(nv, tk["vgprs"])
    src = kernel_source(name, body, ka, lds=lds, vgprs=nv, agprs=0, sgprs=102, block=T)
    return dict(
        name=name,
        src=src,
        ka=ka,
        block=T,
        wmax=tk["wmax"],
        lg_cap=lg_cap,
        lds_s1=lds_s1,
        bt_max=bt_max,
    )


def merge_source(nlf: int, bt_max: int = 4096, name: str | None = None):
    """The wide rows' merge as its own kernel (after source(wide=True, split=True)): one CTA of 512 per row;
    kernargs KA (as the main kernel's). Every candidate slot is complete (stream order), so the polls pass at
    once; the τ_row compaction, the exact top-512 of the compacted list (the full list past CMAX), the lists,
    then the row's tag bump."""
    name = name or f"idx_merge_w{nlf}_b{bt_max}"
    T = 512
    LGm = -(-idx_topk.BT // 128) * 128
    CL = LGm + 4 * bt_max
    lds = CL + 8 * CMAX
    c = []
    A = c.append
    A("  s_load_dwordx16 s[4:19], s[0:1], 0")
    A(f"  s_load_dwordx4 s[20:23], s[0:1], {idx_topk.KA.off['lstride']:#x}")
    A(f"  s_load_dwordx16 s[84:99], s[0:1], {KA.off['q']:#x}")
    A(f"  s_load_dword s{S_SLOG}, s[0:1], {KA.off['slog']:#x}")
    A(f"  v_readfirstlane_b32 s{S_W}, v0")
    A(f"  s_lshr_b32 s{S_W}, s{S_W}, 6")
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_mov_b32 s{S_ROW}, s2")
    A(f"  s_cmp_ge_u32 s{S_ROW}, s96")
    A("  s_cbranch_scc1 .Lend")
    A(f"  s_lshl_b32 s24, s{S_ROW}, 2")
    A(f"  s_and_b32 s29, s{S_ROW}, -4")
    A(f"  s_load_dword s{S_NT}, s[6:7], s24")
    A(f"  s_load_dword s{S_REQ}, s[10:11], s24")
    A(f"  s_load_dword s{S_VW}, s[8:9], s29")
    A(f"  s_lshl_b32 s40, s{S_ROW}, 7")
    A(f"  s_load_dword s{S_TAG}, s[94:95], s40 glc")
    A(f"  s_mul_i32 s42, s{S_ROW}, s20")
    A(f"  s_mul_hi_u32 s43, s{S_ROW}, s20")
    A("  s_add_u32 s84, s4, s42")
    A("  s_addc_u32 s85, s5, s43")
    A(f"  s_lshl_b32 s{S_CAPS}, {CAPS}, s{S_SLOG}")
    A(f"  s_lshl_b32 s76, 1, s{S_SLOG}")
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_add_u32 s{S_TAG}, s{S_TAG}, 1")
    A(f"  s_lshl_b32 s{S_TAG}, s{S_TAG}, 24")
    A(f"  s_mov_b32 s{S_N}, s{S_NT}")
    tkm = idx_topk.source(
        8,
        nlf,
        bt_max=bt_max,
        row=f"s{S_ROW}",
        body_only=True,
        pre=True,
        pfx="m",
        cand_in="s[84:85]",
        cand_range=f"s{S_CAPS}",
        tag=f"s{S_TAG}",
        front=dict(nseg="s76", cmax=CMAX, clds=CL, to_c=".Lmc", to_b=".Lmb"),
    )
    c.extend(tkm["lines"])
    A(".Lmc:")
    tkc = idx_topk.source(
        8,
        CMAX // 2048,
        bt_max=bt_max,
        row=f"s{S_ROW}",
        lg_lds=CL,
        pos_lds=CL + 4 * CMAX,
        body_only=True,
        pre=True,
        pfx="c",
        bt_n=f"s{S_NT}",
        bt_off=LGm,
        exit_to=".Ltagw",
    )
    c.extend(tkc["lines"])
    A(".Lmb:")
    tkb = idx_topk.source(
        8,
        nlf,
        bt_max=bt_max,
        row=f"s{S_ROW}",
        lg_flag="nt",
        body_only=True,
        pre=True,
        pfx="b",
        cand_in="s[84:85]",
        bt_n=f"s{S_NT}",
        bt_off=LGm,
        cand_range=f"s{S_CAPS}",
        tag=f"s{S_TAG}",
        exit_to=".Ltagw",
    )
    c.extend(tkb["lines"])
    assert max(tkm["lds"], tkc["lds"], tkb["lds"]) <= lds
    c += [
        ".Ltagw:",  # the row's tag for the next call
        f"  s_cmp_eq_u32 s{S_W}, 0",
        "  s_cbranch_scc0 .Lend",
        f"  s_lshr_b32 s{S_TAG}, s{S_TAG}, 24",
        f"  s_lshl_b32 s40, s{S_ROW}, 7",
        "  s_mov_b64 exec, 1",
        f"  v_mov_b32 v0, s{S_TAG}",
        "  v_mov_b32 v1, s40",
        "  global_store_dword v1, v0, s[94:95] sc0 sc1",
        ".Lend:",
        "  s_endpgm",
    ]
    nv = max(tkm["vgprs"], tkc["vgprs"], tkb["vgprs"])
    src = kernel_source(name, c, KA, lds=lds, vgprs=nv, agprs=0, sgprs=102, block=T)
    return dict(name=name, src=src, ka=KA, block=T)
