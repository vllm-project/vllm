# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Sparse MLA decode for the attention half, v1: keys split
across CTAs for small M, QK per wave and tile, PV split by head dim, shared
deferred-rescale softmax. Same inputs and output as v0 (attn_dec.py; the
record layout, ragged lists and MXFP8 output are described there).

Grid: 8·S·ceil(M/8) CTAs of 4 waves, S = 2^slog key splits per token. CTA b:
XCD x = b % 8, j = b / 8, token t = x + 8 (j >> slog), split s = j % S, so
all splits of a token run on one XCD (their combine stays in its L2).

A token's tiles (16 keys; SWA tiles, then top-k tiles, as v0) are cut into S
equal ranges [lo, hi). The CTA walks its range in rounds of 4 tiles; wave w
takes tile lo + 4r + w of round r. Per round (2 barriers):
- wave w: wait for its tile (3 tiles in flight per wave, VGPRs); per 64-dim
  NoPE piece: dequantise, V rows (bf16) → LDS slot r % 2 and the piece's 2
  QK MFMAs (S = K·qᵀ over all 512 dims, as v0), interleaved one piece behind;
  scores, tile max per head → LDS;  B1;
- softmax of round r: every wave reads the 4 tile maxima and keeps the same
  per-head reference m. m moves only when a tile max exceeds it by more than
  TAU (log2 units, so P ≤ 2^TAU). Interleaved with it, PV of round r − 1
  (still at the old m): wave w owns dims 128w .. 128w + 127 of Oᵀ (8 blocks ×
  4 AGPRs), over that round's 64 keys: `v_mfma_f32_16x16x32_bf16` with K
  slot 8G + 4u + j = key 4G + j of tile 2i + u (A = Vᵀ from two
  `ds_read_b64_tr_b16`, B = P of two tiles, both in the S layout), and
  tile r + 3's loads (spread out: 4 waves issuing 40 gathers at once cost
  ~800 cycles of TA queueing). Then O
  and l are rescaled if m moved (32 AGPRs), P = exp2(S − m) (bf16) → LDS;  B2.
End: l summed over the 4 waves; S = 1: O·f / (l f + exp2(sink·log2e − M)),
M = max(m, sink·log2e), then v0's epilogue (bf16, inverse RoPE in wave 3,
MXFP8 per 32 dims). S > 1: each CTA stores (m, l, O) to `part` (plain
stores, same XCD), `global_atomic_inc` on `cnt[t]` wrapping at S − 1
(graph-replay safe, no reset); the last arriver loads the other splits'
partials with `nt` (bypasses vL1; same XCD, so L2 is coherent), merges and
runs the epilogue.
"""

from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source

HEADS, D, NOPE = 16, 512, 448
ROW, PITCH = 576, 1056
VT = 16 * PITCH  # one V tile (16 keys × 512 bf16)
VSLOT = 4 * VT  # one round's 4 tiles
PBUF = 2 * VSLOT  # P: [slot][lane][tile] 8 B
MAXB = PBUF + 2 * 2048  # tile maxima: [slot][wave][head] f32
LBUF = MAXB + 2 * 256  # l per wave: [wave][head]
FLAG = LBUF + 256  # last-arriver broadcast
CSB = FLAG + 16  # cos/sin row (256 B, LDS-DMA in the prologue)
LDS = CSB + 256
MAXR = 10  # rounds (40 tiles at S = 1)
TAU = 8.0
PSTRIDE = 32768 + 128  # partial: O [32 blocks][64 lanes][16 B], m [16], l [16]

KA = KArgs(
    ("q", "ptr"),
    ("swa", "ptr"),
    ("comp", "ptr"),
    ("swa_idx", "ptr"),
    ("swa_ptr", "ptr"),
    ("tk_idx", "ptr"),
    ("tk_ptr", "ptr"),
    ("sink", "ptr"),
    ("pos", "ptr"),
    ("cs", "ptr"),
    ("out", "ptr"),
    ("outs", "ptr"),
    ("part", "ptr"),
    ("cnt", "ptr"),
    ("dbg", "ptr"),
    ("m", "u32"),
    ("swa_stride", "u32"),
    ("swa_bsl", "u32"),
    ("comp_stride", "u32"),
    ("comp_bsl", "u32"),
    ("has_comp", "u32"),
    ("qk_scale", "u32"),
    ("slog", "u32"),
)

KA_RR = KArgs(*KA.args, ("ss", "ptr"), ("eps", "u32"))
KA_RRS = KArgs(*KA_RR.args, ("rstd", "ptr"))  # rr_scale "rstd"

# SGPRs: s[0:1] kernargs, s2 workgroup; s[4:41] kernargs (see the loads)
S_Q, S_SWA, S_COMP, S_SIDX, S_SPTR, S_TIDX, S_TPTR, S_SINK = 4, 6, 8, 10, 12, 14, 16, 18
S_POSP, S_CS, S_OUT, S_OUTS, S_PART, S_CNTP, S_DBG = 20, 22, 24, 26, 28, 30, 32
S_M, S_SWST, S_SWBSL, S_CMST, S_CMBSL, S_HASC, S_QKS, S_SLOG = (
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
)
S_W, S_TOK, S_SPL, S_NS, S_NT, S_TS, S_TT, S_LO, S_HI = (
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
)
S_SB, S_TB = 52, 54  # swa_ptr[t], [t+1]; tk_ptr[t], [t+1]
TK_STRIDE = 512  # tk_fixed: entries per token
S_CNT = 56  # 10 per-tile key counts
S_POS = 66  # 2
S_R = 68  # rounds

# VGPRs
V_TID, V_L, V_R, V_G, V_G4 = 0, 1, 2, 3, 4
V_W1, V_W2 = 5, 7  # V tile writes, slot 0 / slot 1 (pairs)
V_TR = 9  # TR reads, slot 0 / 1 (pair)
V_SLOT = 11  # 10 slot ids
V_DA, V_SA, V_T = 22, 24, 26  # address pairs (even), temps 26..32
RAW = (34, 72, 110)  # 3 record buffers: 28 NoPE, 8 RoPE, 2 scale
DEQ = 148  # 56 dequantised NoPE; PV's TR ring (3 × 8) reuses it
OB = 170  # end: O (32), above the 4 split partials loaded at RAW[0] + 34 k
V_PB = 204  # 8: P of the round (B operands)
V_S, V_P, V_PK = 212, 216, 220
V_M, V_LS, V_AL = 222, 223, 224
V_X = 225  # 7 temps
V_NINF, V_CAND, V_SINK, V_PBA, V_MXA = 232, 233, 234, 235, 236
V_Z = 238  # 4 zeros
VGPRS = 244
A_O, A_Q = 0, 32  # Oᵀ a[0:31], q a[32:95]
AGPRS = 96


def f32(v: float) -> str:
    import struct

    return f"{struct.unpack('<I', struct.pack('<f', v))[0]:#x}"


def source(
    name: str = "attn_dec1",
    tk_fixed: bool = False,
    spec: bool = False,
):
    """The MXFP8 output in gemv8's frag layout (wo_a's x; buffers for 256
    tokens). S > 1: the last arriver merges all S partials ("ticket").
    tk_fixed: the top-k list of token t starts at
    tk_idx[t·TK_STRIDE] and tk_ptr holds the counts (lens[t]) instead of an
    indptr; entries past a count are not read.
    q (the front without qkv_norm) comes in unnormalised
    (wq_b of MXFP8(bf16(x · q_norm.weight))) and without the attention norm's
    rstd; the token's rr = rstd · rsq(rstd² · Σ / 1280 + eps) from wqkv's 40 Σx²
    partials (ss `[t][40]`, fixed-order wave sum) and rstd[t] f32 multiplies
    qk_scale (scores are linear in q). Extra kernargs ss, eps, rstd (KA_RRS).
    spec: the slot-id loads go out with the indptr loads,
    for full lists (SWA at 128 t, 128 keys; top-k at 512 t, 512 keys if
    has_comp), ahead of q; once the indptrs land they are checked, and a
    mismatch drains and reloads (the old path).
    The split partials' O in bf16 (16 KB: `[wave][4 × 1 KB][lane 16 B]`,
    m / l f32 at 16 KB + 4 head, + 64), and the last arriver packs its own O from
    registers instead of reloading it; the merge stays in split order on the
    same bf16 values (deterministic)."""
    RRV = RAW[2]  # rr_scale temps (free until tile 2's records)
    ka = KA_RRS
    if spec:
        ka = KArgs(*ka.args, ("sw_cap", "u32"), ("tk_cap", "u32"))
    c = []
    A = c.append

    def nop(n):
        while n > 0:
            A(f"  s_nop {min(n, 8) - 1}")
            n -= 8

    def ladd(dst, a, b):
        """64-bit SGPR pair dst = pair a + 32-bit SGPR / constant b."""
        A(f"  s_add_u32 s{dst}, s{a}, {b}")
        A(f"  s_addc_u32 s{dst + 1}, s{a + 1}, 0")

    # ---------------- prologue
    A("  s_load_dwordx16 s[4:19], s[0:1], 0x0")
    A("  s_load_dwordx16 s[20:35], s[0:1], 0x40")
    A("  s_load_dwordx4 s[36:39], s[0:1], 0x80")
    A("  s_load_dwordx2 s[40:41], s[0:1], 0x90")
    if spec:
        A(
            f"  s_load_dwordx2 s[100:101], s[0:1], {ka.off['sw_cap']:#x}"
        )  # index array sizes (entries)
        A(f"  s_load_dwordx4 s[96:99], s[0:1], {KA_RR.off['ss']:#x}")  # ss, eps (s98)
        A(f"  s_load_dwordx2 s[88:89], s[0:1], {KA_RRS.off['rstd']:#x}")
    A(f"  v_readfirstlane_b32 s{S_W}, v0")
    A(f"  s_lshr_b32 s{S_W}, s{S_W}, 6")
    A(f"  v_and_b32 v{V_L}, 63, v0")
    A(f"  v_and_b32 v{V_R}, 15, v{V_L}")
    A(f"  v_lshrrev_b32 v{V_G}, 4, v{V_L}")
    A(f"  v_lshlrev_b32 v{V_G4}, 2, v{V_G}")
    A("  s_waitcnt lgkmcnt(0)")
    # token / split
    A(f"  s_and_b32 s{S_TOK}, s2, 7")
    A(f"  s_lshr_b32 s{S_SPL}, s2, 3")
    A(f"  s_lshr_b32 s45, s{S_SPL}, s{S_SLOG}")
    A("  s_lshl_b32 s45, s45, 3")
    A(f"  s_add_u32 s{S_TOK}, s{S_TOK}, s45")
    A(f"  s_lshl_b32 s45, 1, s{S_SLOG}")
    A("  s_sub_u32 s45, s45, 1")
    A(f"  s_and_b32 s{S_SPL}, s{S_SPL}, s45")
    A(f"  s_cmp_ge_u32 s{S_TOK}, s{S_M}")
    A("  s_cbranch_scc1 .Lexit")
    A(f"  s_lshl_b32 s45, s{S_TOK}, 2")
    A(f"  s_load_dwordx2 s[{S_SB}:{S_SB + 1}], s[{S_SPTR}:{S_SPTR + 1}], s45")
    A(f"  s_mov_b64 s[{S_TB}:{S_TB + 1}], 0")
    A(f"  s_cmp_eq_u32 s{S_HASC}, 0")
    A("  s_cbranch_scc1 .Lnocomp")
    if tk_fixed:
        A(f"  s_load_dword s{S_TB + 1}, s[{S_TPTR}:{S_TPTR + 1}], s45")  # lens[t]
    else:
        A(f"  s_load_dwordx2 s[{S_TB}:{S_TB + 1}], s[{S_TPTR}:{S_TPTR + 1}], s45")
    A(".Lnocomp:")
    A(f"  s_lshl_b32 s45, s{S_TOK}, 3")
    A(f"  s_load_dwordx2 s[{S_POS}:{S_POS + 1}], s[{S_POSP}:{S_POSP + 1}], s45")
    # spec: s56 LO, s57 HI, s58 SWA start, s59 top-k start, s61 top-k count, s63 tiles (the S_CNT SGPRs, live later)
    if spec:
        # the rr partial first (lanes < 40: Σ partial l of token t, ss + t·160 + 4 l)
        A(f"  s_mul_i32 s45, s{S_TOK}, 160")
        A("  s_add_u32 s96, s96, s45")
        A("  s_addc_u32 s97, s97, 0")
        A(f"  s_lshl_b32 s45, s{S_TOK}, 2")
        A("  s_load_dword s90, s[88:89], s45")  # rstd[t]
        A(f"  v_min_u32 v{RRV}, 39, v{V_L}")
        A(f"  v_lshlrev_b32 v{RRV}, 2, v{RRV}")
        A(f"  global_load_dword v{RRV + 1}, v{RRV}, s[96:97]")
        A("  s_sub_u32 s69, s100, 1")  # last valid entries
        A("  s_sub_u32 s3, s101, 1")
        A(
            f"  s_lshl_b32 s59, s{S_TOK}, 9"
        )  # (before the compare: SALU shifts write SCC)
        A(f"  s_cmp_eq_u32 s{S_HASC}, 0")
        A("  s_cselect_b32 s63, 8, 40")
        A("  s_cselect_b32 s61, 0, 512")
        if not tk_fixed:
            A("  s_cselect_b32 s59, 0, s59")  # no comp: tk_ptr isn't read, TB = 0
        A(f"  s_mul_i32 s56, s{S_SPL}, s63")
        A(f"  s_lshr_b32 s56, s56, s{S_SLOG}")
        A(f"  s_add_u32 s57, s{S_SPL}, 1")
        A("  s_mul_i32 s57, s57, s63")
        A(f"  s_lshr_b32 s57, s57, s{S_SLOG}")
        A(f"  s_lshl_b32 s58, s{S_TOK}, 7")
        for r in range(MAXR):
            A(f"  s_add_u32 s70, s56, s{S_W}")
            A(f"  s_add_u32 s70, s70, {4 * r}")
            A("  s_cmp_ge_u32 s70, s57")
            A("  s_cbranch_scc1 .Lspec_done")
            A("  s_lshl_b32 s71, s70, 4")
            A("  s_cmp_lt_u32 s70, 8")
            A("  s_cselect_b32 s72, s58, s59")
            A("  s_cselect_b32 s73, 0, 128")
            A(
                f"  s_cselect_b64 s[78:79], s[{S_SIDX}:{S_SIDX + 1}], s[{S_TIDX}:{S_TIDX + 1}]"
            )
            A("  s_cselect_b32 s74, s69, s3")
            A("  s_add_u32 s71, s71, s72")
            A("  s_sub_u32 s71, s71, s73")  # entry of key 0
            A(f"  v_add_u32 v{V_T}, s71, v{V_R}")
            A(
                f"  v_min_u32 v{V_T}, s74, v{V_T}"
            )  # a wrong guess may run past the arrays
            A(f"  v_lshlrev_b32 v{V_T}, 2, v{V_T}")
            A(f"  v_mov_b32 v{V_T + 1}, 0")
            A(f"  v_lshl_add_u64 v[{V_T}:{V_T + 1}], v[{V_T}:{V_T + 1}], 0, s[78:79]")
            A(f"  global_load_dword v{V_SLOT + r}, v[{V_T}:{V_T + 1}], off")
        A(".Lspec_done:")
    # q → AGPRs in the MFMA step order (lane: head r, group g), as v0
    A(f"  s_lshl_b32 s45, s{S_TOK}, 14")
    A(f"  s_add_u32 s80, s{S_Q}, s45")
    A(f"  s_addc_u32 s81, s{S_Q + 1}, 0")
    A(f"  v_lshlrev_b32 v{V_T}, 10, v{V_R}")
    A(f"  v_lshl_add_u32 v{V_T}, v{V_G}, 5, v{V_T}")
    A(f"  v_mov_b32 v{V_T + 1}, 0")
    A(f"  v_lshl_add_u64 v[{V_DA}:{V_DA + 1}], v[{V_T}:{V_T + 1}], 0, s[80:81]")
    for kk in range(14):
        off = 128 * (kk // 2) + 16 * (kk % 2)
        A(
            f"  global_load_dwordx4 a[{A_Q + 4 * kk}:{A_Q + 4 * kk + 3}], v[{V_DA}:{V_DA + 1}], off offset:{off}"
        )
    A(f"  v_lshlrev_b32 v{V_T}, 10, v{V_R}")
    A(f"  v_lshl_add_u32 v{V_T}, v{V_G}, 4, v{V_T}")
    A(f"  v_lshl_add_u64 v[{V_SA}:{V_SA + 1}], v[{V_T}:{V_T + 1}], 0, s[80:81]")
    for h in range(2):
        kk = 14 + h
        A(
            f"  global_load_dwordx4 a[{A_Q + 4 * kk}:{A_Q + 4 * kk + 3}], v[{V_SA}:{V_SA + 1}], off offset:{896 + 64 * h}"
        )
    A(f"  v_lshlrev_b32 v{V_T}, 2, v{V_R}")
    A(f"  global_load_dword v{V_SINK}, v{V_T}, s[{S_SINK}:{S_SINK + 1}]")
    if not spec:  # lanes < 40: Σ partial l of token t (ss + t·160 + 4 l)
        A(f"  s_load_dwordx4 s[96:99], s[0:1], {KA_RR.off['ss']:#x}")  # ss, eps (s98)
        A(f"  s_load_dwordx2 s[88:89], s[0:1], {KA_RRS.off['rstd']:#x}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  s_mul_i32 s45, s{S_TOK}, 160")
        A("  s_add_u32 s96, s96, s45")
        A("  s_addc_u32 s97, s97, 0")
        A(f"  s_lshl_b32 s45, s{S_TOK}, 2")
        A("  s_load_dword s90, s[88:89], s45")  # rstd[t]
        A(f"  v_min_u32 v{RRV}, 39, v{V_L}")
        A(f"  v_lshlrev_b32 v{RRV}, 2, v{RRV}")
        A(f"  global_load_dword v{RRV + 1}, v{RRV}, s[96:97]")
    A("  s_waitcnt lgkmcnt(0)")
    # cos/sin row of this token → LDS (wave 3, lanes 0..15)
    A(f"  s_cmp_eq_u32 s{S_W}, 3")
    A("  s_cbranch_scc0 .Lnocs")
    A(f"  s_lshl_b64 s[80:81], s[{S_POS}:{S_POS + 1}], 8")
    A(f"  s_add_u32 s80, s{S_CS}, s80")
    A(f"  s_addc_u32 s81, s{S_CS + 1}, s81")
    A(f"  s_mov_b32 m0, {CSB}")
    A(f"  v_lshlrev_b32 v{V_T}, 4, v{V_L}")
    A("  s_mov_b64 s[84:85], exec")
    A("  s_mov_b64 exec, 0xffff")
    A(f"  global_load_lds_dwordx4 v{V_T}, s[80:81]")
    A("  s_mov_b64 exec, s[84:85]")
    A(".Lnocs:")
    if tk_fixed:  # [t·TK_STRIDE, t·TK_STRIDE + lens[t]) (no comp: lens 0)
        A(f"  s_lshl_b32 s{S_TB}, s{S_TOK}, {TK_STRIDE.bit_length() - 1}")
        A(f"  s_add_u32 s{S_TB + 1}, s{S_TB + 1}, s{S_TB}")
    # tiles: TS SWA, TT total; this split: [LO, HI); rounds R
    A(f"  s_sub_u32 s{S_NS}, s{S_SB + 1}, s{S_SB}")
    A(f"  s_sub_u32 s{S_NT}, s{S_TB + 1}, s{S_TB}")
    A(f"  s_add_u32 s{S_TS}, s{S_NS}, 15")
    A(f"  s_lshr_b32 s{S_TS}, s{S_TS}, 4")
    A(f"  s_add_u32 s{S_TT}, s{S_NT}, 15")
    A(f"  s_lshr_b32 s{S_TT}, s{S_TT}, 4")
    A(f"  s_add_u32 s{S_TT}, s{S_TT}, s{S_TS}")
    A(f"  s_mul_i32 s{S_LO}, s{S_SPL}, s{S_TT}")
    A(f"  s_lshr_b32 s{S_LO}, s{S_LO}, s{S_SLOG}")
    A(f"  s_add_u32 s{S_HI}, s{S_SPL}, 1")
    A(f"  s_mul_i32 s{S_HI}, s{S_HI}, s{S_TT}")
    A(f"  s_lshr_b32 s{S_HI}, s{S_HI}, s{S_SLOG}")
    A(f"  s_sub_u32 s{S_R}, s{S_HI}, s{S_LO}")
    A(f"  s_add_u32 s{S_R}, s{S_R}, 3")
    A(f"  s_lshr_b32 s{S_R}, s{S_R}, 2")
    # LDS addresses: V writes (wave w's region of each slot), TR reads (wave w's dims), P, maxima
    A(f"  s_mul_i32 s45, s{S_W}, {VT}")
    A(f"  v_mul_u32_u24 v{V_W1}, {PITCH}, v{V_R}")
    A(f"  v_add_u32 v{V_W1}, s45, v{V_W1}")
    A(f"  v_lshl_add_u32 v{V_W2}, v{V_G}, 4, v{V_W1}")
    A(f"  v_add_u32 v{V_W2}, {2 * NOPE}, v{V_W2}")
    A(f"  v_lshl_add_u32 v{V_W1}, v{V_G}, 5, v{V_W1}")
    A(f"  v_add_u32 v{V_W1 + 1}, {VSLOT}, v{V_W1}")
    A(f"  v_add_u32 v{V_W2 + 1}, {VSLOT}, v{V_W2}")
    A(f"  v_lshrrev_b32 v{V_T}, 2, v{V_R}")
    A(f"  v_add_u32 v{V_T}, v{V_G4}, v{V_T}")  # row 4G + (r >> 2)
    A(f"  v_mul_u32_u24 v{V_TR}, {PITCH}, v{V_T}")
    A(f"  v_and_b32 v{V_T}, 3, v{V_R}")
    A(f"  v_lshl_add_u32 v{V_TR}, v{V_T}, 3, v{V_TR}")
    A(f"  s_lshl_b32 s45, s{S_W}, 8")  # dims 128w: 256 w bytes
    A(f"  v_add_u32 v{V_TR}, s45, v{V_TR}")
    A(f"  v_add_u32 v{V_TR + 1}, {VSLOT}, v{V_TR}")
    A(f"  v_lshlrev_b32 v{V_PBA}, 5, v{V_L}")
    A(f"  v_add_u32 v{V_PBA}, {PBUF}, v{V_PBA}")  # P reads: PBUF + 32 l; writes + 8 w
    A(f"  v_lshlrev_b32 v{V_MXA}, 2, v{V_R}")
    A(f"  v_add_u32 v{V_MXA}, {MAXB}, v{V_MXA}")  # maxima: MAXB + 4 head (+ 64 w)

    def tile_scalars(r):
        """Wave tile r → s70 = j (token tile), s71 = first entry, s72 = key
        count (≤ 16; garbage if j ≥ HI), s[74:75] cache base, s76 page stride,
        s77 log2 rows per page, s[78:79] index list."""
        A(f"  s_add_u32 s70, s{S_LO}, s{S_W}")
        A(f"  s_add_u32 s70, s70, {4 * r}")
        A("  s_lshl_b32 s72, s70, 4")  # first key if SWA
        A(f"  s_sub_u32 s71, s70, s{S_TS}")
        A("  s_lshl_b32 s71, s71, 4")  # first key if top-k
        A(
            f"  s_cmp_lt_u32 s70, s{S_TS}"
        )  # SCC = SWA tile; only s_cselect until the last use
        A("  s_cselect_b32 s71, s72, s71")
        A(f"  s_cselect_b32 s72, s{S_NS}, s{S_NT}")
        A(f"  s_cselect_b32 s73, s{S_SB}, s{S_TB}")
        A(f"  s_cselect_b64 s[74:75], s[{S_SWA}:{S_SWA + 1}], s[{S_COMP}:{S_COMP + 1}]")
        A(f"  s_cselect_b32 s76, s{S_SWST}, s{S_CMST}")
        A(f"  s_cselect_b32 s77, s{S_SWBSL}, s{S_CMBSL}")
        A(
            f"  s_cselect_b64 s[78:79], s[{S_SIDX}:{S_SIDX + 1}], s[{S_TIDX}:{S_TIDX + 1}]"
        )
        A("  s_sub_u32 s72, s72, s71")
        A("  s_min_u32 s72, s72, 16")  # count
        A("  s_add_u32 s71, s71, s73")  # entry index

    if spec:  # s45 = 1 if the guess was wrong
        A(f"  s_cmp_eq_u32 s{S_NS}, 128")
        A("  s_cselect_b32 s45, 0, 1")
        for real, guess in ((S_NT, 61), (S_SB, 58), (S_TB, 59)):
            A(f"  s_cmp_eq_u32 s{real}, s{guess}")
            A("  s_cselect_b32 s45, s45, 1")
    for r in range(MAXR):
        A(f"  s_mov_b32 s{S_CNT + r}, 0")
    if spec:
        A("  s_cmp_eq_u32 s45, 0")
        A("  s_cbranch_scc1 .Lslots_cnt")
        A("  s_waitcnt vmcnt(0)")  # wrong guess: drain, then the regular loads
    for r in range(MAXR):
        tile_scalars(r)
        A(f"  s_cmp_ge_u32 s70, s{S_HI}")
        A("  s_cbranch_scc1 .Lslots_done")
        A(f"  s_mov_b32 s{S_CNT + r}, s72")
        A("  s_sub_u32 s73, s72, 1")
        A(f"  v_min_u32 v{V_T}, s73, v{V_R}")
        A(f"  v_add_u32 v{V_T}, s71, v{V_T}")
        A(f"  v_lshlrev_b32 v{V_T}, 2, v{V_T}")
        A(f"  v_mov_b32 v{V_T + 1}, 0")
        A(f"  v_lshl_add_u64 v[{V_T}:{V_T + 1}], v[{V_T}:{V_T + 1}], 0, s[78:79]")
        A(f"  global_load_dword v{V_SLOT + r}, v[{V_T}:{V_T + 1}], off")
    if spec:  # right guess: the slot ids are in flight already, only the counts
        A("  s_branch .Lslots_done")
        A(".Lslots_cnt:")
        for r in range(MAXR):
            tile_scalars(r)
            A(f"  s_cmp_ge_u32 s70, s{S_HI}")
            A("  s_cbranch_scc1 .Lslots_done")
            A(f"  s_mov_b32 s{S_CNT + r}, s72")
    A(".Lslots_done:")
    if spec:  # wrong guess: the regular slot loads must land; right: only the rr partial and the slot ids
        A("  s_cmp_eq_u32 s45, 0")
        A("  s_cbranch_scc1 .Lspec_ok")
        A("  s_waitcnt vmcnt(0)")
        A(".Lspec_ok:")
    for i in range(A_O, A_O + 32):
        A(f"  v_accvgpr_write_b32 a{i}, 0")
    A(f"  v_mov_b32 v{V_M}, 0xf149f2ca")  # -1e30
    A(f"  v_mov_b32 v{V_LS}, 0")
    A(f"  v_mov_b32 v{V_NINF}, 0xff800000")
    for i in range(4):
        A(f"  v_mov_b32 v{V_Z + i}, 0")
    A(
        f"  s_waitcnt vmcnt({17 if spec else 0})"
    )  # spec: q (16) and sink may still be in flight
    x, y = RRV + 1, RRV + 2
    A(f"  v_cmp_gt_u32 vcc, 40, v{V_L}")
    A(f"  v_cndmask_b32 v{x}, 0, v{x}, vcc")
    for ctl in (
        "quad_perm:[1,0,3,2]",
        "quad_perm:[2,3,0,1]",
        "row_ror:4",
        "row_ror:8",
    ):
        A(f"  v_mov_b32 v{y}, v{x}")
        nop(2)
        A(f"  v_add_f32_dpp v{x}, v{y}, v{x} {ctl} row_mask:0xf bank_mask:0xf")
    for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32"):
        A(f"  v_mov_b32 v{y}, v{x}")
        nop(2)
        A(f"  {sw} v{x}, v{y}")
        nop(4)
        A(f"  v_add_f32 v{x}, v{x}, v{y}")
    q, b_, r_, e_ = RRV + 3, RRV + 4, RRV + 5, RRV + 6
    A(f"  v_mov_b32 v{b_}, {f32(1280.0)}")
    A(f"  v_rcp_f32 v{r_}, v{b_}")
    A("  s_nop 0")
    A(f"  v_fma_f32 v{e_}, -v{b_}, v{r_}, 1.0")
    A(f"  v_fma_f32 v{r_}, v{e_}, v{r_}, v{r_}")
    A(f"  v_mul_f32 v{q}, v{x}, v{r_}")
    A(f"  v_fma_f32 v{e_}, -v{b_}, v{q}, v{x}")
    A(f"  v_fma_f32 v{q}, v{e_}, v{r_}, v{q}")
    A(f"  v_fma_f32 v{e_}, -v{b_}, v{q}, v{x}")
    A(f"  v_fma_f32 v{q}, v{e_}, v{r_}, v{q}")
    A(f"  v_mul_f32 v{q}, s90, v{q}")
    A(f"  v_mul_f32 v{q}, s90, v{q}")
    A(f"  v_add_f32 v{q}, s98, v{q}")
    A(f"  v_rsq_f32 v{q}, v{q}")
    A("  s_nop 0")
    A(f"  v_mul_f32 v{q}, s90, v{q}")
    A(f"  v_mul_f32 v{q}, s{S_QKS}, v{q}")
    nop(1)  # VALU → v_readfirstlane: 1 wait state
    A(f"  v_readfirstlane_b32 s{S_QKS}, v{q}")
    A("  s_nop 4")

    def exists(r, label):
        A(f"  s_cmp_eq_u32 s{S_CNT + r}, 0")
        A(f"  s_cbranch_scc1 {label}")

    def load(r):
        """Wave tile r's 10 record loads into RAW[r % 3] (caller checks it exists)."""
        raw = RAW[r % 3]
        tile_scalars(r)
        A("  s_lshl_b32 s80, 1, s77")
        A("  s_sub_u32 s80, s80, 1")  # row mask
        A(f"  s_lshl_b32 s81, {ROW}, s77")  # scale array offset
        A(f"  v_max_i32 v{V_T}, 0, v{V_SLOT + r}")
        A(f"  v_lshrrev_b32 v{V_T + 1}, s77, v{V_T}")  # block
        A(f"  v_and_b32 v{V_T + 2}, s80, v{V_T}")  # row
        A(f"  v_mov_b32 v{V_T + 6}, s76")
        A(
            f"  v_mad_u64_u32 v[{V_DA}:{V_DA + 1}], s[86:87], v{V_T + 1}, v{V_T + 6}, s[74:75]"
        )
        A(f"  v_mov_b32 v{V_SA}, v{V_DA}")
        A(f"  v_mov_b32 v{V_SA + 1}, v{V_DA + 1}")
        A(f"  v_mul_u32_u24 v{V_T + 4}, {ROW}, v{V_T + 2}")
        A(f"  v_lshl_add_u32 v{V_T + 4}, v{V_G}, 4, v{V_T + 4}")
        A(f"  v_mov_b32 v{V_T + 5}, 0")
        A(
            f"  v_lshl_add_u64 v[{V_DA}:{V_DA + 1}], v[{V_T + 4}:{V_T + 5}], 0, v[{V_DA}:{V_DA + 1}]"
        )
        A(f"  v_lshl_add_u32 v{V_T + 4}, v{V_T + 2}, 3, s81")
        A(
            f"  v_lshl_add_u64 v[{V_SA}:{V_SA + 1}], v[{V_T + 4}:{V_T + 5}], 0, v[{V_SA}:{V_SA + 1}]"
        )
        for j in range(7):
            A(
                f"  global_load_dwordx4 v[{raw + 4 * j}:{raw + 4 * j + 3}], v[{V_DA}:{V_DA + 1}], off offset:{64 * j}"
            )
        for h in range(2):
            rr = raw + 28 + 4 * h
            A(
                f"  global_load_dwordx4 v[{rr}:{rr + 3}], v[{V_DA}:{V_DA + 1}], off offset:{NOPE + 64 * h}"
            )
        A(f"  global_load_dwordx2 v[{raw + 36}:{raw + 37}], v[{V_SA}:{V_SA + 1}], off")

    def qk_a(r):
        """Wait for tile r; per NoPE piece: dequantise (VALU), V rows → LDS
        slot r % 2 and the piece's 2 QK MFMAs, each one piece behind its
        converts so the three units overlap."""
        raw, slot = RAW[r % 3], r % 2
        if r + 2 < MAXR:
            A(f"  s_cmp_eq_u32 s{S_CNT + r + 2}, 0")
            A(f"  s_cbranch_scc1 .Lw1_{r}")
            A("  s_waitcnt vmcnt(20)")
            A(f"  s_branch .Lwd_{r}")
            A(f".Lw1_{r}:")
        if r + 1 < MAXR:
            A(f"  s_cmp_eq_u32 s{S_CNT + r + 1}, 0")
            A(f"  s_cbranch_scc1 .Lw0_{r}")
            A("  s_waitcnt vmcnt(10)")
            A(f"  s_branch .Lwd_{r}")
            A(f".Lw0_{r}:")
        A("  s_waitcnt vmcnt(0)")
        A(f".Lwd_{r}:")
        w1, w2 = V_W1 + slot, V_W2 + slot

        def mfma(kk):
            a = (
                DEQ + 8 * (kk // 2) + 4 * (kk % 2)
                if kk < 14
                else raw + 28 + 4 * (kk - 14)
            )
            cin = "0" if kk == 0 else f"v[{V_S}:{V_S + 3}]"
            A(
                f"  v_mfma_f32_16x16x32_bf16 v[{V_S}:{V_S + 3}], v[{a}:{a + 3}], a[{A_Q + 4 * kk}:{A_Q + 4 * kk + 3}], {cin}"
            )

        def vw(j, half):
            A(
                f"  ds_write_b128 v{w1}, v[{DEQ + 8 * j + 4 * half}:{DEQ + 8 * j + 4 * half + 3}] offset:{128 * j + 16 * half}"
            )

        for j in range(7):
            sc = V_X + 2 + (j % 2)
            A(f"  v_bfe_u32 v{sc}, v{raw + 36 + j // 4}, {8 * (j % 4)}, 8")
            A(f"  v_lshlrev_b32 v{sc}, 23, v{sc}")
            behind = (
                [
                    lambda j=j: mfma(2 * j - 2),
                    lambda j=j: vw(j - 1, 0),
                    lambda j=j: mfma(2 * j - 1),
                    lambda j=j: vw(j - 1, 1),
                ]
                if j > 0
                else []
            )
            for k in range(4):
                src = raw + 4 * j + k
                A(f"  v_cvt_scalef32_pk_bf16_fp8 v{DEQ + 8 * j + 2 * k}, v{src}, v{sc}")
                A(
                    f"  v_cvt_scalef32_pk_bf16_fp8 v{DEQ + 8 * j + 2 * k + 1}, v{src}, v{sc} op_sel:[1,0,0]"
                )
                if behind:
                    behind.pop(0)()
        nop(2)
        mfma(12)
        vw(6, 0)
        mfma(13)
        vw(6, 1)
        mfma(14)
        A(f"  ds_write_b128 v{w2}, v[{raw + 28}:{raw + 31}]")
        mfma(15)
        A(f"  ds_write_b128 v{w2}, v[{raw + 32}:{raw + 35}] offset:64")

    def qk_b(r):
        """Scores in log2 units (invalid keys → -inf) and the tile max per head → v[V_X]."""
        A(f"  v_cmp_gt_u32 vcc, s{S_CNT + r}, v{V_R}")
        A(f"  v_cmp_le_i32 s[82:83], 0, v{V_SLOT + r}")
        A("  s_and_b64 s[82:83], vcc, s[82:83]")  # valid keys, one bit per lane r
        nop(18)  # last QK MFMA → VALU read
        for i in range(4):
            A(f"  v_mul_f32 v{V_S + i}, s{S_QKS}, v{V_S + i}")
            A(f"  v_add_u32 v{V_X}, {i}, v{V_G4}")
            A(f"  v_bfe_u32 v{V_X}, s82, v{V_X}, 1")
            A(f"  v_cmp_ne_u32 vcc, 0, v{V_X}")
            A(f"  v_cndmask_b32 v{V_S + i}, v{V_NINF}, v{V_S + i}, vcc")
        x, y = V_X, V_X + 1
        A(f"  v_max3_f32 v{x}, v{V_S}, v{V_S + 1}, v{V_S + 2}")
        A(f"  v_max_f32 v{x}, v{x}, v{V_S + 3}")
        for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32"):
            A(f"  v_mov_b32 v{y}, v{x}")
            nop(2)
            A(f"  {sw} v{x}, v{y}")
            nop(4)
            A(f"  v_max_f32 v{x}, v{x}, v{y}")

    def no_tile(r):
        """No tile for this wave in round r: zero V rows, S = -inf."""
        slot = r % 2
        w1, w2 = V_W1 + slot, V_W2 + slot
        for j in range(7):
            A(f"  ds_write_b128 v{w1}, v[{V_Z}:{V_Z + 3}] offset:{128 * j}")
            A(f"  ds_write_b128 v{w1}, v[{V_Z}:{V_Z + 3}] offset:{128 * j + 16}")
        for h in range(2):
            A(f"  ds_write_b128 v{w2}, v[{V_Z}:{V_Z + 3}] offset:{64 * h}")
        for i in range(4):
            A(f"  v_mov_b32 v{V_S + i}, v{V_NINF}")
        A(f"  v_mov_b32 v{V_X}, v{V_NINF}")

    def pv(slot, side=()):
        """Oᵀ (dims 128w ..) += Vᵀ · Pᵀ over the 4 tiles of a slot; `side`:
        asm lines spread between the MFMA pairs. No older LDS
        ops may be outstanding (the lgkm waits count only this function's)."""
        side = list(side)
        tr = V_TR + slot
        A(f"  ds_read_b128 v[{V_PB}:{V_PB + 3}], v{V_PBA} offset:{2048 * slot}")
        A(
            f"  ds_read_b128 v[{V_PB + 4}:{V_PB + 7}], v{V_PBA} offset:{2048 * slot + 16}"
        )

        def issue(bb):
            ring = DEQ + 8 * (bb % 3)
            for tt in range(4):
                A(
                    f"  ds_read_b64_tr_b16 v[{ring + 2 * tt}:{ring + 2 * tt + 1}], v{tr} offset:{VT * tt + 32 * bb}"
                )

        for bb in range(2):
            issue(bb)
        per = -(-len(side) // 8) if side else 0
        for bb in range(8):
            A(f"  s_waitcnt lgkmcnt({4 * min(1, 7 - bb)})")
            ring = DEQ + 8 * (bb % 3)
            for i in range(2):
                A(
                    f"  v_mfma_f32_16x16x32_bf16 a[{A_O + 4 * bb}:{A_O + 4 * bb + 3}], v[{ring + 4 * i}:{ring + 4 * i + 3}], "
                    f"v[{V_PB + 4 * i}:{V_PB + 4 * i + 3}], a[{A_O + 4 * bb}:{A_O + 4 * bb + 3}]"
                )
            if bb + 2 < 8:
                issue(bb + 2)
            for _ in range(per):
                if side:
                    A(side.pop(0))
        c.extend(side)

    def lines(fn, *args):
        """The asm lines fn(*args) would emit."""
        n = len(c)
        fn(*args)
        out = c[n:]
        del c[n:]
        return out

    def phase_b(r, ld: bool, tag: str):
        """After B1: the round's softmax (shared m with deferred rescale) and
        P → LDS, with PV of round r - 1 and (ld) tile r + 3's loads
        interleaved; O and l are rescaled after PV(r - 1), which still used
        the old m."""
        slot = r % 2
        for w in range(4):
            A(f"  ds_read_b32 v{V_X + 2 + w}, v{V_MXA} offset:{256 * slot + 64 * w}")
        A("  s_waitcnt lgkmcnt(0)")
        L = [
            f"  v_max3_f32 v{V_CAND}, v{V_X + 2}, v{V_X + 3}, v{V_X + 4}",
            f"  v_max_f32 v{V_CAND}, v{V_CAND}, v{V_X + 5}",
            f"  v_add_f32 v{V_X}, {f32(TAU)}, v{V_M}",
            f"  v_cmp_gt_f32 s[82:83], v{V_CAND}, v{V_X}",  # heads whose m moves
            f"  v_cndmask_b32 v{V_X}, v{V_M}, v{V_CAND}, s[82:83]",
            f"  v_sub_f32 v{V_AL}, v{V_M}, v{V_X}",
            f"  v_exp_f32 v{V_AL}, v{V_AL}",
            f"  v_mov_b32 v{V_M}, v{V_X}",
            f"  v_mul_f32 v{V_LS}, v{V_LS}, v{V_AL}",
        ]
        for i in range(4):
            L += [
                f"  v_sub_f32 v{V_P + i}, v{V_S + i}, v{V_M}",
                f"  v_exp_f32 v{V_P + i}, v{V_P + i}",
            ]
        L.append("  s_nop 0")  # trans result: 1 wait state before a VALU use
        L += [f"  v_add_f32 v{V_LS}, v{V_LS}, v{V_P + i}" for i in range(4)]
        L += [
            f"  v_cvt_pk_bf16_f32 v{V_PK}, v{V_P}, v{V_P + 1}",
            f"  v_cvt_pk_bf16_f32 v{V_PK + 1}, v{V_P + 2}, v{V_P + 3}",
        ]
        if ld:  # loads and softmax alternate (they use disjoint registers; SCC only inside load)
            Ld = lines(load, r + 3)
            M_, out = L, []
            while Ld or M_:
                out += Ld[:2]
                Ld = Ld[2:]
                out += M_[:1]
                M_ = M_[1:]
            L = out
        if r > 0:
            pv((r - 1) % 2, L)
        else:
            c.extend(L)
        # rescale O only if some head's m moved (rare after the first round)
        A("  s_cmp_eq_u64 s[82:83], 0")
        A(f"  s_cbranch_scc1 .Lnos{r}{tag}")
        nop(20)  # last PV MFMA → accvgpr_read
        for i in range(0, 32, 4):
            for k in range(4):
                A(f"  v_accvgpr_read_b32 v{V_PB + k}, a{A_O + i + k}")
            for k in range(4):
                A(f"  v_mul_f32 v{V_PB + k}, v{V_PB + k}, v{V_AL}")
            for k in range(4):
                A(f"  v_accvgpr_write_b32 a{A_O + i + k}, v{V_PB + k}")
        A(f".Lnos{r}{tag}:")
        A(f"  s_lshl_b32 s45, s{S_W}, 3")
        A(f"  v_add_u32 v{V_X}, s45, v{V_PBA}")
        A(f"  ds_write_b64 v{V_X}, v[{V_PK}:{V_PK + 1}] offset:{2048 * slot}")

    # ---------------- rounds
    for r in range(MAXR):
        if r < 3:
            if r == 0:
                exists(0, ".Lr0")
            else:
                exists(r, ".Lr0")
            load(r)
    for r in range(MAXR):
        A(f".Lr{r}:")
        A(f"  s_cmp_le_u32 s{S_R}, {r}")
        A(f"  s_cbranch_scc1 {'.Lepi' if r == 0 else f'.Lfin{(r - 1) % 2}'}")
        exists(r, f".Lnt{r}")
        qk_a(r)
        qk_b(r)
        A(f"  s_branch .Ltm{r}")
        A(f".Lnt{r}:")
        no_tile(r)
        A(f".Ltm{r}:")
        A(f"  v_cmp_eq_u32 vcc, 0, v{V_G}")
        A("  s_and_saveexec_b64 s[84:85], vcc")
        A(f"  s_lshl_b32 s45, s{S_W}, 6")
        A(f"  v_add_u32 v{V_X + 1}, s45, v{V_MXA}")
        A(f"  ds_write_b32 v{V_X + 1}, v{V_X} offset:{256 * (r % 2)}")
        A("  s_mov_b64 exec, s[84:85]")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
        if r + 3 < MAXR:
            exists(r + 3, f".Lpbn{r}")
            phase_b(r, True, "l")
            A(f"  s_branch .Lpbd{r}")
            A(f".Lpbn{r}:")
            phase_b(r, False, "n")
            A(f".Lpbd{r}:")
        else:
            phase_b(r, False, "n")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_barrier")
    for p in (1, 0):
        A(f".Lfin{p}:")
        pv(p)
        if p == 1:
            A("  s_branch .Lepi")
    A(".Lepi:")
    nop(16)
    # l over the 4 lane groups, then over the waves (LBUF)
    x, y = V_X, V_X + 1
    A(f"  v_mov_b32 v{x}, v{V_LS}")
    for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32"):
        A(f"  v_mov_b32 v{y}, v{x}")
        nop(2)
        A(f"  {sw} v{x}, v{y}")
        nop(4)
        A(f"  v_add_f32 v{x}, v{x}, v{y}")
    A(f"  s_lshl_b32 s45, s{S_W}, 6")
    A(f"  v_lshlrev_b32 v{V_T}, 2, v{V_R}")
    A(f"  v_add_u32 v{V_T}, s45, v{V_T}")
    A(f"  v_add_u32 v{V_T}, {LBUF}, v{V_T}")
    A(f"  v_cmp_eq_u32 vcc, 0, v{V_G}")
    A("  s_and_saveexec_b64 s[84:85], vcc")
    A(f"  ds_write_b32 v{V_T}, v{x}")
    A("  s_mov_b64 exec, s[84:85]")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    A(f"  v_lshlrev_b32 v{V_T}, 2, v{V_R}")
    A(f"  v_add_u32 v{V_T}, {LBUF}, v{V_T}")
    for w in range(4):
        A(f"  ds_read_b32 v{V_X + 2 + w}, v{V_T} offset:{64 * w}")
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  v_add_f32 v{V_LS}, v{V_X + 2}, v{V_X + 3}")
    A(f"  v_add_f32 v{V_LS}, v{V_LS}, v{V_X + 4}")
    A(f"  v_add_f32 v{V_LS}, v{V_LS}, v{V_X + 5}")  # this CTA's l (per head)
    A(f"  v_mul_f32 v{V_SINK}, {f32(1.4426950408889634)}, v{V_SINK}")

    def merge_bf16(pfl):
        """Last arriver, part_bf16: split k's packed O / m / l → v[34 + 18 k ..] (own: packed from OB,
        m, l), merged in split order into OB → .Lnorm."""
        for k in range(4):
            area = RAW[0] + 18 * k
            A(f"  s_cmp_lt_u32 s73, {k}")
            A("  s_cbranch_scc1 .Lpb_done")
            A(f"  s_cmp_eq_u32 s{S_SPL}, {k}")
            A(f"  s_cbranch_scc1 .Lpb_own{k}")
            A(f"  s_lshl_b32 s71, s{S_TOK}, s{S_SLOG}")
            A(f"  s_add_u32 s71, s71, {k}")
            A(f"  s_mul_i32 s71, s71, {PSTRIDE}")
            A(f"  s_add_u32 s80, s{S_PART}, s71")
            A(f"  s_addc_u32 s81, s{S_PART + 1}, 0")
            A(f"  s_lshl_b32 s45, s{S_W}, 12")
            A(f"  v_lshlrev_b32 v{V_T}, 4, v{V_L}")
            A(f"  v_add_u32 v{V_T}, s45, v{V_T}")
            for j in range(4):
                A(
                    f"  global_load_dwordx4 v[{area + 4 * j}:{area + 4 * j + 3}], v{V_T}, s[80:81] offset:{1024 * j}{pfl}"
                )
            A(f"  v_lshlrev_b32 v{V_T + 1}, 2, v{V_R}")
            A(f"  v_add_u32 v{V_T + 1}, 16384, v{V_T + 1}")
            A(f"  global_load_dword v{area + 16}, v{V_T + 1}, s[80:81]{pfl}")
            A(f"  global_load_dword v{area + 17}, v{V_T + 1}, s[80:81] offset:64{pfl}")
            A(f"  s_branch .Lpb_next{k}")
            A(f".Lpb_own{k}:")
            for i in range(16):
                A(f"  v_cvt_pk_bf16_f32 v{area + i}, v{OB + 2 * i}, v{OB + 2 * i + 1}")
            A(f"  v_mov_b32 v{area + 16}, v{V_M}")
            A(f"  v_mov_b32 v{area + 17}, v{V_LS}")
            A(f".Lpb_next{k}:")
        A(".Lpb_done:")
        A("  s_waitcnt vmcnt(0)")
        mx = V_X
        A(f"  v_mov_b32 v{mx}, v{V_SINK}")
        for k in range(4):
            A(f"  s_cmp_lt_u32 s73, {k}")
            A("  s_cbranch_scc1 .Lpbmx_done")
            A(f"  v_max_f32 v{mx}, v{mx}, v{RAW[0] + 18 * k + 16}")
        A(".Lpbmx_done:")
        A(f"  v_sub_f32 v{V_AL}, v{V_SINK}, v{mx}")
        A(f"  v_exp_f32 v{V_AL}, v{V_AL}")  # den, from the sink
        lo, hi = V_X + 3, V_X + 4
        for k in range(4):
            area = RAW[0] + 18 * k
            A(f"  s_cmp_lt_u32 s73, {k}")
            A("  s_cbranch_scc1 .Lpbmg_done")
            A(f"  v_sub_f32 v{V_X + 2}, v{area + 16}, v{mx}")
            A(f"  v_exp_f32 v{V_X + 2}, v{V_X + 2}")
            A("  s_nop 0")
            A(f"  v_fmac_f32 v{V_AL}, v{area + 17}, v{V_X + 2}")
            for i in range(16):
                A(f"  v_lshlrev_b32 v{lo}, 16, v{area + i}")
                A(f"  v_and_b32 v{hi}, 0xffff0000, v{area + i}")
                if k == 0:
                    A(f"  v_mul_f32 v{OB + 2 * i}, v{lo}, v{V_X + 2}")
                    A(f"  v_mul_f32 v{OB + 2 * i + 1}, v{hi}, v{V_X + 2}")
                else:
                    A(f"  v_fmac_f32 v{OB + 2 * i}, v{lo}, v{V_X + 2}")
                    A(f"  v_fmac_f32 v{OB + 2 * i + 1}, v{hi}, v{V_X + 2}")
        A(".Lpbmg_done:")
        A(f"  v_rcp_f32 v{V_AL}, v{V_AL}")
        A("  s_nop 0")
        A("  s_branch .Lnorm")

    # O (AGPRs) → OB[0:32] (block-major, 4 per block)
    for i in range(32):
        A(f"  v_accvgpr_read_b32 v{OB + i}, a{A_O + i}")
    A(f"  s_cmp_eq_u32 s{S_SLOG}, 0")
    A("  s_cbranch_scc1 .Lsingle")
    # ---- S > 1: store the partial, take a ticket
    A(f"  s_lshl_b32 s45, s{S_TOK}, s{S_SLOG}")
    A(f"  s_add_u32 s45, s45, s{S_SPL}")  # partial index t·S + s
    A(f"  s_mul_i32 s45, s45, {PSTRIDE}")
    A(f"  s_add_u32 s80, s{S_PART}, s45")
    A(f"  s_addc_u32 s81, s{S_PART + 1}, 0")
    A(f"  s_lshl_b32 s45, s{S_W}, 13")  # 8 blocks × 1 KB
    A(f"  v_lshlrev_b32 v{V_T}, 4, v{V_L}")
    A(f"  v_add_u32 v{V_T}, s45, v{V_T}")
    A(f"  v_add_u32 v{V_T + 2}, 4096, v{V_T}")
    PK = RAW[0]  # bf16 partials: 16 packed dwords per lane
    A(f"  s_lshl_b32 s45, s{S_W}, 12")  # 4 × 1 KB per wave
    A(f"  v_lshlrev_b32 v{V_T}, 4, v{V_L}")
    A(f"  v_add_u32 v{V_T}, s45, v{V_T}")
    for i in range(16):
        A(f"  v_cvt_pk_bf16_f32 v{PK + i}, v{OB + 2 * i}, v{OB + 2 * i + 1}")
    for j in range(4):
        A(
            f"  global_store_dwordx4 v{V_T}, v[{PK + 4 * j}:{PK + 4 * j + 3}], s[80:81] offset:{1024 * j}"
        )
    MLO = 16384  # m / l of the partial
    A(f"  s_cmp_eq_u32 s{S_W}, 0")
    A("  s_cbranch_scc0 .Lml_done")
    A(f"  v_lshlrev_b32 v{V_T + 1}, 2, v{V_R}")
    A(f"  v_add_u32 v{V_T + 1}, {MLO}, v{V_T + 1}")
    A(f"  v_cmp_eq_u32 vcc, 0, v{V_G}")
    A("  s_and_saveexec_b64 s[84:85], vcc")
    A(f"  global_store_dword v{V_T + 1}, v{V_M}, s[80:81]")
    A(f"  global_store_dword v{V_T + 1}, v{V_LS}, s[80:81] offset:64")
    A("  s_mov_b64 exec, s[84:85]")
    A(".Lml_done:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_barrier")
    A(f"  s_cmp_eq_u32 s{S_W}, 0")
    A("  s_cbranch_scc0 .Lticket_wait")
    A(f"  s_lshl_b32 s45, 1, s{S_SLOG}")
    A("  s_sub_u32 s45, s45, 1")
    A(f"  v_mov_b32 v{V_X}, s45")
    A(f"  s_lshl_b32 s45, s{S_TOK}, 2")
    A(f"  v_mov_b32 v{V_X + 1}, s45")
    A("  s_mov_b64 s[84:85], exec")
    A("  s_mov_b64 exec, 1")
    A(
        f"  global_atomic_inc v{V_X + 2}, v{V_X + 1}, v{V_X}, s[{S_CNTP}:{S_CNTP + 1}] sc0 sc1"
    )
    A("  s_waitcnt vmcnt(0)")
    A(f"  v_mov_b32 v{V_X + 3}, {FLAG}")
    A(f"  ds_write_b32 v{V_X + 3}, v{V_X + 2}")
    A("  s_mov_b64 exec, s[84:85]")
    A("  s_waitcnt lgkmcnt(0)")
    A(".Lticket_wait:")
    A("  s_barrier")
    A(f"  v_mov_b32 v{V_X + 3}, {FLAG}")
    A(f"  ds_read_b32 v{V_X + 2}, v{V_X + 3}")
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  v_readfirstlane_b32 s45, v{V_X + 2}")
    A(f"  s_lshl_b32 s73, 1, s{S_SLOG}")
    A("  s_sub_u32 s73, s73, 1")
    A("  s_cmp_eq_u32 s45, s73")
    A("  s_cbranch_scc0 .Ldone")
    # last arriver: all S partials (its own too) in split order, so the sum
    # doesn't depend on which split came last (deterministic output)
    pfl = " nt"  # same XCD: plain stores + nt / sc1 loads are coherent
    merge_bf16(pfl)
    # ---- S = 1: f = exp2(m - M) / (l exp2(m - M) + exp2(sink' - M))
    A(".Lsingle:")
    mx = V_X
    A(f"  v_max_f32 v{mx}, v{V_M}, v{V_SINK}")
    A(f"  v_sub_f32 v{V_AL}, v{V_SINK}, v{mx}")
    A(f"  v_exp_f32 v{V_AL}, v{V_AL}")
    A(f"  v_sub_f32 v{V_X + 1}, v{V_M}, v{mx}")
    A(f"  v_exp_f32 v{V_X + 1}, v{V_X + 1}")
    A("  s_nop 0")
    A(f"  v_fmac_f32 v{V_AL}, v{V_LS}, v{V_X + 1}")
    A(f"  v_rcp_f32 v{V_AL}, v{V_AL}")
    A("  s_nop 0")
    A(f"  v_mul_f32 v{V_AL}, v{V_AL}, v{V_X + 1}")
    A(".Lnorm:")
    for i in range(32):
        A(f"  v_mul_f32 v{OB + i}, v{OB + i}, v{V_AL}")

    def bf16_round(v):
        A(f"  v_cvt_pk_bf16_f32 v{v}, v{v}, v{v}")
        A(f"  v_lshlrev_b32 v{v}, 16, v{v}")

    for i in range(32):
        bf16_round(OB + i)
    # inverse RoPE on wave 3 (blocks 4..7 = dims 448..511), pairs (i, i+1); cos/sin from LDS
    A(f"  s_cmp_eq_u32 s{S_W}, 3")
    A("  s_cbranch_scc0 .Lnorope")
    A(f"  v_lshlrev_b32 v{V_T}, 3, v{V_G}")  # 4 · 2 g bytes
    A(f"  v_add_u32 v{V_T}, {CSB}, v{V_T}")
    for b in range(4):
        A(
            f"  ds_read_b64 v[{RAW[1] + 4 * b}:{RAW[1] + 4 * b + 1}], v{V_T} offset:{32 * b}"
        )
        A(
            f"  ds_read_b64 v[{RAW[1] + 4 * b + 2}:{RAW[1] + 4 * b + 3}], v{V_T} offset:{128 + 32 * b}"
        )
    A("  s_waitcnt lgkmcnt(0)")
    for b in range(4):
        for pr in range(2):
            e, o = OB + 4 * (4 + b) + 2 * pr, OB + 4 * (4 + b) + 2 * pr + 1
            cs_, sn = RAW[1] + 4 * b + pr, RAW[1] + 4 * b + 2 + pr
            A(f"  v_mul_f32 v{V_X}, v{e}, v{cs_}")
            A(f"  v_fmac_f32 v{V_X}, v{o}, v{sn}")  # e cos + o sin
            A(f"  v_mul_f32 v{V_X + 1}, v{o}, v{cs_}")
            A(f"  v_fma_f32 v{V_X + 1}, -v{e}, v{sn}, v{V_X + 1}")  # o cos − e sin
            A(f"  v_mov_b32 v{e}, v{V_X}")
            A(f"  v_mov_b32 v{o}, v{V_X + 1}")
            bf16_round(e)
            bf16_round(o)
    A(".Lnorope:")
    # MXFP8 per 32 dims (blocks 2c, 2c + 1), 4 lane groups share a block (v0's code)
    # gemv8's frag layout of o [M, 8192] (wo_a's x, 64 K tiles): K = 512 head + dim,
    # lane (head r, g), block pair c, half: tile kg = 4 r + w, half (c / 2), q' = (2c + half) % 4,
    # byte 4 g → ((j·64 + 4r + w)·2 + c/2)·1 KB + (t%16 + 16 q')·16 + 4 g; scales (j·64 + kg)·64 + t%16 + 16 c
    A(f"  s_lshr_b32 s45, s{S_TOK}, 4")
    A("  s_lshl_b32 s45, s45, 6")
    A(f"  s_add_u32 s45, s45, s{S_W}")  # j·64 + w
    A("  s_lshl_b32 s73, s45, 11")
    A(f"  s_and_b32 s86, s{S_TOK}, 15")
    A("  s_lshl_b32 s87, s86, 4")
    A("  s_add_u32 s73, s73, s87")
    A(f"  s_add_u32 s80, s{S_OUT}, s73")
    A(f"  s_addc_u32 s81, s{S_OUT + 1}, 0")
    A(f"  v_lshlrev_b32 v{V_T}, 13, v{V_R}")  # 4 r · 2 KB
    A(f"  v_add_u32 v{V_T}, v{V_G4}, v{V_T}")
    A("  s_lshl_b32 s73, s45, 6")
    A("  s_add_u32 s73, s73, s86")
    A(f"  s_add_u32 s82, s{S_OUTS}, s73")
    A(f"  s_addc_u32 s83, s{S_OUTS + 1}, 0")
    A(f"  v_lshlrev_b32 v{V_T + 1}, 8, v{V_R}")  # 4 r · 64
    for cpair in range(4):
        y0 = OB + 8 * cpair
        a, e, mt, ng = V_X, V_X + 1, V_X + 2, V_X + 3
        A(f"  v_max_f32 v{a}, |v{y0}|, |v{y0 + 1}|")
        for i in range(2, 8):
            A(f"  v_max_f32 v{a}, v{a}, |v{y0 + i}|")
        for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32"):
            A(f"  v_mov_b32 v{e}, v{a}")
            nop(2)
            A(f"  {sw} v{a}, v{e}")
            nop(4)
            A(f"  v_max_f32 v{a}, v{a}, v{e}")
        A(f"  v_max_f32 v{a}, 0x00800000, v{a}")  # FLT_MIN floor
        A(f"  v_frexp_exp_i32_f32 v{e}, v{a}")
        A(f"  v_frexp_mant_f32 v{mt}, v{a}")
        A(f"  v_cmp_lt_f32 vcc, 0x3f600000, v{mt}")
        A(f"  v_addc_co_u32 v{e}, vcc, -9, v{e}, vcc")
        A(f"  v_max_i32 v{e}, 0xffffff81, v{e}")
        A(f"  v_min_i32 v{e}, 0x7f, v{e}")
        A(f"  v_sub_u32 v{ng}, 0, v{e}")
        for i in range(8):
            A(f"  v_ldexp_f32 v{y0 + i}, v{y0 + i}, v{ng}")
        for half in range(2):
            q = RAW[0] + 2 * cpair + half
            yy = y0 + 4 * half
            A(f"  v_cvt_pk_fp8_f32 v{q}, v{yy}, v{yy + 1}")
            A(f"  v_cvt_pk_fp8_f32 v{q}, v{yy + 2}, v{yy + 3} op_sel:[0,0,1]")
            o_ = 1024 * (cpair // 2) + 256 * ((2 * cpair + half) % 4)
            A(f"  global_store_dword v{V_T}, v{q}, s[80:81] offset:{o_}")
        A(f"  v_add_u32 v{e}, 0x7f, v{e}")
        A(f"  v_cmp_eq_u32 vcc, 0, v{V_G}")
        A("  s_and_saveexec_b64 s[84:85], vcc")
        A(f"  global_store_byte v{V_T + 1}, v{e}, s[82:83] offset:{16 * cpair}")
        A("  s_mov_b64 exec, s[84:85]")
    A(".Ldone:")
    A(".Lexit:")
    A("  s_endpgm")
    src = kernel_source(
        name, c, ka, lds=LDS, vgprs=VGPRS, agprs=AGPRS, sgprs=102, block=256
    )
    return dict(name=name, src=src, ka=ka)
