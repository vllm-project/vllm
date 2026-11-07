# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Routed-MoE stage 2 (down), v1: expert-major, per-pick outputs.

v0 (an earlier version) took units (16-row block, pick group) row-block-major: each
unit was a 4.5 KB w2 chunk of a different expert (scattered reads) and the
ring held ~2 units per wave: 4.1-4.6 TB/s, and removing the h gathers gained
only 2-9%. v1 streams each expert's w2 contiguously instead:

- Units (pick group g, chunk c of 16 row blocks = 256 output rows), group-
  major, contiguous equal ranges per CTA; wave w takes row blocks 4w..4w+3 of
  the chunk (18 KB of w2, contiguous apart from the K-padding half-tiles).
- Step = one row block: 5 K-tiles (the last half padding: lanes 0..31 loaded,
  the rest zeroed) into an 8-step AGPR ring; A scales (one dword per row-block
  pair and 2 K-tiles) on even steps; the odd block's scales are byte 1/3 of
  the same dword, so op_sel is static.
- The group's h (B fragments, 5 tiles × 8 VGPRs + 2 scale dwords + the pick's
  weight) stays in VGPRs; the next group's is prefetched only when the group
  changes (two unit-body variants: exact s_waitcnt counts).
- Output: y[p][row] = w_p · (W2 h_p)[row], bf16, one dwordx2 store per lane
  per step (lane t + 16 gq: rows 4 gq..4 gq+3 of pick t). Two accumulator sets
  alternate, so a step's result is read while the next step's MFMAs run. The
  combine (kernels/comb_ar.py) sums each token's 6 rows.

Tables / tensors:
- units int32 [n, 20]: e, c, gid, 0, picks[16] (p = token*6 + k, or -1)
- h fp8 [P, 576]; hs uint8 [P, 4, 8]; wts fp32 [P]; y bf16 [P, 5120]
"""

from vllm.models.deepseek_v41.amd.asm.kernels import gtable
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source, rsrc

H, E = 5120, 384
W2_E = H * 320
S2_E = H * 24
CH = 16  # row blocks per chunk
NCH = 320 // CH  # 20 chunks

KA = KArgs(
    ("w2", "ptr"),
    ("s2", "ptr"),
    ("h", "ptr"),
    ("hs", "ptr"),
    ("wts", "ptr"),
    ("groups", "ptr"),
    ("y", "ptr"),
    ("counts", "ptr"),
    ("h_bytes", "u32"),
    ("hs_bytes", "u32"),
    ("wts_bytes", "u32"),
    ("y_bytes", "u32"),
)


def source(name: str | None = None):
    """Stage 2 as K2 runs it: units split by cost (moe_prep's table) over
    gtable.GRID CTAs, counts read with a vector load, y in bf16 [P, 5120] (half
    the bytes written and read back by the combine; the fp32 stores cost ~3
    bytes of weight stream each): one dwordx2 per lane per step."""
    split, D, grid, PER_MAX, w_pol = "table", 4, gtable.GRID, 128, "nt"
    SPU = 4  # steps (row blocks) per unit per wave
    UST = 1  # unit stride of a CTA
    # D ≤ SPU keeps a prefetch's effect on the wait counts inside its own unit;
    # D | SPU keeps ring slots static (slot = step % D).
    assert D <= SPU and SPU % D == 0
    UPI = 1
    LPS = 5 + 3  # loads per step: W 5 + A scales 3 (every step: no slot race)
    XL = 10 + 2 + 1  # h prefetch: 10 B + 2 SB + 1 weight
    R = 80
    TBL = (PER_MAX * R + 255) // 256 * 256
    lds = TBL
    name = name or f"moe_down1_{split}_d{D}_g{grid}_{w_pol}_e0_yb"
    pol = f" {w_pol}"

    # VGPRs: v0 OOB, v1 lane, v2 lane*16, v3 lane*16+4096, v4 lane*4, v5 t, v6 gq
    vB, vS, vT, vY, vP = (
        8,
        9,
        10,
        11,
        12,
    )  # current group: h off, hs off, wts off, y off, p
    nB, nS, nT, nY, nP = 13, 14, 15, 16, 17
    SA0 = 18  # SA ring: 3 per slot
    XC = SA0 + 3 * D + 1
    XC += XC % 2
    XN = XC + 44
    TMP = XN + 44
    t = [TMP + i for i in range(16)]
    vgprs = TMP + 16
    assert vgprs <= 256, vgprs
    HB, HSB, HW = 0, 40, 42  # offsets inside an h buffer: B, SB (2), weight
    WR0 = 8  # AGPRs: acc a[0:3], a[4:7]; ring a[8 + 20 * slot]
    agprs = WR0 + 20 * D

    def sa(s):
        return SA0 + 3 * (s % D)

    c = []
    A = c.append
    A("  s_load_dwordx16 s[4:19], s[0:1], 0x0")
    A("  s_load_dwordx4 s[20:23], s[0:1], 0x40")
    A("  v_readfirstlane_b32 s48, v0")
    A("  s_lshr_b32 s48, s48, 6")
    A("  v_and_b32 v1, 63, v0")
    A("  v_lshlrev_b32 v2, 4, v1")
    A("  v_add_u32 v3, 0x1000, v2")
    A("  v_lshlrev_b32 v4, 2, v1")
    A("  v_and_b32 v5, 15, v1")
    A("  v_lshrrev_b32 v6, 4, v1")
    A("  s_waitcnt lgkmcnt(0)")
    # kernargs: w2 s[4:5] s2 s[6:7] h s[8:9] hs s[10:11] wts s[12:13] units s[14:15] y s[16:17]
    #           nunits s18 per s19 h_bytes s20 hs_bytes s21 wts_bytes s22 y_bytes s23
    c += rsrc(24, 4, nbytes=-1)
    c += rsrc(28, 6, nbytes=-1)
    c += rsrc(32, 8, nbytes_sgpr=20)
    c += rsrc(36, 10, nbytes_sgpr=21)
    c += rsrc(40, 12, nbytes_sgpr=22)
    c += rsrc(44, 16, nbytes_sgpr=23)
    c += gtable.range_and_copy(
        20,
        counts=18,
        groups=14,
        tv=[t[0], t[1], t[2]],
        end_label=".Lend",
        grid=grid,
        split=split,
    )
    A("  s_mov_b32 s49, s52")
    A("  s_mov_b32 s73, 0xaaaaaaab")
    A("  v_mov_b32 v0, 0x7fff0000")

    # SGPRs per unit: sW (w2 offset of the wave's first block), sA (scale offset), sY (y row offset), gid
    sW, sA, sY, sG = 56, 57, 58, 59
    nW, nA, nY_, nG = 60, 61, 62, 63
    # s75: previous unit's gid; s64..s67: per-step W soffsets (computed per unit)

    def params(u_sgpr, W, Asc, Ys, Gs, B, S, T, Y, P):
        t0, t2 = t[0], t[2]
        o = gtable.decode(20, u_sgpr, Gs, 78)
        o += [
            f"  v_mov_b32 v{t0}, s76",
            f"  ds_read_b32 v{t2}, v{t0}",
            f"  v_lshlrev_b32 v{t0}, 2, v5",
            f"  v_add_u32 v{t0}, s76, v{t0}",
            f"  ds_read_b32 v{P}, v{t0} offset:16",
            "  s_waitcnt lgkmcnt(0)",
            f"  v_readfirstlane_b32 s77, v{t2}",
            # first row block of this wave: n0 = 16 c + 4 w
            "  s_lshl_b32 s79, s78, 4",
            "  s_lshl_b32 s74, s48, 2",
            "  s_add_u32 s79, s79, s74",
            f"  s_mul_i32 s{W}, s77, {W2_E}",
            f"  s_mul_i32 s74, s79, {H}",
            f"  s_add_u32 s{W}, s{W}, s74",
            f"  s_mul_i32 s{Asc}, s77, {S2_E}",
            "  s_lshr_b32 s74, s79, 1",
            "  s_mul_i32 s74, s74, 768",
            f"  s_add_u32 s{Asc}, s{Asc}, s74",
            f"  s_lshl_b32 s{Ys}, s79, 5",  # y row offset: n0 * 16 rows * 2 B
            f"  v_cmp_le_i32 vcc, 0, v{P}",
            f"  v_mul_u32_u24 v{t0}, 0x240, v{P}",
            f"  v_lshl_add_u32 v{t0}, v6, 4, v{t0}",
            f"  v_cndmask_b32 v{B}, v0, v{t0}, vcc",
            f"  v_lshlrev_b32 v{t0}, 5, v{P}",
            f"  v_lshl_add_u32 v{t0}, v6, 3, v{t0}",
            f"  v_cndmask_b32 v{S}, v0, v{t0}, vcc",
            f"  v_lshlrev_b32 v{t0}, 2, v{P}",
            f"  v_cndmask_b32 v{T}, v0, v{t0}, vcc",
            f"  v_mul_u32_u24 v{t0}, {H * 2}, v{P}",
            f"  v_lshl_add_u32 v{t0}, v6, 3, v{t0}",
            f"  v_cndmask_b32 v{Y}, v0, v{t0}, vcc",
        ]
        return o

    def w_loads(s, rel, W, Asc):
        """Loads for global step s = row block `rel` (0..3) of a unit with offsets W/Asc."""
        slot = WR0 + 20 * (s % D)
        o = [f"  s_add_u32 s64, s{W}, {rel * H}"]
        o += [
            f"  buffer_load_dwordx4 a[{slot + 4 * k}:{slot + 4 * k + 3}], v2, s[24:27], s64 offen offset:{k * 1024}{pol}"
            for k in range(4)
        ]
        o += ["  s_mov_b32 exec_lo, 0", "  s_mov_b32 exec_hi, -1"]
        o += [f"  v_accvgpr_write_b32 a{slot + 16 + q}, 0" for q in range(4)]
        o += [
            "  s_mov_b32 exec_lo, -1",
            "  s_mov_b32 exec_hi, 0",
            f"  buffer_load_dwordx4 a[{slot + 16}:{slot + 19}], v3, s[24:27], s64 offen{pol}",
            "  s_mov_b64 exec, -1",
        ]
        o += [
            f"  buffer_load_dword v{sa(s) + k1}, v4, s[28:31], s{Asc} offen offset:{(rel // 2) * 768 + k1 * 256}"
            for k1 in range(3)
        ]
        return o

    def h_loads(buf, B, S, T):
        o = []
        for k0 in range(5):
            r = buf + HB + 8 * k0
            o += [
                f"  buffer_load_dwordx4 v[{r}:{r + 3}], v{B}, s[32:35], 0 offen offset:{k0 * 128}",
                f"  buffer_load_dwordx4 v[{r + 4}:{r + 7}], v{B}, s[32:35], 0 offen offset:{k0 * 128 + 64}",
            ]
        o += [
            f"  buffer_load_dword v{buf + HSB + q}, v{S}, s[36:39], 0 offen offset:{4 * q}"
            for q in range(2)
        ]
        o.append(f"  buffer_load_dword v{buf + HW}, v{T}, s[40:43], 0 offen")
        return o

    def mfmas(s):
        slot = WR0 + 20 * (s % D)
        acc = 4 * (s % 2)
        odd = s % 2  # odd row block of the pair: scale byte +1
        o = []
        for k0 in range(5):
            sel_a = (k0 % 2) * 2 + odd
            sel_b = k0 % 4
            cin = "0" if k0 == 0 else f"a[{acc}:{acc + 3}]"
            o.append(
                f"  v_mfma_scale_f32_16x16x128_f8f6f4 a[{acc}:{acc + 3}], a[{slot + 4 * k0}:{slot + 4 * k0 + 3}], "
                f"v[{XC + HB + 8 * k0}:{XC + HB + 8 * k0 + 7}], {cin}, v{sa(s) + k0 // 2}, v{XC + HSB + k0 // 4} "
                f"op_sel:[{sel_a & 1},{sel_b & 1},0] op_sel_hi:[{sel_a >> 1},{sel_b >> 1},0] cbsz:4 blgp:0"
            )
        return o

    def store_prev(s, rel_prev, Y, Ys, WT):
        """Y for step s-1 (accumulator set (s-1)%2): ×w, bf16, one dwordx2 store."""
        acc = 4 * ((s - 1) % 2)
        o = ["  s_nop 7", "  s_nop 7"]  # MFMA → accumulator read
        o += [f"  v_accvgpr_read_b32 v{t[8 + q]}, a{acc + q}" for q in range(4)]
        o += [f"  v_mul_f32 v{t[8 + q]}, v{t[8 + q]}, v{WT}" for q in range(4)]
        o += [
            f"  v_cvt_pk_bf16_f32 v{t[8]}, v{t[8]}, v{t[9]}",
            f"  v_cvt_pk_bf16_f32 v{t[9]}, v{t[10]}, v{t[11]}",
            f"  buffer_store_dwordx2 v[{t[8]}:{t[9]}], v{Y}, s[44:47], s{Ys} offen offset:{rel_prev * 32}",
        ]
        return o

    # ---- prologue
    c += params(52, sW, sA, sY, sG, vB, vS, vT, vY, vP)
    c += h_loads(XC, vB, vS, vT)
    A("  s_waitcnt vmcnt(0)")
    A(f"  s_mov_b32 s75, s{sG}")
    A(f"  s_add_u32 s53, s52, {UST}")
    A("  s_sub_u32 s54, s50, 1")
    A("  s_min_u32 s53, s53, s54")
    c += params(53, nW, nA, nY_, nG, nB, nS, nT, nY, nP)
    # the wait counts assume exactly one store per iteration after its loads:
    # out-of-range dummy stores (dropped) stand in where there is no result yet
    dummy = "  buffer_store_dword v0, v0, s[44:47], 0 offen"
    for s in range(D - 1):
        u, rel = divmod(s, SPU)
        assert u <= 1
        c += w_loads(s, rel, sW if u == 0 else nW, sA if u == 0 else nA)
        A(dummy)
    # the first step has no previous result: a dummy accumulator store would be
    # out of range; instead the stream starts with step 0 storing nothing
    A("  s_mov_b32 s70, 0")  # first step flag

    # unit loop: UPI units per iteration so ring slots are static. Within a unit
    # body the steps are 4u'..4u'+3 (u' = position in the iteration).
    def unit_body(up: int, pref: bool, tag: str):
        o = []
        for rel in range(SPU):
            s = up * SPU + rel
            nxt = s + D - 1  # global step to issue loads for (relative)
            nu, nrel = divmod(nxt, SPU)
            dist = nu - up  # 0: this unit, 1: next unit, ≥2: beyond (not tracked)
            assert dist <= 1, (
                "ring deeper than one unit ahead: needs a second 'next' set"
            )
            o += w_loads(nxt, nrel, sW if dist == 0 else nW, sA if dist == 0 else nA)
            # newer than step s's loads (issued D-1 iterations ago): the loads of
            # steps s+1..s+D-1, the D-1 stores issued since (one per iteration),
            # and this unit's h prefetch (issued at rel 0) when rel ≥ 1
            # (only steps whose loads were issued before it, i.e. at or before rel 0)
            xl = XL if (pref and 1 <= rel <= D - 1) else 0
            # stores younger than step s's loads: one per iteration
            nst = D - 1
            wait = (D - 1) * LPS + nst + xl
            assert wait <= 63, wait
            o.append(f"  s_waitcnt vmcnt({wait})")
            if rel == 0:
                # previous step's result (its unit's params are still current here)
                o += ["  s_cmp_eq_u32 s70, 0", f"  s_cbranch_scc1 .Lnostore{tag}"]
                o += store_prev(s, SPU - 1, t[12], 71, XC + HW)
                o += (
                    [f"  s_branch .Lstored{tag}", f".Lnostore{tag}:"]
                    + [dummy]
                    + [f".Lstored{tag}:"]
                )
                o += [f"  s_cmp_eq_u32 s{sG}, s75", f"  s_cbranch_scc1 .Lnocopy{tag}"]
                o += [f"  v_mov_b32 v{XC + i}, v{XN + i}" for i in range(44)]
                o += [f".Lnocopy{tag}:", "  s_nop 1"]
                if pref:
                    o += h_loads(XN, nB, nS, nT)
            else:
                o += store_prev(s, rel - 1, vY, sY, XC + HW)
            o += mfmas(s)
        return o

    A(".Lunit:")
    for up in range(UPI):
        A(f"  s_cmp_eq_u32 s{nG}, s{sG}")
        A(f"  s_cbranch_scc0 .Lpref{up}")
        c += unit_body(up, False, f"n{up}")
        A(f"  s_branch .Ladv{up}")
        A(f".Lpref{up}:")
        c += unit_body(up, True, f"p{up}")
        A(f".Ladv{up}:")
        # advance: remember this unit's y offset for the delayed store of its last step
        A(f"  s_mov_b32 s71, s{sY}")
        A("  s_mov_b32 s70, 1")
        A(f"  s_mov_b32 s75, s{sG}")
        for a_, b_ in ((sW, nW), (sA, nA), (sY, nY_), (sG, nG)):
            A(f"  s_mov_b32 s{a_}, s{b_}")
        A(f"  v_mov_b32 v{t[12]}, v{vY}")  # previous y offset for the delayed store
        for a_, b_ in ((vB, nB), (vS, nS), (vT, nT), (vY, nY), (vP, nP)):
            A(f"  v_mov_b32 v{a_}, v{b_}")
        A(f"  s_add_u32 s49, s49, {UST}")
        A("  s_cmp_lt_u32 s49, s50")
        A("  s_cbranch_scc0 .Ldone")
        A(f"  s_add_u32 s53, s49, {UST}")
        A("  s_sub_u32 s54, s50, 1")
        A("  s_min_u32 s53, s53, s54")
        c += params(53, nW, nA, nY_, nG, nB, nS, nT, nY, nP)
    A("  s_branch .Lunit")
    A(".Ldone:")
    # the last step's result
    A("  s_nop 7\n  s_nop 7")
    c += store_prev(SPU * UPI, SPU - 1, t[12], 71, XC + HW)
    A("  s_waitcnt vmcnt(0)")
    A(".Lend:")
    A("  s_endpgm")
    src = kernel_source(
        name, c, KA, lds=lds, vgprs=vgprs, agprs=agprs, sgprs=96, block=256
    )
    return dict(
        name=name,
        src=src,
        ka=KA,
        grid=grid,
        PER_MAX=PER_MAX,
        lds=lds,
        vgprs=vgprs,
        block=256,
    )
