# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Routed-MoE stage 1 (gate/up), v1. Same math and outputs as v0 (an earlier version);
the change is where the activations come from.

v0 gathered each pick's x from L2 for every K-step of every 32-row unit; with
the weight stream flowing through L2 those gathers miss and, since loads
return in order, stall the ring (M = 128: 2.9 TB/s, vs 6.5 TB/s with the
gathers removed). v1:
- G = 1 (16 picks per unit; hotter experts get another unit pass, +0.1-4.6%
  weight bytes on captured routing).
- Each wave keeps its K-quarter of the current group's x as B fragments in
  VGPRs (10 tiles × 8 + 3 scale dwords) and prefetches the next group's into a
  second buffer only when the group changes (two unit-body variants, so every
  s_waitcnt stays exact); the buffer is copied over at the next unit's start.
- The weight ring lives in AGPRs (MFMA A operands can be AGPRs); A scales load
  on even steps only (one dword covers two K-tiles).

Tables / tensors (as v0 except):
- units int32 [n, 20]: e, j, gid, 0, picks[16]; gid identifies the pick group
- xs uint8 [M, 4 (kb), 4 (wave), 12]: E8M0 of (token, K block kb of tile
  k0 = 10 w + kk) at `[t][kb][w][kk]` (kk < 10), so a wave reads 3 dwords and the
  byte select kk % 4 is static
"""

from vllm.models.deepseek_v41.amd.asm.kernels import gtable
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source, rsrc

H, I_PAD, E = 5120, 640, 384
W13_E = 2 * I_PAD * H // 2
S13_E = 2 * I_PAD * H // 32
KT = H // 128
KW = KT // 4
JB = 18
XS_ROW = 4 * 4 * 12  # bytes per token of xs

KA = KArgs(
    ("w13", "ptr"),
    ("s13", "ptr"),
    ("xq", "ptr"),
    ("xs", "ptr"),
    ("groups", "ptr"),
    ("h", "ptr"),
    ("hs", "ptr"),
    ("counts", "ptr"),
    ("xq_bytes", "u32"),
    ("xs_bytes", "u32"),
    ("h_bytes", "u32"),
    ("hs_bytes", "u32"),
)
# rstd (the FFN front without fin): x was quantised without the token's rstd; g and u are scaled by
# rstd[token] f32 before the activation (prefetched with the unit's picks, one unit ahead)
KA_R = KArgs(*KA.args, ("rstd", "ptr"))


def source(name: str | None = None):
    """Stage 1 as K2 runs it: units split by count ("ceil") over gtable.GRID CTAs,
    counts read with a vector load, g / u scaled by rstd."""
    split, D, grid, PER_MAX, w_pol = "ceil", 5, gtable.GRID, 64, "nt"
    G = 1
    assert (
        KW % D == 0 and D % 2 == 1
    )  # odd D: an even step's slot never holds an odd step's SA
    nl = lambda s: 6 if s % 2 == 0 else 4  # loads issued for step s
    XL = 2 * KW + 3  # x prefetch loads
    R = (4 + 16 * G) * 4
    TBL = (PER_MAX * R + 255) // 256 * 256
    RED = 4 * G * 64 * 64
    lds = TBL + 2 * RED
    name = name or f"moe_ug1_{split}_d{D}_g{grid}_{w_pol}_r"
    pol = f" {w_pol}"

    # VGPRs
    vB, vS, vP = 16, 17, 18  # current group: x offset, xs offset, pick
    nB, nS, nP = 19, 20, 21  # next group
    vR, nR = 115, 199  # rstd of the lane's pick (current / next unit), free regs
    SA0 = 22  # SA ring: 2 per slot
    XC, XN = 32, 116  # x buffers: 80 B regs + 3 SB dwords each
    XSB = 80
    T0 = 200
    vsum, vtmp = T0, T0 + 16
    vc10, vcm10, vperm16, vperm32 = T0 + 32, T0 + 33, T0 + 34, T0 + 35
    t = [T0 + 36 + i for i in range(12)]
    vgprs = T0 + 48
    assert SA0 + 2 * D <= XC and XC + 84 <= XN and XN + 84 <= T0
    # AGPRs: acc a[0:15], W ring a[16 + 16 * slot ...]
    WR0 = 16
    agprs = WR0 + 16 * D

    def sa(s):  # SA regs used by step s (loaded with the even step)
        return SA0 + 2 * ((s - s % 2) % D)

    c = []
    A = c.append
    A("  s_load_dwordx16 s[4:19], s[0:1], 0x0")
    A("  s_load_dwordx4 s[20:23], s[0:1], 0x40")
    A(f"  s_load_dwordx2 s[84:85], s[0:1], {KA_R.off['rstd']:#x}")
    A("  v_readfirstlane_b32 s48, v0")
    A("  s_lshr_b32 s48, s48, 6")
    A("  v_and_b32 v1, 63, v0")
    A("  v_lshlrev_b32 v2, 4, v1")
    A("  v_lshlrev_b32 v3, 2, v1")
    A("  v_and_b32 v4, 15, v1")
    A("  v_lshrrev_b32 v5, 4, v1")
    for kk in range(KW):
        A(f"  v_add_u32 v{6 + kk}, {kk * 1024}, v2")
    A(f"  v_mov_b32 v{vc10}, 0x41200000")
    A(f"  v_mov_b32 v{vcm10}, 0xc1200000")
    for r, d in ((vperm16, 16), (vperm32, 32)):
        A(f"  v_add_u32 v{r}, {d}, v1")
        A(f"  v_and_b32 v{r}, 63, v{r}")
        A(f"  v_lshlrev_b32 v{r}, 2, v{r}")
    A("  s_waitcnt lgkmcnt(0)")
    c += rsrc(24, 4, nbytes=-1)
    c += rsrc(28, 6, nbytes=-1)
    c += rsrc(32, 8, nbytes_sgpr=20)
    c += rsrc(36, 10, nbytes_sgpr=21)
    c += rsrc(40, 14, nbytes_sgpr=22)
    c += rsrc(44, 16, nbytes_sgpr=23)
    c += rsrc(84, 84, nbytes=256 * 4)
    c += gtable.range_and_copy(
        18,
        counts=18,
        groups=12,
        tv=[t[0], t[1]],
        end_label=".Lend",
        grid=grid,
        split=split,
    )
    A(f"  s_mul_i32 s71, s48, {KW * 128}")
    A("  s_mul_i32 s72, s48, 12")
    A("  s_mov_b32 s49, s52")
    A("  s_mov_b32 s51, 0")
    A("  s_mov_b32 s73, 0xaaaaaaab")
    A("  v_mov_b32 v0, 0x7fff0000")

    # SGPRs per unit: W streams [4], SA [2], j, gid
    sW, sA, sJ, sG = [56, 57, 58, 59], [60, 61], 62, 63
    nW, nA, nJ, nG = [64, 65, 66, 67], [68, 69], 70, 74
    # s75: gid of the previous unit (copy needed when it differs from sG)

    def params(u_sgpr, W, Asc, J, Gs, B, S, P, R):
        t0, t1, t2, t3 = t[0], t[1], t[2], t[3]
        o = gtable.decode(18, u_sgpr, Gs, J)
        o += [
            f"  v_mov_b32 v{t0}, s76",
            f"  ds_read_b32 v{t2}, v{t0}",
            f"  v_lshlrev_b32 v{t0}, 2, v4",
            f"  v_add_u32 v{t0}, s76, v{t0}",
            f"  ds_read_b32 v{P}, v{t0} offset:16",
            "  s_waitcnt lgkmcnt(0)",
            f"  v_readfirstlane_b32 s77, v{t2}",
            f"  s_mul_i32 s78, s77, {W13_E}",
            f"  s_mul_i32 s79, s{J}, {4 * KT * 1024}",
            "  s_add_u32 s78, s78, s79",
            f"  s_mul_i32 s79, s48, {KW * 1024}",
            "  s_add_u32 s78, s78, s79",
        ]
        o += [f"  s_add_u32 s{W[k]}, s78, {k * KT * 1024}" for k in range(4)]
        o += [
            f"  s_mul_i32 s78, s77, {S13_E}",
            f"  s_mul_i32 s79, s{J}, {2 * 20 * 256}",
            "  s_add_u32 s78, s78, s79",
            f"  s_mul_i32 s79, s48, {KW // 2 * 256}",
            "  s_add_u32 s78, s78, s79",
            f"  s_mov_b32 s{Asc[0]}, s78",
            f"  s_add_u32 s{Asc[1]}, s78, {20 * 256}",
            f"  v_mul_hi_u32 v{t0}, v{P}, s73",
            f"  v_lshrrev_b32 v{t0}, 2, v{t0}",
            f"  v_cmp_le_i32 vcc, 0, v{P}",
            f"  v_mul_u32_u24 v{t1}, {H}, v{t0}",
            f"  v_lshl_add_u32 v{t1}, v5, 4, v{t1}",
            f"  v_cndmask_b32 v{B}, v0, v{t1}, vcc",
            f"  v_mul_u32_u24 v{t1}, {XS_ROW}, v{t0}",
            f"  v_mul_u32_u24 v{t2}, 48, v5",
            f"  v_add_u32 v{t1}, v{t1}, v{t2}",
            f"  v_cndmask_b32 v{S}, v0, v{t1}, vcc",
        ]
        o += [  # rstd[token]; no pick → out of range → 0
            f"  v_lshlrev_b32 v{t3}, 2, v{t0}",
            f"  v_cndmask_b32 v{t3}, v0, v{t3}, vcc",
            f"  buffer_load_dword v{R}, v{t3}, s[84:87], 0 offen",
        ]
        return o

    def w_loads(s, kk, W, Asc):
        slot = WR0 + 16 * (s % D)
        o = [
            f"  buffer_load_dwordx4 a[{slot + 4 * ti}:{slot + 4 * ti + 3}], v{6 + kk}, s[24:27], s{W[ti]} offen{pol}"
            for ti in range(4)
        ]
        if s % 2 == 0:
            o += [
                f"  buffer_load_dword v{sa(s) + b}, v3, s[28:31], s{Asc[b]} offen offset:{(kk // 2) * 256}"
                for b in range(2)
            ]
        return o

    def x_loads(buf, B, S):
        o = []
        for kk in range(KW):
            r = buf + 8 * kk
            o += [
                f"  buffer_load_dwordx4 v[{r}:{r + 3}], v{B}, s[32:35], s71 offen offset:{kk * 128}",
                f"  buffer_load_dwordx4 v[{r + 4}:{r + 7}], v{B}, s[32:35], s71 offen offset:{kk * 128 + 64}",
            ]
        o += [
            f"  buffer_load_dword v{buf + XSB + q}, v{S}, s[36:39], s72 offen offset:{4 * q}"
            for q in range(3)
        ]
        return o

    def mfmas(s, kk):
        slot = WR0 + 16 * (s % D)
        o = []
        for ti in range(4):
            bb, gu = ti // 2, ti % 2
            sel_a = (kk % 2) * 2 + gu
            sel_b = kk % 4
            a = 4 * ti
            cin = "0" if kk == 0 else f"a[{a}:{a + 3}]"
            o.append(
                f"  v_mfma_scale_f32_16x16x128_f8f6f4 a[{a}:{a + 3}], a[{slot + 4 * ti}:{slot + 4 * ti + 3}], "
                f"v[{XC + 8 * kk}:{XC + 8 * kk + 7}], {cin}, v{sa(s) + bb}, v{XC + XSB + kk // 4} "
                f"op_sel:[{sel_a & 1},{sel_b & 1},0] op_sel_hi:[{sel_a >> 1},{sel_b >> 1},0] cbsz:4 blgp:0"
            )
        return o

    # ---- prologue: current = first unit, x for it loaded and waited, next = second
    c += params(52, sW, sA, sJ, sG, vB, vS, vP, vR)
    c += x_loads(XC, vB, vS)
    A("  s_waitcnt vmcnt(0)")
    A(f"  s_mov_b32 s75, s{sG}")
    A("  s_add_u32 s53, s52, 1")
    A("  s_sub_u32 s54, s50, 1")
    A("  s_min_u32 s53, s53, s54")
    c += params(53, nW, nA, nJ, nG, nB, nS, nP, nR)
    for s in range(D - 1):
        c += w_loads(s, s, sW, sA)

    def unit_body(pref: bool, tag: str):
        o = []
        for kk in range(KW):
            s = kk
            nk = kk + D - 1
            if nk < KW:
                o += w_loads(nk, nk, sW, sA)
            else:
                o += w_loads(nk, nk - KW, nW, nA)
            ahead = sum(nl(s + i) for i in range(1, D))
            xl = (
                XL if (pref and 1 <= kk <= D - 1) else 0
            )  # x loads issued after step kk's loads
            assert ahead + xl <= 63, (ahead, xl)
            o.append(f"  s_waitcnt vmcnt({ahead + xl})")
            if kk == 0:
                # the group changed at this unit: its x (prefetched last unit) → current buffer
                o += [f"  s_cmp_eq_u32 s{sG}, s75", f"  s_cbranch_scc1 .Lnocopy{tag}"]
                o += [f"  v_mov_b32 v{XC + i}, v{XN + i}" for i in range(XSB + 3)]
                o += [f".Lnocopy{tag}:", "  s_nop 1"]
                if pref:
                    o += x_loads(XN, nB, nS)
            o += mfmas(s, kk)
        return o

    A(".Lunit:")
    A(f"  s_cmp_eq_u32 s{nG}, s{sG}")
    A("  s_cbranch_scc0 .Lpref")
    c += unit_body(False, "n")
    A("  s_branch .Lepi")
    A(".Lpref:")
    c += unit_body(True, "p")
    A(".Lepi:")
    # ---- epilogue (as v0, G = 1: wave 0 finishes)
    A("  s_nop 7\n  s_nop 7\n  s_nop 7")
    A(f"  v_lshlrev_b32 v{t[11]}, 6, v1")
    A(f"  s_mul_i32 s76, s51, {RED}")
    A(f"  s_add_u32 s76, s76, {TBL}")
    A("  s_lshl_b32 s77, s48, 12")
    A("  s_add_u32 s77, s77, s76")
    A(f"  v_add_u32 v{t[10]}, s77, v{t[11]}")
    for i in range(16):
        A(f"  v_accvgpr_read_b32 v{vsum + i}, a{i}")
    for q in range(4):
        A(
            f"  ds_write_b128 v{t[10]}, v[{vsum + 4 * q}:{vsum + 4 * q + 3}] offset:{16 * q}"
        )
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    A("  s_cmp_eq_u32 s48, 0")
    A("  s_cbranch_scc0 .Lnot0")
    for w in range(4):
        A(f"  v_add_u32 v{t[10]}, s76, v{t[11]}")
        dst = vsum if w == 0 else vtmp
        for q in range(4):
            A(
                f"  ds_read_b128 v[{dst + 4 * q}:{dst + 4 * q + 3}], v{t[10]} offset:{w * 4096 + 16 * q}"
            )
        if w:
            A("  s_waitcnt lgkmcnt(0)")
            for i in range(16):
                A(f"  v_add_f32 v{vsum + i}, v{vsum + i}, v{vtmp + i}")
    A("  s_waitcnt lgkmcnt(0)")
    for i in range(16):
        A(f"  v_mul_f32 v{vsum + i}, v{vR}, v{vsum + i}")
    gs = [vsum + 8 * bb + i for bb in range(2) for i in range(4)]
    us = [g + 4 for g in gs]
    ex = [vtmp + i for i in range(8)]
    for g, u, x in zip(gs, us, ex):
        A(f"  v_min_f32 v{g}, v{vc10}, v{g}")
        A(f"  v_med3_f32 v{u}, v{u}, v{vcm10}, v{vc10}")
        A(f"  v_mul_f32 v{x}, 0xbfb8aa3b, v{g}")
    for x in ex:
        A(f"  v_exp_f32 v{x}, v{x}")
    for x in ex:
        A(f"  v_add_f32 v{x}, 1.0, v{x}")
    for x in ex:
        A(f"  v_rcp_f32 v{x}, v{x}")
    for g, u, x in zip(gs, us, ex):
        A(f"  v_mul_f32 v{g}, v{g}, v{x}")
        A(f"  v_mul_f32 v{g}, v{g}, v{u}")
    hs = gs
    A(f"  v_max_f32 v{t[1]}, |v{hs[0]}|, |v{hs[1]}|")
    for x in hs[2:]:
        A(f"  v_max_f32 v{t[1]}, v{t[1]}, |v{x}|")
    for r in (vperm16, vperm32):
        A(f"  ds_bpermute_b32 v{t[2]}, v{r}, v{t[1]}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_max_f32 v{t[1]}, v{t[1]}, v{t[2]}")
    A(f"  v_frexp_exp_i32_f32 v{t[3]}, v{t[1]}")
    A(f"  v_frexp_mant_f32 v{t[4]}, v{t[1]}")
    A(f"  v_cmp_lt_f32 vcc, 0x3f600000, v{t[4]}")
    A(f"  v_addc_co_u32 v{t[3]}, vcc, -9, v{t[3]}, vcc")
    A(f"  v_max_i32 v{t[3]}, 0xffffff81, v{t[3]}")
    A(f"  v_min_i32 v{t[3]}, 0x7f, v{t[3]}")
    A(f"  v_sub_u32 v{t[5]}, 0, v{t[3]}")
    for x in hs:
        A(f"  v_ldexp_f32 v{x}, v{x}, v{t[5]}")
    for bb in range(2):
        q = hs[4 * bb : 4 * bb + 4]
        A(f"  v_cvt_pk_fp8_f32 v{t[6 + bb]}, v{q[0]}, v{q[1]}")
        A(f"  v_cvt_pk_fp8_f32 v{t[6 + bb]}, v{q[2]}, v{q[3]} op_sel:[0,0,1]")
    A(f"  v_cmp_le_i32 vcc, 0, v{vP}")
    A(f"  v_mul_u32_u24 v{t[8]}, 576, v{vP}")
    A(f"  v_lshl_add_u32 v{t[8]}, v5, 2, v{t[8]}")
    A(f"  v_mov_b32 v{t[9]}, s{sJ}")
    A(f"  v_lshl_add_u32 v{t[8]}, v{t[9]}, 5, v{t[8]}")
    A(f"  v_cndmask_b32 v{t[8]}, v0, v{t[8]}, vcc")
    A(f"  buffer_store_dword v{t[6]}, v{t[8]}, s[40:43], 0 offen")
    A(f"  buffer_store_dword v{t[7]}, v{t[8]}, s[40:43], 0 offen offset:16")
    A(f"  v_lshlrev_b32 v{t[10]}, 5, v{vP}")
    A(f"  s_and_b32 s82, s{sJ}, 3")
    A("  s_lshl_b32 s82, s82, 3")
    A(f"  s_lshr_b32 s83, s{sJ}, 2")
    A("  s_add_u32 s82, s82, s83")
    A(f"  v_add_u32 v{t[10]}, s82, v{t[10]}")
    A("  v_cmp_eq_u32 s[80:81], 0, v5")
    A("  s_and_b64 vcc, vcc, s[80:81]")
    A(f"  v_cndmask_b32 v{t[10]}, v0, v{t[10]}, vcc")
    A(f"  v_add_u32 v{t[11]}, 0x7f, v{t[3]}")
    A(f"  buffer_store_byte v{t[11]}, v{t[10]}, s[44:47], 0 offen")
    A(".Lnot0:")
    # ---- advance
    A(f"  s_mov_b32 s75, s{sG}")
    for a_, b_ in zip(sW + sA + [sJ, sG], nW + nA + [nJ, nG]):
        A(f"  s_mov_b32 s{a_}, s{b_}")
    for a_, b_ in ((vB, nB), (vS, nS), (vP, nP), (vR, nR)):
        A(f"  v_mov_b32 v{a_}, v{b_}")
    A("  s_add_u32 s49, s49, 1")
    A("  s_xor_b32 s51, s51, 1")
    A("  s_cmp_lt_u32 s49, s50")
    A("  s_cbranch_scc0 .Ldone")
    A("  s_add_u32 s53, s49, 1")
    A("  s_sub_u32 s54, s50, 1")
    A("  s_min_u32 s53, s53, s54")
    c += params(53, nW, nA, nJ, nG, nB, nS, nP, nR)
    A("  s_branch .Lunit")
    A(".Ldone:")
    A("  s_waitcnt vmcnt(0)")
    A(".Lend:")
    A("  s_endpgm")
    ka = KA_R
    src = kernel_source(
        name, c, ka, lds=lds, vgprs=vgprs, agprs=agprs, sgprs=96, block=256
    )
    return dict(
        name=name,
        src=src,
        ka=ka,
        grid=grid,
        G=G,
        R=R,
        PER_MAX=PER_MAX,
        lds=lds,
        vgprs=vgprs,
    )
