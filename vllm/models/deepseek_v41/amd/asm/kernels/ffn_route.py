# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""FFN router and shared gate_up (one launch, kernels/multi.py) and their
finish (P3). M ≤ 256.

- `source_router`: logits partials rpart [M, 32, 384] f32 over 32 K slices
  of 160. CTA b: slice s = b % 32, expert group g = b / 32 (48 experts = 3
  tiles of 16). Every wave holds the group's slice of gate as A operands of
  `v_mfma_f32_16x16x32_bf16` (lane l: row l % 16, K 8 (l / 16) .. + 7) and
  computes token tiles w, w + 4, w + 8, w + 12 with B = y (same lane layout).
  D lane l: token l % 16, experts 4 (l / 16) .. + 3, one dwordx4 store.
- `source_shared`: shared gate_up partials gpart [M, 10, 1152] f32 over 10 K
  slices of 512. CTA b: slice s = b % 10, row group = b / 10 (48 rows = 3
  tiles). The weight slice sits in AGPRs as fp8 A operands of the scaled
  16x16x128 F8F6F4 MFMA (lane group q: K 16q .. + 15 in regs 0-3, 64 + 16q ..
  in regs 4-7; scale of 32-block kb from lane row + 16 kb), B = xq / xs as
  moe_ug1 reads them.
- `source_p3`: one CTA per token.
  - Router: wave w sums slices 8w .. 8w + 7 (lane l: experts l + 64 j), wave 0
    sums the 4 in order → logits; score = sqrt(softplus) + bias, 6 argmax
    rounds (ties to the lower index) → ids; weights = (biased − bias)
    renormalised × 1.5.
  - Shared (waves 2-3, thread i < 72: columns 8i .. 8i + 7): sum the 10 slices
    in order, bf16 (the linear's output), silu(min(g, 10)) · clamp(u, ±10),
    bf16, MXFP8 (ceil rule, quads share a block) → hq [M, 576], hs [M, 18].
Tokens ≥ M load 0 and their stores fall outside the buffers.

`source_p3` (the FFN front without a fin launch): the seam wrote
y' = bf16(x1 · ffn_norm.w) without the token's rstd. Every wave sums the 160
Σx1² partials (seam partial value 25) in a fixed order → rstd = rsq(Σ / H +
norm_eps); the router logits and the shared gate_up sums are scaled by it,
wave 0 stores rstd[t] (stage 1 scales its g / u by it). A 5th wave runs fin's
mixes (ffn_seam.mix_wave, KA_MIX's arguments appended with prefix mx_).
"""

from vllm.models.deepseek_v41.amd.asm.kernels import ffn_seam
from vllm.models.deepseek_v41.amd.asm.kernels.ffn_seam import H, f32
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source, rsrc

E, TOPK, I_SH = 384, 6, 576
NS = 2 * I_SH  # shared gate|up rows
RS = 32  # router K slices (160 columns, 5 K steps)
SS = 10  # shared K slices (512 columns, 4 K tiles)
R_GRID = RS * (E // 48)  # 256
S_GRID = SS * (NS // 48)  # 240
RPART_ROW = RS * E * 4  # bytes of rpart per token
GPART_ROW = SS * NS * 4  # bytes of gpart per token
MAX_M = 256

KA_R = KArgs(("gate", "ptr"), ("y", "ptr"), ("part", "ptr"), ("m", "u32"))
KA_S = KArgs(
    ("w", "ptr"),
    ("ws", "ptr"),
    ("xq", "ptr"),
    ("xs", "ptr"),
    ("part", "ptr"),
    ("m", "u32"),
)
KA_3 = KArgs(
    ("rpart", "ptr"),
    ("bias", "ptr"),
    ("logits", "ptr"),
    ("ids", "ptr"),
    ("wts", "ptr"),
    ("gpart", "ptr"),
    ("hq", "ptr"),
    ("hs", "ptr"),
    ("m", "u32"),
)


def source_router(name: str = "ffn_router"):
    """The gate slice in AGPRs a[0:60] (MFMA A operands), 144 VGPRs, so a shared
    gate_up CTA fits beside it on a CU (the two run as one launch)."""
    # gate slice (3 tiles × 5 K steps × 4, AGPRs), y (4 tiles × 5 × 4), D (4 tiles × 3 × 4)
    W, Y, D = 0, 16, 96
    wr = "a"
    c = []
    A = c.append
    A("  s_load_dwordx8 s[4:11], s[0:1], 0x0")  # gate y part, m s10
    A("  v_readfirstlane_b32 s12, v0")
    A("  s_lshr_b32 s12, s12, 6")  # wave
    A("  v_and_b32 v1, 63, v0")
    A("  v_and_b32 v2, 15, v1")  # r
    A("  v_lshrrev_b32 v3, 4, v1")  # lane group q
    A(f"  s_and_b32 s28, s2, {RS - 1}")  # K slice
    A("  s_lshr_b32 s29, s2, 5")  # expert group
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_mul_i32 s13, s10, {2 * H}")
    A(f"  s_mul_i32 s14, s10, {RPART_ROW}")
    c += rsrc(16, 4, nbytes=E * 2 * H)
    c += rsrc(20, 6, nbytes_sgpr=13)
    c += rsrc(24, 8, nbytes_sgpr=14)
    A("  s_add_u32 s33, s10, 15")
    A("  s_lshr_b32 s33, s33, 4")  # token tiles
    A(f"  s_mul_i32 s30, s29, {48 * 2 * H}")
    A("  s_mul_i32 s34, s28, 320")  # slice byte offset
    A("  s_add_u32 s30, s30, s34")
    for i in (1, 2):
        A(f"  s_add_u32 s{30 + i}, s{29 + i}, {16 * 2 * H}")  # gate rows of tile i
    A(f"  v_mul_u32_u24 v4, {2 * H}, v2")
    A("  v_lshl_add_u32 v4, v3, 4, v4")  # r·2H + 16 q
    for i in range(3):
        for k in range(5):
            r = W + 4 * (5 * i + k)
            A(
                f"  buffer_load_dwordx4 {wr}[{r}:{r + 3}], v4, s[16:19], s{30 + i} offen offset:{64 * k}"
            )
    A("  v_lshl_add_u32 v15, v3, 4, s34")  # y: 16 q + 320 s
    A(f"  s_mul_i32 s35, s28, {E * 4}")
    A(f"  s_mul_i32 s36, s29, {48 * 4}")
    A("  s_add_u32 s35, s35, s36")
    A("  v_lshl_add_u32 v14, v3, 4, s35")  # rpart: 16 q + 1536 s + 192 g
    for j in range(4):
        A(f"  s_add_u32 s36, s12, {4 * j}")
        A("  s_lshl_b32 s36, s36, 4")
        A("  v_add_u32 v13, s36, v2")  # t
        A(f"  v_mul_u32_u24 v{5 + j}, {2 * H}, v13")
        A(f"  v_add_u32 v{5 + j}, v{5 + j}, v15")
        A(f"  v_mul_u32_u24 v{9 + j}, {RPART_ROW}, v13")
        A(f"  v_add_u32 v{9 + j}, v{9 + j}, v14")
    for j in range(4):
        for k in range(5):
            r = Y + 20 * j + 4 * k
            A(
                f"  buffer_load_dwordx4 v[{r}:{r + 3}], v{5 + j}, s[20:23], 0 offen offset:{64 * k}"
            )
    A("  s_waitcnt vmcnt(0)")
    # token tiles grow with j: the first one past the end ends the work
    for j in range(4):
        A(f"  s_add_u32 s36, s12, {4 * j}")
        A("  s_cmp_lt_u32 s36, s33")
        A("  s_cbranch_scc0 .Lmfma_end")
        for k in range(5):
            for i in range(3):
                d, w, y = D + 12 * j + 4 * i, W + 4 * (5 * i + k), Y + 20 * j + 4 * k
                cin = "0" if k == 0 else f"v[{d}:{d + 3}]"
                A(
                    f"  v_mfma_f32_16x16x32_bf16 v[{d}:{d + 3}], {wr}[{w}:{w + 3}], v[{y}:{y + 3}], {cin}"
                )
    A(".Lmfma_end:")
    A("  s_nop 7\n  s_nop 7\n  s_nop 7")
    for j in range(4):
        A(f"  s_add_u32 s36, s12, {4 * j}")
        A("  s_cmp_lt_u32 s36, s33")
        A("  s_cbranch_scc0 .Lend")
        for i in range(3):
            d = D + 12 * j + 4 * i
            A(
                f"  buffer_store_dwordx4 v[{d}:{d + 3}], v{9 + j}, s[24:27], 0 offen offset:{64 * i}"
            )
    A(".Lend:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_endpgm")
    src = kernel_source(
        name,
        c,
        KA_R,
        lds=0,
        vgprs=D + 48,
        agprs=60,
        sgprs=48,
        block=256,
    )
    return dict(name=name, src=src, ka=KA_R)


def source_shared(name: str = "ffn_shared_gu"):
    """Two token tiles per round (two rounds at M > 128, 132 VGPRs, so a router
    CTA fits beside it on a CU)."""
    tpr = 2  # token tiles per round
    # W scales (3 × 4), x scales (2 tiles × 4), xq (2 × 4 × 8), D (2 × 3 × 4)
    SA, XS, X, D = 24, 36, 44, 108
    c = []
    A = c.append
    A("  s_load_dwordx8 s[4:11], s[0:1], 0x0")  # w ws xq xs
    A("  s_load_dwordx4 s[12:15], s[0:1], 0x20")  # part, m s14
    A("  v_readfirstlane_b32 s16, v0")
    A("  s_lshr_b32 s16, s16, 6")  # wave
    A("  v_and_b32 v1, 63, v0")
    A("  v_and_b32 v2, 15, v1")  # r
    A("  v_lshrrev_b32 v3, 4, v1")  # lane group q
    A("  s_mul_hi_u32 s40, s2, 0xcccccccd")
    A("  s_lshr_b32 s40, s40, 3")  # row group = b / 10
    A(f"  s_mul_i32 s41, s40, {SS}")
    A("  s_sub_u32 s41, s2, s41")  # K slice
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_mul_i32 s17, s14, {H}")
    A("  s_mul_i32 s18, s14, 192")
    A(f"  s_mul_i32 s19, s14, {GPART_ROW}")
    c += rsrc(20, 4, nbytes=NS * H)
    c += rsrc(24, 6, nbytes=NS // 32 * (H // 32))
    c += rsrc(28, 8, nbytes_sgpr=17)
    c += rsrc(32, 10, nbytes_sgpr=18)
    c += rsrc(36, 12, nbytes_sgpr=19)
    A("  s_add_u32 s42, s14, 15")
    A("  s_lshr_b32 s42, s42, 4")  # token tiles
    A("  s_lshl_b32 s50, s41, 9")  # 512 s
    A(f"  s_mul_i32 s43, s40, {48 * H}")
    A("  s_add_u32 s43, s43, s50")
    for i in (1, 2):
        A(f"  s_add_u32 s{43 + i}, s{42 + i}, {16 * H}")  # weight rows of tile i
    # weight scale rows: tile i covers rows 16 (3 g + i) .. + 15, all in scale row (3 g + i) / 2
    A("  s_mul_i32 s60, s40, 3")
    A("  s_lshl_b32 s61, s41, 4")  # 16 s (4 blocks per K tile × 4 tiles)
    for i in range(3):
        A(f"  s_add_u32 s{46 + i}, s60, {i}")
        A(f"  s_lshr_b32 s{46 + i}, s{46 + i}, 1")
        A(f"  s_mul_i32 s{46 + i}, s{46 + i}, {H // 32}")
        A(f"  s_add_u32 s{46 + i}, s{46 + i}, s61")
    # xs byte of K tile k0 = 4 s + k in moe_ug1's layout: (k0 / 10)·12 + k0 % 10
    A("  s_lshl_b32 s62, s41, 2")
    for k in range(4):
        A(f"  s_add_u32 s63, s62, {k}")
        A("  s_mul_hi_u32 s64, s63, 0xcccccccd")
        A("  s_lshr_b32 s64, s64, 3")
        A("  s_mul_i32 s65, s64, 10")
        A("  s_sub_u32 s65, s63, s65")
        A("  s_mul_i32 s64, s64, 12")
        A(f"  s_add_u32 s{51 + k}, s64, s65")
    A(f"  v_mul_u32_u24 v4, {H}, v2")
    A("  v_lshl_add_u32 v4, v3, 4, v4")  # r·H + 16 q
    for i in range(3):
        for k in range(4):
            a = 8 * (4 * i + k)
            A(
                f"  buffer_load_dwordx4 a[{a}:{a + 3}], v4, s[20:23], s{43 + i} offen offset:{128 * k}"
            )
            A(
                f"  buffer_load_dwordx4 a[{a + 4}:{a + 7}], v4, s[20:23], s{43 + i} offen offset:{128 * k + 64}"
            )
            A(
                f"  buffer_load_ubyte v{SA + 4 * i + k}, v3, s[24:27], s{46 + i} offen offset:{4 * k}"
            )
    A("  v_lshl_add_u32 v5, v3, 4, s50")  # xq: 16 q + 512 s
    A("  v_mul_u32_u24 v6, 48, v3")  # xs: 48 q
    A(f"  s_mul_i32 s55, s41, {NS * 4}")
    A(f"  s_mul_i32 s56, s40, {48 * 4}")
    A("  s_add_u32 s55, s55, s56")
    A("  v_lshl_add_u32 v7, v3, 4, s55")  # gpart: 16 q + 4608 s + 192 group
    for j in range(4):
        A(f"  s_add_u32 s56, s16, {4 * j}")
        A("  s_lshl_b32 s56, s56, 4")
        A("  v_add_u32 v20, s56, v2")  # t
        A(f"  v_mul_u32_u24 v{8 + j}, {H}, v20")
        A(f"  v_add_u32 v{8 + j}, v{8 + j}, v5")
        A(f"  v_mul_u32_u24 v{12 + j}, 192, v20")
        A(f"  v_add_u32 v{12 + j}, v{12 + j}, v6")
        A(f"  v_mul_u32_u24 v{16 + j}, {GPART_ROW}, v20")
        A(f"  v_add_u32 v{16 + j}, v{16 + j}, v7")
    for rnd in range(4 // tpr):
        js = range(rnd * tpr, rnd * tpr + tpr)
        if rnd:  # the round's first tile past the end ends the work
            A(f"  s_add_u32 s56, s16, {4 * js[0]}")
            A("  s_cmp_lt_u32 s56, s42")
            A("  s_cbranch_scc0 .Lend")
        for j in js:
            for k in range(4):
                x = X + 32 * (j - js[0]) + 8 * k
                A(
                    f"  buffer_load_dwordx4 v[{x}:{x + 3}], v{8 + j}, s[28:31], 0 offen offset:{128 * k}"
                )
                A(
                    f"  buffer_load_dwordx4 v[{x + 4}:{x + 7}], v{8 + j}, s[28:31], 0 offen offset:{128 * k + 64}"
                )
                A(
                    f"  buffer_load_ubyte v{XS + 4 * (j - js[0]) + k}, v{12 + j}, s[32:35], s{51 + k} offen"
                )
        A("  s_waitcnt vmcnt(0)")
        for j in js:
            A(f"  s_add_u32 s56, s16, {4 * j}")
            A("  s_cmp_lt_u32 s56, s42")
            A(f"  s_cbranch_scc0 .Lmfma_end{rnd if rnd else ''}")
            for k in range(4):
                for i in range(3):
                    d, a, x = (
                        D + 12 * (j - js[0]) + 4 * i,
                        8 * (4 * i + k),
                        X + 32 * (j - js[0]) + 8 * k,
                    )
                    cin = "0" if k == 0 else f"v[{d}:{d + 3}]"
                    A(
                        f"  v_mfma_scale_f32_16x16x128_f8f6f4 v[{d}:{d + 3}], a[{a}:{a + 7}], v[{x}:{x + 7}], {cin}, "
                        f"v{SA + 4 * i + k}, v{XS + 4 * (j - js[0]) + k} op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0"
                    )
        A(f".Lmfma_end{rnd if rnd else ''}:")
        A("  s_nop 7\n  s_nop 7\n  s_nop 7")
        for j in js:
            A(f"  s_add_u32 s56, s16, {4 * j}")
            A("  s_cmp_lt_u32 s56, s42")
            A("  s_cbranch_scc0 .Lend")
            for i in range(3):
                d = D + 12 * (j - js[0]) + 4 * i
                A(
                    f"  buffer_store_dwordx4 v[{d}:{d + 3}], v{16 + j}, s[36:39], 0 offen offset:{64 * i}"
                )
    A(".Lend:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_endpgm")
    src = kernel_source(
        name, c, KA_S, lds=0, vgprs=D + 12 * tpr, agprs=96, sgprs=72, block=256
    )
    return dict(name=name, src=src, ka=KA_S)


KA_3V = KArgs(*KA_3.args, *[("mx_" + n, t) for n, t in ffn_seam.KA_MIX.args])


def source_p3(name: str = "ffn_p3v3"):
    c = []
    A = c.append
    A("  v_readfirstlane_b32 s21, v0")
    A("  s_lshr_b32 s21, s21, 6")  # wave
    A("  s_cmp_eq_u32 s21, 4")
    A("  s_cbranch_scc1 .Lmxwave")
    A("  s_load_dwordx16 s[4:19], s[0:1], 0x0")  # rpart bias logits ids wts gpart hq hs
    A("  s_load_dword s20, s[0:1], 0x40")  # m
    A(f"  s_load_dwordx2 s[96:97], s[0:1], {KA_3V.off['mx_partial']:#x}")
    A(f"  s_load_dwordx2 s[98:99], s[0:1], {KA_3V.off['mx_rstd']:#x}")
    A(f"  s_load_dword s100, s[0:1], {KA_3V.off['mx_norm_eps']:#x}")
    A("  v_and_b32 v1, 63, v0")  # lane
    A("  v_lshlrev_b32 v2, 2, v1")  # 4 lane
    A("  s_waitcnt lgkmcnt(0)")
    A("  v_mov_b32 v14, 0")
    A(f"  s_mul_i32 s101, s2, {ffn_seam.SLICES * 128}")
    A("  v_lshlrev_b32 v15, 7, v1")
    A("  v_add_u32 v15, s101, v15")
    A("  global_load_dword v12, v15, s[96:97] offset:100")
    A("  v_add_u32 v15, 0x2000, v15")
    A("  global_load_dword v13, v15, s[96:97] offset:100")
    A("  v_cmp_gt_u32 vcc, 32, v1")
    A("  s_and_saveexec_b64 s[22:23], vcc")
    A("  v_add_u32 v15, 0x2000, v15")
    A("  global_load_dword v14, v15, s[96:97] offset:100")
    A("  s_mov_b64 exec, s[22:23]")
    A(f"  s_mul_i32 s22, s20, {RPART_ROW}")
    A(f"  s_mul_i32 s23, s20, {E * 4}")
    A(f"  s_mul_i32 s56, s20, {TOPK * 4}")
    A(f"  s_mul_i32 s57, s20, {GPART_ROW}")
    A(f"  s_mul_i32 s58, s20, {I_SH}")
    A(f"  s_mul_i32 s59, s20, {I_SH // 32}")
    c += rsrc(24, 4, nbytes_sgpr=22)  # rpart
    c += rsrc(28, 6, nbytes=E * 4)  # bias
    c += rsrc(32, 8, nbytes_sgpr=23)  # logits
    c += rsrc(36, 10, nbytes_sgpr=56)  # ids
    c += rsrc(40, 12, nbytes_sgpr=56)  # wts
    c += rsrc(44, 14, nbytes_sgpr=57)  # gpart
    c += rsrc(48, 16, nbytes_sgpr=58)  # hq
    c += rsrc(52, 18, nbytes_sgpr=59)  # hs
    # per-token offsets
    A(f"  s_mul_i32 s60, s2, {RPART_ROW}")
    A(f"  s_mul_i32 s61, s21, {8 * E * 4}")
    A("  s_add_u32 s60, s60, s61")  # rpart: t, slices 8w ..
    A(f"  s_mul_i32 s61, s2, {E * 4}")  # logits
    A(f"  s_mul_i32 s62, s2, {TOPK * 4}")  # ids / wts
    A(f"  s_mul_i32 s63, s2, {GPART_ROW}")  # gpart
    A(f"  s_mul_i32 s64, s2, {I_SH}")  # hq
    A(f"  s_mul_i32 s65, s2, {I_SH // 32}")  # hs
    # ---- router: this wave's 8 slices, experts l + 64 j
    for sl in range(8):
        A(f"  v_add_u32 v{3 + sl}, {sl * E * 4}, v2")
    for sl in range(8):
        for j in range(6):
            A(
                f"  buffer_load_dword v{16 + 6 * sl + j}, v{3 + sl}, s[24:27], s60 offen offset:{256 * j}"
            )
    A("  s_waitcnt vmcnt(0)")
    for sl in range(1, 8):
        for j in range(6):
            A(f"  v_add_f32 v{16 + j}, v{16 + j}, v{16 + 6 * sl + j}")
    A("  v_add_f32 v12, v12, v13")
    A("  v_add_f32 v12, v12, v14")
    for ctl in ("row_ror:8", "row_ror:4", "row_ror:2", "row_ror:1"):
        A("  s_nop 1")
        A(f"  v_add_f32_dpp v12, v12, v12 {ctl} row_mask:0xf bank_mask:0xf")
    for sw in ("v_permlane16_swap_b32", "v_permlane32_swap_b32"):
        A("  v_mov_b32 v13, v12")
        A("  s_nop 1")
        A(f"  {sw} v12, v13")
        A("  s_nop 3")
        A("  v_add_f32 v12, v12, v13")
    A("  v_mov_b32 v13, s100")
    A(
        f"  s_mov_b32 s22, {f32(1.0 / H)}"
    )  # (s22's rpart size is in the descriptor already)
    A("  v_fma_f32 v12, v12, s22, v13")
    A("  v_rsq_f32 v12, v12")
    A("  s_nop 1")
    A("  v_readfirstlane_b32 s101, v12")
    A(f"  s_mul_i32 s66, s21, {E * 4}")
    A("  v_add_u32 v11, s66, v2")
    for j in range(6):
        A(f"  ds_write_b32 v11, v{16 + j} offset:{256 * j}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    A("  s_cmp_eq_u32 s21, 0")
    A("  s_cbranch_scc0 .Lshared")
    # ---- wave 0: logits, scores, top 6, weights
    for j in range(6):
        A(
            f"  buffer_load_dword v{40 + j}, v2, s[28:31], 0 offen offset:{256 * j}"
        )  # bias
    for w in range(4):
        for j in range(6):
            A(f"  ds_read_b32 v{16 + 6 * w + j}, v2 offset:{w * E * 4 + 256 * j}")
    A("  s_waitcnt lgkmcnt(0)")
    for j in range(6):
        A(f"  v_add_f32 v{64 + j}, v{16 + j}, v{22 + j}")
        A(f"  v_add_f32 v{64 + j}, v{64 + j}, v{28 + j}")
        A(f"  v_add_f32 v{64 + j}, v{64 + j}, v{34 + j}")
        A(f"  v_mul_f32 v{64 + j}, s101, v{64 + j}")
        A(f"  buffer_store_dword v{64 + j}, v2, s[32:35], s61 offen offset:{256 * j}")
    A("  s_lshl_b32 s22, s2, 2")
    A("  v_mov_b32 v12, s22")
    A("  v_mov_b32 v13, s101")
    A("  s_mov_b64 exec, 1")
    A("  global_store_dword v12, v13, s[98:99]")
    A("  s_mov_b64 exec, -1")
    # score = sqrt(x > 20 ? x : ln(1 + e^x)); biased = score + bias → v[72:77]
    for j in range(6):
        x, b = 64 + j, 72 + j
        A(f"  v_mul_f32 v{b}, {f32(1.4426950408889634)}, v{x}")
        A(f"  v_exp_f32 v{b}, v{b}")
        A("  s_nop 0")
        A(f"  v_add_f32 v{b}, 1.0, v{b}")
        A(f"  v_log_f32 v{b}, v{b}")
        A("  s_nop 0")
        A(f"  v_mul_f32 v{b}, {f32(0.6931471805599453)}, v{b}")
        A(f"  v_cmp_lt_f32 vcc, {f32(20.0)}, v{x}")
        A(f"  v_cndmask_b32 v{b}, v{b}, v{x}, vcc")
        A(f"  v_sqrt_f32 v{b}, v{b}")
        A("  s_nop 0")
    A("  s_waitcnt vmcnt(0)")
    for j in range(6):
        A(f"  v_add_f32 v{72 + j}, v{72 + j}, v{40 + j}")
        A(f"  v_add_u32 v{78 + j}, {64 * j}, v1")  # expert index
    A("  v_mov_b32 v89, 0xff800000")  # -inf
    for i, d in enumerate((1, 2, 4, 8, 16, 32)):
        A(f"  v_xor_b32 v{90 + i}, {d}, v1")
        A(f"  v_lshlrev_b32 v{90 + i}, 2, v{90 + i}")
    for k in range(TOPK):
        # this lane's best (strict >: the lower index wins ties), then the wave's
        A("  v_mov_b32 v84, v72")
        A("  v_mov_b32 v85, v78")
        for j in range(1, 6):
            A(f"  v_cmp_gt_f32 vcc, v{72 + j}, v84")
            A(f"  v_cndmask_b32 v84, v84, v{72 + j}, vcc")
            A(f"  v_cndmask_b32 v85, v85, v{78 + j}, vcc")
        for i in range(6):
            A(f"  ds_bpermute_b32 v87, v{90 + i}, v84")
            A(f"  ds_bpermute_b32 v88, v{90 + i}, v85")
            A("  s_waitcnt lgkmcnt(0)")
            A("  v_cmp_gt_f32 s[68:69], v87, v84")
            A("  v_cmp_eq_f32 s[70:71], v87, v84")
            A("  v_cmp_lt_u32 s[72:73], v88, v85")
            A("  s_and_b64 s[70:71], s[70:71], s[72:73]")
            A("  s_or_b64 vcc, s[68:69], s[70:71]")
            A("  v_cndmask_b32 v84, v84, v87, vcc")
            A("  v_cndmask_b32 v85, v85, v88, vcc")
        A("  s_nop 0")  # readfirstlane right after a VALU write reads the old value
        A(f"  v_readfirstlane_b32 s{76 + k}, v85")
        A(f"  v_readfirstlane_b32 s{82 + k}, v84")
        for j in range(6):
            A(f"  v_cmp_eq_u32 vcc, s{76 + k}, v{78 + j}")
            A(f"  v_cndmask_b32 v{72 + j}, v{72 + j}, v89, vcc")
    # lanes 0..5: id, biased value, bias[id] → weight
    for k in range(TOPK):
        A(f"  v_writelane_b32 v96, s{76 + k}, {k}")
        A(f"  v_writelane_b32 v97, s{82 + k}, {k}")
    A("  s_mov_b64 exec, 0x3f")
    A("  v_lshlrev_b32 v98, 2, v96")
    A("  buffer_load_dword v99, v98, s[28:31], 0 offen")
    A("  s_waitcnt vmcnt(0)")
    A("  v_sub_f32 v100, v97, v99")
    A("  s_nop 0")
    for k in range(TOPK):
        A(f"  v_readlane_b32 s{88 + k}, v100, {k}")
    A("  v_mov_b32 v101, s88")
    for k in range(1, TOPK):
        A(f"  v_add_f32 v101, s{88 + k}, v101")
    # vLLM's guard (topk_softplus_sqrt: denom = sum > 0 ? sum : 1): a massive-activation row can have every
    # picked score sqrt(softplus(logit)) = 0 (1 + e^x rounds to 1 below x ≈ -17), and rcp(0) would make 0 · inf = NaN
    A("  v_cmp_lt_f32 vcc, 0, v101")
    A("  v_cndmask_b32 v101, 1.0, v101, vcc")
    A("  v_rcp_f32 v102, v101")
    A("  s_nop 0")
    A("  v_mul_f32 v103, v100, v102")
    A(f"  v_mul_f32 v103, {f32(1.5)}, v103")
    A("  buffer_store_dword v96, v2, s[36:39], s62 offen")
    A("  buffer_store_dword v103, v2, s[40:43], s62 offen")
    A("  s_mov_b64 exec, -1")
    A("  s_branch .Lend")
    # ---- waves 2-3: shared activation, thread i = tid − 128 < 72 owns columns 8i .. 8i + 7
    A(".Lshared:")
    A("  s_cmp_lt_u32 s21, 2")
    A("  s_cbranch_scc1 .Lend")
    A("  v_subrev_u32 v26, 128, v0")
    A("  v_cmp_gt_u32 vcc, 72, v26")
    A("  s_and_saveexec_b64 s[66:67], vcc")
    A("  v_lshlrev_b32 v27, 5, v26")  # 32 i
    for s in range(SS):
        A(f"  v_add_u32 v{16 + s}, {s * NS * 4}, v27")
    G, U = 32, 112
    for s in range(SS):
        for h, base in ((0, G), (I_SH * 4, U)):
            A(
                f"  buffer_load_dwordx4 v[{base + 8 * s}:{base + 8 * s + 3}], v{16 + s}, s[44:47], s63 offen offset:{h}"
            )
            A(
                f"  buffer_load_dwordx4 v[{base + 8 * s + 4}:{base + 8 * s + 7}], v{16 + s}, s[44:47], s63 offen "
                f"offset:{h + 16}"
            )
    A("  s_waitcnt vmcnt(0)")
    for base in (G, U):
        for s in range(1, SS):
            for q in range(8):
                A(f"  v_add_f32 v{base + q}, v{base + q}, v{base + 8 * s + q}")
        for q in range(8):
            A(f"  v_mul_f32 v{base + q}, s101, v{base + q}")
        for p in range(4):  # bf16 (gate_up's output dtype)
            A(f"  v_cvt_pk_bf16_f32 v192, v{base + 2 * p}, v{base + 2 * p + 1}")
            A(f"  v_lshlrev_b32 v{base + 2 * p}, 16, v192")
            A(f"  v_and_b32 v{base + 2 * p + 1}, 0xffff0000, v192")
    A("  v_mov_b32 v193, 0x41200000")  # 10
    A("  v_mov_b32 v194, 0xc1200000")  # -10
    gs, us, ex = (
        [G + q for q in range(8)],
        [U + q for q in range(8)],
        [195 + q for q in range(8)],
    )
    for g, u, x in zip(gs, us, ex):
        A(f"  v_min_f32 v{g}, v193, v{g}")
        A(f"  v_med3_f32 v{u}, v{u}, v194, v193")
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
    for p in range(4):  # h → bf16 → f32
        A(f"  v_cvt_pk_bf16_f32 v192, v{G + 2 * p}, v{G + 2 * p + 1}")
        A(f"  v_lshlrev_b32 v{G + 2 * p}, 16, v192")
        A(f"  v_and_b32 v{G + 2 * p + 1}, 0xffff0000, v192")
    # MXFP8: the quad (4 threads = 32 columns) shares one E8M0, ceil rule as quant_block
    A(f"  v_max_f32 v203, |v{G}|, |v{G + 1}|")
    for q in range(2, 8):
        A(f"  v_max_f32 v203, v203, |v{G + q}|")
    for ctl in ("quad_perm:[1,0,3,2]", "quad_perm:[2,3,0,1]"):
        A("  s_nop 1")
        A(f"  v_max_f32_dpp v203, v203, v203 {ctl} row_mask:0xf bank_mask:0xf")
    A("  v_frexp_exp_i32_f32 v204, v203")
    A("  v_frexp_mant_f32 v205, v203")
    A("  v_cmp_lt_f32 vcc, 0x3f600000, v205")
    A("  v_addc_co_u32 v204, vcc, -9, v204, vcc")
    A("  v_max_i32 v204, 0xffffff81, v204")
    A("  v_min_i32 v204, 0x7f, v204")
    A("  v_sub_u32 v206, 0, v204")
    for q in range(8):
        A(f"  v_ldexp_f32 v{G + q}, v{G + q}, v206")
    for p in range(2):
        A(f"  v_cvt_pk_fp8_f32 v{210 + p}, v{G + 4 * p}, v{G + 4 * p + 1}")
        A(
            f"  v_cvt_pk_fp8_f32 v{210 + p}, v{G + 4 * p + 2}, v{G + 4 * p + 3} op_sel:[0,0,1]"
        )
    A("  v_lshlrev_b32 v212, 3, v26")
    A("  buffer_store_dwordx2 v[210:211], v212, s[48:51], s64 offen")
    A("  v_and_b32 v213, 3, v26")
    A("  v_cmp_eq_u32 vcc, 0, v213")
    A("  s_and_b64 exec, exec, vcc")
    A("  v_lshrrev_b32 v214, 2, v26")
    A("  v_add_u32 v215, 0x7f, v204")
    A("  buffer_store_byte v215, v214, s[52:55], s65 offen")
    A(".Lend:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_endpgm")
    # the mixes wave: token = workgroup id, steps through the one s_barrier above
    A(".Lmxwave:")
    c += ffn_seam.mix_wave(
        KA_3V.off["mx_partial"], MAX_M, 4 * E * 4, nbar=1, rstd_store=False
    )
    src = kernel_source(
        name,
        c,
        KA_3V,
        lds=4 * E * 4 + ffn_seam.MIX_LDS,
        vgprs=216,
        agprs=0,
        sgprs=102,
        block=320,
    )
    return dict(name=name, src=src, ka=KA_3V, block=320)


KA_D = KArgs(
    ("w", "ptr"),
    ("ws", "ptr"),
    ("hq", "ptr"),
    ("hs", "ptr"),
    ("out", "ptr"),
    ("m", "u32"),
)
D_GRID = H // 32  # 160


def source_shared_down(name: str = "ffn_shared_down"):
    """Shared down projection: out [M, 5120] bf16 = MXFP8 h [M, 576] (hs
    [M, 18]) · W [5120, 576]^T (e4m3, E8M0 per 32 × 32 block, [160, 18]).
    CTA b owns rows 32b .. 32b + 31 (2 tiles, one scale row); every wave holds
    them in AGPRs and computes token tiles w, w + 4, w + 8, w + 12. K = 576 is
    4.5 K tiles: in the last one each lane loads only its first 16 bytes (K
    512 + 16 q) and the second half (K 576 + ..) is zero on both sides, so the
    scale bytes read there for blocks 18-19 (the next row's, or 0 past the end)
    multiply zeros."""
    SA, XS, X, D = (
        20,
        25,
        48,
        208,
    )  # W scales (5 K tiles), h scales (4 tiles × 5), h (4 × 40), D (4 × 2 × 4)
    KT = 5
    c = []
    A = c.append
    A("  s_load_dwordx8 s[4:11], s[0:1], 0x0")  # w ws hq hs
    A("  s_load_dwordx4 s[12:15], s[0:1], 0x20")  # out, m s14
    A("  v_readfirstlane_b32 s16, v0")
    A("  s_lshr_b32 s16, s16, 6")  # wave
    A("  v_and_b32 v1, 63, v0")
    A("  v_and_b32 v2, 15, v1")  # r
    A("  v_lshrrev_b32 v3, 4, v1")  # lane group q
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_mul_i32 s17, s14, {I_SH}")
    A(f"  s_mul_i32 s18, s14, {I_SH // 32}")
    A(f"  s_mul_i32 s19, s14, {2 * H}")
    c += rsrc(20, 4, nbytes=H * I_SH)
    c += rsrc(24, 6, nbytes=H // 32 * (I_SH // 32))
    c += rsrc(28, 8, nbytes_sgpr=17)
    c += rsrc(32, 10, nbytes_sgpr=18)
    c += rsrc(36, 12, nbytes_sgpr=19)
    A("  s_add_u32 s44, s14, 15")
    A("  s_lshr_b32 s44, s44, 4")  # token tiles
    A(f"  s_mul_i32 s40, s2, {32 * I_SH}")  # W rows of tile 0
    A(f"  s_add_u32 s41, s40, {16 * I_SH}")  # tile 1
    A(f"  s_mul_i32 s42, s2, {I_SH // 32}")  # W scale row b
    A("  s_lshl_b32 s43, s2, 6")  # out: 64 b (32 rows × bf16)
    A(f"  v_mul_u32_u24 v4, {I_SH}, v2")
    A("  v_lshl_add_u32 v4, v3, 4, v4")  # r·576 + 16 q
    for i in range(2):
        for k in range(KT):
            a = 8 * (KT * i + k)
            A(
                f"  buffer_load_dwordx4 a[{a}:{a + 3}], v4, s[20:23], s{40 + i} offen offset:{128 * k}"
            )
            if k < KT - 1:
                A(
                    f"  buffer_load_dwordx4 a[{a + 4}:{a + 7}], v4, s[20:23], s{40 + i} offen offset:{128 * k + 64}"
                )
            else:
                for z in range(4):
                    A(f"  v_accvgpr_write_b32 a{a + 4 + z}, 0")
    for k in range(KT):
        A(f"  buffer_load_ubyte v{SA + k}, v3, s[24:27], s42 offen offset:{4 * k}")
    A("  v_lshl_add_u32 v17, v3, 3, s43")  # out: 8 q + 64 b
    for j in range(4):
        A(f"  s_add_u32 s45, s16, {4 * j}")
        A("  s_lshl_b32 s45, s45, 4")
        A("  v_add_u32 v18, s45, v2")  # t
        A(f"  v_mul_u32_u24 v{5 + j}, {I_SH}, v18")
        A(f"  v_lshl_add_u32 v{5 + j}, v3, 4, v{5 + j}")
        A(f"  v_mul_u32_u24 v{9 + j}, {I_SH // 32}, v18")
        A(f"  v_add_u32 v{9 + j}, v{9 + j}, v3")
        A(f"  v_mul_u32_u24 v{13 + j}, {2 * H}, v18")
        A(f"  v_add_u32 v{13 + j}, v{13 + j}, v17")
    for j in range(4):
        for k in range(KT):
            x = X + 40 * j + 8 * k
            A(
                f"  buffer_load_dwordx4 v[{x}:{x + 3}], v{5 + j}, s[28:31], 0 offen offset:{128 * k}"
            )
            if k < KT - 1:
                A(
                    f"  buffer_load_dwordx4 v[{x + 4}:{x + 7}], v{5 + j}, s[28:31], 0 offen offset:{128 * k + 64}"
                )
            else:
                for z in range(4):
                    A(f"  v_mov_b32 v{x + 4 + z}, 0")
            A(
                f"  buffer_load_ubyte v{XS + KT * j + k}, v{9 + j}, s[32:35], 0 offen offset:{4 * k}"
            )
    A("  s_waitcnt vmcnt(0)")
    for j in range(4):
        A(f"  s_add_u32 s45, s16, {4 * j}")
        A("  s_cmp_lt_u32 s45, s44")
        A("  s_cbranch_scc0 .Lmfma_end")
        for k in range(KT):
            for i in range(2):
                d, a, x = D + 8 * j + 4 * i, 8 * (KT * i + k), X + 40 * j + 8 * k
                cin = "0" if k == 0 else f"v[{d}:{d + 3}]"
                A(
                    f"  v_mfma_scale_f32_16x16x128_f8f6f4 v[{d}:{d + 3}], a[{a}:{a + 7}], v[{x}:{x + 7}], {cin}, "
                    f"v{SA + k}, v{XS + KT * j + k} op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0"
                )
    A(".Lmfma_end:")
    A("  s_nop 7\n  s_nop 7\n  s_nop 7")
    # D lane: token r, rows 4 q .. 4 q + 3 of tile i → bf16 pairs, one dwordx2
    for j in range(4):
        A(f"  s_add_u32 s45, s16, {4 * j}")
        A("  s_cmp_lt_u32 s45, s44")
        A("  s_cbranch_scc0 .Lend")
        for i in range(2):
            d = D + 8 * j + 4 * i
            A(f"  v_cvt_pk_bf16_f32 v{d}, v{d}, v{d + 1}")
            A(f"  v_cvt_pk_bf16_f32 v{d + 1}, v{d + 2}, v{d + 3}")
            A(
                f"  buffer_store_dwordx2 v[{d}:{d + 1}], v{13 + j}, s[36:39], 0 offen offset:{32 * i}"
            )
    A(".Lend:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_endpgm")
    src = kernel_source(
        name, c, KA_D, lds=0, vgprs=D + 32, agprs=8 * KT * 2, sgprs=48, block=256
    )
    return dict(name=name, src=src, ka=KA_D)
