# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""kv RMSNorm + RoPE and the SWA KV insert as a 5th wave of wq_b's launch
(gemv8.source(kv_wave=True)), on wqkv's bf16 qr_kv [M, 1792].

One wave per token (the workgroup's). Lane l loads 32 bf16 of the token's row:
lanes 0..39 the q block l (dims 32l ..), lanes 40..55 the kv dims 32 (l − 40) ..
- RMSNorm as vLLM's Triton kernel: fp32, y = bf16((x · rsqrt(Σx²/n + eps)) · w)
  (`v_rsq_f32`; Σx² over the group's lanes by DPP / permlane, fixed order).
- q: MXFP8 per 32 dims (ceil rule) → qr in gemv8's frag layout (branched over:
  wq_b normalises q itself).
- kv: GPT-J RoPE on dims 448.. in fp32, bf16; then vLLM's fp8_ds_mla record
  (fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu): per 64 dims exponent =
  ceil(log2(max(absmax, 1e-4) / 448)) with an IEEE-rounded quotient and
  `v_log_f32`, codes = e4m3(clamp(x · 2^-exponent, ±448)); RoPE dims stay
  bf16; scale bytes exponent + 127, byte 7 = 0. Written to the SWA cache at
  slot_mapping[t] (block = slot >> bsl, row = slot & (bs − 1)) unless < 0.
"""

from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs

QL, KVD, NOPE = 1280, 512, 448
KGQ = QL // 128  # wq_b's x: K tiles

KA = KArgs(
    ("qr_kv", "ptr"),
    ("qw", "ptr"),
    ("kvw", "ptr"),
    ("pos", "ptr"),
    ("cs", "ptr"),
    ("slots", "ptr"),
    ("cache", "ptr"),
    ("xq", "ptr"),
    ("xs", "ptr"),
    ("m", "u32"),
    ("eps", "u32"),
    ("stride", "u32"),
    ("bsl", "u32"),
)
KA_R = KArgs(*KA.args, ("rstd", "ptr"))


def f32(v: float) -> str:
    import struct

    return f"{struct.unpack('<I', struct.pack('<f', v))[0]:#x}"


def source(ka_base: int, bar_m: str) -> list[str]:
    """The kv wave of wq_b's launch (a 5th wave): token t = workgroup id, q lanes
    don't store (wq_b normalises q itself); returns the code lines (labels .Lkv*).
    KA_R's arguments sit at kernarg offset ka_base (appended to the host GEMV's,
    gemv8.source(kv_wave=True)). qr_kv's row lacks the attention norm's rstd:
    x = bf16(x · rstd[t]) before the norm (vLLM's bf16 qr_kv of the normalised
    input). bar_m: the host GEMV's m in that SGPR (gemv8 mode k with S = 1): all
    loads up front (pos / slot, then cos and sin with the row), then the host's
    first 2·⌈m / 64⌉ − 1 barriers, the math, and its last barrier (measured: a
    spare wave holds the workgroup's barriers while it lives)."""
    CS, SN = 96, 112  # cos / sin of the RoPE lanes
    c = []
    A = c.append
    EXIT = ".Lkv_exit"

    def nop(n):
        while n > 0:
            A(f"  s_nop {min(n, 8) - 1}")
            n -= 8

    X = 8  # 32 values (f32), v[8:39] (v[4:8] are address temps before x)
    T = 40  # temps v[40:63]
    V_L, V_T0 = 1, 2

    def wave_sum(v, tmp):
        """V ← Σ over all 64 lanes (fixed order: DPP within rows, then permlane swaps)."""
        for ctl in (
            "quad_perm:[1,0,3,2]",
            "quad_perm:[2,3,0,1]",
            "row_ror:4",
            "row_ror:8",
        ):
            A(f"  v_mov_b32 v{tmp}, v{v}")
            nop(2)
            A(f"  v_add_f32_dpp v{v}, v{tmp}, v{v} {ctl} row_mask:0xf bank_mask:0xf")
        for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32"):
            A(f"  v_mov_b32 v{tmp}, v{v}")
            nop(2)
            A(f"  {sw} v{v}, v{tmp}")
            nop(4)
            A(f"  v_add_f32 v{v}, v{v}, v{tmp}")

    A(
        f"  s_load_dwordx16 s[4:19], s[0:1], {ka_base:#x}"
    )  # qr_kv qw kvw pos cs slots cache xq
    A(f"  s_load_dwordx4 s[20:23], s[0:1], {ka_base + 0x40:#x}")  # xs, m s22, eps s23
    A(f"  s_load_dwordx2 s[24:25], s[0:1], {ka_base + 0x50:#x}")  # stride, bsl
    A(f"  s_load_dwordx2 s[50:51], s[0:1], {ka_base + KA_R.off['rstd']:#x}")
    A("  v_readfirstlane_b32 s26, v0")
    A("  s_lshr_b32 s26, s26, 6")
    A(f"  v_and_b32 v{V_L}, 63, v0")
    A("  s_mov_b32 s27, s2")  # token t = workgroup
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_cmp_ge_u32 s27, s22")
    A(f"  s_cbranch_scc1 {EXIT}")
    A("  s_lshl_b32 s52, s27, 2")
    A("  s_load_dword s52, s[50:51], s52")  # rstd[t] (waited for with the loads below)
    # row: lane l < 56 loads 64 B at t·3584 + 64 l
    A(f"  s_mul_i32 s28, s27, {2 * (QL + KVD)}")
    A("  s_add_u32 s28, s4, s28")
    A("  s_addc_u32 s29, s5, 0")
    A(f"  v_lshlrev_b32 v{V_T0}, 6, v{V_L}")
    A(f"  v_min_u32 v{V_T0}, {55 * 64}, v{V_T0}")
    for k in range(4):
        A(
            f"  global_load_dwordx4 v[{T + 4 * k}:{T + 4 * k + 3}], v{V_T0}, s[28:29] offset:{16 * k}"
        )
    # weights: q_norm[32 l ..] (lanes < 40), kv_norm[32 (l − 40) ..] (lanes 40..55)
    A(f"  v_cmp_gt_u32 vcc, 40, v{V_L}")
    A(f"  v_lshlrev_b32 v4, 6, v{V_L}")  # q offset 64 l
    A(f"  v_subrev_u32 v5, {40 * 64}, v4")
    A("  v_min_u32 v5, 960, v5")  # kv offset 64 (l − 40), clamped
    A("  v_min_u32 v4, 2496, v4")
    A("  v_cndmask_b32 v4, v5, v4, vcc")
    A("  v_mov_b32 v5, 0")
    A("  v_mov_b32 v6, s6")
    A("  v_mov_b32 v7, s7")
    A("  v_mov_b32 v8, s8")
    A("  v_cndmask_b32 v6, v8, v6, vcc")
    A("  v_mov_b32 v8, s9")
    A("  v_cndmask_b32 v7, v8, v7, vcc")
    A("  v_lshl_add_u64 v[6:7], v[4:5], 0, v[6:7]")
    for k in range(4):
        A(
            f"  global_load_dwordx4 v[{T + 16 + 4 * k}:{T + 16 + 4 * k + 3}], v[6:7], off offset:{16 * k}"
        )
    A("  s_lshl_b32 s34, s27, 3")
    A("  s_load_dwordx2 s[36:37], s[10:11], s34")  # pos[t]
    A("  s_load_dwordx2 s[38:39], s[14:15], s34")  # slot[t]
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_cmp_lt_i32 s39, 0")
    A(f"  s_cbranch_scc1 {EXIT}")  # pad slot: nothing to do (q is wq_b's)
    A("  v_subrev_u32 v4, 54, v1")  # RoPE lanes 54, 55: cos / sin rows now
    A("  v_cmp_gt_u32 vcc, 2, v4")
    A("  s_and_saveexec_b64 s[40:41], vcc")
    A("  s_lshl_b64 s[42:43], s[36:37], 8")
    A("  s_add_u32 s42, s12, s42")
    A("  s_addc_u32 s43, s13, s43")
    A("  v_lshlrev_b32 v5, 6, v4")
    for k in range(4):
        A(
            f"  global_load_dwordx4 v[{CS + 4 * k}:{CS + 3 + 4 * k}], v5, s[42:43] offset:{16 * k}"
        )
    for k in range(4):
        A(
            f"  global_load_dwordx4 v[{SN + 4 * k}:{SN + 3 + 4 * k}], v5, s[42:43] offset:{128 + 16 * k}"
        )
    A("  s_mov_b64 exec, s[40:41]")
    A(f"  s_add_u32 s54, {bar_m}, 63")  # the host's barriers but its last: 2·⌈m/64⌉ − 1
    A("  s_lshr_b32 s54, s54, 6")
    A("  s_lshl_b32 s54, s54, 1")
    A("  s_sub_u32 s54, s54, 1")
    A(".Lkv_bar:")
    A("  s_cmp_eq_u32 s54, 0")
    A("  s_cbranch_scc1 .Lkv_bar_done")
    A("  s_barrier")
    A("  s_sub_u32 s54, s54, 1")
    A("  s_branch .Lkv_bar")
    A(".Lkv_bar_done:")
    A("  s_waitcnt vmcnt(0)")
    # unpack x (bf16 pairs) → f32 v[X:X+32]
    for k in range(16):
        A(f"  v_lshlrev_b32 v{X + 2 * k}, 16, v{T + k}")
        A(f"  v_and_b32 v{X + 2 * k + 1}, 0xffff0000, v{T + k}")
    A("  s_waitcnt lgkmcnt(0)")
    for k in range(16):
        A(f"  v_mul_f32 v{X + 2 * k}, s52, v{X + 2 * k}")
        A(f"  v_mul_f32 v{X + 2 * k + 1}, s52, v{X + 2 * k + 1}")
        A(f"  v_cvt_pk_bf16_f32 v{T}, v{X + 2 * k}, v{X + 2 * k + 1}")
        A(f"  v_lshlrev_b32 v{X + 2 * k}, 16, v{T}")
        A(f"  v_and_b32 v{X + 2 * k + 1}, 0xffff0000, v{T}")
    # Σx² of the lane, then per group (q: lanes < 40, kv: 40..55)
    A(f"  v_mul_f32 v{T}, v{X}, v{X}")
    for i in range(1, 32):
        A(f"  v_fmac_f32 v{T}, v{X + i}, v{X + i}")
    A(f"  v_cmp_gt_u32 vcc, 40, v{V_L}")
    A(f"  v_cndmask_b32 v{T + 1}, 0, v{T}, vcc")  # q part
    A(
        f"  v_cndmask_b32 v{T + 2}, v{T}, 0, vcc"
    )  # kv part (lanes ≥ 56 hold 0: masked load clamps → re-zero)
    A(f"  v_cmp_gt_u32 vcc, 56, v{V_L}")
    A(f"  v_cndmask_b32 v{T + 2}, 0, v{T + 2}, vcc")
    wave_sum(T + 1, T + 3)
    wave_sum(T + 2, T + 3)
    # rrms = rsq(Σ/n + eps): q and kv, select per lane
    A(f"  v_mov_b32 v{T + 4}, {f32(float(QL))}")
    A(f"  v_mov_b32 v{T + 5}, {f32(float(KVD))}")

    def ieee_div(q, a, b_reg, tmp):
        """Q = a / b correctly rounded (b > 0 normal, a ≥ 0 normal or 0)."""
        r, e = tmp, tmp + 1
        A(f"  v_rcp_f32 v{r}, v{b_reg}")
        A("  s_nop 0")
        A(f"  v_fma_f32 v{e}, -v{b_reg}, v{r}, 1.0")
        A(f"  v_fma_f32 v{r}, v{e}, v{r}, v{r}")
        A(f"  v_mul_f32 v{q}, v{a}, v{r}")
        A(f"  v_fma_f32 v{e}, -v{b_reg}, v{q}, v{a}")
        A(f"  v_fma_f32 v{q}, v{e}, v{r}, v{q}")
        A(f"  v_fma_f32 v{e}, -v{b_reg}, v{q}, v{a}")
        A(f"  v_fma_f32 v{q}, v{e}, v{r}, v{q}")

    ieee_div(T + 6, T + 1, T + 4, T + 8)
    ieee_div(T + 7, T + 2, T + 5, T + 8)
    A(f"  v_add_f32 v{T + 6}, s23, v{T + 6}")
    A(f"  v_add_f32 v{T + 7}, s23, v{T + 7}")
    A(f"  v_rsq_f32 v{T + 6}, v{T + 6}")
    A(f"  v_rsq_f32 v{T + 7}, v{T + 7}")
    A("  s_nop 0")
    A(f"  v_cmp_gt_u32 vcc, 40, v{V_L}")
    A(f"  v_cndmask_b32 v{T}, v{T + 7}, v{T + 6}, vcc")  # this lane's rrms
    A("  s_waitcnt vmcnt(0)")
    # y = bf16((x · rrms) · w), kept as f32 of the bf16
    for k in range(16):
        w = T + 16 + k
        for h in range(2):
            xi = X + 2 * k + h
            A(f"  v_mul_f32 v{xi}, v{xi}, v{T}")
            if h == 0:
                A(f"  v_lshlrev_b32 v{T + 1}, 16, v{w}")
            else:
                A(f"  v_and_b32 v{T + 1}, 0xffff0000, v{w}")
            A(f"  v_mul_f32 v{xi}, v{xi}, v{T + 1}")
        A(f"  v_cvt_pk_bf16_f32 v{T + 2}, v{X + 2 * k}, v{X + 2 * k + 1}")
        A(f"  v_lshlrev_b32 v{X + 2 * k}, 16, v{T + 2}")
        A(f"  v_and_b32 v{X + 2 * k + 1}, 0xffff0000, v{T + 2}")
    # ---- q lanes: MXFP8 per lane block → frag [j][kg][h][r + 16 q'][16 B], scales [j][kg][r + 16 (b % 4)]
    A("  s_branch .Lkv_q_done")
    A(f"  v_cmp_gt_u32 vcc, 40, v{V_L}")
    A("  s_and_saveexec_b64 s[30:31], vcc")
    a, e, mt, ng = T, T + 1, T + 2, T + 3
    A(f"  v_max_f32 v{a}, |v{X}|, |v{X + 1}|")
    for i in range(2, 32):
        A(f"  v_max_f32 v{a}, v{a}, |v{X + i}|")
    A(f"  v_max_f32 v{a}, 0x00800000, v{a}")
    A(f"  v_frexp_exp_i32_f32 v{e}, v{a}")
    A(f"  v_frexp_mant_f32 v{mt}, v{a}")
    A(f"  v_cmp_lt_f32 vcc, 0x3f600000, v{mt}")
    A(f"  v_addc_co_u32 v{e}, vcc, -9, v{e}, vcc")
    A(f"  v_max_i32 v{e}, 0xffffff81, v{e}")
    A(f"  v_min_i32 v{e}, 0x7f, v{e}")
    A(f"  v_sub_u32 v{ng}, 0, v{e}")
    for i in range(32):
        A(f"  v_ldexp_f32 v{X + i}, v{X + i}, v{ng}")
    for k in range(8):
        A(f"  v_cvt_pk_fp8_f32 v{T + 8 + k}, v{X + 4 * k}, v{X + 4 * k + 1}")
        A(
            f"  v_cvt_pk_fp8_f32 v{T + 8 + k}, v{X + 4 * k + 2}, v{X + 4 * k + 3} op_sel:[0,0,1]"
        )
    # addresses: j = t / 16, r = t % 16, kg = l / 4, h = (l % 4) / 2, q' = 2 (l % 2)
    A("  s_lshr_b32 s32, s27, 4")
    A(f"  s_mul_i32 s32, s32, {KGQ}")  # j·KGQ
    A("  s_and_b32 s33, s27, 15")  # r
    A(f"  v_lshrrev_b32 v{T + 4}, 2, v{V_L}")  # kg
    A(f"  v_add_u32 v{T + 4}, s32, v{T + 4}")  # j·KGQ + kg
    A(f"  v_lshlrev_b32 v{T + 5}, 11, v{T + 4}")  # block · 2 KB
    A(f"  v_bfe_u32 v{T + 6}, v{V_L}, 1, 1")  # h
    A(f"  v_lshl_add_u32 v{T + 5}, v{T + 6}, 10, v{T + 5}")
    A(f"  v_and_b32 v{T + 6}, 1, v{V_L}")
    A(f"  v_lshl_add_u32 v{T + 5}, v{T + 6}, 9, v{T + 5}")  # q' = 2 (l % 2): + 512
    A("  s_lshl_b32 s34, s33, 4")
    A(f"  v_add_u32 v{T + 5}, s34, v{T + 5}")  # + 16 r
    A(f"  global_store_dwordx4 v{T + 5}, v[{T + 8}:{T + 11}], s[18:19]")
    A(f"  global_store_dwordx4 v{T + 5}, v[{T + 12}:{T + 15}], s[18:19] offset:256")
    A(f"  v_lshlrev_b32 v{T + 5}, 6, v{T + 4}")  # scales: (j·KGQ + kg)·64
    A(f"  v_and_b32 v{T + 6}, 3, v{V_L}")
    A(f"  v_lshl_add_u32 v{T + 5}, v{T + 6}, 4, v{T + 5}")
    A(f"  v_add_u32 v{T + 5}, s33, v{T + 5}")
    A(f"  v_add_u32 v{e}, 0x7f, v{e}")
    A(f"  global_store_byte v{T + 5}, v{e}, s[20:21]")
    A("  s_mov_b64 exec, s[30:31]")
    A(".Lkv_q_done:")
    # ---- kv lanes 40..55: RoPE (lanes 54, 55), record, insert
    A(f"  v_subrev_u32 v{T + 4}, 40, v{V_L}")  # kd
    A(f"  v_cmp_gt_u32 vcc, 16, v{T + 4}")
    A("  s_and_saveexec_b64 s[30:31], vcc")
    A(f"  v_cmp_le_u32 vcc, 14, v{T + 4}")  # RoPE lanes
    A("  s_and_saveexec_b64 s[40:41], vcc")
    for p in range(16):  # e, o → v[64 + 2p], e·cos, o·cos in place
        ev, od, cs_ = X + 2 * p, X + 2 * p + 1, CS + p
        A(f"  v_mov_b32 v{64 + 2 * p}, v{ev}")
        A(f"  v_mov_b32 v{65 + 2 * p}, v{od}")
        A(f"  v_mul_f32 v{ev}, v{ev}, v{cs_}")
        A(f"  v_mul_f32 v{od}, v{od}, v{cs_}")
    for p in range(16):
        ev, od, sn = X + 2 * p, X + 2 * p + 1, SN + p
        e0, o0 = 64 + 2 * p, 65 + 2 * p
        A(f"  v_mul_f32 v{T + 1}, v{o0}, v{sn}")  # o sin
        A(f"  v_sub_f32 v{ev}, v{ev}, v{T + 1}")  # e cos − o sin
        A(f"  v_mul_f32 v{T + 1}, v{e0}, v{sn}")  # e sin
        A(f"  v_add_f32 v{od}, v{T + 1}, v{od}")  # e sin + o cos
    for k in range(16):
        A(f"  v_cvt_pk_bf16_f32 v{T + 8 + k}, v{X + 2 * k}, v{X + 2 * k + 1}")
    A("  s_mov_b64 exec, s[40:41]")
    # record addresses: block = slot >> bsl, row = slot & (bs − 1)
    A("  s_lshr_b64 s[42:43], s[38:39], s25")
    A("  s_mul_hi_u32 s43, s42, s24")
    A("  s_mul_i32 s42, s42, s24")  # block · stride (64-bit)
    A("  s_add_u32 s44, s16, s42")
    A("  s_addc_u32 s45, s17, s43")
    A("  s_lshl_b32 s46, 1, s25")
    A("  s_sub_u32 s47, s46, 1")
    A("  s_and_b32 s47, s38, s47")  # row
    A("  s_mul_i32 s48, s47, 576")  # record offset
    A("  s_mul_i32 s49, s46, 576")
    A("  s_lshl_b32 s47, s47, 3")
    A("  s_add_u32 s49, s49, s47")  # scale offset bs·576 + 8 row
    A(f"  v_cmp_le_u32 vcc, 14, v{T + 4}")
    A("  s_and_saveexec_b64 s[40:41], vcc")
    # RoPE lanes: 64 B of bf16 at 448 + 64 (kd − 14)
    A(f"  v_subrev_u32 v{T + 5}, 14, v{T + 4}")
    A(f"  v_lshlrev_b32 v{T + 5}, 6, v{T + 5}")
    A(f"  v_add_u32 v{T + 5}, s48, v{T + 5}")
    for k in range(4):
        A(
            f"  global_store_dwordx4 v{T + 5}, v[{T + 8 + 4 * k}:{T + 11 + 4 * k}], s[44:45] offset:{NOPE + 16 * k}"
        )
    A("  s_andn2_b64 exec, s[40:41], exec")  # NoPE lanes (kd < 14)
    # 64-dim group = lanes (2g, 2g + 1): absmax over the pair
    a = T
    A(f"  v_max_f32 v{a}, |v{X}|, |v{X + 1}|")
    for i in range(2, 32):
        A(f"  v_max_f32 v{a}, v{a}, |v{X + i}|")
    A(f"  v_mov_b32 v{T + 1}, v{a}")
    nop(2)
    A(
        f"  v_max_f32_dpp v{a}, v{T + 1}, v{a} quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
    )
    A(f"  v_max_f32 v{a}, {f32(1e-4)}, v{a}")
    A(f"  v_mov_b32 v{T + 1}, {f32(448.0)}")
    ieee_div(T + 2, a, T + 1, T + 5)  # absmax / 448
    A(f"  v_log_f32 v{T + 2}, v{T + 2}")
    A("  s_nop 0")
    A(f"  v_ceil_f32 v{T + 2}, v{T + 2}")  # exponent
    A(f"  v_cvt_i32_f32 v{T + 3}, v{T + 2}")
    A(f"  v_sub_u32 v{T + 1}, 0, v{T + 3}")
    A(f"  v_mov_b32 v{T + 6}, {f32(-448.0)}")
    A(f"  v_mov_b32 v{T + 7}, {f32(448.0)}")
    for i in range(32):
        A(f"  v_ldexp_f32 v{X + i}, v{X + i}, v{T + 1}")
        A(f"  v_med3_f32 v{X + i}, v{X + i}, v{T + 6}, v{T + 7}")
    for k in range(8):
        A(f"  v_cvt_pk_fp8_f32 v{T + 8 + k}, v{X + 4 * k}, v{X + 4 * k + 1}")
        A(
            f"  v_cvt_pk_fp8_f32 v{T + 8 + k}, v{X + 4 * k + 2}, v{X + 4 * k + 3} op_sel:[0,0,1]"
        )
    A(f"  v_lshlrev_b32 v{T + 5}, 5, v{T + 4}")
    A(f"  v_add_u32 v{T + 5}, s48, v{T + 5}")
    A(f"  global_store_dwordx4 v{T + 5}, v[{T + 8}:{T + 11}], s[44:45]")
    A(f"  global_store_dwordx4 v{T + 5}, v[{T + 12}:{T + 15}], s[44:45] offset:16")
    # scale byte (even lanes) at bs·576 + 8 row + kd / 2; lane kd 0 also writes byte 7 = 0
    A(f"  v_add_u32 v{T + 3}, 0x7f, v{T + 3}")
    A(f"  v_mov_b32 v{T + 6}, 0xff")
    A(f"  v_med3_i32 v{T + 3}, v{T + 3}, 0, v{T + 6}")
    A(f"  v_and_b32 v{T + 6}, 1, v{T + 4}")
    A(f"  v_cmp_eq_u32 vcc, 0, v{T + 6}")
    A("  s_and_saveexec_b64 s[40:41], vcc")
    A(f"  v_lshrrev_b32 v{T + 5}, 1, v{T + 4}")
    A(f"  v_add_u32 v{T + 5}, s49, v{T + 5}")
    A(f"  global_store_byte v{T + 5}, v{T + 3}, s[44:45]")
    A(f"  v_cmp_eq_u32 vcc, 0, v{T + 4}")
    A("  s_and_b64 exec, exec, vcc")
    A(f"  v_mov_b32 v{T + 6}, 0")
    A(f"  v_mov_b32 v{T + 5}, s49")
    A(f"  global_store_byte v{T + 5}, v{T + 6}, s[44:45] offset:7")
    A("  s_barrier")  # the host's last one
    A(f"{EXIT}:")
    A("  s_endpgm")
    return c
