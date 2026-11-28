# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""Routed-MoE prep on the GPU (replaces the host-side torch prep):

- CTA 0: counting sort of the M*6 picks by expert into the group table
  (gtable.py): LDS atomic adds give each pick its slot within its expert, a
  64-lane scan turns per-expert group counts (ceil(cnt / 16)) into group
  starts, then picks scatter to groups[start + slot // 16].picks[slot % 16];
  slot 0 writes the header (e, 0, 0, 0), the expert's last pick pads its group
  with -1. counts[0] = number of groups. Pick order within an expert follows
  the atomics; no result depends on it (each pick's h and y are computed on
  their own).
- CTAs 1..: x bf16 → MXFP8 (E8M0 2^ceil(log2(amax/448)), RNE), one 32-element
  block per thread; xq [M, 5120] codes, xs `[M][kb 4][wave 4][12]` scale bytes
  (moe_ug1.py's layout).
- the sort CTA also splits stage 2's units
  (20 per group) over the 256 CTAs by cost instead of count: a unit of group g
  costs DS_A + DS_B·picks_g (the y stores make full groups ~18% dearer). counts[4 + b] = first unit of CTA b, counts[260] =
  20·NG; moe_down1 source(split="table") reads them. counts holds 264 int32
  (COUNTS_DSPLIT).
"""

from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source, rsrc

H = 5120
E = 384
KA = KArgs(
    ("x", "ptr"),
    ("ids", "ptr"),
    ("xq", "ptr"),
    ("xs", "ptr"),
    ("groups", "ptr"),
    ("counts", "ptr"),
    ("m", "u32"),
    ("x_bytes", "u32"),
    ("ids_bytes", "u32"),
    ("xq_bytes", "u32"),
    ("xs_bytes", "u32"),
    ("groups_bytes", "u32"),
)


DS_A, DS_B = 88, 1  # stage 2 unit cost: DS_A + DS_B·picks (fit: +1.14% per pick)
DS_UPG = 20
NG_MAX = 480  # 384 experts + 1536 / 16
COUNTS_DSPLIT = 264  # int32: NG, 3 spare, 257 unit starts, pad


def grid(m: int) -> int:
    return 1 + (m * (H // 32) + 255) // 256


def source(name: str = "moe_prep"):
    c = []
    A = c.append
    A(
        "  s_load_dwordx16 s[4:19], s[0:1], 0x0"
    )  # x, ids, xq, xs, groups, counts, m s16, x_bytes s17, ids_bytes s18, xq_bytes s19
    A("  s_load_dwordx2 s[20:21], s[0:1], 0x40")  # xs_bytes, groups_bytes
    # wave id from v0 directly: v_readfirstlane right after a VALU write of its
    # source returns the stale value (1 wait state needed), which made wave 0
    # skip the scan whenever the register held a previous kernel's data
    A("  v_readfirstlane_b32 s22, v0")
    A("  s_lshr_b32 s22, s22, 6")  # wave
    A("  v_and_b32 v1, 63, v0")  # lane
    A("  s_waitcnt lgkmcnt(0)")
    c += rsrc(24, 4, nbytes_sgpr=17)  # x
    c += rsrc(28, 6, nbytes_sgpr=18)  # ids
    c += rsrc(32, 8, nbytes_sgpr=19)  # xq
    c += rsrc(36, 10, nbytes_sgpr=20)  # xs
    c += rsrc(40, 12, nbytes_sgpr=21)  # groups
    c += rsrc(44, 14, nbytes=4 * COUNTS_DSPLIT)  # counts
    A("  s_cmp_eq_u32 s2, 0")
    A("  s_cbranch_scc1 .Lsort")

    # ---------------- quant: thread → block b = (wg - 1) * 256 + tid
    A("  s_sub_u32 s48, s2, 1")
    A("  s_lshl_b32 s48, s48, 8")
    A("  v_add_u32 v2, s48, v0")  # b
    A(f"  s_mul_i32 s49, s16, {H // 32}")
    A("  v_cmp_gt_u32 vcc, s49, v2")
    A("  s_and_saveexec_b64 s[50:51], vcc")
    A("  s_cbranch_execz .Lend")
    A("  s_mov_b32 s52, 0xcccccccd")
    A("  v_lshrrev_b32 v3, 5, v2")
    A("  v_mul_hi_u32 v3, v3, s52")
    A("  v_lshrrev_b32 v3, 2, v3")  # t = b / 160
    A(f"  v_mul_u32_u24 v4, {H // 32}, v3")
    A("  v_sub_u32 v4, v2, v4")  # kblk
    A(
        "  v_lshlrev_b32 v5, 6, v2"
    )  # x offset = b * 64 (bf16 rows are contiguous blocks)
    for q in range(4):
        A(
            f"  buffer_load_dwordx4 v[{8 + 4 * q}:{11 + 4 * q}], v5, s[24:27], 0 offen offset:{16 * q}"
        )
    A("  s_waitcnt vmcnt(0)")
    # bf16 pairs → f32: lo = d << 16, hi = d & 0xffff0000
    for i in range(16):
        A(f"  v_lshlrev_b32 v{24 + 2 * i}, 16, v{8 + i}")
        A(f"  v_and_b32 v{25 + 2 * i}, 0xffff0000, v{8 + i}")
    A("  v_max_f32 v60, |v24|, |v25|")
    for i in range(26, 56):
        A(f"  v_max_f32 v60, v60, |v{i}|")
    A("  v_frexp_exp_i32_f32 v61, v60")
    A("  v_frexp_mant_f32 v62, v60")
    A("  v_cmp_lt_f32 vcc, 0x3f600000, v62")
    A("  v_addc_co_u32 v61, vcc, -9, v61, vcc")
    A("  v_max_i32 v61, 0xffffff81, v61")
    A("  v_min_i32 v61, 0x7f, v61")
    A("  v_sub_u32 v62, 0, v61")
    for i in range(32):
        A(f"  v_ldexp_f32 v{24 + i}, v{24 + i}, v62")
    for q in range(8):
        A(f"  v_cvt_pk_fp8_f32 v{64 + q}, v{24 + 4 * q}, v{25 + 4 * q}")
        A(f"  v_cvt_pk_fp8_f32 v{64 + q}, v{26 + 4 * q}, v{27 + 4 * q} op_sel:[0,0,1]")
    A("  v_lshlrev_b32 v6, 5, v2")  # xq offset = b * 32
    A("  buffer_store_dwordx4 v[64:67], v6, s[32:35], 0 offen")
    A("  buffer_store_dwordx4 v[68:71], v6, s[32:35], 0 offen offset:16")
    # scale byte: k0 = kblk / 4, kb = kblk % 4, w = k0 / 10, kk = k0 % 10 → t*192 + kb*48 + w*12 + kk
    A("  v_lshrrev_b32 v7, 2, v4")  # k0
    A("  v_and_b32 v72, 3, v4")  # kb
    A("  v_mul_hi_u32 v73, v7, s52")
    A("  v_lshrrev_b32 v73, 3, v73")  # w
    A("  v_mul_u32_u24 v74, 10, v73")
    A("  v_sub_u32 v74, v7, v74")  # kk
    A("  v_mul_u32_u24 v75, 192, v3")
    A("  v_mul_u32_u24 v76, 48, v72")
    A("  v_add_u32 v75, v75, v76")
    A("  v_mul_u32_u24 v76, 12, v73")
    A("  v_add_u32 v75, v75, v76")
    A("  v_add_u32 v75, v75, v74")
    A("  v_add_u32 v77, 0x7f, v61")
    A("  buffer_store_byte v77, v75, s[36:39], 0 offen")
    A("  s_branch .Lend")

    # ---------------- sort (CTA 0). LDS: cnt[384] @0, gstart[384] @1536
    A(".Lsort:")
    A("  v_mov_b32 v2, 0")
    A("  v_lshlrev_b32 v3, 2, v0")
    A("  ds_write_b32 v3, v2")
    A("  v_cmp_gt_u32 vcc, 128, v0")
    A("  s_and_saveexec_b64 s[50:51], vcc")
    A("  ds_write_b32 v3, v2 offset:1024")
    A("  s_mov_b64 exec, s[50:51]")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    A("  s_mul_i32 s49, s16, 6")  # P
    # 6 picks per thread: p = tid + 256 i → e in v[10+i], slot in v[20+i] (e = -1 if none)
    A("  v_mov_b32 v4, 1")
    for i in range(6):
        A(f"  v_add_u32 v{30 + i}, {256 * i}, v0")  # p
        A(f"  v_lshlrev_b32 v5, 2, v{30 + i}")
        A(f"  v_mov_b32 v{10 + i}, -1")
        A(f"  v_cmp_gt_u32 vcc, s49, v{30 + i}")
        A("  s_and_saveexec_b64 s[50:51], vcc")
        A(f"  buffer_load_dword v{10 + i}, v5, s[28:31], 0 offen")
        A("  s_mov_b64 exec, s[50:51]")
    A("  s_waitcnt vmcnt(0)")
    for i in range(6):
        A(f"  v_cmp_le_i32 vcc, 0, v{10 + i}")
        A("  s_and_saveexec_b64 s[50:51], vcc")
        A(f"  v_lshlrev_b32 v5, 2, v{10 + i}")
        A(f"  ds_add_rtn_u32 v{20 + i}, v5, v4")
        A("  s_mov_b64 exec, s[50:51]")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    # wave 0: lane l sums groups of experts 6l..6l+5, inclusive scan over lanes, write gstart
    A("  s_cmp_eq_u32 s22, 0")
    A("  s_cbranch_scc0 .Lscanned")
    A("  v_mul_u32_u24 v40, 24, v1")  # LDS offset of expert 6l
    for k in range(6):
        A(f"  ds_read_b32 v{41 + k}, v40 offset:{4 * k}")
    A("  s_waitcnt lgkmcnt(0)")
    for k in range(6):
        A(f"  v_add_u32 v{41 + k}, 15, v{41 + k}")
        A(f"  v_lshrrev_b32 v{41 + k}, 4, v{41 + k}")  # groups of expert 6l+k
    A("  v_add_u32 v47, v41, v42")
    for k in range(2, 6):
        A(f"  v_add_u32 v47, v47, v{41 + k}")  # lane total
    A("  v_mov_b32 v48, v47")  # inclusive scan
    for d in (1, 2, 4, 8, 16, 32):
        A(f"  v_subrev_u32 v49, {d}, v1")
        A("  v_lshlrev_b32 v49, 2, v49")
        A("  ds_bpermute_b32 v50, v49, v48")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_cmp_le_u32 vcc, {d}, v1")
        A("  v_cndmask_b32 v50, 0, v50, vcc")
        A("  v_add_u32 v48, v48, v50")
    A("  v_sub_u32 v48, v48, v47")  # exclusive start of lane's experts
    for k in range(6):
        A(f"  ds_write_b32 v40, v48 offset:{1536 + 4 * k}")
        A(f"  v_add_u32 v48, v48, v{41 + k}")
    # lane 63 holds the total: counts[0]
    A("  s_nop 1")  # VALU write → v_readlane: 1 wait state
    A("  v_readlane_b32 s53, v48, 63")
    A("  v_mov_b32 v51, s53")
    A("  v_mov_b32 v52, 0")
    A("  s_mov_b64 exec, 1")
    A("  buffer_store_dword v51, v52, s[44:47], 0 offen sc0 sc1")
    A("  s_mov_b64 exec, -1")
    A("  s_waitcnt lgkmcnt(0)")
    A(".Lscanned:")
    A("  s_barrier")
    # scatter: g = gstart[e] + slot / 16; groups[g*80 + 16 + (slot % 16)*4] = p
    for i in range(6):
        e, sl, p = 10 + i, 20 + i, 30 + i
        A(f"  v_cmp_le_i32 vcc, 0, v{e}")
        A("  s_and_saveexec_b64 s[50:51], vcc")
        A(f"  s_cbranch_execz .Lsk{i}")
        A(f"  v_lshlrev_b32 v5, 2, v{e}")
        A("  ds_read_b32 v6, v5 offset:1536")  # gstart[e]
        A("  ds_read_b32 v7, v5")  # cnt[e]
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_lshrrev_b32 v8, 4, v{sl}")
        A("  v_add_u32 v6, v6, v8")  # g
        A("  v_mul_u32_u24 v8, 80, v6")  # record offset
        A(f"  v_and_b32 v9, 15, v{sl}")
        A("  v_lshl_add_u32 v53, v9, 2, v8")
        A(f"  buffer_store_dword v{p}, v53, s[40:43], 0 offen offset:16")
        # header by slot % 16 == 0
        A("  v_cmp_eq_u32 s[52:53], 0, v9")
        A("  s_and_saveexec_b64 s[54:55], s[52:53]")
        A(f"  v_mov_b32 v54, v{e}")
        A("  v_mov_b32 v55, 0")
        A("  v_mov_b32 v56, 0")
        A("  v_mov_b32 v57, 0")
        A("  buffer_store_dwordx4 v[54:57], v8, s[40:43], 0 offen")
        A("  s_mov_b64 exec, s[54:55]")
        # the expert's last pick pads its group: slots (slot%16)+1 .. 15 = -1
        A(f"  v_add_u32 v58, 1, v{sl}")
        A("  v_cmp_eq_u32 s[52:53], v58, v7")
        A("  s_and_saveexec_b64 s[54:55], s[52:53]")
        A("  v_mov_b32 v59, -1")
        for k in range(1, 16):
            A(f"  v_add_u32 v58, {k}, v9")
            A("  v_cmp_gt_u32 s[56:57], 16, v58")
            A("  s_and_saveexec_b64 s[58:59], s[56:57]")
            A(f"  buffer_store_dword v59, v53, s[40:43], 0 offen offset:{16 + 4 * k}")
            A("  s_mov_b64 exec, s[58:59]")
        A("  s_mov_b64 exec, s[54:55]")
        A(f".Lsk{i}:")
        A("  s_mov_b64 exec, s[50:51]")
    c += split_code()
    A(".Lend:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_endpgm")
    src = kernel_source(
        name,
        c,
        KA,
        lds=LDS_END,
        vgprs=80,
        agprs=0,
        sgprs=64,
        block=256,
    )
    return dict(name=name, src=src, ka=KA)


# LDS of the split (after cnt @0, gstart @1536)
GC, CP, WT = (
    3072,
    3072 + 4 * 512,
    3072 + 8 * 512 + 16,
)  # unit cost per group, exclusive cost prefix [513], wave totals
LDS_END = WT + 16


def split_code() -> list[str]:
    """Sort CTA, after the scatter (cnt / gstart in LDS): stage 2 unit starts
    by cost into counts[4 + b]. v1 = lane, v0 = tid, s22 = wave."""
    c = []
    A = c.append
    A("  s_waitcnt vmcnt(0) lgkmcnt(0)")
    # 1. unit cost of every group: expert e = tid (+ 256): groups gstart[e] + j, picks min(16, cnt - 16 j)
    for half in range(2):
        A(f"  v_add_u32 v2, {256 * half}, v0")  # e
        A("  s_mov_b64 s[60:61], exec")
        if half:
            A(f"  v_cmp_gt_u32 vcc, {384 - 256}, v0")
            A("  s_and_b64 exec, exec, vcc")
        A("  v_lshlrev_b32 v3, 2, v2")
        A("  ds_read_b32 v4, v3")  # cnt
        A("  ds_read_b32 v5, v3 offset:1536")  # gstart
        A("  s_waitcnt lgkmcnt(0)")
        A("  v_mov_b32 v6, 0")  # 16 j
        A("  v_lshlrev_b32 v7, 2, v5")  # GC[gstart] (LDS offset GC)
        A(f".Lgc{half}:")
        A("  v_cmp_gt_u32 vcc, v4, v6")
        A("  s_and_b64 exec, exec, vcc")
        A(f"  s_cbranch_execz .Lgc{half}_done")
        A("  v_sub_u32 v8, v4, v6")
        A("  v_min_u32 v8, 16, v8")  # picks
        assert DS_B == 1
        A(f"  v_add_u32 v8, {DS_A}, v8")
        A(f"  ds_write_b32 v7, v8 offset:{GC}")
        A("  v_add_u32 v6, 16, v6")
        A("  v_add_u32 v7, 4, v7")
        A(f"  s_branch .Lgc{half}")
        A(f".Lgc{half}_done:")
        A("  s_mov_b64 exec, s[60:61]")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    # NG = gstart[383] + ceil(cnt[383] / 16), read by every lane
    A("  v_mov_b32 v2, 0")
    A(f"  ds_read_b32 v3, v2 offset:{4 * 383}")
    A(f"  ds_read_b32 v4, v2 offset:{1536 + 4 * 383}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  v_add_u32 v3, 15, v3")
    A("  v_lshrrev_b32 v3, 4, v3")
    A("  v_add_u32 v3, v3, v4")
    A("  s_nop 1")
    A("  v_readfirstlane_b32 s56, v3")  # NG
    # 2. exclusive prefix of 20·GC: thread t owns groups 2t, 2t + 1
    A("  v_lshlrev_b32 v2, 1, v0")  # 2t
    A("  v_lshlrev_b32 v3, 2, v2")
    A(f"  ds_read_b64 v[4:5], v3 offset:{GC}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  v_cmp_gt_u32 vcc, s56, v2")
    A("  v_cndmask_b32 v4, 0, v4, vcc")
    A("  v_add_u32 v6, 1, v2")
    A("  v_cmp_gt_u32 vcc, s56, v6")
    A("  v_cndmask_b32 v5, 0, v5, vcc")
    A(f"  v_mul_u32_u24 v4, {DS_UPG}, v4")  # a
    A(f"  v_mul_u32_u24 v5, {DS_UPG}, v5")  # b
    A("  v_add_u32 v7, v4, v5")  # thread total
    A("  v_mov_b32 v8, v7")  # inclusive wave scan
    for d in (1, 2, 4, 8, 16, 32):
        A(f"  v_subrev_u32 v9, {d}, v1")
        A("  v_lshlrev_b32 v9, 2, v9")
        A("  ds_bpermute_b32 v10, v9, v8")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_cmp_le_u32 vcc, {d}, v1")
        A("  v_cndmask_b32 v10, 0, v10, vcc")
        A("  v_add_u32 v8, v8, v10")
    A("  s_nop 1")
    A("  v_readlane_b32 s57, v8, 63")  # wave total
    A("  v_mov_b32 v9, s57")
    A("  s_lshl_b32 s58, s22, 2")
    A("  v_mov_b32 v10, s58")
    A("  s_mov_b64 s[60:61], exec")
    A("  s_mov_b64 exec, 1")
    A(f"  ds_write_b32 v10, v9 offset:{WT}")
    A("  s_mov_b64 exec, s[60:61]")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    A("  v_mov_b32 v9, 0")
    A(f"  ds_read_b128 v[10:13], v9 offset:{WT}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  v_sub_u32 v8, v8, v7")  # exclusive in wave
    for w in range(3):  # + totals of earlier waves
        A(f"  s_cmp_gt_u32 s22, {w}")
        A("  s_cselect_b64 vcc, exec, 0")
        A(f"  v_cndmask_b32 v14, 0, v{10 + w}, vcc")
        A("  v_add_u32 v8, v8, v14")
    A("  v_add_u32 v15, v10, v11")
    A("  v_add_u32 v15, v15, v12")
    A("  v_add_u32 v15, v15, v13")  # T (every lane)
    A("  v_lshlrev_b32 v3, 2, v2")
    A(f"  v_add_u32 v3, {CP}, v3")
    A("  v_add_u32 v9, v8, v4")
    A("  ds_write2_b32 v3, v8, v9 offset1:1")  # CP[2t], CP[2t + 1]
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    # 3. CTA b = tid: target b·T / 256, g = last group with CP[g] <= target, u = 20 g + ceil((target - CP[g]) / GC[g])
    A("  v_mul_lo_u32 v16, v0, v15")
    A("  v_lshrrev_b32 v16, 8, v16")  # target
    A("  v_mov_b32 v17, 0")  # g
    for k in range(8, -1, -1):
        A(f"  v_add_u32 v18, {1 << k}, v17")
        A("  v_lshlrev_b32 v19, 2, v18")
        A(f"  ds_read_b32 v20, v19 offset:{CP}")
        A("  s_waitcnt lgkmcnt(0)")
        A("  v_cmp_gt_u32 s[60:61], s56, v18")
        A("  v_cmp_le_u32 s[62:63], v20, v16")
        A("  s_and_b64 vcc, s[60:61], s[62:63]")
        A("  v_cndmask_b32 v17, v17, v18, vcc")
    A("  v_lshlrev_b32 v19, 2, v17")
    A(f"  ds_read_b32 v20, v19 offset:{CP}")
    A(f"  ds_read_b32 v21, v19 offset:{GC}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  v_sub_u32 v20, v16, v20")  # r < 20 · cu
    A("  v_cvt_f32_u32 v22, v20")
    A("  v_cvt_f32_u32 v23, v21")
    A("  v_rcp_f32 v23, v23")
    A("  s_nop 1")
    A("  v_mul_f32 v22, v22, v23")
    A("  v_cvt_u32_f32 v22, v22")  # floor(r / cu) ± 1
    A("  v_mul_u32_u24 v23, v22, v21")
    A("  v_cmp_lt_u32 vcc, v23, v20")
    A("  v_addc_co_u32 v22, vcc, 0, v22, vcc")  # q = ceil(r / cu)
    A(f"  v_min_u32 v22, {DS_UPG}, v22")
    A(f"  v_mad_u32_u24 v22, v17, {DS_UPG}, v22")  # u_begin of CTA tid
    A("  v_lshlrev_b32 v23, 2, v0")
    A("  buffer_store_dword v22, v23, s[44:47], 0 offen offset:16")
    A("  v_cmp_eq_u32 vcc, 0, v0")
    A("  s_and_saveexec_b64 s[60:61], vcc")
    A(f"  s_mul_i32 s57, s56, {DS_UPG}")
    A("  v_mov_b32 v22, s57")
    A(f"  buffer_store_dword v22, v23, s[44:47], 0 offen offset:{16 + 4 * 256}")
    A("  s_mov_b64 exec, s[60:61]")
    return c
