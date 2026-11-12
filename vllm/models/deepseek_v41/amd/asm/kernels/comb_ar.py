# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""K2's MoE combine with all-reduce #2 folded in, in kernels/wob_ar.py's shape
: one launch from the weighted expert rows to the all-reduced
FFN output, replacing a combine launch + vLLM's all-reduce.

part = bf16(bf16(Σ_k y[t·6 + k]) + add[t]) (y bf16, widened to f32 and summed in
pick order; picks with ids < 0 read 0) is split into 4 flat quarters as in ar2
(n = M·160 granules of 16 B each); rank q owns quarter q. Grid ceil(n / 64)
CTAs of 4 waves; wave w of CTA c takes granule x = 64 c + lane of quarter w:
- every wave computes its granule of part; a non-owner wave (w ≠ R) pushes it
  (16 B) to rank w's `land` slot R and flags hop 0, then waits for rank w's
  hop-1 flag and copies its `land2` slot w to `out`;
- the owner wave (w = R) keeps its granule, waits for the 3 hop-0 flags, sums
  the 4 partials in fp32 in rank order, rounds to bf16 (RNE), pushes the
  result to the peers' land2 slot R, writes its own `out` granule and flags
  hop 1.
So `out` = bf16(((p0 + p1) + p2) + p3), vLLM's custom all-reduce's sum.

Buffers (IPC, this rank's address space): land[d] / land2[d] as ar2's
(`[4 slots][QMAX]`, granule x at 16·x), flags[d]:
`[2 hops][4 sources][NC CTAs][4 waves]` dwords, ctr [NC·4], err [NC·4]. Seq per
(CTA, wave) as in ar2 / wob_ar; every rank launches the same grids, so a
CTA's counters move together on every rank even though the grid follows M.
"""

from vllm.models.deepseek_v41.amd.asm.kernels.ar2 import POLLS, QMAX
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source, rsrc

H, MAX_M = 5120, 256
NC = MAX_M * 160 // 64  # CTAs at M 256
NW = NC * 4
CTR_OFF = 2 * 4 * NW * 4
ERR_OFF = CTR_OFF + NW * 4
FLAG_BYTES = ERR_OFF + NW * 4

KA = KArgs(
    *[(f"land{d}", "ptr") for d in range(4)],
    *[(f"land2_{d}", "ptr") for d in range(4)],
    *[(f"flags{d}", "ptr") for d in range(4)],
    ("out", "ptr"),
    ("y", "ptr"),
    ("ids", "ptr"),
    ("add", "ptr"),
    ("m", "u32"),
    ("y_bytes", "u32"),
    ("ids_bytes", "u32"),
    ("add_bytes", "u32"),
)


def grid(m: int) -> int:
    return -(-m * 160 // 64)


def source(R: int, name: str | None = None):
    """Rank R's kernel; y bf16 [M·6, 5120] (moe_down1 y_bf16), one dwordx4 per pick."""
    assert 0 <= R < 4
    name = name or f"comb_ar_bf16_r{R}_yb"
    peers = [d for d in range(4) if d != R]
    vA, vF = 2, 5  # 16 x, 16 f (flat byte offset in part / out)
    I, Y, AD, B = 8, 16, 64, 68  # noqa: E741 ids 6, y 6 × 8, add 4, own granule (packed bf16) 4
    F, T = 72, 80  # 8 f32, temps 12
    L = 92  # received: 3 × 4
    SM = 104  # sum (8 f32)
    FV = 120  # flag offset, value, 3 poll offsets, 3 polled
    LAND, LAND2, FLAGS = 20, 28, 36  # SGPR pairs (+2 d)
    c = []
    A = c.append

    def pair(base, d):
        return f"s[{base + 2 * d}:{base + 2 * d + 1}]"

    def unpack(src, f):
        for k in range(4):
            A(f"  v_lshlrev_b32 v{f + 2 * k}, 16, v{src + k}")
            A(f"  v_and_b32 v{f + 2 * k + 1}, 0xffff0000, v{src + k}")

    def pack(f, dst):
        for k in range(4):
            A(f"  v_cvt_pk_bf16_f32 v{dst + k}, v{f + 2 * k}, v{f + 2 * k + 1}")

    def poll(offs, tag):
        """Wait until the dwords at flags[R] + the SGPR offsets all equal seq (sc1 polls)."""
        A("  s_mov_b64 exec, 1")
        for k, so in enumerate(offs):
            A(f"  v_mov_b32 v{FV + 2 + k}, {so}")
        A(f"  s_mov_b32 s19, {POLLS & 0xFFFFFFFF}")
        A(f".Lw{tag}:")
        A("  s_sub_u32 s19, s19, 1")
        A("  s_cbranch_scc1 .Ltimeout")
        for k in range(len(offs)):
            A(f"  global_load_dword v{FV + 5 + k}, v{FV + 2 + k}, {pair(FLAGS, R)} sc1")
        A("  s_waitcnt vmcnt(0)")
        for k in range(len(offs)):
            A(f"  v_readfirstlane_b32 s{88 + k}, v{FV + 5 + k}")
        for k in range(len(offs)):
            A(f"  s_cmp_eq_u32 s{88 + k}, s13")
            A(f"  s_cbranch_scc0 .Lw{tag}")
        A("  s_mov_b64 exec, s[8:9]")

    def flag_to(d_pair, off_sgpr):
        A("  s_mov_b64 exec, 1")
        A(f"  v_mov_b32 v{FV}, {off_sgpr}")
        A(f"  v_mov_b32 v{FV + 1}, s13")
        A(f"  global_store_dword v{FV}, v{FV + 1}, {d_pair} sc0 sc1")
        A("  s_mov_b64 exec, s[8:9]")

    # ---- prologue
    A("  s_load_dwordx16 s[20:35], s[0:1], 0")  # land 0..3, land2 0..3
    A("  s_load_dwordx8 s[36:43], s[0:1], 64")  # flags 0..3
    A(f"  s_load_dwordx8 s[44:51], s[0:1], {KA.off['out']:#x}")  # out y ids add
    A(
        f"  s_load_dwordx4 s[52:55], s[0:1], {KA.off['m']:#x}"
    )  # m, y_bytes, ids_bytes, add_bytes
    A("  v_readfirstlane_b32 s74, v0")
    A("  s_lshr_b32 s74, s74, 6")  # wave = quarter
    A("  v_and_b32 v1, 63, v0")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_mul_i32 s7, s52, 160")  # n
    A("  s_lshl_b32 s14, s2, 6")
    A(f"  v_add_u32 v{vA}, s14, v1")  # x
    A(f"  v_cmp_gt_u32 s[8:9], s7, v{vA}")  # valid lanes
    A("  s_mul_i32 s15, s7, s74")
    A(f"  v_add_u32 v{vF}, s15, v{vA}")  # flat granule f
    A(f"  v_lshlrev_b32 v{vA}, 4, v{vA}")
    A("  s_lshl_b32 s16, s2, 2")
    A("  s_add_u32 s16, s16, s74")
    A("  s_lshl_b32 s16, s16, 2")  # (c·4 + w)·4
    A(f"  s_add_u32 s17, s16, {CTR_OFF}")
    A(f"  s_load_dword s13, {pair(FLAGS, R)}, s17")
    for line in (
        rsrc(56, 46, nbytes_sgpr=53)
        + rsrc(60, 48, nbytes_sgpr=54)
        + rsrc(64, 50, nbytes_sgpr=55)
    ):
        A(line)
    A("  s_mov_b32 s68, 0xcccccccd")
    A("  s_mov_b64 exec, s[8:9]")

    # ---- this wave's granule of part
    A(f"  v_lshrrev_b32 v{T}, 7, v{vF}")
    A(f"  v_mul_hi_u32 v{T}, v{T}, s68")
    A(f"  v_lshrrev_b32 v{T}, 2, v{T}")  # token = f / 640
    A(f"  v_mul_u32_u24 v{T + 1}, {H // 8}, v{T}")
    A(f"  v_sub_u32 v{T + 1}, v{vF}, v{T + 1}")  # chunk
    A(f"  v_mul_u32_u24 v{T + 2}, 24, v{T}")
    for k in range(6):
        A(f"  buffer_load_dword v{I + k}, v{T + 2}, s[60:63], 0 offen offset:{4 * k}")
    A(f"  v_mul_u32_u24 v{T + 3}, {H * 2}, v{T}")
    A(f"  v_lshl_add_u32 v{T + 3}, v{T + 1}, 4, v{T + 3}")
    A(f"  buffer_load_dwordx4 v[{AD}:{AD + 3}], v{T + 3}, s[64:67], 0 offen")
    A(f"  v_mul_u32_u24 v{T + 2}, {6 * H * 2}, v{T}")
    A(f"  v_lshl_add_u32 v{T + 2}, v{T + 1}, 4, v{T + 2}")
    A(f"  v_mov_b32 v{T + 10}, 0x7fff0000")
    A(f"  v_lshlrev_b32 v{vF}, 4, v{vF}")  # flat byte offset (out)
    A("  s_waitcnt vmcnt(1)")  # the ids (add may still be in flight)
    for k in range(6):
        A(f"  v_add_u32 v{T + 3 + k}, {k * H * 2}, v{T + 2}")
        A(f"  v_cmp_le_i32 vcc, 0, v{I + k}")
        A(f"  v_cndmask_b32 v{T + 3 + k}, v{T + 10}, v{T + 3 + k}, vcc")
    for k in range(6):
        A(
            f"  buffer_load_dwordx4 v[{Y + 8 * k + 4}:{Y + 8 * k + 7}], v{T + 3 + k}, s[56:59], 0 offen"
        )
    A("  s_waitcnt vmcnt(0)")
    for k in range(6):
        unpack(Y + 8 * k + 4, Y + 8 * k)
    for k in range(1, 6):
        for i in range(8):
            A(f"  v_add_f32 v{Y + i}, v{Y + i}, v{Y + 8 * k + i}")
    for w in range(4):
        A(f"  v_cvt_pk_bf16_f32 v{B + w}, v{Y + 2 * w}, v{Y + 2 * w + 1}")
        A(f"  v_lshlrev_b32 v{T}, 16, v{B + w}")
        A(f"  v_and_b32 v{T + 1}, 0xffff0000, v{B + w}")
        A(f"  v_lshlrev_b32 v{T + 2}, 16, v{AD + w}")
        A(f"  v_and_b32 v{T + 3}, 0xffff0000, v{AD + w}")
        A(f"  v_add_f32 v{T}, v{T}, v{T + 2}")
        A(f"  v_add_f32 v{T + 1}, v{T + 1}, v{T + 3}")
        A(f"  v_cvt_pk_bf16_f32 v{B + w}, v{T}, v{T + 1}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_add_u32 s13, s13, 1")  # this launch's seq
    A(f"  s_cmp_eq_u32 s74, {R}")
    A("  s_cbranch_scc1 .Lowner")

    # ---- non-owner wave w: push to land[w] slot R, hop-0 flag, wait for hop 1, land2[R] slot w → out
    A(f"  s_mov_b64 s[70:71], {pair(LAND, 0)}")  # land[w], flags[w]
    A(f"  s_mov_b64 s[72:73], {pair(FLAGS, 0)}")
    for d in (1, 2, 3):
        A(f"  s_cmp_eq_u32 s74, {d}")
        A(f"  s_cselect_b64 s[70:71], {pair(LAND, d)}, s[70:71]")
        A(f"  s_cselect_b64 s[72:73], {pair(FLAGS, d)}, s[72:73]")
    A(f"  s_add_u32 s70, s70, {R * QMAX}")
    A("  s_addc_u32 s71, s71, 0")
    A(f"  global_store_dwordx4 v{vA}, v[{B}:{B + 3}], s[70:71] sc1")
    A("  s_waitcnt vmcnt(0)")
    A(f"  s_add_u32 s17, s16, {(0 * 4 + R) * NW * 4}")
    flag_to("s[72:73]", "s17")
    A(f"  s_mul_i32 s18, s74, {NW * 4}")  # hop 1, source w
    A(f"  s_add_u32 s18, s18, {4 * NW * 4}")
    A("  s_add_u32 s18, s18, s16")
    poll(["s18"], "ag")
    A(f"  s_mul_i32 s17, s74, {QMAX}")  # land2[R] slot w
    A(f"  s_add_u32 s70, s{LAND2 + 2 * R}, s17")
    A(f"  s_addc_u32 s71, s{LAND2 + 2 * R + 1}, 0")
    A(f"  global_load_dwordx4 v[{L}:{L + 3}], v{vA}, s[70:71] sc1")
    A("  s_waitcnt vmcnt(0)")
    A(f"  global_store_dwordx4 v{vF}, v[{L}:{L + 3}], s[44:45]")
    A("  s_branch .Lbump")

    # ---- owner wave R
    A(".Lowner:")
    for k, s_ in enumerate(peers):
        A(f"  s_add_u32 s{80 + k}, s16, {(0 * 4 + s_) * NW * 4}")
    poll([f"s{80 + k}" for k in range(3)], "rs")
    for k, s_ in enumerate(peers):  # land[R] slot s
        A(f"  s_add_u32 s70, s{LAND + 2 * R}, {s_ * QMAX}")
        A(f"  s_addc_u32 s71, s{LAND + 2 * R + 1}, 0")
        r = L + 4 * k
        A(f"  global_load_dwordx4 v[{r}:{r + 3}], v{vA}, s[70:71] sc1")
    A("  s_waitcnt vmcnt(0)")
    for r in range(4):
        dst = SM if r == 0 else F
        unpack(B if r == R else L + 4 * peers.index(r), dst)
        if r:
            for e in range(8):
                A(f"  v_add_f32 v{SM + e}, v{SM + e}, v{F + e}")
    pack(SM, B)
    for d in peers:  # land2[d] slot R
        A(f"  s_add_u32 s70, s{LAND2 + 2 * d}, {R * QMAX}")
        A(f"  s_addc_u32 s71, s{LAND2 + 2 * d + 1}, 0")
        A(f"  global_store_dwordx4 v{vA}, v[{B}:{B + 3}], s[70:71] sc1")
    A(f"  global_store_dwordx4 v{vF}, v[{B}:{B + 3}], s[44:45]")
    A("  s_waitcnt vmcnt(0)")
    A(f"  s_add_u32 s17, s16, {(1 * 4 + R) * NW * 4}")
    for d in peers:
        flag_to(pair(FLAGS, d), "s17")
    A("  s_branch .Lbump")

    A(".Ltimeout:")
    A("  s_mov_b64 exec, 1")
    A(f"  s_add_u32 s17, s16, {ERR_OFF}")
    A(f"  v_mov_b32 v{FV}, s17")
    A(f"  v_mov_b32 v{FV + 1}, s13")
    A(f"  global_store_dword v{FV}, v{FV + 1}, {pair(FLAGS, R)} sc1")
    A(".Lbump:")
    A("  s_mov_b64 exec, 1")
    A(f"  s_add_u32 s17, s16, {CTR_OFF}")
    A(f"  v_mov_b32 v{FV}, s17")
    A(f"  v_mov_b32 v{FV + 1}, s13")
    A(f"  global_store_dword v{FV}, v{FV + 1}, {pair(FLAGS, R)}")
    A("  s_endpgm")
    src = kernel_source(name, c, KA, lds=0, vgprs=FV + 8, agprs=0, sgprs=96, block=256)
    return dict(name=name, src=src, ka=KA)
