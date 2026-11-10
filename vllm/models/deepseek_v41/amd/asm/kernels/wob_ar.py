# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""wo_b with all-reduce #1 folded in: gemv8's
WO_B1 (160 CTAs of 32 output columns over all K, mode "k") whose epilogue
pushes the reduce-scatter and whose tail finishes the all-reduce, so the
attention half ends in one launch with `out` = Σ_ranks wo_b(z_rank) on every
rank. Replaces K1's wo_b launch + the AR#1 launch (ours or vLLM's).

Ownership: in mode k, wave w of every CTA finishes token tiles w, w + 4, ..;
rank R owns the tiles of wave R (every CTA has one owner wave per rank: the
same work everywhere). Per tile (16 tokens × the CTA's 32 columns):
- epilogue (inside the main loop): D → bf16 (wo_b's partial, as WO_B1
  writes it). Non-owner waves push it to the owner's `land` slot R, 16 B per
  lane. The owner wave keeps its own tile: a plain store to its own land slot
  R (the same number of vm ops, so the next batch's x waits stay static:
  gemv8 P["epi_vm"]).
- tail (after the main loop, per wave): a non-owner flags the owner (hop 0),
  waits for the owner's hop-1 flag, reads the reduced tiles from its own
  `land2` slot w and writes them to `out`. The owner waits for the 3
  hop-0 flags, sums the 4 partials in fp32 in rank order, rounds to bf16
  (RNE), pushes the result to the peers' land2 slot R, writes its own `out`
  rows, then flags hop 1.
So `out` = bf16(((p0 + p1) + p2) + p3), vLLM's custom all-reduce's sum.

Buffers (IPC, this rank's address space): land[d] / land2[d]:
`[4 sources][160 CTAs][16 tiles][1 KB]`. flags[d]: `[2 hops][4 sources][160][4 waves]`
dwords, ctr [640], err [640] (per wave: launches so far; seq of the last
timed-out launch). Seq numbers as in ar2: graph replays need no host input,
every rank runs the same launches. Single-buffered, safe for the same reason
as ar2's: a rank's next push into a slot needs a flag that the slot's reader
only sends after reading it.
"""

from vllm.models.deepseek_v41.amd.asm.kernels import gemv8
from vllm.models.deepseek_v41.amd.asm.kernels.ar2 import POLLS
from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source

H, NCTA, MAXJ = 5120, 160, gemv8.MAXJ
CFG = gemv8.WO_B1
CH = 1024  # chunk bytes (one tile of one CTA, bf16)
SLOT = NCTA * MAXJ * CH  # one source's slot
LAND_BYTES = 4 * SLOT
NW = NCTA * 4  # waves
CTR_OFF = 2 * 4 * NW * 4
ERR_OFF = CTR_OFF + NW * 4
FLAG_BYTES = ERR_OFF + NW * 4

KA = KArgs(
    *gemv8.KA.args,
    *[(f"land{d}", "ptr") for d in range(4)],
    *[(f"land2_{d}", "ptr") for d in range(4)],
    *[(f"flags{d}", "ptr") for d in range(4)],
)
assert KA.off["land0"] == 0x60


def source(R: int, name: str | None = None):
    """Rank R's kernel."""
    assert 0 <= R < 4
    name = name or f"wob_ar_bf16_r{R}"
    peers = [d for d in range(4) if d != R]
    LAND, LAND2, FLAGS = 72, 80, 88  # SGPR pairs of land[d] (+2d), land2[d], flags[d]
    SEQ, TMP, TGT = 96, 97, 98  # s98:99: land[w] (epilogue), flags[w] (tail)
    c = []
    A = c.append

    def pair(base, d):
        return f"s[{base + 2 * d}:{base + 2 * d + 1}]"

    def by_wave(base):
        """s[TGT:TGT+1] = pair base + 2·w (w = s26)."""
        A(f"  s_mov_b64 s[{TGT}:{TGT + 1}], {pair(base, 0)}")
        for d in (1, 2, 3):
            A(f"  s_cmp_eq_u32 s26, {d}")
            A(
                f"  s_cselect_b64 s[{TGT}:{TGT + 1}], {pair(base, d)}, s[{TGT}:{TGT + 1}]"
            )

    def after_map(ctx):
        A(f"  s_lshl_b32 s{TMP}, s27, 2")
        A(f"  s_add_u32 s{TMP}, s{TMP}, s26")
        A(f"  s_lshl_b32 s{TMP}, s{TMP}, 2")
        A(f"  s_add_u32 s{TMP}, s{TMP}, {CTR_OFF}")
        A(f"  s_load_dword s{SEQ}, {pair(FLAGS, R)}, s{TMP}")
        by_wave(LAND)

    # ---- helpers on 8 f32 / packed values (lane: token t = l % 16, columns 4q + k of tiles 0 / 1)
    def unpack(src, f):
        """4 packed bf16 dwords v[src..] → 8 f32 v[f..] (exact)."""
        for k in range(4):
            A(f"  v_lshlrev_b32 v{f + 2 * k}, 16, v{src + k}")
            A(f"  v_and_b32 v{f + 2 * k + 1}, 0xffff0000, v{src + k}")

    def pack(f, dst):
        for k in range(4):
            A(f"  v_cvt_pk_bf16_f32 v{dst + k}, v{f + 2 * k}, v{f + 2 * k + 1}")

    def chunk_off(sreg, src, tile_sreg):
        """s[sreg] = byte offset of chunk (src, CTA s27, tile) in a land buffer."""
        A(f"  s_mul_i32 s{sreg}, s27, {MAXJ}")
        A(f"  s_add_u32 s{sreg}, s{sreg}, {src * NCTA * MAXJ}")
        A(f"  s_add_u32 s{sreg}, s{sreg}, {tile_sreg}")
        A(f"  s_lshl_b32 s{sreg}, s{sreg}, 10")

    def out_store(q4, tile_sreg, ctx, t):
        """4 packed bf16 dwords v[q4..] of tile s[tile_sreg] → out rows (t < m); t: 1 temp; exec saved/restored."""
        A("  s_mov_b64 s[46:47], exec")
        A(f"  s_lshl_b32 s41, {tile_sreg}, 4")
        A(f"  v_add_u32 v{t}, s41, v{ctx['V_R']}")  # token
        A(f"  v_cmp_gt_u32 vcc, s24, v{t}")
        A("  s_and_b64 exec, exec, vcc")
        A(f"  v_mul_u32_u24 v{t}, {H}, v{t}")
        A(f"  v_add_u32 v{t}, s32, v{t}")
        A(f"  v_lshl_add_u32 v{t}, v{ctx['V_Q']}, 2, v{t}")
        A(f"  v_lshlrev_b32 v{t}, 1, v{t}")
        A(f"  global_store_dwordx2 v{t}, v[{q4}:{q4 + 1}], s[12:13]")
        A(f"  global_store_dwordx2 v{t}, v[{q4 + 2}:{q4 + 3}], s[12:13] offset:32")
        A("  s_mov_b64 exec, s[46:47]")

    # ---- epilogue: tile s45's D (SUM, TI = 2 × 4 f32) → push / keep
    def epi(ctx):
        SUM, E = ctx["SUM"], ctx["V_E"]
        B = E  # 4 packed bf16
        for i in range(2):
            for k in (0, 2):
                A(
                    f"  v_cvt_pk_bf16_f32 v{B + 2 * i + k // 2}, v{SUM + 4 * i + k}, v{SUM + 4 * i + k + 1}"
                )
        chunk_off(40, R, "s45")
        A(f"  v_add_u32 v{E + 4}, s40, v{ctx['V_T']}")  # + 16 l
        own, done = ctx["label"]("own"), ctx["label"]("epid")
        A(f"  s_cmp_eq_u32 s26, {R}")
        A(f"  s_cbranch_scc1 {own}")
        A(f"  global_store_dwordx4 v{E + 4}, v[{B}:{B + 3}], s[{TGT}:{TGT + 1}] sc1")
        A(f"  s_branch {done}")
        A(f"{own}:")
        A(f"  global_store_dwordx4 v{E + 4}, v[{B}:{B + 3}], s[{TGT}:{TGT + 1}]")
        A(f"{done}:")

    c_hdr = []
    gemv8._header(c_hdr)
    c += c_hdr
    A(f"  s_load_dwordx16 s[{LAND}:{LAND2 + 7}], s[0:1], {KA.off['land0']:#x}")
    A(f"  s_load_dwordx8 s[{FLAGS}:{FLAGS + 7}], s[0:1], {KA.off['flags0']:#x}")
    P = dict(epi_fn=epi, epi_vm=1, after_map=after_map)
    r = gemv8._emit(c, CFG, "k", P)
    V0 = 48  # tail VGPRs (gemv8's x ring is free now)
    assert r["vgprs"] >= V0 + 120, r["vgprs"]

    # ---- tail
    ctx = dict(V_R=2, V_Q=3)
    FV = V0  # flag offset, value, 3 poll offsets, 3 polled
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_add_u32 s{SEQ}, s{SEQ}, 1")  # this launch's seq
    A("  s_cmp_ge_u32 s26, s29")
    A("  s_cbranch_scc1 .Lbump")  # no tiles for this wave
    A("  s_waitcnt vmcnt(0)")  # my pushes have landed
    A("  s_lshl_b32 s40, s27, 2")
    A("  s_add_u32 s40, s40, s26")
    A("  s_lshl_b32 s40, s40, 2")  # (b·4 + w)·4
    A(f"  v_mov_b32 v{FV + 1}, s{SEQ}")

    def poll(offsets, tag):
        """Wait until the dwords at flags[R] + s-reg offsets all equal seq (sc1 polls)."""
        for k, so in enumerate(offsets):
            A(f"  v_mov_b32 v{FV + 2 + k}, {so}")
        A(f"  s_mov_b32 s41, {POLLS & 0xFFFFFFFF}")
        A(f".Lw{tag}:")
        A("  s_sub_u32 s41, s41, 1")
        A("  s_cbranch_scc1 .Ltimeout")
        for k in range(len(offsets)):
            A(f"  global_load_dword v{FV + 5 + k}, v{FV + 2 + k}, {pair(FLAGS, R)} sc1")
        A("  s_waitcnt vmcnt(0)")
        for k in range(len(offsets)):
            A(f"  v_readfirstlane_b32 s{42 + k}, v{FV + 5 + k}")
        for k in range(len(offsets)):
            A(f"  s_cmp_eq_u32 s{42 + k}, s{SEQ}")
            A(f"  s_cbranch_scc0 .Lw{tag}")

    def tiles(fn):
        """fn(u) for this wave's tiles j = w + 4u < s29 (s45 = j); scalar-guarded."""
        end = f".Lt{len(c)}"
        for u in range(MAXJ // 4):
            A(f"  s_add_u32 s45, s26, {4 * u}")
            A("  s_cmp_ge_u32 s45, s29")
            A(f"  s_cbranch_scc1 {end}_{u}")
            fn(u)
            A(f"{end}_{u}:")

    A("  s_mov_b64 exec, -1")
    A(f"  s_cmp_eq_u32 s26, {R}")
    A("  s_cbranch_scc1 .Lowner")

    # ---- non-owner wave w: hop-0 flag to rank w, wait for its hop-1 flag, land2[R] slot w → out
    by_wave(FLAGS)
    A(f"  s_add_u32 s43, s40, {((0 * 4 + R) * NCTA * 4) * 4}")
    A(f"  v_mov_b32 v{FV}, s43")
    A("  s_mov_b64 exec, 1")
    A(f"  global_store_dword v{FV}, v{FV + 1}, s[{TGT}:{TGT + 1}] sc0 sc1")
    A("  s_mov_b64 exec, -1")
    A(f"  s_mul_i32 s43, s26, {NCTA * 4 * 4}")  # hop 1, source w: ((4 + w)·160·4)·4
    A(f"  s_add_u32 s43, s43, {4 * NCTA * 4 * 4}")
    A("  s_add_u32 s43, s43, s40")
    poll(["s43"], "ag")
    RX = V0 + 8  # per u: 4 dwords

    def nload(u):
        r = RX + 4 * u
        A(f"  s_mul_i32 s40, s26, {NCTA * MAXJ}")  # source w
        A(f"  s_mul_i32 s43, s27, {MAXJ}")
        A("  s_add_u32 s40, s40, s43")
        A("  s_add_u32 s40, s40, s45")  # chunk index
        A("  s_lshl_b32 s43, s40, 10")
        A(f"  v_add_u32 v{r}, s43, v4")
        A(f"  global_load_dwordx4 v[{r}:{r + 3}], v{r}, {pair(LAND2, R)} sc1")

    tiles(nload)
    A("  s_waitcnt vmcnt(0)")
    T = V0 + 32

    def nstore(u):
        out_store(RX + 4 * u, "s45", ctx, T + 9)

    tiles(nstore)
    A("  s_branch .Lbump")

    # ---- owner wave R: wait for the 3 hop-0 flags, reduce, push to land2[d] slot R, own out, hop-1 flags
    A(".Lowner:")
    for k, s_ in enumerate(peers):
        A(
            f"  s_add_u32 s{36 + k}, s40, {((0 * 4 + s_) * NCTA * 4) * 4}"
        )  # s36..38 (weights ptrs: dead)
    poll([f"s{36 + k}" for k in range(3)], "rs")
    # per u: own 4 dwords + per peer 4 dwords: 16 VGPRs
    OX = V0 + 8

    def oload(u):
        base = OX + 16 * u
        chunk_off(40, R, "s45")
        A(f"  v_add_u32 v{base}, s40, v4")
        A(f"  global_load_dwordx4 v[{base}:{base + 3}], v{base}, {pair(LAND, R)}")
        for k, s_ in enumerate(peers):
            r = base + 4 + 4 * k
            chunk_off(40, s_, "s45")
            A(f"  v_add_u32 v{r}, s40, v4")
            A(f"  global_load_dwordx4 v[{r}:{r + 3}], v{r}, {pair(LAND, R)} sc1")

    tiles(oload)
    A("  s_waitcnt vmcnt(0)")
    S8, T8 = V0 + 72, V0 + 80  # sum (8 f32), terms (8 f32)
    QO, TT = V0 + 88, V0 + 92  # out bf16 (4), temps (10)

    def oreduce(u):
        base = OX + 16 * u
        for r in range(4):
            dst = S8 if r == 0 else T8
            unpack(base if r == R else base + 4 + 4 * peers.index(r), dst)
            if r:
                for e in range(8):
                    A(f"  v_add_f32 v{S8 + e}, v{S8 + e}, v{T8 + e}")
        A(f"  s_mul_i32 s40, s27, {MAXJ}")
        A(f"  s_add_u32 s40, s40, {R * NCTA * MAXJ}")
        A("  s_add_u32 s40, s40, s45")  # chunk (R, b, j)
        pack(S8, QO)
        A("  s_lshl_b32 s43, s40, 10")
        A(f"  v_add_u32 v{TT}, s43, v4")
        for d in peers:
            A(f"  global_store_dwordx4 v{TT}, v[{QO}:{QO + 3}], {pair(LAND2, d)} sc1")
        out_store(QO, "s45", ctx, TT + 1)

    tiles(oreduce)
    A("  s_waitcnt vmcnt(0)")
    A("  s_lshl_b32 s40, s27, 2")
    A("  s_add_u32 s40, s40, s26")
    A("  s_lshl_b32 s40, s40, 2")
    A(f"  s_add_u32 s40, s40, {((1 * 4 + R) * NCTA * 4) * 4}")
    A(f"  v_mov_b32 v{FV}, s40")
    A(f"  v_mov_b32 v{FV + 1}, s{SEQ}")
    A("  s_mov_b64 exec, 1")
    for d in peers:
        A(f"  global_store_dword v{FV}, v{FV + 1}, {pair(FLAGS, d)} sc0 sc1")
    A("  s_branch .Lbump")

    A(".Ltimeout:")
    A("  s_mov_b64 exec, 1")
    A("  s_lshl_b32 s40, s27, 2")
    A("  s_add_u32 s40, s40, s26")
    A("  s_lshl_b32 s40, s40, 2")
    A(f"  s_add_u32 s40, s40, {ERR_OFF}")
    A(f"  v_mov_b32 v{FV}, s40")
    A(f"  v_mov_b32 v{FV + 1}, s{SEQ}")
    A(f"  global_store_dword v{FV}, v{FV + 1}, {pair(FLAGS, R)} sc1")
    A(".Lbump:")
    A("  s_mov_b64 exec, 1")
    A("  s_lshl_b32 s40, s27, 2")
    A("  s_add_u32 s40, s40, s26")
    A("  s_lshl_b32 s40, s40, 2")
    A(f"  s_add_u32 s40, s40, {CTR_OFF}")
    A(f"  v_mov_b32 v{FV}, s40")
    A(f"  v_mov_b32 v{FV + 1}, s{SEQ}")
    A(f"  global_store_dword v{FV}, v{FV + 1}, {pair(FLAGS, R)}")
    gemv8._footer(c)
    src = kernel_source(
        name,
        c,
        KA,
        lds=r["lds"],
        vgprs=r["vgprs"],
        agprs=r["agprs"],
        sgprs=102,
        block=256,
    )
    return dict(name=name, src=src, ka=KA)
