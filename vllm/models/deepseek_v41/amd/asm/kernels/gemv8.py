# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""MXFP8 × MXFP8 GEMV for the attention half's dense projections (A1 wqkv, A2 wq_b, A4 wo_a, A5 wo_b), M ≤ 256:

    out[t][n] = Σ_k W[n][k] · x[t][xk0(n) + k]

W: e4m3 [N, K], E8M0 per 32 × 32 block; x: e4m3 [M, G·K], E8M0 per 32
columns; G groups split the rows evenly and group g reads x columns g·K ..
(wo_a: two o-groups of 8 heads).

**Every global access is one contiguous 1 KB (or 64 B) per wave-instruction**
(a scattered dwordx4 costs ~50 CU cycles in the TA against 18, measured;
the first, row-major version of this kernel spent most
of its time there). So the operands live in MFMA-fragment order:
- weights, shuffled once (``frag_w``): per CTA (row group rg, slice s), row
  tile i, K tile k: [half h][lane l][16 B] = W[row0 + 16 i + l % 16]
  [s·KS + 128 k + 64 h + 16 (l / 16) ..]; scales [lane] (lane group q = l / 16
  holds 32-block q of the K tile);
- activations ("frag" layout, ``frag_x``; our producers write it): per token
  tile j (16 tokens) and global K tile kg: [h][l][16 B] = x[16 j + l % 16]
  [128 kg + 64 h + 16 (l / 16) ..], scales [j][kg][l];
- split partials: per CTA, token tile, row tile: [l][4 × f32].

Grid: NGP·S CTAs of 4 waves. A CTA owns R rows (TI = R/16 tiles) and K slice
s of KS columns (KT = KS/128 K tiles). CTA b: XCD x = b % 8, j = b / 8, s =
j % S, rg = x + 8 (j / S): the S slices of a row group share an XCD.
- The CTA's weight block (contiguous in frag order) goes to LDS with LDS-DMA
  (1 KB per instruction, a quarter per wave), then every wave reads it into
  its AGPRs (A operands of ``v_mfma_scale_f32_16x16x128_f8f6f4``). (A first
  version split K across the waves and reduced D through LDS: at M 256 that
  moved 320 KB of LDS per CU and was slower.)
- Wave w runs token tiles w, w + 4, w + 8, w + 12 over all KT K tiles (x
  prefetched NB − 1 tiles ahead); D lane l: token l % 16, rows 4q .. 4q + 3.
- S > 1: D → ``part``; then ``global_atomic_inc`` on cnt[rg] wrapping at
  S − 1; the last arriver sums all S partials in slice order (``nt``, same
  XCD; chunks of ≤ 4 slices, double-buffered) and runs the epilogue. S = 1:
  the epilogue directly.
Epilogues: "bf16" → out [M, N] bf16 row-major; "rope" → bf16, GPT-J RoPE on
rows ≥ 448 of each 512-row head, bf16 (vLLM's wq_b + q RoPE); "mxfp8" →
bf16-rounded, MXFP8 per 32 rows (ceil rule) → out / outs in the frag layout
with N/128 K tiles (the next GEMV's x); "idxq" (the indexer's wq_b) → bf16,
GPT-J RoPE on rows ≥ 64 of each 128-row head (bf16), MXFP4 per 32 rows with
aiter's ceil scale → codes [M, N/2] and E8M0 [M, N/32] row-major (vLLM's
q_rope_mxfp4_quant). Tokens ≥ M are computed but not stored (frag outputs
store them: the buffers hold 256 tokens).
"""

from dataclasses import dataclass

from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source

MAXJ = 16  # token tiles (M ≤ 256)
QN_Q = 1280  # epi "qn2": wqkv's q-latent rows (the rest is kv)
# split-K reduce (S > 1): the last arriver reduces all tiles, no inter-CTA waiting (a rendezvous deadlocked
# FULL graph capture at long context, when other streams held the CUs the slices needed)
MIX_WAVES = 2  # source(mix_wave=True): the extra waves per CTA

KA = KArgs(
    ("w", "ptr"),
    ("ws", "ptr"),
    ("x", "ptr"),
    ("xs", "ptr"),
    ("out", "ptr"),
    ("outs", "ptr"),
    ("part", "ptr"),
    ("cnt", "ptr"),
    ("cs", "ptr"),
    ("pos", "ptr"),
    ("dbg", "ptr"),
    ("m", "u32"),
)


@dataclass(frozen=True)
class Cfg:
    name: str
    N: int
    K: int  # per group
    R: int
    KS: int
    G: int = 1
    epi: str = "bf16"

    @property
    def S(self):
        return self.K // self.KS

    @property
    def KT(self):
        return self.KS // 128

    @property
    def KW(self):
        return (self.KT + 3) // 4

    @property
    def TI(self):
        return self.R // 16

    @property
    def NG(self):
        return self.N // self.R

    @property
    def NGP(self):
        return (self.NG + 7) // 8 * 8

    @property
    def grid(self):
        return self.NGP * self.S

    @property
    def KGT(self):
        return self.G * self.K // 128  # K tiles of x

    def part_bytes(self) -> int:
        return self.NG * self.S * MAXJ * self.TI * 1024 if self.S > 1 else 4


WO_A = Cfg("gemv_wo_a", N=2048, K=4096, R=32, KS=1024, G=2, epi="mxfp8")
WQ_B = Cfg("gemv_wq_b", N=8192, K=1280, R=32, KS=1280, epi="rope")


# ---------------------------------------------------------------- host-side layouts


def frag_w(w, ws, cfg: Cfg):
    """W e4m3 [N, K], ws E8M0 [N/32, K/32] → (frag bytes, frag scale bytes)."""
    import torch

    N, R, TI, KT, S = cfg.N, cfg.R, cfg.TI, cfg.KT, cfg.S
    w = w.view(torch.uint8).reshape(
        N // R, TI, 16, S, KT, 2, 4, 16
    )  # [rg, i, r, s, k, h, q, b]
    w = w.permute(0, 3, 1, 4, 5, 6, 2, 7).contiguous()  # [rg, s, i, k, h, q, r, b]
    sc = ws.view(torch.uint8).reshape(N // 32, S, KT, 4)  # [row block, s, k, q]
    rows = torch.arange(N // 16, device=sc.device) // 2  # row tile → row block
    sc = (
        sc[rows].reshape(N // R, TI, S, KT, 4).permute(0, 2, 1, 3, 4)
    )  # [rg, s, i, k, q]
    sc = sc.unsqueeze(-1).expand(*sc.shape, 16).contiguous()  # [.., q, r]
    return w.view(-1), sc.view(-1)


def frag_x(xq, xs):
    """X e4m3 [M, KX], xs [M, KX/32] → frag (MAXJ token tiles, zero padded):
    ([j][kg][h][l][16 B], [j][kg][l])."""
    import torch

    M, KX = xq.shape
    x = torch.zeros(MAXJ * 16, KX, dtype=torch.uint8, device=xq.device)
    x[:M] = xq.view(torch.uint8)
    s = torch.zeros(MAXJ * 16, KX // 32, dtype=torch.uint8, device=xq.device)
    s[:M] = xs.view(torch.uint8)
    x = (
        x.view(MAXJ, 16, KX // 128, 2, 4, 16).permute(0, 2, 3, 4, 1, 5).contiguous()
    )  # [j, kg, h, q, r, b]
    s = s.view(MAXJ, 16, KX // 128, 4).permute(0, 2, 3, 1).contiguous()  # [j, kg, q, r]
    return x.view(-1), s.view(-1)


# ---------------------------------------------------------------- kernel


def _emit(c, cfg: Cfg, mode: str, P: dict):
    """One GEMV phase's code appended to c (see source()). P: pf label prefix,
    end / exit: labels to leave by. Returns resources.

    mode "t" (large M): the CTA's weights go to LDS once (LDS-DMA) and every
    wave reads all of them into AGPRs, then runs its own token tiles over all
    K. Mode "k" (small M): wave w loads only its K tiles w, w + 4, .. straight
    into AGPRs and runs every token tile over them; the 4 waves' D meet in
    LDS (batches of 4 token tiles) and wave w finishes token tile 4b + w.
    Hooks (kernels/wob_ar.py): P["epi_fn"](ctx) replaces the S = 1 epilogue (ctx: register map, token tile in
    s45); P["epi_vm"] = the vm ops it issues per tile (mode k: the x waits of the next batch count them, so
    they don't wait for the epilogue's stores); P["after_map"](ctx) runs once the CTA mapping is known."""
    assert mode in ("t", "k")
    pf, end, ex = P.get("pf", ""), P.get("end", ".Lstamps"), P.get("exit", ".Lexit")
    N, K, R, KS, G, KT, S, TI, KW = (
        cfg.N,
        cfg.K,
        cfg.R,
        cfg.KS,
        cfg.G,
        cfg.KT,
        cfg.S,
        cfg.TI,
        cfg.KW,
    )
    assert (
        R % 16 == 0
        and KS % 128 == 0
        and K % KS == 0
        and N % R == 0
        and (N // G) % R == 0
    )
    assert S & (S - 1) == 0, "S must be a power of 2"
    assert TI <= 8
    if mode == "t":
        assert TI * KT * 8 <= 256, "weights don't fit the AGPRs"
        assert (TI * KT * 2) % 4 == 0
    else:
        assert KT >= 4 and TI * KW * 8 <= 256
    if cfg.epi in ("mxfp8", "qn2"):
        assert R % 32 == 0 and N % 128 == 0
    KX = KT if mode == "t" else KW  # K tiles per wave
    XB = KX * 9  # one token tile of x: KX × (8 data + 1 scale)
    V_L, V_R, V_Q = 1, 2, 3
    V_T = 4  # 16 l; mode k: LDS write base, 2 read bases (4..7)
    vmax = 256  # VGPR budget
    V_SA = 8  # weight scales TI × KX
    V_E = 8 + TI * KX  # epilogue temps (8)
    SUM = (V_E + 8 + 3) // 4 * 4  # D / summed tile: TI × 4
    D0 = SUM if mode == "t" else SUM + 4 * TI
    STG = D0 + 4 * TI  # mode k: LDS-reduce staging
    U0 = STG if mode == "t" else STG + 4 * TI  # union: x ring | split partial chunks
    XS_ = (XB + 1) // 2 * 2
    NB = max(2, min(4, (vmax - U0) // XS_))
    XBUF = tuple(U0 + XS_ * k for k in range(NB))
    SC = min(S, 4)  # slices per partial chunk
    PCH = (U0, U0 + 4 * TI * SC)  # two chunk buffers
    VGPRS = max(U0 + NB * XS_, PCH[1] + 4 * TI * SC if S > 1 else 0)
    assert vmax >= VGPRS, VGPRS
    AGPRS = TI * KX * 8
    WBYTES = TI * KT * 2048  # the CTA's weight block
    SLOT = TI * 1024  # mode k: one wave's D for one token tile
    LDS = (WBYTES if mode == "t" else 16 * SLOT) + 64  # + ticket word
    TICKET = LDS - 64

    A = c.append
    uid = P.setdefault("uid", [0])

    def label(p):
        uid[0] += 1
        return f".L{pf}{p}{uid[0]}"

    def nop(n):
        while n > 0:
            A(f"  s_nop {min(n, 8) - 1}")
            n -= 8

    # SGPRs: s[4:23] pointers, s24 m; s26 wave, s27 rg, s28 slice, s29 token tiles,
    # s32 row0, s33 first K tile of the slice in x; s[36:37] weights, s[38:39] their scales
    A("  s_and_b32 s27, s2, 7")
    A("  s_lshr_b32 s28, s2, 3")
    A(f"  s_lshr_b32 s30, s28, {S.bit_length() - 1}")
    A(f"  s_and_b32 s28, s28, {S - 1}")  # slice
    A("  s_lshl_b32 s30, s30, 3")
    A("  s_add_u32 s27, s27, s30")  # row group
    A(f"  s_cmp_ge_u32 s27, {cfg.NG}")
    A(f"  s_cbranch_scc1 {ex}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_add_u32 s29, s24, 15")
    A("  s_lshr_b32 s29, s29, 4")  # token tiles
    A(f"  s_mul_i32 s32, s27, {R}")  # row0
    A(f"  s_mul_i32 s33, s28, {KT}")
    if G > 1:
        A(f"  s_cmp_ge_u32 s32, {N // G}")
        A(f"  s_cselect_b32 s34, {K // 128}, 0")
        A("  s_add_u32 s33, s33, s34")
    if P.get("after_map"):
        P["after_map"](dict(A=A, V_L=V_L, V_R=V_R, V_Q=V_Q, V_T=V_T))
    A(f"  s_mul_i32 s34, s27, {S}")
    A("  s_add_u32 s34, s34, s28")  # CTA block rg·S + s
    A(f"  s_mul_i32 s35, s34, {TI * KT * 2048}")
    A("  s_add_u32 s36, s4, s35")
    A("  s_addc_u32 s37, s5, 0")
    A(f"  s_mul_i32 s35, s34, {TI * KT * 64}")
    A("  s_add_u32 s38, s6, s35")
    A("  s_addc_u32 s39, s7, 0")

    def wave_k(kk):
        """s40 = this wave's K tile kk (mode k), clamped to KT - 1 (clamped tiles
        load valid memory; their MFMAs are skipped)."""
        A(f"  s_add_u32 s40, s26, {4 * kk}")
        A(f"  s_min_u32 s40, s40, {KT - 1}")

    if mode == "t":
        # weights → LDS: wave w DMAs 1 KB pieces w, w + 4, ..
        NP = TI * KT * 2
        for p in range(NP // 4):
            A(f"  s_add_u32 s40, s26, {4 * p}")
            A("  s_lshl_b32 s41, s40, 10")
            A("  s_add_u32 s42, s36, s41")
            A("  s_addc_u32 s43, s37, 0")
            A("  s_mov_b32 m0, s41")
            A(f"  global_load_lds_dwordx4 v{V_T}, s[42:43]")
        # scales: [i][k][lane] bytes, contiguous 64 B per load
        for i in range(TI):
            for k in range(KT):
                A(
                    f"  global_load_ubyte v{V_SA + KT * i + k}, v{V_L}, s[38:39] offset:{64 * (KT * i + k)}"
                )
    else:
        for i in range(TI):
            for kk in range(KW):
                wave_k(kk)
                A(f"  s_add_u32 s40, s40, {KT * i}")
                A("  s_lshl_b32 s41, s40, 11")
                A("  s_add_u32 s42, s36, s41")
                A("  s_addc_u32 s43, s37, 0")
                a = 8 * (KW * i + kk)
                A(f"  global_load_dwordx4 a[{a}:{a + 3}], v{V_T}, s[42:43]")
                A(
                    f"  global_load_dwordx4 a[{a + 4}:{a + 7}], v{V_T}, s[42:43] offset:1024"
                )
                A("  s_lshl_b32 s41, s40, 6")
                A("  s_add_u32 s42, s38, s41")
                A("  s_addc_u32 s43, s39, 0")
                A(f"  global_load_ubyte v{V_SA + KW * i + kk}, v{V_L}, s[42:43]")

    def x_load(buf):
        """Token tile s44 → XBUF[buf]: all KT K tiles (mode t) or the wave's KW
        (mode k, clamped). Always 3·KX loads."""
        xb = XBUF[buf]
        A(f"  s_mul_i32 s45, s44, {cfg.KGT}")
        A("  s_add_u32 s45, s45, s33")  # j·KGT + kg0
        if mode == "t":
            A("  s_lshl_b32 s41, s45, 11")
            A("  s_add_u32 s42, s8, s41")
            A("  s_addc_u32 s43, s9, 0")
            A("  s_lshl_b32 s41, s45, 6")
            A("  s_add_u32 s46, s10, s41")
            A("  s_addc_u32 s47, s11, 0")
            for k in range(KT):
                if k and k % 2 == 0:
                    A("  s_add_u32 s42, s42, 4096")
                    A("  s_addc_u32 s43, s43, 0")
                o = 2048 * (k % 2)
                A(
                    f"  global_load_dwordx4 v[{xb + 8 * k}:{xb + 8 * k + 3}], v{V_T}, s[42:43] offset:{o}"
                )
                A(
                    f"  global_load_dwordx4 v[{xb + 8 * k + 4}:{xb + 8 * k + 7}], v{V_T}, s[42:43] offset:{o + 1024}"
                )
                A(
                    f"  global_load_ubyte v{xb + 8 * KT + k}, v{V_L}, s[46:47] offset:{64 * k}"
                )
        else:
            for kk in range(KW):
                wave_k(kk)
                A("  s_add_u32 s40, s40, s45")
                A("  s_lshl_b32 s41, s40, 11")
                A("  s_add_u32 s42, s8, s41")
                A("  s_addc_u32 s43, s9, 0")
                A(
                    f"  global_load_dwordx4 v[{xb + 8 * kk}:{xb + 8 * kk + 3}], v{V_T}, s[42:43]"
                )
                A(
                    f"  global_load_dwordx4 v[{xb + 8 * kk + 4}:{xb + 8 * kk + 7}], v{V_T}, s[42:43] offset:1024"
                )
                A("  s_lshl_b32 s41, s40, 6")
                A("  s_add_u32 s42, s10, s41")
                A("  s_addc_u32 s43, s11, 0")
                A(f"  global_load_ubyte v{xb + 8 * KW + kk}, v{V_L}, s[42:43]")

    def mfma(buf):
        xb = XBUF[buf]
        for k in range(KX):
            skip = None
            if mode == "k" and 4 * (k + 1) > KT:  # some waves have no tile k
                skip = label("mskip")
                A(f"  s_add_u32 s40, s26, {4 * k}")
                A(f"  s_cmp_ge_u32 s40, {KT}")
                A(f"  s_cbranch_scc1 {skip}")
            for i in range(TI):
                d, a = D0 + 4 * i, 8 * (KX * i + k)
                cin = "0" if k == 0 else f"v[{d}:{d + 3}]"
                A(
                    f"  v_mfma_scale_f32_16x16x128_f8f6f4 v[{d}:{d + 3}], a[{a}:{a + 7}], v[{xb + 8 * k}:{xb + 8 * k + 7}], "
                    f"{cin}, v{V_SA + KX * i + k}, v{xb + 8 * KX + k} op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:0 blgp:0"
                )
            if skip:
                A(f"{skip}:")

    def epilogue():
        """SUM (TI × 4; lane: token t = 16·s45 + l % 16, rows 4q ..) → out."""
        A("  s_lshl_b32 s41, s45, 4")
        A(f"  v_add_u32 v{V_E}, s41, v{V_R}")  # t
        A("  s_mov_b64 s[46:47], exec")
        for i in range(TI):
            for k in range(0, 4, 2):
                d = SUM + 4 * i + k
                A(f"  v_cvt_pk_bf16_f32 v{d}, v{d}, v{d + 1}")

        def rope(head, first):
            """GPT-J RoPE on rows first .. head - 1 of each head of ``head`` rows (bf16 pairs in SUM, cos / sin f32
            [pos][32 + 32] at cs), rotated values rounded to bf16."""
            done = label("norope")
            A(f"  s_and_b32 s40, s32, {head - 1}")
            A(f"  s_cmp_ge_u32 s40, {first}")
            A(f"  s_cbranch_scc0 {done}")
            A("  s_sub_u32 s41, s24, 1")
            A(f"  v_min_u32 v{V_E + 1}, s41, v{V_E}")  # clamp t for the load
            A(f"  v_lshlrev_b32 v{V_E + 1}, 3, v{V_E + 1}")
            A(
                f"  global_load_dwordx2 v[{V_E + 2}:{V_E + 3}], v{V_E + 1}, s[22:23]"
            )  # pos[t]
            A("  s_waitcnt vmcnt(0)")
            A(f"  v_lshlrev_b32 v{V_E + 1}, 8, v{V_E + 2}")  # pos·256
            A(f"  s_sub_u32 s40, s40, {first}")
            A("  s_lshr_b32 s40, s40, 1")  # first pair j0
            A(f"  v_lshl_add_u32 v{V_E + 1}, v{V_Q}, 3, v{V_E + 1}")  # + 4 · 2 q
            A("  s_lshl_b32 s40, s40, 2")
            A(f"  v_add_u32 v{V_E + 1}, s40, v{V_E + 1}")
            for i in range(TI):
                A(
                    f"  global_load_dwordx2 v[{V_E + 2}:{V_E + 3}], v{V_E + 1}, s[20:21] offset:{32 * i}"
                )
                A(
                    f"  global_load_dwordx2 v[{V_E + 4}:{V_E + 5}], v{V_E + 1}, s[20:21] offset:{128 + 32 * i}"
                )
                A("  s_waitcnt vmcnt(0)")
                for pr in range(2):
                    d = SUM + 4 * i + 2 * pr
                    e, o, cs_, sn = V_E + 6, V_E + 7, V_E + 2 + pr, V_E + 4 + pr
                    A(f"  v_lshlrev_b32 v{e}, 16, v{d}")
                    A(f"  v_and_b32 v{o}, 0xffff0000, v{d}")
                    A(f"  v_mul_f32 v{d}, v{e}, v{cs_}")
                    A(f"  v_fma_f32 v{d}, -v{o}, v{sn}, v{d}")  # e cos − o sin
                    A(f"  v_mul_f32 v{o}, v{o}, v{cs_}")
                    A(f"  v_fmac_f32 v{o}, v{e}, v{sn}")  # e sin + o cos
                    A(f"  v_cvt_pk_bf16_f32 v{d}, v{d}, v{o}")
            A(f"{done}:")

        if cfg.epi == "rope":
            rope(512, 448)
        if cfg.epi == "idxq":
            rope(128, 64)

        def store_bf16():
            A(f"  v_cmp_gt_u32 vcc, s24, v{V_E}")
            A("  s_and_b64 exec, exec, vcc")
            A(f"  v_mul_u32_u24 v{V_E + 1}, {N}, v{V_E}")
            A(f"  v_add_u32 v{V_E + 1}, s32, v{V_E + 1}")
            A(f"  v_lshl_add_u32 v{V_E + 1}, v{V_Q}, 2, v{V_E + 1}")
            A(f"  v_lshlrev_b32 v{V_E + 1}, 1, v{V_E + 1}")
            for i in range(TI):
                d = SUM + 4 * i
                A(f"  v_mov_b32 v{d + 1}, v{d + 2}")
                A(
                    f"  global_store_dwordx2 v{V_E + 1}, v[{d}:{d + 1}], s[12:13] offset:{32 * i}"
                )

        def store_mxfp8(
            codes="s[12:13]", scales="s[14:15]", KGO=N // 128, unpacked=False
        ):
            """MXFP8 per 32 rows (tiles 2cb, 2cb + 1) → frag layout of an [M, KGO·128] activation."""
            for cb in range(TI // 2):
                ys = []
                for i in (2 * cb, 2 * cb + 1):
                    for k in range(0, 4, 2):
                        d = SUM + 4 * i + k
                        if not unpacked:
                            A(f"  v_and_b32 v{d + 1}, 0xffff0000, v{d}")
                            A(f"  v_lshlrev_b32 v{d}, 16, v{d}")
                    ys += [SUM + 4 * i + k for k in range(4)]
                a, e, mt, ng = V_E + 3, V_E + 4, V_E + 5, V_E + 6
                A(f"  v_max_f32 v{a}, |v{ys[0]}|, |v{ys[1]}|")
                for y in ys[2:]:
                    A(f"  v_max_f32 v{a}, v{a}, |v{y}|")
                for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32"):
                    A(f"  v_mov_b32 v{e}, v{a}")
                    nop(2)
                    A(f"  {sw} v{a}, v{e}")
                    nop(4)
                    A(f"  v_max_f32 v{a}, v{a}, v{e}")
                A(f"  v_max_f32 v{a}, 0x00800000, v{a}")
                A(f"  v_frexp_exp_i32_f32 v{e}, v{a}")
                A(f"  v_frexp_mant_f32 v{mt}, v{a}")
                A(f"  v_cmp_lt_f32 vcc, 0x3f600000, v{mt}")
                A(f"  v_addc_co_u32 v{e}, vcc, -9, v{e}, vcc")
                A(f"  v_max_i32 v{e}, 0xffffff81, v{e}")
                A(f"  v_min_i32 v{e}, 0x7f, v{e}")
                A(f"  v_sub_u32 v{ng}, 0, v{e}")
                for y in ys:
                    A(f"  v_ldexp_f32 v{y}, v{y}, v{ng}")
                # rows row0 + 32 cb + 16 h + 4 q: kg = row / 128, half = (row % 128) / 64,
                # q' = (row % 64) / 16 (tile 2cb + h → q' + h), byte 4 q
                A(f"  s_add_u32 s40, s32, {32 * cb}")
                A("  s_lshr_b32 s41, s40, 7")  # kg
                A(f"  s_mul_i32 s42, s45, {KGO}")
                A("  s_add_u32 s41, s41, s42")  # j·KGO + kg
                A("  s_lshl_b32 s42, s41, 11")
                A("  s_and_b32 s43, s40, 127")
                A("  s_lshr_b32 s43, s43, 6")
                A("  s_lshl_b32 s43, s43, 10")  # half · 1 KB
                A("  s_add_u32 s42, s42, s43")
                A("  s_and_b32 s43, s40, 63")
                A("  s_lshr_b32 s43, s43, 4")
                A("  s_lshl_b32 s43, s43, 8")  # q' · 256
                A("  s_add_u32 s42, s42, s43")
                A(f"  v_lshlrev_b32 v{V_E + 1}, 4, v{V_R}")
                A(f"  v_lshl_add_u32 v{V_E + 1}, v{V_Q}, 2, v{V_E + 1}")
                A(f"  v_add_u32 v{V_E + 1}, s42, v{V_E + 1}")  # + 16 r + 4 q
                for h, i in enumerate((2 * cb, 2 * cb + 1)):
                    d = SUM + 4 * i
                    A(f"  v_cvt_pk_fp8_f32 v{d}, v{d}, v{d + 1}")
                    A(f"  v_cvt_pk_fp8_f32 v{d}, v{d + 2}, v{d + 3} op_sel:[0,0,1]")
                    A(
                        f"  global_store_dword v{V_E + 1}, v{d}, {codes} offset:{256 * h}"
                    )
                A(
                    f"  v_add_u32 v{e}, 0x7f, v{e}"
                )  # scale → [j][kg][r + 16 ((row % 128) / 32)]
                A("  s_lshl_b32 s42, s41, 6")
                A("  s_and_b32 s43, s40, 127")
                A("  s_lshr_b32 s43, s43, 5")
                A("  s_lshl_b32 s43, s43, 4")
                A("  s_add_u32 s42, s42, s43")
                A(f"  v_add_u32 v{V_E + 2}, s42, v{V_R}")
                A(f"  v_cmp_eq_u32 vcc, 0, v{V_Q}")
                A("  s_and_saveexec_b64 s[42:43], vcc")
                A(f"  global_store_byte v{V_E + 2}, v{e}, {scales}")
                A("  s_mov_b64 exec, s[42:43]")

        if cfg.epi in ("bf16", "rope"):
            store_bf16()
        if cfg.epi == "idxq":
            # the indexer's q: after the RoPE above, MXFP4 per 32 rows = the CTA's block of head
            # row0 / 128 (aiter's q_rope_mxfp4_quant: ceil scale e = (bits(max(amax, 6·2^-126) · fl(1/6)) +
            # 0x7fffff) >> 23, e2m1 RNE) → codes out + t·2048 + row0 / 2 + {2q, 8 + 2q} (16 bits each: rows
            # 4q .. + 3 of tiles 0 / 1), scale byte outs + t·128 + row0 / 32
            assert R == 32 and TI == 2
            ys = []
            for i in range(TI):
                for k in range(0, 4, 2):
                    d = SUM + 4 * i + k
                    A(f"  v_and_b32 v{d + 1}, 0xffff0000, v{d}")
                    A(f"  v_lshlrev_b32 v{d}, 16, v{d}")
                ys += [SUM + 4 * i + k for k in range(4)]
            a, e, pk = V_E + 3, V_E + 4, V_E + 5
            A(f"  v_max_f32 v{a}, |v{ys[0]}|, |v{ys[1]}|")
            for y in ys[2:]:
                A(f"  v_max_f32 v{a}, v{a}, |v{y}|")
            for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32"):
                A(f"  v_mov_b32 v{e}, v{a}")
                nop(2)
                A(f"  {sw} v{a}, v{e}")
                nop(4)
                A(f"  v_max_f32 v{a}, v{a}, v{e}")
            A(f"  v_max_f32 v{a}, 0x01c00000, v{a}")  # 6 · 2^-126
            A(f"  v_mul_f32 v{a}, 0x3e2aaaab, v{a}")  # · fl(1/6)
            A(f"  v_add_u32 v{a}, 0x7fffff, v{a}")
            A(f"  v_lshrrev_b32 v{e}, 23, v{a}")  # e8m0
            A(f"  v_lshlrev_b32 v{a}, 23, v{e}")  # scale operand 2^(e - 127)
            for n, (y0, y1) in enumerate(zip(ys[0::2], ys[1::2])):
                A(
                    f"  v_cvt_scalef32_pk_fp4_f32 v{pk}, v{y0}, v{y1}, v{a} op_sel:[0,0,{n & 1},{n >> 1}]"
                )
            A(f"  v_cmp_gt_u32 vcc, s24, v{V_E}")
            A("  s_and_b64 exec, exec, vcc")
            A(f"  v_lshlrev_b32 v{V_E + 1}, 11, v{V_E}")  # t · 2048
            A("  s_lshr_b32 s40, s32, 1")
            A(f"  v_add_u32 v{V_E + 1}, s40, v{V_E + 1}")
            A(f"  v_lshl_add_u32 v{V_E + 1}, v{V_Q}, 1, v{V_E + 1}")
            A(f"  global_store_short v{V_E + 1}, v{pk}, s[12:13]")
            A(f"  global_store_short_d16_hi v{V_E + 1}, v{pk}, s[12:13] offset:8")
            A(f"  v_cmp_eq_u32 vcc, 0, v{V_Q}")
            A("  s_and_b64 exec, exec, vcc")
            A(f"  v_lshlrev_b32 v{V_E + 1}, 7, v{V_E}")  # t · 128
            A("  s_lshr_b32 s40, s32, 5")
            A(f"  v_add_u32 v{V_E + 1}, s40, v{V_E + 1}")
            A(f"  global_store_byte v{V_E + 1}, v{e}, s[14:15]")
        if cfg.epi == "mxfp8":
            store_mxfp8()
        if cfg.epi == "qn2":
            # wqkv for the K1 front without qkv_norm: q rows (< QN_Q) → x = bf16 of the
            # GEMV (vLLM's qr), Σx² over the CTA's 32 rows → pos + (t·40 + row0 / 32)·4 (token-major), then
            # y = bf16(x · q_norm.weight) (q_norm.weight bf16 at dbg) → MXFP8 in wq_b's x frag order: codes at
            # outs, scales at cs. The token's rrms is applied later, by A3 (attn_dec1 rr_scale). kv rows: bf16
            # row-major at out (qr_kv, qkv_norm's kv input).
            assert R == 32 and TI == 2
            kvl, done = label("qn2kv"), label("qn2done")
            A(f"  s_cmp_ge_u32 s32, {QN_Q}")
            A(f"  s_cbranch_scc1 {kvl}")
            ys = []
            for i in range(TI):
                for k in range(0, 4, 2):
                    d = SUM + 4 * i + k
                    A(f"  v_and_b32 v{d + 1}, 0xffff0000, v{d}")
                    A(f"  v_lshlrev_b32 v{d}, 16, v{d}")
                ys += [SUM + 4 * i + k for k in range(4)]
            acc, t1 = V_E + 2, V_E + 3
            A(f"  v_mul_f32 v{acc}, v{ys[0]}, v{ys[0]}")
            for y in ys[1:]:
                A(f"  v_fmac_f32 v{acc}, v{y}, v{y}")
            for sw in ("v_permlane32_swap_b32", "v_permlane16_swap_b32"):
                A(f"  v_mov_b32 v{t1}, v{acc}")
                nop(2)
                A(f"  {sw} v{acc}, v{t1}")
                nop(4)
                A(f"  v_add_f32 v{acc}, v{acc}, v{t1}")
            A(f"  v_cmp_gt_u32 vcc, s24, v{V_E}")
            A("  s_and_b64 exec, exec, vcc")
            A(f"  v_cmp_eq_u32 vcc, 0, v{V_Q}")
            A("  s_and_b64 exec, exec, vcc")
            A("  s_lshr_b32 s40, s32, 5")
            A("  s_lshl_b32 s40, s40, 2")
            A(f"  v_mul_u32_u24 v{t1}, {4 * QN_Q // 32}, v{V_E}")
            A(f"  v_add_u32 v{t1}, s40, v{t1}")
            A(f"  global_store_dword v{t1}, v{acc}, s[22:23]")
            A("  s_mov_b64 exec, s[46:47]")
            # q_norm.weight of the lane's rows: tile i rows row0 + 16 i + 4 q .. + 3
            A(f"  v_lshlrev_b32 v{t1}, 3, v{V_Q}")
            A("  s_lshl_b32 s40, s32, 1")
            A(f"  v_add_u32 v{t1}, s40, v{t1}")
            for i in range(TI):
                A(
                    f"  global_load_dwordx2 v[{V_E + 4 + 2 * i}:{V_E + 5 + 2 * i}], v{t1}, s[50:51] offset:{32 * i}"
                )
            A("  s_waitcnt vmcnt(0)")
            for i in range(TI):
                for k in range(0, 4, 2):
                    d, wv = SUM + 4 * i + k, V_E + 4 + 2 * i + k // 2
                    A(f"  v_lshlrev_b32 v{t1}, 16, v{wv}")
                    A(f"  v_mul_f32 v{d}, v{d}, v{t1}")
                    A(f"  v_and_b32 v{t1}, 0xffff0000, v{wv}")
                    A(f"  v_mul_f32 v{d + 1}, v{d + 1}, v{t1}")
                    A(f"  v_cvt_pk_bf16_f32 v{d}, v{d}, v{d + 1}")
                    A(f"  v_and_b32 v{d + 1}, 0xffff0000, v{d}")
                    A(f"  v_lshlrev_b32 v{d}, 16, v{d}")
            store_mxfp8("s[14:15]", "s[20:21]", QN_Q // 128, unpacked=True)
            A(f"  s_branch {done}")
            A(f"{kvl}:")
            store_bf16()
            A(f"{done}:")
        A("  s_mov_b64 exec, s[46:47]")

    def part_base(sreg):
        """s[42:43] = part + ((rg·S + slice)·MAXJ + s45)·TI KB (slice in sreg)."""
        A(f"  s_mul_i32 s40, s27, {S}")
        A(f"  s_add_u32 s40, s40, {sreg}")
        A(f"  s_lshl_b32 s40, s40, {MAXJ.bit_length() - 1}")
        A("  s_add_u32 s40, s40, s45")
        A(f"  s_mul_i32 s40, s40, {TI * 1024}")
        A("  s_add_u32 s42, s16, s40")
        A("  s_addc_u32 s43, s17, 0")

    def finish_tile():
        """SUM = the CTA's D of token tile s45 → part (S > 1) or out (S = 1)."""
        if S > 1:
            part_base("s28")
            for i in range(TI):
                if i == 4:
                    A("  s_add_u32 s42, s42, 4096")
                    A("  s_addc_u32 s43, s43, 0")
                A(
                    f"  global_store_dwordx4 v{V_T}, v[{SUM + 4 * i}:{SUM + 4 * i + 3}], s[42:43] offset:{1024 * (i % 4)}"
                )
        elif P.get("epi_fn"):
            P["epi_fn"](
                dict(
                    A=A,
                    SUM=SUM,
                    V_E=V_E,
                    V_L=V_L,
                    V_R=V_R,
                    V_Q=V_Q,
                    V_T=V_T,
                    label=label,
                    nop=nop,
                )
            )
        else:
            epilogue()

    NT = MAXJ // 4
    PDn = NB - 1
    if mode == "t":
        # ---- weights LDS → AGPRs, then token tiles w + 4 n over all K, x prefetched NB - 1 ahead
        def x_load_n(n):
            """Token tile min(w + 4 n, TT - 1) → XBUF[n % NB] (always issued: static vmcnt counts)."""
            A(f"  s_add_u32 s44, s26, {4 * n}")
            A("  s_sub_u32 s45, s29, 1")
            A("  s_min_u32 s44, s44, s45")
            x_load(n % NB)

        A("  s_waitcnt vmcnt(0)")
        A("  s_barrier")
        rb_ = V_T
        for i in range(TI):
            for k in range(KT):
                a = 8 * (KT * i + k)
                for h in range(2):
                    off = ((KT * i + k) * 2 + h) * 1024
                    A(
                        f"  ds_read_b128 a[{a + 4 * h}:{a + 4 * h + 3}], v{rb_} offset:{off}"
                    )
        for n in range(min(PDn, NT)):
            x_load_n(n)
        A("  s_waitcnt lgkmcnt(0)")
        for n in range(NT):
            if n + PDn < NT:
                x_load_n(n + PDn)
            ahead = min(PDn, NT - 1 - n)
            A(f"  s_waitcnt vmcnt({3 * KT * ahead})")
            A(f"  s_add_u32 s45, s26, {4 * n}")
            A("  s_cmp_lt_u32 s45, s29")
            A(f"  s_cbranch_scc0 .L{pf}main_done")
            mfma(n % NB)
            nop(20)
            finish_tile()
    else:
        # ---- every token tile over the wave's K tiles; batches of 4 meet in LDS
        def x_load_n(n):
            A("  s_sub_u32 s44, s29, 1")
            A(f"  s_min_u32 s44, s44, {n}")
            x_load(n % NB)

        A(f"  s_mul_i32 s48, s26, {4 * SLOT}")
        A(f"  v_add_u32 v{V_T + 1}, s48, v{V_T}")  # write base (4 w)·SLOT + 16 l
        A(f"  s_mul_i32 s48, s26, {SLOT}")
        A(f"  v_add_u32 v{V_T + 2}, s48, v{V_T}")  # read base for jj = w
        A(f"  v_add_u32 v{V_T + 3}, {8 * SLOT}, v{V_T + 2}")  # waves 2, 3
        for n in range(PDn):
            x_load_n(n)
        for b in range(MAXJ // 4):
            A(f"  s_cmp_le_u32 s29, {4 * b}")
            A(f"  s_cbranch_scc1 .L{pf}main_done")
            for jj in range(4):
                n = 4 * b + jj
                zero = label("zero")
                if n + PDn < MAXJ:
                    x_load_n(n + PDn)
                ahead = min(PDn, MAXJ - 1 - n)
                # the previous batch's epilogue ops are younger than tile n's loads for jj < PDn (if a batch
                # runs, every wave finished a tile in the one before)
                ev = P.get("epi_vm", 0) if b and jj < PDn else 0
                A(f"  s_waitcnt vmcnt({3 * KW * ahead + ev})")
                A(f"  s_cmp_le_u32 s29, {n}")
                A(f"  s_cbranch_scc1 {zero}")
                mfma(n % NB)
                nop(20)
                for i in range(TI):
                    d = D0 + 4 * i
                    A(
                        f"  ds_write_b128 v{V_T + 1}, v[{d}:{d + 3}] offset:{jj * SLOT + 1024 * i}"
                    )
                A(f"{zero}:")
            A("  s_waitcnt lgkmcnt(0)")
            A("  s_barrier")
            fin = label("fin")
            A(f"  s_add_u32 s45, s26, {4 * b}")
            A("  s_cmp_lt_u32 s45, s29")
            A(f"  s_cbranch_scc0 {fin}")
            for wv in range(4):  # sum in wave order
                base, off = (
                    (V_T + 2, 4 * SLOT * wv)
                    if wv < 2
                    else (V_T + 3, 4 * SLOT * (wv - 2))
                )
                dst = SUM if wv == 0 else STG
                for i in range(TI):
                    A(
                        f"  ds_read_b128 v[{dst + 4 * i}:{dst + 4 * i + 3}], v{base} offset:{off + 1024 * i}"
                    )
                A("  s_waitcnt lgkmcnt(0)")
                if wv:
                    for i in range(TI):
                        for k in range(4):
                            A(
                                f"  v_add_f32 v{SUM + 4 * i + k}, v{SUM + 4 * i + k}, v{STG + 4 * i + k}"
                            )
            finish_tile()
            A(f"{fin}:")
            A("  s_barrier")  # LDS slots free again
    A(f".L{pf}main_done:")
    if S > 1:
        # no waiting: the last arriver reduces every tile
        A("  s_waitcnt vmcnt(0)")
        A("  s_barrier")
        A("  s_cmp_eq_u32 s26, 0")
        A(f"  s_cbranch_scc0 .L{pf}ticket_wait")
        A(f"  v_mov_b32 v{V_E}, 1")
        A("  s_lshl_b32 s40, s27, 2")
        A(f"  v_mov_b32 v{V_E + 1}, s40")
        A("  s_mov_b64 exec, 1")
        A(f"  global_atomic_add v{V_E + 2}, v{V_E + 1}, v{V_E}, s[18:19] sc0 sc1")
        A("  s_waitcnt vmcnt(0)")
        A(f"  v_readfirstlane_b32 s40, v{V_E + 2}")
        # the group's last arriver (old count ≡ S − 1 mod S; the counter is monotonic) reduces; LDS tells the waves
        A(f"  s_and_b32 s40, s40, {S - 1}")
        A(f"  s_cmp_eq_u32 s40, {S - 1}")
        A("  s_cselect_b32 s40, 1, 0")
        A(f"  v_mov_b32 v{V_E + 2}, s40")
        A(f"  v_mov_b32 v{V_E + 1}, {TICKET}")
        A(f"  ds_write_b32 v{V_E + 1}, v{V_E + 2}")
        A("  s_waitcnt lgkmcnt(0)")
        A("  s_mov_b64 exec, -1")
        A(f".L{pf}ticket_wait:")
        A("  s_barrier")
        A(f"  v_mov_b32 v{V_E + 1}, {TICKET}")
        A(f"  ds_read_b32 v{V_E + 2}, v{V_E + 1}")
        A("  s_waitcnt lgkmcnt(0)")
        A(f"  v_readfirstlane_b32 s40, v{V_E + 2}")
        A("  s_cmp_eq_u32 s40, 0")
        A(f"  s_cbranch_scc1 {end}")
        A("  s_mov_b32 s28, 0")  # every token tile: as slice 0 of a 1-slice split
        items = [(u, ch) for u in range(MAXJ // 4 or 1) for ch in range(S // SC)]

        def tile_of(u):
            """s45 = s + w + 4u."""
            A(f"  s_add_u32 s45, s26, {4 * u}")
            A("  s_add_u32 s45, s45, s28")

        def chunk_load(k):
            u, ch = items[k]
            buf = PCH[k % 2]
            tile_of(u)
            A("  s_sub_u32 s49, s29, 1")
            A("  s_min_u32 s45, s45, s49")  # clamped (loads past the end stay valid)
            for sl in range(SC):
                A(f"  s_mov_b32 s49, {ch * SC + sl}")
                part_base("s49")
                for i in range(TI):
                    if i == 4:
                        A("  s_add_u32 s42, s42, 4096")
                        A("  s_addc_u32 s43, s43, 0")
                    r = buf + 4 * (TI * sl + i)
                    A(
                        f"  global_load_dwordx4 v[{r}:{r + 3}], v{V_T}, s[42:43] offset:{1024 * (i % 4)} nt"
                    )

        chunk_load(0)
        for k, (u, ch) in enumerate(items):
            if ch == 0:
                tile_of(u)
                A("  s_cmp_lt_u32 s45, s29")
                A(f"  s_cbranch_scc0 {end}")
            if k + 1 < len(items):
                chunk_load(k + 1)
                A(f"  s_waitcnt vmcnt({SC * TI})")
            else:
                A("  s_waitcnt vmcnt(0)")
            buf = PCH[k % 2]
            for sl in range(SC):
                for i in range(TI):
                    for q in range(4):
                        d, r = SUM + 4 * i + q, buf + 4 * (TI * sl + i) + q
                        A(
                            f"  v_mov_b32 v{d}, v{r}"
                            if ch == 0 and sl == 0
                            else f"  v_add_f32 v{d}, v{d}, v{r}"
                        )
            if ch == S // SC - 1:
                tile_of(u)
                epilogue()
    return dict(vgprs=VGPRS, agprs=AGPRS, lds=LDS, V_E=V_E)


def _header(c):
    A = c.append
    A("  s_load_dwordx16 s[4:19], s[0:1], 0x0")  # w ws x xs out outs part cnt
    A("  s_load_dwordx4 s[20:23], s[0:1], 0x40")  # cs pos
    A("  s_load_dwordx2 s[50:51], s[0:1], 0x50")  # dbg
    A("  s_load_dword s24, s[0:1], 0x58")  # m
    A("  v_readfirstlane_b32 s26, v0")
    A("  s_lshr_b32 s26, s26, 6")
    A("  v_and_b32 v1, 63, v0")
    A("  v_and_b32 v2, 15, v1")
    A("  v_lshrrev_b32 v3, 4, v1")
    A("  v_lshlrev_b32 v4, 4, v1")  # 16 l


def _footer(c):
    A = c.append
    A(".Lstamps:")
    A(".Lexit:")
    A("  s_endpgm")


def source(
    cfg: Cfg,
    mode: str = "t",
    kv_wave: bool = False,
    mix_wave: bool = False,
    extra: tuple | None = None,
):
    """One GEMV kernel (see the module docstring and _emit). kv_wave (wq_b without a qkv_norm launch):
    a 5th wave per CTA runs qkv_norm's kv path (record + SWA insert) for token = workgroup id, with
    qkv_norm's arguments appended (KA_KVR, prefixed kv_); kv comes in without the attention norm's rstd,
    which the wave reads from kv_rstd[t]; it loads everything first and steps through the GEMV's barriers
    (mode k, S = 1; qkv_norm bar_m). mix_wave: the seam fin's mixes
    (ffn_seam.mix_wave; KA_MX, prefixed mx_) in MIX_WAVES extra waves: wave 4 + i takes tokens
    b + grid·i, + grid·MIX_WAVES, .. (one each up to M = grid·MIX_WAVES); they step through the GEMV's
    barriers before their math (mode k, split-K).
    extra = (cfg2, mode2) (wq_b + the indexer's wq_b in one launch): CTAs cfg.grid ..
    cfg.grid + cfg2.grid − 1 run cfg2's GEMV on the same x / xs / pos / m with their own w / ws / out / outs /
    part / cnt / cs (KArgs appended, prefixed x2_; KA_X2 below); their extra waves (kv / mix) end at once."""
    from vllm.models.deepseek_v41.amd.asm.kernels import ffn_seam, qkv_norm

    assert not (kv_wave and mix_wave)
    c = []
    if extra:
        c += [f"  s_cmp_ge_u32 s2, {cfg.grid}", "  s_cbranch_scc1 .Lx2"]
    if kv_wave or mix_wave:
        c += [
            "  v_readfirstlane_b32 s26, v0",
            "  s_lshr_b32 s26, s26, 6",
            f"  s_cmp_{'ge' if mix_wave else 'eq'}_u32 s26, 4",
            f"  s_cbranch_scc1 {'.Lkvwave' if kv_wave else '.Lmxwave'}",
        ]
    _header(c)
    r = _emit(c, cfg, mode, {})
    _footer(c)
    ka, block, name, lds, sgprs = KA, 256, cfg.name, r["lds"], 72
    if kv_wave:
        c.append(".Lkvwave:")
        ka = KA_KVR
        assert mode == "k" and cfg.S == 1, "barrier count is mode k S = 1's"
        c.append(f"  s_load_dword s53, s[0:1], {KA.off['m']:#x}")
        c += qkv_norm.source(ka_base=ka.off["kv_qr_kv"], bar_m="s53")
        block, name = 320, cfg.name + "_kvrb"
        assert max(r["vgprs"], 96) + r["agprs"] <= 256, "5 waves need 2 per SIMD"
    if mix_wave:
        c.append(".Lmxwave:")
        assert mode == "k" and cfg.S > 1, "barrier count is mode k split-K's"
        c += [
            f"  s_load_dword s24, s[0:1], {KA.off['m']:#x}",
            "  s_sub_u32 s26, s26, 4",
            f"  s_lshl_b32 s25, s26, {ffn_seam.MIX_LDS.bit_length() - 1}",
            f"  s_mul_i32 s26, s26, {cfg.grid}",
            "  s_add_u32 s26, s26, s2",
        ]
        c += ffn_seam.mix_wave(
            KA_MX.off["mx_partial"],
            cfg.grid * MIX_WAVES,
            lds,
            first="s26",
            slot="s25",
            bar_m="s24",
        )
        ka, block, name = KA_MX, 64 * (4 + MIX_WAVES), cfg.name + "_mx"
        # every wave of a CTA has its own LDS slot (MIX_LDS each)
        lds, sgprs = lds + ffn_seam.MIX_LDS * MIX_WAVES, 102
        assert MIX_WAVES <= 4 and max(r["vgprs"], 136) + r["agprs"] <= 256, (
            "the extra waves need 2 per SIMD"
        )
    vgprs, agprs = (
        max(r["vgprs"], 136 if mix_wave else 96 if kv_wave else 0),
        r["agprs"],
    )
    if extra:
        cfg2, mode2 = extra
        ka = KArgs(*ka.args, *[("x2_" + n, "ptr") for n in X2])
        c += [
            ".Lx2:",
            "  v_readfirstlane_b32 s26, v0",
            "  s_lshr_b32 s26, s26, 6",
            "  s_cmp_ge_u32 s26, 4",
            "  s_cbranch_scc1 .Lx2_end",
            f"  s_sub_u32 s2, s2, {cfg.grid}",
        ]
        _header(c)
        c.append(
            "  s_waitcnt lgkmcnt(0)"
        )  # (scalar loads may land out of order: no overlap)
        for i, sreg in enumerate((4, 12, 16)):  # w ws | out outs | part cnt
            c.append(
                f"  s_load_dwordx4 s[{sreg}:{sreg + 3}], s[0:1], {ka.off['x2_' + X2[2 * i]]:#x}"
            )
        c.append(f"  s_load_dwordx2 s[20:21], s[0:1], {ka.off['x2_cs']:#x}")
        r2 = _emit(c, cfg2, mode2, dict(pf="x2", exit=".Lx2_end", end=".Lx2_end"))
        c += [".Lx2_end:", "  s_waitcnt vmcnt(0)", "  s_endpgm"]
        vgprs, agprs, lds = (
            max(vgprs, r2["vgprs"]),
            max(agprs, r2["agprs"]),
            max(lds, r2["lds"]),
        )
        name += "_x2" + cfg2.name.removeprefix("gemv") + "_" + mode2
        if kv_wave:
            assert max(vgprs, 96) + agprs <= 256, "5 waves need 2 per SIMD"
    src = kernel_source(
        name, c, ka, lds=lds, vgprs=vgprs, agprs=agprs, sgprs=sgprs, block=block
    )
    return dict(
        name=name,
        src=src,
        ka=ka,
        block=block,
        grid=cfg.grid + (extra[0].grid if extra else 0),
    )


X2 = ("w", "ws", "out", "outs", "part", "cnt", "cs")  # source(extra=)'s own pointers


# wo_b unsplit (160 CTAs of 32 rows over all K, mode "k" only): no split-K
# rendezvous
WO_B1 = Cfg("gemv_wo_b1", N=5120, K=2048, R=32, KS=2048)
# wqkv in 32-row tiles with epilogue "qn2": q → wq_b's MXFP8 x directly
WQKV32N = Cfg("gemv_wqkv32n", N=1792, K=5120, R=32, KS=1280, epi="qn2")

# the indexer's wq_b on index layers: qr's MXFP8 (wq_b's x) → bf16 [M, 32 × 128];
# 256 CTAs of 16 rows: cold 4.4 / 5.8 / 7.2 µs at M 64 / 128 / 256 (k, k, t) vs 4.8 / 6.3 / 7.9 with 32 rows
IDX_WQ_B = Cfg("gemv_idx_wq_b", N=4096, K=1280, R=16, KS=1280)
# + the indexer q's RoPE and MXFP4 in the epilogue: out = codes [M, 32, 64], outs = scales [M, 32, 4]
IDX_WQ_BQ = Cfg("gemv_idx_wq_bq", N=4096, K=1280, R=32, KS=1280, epi="idxq")


def _ka_ext():
    from vllm.models.deepseek_v41.amd.asm.kernels import ffn_seam, qkv_norm

    return (
        KArgs(*KA.args, *[("kv_" + n, t) for n, t in qkv_norm.KA_R.args]),
        KArgs(*KA.args, *[("mx_" + n, t) for n, t in ffn_seam.KA_MIX.args]),
    )


KA_KVR, KA_MX = _ka_ext()
