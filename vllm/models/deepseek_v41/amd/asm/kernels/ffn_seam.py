# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501  (one asm instruction or template line per source line)
"""FFN seam, main pass: post-mix, collapse and the projection
partials, the same outputs as aiter's `_mhc_fused_post_pre_delayed_rmsnorm_main_kernel`
(NUM_KSPLIT 160).

Task = (token tile tt of 16, column slice cs of 32), one wave each (the
schedule: `source_v2`). Lane l: token t = 16 tt + l % 16,
columns c0 = 32 cs + 8 (l / 16) .. c0 + 7, all 4 streams.
- R'_j = bf16(fma chain: p_j·x + Σ_h comb`[h][j]`·R_h) → r_out [M, 4, H].
- x1 = bf16(Σ_j pre_j·R'_j) → x1 [M, H] (un-normalised collapse).
- partial [M, 160, 32] f32: [0:24] = R'_slice · fn_slice^T, [24] Σ R'²,
  [25] Σ x1² (fp32 of the bf16 values). [26:32] are not written.
- The projection is two bf16 MFMAs per stream against fn = hi + lo
  (hi = bf16(fn), lo = bf16(fn − hi)), as aiter does. The lane's 8 packed
  bf16 of R'_j are exactly its A operand of `v_mfma_f32_16x16x32_bf16`
  (lane group g holds K 8g..8g+7); B lane l holds fn row l % 16 (+16), the
  same 8 columns. Rows 24..31 of fn read out of range → 0.
Rows t ≥ M load as 0 and their stores fall outside the buffers.
"""

import struct

from vllm.models.deepseek_v41.amd.asm.kernels.kmod import KArgs, kernel_source, rsrc

H = 5120
SLICES = H // 32
# ---------------------------------------------------------------- main pass

KA2 = KArgs(
    ("residual", "ptr"),
    ("x", "ptr"),
    ("post", "ptr"),
    ("comb", "ptr"),
    ("pre", "ptr"),
    ("fn_hl", "ptr"),
    ("r_out", "ptr"),
    ("x1", "ptr"),
    ("partial", "ptr"),
    ("m", "u32"),
    ("tg", "u32"),
)
NST = 14  # vector memory instructions after a tile's loads before the next wait (its stores)
# yq: + norm_w (attn_norm.weight bf16), xq / xs (wqkv's x, MXFP8 frag, 256 tokens)
KA2Q = KArgs(*KA2.args, ("norm_w", "ptr"), ("xq", "ptr"), ("xs", "ptr"))


def fn_hl(fn):
    """Hc fn [24, 4H] f32 → source_v2's B operands, bf16 hi = bf16(fn) and lo =
    bf16(fn − hi) (aiter's split), in fragment order:
    `[slice cs][q = 2·stream + n-block][hi, lo][lane l][8 bf16]`: row l % 16 + 16 nb
    (rows ≥ 24 zero), columns 32 cs + 8 (l / 16) .. + 7 of stream q / 2."""
    import torch

    f = torch.zeros(32, 4 * H, dtype=torch.float32, device=fn.device)
    f[: fn.shape[0]] = fn.float()
    hi = f.to(torch.bfloat16)
    lo = (f - hi.float()).to(torch.bfloat16)
    out = []
    for t in (hi, lo):
        # [nb, r, stream j, cs, g, 8] → [cs, j, nb, (g, r), 8]
        v = (
            t.view(2, 16, 4, SLICES, 4, 8)
            .permute(3, 2, 0, 4, 1, 5)
            .reshape(SLICES, 4, 2, 64, 8)
        )
        out.append(v)
    return (
        torch.stack(out, 3).reshape(SLICES, 8, 2, 64, 8).contiguous().view(torch.int32)
    )


def tg_for(m: int) -> int:
    """Token-tile groups: waves per column slice (each takes tiles g, g + tg, ..).
    6 was best or within 0.1 µs of it at M 64–256."""
    return min((m + 15) // 16, 6)


def source_v2(name: str = "ffn_seam2", post_stride: int = 16, yq: bool | str = False):
    """FFN seam main pass, scheduled as follows:
    - fn split into bf16 hi / lo once on the host (`fn_hl`, fragment order:
      16 contiguous 1 KB loads per wave instead of 16 scattered ones + 192 VALU
      per task);
    - wave gw = (column slice cs = gw % 160, group g = gw / 160) keeps its fn
      slice in VGPRs and walks token tiles g, g + tg, .. (tg a kernarg; grid
      40·tg CTAs of 4 waves);
    - the next tile's loads (11) go out before this tile's math: two load
      buffers (v12.., v56..), tiles moved by `soffset` (part of the raw range
      check on gfx950, so rows ≥ M still load 0).
    post_stride 64: post`[t][h]` at byte 64 t + 16 h, i.e.
    the Engram gate all-gather's output as is (comb's offsets, 4 dword loads).
    yq (the front without the fin launch): also y' = bf16(x1 ·
    attn_norm.weight) → MXFP8 (quant_block's rule) in gemv8's x frag layout
    (wqkv's x; KA2Q). The token's rstd = rsq(Σx1² / H + eps) is not applied:
    wqkv's two outputs are RMS-normalised again, so the consumers fold it in.
    The wave's slice cs is one MXFP8 block; lane (r, g) holds codes 8g .. 8g + 7
    of token r: frag bytes ((j·40 + cs/4)·2 + (cs/2)%2)·1 KB + (r + 32 (cs%2))·16
    + 8 (g%2) + 256 (g/2), scale (j·40 + cs/4)·64 + 16 (cs%4) + r.
    yq "ug" (the FFN front without fin): norm_w = ffn_norm.weight,
    the MXFP8 in moe_ug1's layout (xq [M, H] row-major, xs [M, 192]: token t's
    block b at t·192 + kb·48 + w·12 + kk, k0 = b/4, kb = b%4, w = k0/10, kk = k0%10) for
    the shared gate_up and stage 1, and bf16 y' instead of x1 in `x1` (the
    router's input; nothing reads x1 then). rstd is applied by p3 / stage 1.
    yq "fy" (the index layers' front): the frag MXFP8 as yq, plus bf16 y'
    instead of x1 in `x1`, as "ug" (the compressor GEMM and the indexer's
    weights_proj read it; the front reads no x1).
    Global stores, so xq / xs must hold whole 16-token tiles (MAX_M rows)."""
    assert post_stride in (16, 64)
    assert yq in (False, True, "ug", "fy"), yq
    ug = yq == "ug"
    ystore = yq in ("ug", "fy")  # bf16 y' into x1's slot
    nst = NST + 2 if yq else NST
    c = []
    A = c.append
    A(
        "  s_load_dwordx16 s[4:19], s[0:1], 0x0"
    )  # residual x post comb pre fn_hl r_out x1
    A("  s_load_dwordx4 s[20:23], s[0:1], 0x40")  # partial, m s22, tg s23
    if yq:
        A(f"  s_load_dwordx2 s[100:101], s[0:1], {KA2Q.off['norm_w']:#x}")
    A("  v_readfirstlane_b32 s29, v0")
    A("  s_lshr_b32 s29, s29, 6")  # wave
    A("  v_and_b32 v0, 63, v0")  # lane
    A("  v_and_b32 v1, 15, v0")  # r
    A("  v_lshrrev_b32 v2, 4, v0")  # g (lane group)
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_mul_i32 s24, s22, {4 * 2 * H}")
    A(f"  s_mul_i32 s25, s22, {2 * H}")
    A("  s_lshl_b32 s26, s22, 4")
    A("  s_lshl_b32 s27, s22, 6")
    A(f"  s_mul_i32 s28, s22, {SLICES * 128}")
    c += rsrc(32, 4, nbytes_sgpr=24)
    c += rsrc(36, 6, nbytes_sgpr=25)
    c += rsrc(40, 8, nbytes_sgpr=26 if post_stride == 16 else 27)
    c += rsrc(44, 10, nbytes_sgpr=27)
    c += rsrc(48, 12, nbytes_sgpr=26)
    c += rsrc(52, 14, nbytes=SLICES * 16384)
    c += rsrc(56, 16, nbytes_sgpr=24)
    c += rsrc(60, 18, nbytes_sgpr=25)
    c += rsrc(64, 20, nbytes_sgpr=28)
    A("  s_lshl_b32 s30, s2, 2")
    A("  s_add_u32 s31, s30, s29")  # gw
    A("  s_add_u32 s90, s22, 15")
    A("  s_lshr_b32 s90, s90, 4")  # nt
    A("  s_mov_b32 s94, 0xcccccccd")
    A("  s_lshr_b32 s92, s31, 5")
    A("  s_mul_hi_u32 s92, s92, s94")
    A("  s_lshr_b32 s92, s92, 2")  # g = gw / 160 = first tile
    A(f"  s_mul_i32 s91, s92, {SLICES}")
    A("  s_sub_u32 s91, s31, s91")  # cs
    A("  s_cmp_ge_u32 s92, s23")
    A("  s_cbranch_scc1 .Lend")
    A("  s_cmp_ge_u32 s92, s90")
    A("  s_cbranch_scc1 .Lend")
    # tile-0 VGPR offsets (t = r)
    A("  s_lshl_b32 s94, s91, 5")  # 32 cs
    A("  v_lshl_add_u32 v4, v2, 3, s94")  # c0
    A(f"  v_mul_u32_u24 v5, {4 * 2 * H}, v1")
    A("  v_lshl_add_u32 v5, v4, 1, v5")  # R / r_out
    A(f"  v_mul_u32_u24 v6, {2 * H}, v1")
    A("  v_lshl_add_u32 v6, v4, 1, v6")  # x / x1
    A("  v_lshlrev_b32 v7, 4, v1")  # post / pre
    A("  v_lshlrev_b32 v8, 6, v1")  # comb
    A("  s_lshl_b32 s94, s91, 14")
    A("  v_lshl_add_u32 v9, v0, 4, s94")  # fn_hl: cs·16 KB + lane·16
    A("  v_lshlrev_b32 v10, 2, v2")  # 4 g
    A(f"  v_mul_u32_u24 v10, {SLICES}, v10")
    A("  v_add_u32 v10, s91, v10")
    A("  v_lshlrev_b32 v10, 7, v10")
    A(
        "  v_lshl_add_u32 v10, v1, 2, v10"
    )  # partial rows 4g + i (+ i·160·128 in soffset)
    A(f"  v_mul_u32_u24 v11, {SLICES}, v1")
    A("  v_add_u32 v11, s91, v11")
    A("  v_lshlrev_b32 v11, 7, v11")  # sq slots of row t
    if yq:
        A("  v_lshlrev_b32 v244, 1, v4")
        A(
            "  global_load_dwordx4 v[100:103], v244, s[100:101]"
        )  # norm_w of the lane's 8 columns
        A(
            f"  s_load_dwordx2 s[98:99], s[0:1], {KA2Q.off['xq']:#x}"
        )  # xq, xs: global store bases
        A(f"  s_load_dwordx2 s[100:101], s[0:1], {KA2Q.off['xs']:#x}")
        if ug:
            A(f"  v_mul_u32_u24 v246, {H}, v1")
            A("  v_add_u32 v246, v246, v4")  # code offset r·H + c0 (+ 16 H per tile)
            # scale byte (t, cs): t·192 + kb·48 + w·12 + kk, k0 = cs / 4, kb = cs % 4, w = k0 / 10, kk = k0 % 10
            A("  s_lshr_b32 s94, s91, 2")  # k0
            A("  s_mul_hi_u32 s95, s94, 0xcccccccd")
            A("  s_lshr_b32 s95, s95, 3")  # w
            A("  s_mul_i32 s96, s95, 10")
            A("  s_sub_u32 s94, s94, s96")  # kk
            A("  s_mul_i32 s95, s95, 12")
            A("  s_add_u32 s94, s94, s95")
            A("  s_and_b32 s95, s91, 3")
            A("  s_mul_i32 s95, s95, 48")
            A("  s_add_u32 s94, s94, s95")
            A("  v_mul_u32_u24 v247, 192, v1")
            A("  v_add_u32 v247, s94, v247")  # scale offset (+ 16·192 per tile)
        else:
            A("  s_lshr_b32 s94, s91, 2")  # kg = cs / 4
            A("  s_lshl_b32 s94, s94, 1")
            A("  s_bfe_u32 s95, s91, 0x10001")  # (cs / 2) % 2
            A("  s_add_u32 s94, s94, s95")
            A("  s_lshl_b32 s94, s94, 10")
            A("  s_and_b32 s95, s91, 1")
            A("  s_lshl_b32 s95, s95, 9")
            A("  s_add_u32 s94, s94, s95")  # scalar part of the code offset
            A("  v_and_b32 v244, 1, v2")
            A("  v_lshlrev_b32 v244, 3, v244")  # 8 (g % 2)
            A("  v_lshrrev_b32 v245, 1, v2")
            A("  v_lshl_add_u32 v244, v245, 8, v244")  # + 256 (g / 2)
            A("  v_lshl_add_u32 v244, v1, 4, v244")  # + 16 r
            A("  v_add_u32 v246, s94, v244")  # code offset (+ j·80 KB per tile)
            A("  s_lshr_b32 s94, s91, 2")
            A("  s_lshl_b32 s94, s94, 6")
            A("  s_and_b32 s95, s91, 3")
            A("  s_lshl_b32 s95, s95, 4")
            A("  s_add_u32 s94, s94, s95")
            A("  v_add_u32 v247, s94, v1")  # scale offset (+ j·2560 per tile)
    # fn hi / lo → v[120 + 8q .. +3] / v[124 + 8q .. +3]
    for q in range(8):
        A(f"  s_mov_b32 s94, {q * 2048}")
        A(
            f"  buffer_load_dwordx4 v[{120 + 8 * q}:{123 + 8 * q}], v9, s[52:55], s94 offen"
        )
        A(
            f"  buffer_load_dwordx4 v[{124 + 8 * q}:{127 + 8 * q}], v9, s[52:55], s94 offen offset:1024"
        )

    def set_offsets(S, k):
        """Tile k (SGPR) → set S: R0..R3, X, P, C, Part0..3."""
        A(f"  s_mul_i32 s{S}, s{k}, {16 * 4 * 2 * H}")
        for h in range(1, 4):
            A(f"  s_add_u32 s{S + h}, s{S}, {h * 2 * H}")
        A(f"  s_mul_i32 s{S + 4}, s{k}, {16 * 2 * H}")
        A(f"  s_lshl_b32 s{S + 5}, s{k}, 8")
        A(f"  s_lshl_b32 s{S + 6}, s{k}, 10")
        A(f"  s_mul_i32 s{S + 7}, s{k}, {16 * SLICES * 128}")
        for i in range(1, 4):
            A(f"  s_add_u32 s{S + 7 + i}, s{S + 7}, {i * SLICES * 128}")

    def loads(B, S):
        for h in range(4):
            A(
                f"  buffer_load_dwordx4 v[{B + 4 * h}:{B + 3 + 4 * h}], v5, s[32:35], s{S + h} offen"
            )
        A(f"  buffer_load_dwordx4 v[{B + 16}:{B + 19}], v6, s[36:39], s{S + 4} offen")
        if post_stride == 16:
            A(
                f"  buffer_load_dwordx4 v[{B + 20}:{B + 23}], v7, s[40:43], s{S + 5} offen"
            )
        else:
            for h in range(4):
                A(
                    f"  buffer_load_dword v{B + 20 + h}, v8, s[40:43], s{S + 6} offen offset:{16 * h}"
                )
        for q in range(4):
            A(
                f"  buffer_load_dwordx4 v[{B + 24 + 4 * q}:{B + 27 + 4 * q}], v8, s[44:47], s{S + 6} offen offset:{16 * q}"
            )
        A(f"  buffer_load_dwordx4 v[{B + 40}:{B + 43}], v7, s[48:51], s{S + 5} offen")

    def compute(B, S, tile):
        """The task's math on buffer B (R_h at B + 4h, x B + 16, post B + 20, comb B + 24, pre B + 40); stores with set S;
        tile: its SGPR (yq's offsets)."""
        A("  v_mov_b32 v234, 0")
        A("  v_mov_b32 v235, 0")
        for q in range(236, 244):
            A(f"  v_mov_b32 v{q}, 0")
        for h in range(5):
            src, dst = (B + 4 * h, 186 + 8 * h) if h < 4 else (B + 16, 218)
            for k in range(4):
                A(f"  v_lshlrev_b32 v{dst + 2 * k}, 16, v{src + k}")
                A(f"  v_and_b32 v{dst + 2 * k + 1}, 0xffff0000, v{src + k}")
        for j in range(4):
            for i in range(8):
                a = 226 + i
                A(f"  v_mul_f32 v{a}, v{B + 20 + j}, v{218 + i}")
                for h in range(4):
                    A(f"  v_fmac_f32 v{a}, v{B + 24 + 4 * h + j}, v{186 + 8 * h + i}")
            for k in range(4):
                A(
                    f"  v_cvt_pk_bf16_f32 v{B + 4 * j + k}, v{226 + 2 * k}, v{227 + 2 * k}"
                )
        for j in range(4):
            for k in range(4):
                A(f"  v_lshlrev_b32 v{186 + 8 * j + 2 * k}, 16, v{B + 4 * j + k}")
                A(f"  v_and_b32 v{187 + 8 * j + 2 * k}, 0xffff0000, v{B + 4 * j + k}")
        for j in range(4):
            for i in range(8):
                A(f"  v_fmac_f32 v234, v{186 + 8 * j + i}, v{186 + 8 * j + i}")
        A("  s_nop 2")
        for j in range(4):
            for part in (0, 4):
                for nb in range(2):
                    b = 120 + 8 * (2 * j + nb) + part
                    d = 236 + 4 * nb
                    A(
                        f"  v_mfma_f32_16x16x32_bf16 v[{d}:{d + 3}], v[{B + 4 * j}:{B + 3 + 4 * j}], v[{b}:{b + 3}], "
                        f"v[{d}:{d + 3}]"
                    )
        for i in range(8):
            a = 226 + i
            A(f"  v_mul_f32 v{a}, v{B + 40}, v{186 + i}")
            for j in range(1, 4):
                A(f"  v_fmac_f32 v{a}, v{B + 40 + j}, v{186 + 8 * j + i}")
        for k in range(4):
            A(f"  v_cvt_pk_bf16_f32 v{B + 16 + k}, v{226 + 2 * k}, v{227 + 2 * k}")
        if not ystore:
            A(
                f"  buffer_store_dwordx4 v[{B + 16}:{B + 19}], v6, s[60:63], s{S + 4} offen"
            )
        for j in range(4):
            A(
                f"  buffer_store_dwordx4 v[{B + 4 * j}:{B + 3 + 4 * j}], v5, s[56:59], s{S + j} offen"
            )
        for k in range(4):
            A(f"  v_lshlrev_b32 v184, 16, v{B + 16 + k}")
            A(f"  v_and_b32 v185, 0xffff0000, v{B + 16 + k}")
            A("  v_fmac_f32 v235, v184, v184")
            A("  v_fmac_f32 v235, v185, v185")
        if yq:  # y' = bf16(x1 · w) → f32 v[104:112], its |max| v114
            for k in range(4):
                A(f"  v_lshlrev_b32 v{104 + 2 * k}, 16, v{B + 16 + k}")
                A(f"  v_and_b32 v{105 + 2 * k}, 0xffff0000, v{B + 16 + k}")
                A(f"  v_lshlrev_b32 v112, 16, v{100 + k}")
                A(f"  v_and_b32 v113, 0xffff0000, v{100 + k}")
                A(f"  v_mul_f32 v{104 + 2 * k}, v{104 + 2 * k}, v112")
                A(f"  v_mul_f32 v{105 + 2 * k}, v{105 + 2 * k}, v113")
                A(f"  v_cvt_pk_bf16_f32 v112, v{104 + 2 * k}, v{105 + 2 * k}")
                A(f"  v_lshlrev_b32 v{104 + 2 * k}, 16, v112")
                A(f"  v_and_b32 v{105 + 2 * k}, 0xffff0000, v112")
            if ystore:  # bf16 y' → x1's slot (ug: the router's input)
                for k in range(4):
                    A(f"  v_cvt_pk_bf16_f32 v{248 + k}, v{104 + 2 * k}, v{105 + 2 * k}")
                A(f"  buffer_store_dwordx4 v[248:251], v6, s[60:63], s{S + 4} offen")
            A("  v_max_f32 v114, |v104|, |v105|")
            for i in range(2, 8):
                A(f"  v_max_f32 v114, v114, |v{104 + i}|")
        for d in (16, 32):
            A(f"  v_xor_b32 v184, {d}, v0")
            A("  v_lshlrev_b32 v184, 2, v184")
            A("  ds_bpermute_b32 v185, v184, v234")
            A("  ds_bpermute_b32 v186, v184, v235")
            if yq:
                A("  ds_bpermute_b32 v115, v184, v114")
            A("  s_waitcnt lgkmcnt(0)")
            A("  v_add_f32 v234, v234, v185")
            A("  v_add_f32 v235, v235, v186")
            if yq:
                A("  v_max_f32 v114, v114, v115")
        if (
            yq
        ):  # MXFP8 of the token's 32 columns (the 4 lane groups): E8M0 as quant_block
            A("  v_frexp_exp_i32_f32 v116, v114")
            A("  v_frexp_mant_f32 v117, v114")
            A("  v_cmp_lt_f32 vcc, 0x3f600000, v117")
            A("  v_addc_co_u32 v116, vcc, -9, v116, vcc")
            A("  v_max_i32 v116, 0xffffff81, v116")
            A("  v_min_i32 v116, 0x7f, v116")
            A("  v_sub_u32 v117, 0, v116")
            for i in range(8):
                A(f"  v_ldexp_f32 v{104 + i}, v{104 + i}, v117")
            for k in range(2):
                A(f"  v_cvt_pk_fp8_f32 v{118 + k}, v{104 + 4 * k}, v{105 + 4 * k}")
                A(
                    f"  v_cvt_pk_fp8_f32 v{118 + k}, v{106 + 4 * k}, v{107 + 4 * k} op_sel:[0,0,1]"
                )
            A(f"  s_mul_i32 s94, s{tile}, {16 * H if ug else 40 * 2048}")
            A(f"  s_mul_i32 s95, s{tile}, {16 * 192 if ug else 40 * 64}")
            A("  v_add_u32 v112, s94, v246")
            A("  v_add_u32 v113, s95, v247")
            A("  global_store_dwordx2 v112, v[118:119], s[98:99]")
            A("  v_add_u32 v116, 0x7f, v116")
        A("  s_mov_b64 exec, 0xffff")
        A(f"  buffer_store_dwordx2 v[234:235], v11, s[64:67], s{S + 7} offen offset:96")
        if yq:
            A("  global_store_byte v113, v116, s[100:101]")
        A("  s_mov_b64 exec, -1")
        A("  s_nop 7")
        A("  s_nop 7")
        A("  s_nop 7")
        for i in range(4):
            A(f"  buffer_store_dword v{236 + i}, v10, s[64:67], s{S + 7 + i} offen")
        A("  v_cmp_gt_u32 vcc, 8, v1")
        A("  s_and_saveexec_b64 s[96:97], vcc")
        for i in range(4):
            A(
                f"  buffer_store_dword v{240 + i}, v10, s[64:67], s{S + 7 + i} offen offset:64"
            )
        A("  s_mov_b64 exec, s[96:97]")

    SA, SB, BA, BB = 68, 79, 12, 56
    # s92 = tile in buffer A, s93 = tile in buffer B
    set_offsets(SA, 92)
    loads(BA, SA)
    A("  s_waitcnt vmcnt(0)")
    A(".LloopA:")
    A("  s_add_u32 s93, s92, s23")
    A("  s_cmp_lt_u32 s93, s90")
    A("  s_cbranch_scc0 .LnoB")
    set_offsets(SB, 93)
    loads(BB, SB)
    A(".LnoB:")
    compute(BA, SA, 92)
    A("  s_cmp_lt_u32 s93, s90")
    A("  s_cbranch_scc0 .Lend")
    A(f"  s_waitcnt vmcnt({nst})")
    A("  s_add_u32 s92, s93, s23")
    A("  s_cmp_lt_u32 s92, s90")
    A("  s_cbranch_scc0 .LnoA")
    set_offsets(SA, 92)
    loads(BA, SA)
    A(".LnoA:")
    compute(BB, SB, 93)
    A("  s_cmp_lt_u32 s92, s90")
    A("  s_cbranch_scc0 .Lend")
    A(f"  s_waitcnt vmcnt({nst})")
    A("  s_branch .LloopA")
    A(".Lend:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_endpgm")
    ka = KA2Q if yq else KA2
    src = kernel_source(
        name,
        c,
        ka,
        lds=0,
        vgprs=252 if ystore else 248 if yq else 244,
        agprs=0,
        sgprs=102,
        block=256,
    )
    return dict(name=name, src=src, ka=ka)


KA_FIN = KArgs(
    ("partial", "ptr"),
    ("x1", "ptr"),
    ("norm_w", "ptr"),
    ("scale", "ptr"),
    ("base", "ptr"),
    ("post_out", "ptr"),
    ("comb_out", "ptr"),
    ("pre_out", "ptr"),
    ("y", "ptr"),
    ("xq", "ptr"),
    ("xs", "ptr"),
    ("m", "u32"),
    ("rms_eps", "u32"),
    ("hc_eps", "u32"),
    ("post_alpha", "u32"),
    ("norm_eps", "u32"),
)
SINKHORN = 20


def f32(v: float) -> str:
    return hex(struct.unpack("<I", struct.pack("<f", v))[0])


def quant_block(
    A,
    y: int,
    scratch: int,
    t: str,
    b: str,
    xq_rsrc: int,
    xs_rsrc: int,
) -> None:
    """MXFP8 of the 32 floats in v[y:y+32] (block b of token t, VGPRs t and b),
    E8M0 = 2^ceil(log2(amax/448)) as moe_prep.py, in gemv8's frag layout (codes
    → xq, scale byte → xs). scratch must be even; uses v[scratch:scratch+17]."""
    a, e, mt, ng, off, q = (
        scratch,
        scratch + 1,
        scratch + 2,
        scratch + 3,
        scratch + 4,
        scratch + 6,
    )  # q even
    A(f"  v_max_f32 v{a}, |v{y}|, |v{y + 1}|")
    for i in range(2, 32):
        A(f"  v_max_f32 v{a}, v{a}, |v{y + i}|")
    A(f"  v_frexp_exp_i32_f32 v{e}, v{a}")
    A(f"  v_frexp_mant_f32 v{mt}, v{a}")
    A(f"  v_cmp_lt_f32 vcc, 0x3f600000, v{mt}")
    A(f"  v_addc_co_u32 v{e}, vcc, -9, v{e}, vcc")
    A(f"  v_max_i32 v{e}, 0xffffff81, v{e}")
    A(f"  v_min_i32 v{e}, 0x7f, v{e}")
    A(f"  v_sub_u32 v{ng}, 0, v{e}")
    for i in range(32):
        A(f"  v_ldexp_f32 v{y + i}, v{y + i}, v{ng}")
    for k in range(8):
        A(f"  v_cvt_pk_fp8_f32 v{q + k}, v{y + 4 * k}, v{y + 4 * k + 1}")
        A(
            f"  v_cvt_pk_fp8_f32 v{q + k}, v{y + 4 * k + 2}, v{y + 4 * k + 3} op_sel:[0,0,1]"
        )
    # gemv8's frag layout (K tiles of 128, KGT = H / 128): codes of block b at
    # ((j·KGT + b/4)·2 + (b/2)%2)·1 KB + (r + 32 (b%2))·16 and + 256; scale at
    # (j·KGT + b/4)·64 + r + 16 (b%4)
    kgt = H // 128
    jr, kg, tmp = scratch + 14, scratch + 15, scratch + 16
    A(f"  v_lshrrev_b32 v{jr}, 4, v{t}")
    A(f"  v_mul_u32_u24 v{jr}, {kgt}, v{jr}")
    A(f"  v_lshrrev_b32 v{kg}, 2, v{b}")
    A(f"  v_add_u32 v{kg}, v{jr}, v{kg}")  # j·KGT + kg
    A(f"  v_lshlrev_b32 v{off}, 11, v{kg}")
    A(f"  v_bfe_u32 v{tmp}, v{b}, 1, 1")
    A(f"  v_lshl_add_u32 v{off}, v{tmp}, 10, v{off}")
    A(f"  v_and_b32 v{tmp}, 1, v{b}")
    A(f"  v_lshl_add_u32 v{off}, v{tmp}, 9, v{off}")
    A(f"  v_and_b32 v{tmp}, 15, v{t}")
    A(f"  v_lshl_add_u32 v{off}, v{tmp}, 4, v{off}")
    A(
        f"  buffer_store_dwordx4 v[{q}:{q + 3}], v{off}, s[{xq_rsrc}:{xq_rsrc + 3}], 0 offen"
    )
    A(
        f"  buffer_store_dwordx4 v[{q + 4}:{q + 7}], v{off}, s[{xq_rsrc}:{xq_rsrc + 3}], 0 offen offset:256"
    )
    A(f"  v_lshlrev_b32 v{off}, 6, v{kg}")
    A(f"  v_add_u32 v{off}, v{tmp}, v{off}")
    A(f"  v_and_b32 v{tmp}, 3, v{b}")
    A(f"  v_lshl_add_u32 v{off}, v{tmp}, 4, v{off}")
    A(f"  v_add_u32 v{e}, 0x7f, v{e}")
    A(f"  buffer_store_byte v{e}, v{off}, s[{xs_rsrc}:{xs_rsrc + 3}], 0 offen")


def gates_sinkhorn(A, tok: str = "s2", base_loaded: bool = False) -> None:
    """Fin's step 2 (also the mixes wave, `mix_wave`): lane L < 32 holds value
    v(L) in v31 (v4 = v(L), v7 = 4 v(L), v1 = lane), Σ R'² in v32 → rstd, the
    pre' / post' gates and comb' by Sinkhorn → pre / post / comb of token
    `tok`. SGPRs as fin: rms_eps s27, hc_eps s28, alpha s29, scale s72..74,
    constants s86..s89; base s[44:47], post s[48:51], comb s[52:55], pre
    s[56:59]. Uses v9, v34..v46. base_loaded: v34 already holds base[v(L)]."""
    if not base_loaded:
        A("  buffer_load_dword v34, v7, s[44:47], 0 offen")  # base[v] (0 past 24)
    A("  v_mov_b32 v9, s27")
    A("  v_fma_f32 v32, v32, s86, v9")
    A("  v_rsq_f32 v32, v32")  # rstd
    A("  s_nop 0")
    A("  v_mul_f32 v35, v31, v32")  # mixes · rstd
    A("  v_mov_b32 v36, s74")  # scale per lane: s0 (v < 4), s1 (v < 8), s2
    A("  v_mov_b32 v9, s73")
    A("  v_cmp_gt_u32 vcc, 8, v4")
    A("  v_cndmask_b32 v36, v36, v9, vcc")
    A("  v_mov_b32 v9, s72")
    A("  v_cmp_gt_u32 vcc, 4, v4")
    A("  v_cndmask_b32 v36, v36, v9, vcc")
    if not base_loaded:
        A("  s_waitcnt vmcnt(0)")
    A("  v_fma_f32 v37, v35, v36, v34")  # z = mixes · rstd · s + base
    # sigmoid(z) = 1 / (1 + 2^(−z log2 e))
    A("  v_mul_f32 v38, s89, v37")
    A("  v_exp_f32 v38, v38")
    A("  s_nop 0")
    A("  v_add_f32 v38, 1.0, v38")
    A("  v_rcp_f32 v38, v38")
    A("  s_nop 0")
    A("  v_add_f32 v39, s28, v38")  # pre' = σ + hc_eps
    A("  v_mul_f32 v40, s29, v38")  # post' = σ · alpha
    A("  v_lshlrev_b32 v41, 2, v1")
    A(f"  v_mov_b32 v42, {tok}")
    A("  v_lshl_add_u32 v41, v42, 4, v41")  # t·16 + 4 L
    A("  v_subrev_u32 v46, 16, v41")  # post offset, before exec narrows to lanes 0..3
    A("  s_mov_b64 exec, 0xf")
    A("  buffer_store_dword v39, v41, s[56:59], 0 offen")
    A("  s_mov_b64 exec, 0xf0")
    A("  buffer_store_dword v40, v46, s[48:51], 0 offen")
    A("  s_mov_b64 exec, -1")

    # Sinkhorn on lanes 16..31 (row r = (L−16)/4, column c = L%4)
    def dpp(op, d, a, b, ctl):
        A("  s_nop 1")
        A(f"  v_{op}_f32_dpp v{d}, v{a}, v{b} {ctl} row_mask:0xf bank_mask:0xf")

    QA, QB, R4, R8 = (
        "quad_perm:[1,0,3,2]",
        "quad_perm:[2,3,0,1]",
        "row_ror:4",
        "row_ror:8",
    )

    def rownorm(p, first=False):
        dpp("add", 44, p, p, QA)
        dpp("add", 44, 44, 44, QB)
        if not first:
            A("  v_add_f32 v44, s28, v44")
        A("  v_rcp_f32 v44, v44")
        A("  s_nop 0")
        A(f"  v_mul_f32 v{p}, v{p}, v44")
        if first:
            A(f"  v_add_f32 v{p}, s28, v{p}")

    def colnorm(p):
        dpp("add", 44, p, p, R4)
        dpp("add", 44, 44, 44, R8)
        A("  v_add_f32 v44, s28, v44")
        A("  v_rcp_f32 v44, v44")
        A("  s_nop 0")
        A(f"  v_mul_f32 v{p}, v{p}, v44")

    dpp("max", 43, 37, 37, QA)
    dpp("max", 43, 43, 43, QB)
    A("  v_sub_f32 v45, v37, v43")
    A("  v_mul_f32 v45, s88, v45")
    A("  v_exp_f32 v45, v45")
    A("  s_nop 0")
    rownorm(45, first=True)
    colnorm(45)
    for _ in range(SINKHORN - 1):
        rownorm(45)
        colnorm(45)
    A("  v_subrev_u32 v46, 16, v1")
    A("  v_lshlrev_b32 v46, 2, v46")
    A("  v_lshl_add_u32 v46, v42, 6, v46")  # t·64 + 4 (L − 16)
    A(
        "  s_mov_b32 exec_lo, 0xffff0000"
    )  # lanes 16..31 only (a b64 literal would sign-extend)
    A("  s_mov_b32 exec_hi, 0")
    A("  buffer_store_dword v45, v46, s[52:55], 0 offen")
    A("  s_mov_b64 exec, -1")


# the mixes wave: fin's partial sums, gates and Sinkhorn for one token at a time, as a spare
# wave of another kernel (gemv8.source(mix_wave=True), kernargs appended with prefix mx_)
KA_MIX = KArgs(
    ("partial", "ptr"),
    ("scale", "ptr"),
    ("base", "ptr"),
    ("post_out", "ptr"),
    ("comb_out", "ptr"),
    ("pre_out", "ptr"),
    ("rstd", "ptr"),
    ("m", "u32"),
    ("rms_eps", "u32"),
    ("hc_eps", "u32"),
    ("post_alpha", "u32"),
    ("norm_eps", "u32"),
    ("dbg", "ptr"),
)
MIX_LDS = 128


def mix_wave(
    ka_base: int,
    stride: int,
    lds: int,
    first: str = "s2",
    slot: str = "",
    bar_m: str = "",
    nbar: int = 0,
    rstd_store: bool = True,
) -> list[str]:
    """One wave doing fin's work except y: tokens t = `first` (an SGPR set by
    the caller, e.g. workgroup id + grid · spare-wave index), + stride, .. < m.
    The 160 partial rows of t in 20 contiguous 1 KB loads, all in flight at
    once (lane l: row 8k + l / 8, floats 4 (l % 8) ..), summed per lane in k
    order, then over the 8 row groups (row_ror 8, permlane 16 / 32 swaps);
    lanes 0..7 → LDS (MIX_LDS bytes at `lds` + SGPR `slot`) → lane L reads value v(L) as
    fin's wave 0, then `gates_sinkhorn` → pre / post / comb, and rstd_n =
    rsq(Σx1² / H + norm_eps) → rstd[t] (f32). Not bitwise = fin (summation
    order). KA_MIX's arguments at kernarg offset ka_base. Uses v0..v134,
    s0..s91 (`first`, `slot` and `bar_m` are read before s4). bar_m: the
    host kernel's barriers (s_barrier waits for every live wave of the
    workgroup, so a spare wave that skips them holds the others at their first
    one until it ends): with its loads in flight, the wave steps through
    2·⌈m / 64⌉ + 2 barriers (gemv8 mode k with split-K; m in SGPR bar_m), then
    sums and runs the gates while the last arrivers reduce. nbar: instead, a fixed
    number of host barriers to step through (e.g. p3's one). rstd_store False:
    rstd[t] is the host's to write (K2's p3)."""
    c = []
    A = c.append
    LB, ACC, T = 50, 130, 134
    A(f"  s_mov_b32 s90, {first}")  # t
    A(
        f"  s_load_dwordx16 s[4:19], s[0:1], {ka_base:#x}"
    )  # partial scale base post comb pre rstd, m s18
    A(
        f"  s_load_dwordx4 s[20:23], s[0:1], {ka_base + 0x40:#x}"
    )  # hc_eps s20, alpha s21, norm_eps s22
    A("  v_and_b32 v1, 63, v0")  # lane
    A("  v_lshlrev_b32 v5, 4, v1")  # 16 l: load offset, LDS write offset (lanes < 8)
    A(f"  v_mov_b32 v8, {slot or 0}")  # the wave's LDS slot
    A("  v_add_u32 v10, v5, v8")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_cmp_ge_u32 s90, s18")
    A("  s_cbranch_scc1 .Lmx_end")
    A(f"  s_mul_i32 s77, s18, {SLICES * 128}")
    c.extend(rsrc(32, 4, nbytes_sgpr=77))  # partial
    c.extend(rsrc(44, 8, nbytes=24 * 4))  # base
    # v(j), j = L % 32: j (j < 8), j + 16 (8 ≤ j < 16), j − 8 (16 ≤ j < 32)
    A("  v_and_b32 v3, 31, v1")
    A("  v_add_u32 v4, 16, v3")
    A("  v_cmp_gt_u32 vcc, 8, v3")
    A("  v_cndmask_b32 v4, v4, v3, vcc")
    A("  v_subrev_u32 v6, 8, v3")
    A("  v_cmp_le_u32 vcc, 16, v3")
    A("  v_cndmask_b32 v4, v4, v6, vcc")  # v(j)
    A("  v_lshlrev_b32 v7, 2, v4")
    A(
        "  buffer_load_dword v34, v7, s[44:47], 0 offen"
    )  # base[v] (0 past 24), for every token

    def loads():
        A(f"  s_mul_i32 s81, s90, {SLICES * 128}")
        for k in range(20):
            A(f"  s_add_u32 s82, s81, {1024 * k}")
            A(
                f"  buffer_load_dwordx4 v[{LB + 4 * k}:{LB + 4 * k + 3}], v5, s[32:35], s82 offen"
            )

    loads()
    A("  s_mov_b32 s27, s19")  # rms_eps
    A("  s_mov_b32 s28, s20")  # hc_eps
    A("  s_mov_b32 s29, s21")  # alpha
    A("  s_mov_b32 s30, s22")  # norm_eps
    A("  s_load_dwordx2 s[72:73], s[6:7], 0x0")  # scale[0:2]
    A("  s_load_dword s74, s[6:7], 0x8")
    A("  s_lshl_b32 s79, s18, 4")
    A("  s_lshl_b32 s80, s18, 6")
    A("  s_lshl_b32 s78, s18, 2")
    c.extend(rsrc(48, 10, nbytes_sgpr=79))  # post_out
    c.extend(rsrc(52, 12, nbytes_sgpr=80))  # comb_out
    c.extend(rsrc(56, 14, nbytes_sgpr=79))  # pre_out
    c.extend(rsrc(60, 16, nbytes_sgpr=78))  # rstd
    A(f"  s_mov_b32 s86, {f32(1.0 / (4 * H))}")
    A(f"  s_mov_b32 s87, {f32(1.0 / H)}")
    A(f"  s_mov_b32 s88, {f32(1.4426950408889634)}")  # log2 e
    A(f"  s_mov_b32 s89, {f32(-1.4426950408889634)}")
    A("  v_add_u32 v11, v7, v8")
    A("  v_mov_b32 v12, 0")
    if bar_m:
        A(f"  s_add_u32 s83, {bar_m}, 63")
        A("  s_lshr_b32 s83, s83, 6")
        A("  s_lshl_b32 s83, s83, 1")
        A("  s_add_u32 s83, s83, 2")
        A(".Lmx_bar:")
        A("  s_barrier")
        A("  s_sub_u32 s83, s83, 1")
        A("  s_cmp_lg_u32 s83, 0")
        A("  s_cbranch_scc1 .Lmx_bar")
    for _ in range(nbar):
        A("  s_barrier")
    A(".Lmx_tok:")
    A("  s_waitcnt lgkmcnt(0)")
    for k in range(20):
        # sum as the loads land (in order; later tokens: their loads landed in gates_sinkhorn's
        # wait, only its stores may be outstanding, so the count is conservative)
        A(f"  s_waitcnt vmcnt({19 - k})")
        for i in range(4):
            if k == 0:
                A(f"  v_mov_b32 v{ACC + i}, v{LB + i}")
            else:
                A(f"  v_add_f32 v{ACC + i}, v{ACC + i}, v{LB + 4 * k + i}")
    for i in range(4):
        x = ACC + i
        A(f"  v_mov_b32 v{T}, v{x}")
        A("  s_nop 1")
        A(f"  v_add_f32_dpp v{x}, v{T}, v{x} row_ror:8 row_mask:0xf bank_mask:0xf")
        for sw in ("v_permlane16_swap_b32", "v_permlane32_swap_b32"):
            A(f"  v_mov_b32 v{T}, v{x}")
            A("  s_nop 1")
            A(f"  {sw} v{x}, v{T}")
            A("  s_nop 3")
            A(f"  v_add_f32 v{x}, v{x}, v{T}")
    A("  s_mov_b64 exec, 0xff")
    A(f"  ds_write_b128 v10, v[{ACC}:{ACC + 3}] offset:{lds}")
    A("  s_mov_b64 exec, -1")
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  ds_read_b32 v31, v11 offset:{lds}")
    A(f"  ds_read_b32 v32, v8 offset:{lds + 96}")  # Σ R'²
    A(f"  ds_read_b32 v33, v8 offset:{lds + 100}")  # Σ x1²
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_mov_b32 s91, s90")  # this token; s90 → the next one, its loads in flight
    A(f"  s_add_u32 s90, s90, {stride}")
    A("  s_cmp_lt_u32 s90, s18")
    A("  s_cbranch_scc0 .Lmx_last")
    loads()
    A(".Lmx_last:")
    A("  v_mov_b32 v9, s30")
    A("  v_fma_f32 v33, v33, s87, v9")
    A("  v_rsq_f32 v33, v33")  # rstd_n
    if rstd_store:
        A("  s_lshl_b32 s82, s91, 2")
        A("  s_mov_b64 exec, 1")
        A("  buffer_store_dword v33, v12, s[60:63], s82 offen")
        A("  s_mov_b64 exec, -1")
    gates_sinkhorn(A, tok="s91", base_loaded=True)
    A("  s_cmp_lt_u32 s90, s18")
    A("  s_cbranch_scc1 .Lmx_tok")
    A(".Lmx_end:")
    A("  s_endpgm")
    return c


def source_fin(name: str = "ffn_seam_fin"):
    """FFN seam, second pass (aiter's reduce kernel + the mixes): one CTA per
    token t.
    1. Sum the 160 partial rows in fixed order: thread (rg = tid/32, j = tid%32)
       sums rows 20 rg .. 20 rg + 19 of value v(j), then threads 0..31 sum the
       8 groups. v(j) puts pre/post on lanes 0..7, Σ R'² / Σ x1² on 8..9 and
       the 16 comb logits on lanes 16..31 (one DPP row: row sums are quads,
       column sums row_ror 4 + 8).
    2. Wave 0: rstd, pre' / post' gates, comb' by Sinkhorn (softmax, then
       20 × row / column normalise, x / s as x · rcp(s) like aiter's
       fast_dividef) → pre_out, post_out, comb_out.
    3. Threads 0..159, one 32-column block each: y = bf16((x1 · rstd_n) · w)
       → y [M, H] (the router's input), then MXFP8 of y → xq, xs (gemv8's frag
       layout: K1's Engram-layer seam, the only user)."""
    c = []
    A = c.append
    A(
        "  s_load_dwordx16 s[4:19], s[0:1], 0x0"
    )  # partial x1 norm_w scale base post comb pre
    A("  s_load_dwordx8 s[20:27], s[0:1], 0x40")  # y xq xs, m s26, rms_eps s27
    A("  s_load_dwordx4 s[28:31], s[0:1], 0x60")  # hc_eps s28, alpha s29, norm_eps s30
    A("  v_readfirstlane_b32 s76, v0")
    A("  s_lshr_b32 s76, s76, 6")  # wave
    A("  v_and_b32 v1, 63, v0")  # lane
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_load_dwordx2 s[72:73], s[10:11], 0x0")  # scale[0:2]
    A("  s_load_dword s74, s[10:11], 0x8")
    A(f"  s_mul_i32 s77, s26, {SLICES * 128}")
    A(f"  s_mul_i32 s78, s26, {2 * H}")
    A("  s_lshl_b32 s79, s26, 4")
    A("  s_lshl_b32 s80, s26, 6")
    # x for gemv8 (frag layout, 256 tokens)
    A(f"  s_mov_b32 s81, {256 * H}")
    A(f"  s_mov_b32 s82, {256 * H // 32}")
    c += rsrc(32, 4, nbytes_sgpr=77)  # partial
    c += rsrc(36, 6, nbytes_sgpr=78)  # x1
    c += rsrc(40, 8, nbytes=2 * H)  # norm_w
    c += rsrc(44, 12, nbytes=24 * 4)  # base
    c += rsrc(48, 14, nbytes_sgpr=79)  # post_out
    c += rsrc(52, 16, nbytes_sgpr=80)  # comb_out
    c += rsrc(56, 18, nbytes_sgpr=79)  # pre_out
    c += rsrc(60, 20, nbytes_sgpr=78)  # y
    c += rsrc(64, 22, nbytes_sgpr=81)  # xq
    c += rsrc(68, 24, nbytes_sgpr=82)  # xs
    # ---- 1. partial sums. v(j) = j (j < 8), j + 16 (8 ≤ j < 16), j − 8 (16 ≤ j < 32)
    A("  v_lshrrev_b32 v2, 5, v0")  # rg
    A("  v_and_b32 v3, 31, v0")  # j
    A("  v_add_u32 v4, 16, v3")
    A("  v_cmp_gt_u32 vcc, 8, v3")
    A("  v_cndmask_b32 v4, v4, v3, vcc")
    A("  v_subrev_u32 v5, 8, v3")
    A("  v_cmp_le_u32 vcc, 16, v3")
    A("  v_cndmask_b32 v4, v4, v5, vcc")  # v(j)
    A("  v_lshlrev_b32 v7, 2, v4")
    A(f"  s_mul_i32 s83, s2, {SLICES}")
    A("  v_mad_u32_u24 v5, 20, v2, s83")
    A("  v_lshlrev_b32 v5, 7, v5")
    A("  v_lshl_add_u32 v5, v4, 2, v5")
    for k in range(20):
        A(f"  buffer_load_dword v{10 + k}, v5, s[32:35], 0 offen offset:{128 * k}")
    A("  s_waitcnt vmcnt(0)")
    A("  v_mov_b32 v30, v10")
    for k in range(1, 20):
        A(f"  v_add_f32 v30, v30, v{10 + k}")
    A("  v_lshlrev_b32 v6, 2, v0")
    A("  ds_write_b32 v6, v30")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    A("  v_cmp_gt_u32 vcc, 32, v0")
    A("  s_and_saveexec_b64 s[84:85], vcc")
    for g in range(8):
        A(f"  ds_read_b32 v{10 + g}, v6 offset:{128 * g}")
    A("  s_waitcnt lgkmcnt(0)")
    A("  v_mov_b32 v31, v10")
    for g in range(1, 8):
        A(f"  v_add_f32 v31, v31, v{10 + g}")
    A("  ds_write_b32 v7, v31 offset:1024")  # total of value v at 1024 + 4 v
    A("  s_mov_b64 exec, s[84:85]")
    A("  s_waitcnt lgkmcnt(0)")
    A("  s_barrier")
    A("  v_mov_b32 v8, 0")
    A("  ds_read_b32 v32, v8 offset:1120")
    A("  ds_read_b32 v33, v8 offset:1124")  # Σ x1² (value 25)
    A("  s_waitcnt lgkmcnt(0)")
    A(f"  s_mov_b32 s86, {f32(1.0 / (4 * H))}")
    A(f"  s_mov_b32 s87, {f32(1.0 / H)}")
    A(f"  s_mov_b32 s88, {f32(1.4426950408889634)}")  # log2 e
    A(f"  s_mov_b32 s89, {f32(-1.4426950408889634)}")
    A("  v_mov_b32 v9, s30")
    A("  v_fma_f32 v33, v33, s87, v9")
    A("  v_rsq_f32 v33, v33")  # rstd_n (every wave)
    A("  s_cmp_eq_u32 s76, 0")
    A("  s_cbranch_scc0 .Lnorm")
    # ---- 2. gates (wave 0; lane L < 32 holds value v(L) in v31)
    gates_sinkhorn(A)
    # ---- 3. norm + MXFP8: thread b < 160 owns columns 32 b .. 32 b + 31
    A(".Lnorm:")
    A(f"  v_cmp_gt_u32 vcc, {SLICES}, v0")
    A("  s_and_saveexec_b64 s[84:85], vcc")
    A("  s_cbranch_execz .Lend")
    A("  v_mov_b32 v47, s2")  # t
    A(f"  s_mul_i32 s90, s2, {2 * H}")
    A("  v_lshlrev_b32 v48, 6, v0")  # b · 64
    A("  v_add_u32 v49, s90, v48")  # x1 / y offset
    for q in range(4):
        A(
            f"  buffer_load_dwordx4 v[{50 + 4 * q}:{53 + 4 * q}], v49, s[36:39], 0 offen offset:{16 * q}"
        )
        A(
            f"  buffer_load_dwordx4 v[{66 + 4 * q}:{69 + 4 * q}], v48, s[40:43], 0 offen offset:{16 * q}"
        )
    A("  s_waitcnt vmcnt(0)")
    # y_i = bf16((x1_i · rstd_n) · w_i) → packed v[82:97]; f32 of y → v[100:131]
    for k in range(16):
        lo, hi = 100 + 2 * k, 101 + 2 * k
        A(f"  v_lshlrev_b32 v{lo}, 16, v{50 + k}")
        A(f"  v_and_b32 v{hi}, 0xffff0000, v{50 + k}")
        A(f"  v_mul_f32 v{lo}, v{lo}, v33")
        A(f"  v_mul_f32 v{hi}, v{hi}, v33")
        A(f"  v_lshlrev_b32 v98, 16, v{66 + k}")
        A(f"  v_and_b32 v99, 0xffff0000, v{66 + k}")
        A(f"  v_mul_f32 v{lo}, v{lo}, v98")
        A(f"  v_mul_f32 v{hi}, v{hi}, v99")
        A(f"  v_cvt_pk_bf16_f32 v{82 + k}, v{lo}, v{hi}")
        A(f"  v_lshlrev_b32 v{lo}, 16, v{82 + k}")
        A(f"  v_and_b32 v{hi}, 0xffff0000, v{82 + k}")
    for q in range(4):
        A(
            f"  buffer_store_dwordx4 v[{82 + 4 * q}:{85 + 4 * q}], v49, s[60:63], 0 offen offset:{16 * q}"
        )
    quant_block(A, 100, 132, "47", "0", 64, 68)
    A(".Lend:")
    A("  s_waitcnt vmcnt(0)")
    A("  s_endpgm")
    src = kernel_source(
        name, c, KA_FIN, lds=1152, vgprs=160, agprs=0, sgprs=96, block=256
    )
    return dict(name=name, src=src, ka=KA_FIN)
