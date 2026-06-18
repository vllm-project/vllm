# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software fp8e4m3 (E4M3FN / fp8e4nv) <-> fp16 conversion for Triton, SM75+.

Companion to fp8e4nv_sm80.py (which converts fp8 <-> *bf16*). bf16 is an SM80+
hardware type, so the bf16 path cannot serve SM75 (Turing, e.g. Tesla T4). On
SM75 the only software dtype for an fp8 KV cache is fp16, so these helpers
convert fp8e4m3 <-> fp16 directly (NOT via bf16) using ONLY SM75-legal integer
PTX (and/or/shl/shr/add/sub/min/max/setp/selp/prmt/cvt.u32.u16) -- no native
fp8 cvt, no bf16, no video-SIMD (vset2/vsub2). Verified to assemble for sm_75
(`ptxas -arch=sm_75`). Runs on SM75-SM88; on SM89+ use the native fp8e4nv cvt.

Validated bit-exact (decode and RNE encode) / <=2 FP8-code ULP (truncating
encode) vs torch.float8_e4m3fn over the entire finite domain (all 254 finite fp8
bytes; all 63,488 finite fp16 patterns), denormals included.

FP8 E4M3FN byte:  S EEEE MMM        bias 7,  finite-only (0x7f/0xff = NaN), max 448
FP16 (IEEE half): S EEEEE MMMMMMMMMM  bias 15
Normal map: fp16_exp = fp8_exp + 8, fp16_mant = fp8_mant << 7
  decode normal:  fp16 = (mag << 7) + 0x2000      (0x2000 == 8<<10)
  encode normal:  fp8  = ((e5 - 8) << 3) + round(m10 >> 7)

# ---------------------------------------------------------------------------
# Encode: fp16 -> fp8e4m3  (write path -- KV-cache store, O(new_tokens))
#
#   RNE   -- round-to-nearest-even, saturating overflow (>448 -> 0x7e, never
#            NaN). Bit-exact vs torch over all finite fp16. ~164 PTX instr/pack-4.
#   TRUNC -- truncation, saturating; <=2 FP8-code ULP. ~84 PTX instr/pack-4.
# Encode fires once per written KV position, not in the attention inner loop,
# so RNE is the better default; TRUNC is offered for the throughput-sensitive
# write path.
#
# Decode: fp8e4m3 -> fp16  (read path -- KV-cache load, attention inner loop)
#
#   fp8->fp16 is an exact expansion. The 7 fp8-subnormal targets all have fp16
#   low-byte 0, so the subnormal value comes from a single prmt byte-LUT whose
#   control nibble is the (shifted) fp8 code -- LUT[m] << 8 in one prmt, no
#   region arithmetic, no vset2, no post and/shl. Bit-exact incl. denormals.
#   This is the hot path (every KV tile load), so it is the leanest + lowest
#   register pressure.
#
# NaN/Inf: not handled specially (fp16 Inf/NaN -> saturated fp8 byte on encode;
# fp8 NaN bytes 0x7f/0xff -> large-finite fp16 on decode). KV activations do not
# contain NaN/Inf; this matches the finite-conversion contract.
# ---------------------------------------------------------------------------
"""

from vllm.triton_utils import tl, triton

# ===========================================================================
# PTX builders (codegen at import; pure SM75 integer ISA). Expand to string
# literals if preferred -- the builders just keep the 4 lanes in sync.
# ===========================================================================


def _enc_input_header() -> list[str]:
    return [
        "    mov.b32 {b0, b1}, $1;",
        "    mov.b32 {b2, b3}, $2;",
        "    cvt.u32.u16 raw0, b0;",
        "    cvt.u32.u16 raw1, b1;",
        "    cvt.u32.u16 raw2, b2;",
        "    cvt.u32.u16 raw3, b3;",
    ]


def _enc_pack_tail() -> list[str]:
    return [
        "    shl.b32 o1, o1, 8;",
        "    shl.b32 o2, o2, 16;",
        "    shl.b32 o3, o3, 24;",
        "    or.b32  out, o0, o1;",
        "    or.b32  out, out, o2;",
        "    or.b32  $0, out, o3;",
    ]


def _enc_lane_rne(i: int, overflow_code: str) -> str:
    # round-to-nearest-even via half-up + single even-tie correction (normal &
    # subnormal); saturating overflow. Preserves all fp8 denormals.
    return f"""
    and.b32  a{i}, raw{i}, 0x7fff;
    and.b32  sgn{i}, raw{i}, 0x8000;
    shr.u32  sgn{i}, sgn{i}, 8;
    shr.u32  e{i}, a{i}, 10;
    and.b32  m{i}, a{i}, 0x3ff;

    add.u32  r{i}, m{i}, 0x40;
    shr.u32  r{i}, r{i}, 7;
    and.b32  tmp{i}, m{i}, 0xff;
    setp.eq.u32 p_ntie{i}, tmp{i}, 0x40;
    sub.u32  ndec{i}, r{i}, 1;
    selp.u32 r{i}, ndec{i}, r{i}, p_ntie{i};
    sub.u32  norm{i}, e{i}, 8;
    shl.b32  norm{i}, norm{i}, 3;
    add.u32  norm{i}, norm{i}, r{i};
    min.u32  norm{i}, norm{i}, 0x7f;

    max.u32  ec{i}, e{i}, 5;
    sub.u32  sh{i}, 16, ec{i};
    sub.u32  shm1{i}, sh{i}, 1;
    mov.u32  one{i}, 1;
    shl.b32  half{i}, one{i}, shm1{i};
    or.b32   tmp{i}, m{i}, 0x400;
    add.u32  sub{i}, tmp{i}, half{i};
    shr.u32  sub{i}, sub{i}, sh{i};
    add.u32  shp1{i}, sh{i}, 1;
    shl.b32  mask{i}, one{i}, shp1{i};
    sub.u32  mask{i}, mask{i}, 1;
    and.b32  rem{i}, tmp{i}, mask{i};
    setp.eq.u32 p_stie{i}, rem{i}, half{i};
    sub.u32  sdec{i}, sub{i}, 1;
    selp.u32 sub{i}, sdec{i}, sub{i}, p_stie{i};
    min.u32  sub{i}, sub{i}, 8;
    setp.lt.u32 p_tiny{i}, e{i}, 5;
    selp.u32 sub{i}, 0, sub{i}, p_tiny{i};

    setp.gt.u32 p_norm{i}, e{i}, 8;
    selp.u32 o{i}, norm{i}, sub{i}, p_norm{i};
    setp.ge.u32 p_hi{i}, a{i}, 0x5f41;
    selp.u32 o{i}, {overflow_code}, o{i}, p_hi{i};
    or.b32 o{i}, o{i}, sgn{i};
"""


def _make_enc_rne_asm(overflow_code: str) -> str:
    regs, preds = [], []
    for i in range(4):
        regs.append(
            "    .reg .u32 "
            + ", ".join(
                f"{n}{i}"
                for n in (
                    "a",
                    "e",
                    "m",
                    "r",
                    "tmp",
                    "ndec",
                    "norm",
                    "ec",
                    "sh",
                    "shm1",
                    "shp1",
                    "one",
                    "half",
                    "mask",
                    "rem",
                    "sub",
                    "sdec",
                    "sgn",
                    "o",
                )
            )
            + ";"
        )
        preds.append(
            "    .reg .pred "
            + ", ".join(
                f"{p}{i}" for p in ("p_ntie", "p_stie", "p_tiny", "p_norm", "p_hi")
            )
            + ";"
        )
    lanes = "".join(_enc_lane_rne(i, overflow_code) for i in range(4))
    return "\n".join(
        [
            "{",
            "    .reg .u16 b<4>;",
            "    .reg .u32 raw<4>;",
            *regs,
            "    .reg .u32 out;",
            *preds,
            "",
            *_enc_input_header(),
            lanes,
            *_enc_pack_tail(),
            "}",
        ]
    )


def _enc_lane_trunc(i: int, overflow_code: str) -> str:
    # truncating encode (no rounding, no tie logic); saturating overflow.
    return f"""
    and.b32  a{i}, raw{i}, 0x7fff;
    and.b32  sgn{i}, raw{i}, 0x8000;
    shr.u32  sgn{i}, sgn{i}, 8;
    shr.u32  e{i}, a{i}, 10;
    and.b32  m{i}, a{i}, 0x3ff;

    shr.u32  r{i}, m{i}, 7;
    sub.u32  norm{i}, e{i}, 8;
    shl.b32  norm{i}, norm{i}, 3;
    add.u32  norm{i}, norm{i}, r{i};

    max.u32  ec{i}, e{i}, 5;
    sub.u32  sh{i}, 16, ec{i};
    or.b32   tmp{i}, m{i}, 0x400;
    shr.u32  sub{i}, tmp{i}, sh{i};

    setp.gt.u32 p_norm{i}, e{i}, 8;
    selp.u32 o{i}, norm{i}, sub{i}, p_norm{i};
    setp.ge.u32 p_hi{i}, a{i}, 0x5f41;
    selp.u32 o{i}, {overflow_code}, o{i}, p_hi{i};
    or.b32 o{i}, o{i}, sgn{i};
"""


def _make_enc_trunc_asm(overflow_code: str) -> str:
    regs, preds = [], []
    for i in range(4):
        regs.append(
            "    .reg .u32 "
            + ", ".join(
                f"{n}{i}"
                for n in (
                    "a",
                    "e",
                    "m",
                    "r",
                    "norm",
                    "ec",
                    "sh",
                    "tmp",
                    "sub",
                    "sgn",
                    "o",
                )
            )
            + ";"
        )
        preds.append("    .reg .pred " + ", ".join((f"p_norm{i}", f"p_hi{i}")) + ";")
    lanes = "".join(_enc_lane_trunc(i, overflow_code) for i in range(4))
    return "\n".join(
        [
            "{",
            "    .reg .u16 b<4>;",
            "    .reg .u32 raw<4>;",
            *regs,
            "    .reg .u32 out;",
            *preds,
            "",
            *_enc_input_header(),
            lanes,
            *_enc_pack_tail(),
            "}",
        ]
    )


def _dec_extract_header() -> str:
    return """
    and.b32 raw0, $2, 0xff;
    shr.u32 raw1, $2, 8;
    and.b32 raw1, raw1, 0xff;
    shr.u32 raw2, $2, 16;
    and.b32 raw2, raw2, 0xff;
    shr.u32 raw3, $2, 24;
    and.b32 raw3, raw3, 0xff;
"""


def _dec_lane_lut(i: int) -> str:
    # normal: (mag<<7)+0x2000; subnormal value from a prmt byte-LUT (the 7 fp16
    # subnormal targets all have low byte 0 -> sub bits = LUT[m] << 8); zero via
    # m==0 LUT entry 0x00. Bit-exact incl. denormals.
    #
    # The shift `m << 4` places the LUT index in prmt control nibble 1, so prmt
    # writes LUT[m] straight into result byte 1 while nibbles 0/2/3 select sublut
    # byte 0 (= 0x00). That yields LUT[m] << 8 in ONE prmt -- no post and/shl --
    # which keeps the decode register-lean (the hot KV-load path).
    return f"""
    and.b32 mag{i}, raw{i}, 0x7f;
    and.b32 sign{i}, raw{i}, 0x80;
    shl.b32 sign{i}, sign{i}, 8;
    shl.b32 norm{i}, mag{i}, 7;
    add.u32 norm{i}, norm{i}, 0x2000;
    and.b32 m{i}, raw{i}, 0x07;
    shl.b32 m{i}, m{i}, 4;
    prmt.b32 sub{i}, sublut, subhi, m{i};
    setp.ge.u32 p_norm{i}, mag{i}, 8;
    selp.u32 o{i}, norm{i}, sub{i}, p_norm{i};
    or.b32 o{i}, o{i}, sign{i};
"""


def _make_dec_lut_asm() -> str:
    regs, preds = [], []
    for i in range(4):
        regs.append(
            "    .reg .u32 "
            + ", ".join(
                f"{n}{i}" for n in ("raw", "mag", "m", "sign", "norm", "sub", "o")
            )
            + ";"
        )
        preds.append(f"    .reg .pred p_norm{i};")
    lanes = "".join(_dec_lane_lut(i) for i in range(4))
    # LUT high bytes by m (0..7): 0x00,0x18,0x1c,0x1e,0x20,0x21,0x22,0x23
    return "\n".join(
        [
            "{",
            "    .reg .u32 sublut, subhi;",
            *regs,
            "    .reg .u32 out0, out1;",
            *preds,
            "    mov.u32 sublut, 0x1e1c1800;",
            "    mov.u32 subhi, 0x23222120;",
            _dec_extract_header(),
            lanes,
            "    shl.b32 o1, o1, 16;",
            "    or.b32  out0, o0, o1;",
            "    shl.b32 o3, o3, 16;",
            "    or.b32  out1, o2, o3;",
            "    mov.b32 $0, out0;",
            "    mov.b32 $1, out1;",
            "}",
        ]
    )


# Saturating overflow (0x7e, never NaN) is the KV-cache default (no NaN/Inf in
# activations; overflow clamps to fp8 max 448).
_FP16_TO_FP8E4M3_RNE_ASM = _make_enc_rne_asm("0x7e")
_FP16_TO_FP8E4M3_TRUNC_ASM = _make_enc_trunc_asm("0x7e")
_FP8E4M3_TO_FP16_ASM = _make_dec_lut_asm()

# Select the encode variant (RNE default; swap to TRUNC for the fast write path):
_FP16_TO_FP8E4M3_ASM = _FP16_TO_FP8E4M3_RNE_ASM
# _FP16_TO_FP8E4M3_ASM = _FP16_TO_FP8E4M3_TRUNC_ASM

_FP16_TO_FP8E4M3_ASM = tl.constexpr(_FP16_TO_FP8E4M3_ASM)
_FP16_TO_FP8E4M3_TRUNC_ASM = tl.constexpr(_FP16_TO_FP8E4M3_TRUNC_ASM)
_FP8E4M3_TO_FP16_ASM = tl.constexpr(_FP8E4M3_TO_FP16_ASM)


@triton.jit
def fp8e4m3_to_fp16(x):
    """4 packed uint8 fp8e4m3 bytes -> 4 fp16 (pack-4, exact prmt-LUT, SM75)."""
    return tl.inline_asm_elementwise(
        _FP8E4M3_TO_FP16_ASM,
        "=r,=r,r",
        [x],
        dtype=tl.float16,
        is_pure=True,
        pack=4,
    )


@triton.jit
def fp16_to_fp8e4m3(x):
    """4 fp16 -> 4 packed uint8 fp8e4m3 bytes (pack-4, RNE saturating, SM75)."""
    return tl.inline_asm_elementwise(
        _FP16_TO_FP8E4M3_ASM,
        "=r,r,r",
        [x],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def fp16_to_fp8e4m3_trunc(x):
    """4 fp16 -> 4 packed uint8 fp8e4m3 bytes (pack-4, truncating saturating, SM75)."""
    return tl.inline_asm_elementwise(
        _FP16_TO_FP8E4M3_TRUNC_ASM,
        "=r,r,r",
        [x],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )
