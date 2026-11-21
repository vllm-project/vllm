# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct packed E4M3 conversion with compile-time PTX policy fragments.

NaN handling and E4M3 underflow flushing are independently opt-in.
Flushing underflows also canonicalizes zero to positive zero.
pack selects one, two, or four elements per inline-assembly invocation.
Flushing removes subnormal work before compilation. The production helper remains
unchanged while this implementation is benchmarked.
"""

from functools import lru_cache

import regex as re

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8e4nv import _check_software_conversion

ENCODE = {
    "fp16": {
        "declarations": r"""
    .reg .u32 raw{i}, a{i}, e{i}, m{i}, r{i}, tmp{i}, ndec{i}, norm{i}, ec{i},
        sh{i}, shm1{i}, shp1{i}, one{i}, half{i}, mask{i},
        rem{i}, sub{i}, sdec{i}, sgn{i}, o{i};
    .reg .pred p_ntie{i}, p_stie{i}, p_tiny{i}, p_norm{i}, p_hi{i};
""",
        "normal": r"""
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

""",
        "subnormal": r"""    max.u32  ec{i}, e{i}, 5;
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

""",
        "selection": r"""    setp.gt.u32 p_norm{i}, e{i}, 8;
    selp.u32 o{i}, norm{i}, sub{i}, p_norm{i};
""",
        "saturation": r"""    setp.ge.u32 p_hi{i}, a{i}, 0x5f41;
    selp.u32 o{i}, 0x7e, o{i}, p_hi{i};
""",
    },
    "bf16": {
        "declarations": r"""
    .reg .u32 raw{i}, a{i}, e{i}, m{i}, r{i}, tmp{i}, ntmp{i}, ndec{i},
        norm{i}, ec{i}, sh{i}, sh2{i}, shp1{i}, one{i}, rnd{i},
        mask{i}, rem{i}, sub{i}, sdec{i}, sgn{i}, o{i};
    .reg .pred p_ntie{i}, p_stie{i}, p_tiny{i}, p_norm{i}, p_hi{i};
""",
        "normal": r"""
    and.b32  a{i}, raw{i}, 0x7fff;
    and.b32  sgn{i}, raw{i}, 0x8000;
    shr.u32  sgn{i}, sgn{i}, 8;
    shr.u32  e{i}, a{i}, 7;
    and.b32  m{i}, a{i}, 0x7f;
    add.u32  r{i}, m{i}, 8;
    shr.u32  r{i}, r{i}, 4;
    and.b32  ntmp{i}, m{i}, 0x1f;
    setp.eq.u32 p_ntie{i}, ntmp{i}, 8;
    sub.u32  ndec{i}, r{i}, 1;
    selp.u32 r{i}, ndec{i}, r{i}, p_ntie{i};
    sub.u32  norm{i}, e{i}, 120;
    shl.b32  norm{i}, norm{i}, 3;
    add.u32  norm{i}, norm{i}, r{i};
    min.u32  norm{i}, norm{i}, 0x7e;
""",
        "subnormal": r"""    max.u32  ec{i}, e{i}, 117;
    sub.u32  sh{i}, 125, ec{i};
    sub.u32  sh2{i}, sh{i}, 1;
    mov.u32  rnd{i}, 1;
    shl.b32  rnd{i}, rnd{i}, sh2{i};
    or.b32   tmp{i}, m{i}, 0x80;
    add.u32  sub{i}, tmp{i}, rnd{i};
    shr.u32  sub{i}, sub{i}, sh{i};
    add.u32  shp1{i}, sh{i}, 1;
    mov.u32  one{i}, 1;
    shl.b32  mask{i}, one{i}, shp1{i};
    sub.u32  mask{i}, mask{i}, 1;
    and.b32  rem{i}, tmp{i}, mask{i};
    setp.eq.u32 p_stie{i}, rem{i}, rnd{i};
    sub.u32  sdec{i}, sub{i}, 1;
    selp.u32 sub{i}, sdec{i}, sub{i}, p_stie{i};
    min.u32  sub{i}, sub{i}, 8;
    setp.lt.u32 p_tiny{i}, e{i}, 117;
    selp.u32 sub{i}, 0, sub{i}, p_tiny{i};
""",
        "selection": r"""    setp.gt.u32 p_norm{i}, e{i}, 120;
    selp.u32 o{i}, norm{i}, sub{i}, p_norm{i};
""",
        "saturation": r"""    setp.ge.u32 p_hi{i}, a{i}, 0x43e0;
    selp.u32 o{i}, 0x7e, o{i}, p_hi{i};
""",
    },
}

FULL = {
    "fp16": r"""{
    .reg .u32 zero, reg0, reg1, tmp0, tmp1, ctrl0, ctrl1, lt0, lt1;
    .reg .u32 exp0, exp1, norm0, norm1, sub0, sub1, lo0, lo1, hi0, hi1;
    .reg .u32 sign0, sign1, mask0, mask1, out0, out1;
    mov.u32 zero, 0;
    prmt.b32 reg0, $2, zero, 0x4140;
    prmt.b32 reg1, $2, zero, 0x4342;
    and.b32 tmp0, reg0, 0x007f007f;
    and.b32 tmp1, reg1, 0x007f007f;
    shl.b32 norm0, tmp0, 7;
    shl.b32 norm1, tmp1, 7;
    add.u32 norm0, norm0, 0x20002000;
    add.u32 norm1, norm1, 0x20002000;
    // PRMT uses one selector nibble per output byte. The original FP8
    // mantissas index bytes 0 and 2 for each pair of 16-bit outputs.
    and.b32 ctrl0, $2, 0x07070707;
    shr.u32 ctrl1, ctrl0, 16;
    mov.u32 lt0, 0x1e1c1800;
    mov.u32 lt1, 0x23222120;
    prmt.b32 hi0, lt0, lt1, ctrl0;
    prmt.b32 hi1, lt0, lt1, ctrl1;
    prmt.b32 sub0, zero, hi0, 0x6040;
    prmt.b32 sub1, zero, hi1, 0x6040;
    and.b32 mask0, reg0, 0x00780078;
    and.b32 mask1, reg1, 0x00780078;
    add.u32 mask0, mask0, 0x007f007f;
    and.b32 mask0, mask0, 0x00800080;
    shr.u32 mask0, mask0, 7;
    mul.lo.u32 mask0, mask0, 0xffff;
    add.u32 mask1, mask1, 0x007f007f;
    and.b32 mask1, mask1, 0x00800080;
    shr.u32 mask1, mask1, 7;
    mul.lo.u32 mask1, mask1, 0xffff;
    lop3.b32 out0, mask0, norm0, sub0, 0xca;
    lop3.b32 out1, mask1, norm1, sub1, 0xca;
    and.b32 sign0, reg0, 0x00800080;
    and.b32 sign1, reg1, 0x00800080;
    shl.b32 sign0, sign0, 8;
    shl.b32 sign1, sign1, 8;
    or.b32 $0, out0, sign0;
    or.b32 $1, out1, sign1;
  }""",
    "bf16": r"""{
    .reg .u32 zero, reg0, reg1, tmp0, tmp1, ctrl0, ctrl1, lt0, lt1;
    .reg .u32 exp0, exp1, norm0, norm1, sub0, sub1, lo0, lo1, hi0, hi1;
    .reg .u32 sign0, sign1, mask0, mask1, out0, out1;
    mov.u32 zero, 0;
    prmt.b32 reg0, $2, zero, 0x4140;
    prmt.b32 reg1, $2, zero, 0x4342;
    mov.u32 lt0, 0x3f3e3d3c;
    mov.u32 lt1, 0x43424140;
    shl.b32 tmp0, reg0, 4;
    shl.b32 tmp1, reg1, 4;
    prmt.b32 ctrl0, tmp0, zero, 0x4331;
    prmt.b32 ctrl1, tmp1, zero, 0x4331;
    and.b32 ctrl0, ctrl0, 0x07070707;
    and.b32 ctrl1, ctrl1, 0x07070707;
    prmt.b32 exp0, lt0, lt1, ctrl0;
    prmt.b32 exp1, lt0, lt1, ctrl1;
    prmt.b32 norm0, tmp0, exp0, 0x6240;
    prmt.b32 norm1, tmp1, exp1, 0x6240;
    // PRMT uses one selector nibble per output byte. The original FP8
    // mantissas index bytes 0 and 2 for each pair of 16-bit outputs.
    and.b32 ctrl0, $2, 0x07070707;
    shr.u32 ctrl1, ctrl0, 16;
    mov.u32 lt0, 0xc0800000;
    mov.u32 lt1, 0x60402000;
    prmt.b32 lo0, lt0, lt1, ctrl0;
    prmt.b32 lo1, lt0, lt1, ctrl1;
    mov.u32 lt0, 0x3b3b3b00;
    mov.u32 lt1, 0x3c3c3c3c;
    prmt.b32 hi0, lt0, lt1, ctrl0;
    prmt.b32 hi1, lt0, lt1, ctrl1;
    prmt.b32 sub0, lo0, hi0, 0x6240;
    prmt.b32 sub1, lo1, hi1, 0x6240;
    and.b32 mask0, reg0, 0x00780078;
    and.b32 mask1, reg1, 0x00780078;
    add.u32 mask0, mask0, 0x007f007f;
    and.b32 mask0, mask0, 0x00800080;
    shr.u32 mask0, mask0, 7;
    mul.lo.u32 mask0, mask0, 0xffff;
    add.u32 mask1, mask1, 0x007f007f;
    and.b32 mask1, mask1, 0x00800080;
    shr.u32 mask1, mask1, 7;
    mul.lo.u32 mask1, mask1, 0xffff;
    lop3.b32 out0, mask0, norm0, sub0, 0xca;
    lop3.b32 out1, mask1, norm1, sub1, 0xca;
    and.b32 sign0, reg0, 0x00800080;
    and.b32 sign1, reg1, 0x00800080;
    shl.b32 sign0, sign0, 8;
    shl.b32 sign1, sign1, 8;
    or.b32 $0, out0, sign0;
    or.b32 $1, out1, sign1;
  }""",
}


def _encode(name, width, nan, flush):
    """Compose encoder phases; omit subnormal arithmetic when flushing."""
    pieces = ["{"]
    for i in range(width):
        f = ENCODE[name]
        pieces.append(f["declarations"].format(i=i))
        operand = 1 + i // 2
        if width == 1:
            pieces.append("cvt.u32.u16 raw0, $1;")
        elif i % 2 == 0:
            pieces.append(f"and.b32 raw{i}, ${operand}, 0xffff;")
        else:
            pieces.append(f"shr.u32 raw{i}, ${operand}, 16;")
        pieces.append(f["normal"].format(i=i))
        if flush:
            # Match the previously developed input cutoff: |x| < E4M3 min normal.
            cutoff = "0x2400" if name == "fp16" else "0x3c80"
            pieces.append(
                f".reg .pred p_flush{i}; "
                f"setp.lt.u32 p_flush{i},a{i},{cutoff};"
                f"selp.u32 o{i},0,norm{i},p_flush{i};"
            )
        else:
            pieces.extend([f["subnormal"].format(i=i), f["selection"].format(i=i)])
        pieces.append(f["saturation"].format(i=i))
        pieces.append(f"or.b32 o{i},o{i},sgn{i};")
        if flush:
            pieces.append(f"selp.u32 o{i},0,o{i},p_flush{i};")
        if nan:
            limit = "0x7c00" if name == "fp16" else "0x7f80"
            pieces.append(
                f".reg .pred p_nan{i}; "
                f"setp.gt.u32 p_nan{i},a{i},{limit}; "
                f"selp.u32 o{i},0x7f,o{i},p_nan{i};"
            )
    if width == 1:
        pieces.append("cvt.u16.u32 $0,o0;")
    else:
        pieces.append(".reg .u32 assembled; mov.b32 assembled,o0;")
        for i in range(1, width):
            pieces.append(
                f"shl.b32 o{i},o{i},{8 * i}; or.b32 assembled,assembled,o{i};"
            )
        pieces.append("mov.b32 $0,assembled;")
    return "\n".join(pieces + ["}"])


def _scalar_decode(name, nan, flush):
    """Use a scalar decoder without calculating a discarded second lane."""
    shift, bias = (7, "0x2000") if name == "fp16" else (4, "0x3c00")
    p = [
        "{",
        ".reg .u32 raw,mag,sign,norm,out,sub,m,hi,lo; .reg .pred normal;",
        "and.b32 raw,$1,0xff;",
        "and.b32 mag,raw,0x7f;",
        "and.b32 sign,raw,0x80;",
        "shl.b32 sign,sign,8;",
        f"shl.b32 norm,mag,{shift};",
        f"add.u32 norm,norm,{bias};",
    ]
    if flush:
        p += [
            "or.b32 norm,norm,sign;",
            "setp.ge.u32 normal,mag,8;",
            "selp.u32 out,norm,0,normal;",
        ]
    else:
        p += ["and.b32 m,raw,7;"]
        if name == "fp16":
            p += ["shl.b32 m,m,4;", "prmt.b32 sub,0x1e1c1800,0x23222120,m;"]
        else:
            p += [
                "prmt.b32 hi,0x3b3b3b00,0x3c3c3c3c,m;",
                "and.b32 hi,hi,0xff;",
                "shl.b32 hi,hi,8;",
                "prmt.b32 lo,0xc0800000,0x60402000,m;",
                "and.b32 lo,lo,0xff;",
                "or.b32 sub,hi,lo;",
            ]
        p += [
            "setp.ge.u32 normal,mag,8;",
            "selp.u32 out,norm,sub,normal;",
            "or.b32 out,out,sign;",
        ]
    if nan:
        p += [
            ".reg .pred n;",
            "setp.eq.u32 n,mag,0x7f;",
            f"selp.u32 out,{'0x7e00' if name == 'fp16' else '0x7fc0'},out,n;",
        ]
    return "\n".join(p + ["cvt.u16.u32 $0,out;", "}"])


def _decode(name, width, nan, flush):
    """Cut packed decoding to its width and select only enabled policy phases."""
    if width == 1:
        return _scalar_decode(name, nan, flush)
    body = FULL[name]
    if width == 2:
        lines = []
        for line in body.splitlines():
            if not line.strip().startswith((".reg", "//")) and re.search(
                r"\b(?:reg|tmp|ctrl|exp|norm|sub|lo|hi|sign|mask|out)1\b|\$1", line
            ):
                continue
            lines.append(line.replace("$2", "$1"))
        body = "\n".join(lines)
    if flush:
        # Remove exact-subnormal LUT work; apply the normal mask after inserting sign.
        lines = []
        for line in body.splitlines():
            s = line.strip()
            if not s.startswith((".reg", "//")) and (
                "sub" in s or re.search(r"\b(?:lo|hi)[01]\b", s)
            ):
                continue
            if s.startswith("lop3.b32"):
                continue
            if re.match(r"or.b32 \$\d,", s):
                line = re.sub(r"out([01]),", r"norm\1,", line)
                lines.append(line)
                match = re.search(r"\$\d", line)
                assert match is not None
                out = match.group()
                pair = 0 if out == "$0" else 1
                lines.append(f"and.b32 {out},{out},mask{pair};")
                continue
            # Keep the normal BF16 LUT; remove subnormal constants only.
            if any(
                c in s
                for c in [
                    "0xc0800000",
                    "0x60402000",
                    "0x3b3b3b00",
                    "0x3c3c3c3c",
                    "0x1e1c1800",
                    "0x23222120",
                ]
            ):
                continue
            if re.match(r"and.b32 ctrl0, \$\d, 0x07070707;", s):
                continue
            if s.startswith("shr.u32 ctrl1, ctrl0, 16;"):
                continue
            lines.append(line)
        body = "\n".join(lines)
    suffix = []
    for i in range(width):
        out = f"${i // 2}"
        shift = (i % 2) * 16
        if nan:
            bits = (0x7E00 if name == "fp16" else 0x7FC0) << shift
            suffix.extend(
                [
                    f".reg .u32 nraw{i},nval{i}; .reg .pred n{i};",
                    f"shr.u32 nraw{i}, ${width // 2}, {i * 8};",
                    f"and.b32 nraw{i},nraw{i},0x7f;",
                    f"setp.eq.u32 n{i},nraw{i},0x7f;",
                    f"and.b32 nval{i},{out},{hex(0xFFFFFFFF ^ (0xFFFF << shift))};",
                    f"or.b32 nval{i},nval{i},{hex(bits)};",
                    f"selp.u32 {out},nval{i},{out},n{i};",
                ]
            )
    return body.rstrip()[:-1] + "\n" + "\n".join(suffix) + "\n}"


@lru_cache(None)
def ptx(direction, name, width, nan=False, flush=False):
    """Build a constant assembly string before Triton compilation."""
    assert (
        direction in ("encode", "decode")
        and name in ("fp16", "bf16")
        and width in (1, 2, 4)
    )
    return (_encode if direction == "encode" else _decode)(name, width, nan, flush)


@tl.core.builtin
def _convert(
    x,
    dtype,
    width,
    encode,
    propagate_nan,
    enable_ftz,
    _semantic=None,
):
    """Pass packed tensors directly to inline PTX without caller repacking."""
    unwrap = tl.core._unwrap_if_constexpr
    dtype, width, encode, nan, flush = map(
        unwrap, (dtype, width, encode, propagate_nan, enable_ftz)
    )
    name = "fp16" if dtype == tl.float16 else "bf16"
    assert dtype in (tl.float16, tl.bfloat16)
    asm = ptx("encode" if encode else "decode", name, width, nan, flush)
    constraints = (
        ("=h,h" if width == 1 else "=r,r" if width == 2 else "=r,r,r")
        if encode
        else ("=h,r" if width == 1 else "=r,r" if width == 2 else "=r,=r,r")
    )
    return tl.core.inline_asm_elementwise(
        asm,
        constraints,
        [x],
        dtype=tl.uint8 if encode else dtype,
        is_pure=True,
        pack=width,
        _semantic=_semantic,
    )


@triton.jit
def convert_to_fp8e4m3(
    x,
    pack: tl.constexpr,
    propagate_nan: tl.constexpr = False,
    FORCE_SOFTWARE_CONVERSION: tl.constexpr = False,
    enable_ftz: tl.constexpr = False,
):
    """Encode FP16/BF16 to saturating RNE E4M3 bytes.

    pack is required and must be 1, 2, or 4. Triton supplies packed input
    registers and consumes every output; callers must not repack elements.
    propagate_nan=False and enable_ftz=False are compile-time defaults,
    preserving finite subnormals and signed zeros without NaN checking.
    Software use on SM89+ requires FORCE_SOFTWARE_CONVERSION=True; the
    default rejects accidental use where native conversion is available.

    With enable_ftz=True, positive and negative underflows and negative
    zero all flush to +0. The input cutoff is |x| < 2**-6. With flushing
    disabled, subnormals and signed zeros are preserved. NaN sign/payload are
    unspecified; opt-in NaN handling guarantees a NaN output only.
    """
    tl.static_assert(
        x.dtype == tl.float16 or x.dtype == tl.bfloat16, "expected FP16 or BF16 input"
    )
    _check_software_conversion(FORCE_SOFTWARE_CONVERSION)
    tl.static_assert(pack == 1 or pack == 2 or pack == 4, "pack must be 1, 2, or 4")
    return _convert(x, x.dtype, pack, True, propagate_nan, enable_ftz)


@triton.jit
def convert_from_fp8e4m3(
    x,
    dtype: tl.constexpr,
    pack: tl.constexpr,
    propagate_nan: tl.constexpr = False,
    FORCE_SOFTWARE_CONVERSION: tl.constexpr = False,
    enable_ftz: tl.constexpr = False,
):
    """Decode E4M3 bytes to FP16/BF16 with optional compile-time flushing.

    pack is required and must be 1, 2, or 4. Triton supplies packed input
    registers and consumes every output; callers must not repack elements.
    propagate_nan=False and enable_ftz=False are compile-time defaults,
    preserving finite subnormals and signed zeros without NaN checking.
    Software use on SM89+ requires FORCE_SOFTWARE_CONVERSION=True; the
    default rejects accidental use where native conversion is available.

    With enable_ftz=True, positive and negative E4M3 denormals and
    negative zero all flush to +0. With flushing disabled, denormals and
    signed zeros are preserved. Opt-in NaN handling guarantees NaN output
    without a sign/payload preservation requirement:
    https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
    """
    tl.static_assert(x.dtype == tl.uint8, "expected E4M3 bytes as uint8")
    tl.static_assert(
        dtype == tl.float16 or dtype == tl.bfloat16, "expected FP16 or BF16 output"
    )
    _check_software_conversion(FORCE_SOFTWARE_CONVERSION)
    tl.static_assert(pack == 1 or pack == 2 or pack == 4, "pack must be 1, 2, or 4")
    return _convert(x, dtype, pack, False, propagate_nan, enable_ftz)
