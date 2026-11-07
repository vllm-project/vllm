# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernels of the fused DSpark Markov sampling loop.

Per draft step the Markov w2 GEMV runs unchanged, then ``markov_step``
(bias, output transform, cache store, Gumbel block argmax, and in the row's
last block the final argmax, token and next w1 row). With a vocab-parallel w2
each rank works on its vocab range, its last block writes a packet instead of
the token, and ``markov_pick`` picks the winner from the gathered per-rank
packets; ``cache_fill`` scatters the gathered draft-logits shards after the
loop. ``anchor_embed`` starts a call.

The arithmetic is bit for bit that of the speculator's eager loop: its
output transform (BF16 add, multiplier, optional soft cap
``cap * tanh(x * inv_cap)``) and the stock ``gumbel_noised_argmax`` and
``gumbel_block_argmax`` of ``vllm/v1/worker/gpu/sample/gumbel.py``
(MurmurHash3 noise of seed, step, position and global vocab id). It is written
as explicit PTX, the instructions Triton 3.8 emits for those kernels on
SM 10.x, including libdevice ``tanhf`` and ``logf`` under FTZ.

Every argmax takes the first NaN, else the first maximum (the order of
``torch.argmax``). That is a total order on (value, index), so the block
argmax, the argmax over the blocks and the pick over the ranks give the same
id in any reduction order: the lowest vocab id whose noised logit is NaN, else
the lowest id of the maximum, at every block size, layout and vocab shard. On
rows without NaN this is the eager loop's id; on a row with a NaN the eager
loop's Triton block argmax can differ, and is not reproduced.

Each step's blocks count their arrivals; the row's last block reduces the
blocks' (value, id) results with the same combine.

A token's w1 row is read at the token clamped to the table's last row, so a
corrupt id never reads past the table; valid ids are unchanged. Fixed
pointers and host grids keep the loop capturable. Where the GPU has
programmatic dependent launch, the step and pick kernels can launch with it:
until the dependency wait they read only inputs written before the loop and
store nothing; they never trigger their dependents early.
"""

import functools
from collections.abc import Iterable
from typing import Any, NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cu_utils
import torch
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..common.host import assert_not_capturing
from ..common.primitives import (
    _asm,
    atomic_add1_acq_rel_u32_address,
    current_cu_stream,
    load_global_s64_address,
    load_global_u32_address,
    store_global_u32_address,
)

# The step and cache-fill kernels' threads (4 warps); the pick kernel's (one
# warp).
THREADS = 128
WARP = 32
PICK_THREADS = WARP
# The cache fill's vocab entries per CTA.
FILL_BLOCK = 1024
# Bytes of the widest access of one thread.
VECTOR_BYTES = 16
NEG_INF = float("-inf")
# Larger than every vocab id: the id of the finish's padding results.
BIG_INDEX = 2147483647


def _values(x, *, loc, ip) -> list:
    return [v.ir_value(loc=loc, ip=ip) for v in x]


def _pair(struct, types, *, loc, ip):
    return tuple(
        kind(llvm.extractvalue(kind.mlir_type, struct, [index], loc=loc, ip=ip))
        for index, kind in enumerate(types)
    )


# Global memory, by address. A false predicate loads zero and stores nothing.


@dsl_user_op
def _ld_u16(address: Int64, predicate: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            _values((Int64(address), Int32(predicate)), loc=loc, ip=ip),
            "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.s32 p, $2, 0;\n\t"
            "mov.u16 h, 0;\n\t@p ld.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,l,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_volatile_b32_or(
    address: Int64, predicate: Int32, other: Uint32, *, loc=None, ip=None
) -> Uint32:
    """A volatile load if ``predicate``, else ``other``."""
    return Uint32(
        _asm(
            T.i32(),
            _values((Int64(address), Int32(predicate), Uint32(other)), loc=loc, ip=ip),
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\tmov.u32 $0, $3;\n\t"
            "@p ld.volatile.global.b32 $0, [$1];\n\t}",
            "=r,l,r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_u16(
    address: Int64, bits: Uint32, predicate: Int32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        _values((Int64(address), Uint32(bits), Int32(predicate)), loc=loc, ip=ip),
        "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.s32 p, $2, 0;\n\t"
        "cvt.u16.u32 h, $1;\n\t@p st.global.b16 [$0], h;\n\t}",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_b64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    _asm(
        None,
        _values((Int64(address), Int64(value)), loc=loc, ip=ip),
        "st.global.b64 [$0], $1;",
        "l,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


# A copy's PTX vector type and 32-bit words by its bytes (2: one b16).
_COPY_OPS = {16: ("v4.b32", 4), 8: ("v2.b32", 2), 4: ("b32", 1), 2: ("b16", 1)}


@dsl_user_op
def _load_words(
    source: Int64,
    nbytes: cutlass.Constexpr[int],
    predicate: Int32,
    *,
    loc=None,
    ip=None,
) -> list:
    """``nbytes`` (2, 4, 8 or 16, aligned) as 32-bit words if ``predicate``
    (else zeros)."""
    kind, words = _COPY_OPS[nbytes]
    outs = ", ".join(f"${i}" for i in range(words))
    braced = f"{{{outs}}}" if words > 1 else outs
    zero = "".join(f"mov.u32 ${i}, 0;\n\t" for i in range(words))
    if nbytes == 2:
        body = f"{zero}@p ld.global.b16 h, [${words}];\n\t@p cvt.u32.u16 $0, h;"
    else:
        body = f"{zero}@p ld.global.{kind} {braced}, [${words}];"
    struct = _asm(
        llvm.StructType.get_literal([T.i32()] * words),
        _values((Int64(source), Int32(predicate)), loc=loc, ip=ip),
        "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\t"
        f"setp.ne.s32 p, ${words + 1}, 0;\n\t{body}\n\t}}",
        ",".join(["=r"] * words + ["l", "r"]),
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return list(_pair(struct, (Uint32,) * words, loc=loc, ip=ip))


@dsl_user_op
def _store_words(
    destination: Int64,
    words: list,
    nbytes: cutlass.Constexpr[int],
    predicate: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store the words of ``_load_words`` if ``predicate``."""
    kind, count = _COPY_OPS[nbytes]
    ins = ", ".join(f"${i + 2}" for i in range(count))
    braced = f"{{{ins}}}" if count > 1 else ins
    if nbytes == 2:
        body = "cvt.u16.u32 h, $2;\n\t@p st.global.b16 [$0], h;"
    else:
        body = f"@p st.global.{kind} [$0], {braced};"
    _asm(
        None,
        _values((Int64(destination), Int32(predicate), *words), loc=loc, ip=ip),
        f"{{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.s32 p, $1, 0;\n\t{body}\n\t}}",
        ",".join(["l", "r"] + ["r"] * count),
        side_effects=True,
        loc=loc,
        ip=ip,
    )


def _copy(copies: list, nbytes: int) -> None:
    """Copy ``nbytes`` (2, 4, 8 or 16, aligned) for each ``(source,
    destination, predicate)``: every load issued before the first store
    (traced inline)."""
    loaded = [_load_words(source, nbytes, predicate) for source, _, predicate in copies]
    for (_, destination, predicate), words in zip(copies, loaded):
        _store_words(destination, words, nbytes, predicate)


# Shared memory and warp exchange.


@dsl_user_op
def _st_shared_b32(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        _values((Int32(address), Uint32(value)), loc=loc, ip=ip),
        "st.shared.b32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_b32(address: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            _values((Int32(address),), loc=loc, ip=ip),
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            _values((Float32(value),), loc=loc, ip=ip),
            "mov.b32 $0, $1;",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _float(bits: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            _values((Uint32(bits),), loc=loc, ip=ip),
            "mov.b32 $0, $1;",
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# The argmax combines on (value, index): the first NaN, else the first
# maximum (torch.argmax's order).


@dsl_user_op
def _combine_in_order(
    v1: Float32, i1: Int32, v2: Float32, i2: Int32, *, loc=None, ip=None
):
    """``_combine`` of two pairs with ``i1 < i2``: the first if ``v1`` is NaN
    or ``v1 >= v2`` (ordered), else the second."""
    struct = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        _values((Float32(v1), Int32(i1), Float32(v2), Int32(i2)), loc=loc, ip=ip),
        "{\n\t.reg .pred pn, p;\n\tsetp.nan.f32 pn, $2, $2;\n\t"
        "setp.ge.or.f32 p, $2, $4, pn;\n\t"
        "selp.f32 $0, $2, $4, p;\n\tselp.b32 $1, $3, $5, p;\n\t}",
        "=f,=r,f,r,f,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _pair(struct, (Float32, Int32), loc=loc, ip=ip)


@dsl_user_op
def _combine(v1: Float32, i1: Int32, v2: Float32, i2: Int32, *, loc=None, ip=None):
    """The first pair if ``v1 > v2``, or ``v1 == v2`` with ``i1 < i2``
    (ordered compares), or ``v1`` is NaN and ``v2`` is not or ``i1 < i2``;
    else the second."""
    struct = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        _values((Float32(v1), Int32(i1), Float32(v2), Int32(i2)), loc=loc, ip=ip),
        "{\n\t.reg .pred pl, pk, pn;\n\t"
        "setp.lt.s32 pl, $3, $5;\n\t"
        "setp.eq.and.f32 pk, $2, $4, pl;\n\tsetp.gt.or.f32 pk, $2, $4, pk;\n\t"
        "setp.num.or.f32 pn, $4, $4, pl;\n\tsetp.nan.and.f32 pn, $2, $2, pn;\n\t"
        "or.pred pk, pk, pn;\n\t"
        "selp.f32 $0, $2, $4, pk;\n\tselp.b32 $1, $3, $5, pk;\n\t}",
        "=f,=r,f,r,f,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _pair(struct, (Float32, Int32), loc=loc, ip=ip)


@dsl_user_op
def _w1_row(token: Int64, last: int, *, loc=None, ip=None) -> Int64:
    """The w1 row a token's embedding is copied from: the token, clamped to
    ``last`` (the table's last row) as an unsigned value, so an id outside
    the table (negative included) reads its last row instead of memory
    around it. A valid id is unchanged."""
    return Int64(
        _asm(
            T.i64(),
            _values((Int64(token),), loc=loc, ip=ip),
            f"min.u64 $0, $1, {last};",
            "=l,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# The step's arithmetic, as Triton 3.8 compiles the eager loop for SM 10.x.

# libdevice tanhf under FTZ: an odd polynomial below |x| 0.6, else
# 1 - 2 / (2**(2 |x| log2 e) + 1) with the sign of x (1 from |x| 9.01).
_TANH = (
    "abs.ftz.f32 ax, z;\n\t"
    "setp.ltu.f32 ps, ax, 0f3F19999A;\n\t"
    "mul.rn.f32 z2, z, z;\n\t"
    "fma.rn.ftz.f32 s, 0f3C80F082, z2, 0fBD563CAE;\n\t"
    "fma.rn.ftz.f32 s, s, z2, 0f3E085941;\n\t"
    "fma.rn.ftz.f32 s, s, z2, 0fBEAAA9ED;\n\t"
    "fma.rn.ftz.f32 s, s, z2, 0f00000000;\n\t"
    "fma.rn.ftz.f32 s, s, z, z;\n\t"
    "mul.rn.f32 e, ax, 0f4038AA3B;\n\t"
    "ex2.approx.ftz.f32 e, e;\n\t"
    "add.rn.f32 d, e, 0f3F800000;\n\t"
    "rcp.approx.ftz.f32 g, d;\n\t"
    "fma.rn.ftz.f32 g, g, 0fC0000000, 0f3F800000;\n\t"
    "setp.ge.f32 pb, ax, 0f41102CB4;\n\t"
    "selp.f32 g, 0f3F800000, g, pb;\n\t"
    "mov.b32 gb, g;\n\t"
    "mov.b32 zb, z;\n\t"
    "and.b32 zb, zb, -2147483648;\n\t"
    "or.b32 gb, gb, zb;\n\t"
    "mov.b32 g, gb;\n\t"
    "selp.f32 w, s, g, ps;\n\t"
)


@dsl_user_op
def _logit(
    a_bits: Uint32,
    b_bits: Uint32,
    mult: Float32,
    inv_cap: Float32,
    cap: Float32,
    mask: Int32,
    has_cap: cutlass.Constexpr[bool],
    *,
    loc=None,
    ip=None,
):
    """(BF16 bits of the cached logit, the logit as FP32 or -inf off the mask)
    of BF16 ``a`` (base) and ``b`` (bias)."""
    if has_cap:
        transform = "mul.rn.f32 z, $5, y;\n\t" + _TANH + "mul.rn.f32 w, $6, w;\n\t"
    else:
        transform = "mov.f32 w, y;\n\t"
    struct = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        _values(
            (
                Uint32(a_bits),
                Uint32(b_bits),
                Float32(mult),
                Float32(inv_cap),
                Float32(cap),
                Int32(mask),
            ),
            loc=loc,
            ip=ip,
        ),
        "{\n\t.reg .pred pm, ps, pb;\n\t.reg .b16 ha, hb, hx;\n\t"
        ".reg .f32 x, y, z, ax, z2, s, e, d, g, w, l;\n\t.reg .b32 gb, zb;\n\t"
        "cvt.u16.u32 ha, $2;\n\tcvt.u16.u32 hb, $3;\n\t"
        "add.bf16 hx, ha, hb;\n\tcvt.f32.bf16 x, hx;\n\t"
        "mul.rn.f32 y, $4, x;\n\t" + transform + "cvt.rn.bf16.f32 hx, w;\n\t"
        "cvt.u32.u16 $0, hx;\n\tcvt.f32.bf16 l, hx;\n\t"
        "setp.ne.s32 pm, $7, 0;\n\tselp.f32 $1, l, 0fFF800000, pm;\n\t}",
        "=r,=f,r,r,f,f,f,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _pair(struct, (Uint32, Float32), loc=loc, ip=ip)


_MURMUR_MIX = (
    "mul.lo.s32 k, k, -862048943;\n\t"
    "shf.l.wrap.b32 k, k, k, 15;\n\t"
    "mul.lo.s32 k, k, 461845907;\n\t"
    "xor.b32 h, h, k;\n\t"
    "shf.l.wrap.b32 h, h, h, 13;\n\t"
    "mad.lo.s32 h, h, 5, -430675100;\n\t"
)


@dsl_user_op
def _murmur_prefix(
    seed_lo: Uint32, seed_hi: Uint32, pos_lo: Uint32, *, loc=None, ip=None
) -> Uint32:
    """MurmurHash3 state after the request's three words (seed low, seed high,
    position), from 0."""
    return Uint32(
        _asm(
            T.i32(),
            _values((Uint32(seed_lo), Uint32(seed_hi), Uint32(pos_lo)), loc=loc, ip=ip),
            "{\n\t.reg .b32 h, k;\n\tmov.u32 h, 0;\n\t"
            "mov.u32 k, $1;\n\t"
            + _MURMUR_MIX
            + "mov.u32 k, $2;\n\t"
            + _MURMUR_MIX
            + "mov.u32 k, $3;\n\t"
            + _MURMUR_MIX
            + "mov.u32 $0, h;\n\t}",
            "=r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def _logf(x: str, out: str, tag: str, addend: str) -> str:
    """Libdevice logf under FTZ of the normal-or-special ``x`` (the exponent
    term ``addend`` added), into ``out``; NaN, +-inf and negatives through
    ``fma(x, inf, inf)``."""
    return (
        f"mov.b32 i{tag}, {x};\n\t"
        f"add.s32 e{tag}, i{tag}, -1059760811;\n\t"
        f"and.b32 e{tag}, e{tag}, -8388608;\n\t"
        f"sub.s32 m{tag}, i{tag}, e{tag};\n\t"
        f"cvt.rn.f32.s32 fe{tag}, e{tag};\n\t"
        f"fma.rn.ftz.f32 t{tag}, fe{tag}, 0f34000000, {addend};\n\t"
        f"mov.b32 f{tag}, m{tag};\n\t"
        f"add.rn.f32 f{tag}, f{tag}, 0fBF800000;\n\t"
        f"fma.rn.ftz.f32 q{tag}, 0fBE055027, f{tag}, 0f3E1039F6;\n\t"
        f"fma.rn.ftz.f32 q{tag}, q{tag}, f{tag}, 0fBDF8CDCC;\n\t"
        f"fma.rn.ftz.f32 q{tag}, q{tag}, f{tag}, 0f3E0F2955;\n\t"
        f"fma.rn.ftz.f32 q{tag}, q{tag}, f{tag}, 0fBE2AD8B9;\n\t"
        f"fma.rn.ftz.f32 q{tag}, q{tag}, f{tag}, 0f3E4CED0B;\n\t"
        f"fma.rn.ftz.f32 q{tag}, q{tag}, f{tag}, 0fBE7FFF22;\n\t"
        f"fma.rn.ftz.f32 q{tag}, q{tag}, f{tag}, 0f3EAAAA78;\n\t"
        f"fma.rn.ftz.f32 q{tag}, q{tag}, f{tag}, 0fBF000000;\n\t"
        f"mul.rn.f32 q{tag}, f{tag}, q{tag};\n\t"
        f"fma.rn.ftz.f32 q{tag}, q{tag}, f{tag}, f{tag};\n\t"
        f"fma.rn.ftz.f32 {out}, t{tag}, 0f3F317218, q{tag};\n\t"
        f"setp.lt.u32 p{tag}, i{tag}, 2139095040;\n\t"
        f"@!p{tag} fma.rn.ftz.f32 {out}, {x}, 0f7F800000, 0f7F800000;\n\t"
    )


def _logf_regs(tag: str) -> str:
    return (
        f".reg .pred p{tag};\n\t.reg .b32 i{tag}, e{tag}, m{tag};\n\t"
        f".reg .f32 fe{tag}, t{tag}, f{tag}, q{tag};\n\t"
    )


@dsl_user_op
def _gumbel(
    logit: Float32,
    temp: Float32,
    prefix: Uint32,
    gid: Int32,
    mask: Int32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """``logit / temp`` plus the Gumbel noise of vocab id ``gid``
    (``gumbel_noised_argmax`` with fp32 noise while drafting), or -inf off
    the mask. ``prefix`` is ``_murmur_prefix`` of the row."""
    return Float32(
        _asm(
            T.f32(),
            _values(
                (
                    Float32(logit),
                    Float32(temp),
                    Uint32(prefix),
                    Int32(gid),
                    Int32(mask),
                ),
                loc=loc,
                ip=ip,
            ),
            "{\n\t.reg .pred ps, pg, pz, pm;\n\t.reg .b32 h, k, hi, lo;\n\t"
            ".reg .f32 fh, fl, u, poly, nu, ser, om, xa, la, sel, nsel, xb, adj, lb, "
            "q;\n\t"
            + _logf_regs("A")
            + _logf_regs("B")
            # MurmurHash3 of the vocab id, finalized; its uniform in (0, 1).
            + "mov.u32 h, $3;\n\tmov.u32 k, $4;\n\t"
            + _MURMUR_MIX
            + "xor.b32 h, h, 16;\n\t"
            "shr.u32 k, h, 16;\n\txor.b32 h, h, k;\n\t"
            "mul.lo.s32 h, h, -2048144789;\n\t"
            "shr.u32 k, h, 13;\n\txor.b32 h, h, k;\n\t"
            "mul.lo.s32 h, h, -1028477387;\n\t"
            "shr.u32 k, h, 16;\n\txor.b32 h, h, k;\n\t"
            "shr.u32 hi, h, 16;\n\tand.b32 lo, h, 65535;\n\t"
            "cvt.rn.f32.u32 fh, hi;\n\tcvt.rn.f32.u32 fl, lo;\n\t"
            "add.rn.f32 fl, fl, 0f3F000000;\n\tmul.rn.f32 fl, fl, 0f2F800000;\n\t"
            "fma.rn.f32 u, fh, 0f37800000, fl;\n\t"
            # log(1 - u): the series below 0.25, else logf.
            "fma.rn.f32 poly, u, 0f3E000000, 0f3E124925;\n\t"
            "fma.rn.f32 poly, u, poly, 0f3E2AAAAB;\n\t"
            "fma.rn.f32 poly, u, poly, 0f3E4CCCCD;\n\t"
            "fma.rn.f32 poly, u, poly, 0f3E800000;\n\t"
            "fma.rn.f32 poly, u, poly, 0f3EAAAAAB;\n\t"
            "fma.rn.f32 poly, u, poly, 0f3F000000;\n\t"
            "fma.rn.f32 poly, u, poly, 0f3F800000;\n\t"
            "neg.f32 nu, u;\n\tmul.rn.f32 ser, poly, nu;\n\t"
            "sub.rn.f32 om, 0f3F800000, u;\n\tmax.f32 xa, om, 0f33800000;\n\t"
            + _logf("xa", "la", "A", "0f00000000")
            + "setp.lt.f32 ps, u, 0f3E800000;\n\tselp.f32 sel, ser, la, ps;\n\t"
            # log(-sel), subnormals scaled by 2**23 first.
            "setp.gt.f32 pg, sel, 0f80800000;\n\tneg.f32 nsel, sel;\n\t"
            "mul.rn.f32 xb, sel, 0fCB000000;\n\tselp.f32 xb, xb, nsel, pg;\n\t"
            "selp.f32 adj, 0fC1B80000, 0f00000000, pg;\n\t"
            + _logf("xb", "lb", "B", "adj")
            + "setp.eq.f32 pz, xb, 0f00000000;\n\tselp.f32 lb, 0fFF800000, lb, pz;\n\t"
            # logit / temp - log(-log(1 - u)), -inf off the mask.
            "div.full.f32 q, $1, $2;\n\tsub.rn.f32 q, q, lb;\n\t"
            "setp.ne.s32 pm, $5, 0;\n\tselp.f32 $0, q, 0fFF800000, pm;\n\t}",
            "=f,f,f,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nonzero_f32(value: Float32, *, loc=None, ip=None) -> Int32:
    """1 where ``value != 0.0`` (ordered, as Triton's ``temp != 0.0``)."""
    return Int32(
        _asm(
            T.i32(),
            _values((Float32(value),), loc=loc, ip=ip),
            "{\n\t.reg .pred p;\n\tsetp.eq.f32 p, $1, 0f00000000;\n\t"
            "selp.u32 $0, 0, 1, p;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# Device building blocks.


@cute.jit
def _copy_row(
    source: Int64,
    destination: Int64,
    tidx: Int32,
    threads: cutlass.Constexpr[int],
    row_bytes: cutlass.Constexpr[int],
    chunk: cutlass.Constexpr[int],
):
    """Copy ``row_bytes`` from ``source`` to ``destination`` with ``threads``
    threads in ``chunk``-byte accesses."""
    per_pass = threads * chunk
    offset = Int64(tidx * chunk)
    if const_expr(row_bytes >= per_pass):
        _copy(
            [
                (
                    source + offset + rep * per_pass,
                    destination + offset + rep * per_pass,
                    Int32(1),
                )
                for rep in range(row_bytes // per_pass)
            ],
            chunk,
        )
    else:
        _copy(
            [(source + offset, destination + offset, Int32(tidx * chunk < row_bytes))],
            chunk,
        )


def _load_bf16(row: Int64, offset: list, mask: list, vector: bool) -> list:
    """A thread's BF16 elements of a row (the bits in a 32-bit word each),
    one load per element, or with ``vector`` (consecutive elements, which
    share a mask) one load of all of them (traced inline)."""
    if not vector:
        return [_ld_u16(row + offset[k], mask[k]) for k in range(len(offset))]
    bits = []
    for word in _load_words(row + offset[0], len(offset) * 2, mask[0]):
        # _logit reads the low 16 bits of an element's word.
        bits += [word, word >> Uint32(16)]
    return bits


def _store_bf16(
    row: Int64, offset: list, store: list, bits: list, vector: bool
) -> None:
    """Store a thread's BF16 elements as ``_load_bf16`` loads them."""
    if not vector:
        for k in range(len(offset)):
            _st_u16(row + offset[k], bits[k], store[k])
        return
    words = [bits[j] | (bits[j + 1] << Uint32(16)) for j in range(0, len(bits), 2)]
    _store_words(row + offset[0], words, len(offset) * 2, store[0])


def _lane_results(
    block_max: Int64, block_argmax: Int64, lane: Int32, blocks: int, threads: int
) -> list:
    """The (FP32 value, global id) results ``lane, lane + threads, ...`` of
    ``blocks``, all loaded at once (traced inline); an id is the low word of
    its int64. Past the last block a lane loads (-inf, BIG_INDEX): its id is
    above every block's, so a block's -inf wins over it."""
    got = []
    for q in range(-(-blocks // threads)):
        index = lane + q * threads
        valid = Int32(1) if (q + 1) * threads <= blocks else Int32(index < blocks)
        bits = _ld_volatile_b32_or(
            block_max + Int64(index) * 4, valid, _bits(Float32(NEG_INF))
        )
        gid = _ld_volatile_b32_or(
            block_argmax + Int64(index) * 8, valid, Uint32(BIG_INDEX)
        )
        got.append((_float(bits), Int32(gid)))
    return got


def _tree_argmax(pairs: list, combine) -> tuple:
    """The first NaN, else the first maximum, of (value, index) pairs,
    reduced level by level as a tree (traced inline); ``combine`` is
    ``_combine_in_order`` when the pairs are in index order."""
    while len(pairs) > 1:
        pairs = [
            combine(*pairs[j], *pairs[j + 1]) if j + 1 < len(pairs) else pairs[j]
            for j in range(0, len(pairs), 2)
        ]
    return pairs[0]


@cute.jit
def _warp_argmax(value: Float32, index: Int32, top: cutlass.Constexpr[int]):
    """Butterfly over the warp's lanes (``top`` .. 1): every lane ends with
    the first NaN, else the first maximum, of the lanes' pairs."""
    offset = top
    for _ in cutlass.range_constexpr(top.bit_length()):
        other_v = cute.arch.shuffle_sync_bfly(value, offset=offset)
        other_i = cute.arch.shuffle_sync_bfly(index, offset=offset)
        value, index = _combine(value, index, other_v, other_i)
        offset //= 2
    return value, index


class _StepKernel:
    """Grid (rows, blocks), THREADS threads, ``block`` vocab entries per CTA;
    ``blocks`` (the vocab width's) is a compile-time constant."""

    def __init__(
        self,
        *,
        block: int,
        blocks: int,
        vector: bool,
        has_cap: bool,
        sharded: bool,
        pdl: bool,
        row_bytes: int,
        chunk: int,
        w1_rows: int,
    ) -> None:
        assert block % THREADS == 0 and block & (block - 1) == 0
        self.block, self.blocks = block, blocks
        # Thread t holds per_thread of the block's elements: with vector the
        # ones from t * per_thread on, else t, t + THREADS, ...
        self.per_thread = block // THREADS
        # vector: the base, bias and cache rows are read and written a
        # thread's elements at a time.
        assert not vector or vector_bytes(block)
        self.vector = vector
        # The warps of the row's last block that reduce the blocks' results
        # (and copy the w1 row).
        self.finish_warps = finish_warps(blocks)
        self.has_cap, self.sharded, self.pdl = has_cap, sharded, pdl
        # The last block's warp 0 copies the w1 row in chunk-byte accesses,
        # from one of the table's w1_rows rows.
        self.row_bytes, self.chunk, self.w1_rows = row_bytes, chunk, w1_rows

    @cute.jit
    def __call__(
        self,
        base: Int64,
        base_s0: Int64,
        bias: Int64,
        bias_s0: Int64,
        cache: Int64,
        cache_s0: Int64,
        cache_s1: Int64,
        col: Int32,
        idx_map: Int64,
        idx_map_s0: Int64,
        pos: Int64,
        pos_s0: Int64,
        seeds: Int64,
        temperature: Int64,
        block_argmax: Int64,
        block_max: Int64,
        arrivals: Int64,
        tokens: Int64,
        tokens_s0: Int64,
        w1: Int64,
        emb: Int64,
        packets: Int64,
        width: Int32,
        vocab_offset: Int32,
        mult: Float32,
        inv_cap: Float32,
        cap: Float32,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            base,
            base_s0,
            bias,
            bias_s0,
            cache,
            cache_s0,
            cache_s1,
            col,
            idx_map,
            idx_map_s0,
            pos,
            pos_s0,
            seeds,
            temperature,
            block_argmax,
            block_max,
            arrivals,
            tokens,
            tokens_s0,
            w1,
            emb,
            packets,
            width,
            vocab_offset,
            mult,
            inv_cap,
            cap,
        ).launch(
            grid=(rows, self.blocks, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        base: Int64,
        base_s0: Int64,
        bias: Int64,
        bias_s0: Int64,
        cache: Int64,
        cache_s0: Int64,
        cache_s1: Int64,
        col: Int32,
        idx_map: Int64,
        idx_map_s0: Int64,
        pos: Int64,
        pos_s0: Int64,
        seeds: Int64,
        temperature: Int64,
        block_argmax: Int64,
        block_max: Int64,
        arrivals: Int64,
        tokens: Int64,
        tokens_s0: Int64,
        w1: Int64,
        emb: Int64,
        packets: Int64,
        width: Int32,
        vocab_offset: Int32,
        mult: Float32,
        inv_cap: Float32,
        cap: Float32,
    ) -> None:
        block, blocks = self.block, self.blocks
        per_thread = self.per_thread
        tidx, _, _ = cute.arch.thread_idx()
        r, blk, _ = cute.arch.block_idx()
        warp = tidx // WARP
        lane = tidx % WARP
        smem = cu_utils.SmemAllocator()
        warp_values = smem.allocate_array(Uint32, 4)
        warp_indices = smem.allocate_array(Uint32, 4)
        word = smem.allocate_array(Uint32, 4)
        row = Int64(r)

        # Element k of this thread: consecutive elements for one vector
        # access, else strided by THREADS so that a warp's element accesses
        # are coalesced. Either way the index rises with k, as the in-order
        # combine of the thread's reduction needs.
        index = []
        loc = []
        mask = []
        for k in cutlass.range_constexpr(per_thread):
            if const_expr(self.vector):
                element = tidx * per_thread + k
            else:
                element = k * THREADS + tidx
            index.append(Int32(element))
            loc.append(blk * block + element)
            mask.append(Int32(blk * block + element < width))
        # Byte offsets of the elements in a row: never negative, so
        # zero-extended (one wide multiply-add per address).
        offset = [Int64(Uint32(loc[k])) * 2 for k in range(per_thread)]

        # Inputs written before the loop: read while the w2 GEMV still runs.
        base_row = base + row * base_s0 * 2
        a_bits = _load_bf16(base_row, offset, mask, self.vector)
        req = load_global_s64_address(
            idx_map + row * idx_map_s0 * 8, Int32(1), ordered=True
        )
        position = load_global_s64_address(
            pos + row * pos_s0 * 8, Int32(1), ordered=True
        )
        position = position - Int64(1)
        valid = Int32(req >= Int64(0))
        bias_row = bias + row * bias_s0 * 2
        # The request's temperature and seed load once its index has
        # arrived; the loads issue in program order, so without PDL they
        # follow the bias loads instead of holding them back.
        if const_expr(self.pdl):
            temp = _float(
                load_global_u32_address(temperature + req * 4, valid, ordered=True)
            )
            seed = load_global_s64_address(seeds + req * 8, valid, ordered=True)
            # The bias is the GEMV's output; every store also waits for it.
            cute.arch.griddepcontrol_wait()
            b_bits = _load_bf16(bias_row, offset, mask, self.vector)
        else:
            b_bits = _load_bf16(bias_row, offset, mask, self.vector)
            temp = _float(
                load_global_u32_address(temperature + req * 4, valid, ordered=True)
            )
            seed = load_global_s64_address(seeds + req * 8, valid, ordered=True)
        values = cute.make_rmem_tensor(cute.make_layout((per_thread,)), Float32)
        if const_expr(self.sharded):
            cache_row = cache + (row * cache_s0 + Int64(col) * cache_s1) * 2
        else:
            cache_row = cache + (req * cache_s0 + Int64(col) * cache_s1) * 2
        cache_bits, store = [], []
        for k in cutlass.range_constexpr(per_thread):
            bits, logit = _logit(
                a_bits[k], b_bits[k], mult, inv_cap, cap, mask[k], self.has_cap
            )
            values[k] = logit
            cache_bits.append(bits)
            store.append(mask[k] if const_expr(self.sharded) else mask[k] & valid)
        _store_bf16(cache_row, offset, store, cache_bits, self.vector)

        if _nonzero_f32(temp) != Int32(0):
            # The step seed of gumbel_block_argmax when it writes a logits
            # cache, and the draft's noise salt.
            step_seed = seed ^ (Int64(col) << Int64(32))
            salted = position + Int64(1 << 30)
            prefix = _murmur_prefix(
                Uint32(step_seed & Int64(0xFFFFFFFF)),
                Uint32((step_seed >> Int64(32)) & Int64(0xFFFFFFFF)),
                Uint32(salted & Int64(0xFFFFFFFF)),
            )
            for k in cutlass.range_constexpr(per_thread):
                values[k] = _gumbel(
                    values[k], temp, prefix, vocab_offset + loc[k], mask[k]
                )

        # This thread's elements in pairs, then the warp, then the 4 warps.
        value, idx = _tree_argmax(
            [(values[k], index[k]) for k in range(per_thread)], _combine_in_order
        )
        value, idx = _warp_argmax(value, idx, WARP // 2)
        # Every lane holds its warp's pair and stores it to the warp's slot.
        _st_shared_b32(Int32(warp_values.toint()) + warp * 4, _bits(value))
        _st_shared_b32(Int32(warp_indices.toint()) + warp * 4, Uint32(idx))
        cute.arch.barrier()
        value = _float(_ld_shared_b32(Int32(warp_values.toint()) + (tidx % 4) * 4))
        idx = Int32(_ld_shared_b32(Int32(warp_indices.toint()) + (tidx % 4) * 4))
        value, idx = _warp_argmax(value, idx, 2)
        slot = row * Int64(blocks) + Int64(blk)
        if tidx == 0:
            _st_b64(block_argmax + slot * 8, Int64(vocab_offset + blk * block + idx))
            store_global_u32_address(block_max + slot * 4, _bits(value))

        # Release this block's result; the block that completes the row's
        # count (blocks per step, from zero per call) acquires all of them.
        cute.arch.barrier()
        if tidx == 0:
            ticket = atomic_add1_acq_rel_u32_address(arrivals + row * 4)
            _st_shared_b32(Int32(word.toint()), ticket)
        cute.arch.barrier()
        arrival = Int32(_ld_shared_b32(Int32(word.toint())))
        if ((arrival + 1) % blocks == 0) & (warp < self.finish_warps):
            self._finish(
                tidx,
                warp,
                lane,
                Int32(warp_values.toint()),
                Int32(warp_indices.toint()),
                row,
                col,
                block_argmax,
                block_max,
                tokens,
                tokens_s0,
                w1,
                emb,
                packets,
            )

    @cute.jit
    def _finish(
        self,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        warp_values: Int32,
        warp_indices: Int32,
        row: Int64,
        col: Int32,
        block_argmax: Int64,
        block_max: Int64,
        tokens: Int64,
        tokens_s0: Int64,
        w1: Int64,
        emb: Int64,
        packets: Int64,
    ) -> None:
        """The first ``finish_warps`` warps of the row's last block: the
        first NaN, else the first maximum, over the blocks' (value, global
        id) results (a global id rises with the block), then the token and
        next w1 row, or the packet ``[value bits, global id, 0, 0]``. Every
        thread loads its share of the results at once, then reduces them,
        then the warps, then (several warps) their results in shared
        memory; the combine is a total order, so every lane ends with the
        winner and no load waits for it."""
        blocks, threads = self.blocks, self.finish_warps * WARP
        first = row * Int64(blocks)
        got = _lane_results(
            block_max + first * 4, block_argmax + first * 8, tidx, blocks, threads
        )
        best_v, best_id = _tree_argmax(got, _combine_in_order)
        best_v, best_id = _warp_argmax(best_v, best_id, WARP // 2)
        if const_expr(self.finish_warps > 1):
            if lane == 0:
                _st_shared_b32(warp_values + warp * 4, _bits(best_v))
                _st_shared_b32(warp_indices + warp * 4, Uint32(best_id))
            cute.arch.barrier()
            best_v, best_id = _tree_argmax(
                [
                    (
                        _float(_ld_shared_b32(warp_values + w * 4)),
                        Int32(_ld_shared_b32(warp_indices + w * 4)),
                    )
                    for w in range(self.finish_warps)
                ],
                _combine,
            )
        if const_expr(self.sharded):
            if tidx == 0:
                packet = packets + row * 16
                store_global_u32_address(packet, _bits(best_v))
                store_global_u32_address(packet + 4, Uint32(best_id))
                store_global_u32_address(packet + 8, Uint32(0))
                store_global_u32_address(packet + 12, Uint32(0))
        else:
            token = Int64(best_id)
            if tidx == 0:
                _st_b64(tokens + (row * tokens_s0 + Int64(col)) * 8, token)
            _copy_row(
                w1 + _w1_row(token, self.w1_rows - 1) * self.row_bytes,
                emb + row * self.row_bytes,
                tidx,
                threads,
                self.row_bytes,
                self.chunk,
            )


class _PickKernel:
    """Grid (rows,), one warp: the winner over the gathered ``[tp, rows, 4]``
    packets (the first NaN, else the first maximum, by global id: with
    shard k holding the ids above shard k - 1's, the unsharded argmax)."""

    def __init__(
        self, *, tp: int, pdl: bool, row_bytes: int, chunk: int, w1_rows: int
    ) -> None:
        assert tp & (tp - 1) == 0 and 1 <= tp <= WARP
        self.tp, self.pdl = tp, pdl
        self.row_bytes, self.chunk, self.w1_rows = row_bytes, chunk, w1_rows

    @cute.jit
    def __call__(
        self,
        packets: Int64,
        rows: Int32,
        tokens: Int64,
        tokens_s0: Int64,
        step: Int32,
        w1: Int64,
        emb: Int64,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(packets, rows, tokens, tokens_s0, step, w1, emb).launch(
            grid=(rows, 1, 1),
            block=(PICK_THREADS, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        packets: Int64,
        rows: Int32,
        tokens: Int64,
        tokens_s0: Int64,
        step: Int32,
        w1: Int64,
        emb: Int64,
    ) -> None:
        if const_expr(self.pdl):
            # The packets are the all-gather's output.
            cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        r, _, _ = cute.arch.block_idx()
        row = Int64(r)
        rank = Int64(tidx % self.tp)
        packet = packets + ((rank * Int64(rows) + row) * 4) * 4
        value = _float(load_global_u32_address(packet, Int32(1), ordered=True))
        gid = Int32(load_global_u32_address(packet + 4, Int32(1), ordered=True))
        # Lane l holds rank l % tp: a butterfly over tp lanes reduces the
        # ranks' packets in every lane.
        _, winner = _warp_argmax(value, gid, self.tp // 2)
        token = Int64(winner)
        if tidx == 0:
            _st_b64(tokens + (row * tokens_s0 + Int64(step)) * 8, token)
        _copy_row(
            w1 + _w1_row(token, self.w1_rows - 1) * self.row_bytes,
            emb + row * self.row_bytes,
            tidx,
            PICK_THREADS,
            self.row_bytes,
            self.chunk,
        )


class _AnchorKernel:
    """Grid (rows,), THREADS threads: step 0's Markov embedding (the w1 row
    of the anchor token) and the row's arrival count reset."""

    def __init__(self, *, row_bytes: int, chunk: int, w1_rows: int) -> None:
        self.row_bytes, self.chunk, self.w1_rows = row_bytes, chunk, w1_rows

    @cute.jit
    def __call__(
        self,
        input_ids: Int64,
        anchor_idx: Int64,
        w1: Int64,
        emb: Int64,
        arrivals: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(input_ids, anchor_idx, w1, emb, arrivals).launch(
            grid=(rows, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        input_ids: Int64,
        anchor_idx: Int64,
        w1: Int64,
        emb: Int64,
        arrivals: Int64,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        r, _, _ = cute.arch.block_idx()
        row = Int64(r)
        anchor = load_global_s64_address(anchor_idx + row * 8, Int32(1), ordered=True)
        token = load_global_u32_address(input_ids + anchor * 4, Int32(1), ordered=True)
        token = Int64(Int32(token))
        _copy_row(
            w1 + _w1_row(token, self.w1_rows - 1) * self.row_bytes,
            emb + row * self.row_bytes,
            tidx,
            THREADS,
            self.row_bytes,
            self.chunk,
        )
        if tidx == 0:
            # markov_step counts the row's arriving blocks from zero per call.
            store_global_u32_address(arrivals + row * 4, Uint32(0))


def _fill_entries(
    src: Int64,
    src_s0: Int64,
    row: Int64,
    target: Int64,
    first: Int64,
    shard: Int32,
    vocab: Int32,
) -> list:
    """The cache fill's one-entry copies of a thread (traced inline)."""
    copies = []
    for e in range(FILL_BLOCK // THREADS):
        c = first + e
        k = c // Int64(shard)
        local = c - k * Int64(shard)
        copies.append(
            (
                src + (k * src_s0 + row + local) * 2,
                target + c * 2,
                Int32(c < Int64(vocab)),
            )
        )
    return copies


class _CacheFillKernel:
    """Grid (rows * steps, vocab blocks), THREADS threads of FILL_BLOCK //
    THREADS consecutive entries: scatter the gathered ``[tp, rows, steps,
    shard]`` draft logits (rank stride ``src_s0``) into the full-vocab
    cache. ``vectorized``: one 16-byte copy per thread (shard, vocab,
    strides and pointers 16-byte aligned)."""

    def __init__(self, *, vectorized: bool) -> None:
        self.vectorized = vectorized

    @cute.jit
    def __call__(
        self,
        src: Int64,
        src_s0: Int64,
        cache: Int64,
        cache_s0: Int64,
        cache_s1: Int64,
        idx_map: Int64,
        steps: Int32,
        shard: Int32,
        vocab: Int32,
        grid_x: Int32,
        grid_y: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            src, src_s0, cache, cache_s0, cache_s1, idx_map, steps, shard, vocab
        ).launch(grid=(grid_x, grid_y, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        src: Int64,
        src_s0: Int64,
        cache: Int64,
        cache_s0: Int64,
        cache_s1: Int64,
        idx_map: Int64,
        steps: Int32,
        shard: Int32,
        vocab: Int32,
    ) -> None:
        per_thread = FILL_BLOCK // THREADS
        tidx, _, _ = cute.arch.thread_idx()
        rs, blk, _ = cute.arch.block_idx()
        r = rs // steps
        i = rs - r * steps
        req = load_global_s64_address(idx_map + Int64(rs) * 8, Int32(1), ordered=True)
        if req >= Int64(0):
            first = Int64(blk) * FILL_BLOCK + Int64(tidx * per_thread)
            row = Int64(rs) * Int64(shard)
            target = cache + (req * cache_s0 + Int64(i) * cache_s1) * 2
            if const_expr(self.vectorized):
                k = first // Int64(shard)
                local = first - k * Int64(shard)
                _copy(
                    [
                        (
                            src + (k * src_s0 + row + local) * 2,
                            target + first * 2,
                            Int32(first < Int64(vocab)),
                        )
                    ],
                    per_thread * 2,
                )
            else:
                _copy(_fill_entries(src, src_s0, row, target, first, shard, vocab), 2)


# Host side.


class Operand(NamedTuple):
    """Where a BF16 operand of the step lies: its byte address and element
    strides, which decide whether the step moves it in vectors
    (``vectorized``)."""

    address: int
    strides: tuple[int, ...]

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "Operand":
        assert tensor.dtype == torch.bfloat16, tensor.dtype
        return cls(tensor.data_ptr(), tuple(tensor.stride()))


def _aligned(address: int) -> bool:
    return address % VECTOR_BYTES == 0


def finish_warps(blocks: int) -> int:
    """Warps of a step's last block that reduce its ``blocks`` results:
    warp 0 up to four results per lane, else all four warps."""
    return 1 if blocks <= 4 * WARP else THREADS // WARP


def vector_bytes(block: int) -> int:
    """The bytes of one thread's ``block / 128`` elements where a step at
    ``block`` may move them as one 8- or 16-byte access, else 0."""
    nbytes = block // THREADS * 2
    return nbytes if 8 <= nbytes <= VECTOR_BYTES else 0


def vectorized(
    block: int,
    *,
    width: int,
    col: int,
    base: Operand,
    bias: Operand,
    cache: Operand,
) -> bool:
    """Whether a step at ``block`` reads and writes its base, bias and cache
    rows a thread's ``block / 128`` elements at a time (``vector_bytes``):
    every such group then starts aligned to its bytes and lies wholly inside
    or outside the width. The values are the same either way."""
    nbytes = vector_bytes(block)
    per_thread = nbytes // 2
    return (
        nbytes > 0
        and width % per_thread == 0
        and all(
            t.address % nbytes == 0 and t.strides[0] % per_thread == 0
            for t in (base, bias, cache)
        )
        and col * cache.strides[1] % per_thread == 0
    )


def fill_vectorized(
    *, gathered: int, src_s0: int, cache: Operand, shard: int, vocab: int
) -> bool:
    """Whether the cache fill copies 16-byte vectors: the shard, the vocab,
    the strides and the pointers (``gathered`` the address of the gathered
    logits) 16-byte aligned."""
    return (
        shard % 8 == 0
        and vocab % 8 == 0
        and src_s0 % 8 == 0
        and cache.strides[0] % 8 == 0
        and cache.strides[1] % 8 == 0
        and _aligned(gathered)
        and _aligned(cache.address)
    )


def copy_chunk(row_bytes: int, threads: int, *tensors: torch.Tensor) -> int:
    """Bytes per access of a row copy by ``threads`` threads: up to 16, no
    more than a thread's share (2 when a row is narrower than the threads),
    within the alignment of every row pointer. The chunk divides the row
    bytes (both powers of two), so it is the same for every row of a
    row-major table."""
    chunk = 2
    while chunk < VECTOR_BYTES and chunk * 2 * threads <= row_bytes:
        chunk *= 2
    while any(t.data_ptr() % chunk for t in tensors):
        chunk //= 2
    return chunk


@functools.cache
def compile_step(
    *,
    block: int,
    blocks: int,
    vector: bool,
    has_cap: bool,
    sharded: bool,
    pdl: bool,
    row_bytes: int,
    chunk: int,
    w1_rows: int,
):
    """cute.compile one step kernel (before CUDA graph capture)."""
    assert_not_capturing("the Markov step kernel")
    i64, i32, f32 = Int64(0), Int32(0), Float32(0.0)
    return cute.compile(
        _StepKernel(
            block=block,
            blocks=blocks,
            vector=vector,
            has_cap=has_cap,
            sharded=sharded,
            pdl=pdl,
            row_bytes=row_bytes,
            chunk=chunk,
            w1_rows=w1_rows,
        ),
        *([i64] * 7),
        i32,
        *([i64] * 14),
        i32,
        i32,
        f32,
        f32,
        f32,
        Int32(1),
        current_cu_stream(),
    )


@functools.cache
def compile_pick(*, tp: int, pdl: bool, row_bytes: int, chunk: int, w1_rows: int):
    assert_not_capturing("the Markov pick kernel")
    i64, i32 = Int64(0), Int32(0)
    return cute.compile(
        _PickKernel(tp=tp, pdl=pdl, row_bytes=row_bytes, chunk=chunk, w1_rows=w1_rows),
        i64,
        Int32(1),
        i64,
        i64,
        i32,
        i64,
        i64,
        current_cu_stream(),
    )


@functools.cache
def compile_anchor(*, row_bytes: int, chunk: int, w1_rows: int):
    assert_not_capturing("the Markov anchor kernel")
    i64 = Int64(0)
    return cute.compile(
        _AnchorKernel(row_bytes=row_bytes, chunk=chunk, w1_rows=w1_rows),
        *([i64] * 5),
        Int32(1),
        current_cu_stream(),
    )


@functools.cache
def compile_cache_fill(*, vectorized: bool):
    assert_not_capturing("the Markov cache fill kernel")
    i64, i32 = Int64(0), Int32(1)
    return cute.compile(
        _CacheFillKernel(vectorized=vectorized),
        *([i64] * 6),
        i32,
        i32,
        i32,
        i32,
        i32,
        current_cu_stream(),
    )


def _row_copy(w1: torch.Tensor, emb: torch.Tensor, threads: int) -> tuple[int, int]:
    row_bytes = w1.shape[1] * w1.element_size()
    return row_bytes, copy_chunk(row_bytes, threads, w1, emb)


def _launch_step(
    kernel,
    base: torch.Tensor,
    bias: torch.Tensor,
    cache: torch.Tensor,
    col: int,
    idx_map: torch.Tensor,
    pos: torch.Tensor,
    seeds: torch.Tensor,
    temperature: torch.Tensor,
    block_argmax: torch.Tensor,
    block_max: torch.Tensor,
    arrivals: torch.Tensor,
    tokens: torch.Tensor,
    w1: torch.Tensor,
    emb: torch.Tensor,
    packets: torch.Tensor,
    width: int,
    vocab_offset: int,
    mult: float,
    inv_cap: float,
    cap: float,
) -> None:
    kernel(
        Int64(base.data_ptr()),
        Int64(base.stride(0)),
        Int64(bias.data_ptr()),
        Int64(bias.stride(0)),
        Int64(cache.data_ptr()),
        Int64(cache.stride(0)),
        Int64(cache.stride(1)),
        Int32(col),
        Int64(idx_map.data_ptr()),
        Int64(idx_map.stride(0)),
        Int64(pos.data_ptr()),
        Int64(pos.stride(0)),
        Int64(seeds.data_ptr()),
        Int64(temperature.data_ptr()),
        Int64(block_argmax.data_ptr()),
        Int64(block_max.data_ptr()),
        Int64(arrivals.data_ptr()),
        Int64(tokens.data_ptr()),
        Int64(tokens.stride(0)),
        Int64(w1.data_ptr()),
        Int64(emb.data_ptr()),
        Int64(packets.data_ptr()),
        Int32(width),
        Int32(vocab_offset),
        Float32(mult),
        Float32(inv_cap),
        Float32(cap),
        Int32(base.shape[0]),
        current_cu_stream(),
    )


def _launch_pick(
    kernel,
    packets: torch.Tensor,
    rows: int,
    tokens: torch.Tensor,
    step: int,
    w1: torch.Tensor,
    emb: torch.Tensor,
) -> None:
    kernel(
        Int64(packets.data_ptr()),
        Int32(rows),
        Int64(tokens.data_ptr()),
        Int64(tokens.stride(0)),
        Int32(step),
        Int64(w1.data_ptr()),
        Int64(emb.data_ptr()),
        current_cu_stream(),
    )


class MarkovKernels:
    """The kernels of one loop over the BF16 Markov w1 table ``w1`` ``[V,
    R]`` with Markov embedding rows in ``embeds`` and ``width`` vocab entries
    per rank (a shard over ``tp`` ranks, or the whole vocab with ``tp``
    None), compiled here, before CUDA graph capture: the anchor, the step at
    each ``(block, PDL, vector)`` of ``steps``, and with a vocab shard
    the pick (``pick_pdl``) and the cache fill (``fill_vector``). A step at
    another ``(block, PDL, vector)`` compiles when it first runs (never under
    capture)."""

    def __init__(
        self,
        w1: torch.Tensor,
        embeds: torch.Tensor,
        *,
        width: int,
        has_cap: bool,
        tp: int | None,
        steps: Iterable[tuple[int, bool, bool]],
        pick_pdl: bool,
        fill_vector: bool,
    ) -> None:
        assert w1.dtype == embeds.dtype == torch.bfloat16
        self.w1, self.embeds, self.width, self.has_cap = w1, embeds, width, has_cap
        self.sharded = tp is not None
        row_bytes, chunk = _row_copy(w1, embeds, THREADS)
        compile_anchor(row_bytes=row_bytes, chunk=chunk, w1_rows=w1.shape[0])
        self._steps: dict[tuple[int, bool, bool], Any] = {}
        for variant in steps:
            self._step(*variant)
        if tp is not None:
            row_bytes, chunk = _row_copy(w1, embeds, PICK_THREADS)
            self._pick = compile_pick(
                tp=tp,
                pdl=pick_pdl,
                row_bytes=row_bytes,
                chunk=chunk,
                w1_rows=w1.shape[0],
            )
            compile_cache_fill(vectorized=fill_vector)

    @property
    def step_variants(self) -> frozenset[tuple[int, bool, bool]]:
        """The ``(block, PDL, vector)`` of the compiled steps."""
        return frozenset(self._steps)

    def _step(self, block: int, pdl: bool, vector: bool):
        kernel = self._steps.get((block, pdl, vector))
        if kernel is None:
            blocks = -(-self.width // block)
            row_bytes, chunk = _row_copy(
                self.w1, self.embeds, WARP * finish_warps(blocks)
            )
            kernel = self._steps[(block, pdl, vector)] = compile_step(
                block=block,
                blocks=blocks,
                vector=vector,
                has_cap=self.has_cap,
                sharded=self.sharded,
                pdl=pdl,
                row_bytes=row_bytes,
                chunk=chunk,
                w1_rows=self.w1.shape[0],
            )
        return kernel

    def step(
        self,
        base: torch.Tensor,
        bias: torch.Tensor,
        cache: torch.Tensor,
        col: int,
        idx_map: torch.Tensor,
        pos: torch.Tensor,
        seeds: torch.Tensor,
        temperature: torch.Tensor,
        block_argmax: torch.Tensor,
        block_max: torch.Tensor,
        arrivals: torch.Tensor,
        tokens: torch.Tensor,
        emb: torch.Tensor,
        packets: torch.Tensor,
        vocab_offset: int,
        mult: float,
        inv_cap: float,
        cap: float,
        *,
        block: int,
        pdl: bool,
    ) -> None:
        """``markov_step`` over this loop's w1 and width, ``emb`` a row view
        of ``embeds``; the caller checks the operands' dtypes."""
        vector = vectorized(
            block,
            width=self.width,
            col=col,
            base=Operand(base.data_ptr(), base.stride()),
            bias=Operand(bias.data_ptr(), bias.stride()),
            cache=Operand(cache.data_ptr(), cache.stride()),
        )
        _launch_step(
            self._step(block, pdl, vector),
            base,
            bias,
            cache,
            col,
            idx_map,
            pos,
            seeds,
            temperature,
            block_argmax,
            block_max,
            arrivals,
            tokens,
            self.w1,
            emb,
            packets,
            self.width,
            vocab_offset,
            mult,
            inv_cap,
            cap,
        )

    def pick(
        self,
        packets: torch.Tensor,
        rows: int,
        tokens: torch.Tensor,
        step: int,
        emb: torch.Tensor,
    ) -> None:
        """``markov_pick`` over this loop's w1, ``emb`` a row view of
        ``embeds``."""
        _launch_pick(self._pick, packets, rows, tokens, step, self.w1, emb)


def anchor_embed(
    input_ids: torch.Tensor,
    anchor_idx: torch.Tensor,
    w1: torch.Tensor,
    emb: torch.Tensor,
    arrivals: torch.Tensor,
) -> None:
    """``emb[r] = w1[input_ids[anchor_idx[r]]]`` and ``arrivals[r] = 0`` for
    the ``arrivals.numel()`` rows (int32 ids, int64 anchors, rows of ``emb``
    contiguous)."""
    assert input_ids.dtype == torch.int32 and anchor_idx.dtype == torch.int64
    row_bytes, chunk = _row_copy(w1, emb, THREADS)
    compile_anchor(row_bytes=row_bytes, chunk=chunk, w1_rows=w1.shape[0])(
        Int64(input_ids.data_ptr()),
        Int64(anchor_idx.data_ptr()),
        Int64(w1.data_ptr()),
        Int64(emb.data_ptr()),
        Int64(arrivals.data_ptr()),
        Int32(arrivals.numel()),
        current_cu_stream(),
    )


def markov_step(
    base: torch.Tensor,
    bias: torch.Tensor,
    cache: torch.Tensor,
    col: int,
    idx_map: torch.Tensor,
    pos: torch.Tensor,
    seeds: torch.Tensor,
    temperature: torch.Tensor,
    block_argmax: torch.Tensor,
    block_max: torch.Tensor,
    arrivals: torch.Tensor,
    tokens: torch.Tensor,
    w1: torch.Tensor,
    emb: torch.Tensor,
    packets: torch.Tensor,
    width: int,
    vocab_offset: int,
    mult: float,
    inv_cap: float,
    cap: float,
    *,
    has_cap: bool,
    sharded: bool,
    block: int,
    pdl: bool,
) -> None:
    """One step over ``width`` vocab entries from ``vocab_offset`` for the
    ``base.shape[0]`` rows: ``base`` and ``bias`` BF16 rows, ``cache`` the
    draft-logits cache ``[req, step, vocab]`` or, sharded, ``[row, step,
    width]``, ``idx_map`` and ``pos`` the step's int64 columns,
    ``block_argmax`` int64 and ``block_max`` FP32 with ``rows * blocks``
    elements, ``packets`` int32 ``[rows, 4]``, ``w1`` and ``emb`` BF16."""
    for tensor, dtype in (
        (idx_map, torch.int64),
        (pos, torch.int64),
        (seeds, torch.int64),
        (temperature, torch.float32),
        (block_argmax, torch.int64),
        (block_max, torch.float32),
        (arrivals, torch.int32),
        (tokens, torch.int64),
        (w1, torch.bfloat16),
        (emb, torch.bfloat16),
        (packets, torch.int32),
    ):
        assert tensor.dtype == dtype, (tensor.dtype, dtype)
    vector = vectorized(
        block,
        width=width,
        col=col,
        base=Operand.of(base),
        bias=Operand.of(bias),
        cache=Operand.of(cache),
    )
    blocks = -(-width // block)
    row_bytes, chunk = _row_copy(w1, emb, WARP * finish_warps(blocks))
    kernel = compile_step(
        block=block,
        blocks=blocks,
        vector=vector,
        has_cap=has_cap,
        sharded=sharded,
        pdl=pdl,
        row_bytes=row_bytes,
        chunk=chunk,
        w1_rows=w1.shape[0],
    )
    _launch_step(
        kernel,
        base,
        bias,
        cache,
        col,
        idx_map,
        pos,
        seeds,
        temperature,
        block_argmax,
        block_max,
        arrivals,
        tokens,
        w1,
        emb,
        packets,
        width,
        vocab_offset,
        mult,
        inv_cap,
        cap,
    )


def markov_pick(
    packets: torch.Tensor,
    rows: int,
    tokens: torch.Tensor,
    step: int,
    w1: torch.Tensor,
    emb: torch.Tensor,
    *,
    tp: int,
    pdl: bool,
) -> None:
    """``tokens[r, step]`` and ``emb[r]`` (its w1 row) of the winner over the
    gathered int32 ``[tp, rows, 4]`` packets."""
    assert packets.dtype == torch.int32 and tokens.dtype == torch.int64
    row_bytes, chunk = _row_copy(w1, emb, PICK_THREADS)
    kernel = compile_pick(
        tp=tp, pdl=pdl, row_bytes=row_bytes, chunk=chunk, w1_rows=w1.shape[0]
    )
    _launch_pick(kernel, packets, rows, tokens, step, w1, emb)


def cache_fill(
    gathered: torch.Tensor,
    src_s0: int,
    cache: torch.Tensor,
    idx_map: torch.Tensor,
    rows: int,
    steps: int,
    shard: int,
    vocab: int,
) -> None:
    """Scatter the gathered ``[tp, rows, steps, shard]`` BF16 draft logits
    (rank stride ``src_s0``) into the full-vocab cache rows ``idx_map``
    names (int64, one per row and step; negative: skipped)."""
    assert gathered.dtype == cache.dtype == torch.bfloat16
    assert idx_map.dtype == torch.int64
    vector = fill_vectorized(
        gathered=gathered.data_ptr(),
        src_s0=src_s0,
        cache=Operand.of(cache),
        shard=shard,
        vocab=vocab,
    )
    compile_cache_fill(vectorized=vector)(
        Int64(gathered.data_ptr()),
        Int64(src_s0),
        Int64(cache.data_ptr()),
        Int64(cache.stride(0)),
        Int64(cache.stride(1)),
        Int64(idx_map.data_ptr()),
        Int32(steps),
        Int32(shard),
        Int32(vocab),
        Int32(rows * steps),
        Int32(-(-vocab // FILL_BLOCK)),
        current_cu_stream(),
    )
