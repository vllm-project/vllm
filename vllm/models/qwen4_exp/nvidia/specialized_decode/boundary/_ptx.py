# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the sublayer boundary kernel, beside the read kernels' own
(`..hc._ptx`).

Memory accesses are side-effecting assembly, kept where the kernel issues
them: data loads after the grid-dependency wait, the Lamport polling loads in
every iteration of their loop. Floating-point steps
whose rounding must match the stock kernels are spelled out instruction by
instruction, so no contraction or reassociation can change their bits.

Registers holding BF16 data are 32-bit words of two values (the lower half is
the lower address).
"""

from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import Words4, _asm, _words

# A BF16 Lamport slot is empty while it holds BF16 negative zero in every
# value; senders replace a real negative zero by a positive one.
SENTINEL_BF16X2 = 0x80008000


@dsl_user_op
def load_volatile_x4(address: Int64, *, loc=None, ip=None) -> Words4:
    """16 bytes of a Lamport mailbox slot, re-read on every call (polling)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, loc, ip)


@dsl_user_op
def store_x4(address: Int64, words: Words4, *, loc=None, ip=None) -> None:
    """Store 16 bytes at a 16-byte aligned global address."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def multicast_store_x4(address: Int64, words: Words4, *, loc=None, ip=None) -> None:
    """Store 16 bytes through an NVLink multicast address: the same slot of
    every rank's mailbox receives them."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        "multimem.st.relaxed.sys.global.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sanitize_bf16x2(word: Uint32, *, loc=None, ip=None) -> Uint32:
    """`word` with each BF16 negative zero replaced by positive zero, so that
    a published value never reads as an empty Lamport slot."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 lo, hi;\n\t.reg .pred plo, phi;\n\t"
            "and.b32 lo, $1, 0xFFFF;\n\tand.b32 hi, $1, 0xFFFF0000;\n\t"
            "setp.eq.b32 plo, lo, 0x8000;\n\tsetp.eq.b32 phi, hi, 0x80000000;\n\t"
            "@plo mov.b32 lo, 0;\n\t@phi mov.b32 hi, 0;\n\t"
            "or.b32 $0, lo, hi;\n\t}",
            "=r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def empty_halves_x4(words: Words4, *, loc=None, ip=None) -> Int32:
    """1 when any BF16 value of the 16 bytes is the empty-slot sentinel
    (negative zero), else 0."""
    return Int32(
        _asm(
            T.i32(),
            [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
            "{\n\t.reg .b32 t;\n\t.reg .pred p, q;\n\t"
            "and.b32 t, $1, 0xFFFF;\n\tsetp.eq.b32 p, t, 0x8000;\n\t"
            "and.b32 t, $1, 0xFFFF0000;\n\tsetp.eq.or.b32 p, t, 0x80000000, p;\n\t"
            "and.b32 t, $2, 0xFFFF;\n\tsetp.eq.or.b32 p, t, 0x8000, p;\n\t"
            "and.b32 t, $2, 0xFFFF0000;\n\tsetp.eq.or.b32 p, t, 0x80000000, p;\n\t"
            "and.b32 t, $3, 0xFFFF;\n\tsetp.eq.or.b32 p, t, 0x8000, p;\n\t"
            "and.b32 t, $3, 0xFFFF0000;\n\tsetp.eq.or.b32 p, t, 0x80000000, p;\n\t"
            "and.b32 t, $4, 0xFFFF;\n\tsetp.eq.or.b32 p, t, 0x8000, p;\n\t"
            "and.b32 t, $4, 0xFFFF0000;\n\tsetp.eq.or.b32 p, t, 0x80000000, p;\n\t"
            "selp.b32 $0, 1, 0, p;\n\t}",
            "=r,r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def unpack_bf16x2(word: Uint32) -> tuple[Float32, Float32]:
    """The two BF16 values of a word, widened (exactly) to FP32."""
    return _bf16_lo(word), _bf16_hi(word)


@dsl_user_op
def _bf16_lo(word: Uint32, *, loc=None, ip=None) -> Float32:
    """The lower BF16 value of a word as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_hi(word: Uint32, *, loc=None, ip=None) -> Float32:
    """The upper BF16 value of a word as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 t;\n\tand.b32 t, $1, 0xFFFF0000;\n\tmov.b32 $0, t;\n\t}",
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Two FP32 values rounded to nearest BF16 in one word, `lo` in the
    lower half."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Float32(hi).ir_value(loc=loc, ip=ip),
                Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def round_bf16(value: Float32, *, loc=None, ip=None) -> Float32:
    """`value` rounded to nearest BF16, as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 t;\n\tcvt.rn.bf16.f32 h, $1;\n\t"
            "cvt.u32.u16 t, h;\n\tshl.b32 t, t, 16;\n\tmov.b32 $0, t;\n\t}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a + b`, one rounding, never contracted into an FMA."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b`, one rounding, never contracted into an FMA."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b + c` with one rounding."""
    return Float32(
        _asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def norm_scale(square_sum: Float32, eps: Float32, *, loc=None, ip=None) -> Float32:
    """`rsqrt(square_sum / 2560 + eps)` in the instruction sequence Triton
    emits for the stock combine-norm: a full-range division, then
    `rsqrt.approx.ftz`."""
    return Float32(
        _asm(
            T.f32(),
            [
                Float32(square_sum).ir_value(loc=loc, ip=ip),
                Float32(eps).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .f32 t;\n\t"
            "div.full.f32 t, $1, 0f45200000;\n\t"
            "add.f32 t, t, $2;\n\t"
            "rsqrt.approx.ftz.f32 $0, t;\n\t}",
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def injection_scale(logit: Float32, *, loc=None, ip=None) -> Float32:
    """`2 sigmoid(logit / 4)`, the stream weight of the combine, in the
    instruction sequence Triton emits for the stock combine-norm's
    `2.0 * tl.sigmoid(inj / HC)`: full-range divisions, the exponential
    as `ex2.approx` of a product with -log2(e)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(logit).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 x, e, d, s;\n\t"
            "div.full.f32 x, $1, 0f40800000;\n\t"
            "mul.f32 e, x, 0fBFB8AA3B;\n\t"
            "ex2.approx.f32 e, e;\n\t"
            "add.f32 d, e, 0f3F800000;\n\t"
            "div.full.f32 s, 0f3F800000, d;\n\t"
            "add.f32 $0, s, s;\n\t}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """One 32-bit integer an earlier kernel wrote."""
    return Int32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.s32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_x4_if(address: Int64, slot: Int32, *, loc=None, ip=None) -> Words4:
    """16 bytes an earlier kernel wrote, loaded only when `slot` is not
    negative (else zeros, and `address` is not accessed)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int32(slot).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ge.s32 p, $5, 0;\n\t"
        "mov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "@p ld.global.v4.u32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, loc, ip)


@dsl_user_op
def fma_if(
    a: Float32, b: Float32, c: Float32, slot: Int32, *, loc=None, ip=None
) -> Float32:
    """`a * b + c` with one rounding when `slot` is not negative, else
    `c` unchanged."""
    return Float32(
        _asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
                Int32(slot).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\t.reg .f32 t;\n\tsetp.ge.s32 p, $4, 0;\n\t"
            "fma.rn.f32 t, $1, $2, $3;\n\tselp.f32 $0, t, $3, p;\n\t}",
            "=f,f,f,f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_volatile_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    """One 32-bit word another kernel of the chain may have changed."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """Store one 32-bit word."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrive_u32(address: Int64, after: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add one to a GPU-scope arrival counter and return its previous value.

    The atomic is relaxed, so this thread's later memory accesses do not wait
    for it, and it issues only once the value `after` (the generation word)
    has been loaded: its increment, `(after >> 31) + 1`, which is one for
    any generation, depends on that value. An arrival therefore implies that
    the arriving thread has read the generation; other threads of its CTA are
    not ordered by it.
    """
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(after).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 one;\n\tshr.u32 one, $2, 31;\n\tadd.u32 one, one, 1;\n\t"
            "atom.relaxed.gpu.global.add.u32 $0, [$1], one;\n\t}",
            "=r,l,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_cluster_f32(
    address: Int32, cta_rank: Int32, value: Float32, *, loc=None, ip=None
) -> None:
    """Store one FP32 value at `address` in the shared memory of the
    cluster's CTA `cta_rank` (distributed shared memory)."""
    _asm(
        None,
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Int32(cta_rank).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 a;\n\tmapa.shared::cluster.u32 a, $0, $1;\n\t"
        "st.shared::cluster.f32 [a], $2;\n\t}",
        "r,r,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_shared_u32(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    """Store one 32-bit word in this CTA's shared memory."""
    _asm(
        None,
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_u32(address: Int32, *, loc=None, ip=None) -> Uint32:
    """One 32-bit word from this CTA's shared memory."""
    return Uint32(
        _asm(
            T.i32(),
            [Int32(address).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def global_timer(*, loc=None, ip=None) -> Int64:
    """The GPU's global nanosecond timer, read anew on every call."""
    return Int64(
        _asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def report_stall(address: Int64, word: Int64, *, loc=None, ip=None) -> None:
    """Store `word` at `address` (host memory) at system scope, make it
    visible to the host, then trap: the kernel aborts with a CUDA error."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int64(word).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.sys.global.u64 [$0], $1;\n\tfence.sc.sys;\n\ttrap;",
        "l,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
