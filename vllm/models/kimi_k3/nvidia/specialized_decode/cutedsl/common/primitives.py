# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline-PTX, memory, synchronization and math helpers of the decode kernels."""

import math
from functools import partial, wraps

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, Int64, Uint8, Uint16, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as nvvm


def ptx(assembly, constraints, operands=(), result=None, *, effects, loc=None, ip=None):
    """One verbatim asm block; a tuple ``result`` (even a 1-tuple) returns a tuple."""
    if isinstance(result, tuple) and len(result) == 1:
        args = (assembly, constraints, operands, result[0])
        return (ptx(*args, effects=effects, loc=loc, ip=ip),)
    if result is None:
        ir_type = None
    elif isinstance(result, tuple):
        ir_type = llvm.StructType.get_literal([dtype.mlir_type for dtype in result])
    else:
        ir_type = result.mlir_type
    value = llvm.inline_asm(
        ir_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        assembly,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            dtype(llvm.extractvalue(dtype.mlir_type, value, [i], loc=loc, ip=ip))
            for i, dtype in enumerate(result)
        )
    return result(value)


def asm(result=None, *, pure=False):
    """``f(*args) -> (code, constraints, operands[, result, pure])`` as one inline-PTX
    op. Pointer operands pass their address: 32-bit for "r" (shared), 64-bit for "l"."""

    def wrap(fn):
        @dsl_user_op
        @wraps(fn)
        def op(*args, loc=None, ip=None, **kwargs):
            code, constraints, operands, *rest = fn(*args, **kwargs)
            res, fx = (*rest, pure)[:2] if rest else (result, pure)
            kinds = [c for c in constraints.split(",") if c[:1] not in ("=", "~", "")]
            values = []
            for kind, v in zip(kinds, operands):
                if hasattr(v, "toint"):
                    v = v.toint(loc=loc, ip=ip)
                    v = Int32(v) if kind == "r" else v
                values.append(v)
            return ptx(code, constraints, values, res, effects=not fx, loc=loc, ip=ip)

        return op

    return wrap


def _vector(count, first=0):
    """(".vN", "{$first,...}") of an N-word access; ("", "$first") for one word."""
    names = ",".join(f"${first + i}" for i in range(count))
    return (f".v{count}", "{" + names + "}") if count > 1 else ("", names)


div_rn = partial(cute.math.div, rounding=cute.math.RoundingMode.NEAREST_EVEN)


@asm(Float32, pure=True)
def add_rn(a, b):
    return "add.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def routing_add(a: Float32, b: Float32):
    return "add.ftz.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def fma(a, b, c, *, ftz=False):
    return f"fma.rn{'.ftz' if ftz else ''}.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, c)


def dot(a, b, acc=None, **kwargs):
    """``fma`` chain over ``zip(a, b)`` from ``acc``, or from ``a[0] * b[0]``."""
    for x, y in zip(a, b):
        acc = x * y if acc is None else fma(x, y, acc, **kwargs)
    return acc


@asm(Float32, pure=True)
def fma_bf16(a, b, c, *, high=False):
    half = "h" if high else "l"
    code = "{ .reg .b16 al,ah,bl,bh; mov.b32 {al,ah},$1; mov.b32 {bl,bh},$2; "
    code += f"fma.rn.f32.bf16 $0,a{half},b{half},$3; }}"
    return code, "=f,r,r,f", (Uint32(a), Uint32(b), Float32(c))


@asm(Uint32, pure=True)
def round_pair_square(word):
    return "mul.rn.bf16x2 $0,$1,$1;", "=r,r", (Uint32(word),)


@asm(Float32, pure=True)
def inv_rms(value, eps, *, m8=False):
    """rsqrt(value / 3584 + eps); ``m8``: the mean as a reciprocal product refined by
    its FMA residual."""
    mean = "div.rn.f32 r,$1,0f45600000;"
    if m8:
        mean = "mul.f32 q,$1,0f39924925; neg.f32 r,q; fma.rn.f32 r,r,0f45600000,$1; "
        mean += "fma.rn.f32 r,r,0f39924925,q;"
    code = f"{{ .reg .f32 {'q,' * m8}r; {mean} add.f32 r,r,$2; "
    code += "rsqrt.approx.ftz.f32 $0,r; }"
    return code, "=f,f,f", (Float32(value), Float32(eps))


@asm(Float32, pure=True)
def query_norm_rsqrt(total: Float32, eps: Float32):
    code = "{ .reg .f32 mean; div.rn.f32 mean,$1,0f44c00000; "
    code += "add.f32 mean,mean,$2; rsqrt.approx.f32 $0,mean; }"
    return code, "=f,f,f", (total, eps)


@asm(Float64, pure=True)
def reciprocal64(value: Float64):
    step = "fma.rn.f64 e,n,r,0d3ff0000000000000; fma.rn.f64 {},r,e,r; "
    code = "{ .reg .f64 r,e,n; neg.f64 n,$1; rcp.approx.ftz.f64 r,$1; "
    code += step.format("r") * 2 + step.format("$0") + "}"
    return code, "=d,d", (value,)


@asm(Float32, pure=True)
def sigmoid_reciprocal(denominator: Float32, *, fast_math):
    """1 / (1 + exp(-x)) from its denominator: an approximate divide with ``fast_math``,
    else the approximate reciprocal refined by two IEEE FMAs (out-of-range: div.rn)."""
    if fast_math:
        return "div.approx.ftz.f32 $0, 0f3f800000, $1;", "=f,f", (denominator,)
    code = (
        "{\n.reg .f32 r,error,neg_error; .reg .b32 bits; .reg .pred regular;\n"
        "mov.b32 bits,$1; add.u32 bits,bits,0x01800000; and.b32 bits,bits,0x7f800000;\n"
        "setp.gt.u32 regular,bits,0x01ffffff; rcp.approx.ftz.f32 r,$1;\n"
        "fma.rn.f32 error,$1,r,0fbf800000; neg.f32 neg_error,error;\n"
        "fma.rn.f32 $0,r,neg_error,r; @!regular div.rn.f32 $0,0f3f800000,$1;\n}"
    )
    return code, "=f,f", (denominator,)


@cute.jit
def sigmoid(value: Float32, fast_math: cutlass.Constexpr = None):
    """1 / (1 + exp(-x)); correctly rounded when ``fast_math`` is None."""
    if cutlass.const_expr(fast_math is None):
        result = div_rn(Float32(1), Float32(1) + cute.math.exp(-value, fastmath=False))
    else:
        denominator = Float32(1) + cute.math.exp(-value, fastmath=True)
        result = sigmoid_reciprocal(denominator, fast_math=fast_math)
    return result


@asm(Float32, pure=True)
def routing_sigmoid(value: Float32):
    code = "{ .reg .f32 h,t; mul.ftz.f32 h,$1,0f3f000000; tanh.approx.f32 t,h; "
    code += "fma.rn.ftz.f32 $0,t,0f3f000000,0f3f000000; }"
    return code, "=f,f", (value,)


@asm(Float32, pure=True)
def routing_weight(score: Float32, total: Float32, scale: Float32, *, renorm=True):
    code = "mul.ftz.f32 $0,$3,$1;"
    if renorm:
        code = (
            "{ .reg .f32 n,d; mul.ftz.f32 n,$3,$1; "
            "add.ftz.f32 d,0f1e3ce508,$2; div.approx.ftz.f32 $0,n,d; }"
        )
    return code, "=f,f,f,f", (score, total, scale)


@asm(Float32, pure=True)
def situ(g: Float32, u: Float32, beta_lb: Float32, inv_beta: Float32, inv_lb: Float32):
    code = "{ .reg .f32 tg,tu,ex,rc,z; mul.f32 tg,$1,$4; tanh.approx.f32 tg,tg; "
    code += "mul.f32 tu,$2,$5; tanh.approx.f32 tu,tu; mul.f32 z,$1,0fbfb8aa3b; "
    code += "ex2.approx.ftz.f32 ex,z; add.f32 ex,ex,0f3f800000; "
    code += "rcp.approx.ftz.f32 rc,ex; mul.f32 z,$3,tg; mul.f32 z,z,rc; "
    code += "mul.f32 $0,z,tu; }"
    return code, "=f,f,f,f,f,f", (g, u, beta_lb, inv_beta, inv_lb)


@cute.jit
def shared_situ(gate, up, beta, inverse_beta, linear_beta, inverse_linear_beta):
    denominator = Float32(1) + cute.math.exp(-gate, fastmath=False)
    sigmoid = Float32(0)
    if denominator != Float32(float("inf")):
        sigmoid = Float32(reciprocal64(Float64(denominator)))
    g = (beta * cute.math.tanh(gate * inverse_beta, fastmath=False)) * sigmoid
    u = up
    if linear_beta > 0:
        divided = Float32(Float64(up) * inverse_linear_beta)
        u = linear_beta * cute.math.tanh(divided, fastmath=False)
    return g * u


def bits_float(bits):
    return Float32(llvm.bitcast(T.f32(), Uint32(bits).ir_value()))


def float_bits(value: Float32) -> Int32:
    return Int32(llvm.bitcast(T.i32(), value.ir_value()))


def bf16(word, high=False):
    """FP32 of the low (or high) BF16 half of ``word``."""
    return bits_float(word & 0xFFFF0000 if high else word << 16)


@asm((Float32, Float32), pure=True)
def unpack_bf16(word: Uint32):
    return "shl.b32 $0,$2,16; and.b32 $1,$2,0xffff0000;", "=f,=f,r", (word,)


@asm((Float32, Float32), pure=True)
def fma2(a, b, c):
    """Lanewise a * b + c of FP32 pairs in one packed instruction (IEEE per lane)."""
    code = "{ .reg .b64 a,b,c,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    code += "mov.b64 c,{$6,$7}; fma.rn.f32x2 d,a,b,c; mov.b64 {$0,$1},d; }"
    return code, "=f,=f,f,f,f,f,f,f", (*a, *b, *c)


@asm((Float32, Float32), pure=True)
def mul2(a, b):
    """Lanewise a * b of FP32 pairs in one packed instruction (IEEE per lane)."""
    code = "{ .reg .b64 a,b,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    code += "mul.rn.f32x2 d,a,b; mov.b64 {$0,$1},d; }"
    return code, "=f,=f,f,f,f,f", (*a, *b)


@asm(Uint16, pure=True)
def gate_bf16_bits(value: Float32):
    return "cvt.rn.bf16.f32 $0, $1;", "=h,f", (value,)


@asm(Uint32, pure=True)
def pack_pair(low, high, *, policy="plain"):
    """BF16x2 of (low, high); ``policy=None`` is the bare conversion."""
    code = "cvt.rn.bf16x2.f32 $0,$2,$1; "
    if policy == "stage":
        # NaN halves become canonical, so no word is the 0xffffffff staging sentinel.
        code += (
            "and.b32 bits,$0,0x7fff; setp.gt.u32 p,bits,0x7f80; "
            "@p and.b32 $0,$0,0xffff0000; @p or.b32 $0,$0,0x7fc0; "
            "and.b32 bits,$0,0x7fff0000; setp.gt.u32 p,bits,0x7f800000; "
            "@p and.b32 $0,$0,0xffff; @p or.b32 $0,$0,0x7fc00000; "
        )
    if policy == "latent":
        # Negative BF16 zero is the downstream mailbox's unpublished sentinel.
        code += (
            "and.b32 bits,$0,0xffff; setp.eq.u32 p,bits,0x8000; "
            "@p and.b32 $0,$0,0xffff0000; shr.u32 bits,$0,16; "
            "setp.eq.u32 p,bits,0x8000; @p and.b32 $0,$0,0xffff; "
        )
    if policy is not None:
        code = "{ .reg .b32 bits; .reg .pred p; " + code + "}"
    return code, "=r,f,f", (Float32(low), Float32(high))


bf16x2 = partial(pack_pair, policy=None)


@cute.jit
def no_neg_zero(word):
    """``word`` with each negative-zero BF16 half made +0."""
    result = Uint32(word)
    if (result & 0xFFFF) == 0x8000:
        result &= Uint32(0xFFFF0000)
    if (result >> 16) == 0x8000:
        result &= Uint32(0xFFFF)
    return result


def published(words, ready=True, *, sentinel=0x80000000):
    """``ready`` and no word is the mailbox ``sentinel``."""
    for word in words:
        ready &= word != Uint32(sentinel)
    return ready


@cute.jit
def front_dirty(word):
    return ((word & 0xFFFF) == 0xFFFF) | ((word >> 16) == 0xFFFF)


def fp8_bits(value: Float32):
    code = (
        "{ .reg .b16 pair; cvt.rn.satfinite.e4m3x2.f32 pair, 0f00000000, $1; "
        "cvt.u32.u16 $0, pair; }"
    )
    return Uint8(ptx(code, "=r,f", (value,), Uint32, effects=False))


@asm()
def store_fp8_pair(pointer, low: Float32, high: Float32):
    """FP8 E4M3 (satfinite) of ``low`` and ``high`` as the two bytes at ``pointer``."""
    code = "{ .reg .b16 pair; cvt.rn.satfinite.e4m3x2.f32 pair,$2,$1; "
    code += "st.global.b16 [$0],pair; }"
    return code, "l,f,f,~{memory}", (pointer, low, high)


@asm(Uint32, pure=True)
def pack_fp8(values):
    code = "{ .reg .b16 lo,hi; cvt.rn.satfinite.e4m3x2.f32 lo,$2,$1; "
    code += "cvt.rn.satfinite.e4m3x2.f32 hi,$4,$3; mov.b32 $0,{lo,hi}; }"
    return code, "=r,f,f,f,f", values


@cute.jit
def mx_exponent(amax: Float32, floor: cutlass.Constexpr[bool] = False):
    bits = Uint32(float_bits(amax))
    e = Int32(bits >> 23)
    result = Int32(0)
    if cutlass.const_expr(floor):
        if e == 255:
            result = 254
        elif e > 8:
            result = e - 8
    else:
        if e != 0:
            result = min(max(e - 8 + Int32(bits & 8388607 > 6291456), 0), 254)
    return result


@cute.jit
def mx_rescale(scale: Int32):
    e = 254 - scale
    return bits_float(Uint32(e) << 23) if e >= 1 else Float32(0)


@cute.jit
def warp_sum(value: Float32, lanes: cutlass.Constexpr[int] = 32):
    for shift in cutlass.range_constexpr(lanes.bit_length() - 2, -1, -1):
        value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
    return value


@asm(Uint32, pure=True)
def redux(value, op="min"):
    return f"redux.sync.{op}.u32 $0,$1,0xffffffff;", "=r,r", (Uint32(value),)


def warp_max_abs(value):
    return bits_float(redux(float_bits(abs(value)), "max"))


@asm((Float32,) * 4, pure=True)
def mma(a, b, accumulator, *, fp8=False, bf16=False):
    """m16n8k16 (FP8: k32) mma.sync."""
    kind = "e4m3" if fp8 else "bf16" if bf16 else "f16"
    code = f"mma.sync.aligned.m16n8k{32 if fp8 else 16}.row.col.f32.{kind}.{kind}.f32 "
    code += "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};"
    return code, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", (*a, *b, *accumulator)


def mma_into(acc, a, b, **kwargs):
    """``acc += a @ b`` with one ``mma``; ``acc`` is a four-element register view."""
    result = mma(a, b, tuple(acc[k] for k in range(4)), **kwargs)
    for k in range(4):
        acc[k] = result[k]


@asm()
def ldmatrix(pointer, *, trans=False, count=4):
    """``count`` 8x8 b16 shared matrices; lanes 8i..8i+7 address matrix i's rows."""
    names = ",".join(f"${i}" for i in range(count))
    code = f"ldmatrix.sync.aligned.m8n8.x{count}{'.trans' * trans}.shared.b16 "
    code += f"{{{names}}},[${count}];"
    constraints = ",".join(["=r"] * count + ["r"])
    return code, constraints, (pointer,), (Uint32,) * count


def split(total, i, n):
    """(first, count) of part ``i`` of ``total`` items split ``n`` ways."""
    first = total * i // n
    return first, total * (i + 1) // n - first


def thread():
    """(thread, warp, lane, CTA) indices."""
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    return tid, tid // 32, tid % 32, cta


def shared_address(pointer):
    return Uint32(pointer.toint())


def at(tensor, *coord):
    """Pointer to ``tensor[coord]``."""
    return tensor.iterator + cute.crd2idx(coord, tensor.layout)


def view(base, offset, dtype, shape, stride=None):
    """``dtype`` tensor at byte ``offset`` of pointer ``base``."""
    pointer = cute.recast_ptr(base + offset, dtype=dtype)
    return cute.make_tensor(pointer, cute.make_layout(shape, stride=stride))


def zeros(shape, dtype):
    tensor = cute.make_rmem_tensor(shape, dtype)
    tensor.fill(0)
    return tensor


def alloc(allocator, dtype, shape, align=16):
    """Row-major ``shape`` tensor from a ``SmemAllocator``."""
    stride = None
    if isinstance(shape, tuple):
        stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    layout = cute.make_layout(shape, stride=stride)
    return allocator.allocate_tensor(dtype, layout, align)


@asm()
def load_words(
    pointer,
    *,
    count=2,
    shared=False,
    evict_first=False,
    mutable=False,
    volatile=False,
    scope=None,
    cache_policy="cg",
    word_type="b32",
    result_type=Uint32,
):
    """``count`` words (a tuple); ``scope`` is a relaxed load at that scope.
    ``mutable`` (a side-effecting load) for data written during the step (activations,
    tables, refills). Only static weights use ``evict_first``."""
    space = "global" + (f".{cache_policy}" if cache_policy else "")
    if shared:
        space = "shared"
    if volatile:
        space = "volatile.global"
    if scope:
        space = f"relaxed.{scope}.global"
    policy = hint = tail = ""
    if evict_first:
        assert not shared
        space = "global.nc.L1::no_allocate"
        policy = ".reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
        hint, tail = ".L2::cache_hint", ", pol"
    vector, registers = _vector(count)
    address = f"[${count}]{tail}"
    code = f"{{ {policy}ld.{space}{hint}{vector}.{word_type} {registers}, {address}; }}"
    constraints = ["=f" if result_type is Float32 else "=r"] * count
    constraints += ["r" if shared else "l"] + (["~{memory}"] if scope else [])
    # Shared data is written by other threads between reads: never pure.
    pure = not (shared or mutable or volatile or scope)
    return code, ",".join(constraints), (pointer,), (result_type,) * count, pure


smem_words = partial(load_words, count=4, shared=True, mutable=True)
load_volatile = partial(load_words, count=1, volatile=True, word_type="u32")
mailbox_load = partial(load_words, count=1, scope="gpu", word_type="u32")
load_f32x4 = partial(load_words, count=4, result_type=Float32, word_type="f32")
early_weights = partial(
    load_words, cache_policy="nc.L1::no_allocate", word_type="u32", mutable=True
)


@asm()
def load_bf16x8(pointer, *, shared=False, mutable=False):
    opcode = "ld.shared.v4.b32" if shared else "ld.global.cg.v4.b32"
    # Shifting BF16 bits into the high FP32 half preserves values exactly.
    code = "{ .reg .b32 x<4>; " + opcode + " {x0,x1,x2,x3}, [$8]; "
    for j in range(4):
        code += f"shl.b32 ${2 * j}, x{j}, 16; and.b32 ${2 * j + 1}, x{j}, 0xffff0000; "
    constraints = ",".join(["=f"] * 8 + ["r" if shared else "l"])
    return code + "}", constraints, (pointer,), (Float32,) * 8, not (shared or mutable)


@asm()
def load_vector(pointer, *, cache="cg"):
    """``cache`` "weight": a streaming L2 evict_first load of static weights; else
    ``ld.global.<cache>``. Side-effecting in every mode."""
    opcode, policy, tail = "ld.global." + cache, "", ""
    if cache == "weight":
        policy = ".reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
        opcode = "ld.global.L1::no_allocate.L2::cache_hint.L2::256B"
        tail = ",pol"
    if cache == "volatile":
        opcode = "ld.volatile.global"
    code = "{ " + policy + opcode + ".v4.b32 {$0,$1,$2,$3},[$4]" + tail + "; }"
    clobber = ",~{memory}" if cache == "volatile" else ""
    return code, "=r,=r,=r,=r,l" + clobber, (pointer,), (Uint32,) * 4


@asm(Float32, pure=True)
def table_load(pointer):
    return "ld.global.nc.f32 $0,[$1];", "=f,l", (pointer,)


@asm()
def store_words(pointer, words, *, shared=False, scope=None):
    """``scope`` is a relaxed global store at that scope."""
    vector, registers = _vector(len(words), 1)
    if shared:
        # Other threads read it: a memory side effect, like the global stores.
        space, constraints = (
            f"shared{vector}.b32",
            "r" + ",r" * len(words) + ",~{memory}",
        )
    else:
        space = (
            f"relaxed.{scope}.global{vector}.u32" if scope else f"global{vector}.b32"
        )
        constraints = "l" + ",r" * len(words) + ",~{memory}"
        words = tuple(Uint32(v) for v in words)
    return f"st.{space} [$0], {registers};", constraints, (pointer, *words)


store_shared_words = partial(store_words, shared=True)
store_relaxed = partial(store_words, scope="sys")
mailbox_store = partial(store_words, scope="gpu")


@asm()
def store_f32x4(pointer, values, when=None):
    """``when``: a predicate, so callers need no branch."""
    code = "st.global.v4.f32 [$0], {$1, $2, $3, $4};"
    if when is None:
        return code, "l,f,f,f,f", (pointer, *values)
    code = "{ .reg .pred p; setp.ne.b32 p,$5,0; @p " + code + " }"
    return code, "l,f,f,f,f,r", (pointer, *values, Int32(when))


@asm()
def store_gate_mailbox(pointer, bits: Uint16):
    return "st.relaxed.gpu.global.u16 [$0], $1;", "l,h,~{memory}", (pointer, bits)


@asm()
def store_bf16_pair(pointer, a, b):
    code = "{ .reg .b32 pair; cvt.rn.bf16x2.f32 pair,$2,$1; "
    return code + "st.relaxed.gpu.global.b32 [$0],pair; }", "l,f,f", (pointer, a, b)


@asm()
def store_pair(pointer, a, b):
    """BF16x2 (a, b) with negative-zero halves made +0."""
    code = "{ .reg .b32 pair,half; .reg .pred p; cvt.rn.bf16x2.f32 pair,$2,$1; "
    code += "and.b32 half,pair,0xffff; setp.eq.b32 p,half,0x8000; "
    code += "@p and.b32 pair,pair,0xffff0000; shr.u32 half,pair,16; "
    code += "setp.eq.b32 p,half,0x8000; @p and.b32 pair,pair,0x0000ffff; "
    return code + "st.global.b32 [$0],pair; }", "l,f,f", (pointer, a, b)


@asm()
def push_float(pointer, value, barrier, peer, when=None):
    """DSMEM store of one FP32 ``value`` (or a 4-tuple) completing on ``barrier``
    (``when``: predicated on it)."""
    values = tuple(map(Float32, value if isinstance(value, tuple) else (value,)))
    vector, registers = _vector(len(values), 1)
    b, p, w = (f"${len(values) + i}" for i in (1, 2, 3))
    code = f"{{ .reg .b32 dst,bar; mapa.shared::cluster.u32 dst,$0,{p}; "
    code += f"mapa.shared::cluster.u32 bar,{b},{p}; "
    guard = "" if when is None else f".reg .pred w; setp.ne.b32 w,{w},0; @w "
    code += (
        guard + f"st.async.shared::cluster.mbarrier::complete_tx::bytes{vector}.b32 "
    )
    code += f"[dst],{registers},[bar]; }}"
    constraints = "r" + ",f" * len(values) + ",r,r" + ",r" * (when is not None)
    operands = (pointer, *values, barrier, Int32(peer))
    operands += () if when is None else (Int32(when),)
    return code, constraints + ",~{memory}", operands


@asm()
def multicast_words(address, words, *, kind="f32"):
    """Relaxed multimem store of FP32 ("f32") or integer ("b32", "bf16x2") words."""
    vector, registers = _vector(len(words), 1)
    f32 = kind == "f32"
    values = (bits_float(v) if f32 else Uint32(v) for v in words)
    constraints = "l" + (",f" if f32 else ",r") * len(words) + ",~{memory}"
    code = f"multimem.st.relaxed.sys.global{vector}.{kind} [$0],{registers};"
    return code, constraints, (Int64(address), *values)


@asm()
def publish_hint(pointer):
    return "red.relaxed.gpu.global.add.u32 [$0],1;", "l,~{memory}", (pointer,)


@asm()
def copy_g2s(dst, src, size=16, barrier=None, *, evict_first=False):
    """16-byte cp.async, or a static/dynamic ``size`` bulk copy on ``barrier``."""
    operands = [dst, src]
    policy = hint = tail = ""
    if evict_first:
        policy = ".reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
        hint, tail = ".L2::cache_hint", ", pol"
    if barrier is None:
        assert size == 16
        code = f"cp.async.cg.shared.global{hint} [$0], [$1], 16{tail};"
    else:
        operands.append(barrier)
        if not isinstance(size, int):
            operands.append(size)
            size = "$3"
        code = (
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            f"{hint} [$0], [$1], {size}, [$2]{tail};"
        )
    constraints = ",".join(["r", "l"] + ["r"] * (len(operands) - 2))
    return "{ " + policy + code + " }", constraints, operands


copy_evict = partial(copy_g2s, evict_first=True)


@asm()
def copy_bulk(dst, src, barrier, size):
    code = "{ .reg .b64 p; createpolicy.fractional.L2::evict_first.b64 p,1.0; "
    code += "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
    code += ".L2::cache_hint [$0],[$1],$3,[$2],p; }"
    return code, "r,l,r,r,~{memory}", (dst, src, barrier, Int32(size))


@asm()
def copy_s2cluster(dst, src, barrier, peer, *, size):
    code = "{ .reg .b32 dst,bar; mapa.shared::cluster.u32 dst,$0,$3; "
    code += "mapa.shared::cluster.u32 bar,$2,$3; cp.async.bulk.shared::cluster"
    code += f".shared::cta.mbarrier::complete_tx::bytes [dst],[$1],{size},[bar]; }}"
    return code, "r,r,r,r", (dst, src, barrier, Int32(peer))


@asm()
def stage_weight(dst, src):
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
    code += "cp.async.cg.shared.global.L2::cache_hint.L2::256B [$0],[$1],16,pol; }"
    return code, "r,l,~{memory}", (dst, src)


@asm()
def load_tma2(dst, descriptor, barrier, column, row):
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
    code += "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier"
    code += "::complete_tx::bytes.L2::cache_hint [$0],[$1,{$3,$4}],[$2],pol; }"
    return code, "r,l,r,r,r", (dst, descriptor, barrier, Int32(column), Int32(row))


@asm()
def load_tma3(dst, descriptor, barrier, row, chunk):
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
    code += "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier"
    code += "::complete_tx::bytes.L2::cache_hint [$0],[$1,{0,$3,$4}],[$2],pol; }"
    return code, "r,l,r,r,r", (dst, descriptor, barrier, Int32(row), Int32(chunk))


@asm()
def prefetch_bytes(src, size):
    return "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r", (src, Int32(size))


@dsl_user_op
def prefetch_bulk(address, size, *, loc=None, ip=None):
    """``prefetch_bytes`` of a 64-bit address through the NVVM op."""
    pointer = cute.make_ptr(cutlass.Uint8, address, cute.AddressSpace.gmem)
    nvvm.cp_async_bulk_prefetch(pointer, size, loc=loc, ip=ip)


@asm()
def prefetch_l2(pointer):
    return "prefetch.global.L2 [$0];", "l", (pointer,)


@asm()
def prefetch_map(descriptor, row, chunk):
    code = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [$0,{0,$1,$2}];"
    return code, "l,r,r", (descriptor, Int32(row), Int32(chunk))


def acquire_tensormap(pointer):
    generic = cute.make_ptr(cutlass.Uint8, pointer.toint(), cute.AddressSpace.generic)
    nvvm.fence_proxy_acquire("sys", generic, 128)
    nvvm.prefetch_tensormap(generic)


@asm()
def acquire_tensormap_asm(pointer):
    """``acquire_tensormap`` as one asm block on the unfolded address."""
    code = "fence.proxy.tensormap::generic.acquire.sys [$0],128; "
    return code + "prefetch.tensormap [$0];", "l", (pointer,)


@asm(Int32)
def try_wait(barrier, phase=0, *, test=False, scope="cluster"):
    """Acquire parity probe at ``scope`` ("cluster", or "cta" when every arrival is
    CTA-local); ``test`` is the non-blocking probe of phase 0."""
    op, parity = ("test_wait", "0") if test else ("try_wait", "$2")
    code = f"{{ .reg .pred p; mbarrier.{op}.parity.acquire.{scope}.shared::cta.b64 "
    code += f"p,[$1],{parity}; selp.u32 $0,1,0,p; }}"
    if test:
        return code, "=r,r", (barrier,)
    return code, "=r,r,r,~{memory}", (barrier, Int32(phase))


@cute.jit
def wait(barrier, phase=0, test: cutlass.Constexpr[bool] = False, scope="cluster"):
    done = try_wait(barrier, phase, test=test, scope=scope)
    while done == 0:
        done = try_wait(barrier, phase, test=test, scope=scope)


@asm(Int32)
def try_wait_phase_zero(barrier):
    code = "{ .reg .pred ready; mbarrier.try_wait.parity.acquire.cluster"
    code += ".shared::cta.b64 ready,[$1],0; selp.u32 $0,1,0,ready; }"
    return code, "=r,r,~{memory}", (barrier,)


@cute.jit
def wait_phase_zero_cluster(barrier: cute.Pointer):
    done = try_wait_phase_zero(barrier)
    while done == 0:
        done = try_wait_phase_zero(barrier)


def init_barriers(barriers, count, *, async_proxy=False):
    for i in range(count):
        cute.arch.mbarrier_init(barriers + i, 1)
    cute.arch.mbarrier_init_fence()
    if async_proxy:
        cute.arch.fence_proxy("async.shared", space="cta")


@asm()
def named_barrier(barrier_id=1, count=128, *, arrive=False, all_groups=False):
    if all_groups:
        barrier_id, count = 2, 512
    op = "arrive" if arrive else "sync"
    return f"bar.{op} {barrier_id},{count};", "~{memory}", ()


def group_barrier(group):
    cute.arch.barrier(barrier_id=1 + 3 * group, number_of_threads=128)


@asm()
def sleep_ns(ns):
    """``ns`` is an immediate if a Python int, else a register."""
    if isinstance(ns, int):
        return f"nanosleep.u32 {ns};", "", ()
    return "nanosleep.u32 $0;", "r", (Uint32(ns),)


@cute.jit
def smem_desc(pointer: cute.Pointer, swizzle: cutlass.Constexpr[bool]):
    address = shared_address(pointer)
    value = Uint64((address & 262143) >> 4) | Uint64(1) << 46
    if cutlass.const_expr(swizzle):
        value |= Uint64(1) << 16 | Uint64(64) << 32 | Uint64(2) << 61
    else:
        value |= Uint64(8) << 32
    return value


def fence_tmem(*, before=False):
    nvvm.tcgen05_fence("before_thread_sync" if before else "after_thread_sync")


def wait_tmem(barrier, phase=0, scope="cluster"):
    """``wait``, then order later tcgen05 operations after it."""
    wait(barrier, phase, scope=scope)
    fence_tmem()


@asm()
def free_tmem(address, columns):
    code = f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0,{columns};"
    return code, "r", (Uint32(address),)


@cute.jit
def release_tmem(tmem, columns: cutlass.Constexpr[int], warp, owner):
    """CTA teardown: warp ``owner`` frees the TMEM after every tcgen05 operation."""
    fence_tmem(before=True)
    cute.arch.sync_threads()
    if warp == owner:
        fence_tmem()
        free_tmem(tmem, columns)


@asm()
def require_zero_tmem(address):
    return "{ .reg .pred p; setp.ne.u32 p,$0,0; @p trap; }", "r", (Uint32(address),)


@asm()
def load_tmem(address, count=1):
    """``count`` FP32 TMEM columns of this lane, waiting for the load."""
    registers = ",".join(f"${i}" for i in range(count))
    code = f"tcgen05.ld.sync.aligned.32x32b.x{count}.b32 {{{registers}}},[${count}]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    constraints = ",".join(["=f"] * count + ["r", "~{memory}"])
    result = Float32 if count == 1 else (Float32,) * count
    return code, constraints, (Uint32(address),), result


@asm()
def copy_scale(column, descriptor):
    code = "tcgen05.cp.cta_group::1.32x128b.warpx4 [$0],$1;"
    return code, "r,l", (Uint32(column), Uint64(descriptor))


commit = nvvm.tcgen05_commit


@asm()
def commit_asm(barrier):
    """``commit`` as an asm block on the unfolded address."""
    code = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];"
    return code, "r", (barrier,)


@asm()
def mma_bf16_block(d, a, b, instruction, accumulate, empty, release):
    """Warp-uniform issue of one 64-K block (K-major 128B-swizzled A and B): the
    elected lane runs the four K16 BF16 MMAs into ``d`` (the first accumulates iff
    ``accumulate`` is nonzero) and, if ``release``, commits ``empty``."""
    mma = "tcgen05.mma.cta_group::1.kind::f16"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = "{ .reg .pred e,p0,pt,pc; .reg .b64 a1,a2,a3,b1,b2,b3; "
    code += "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 p0,$4,0; "
    code += "setp.ne.b32 pc,$6,0; and.pred pc,pc,e; "
    code += "add.s64 a1,$1,2; add.s64 a2,$1,4; add.s64 a3,$1,6; "
    code += "add.s64 b1,$2,2; add.s64 b2,$2,4; add.s64 b3,$2,6; "
    code += f"@e {mma} [$0],$1,$2,$3,p0; @e {mma} [$0],a1,b1,$3,pt; "
    code += f"@e {mma} [$0],a2,b2,$3,pt; @e {mma} [$0],a3,b3,$3,pt; "
    code += f"@pc {commit} [$5]; }}"
    operands: tuple = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction))
    operands += (Uint32(accumulate), empty, Uint32(release))
    return code, "r,l,l,r,r,r,r", operands


@asm()
def mma_mx(d, a, b, instruction, sfa, sfb, accumulate):
    """One block-scaled MXF8 tcgen05 MMA; ``accumulate`` is a runtime flag."""
    words = (Uint32(v) for v in (instruction, sfa, sfb, accumulate))
    code = "{ .reg .pred p; setp.ne.b32 p,$6,0; tcgen05.mma.cta_group::1"
    code += ".kind::mxf8f6f4.block_scale [$0],$1,$2,$3,[$4],[$5],p; }"
    return code, "r,l,l,r,r,r,r", (Uint32(d), Uint64(a), Uint64(b), *words)


@cute.jit
def mma_steps(d, a, b, sfa, sfb, accumulate, fc2: cutlass.Constexpr[bool]):
    for k in cutlass.range_constexpr(6 if fc2 else 8):
        a_offset: cutlass.Constexpr = k // 4 * 1024 + k % 4 * 2 + 4 * (fc2 and k >= 4)
        instruction: cutlass.Constexpr = 142738048 | (k % 4 << 29) | (k % 4 << 4)
        ops: tuple = (d, a + a_offset, b + k // 4 * 64 + k % 4 * 2, instruction)
        ops += (sfa + k // 4 * 4, sfb + (k * 4 if fc2 else k // 4 * 4))
        mma_mx(*ops, accumulate if k == 0 else 1)


@asm()
def chunk_mma(*args):
    """(d, adesc, bdesc, sfa, sfa_src, sfb, sfb_src, copy_sfb, accum, empty_bar,
    acc_bar): four K32 MMAs of one K128 chunk, with optional completion arrivals."""
    constraints = "r,l,l,r,l,r,l,r,r,r,r"
    types = [Uint32 if kind == "r" else Uint64 for kind in constraints.split(",")]
    cp, mma = "tcgen05.cp.cta_group::1.32x128b.warpx4", "tcgen05.mma.cta_group::1"
    commit = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
    code = (
        "{\n.reg .pred e,p0,pt,pc,pa,pe; .reg .b32 i<4>; .reg .b64 xa<4>,xb<4>;\n"
        "elect.sync _|e,0xffffffff; setp.eq.b32 pt,0,0; setp.ne.b32 pc,$7,0;\n"
        "and.pred pc,pc,e; setp.ne.b32 pa,$10,0; and.pred pa,pa,e;\n"
        "setp.ne.b32 pe,$9,0; and.pred pe,pe,e; setp.ne.b32 p0,$8,0;\n"
        "mov.b32 i0,142738048; or.b32 i1,i0,536870928; or.b32 i2,i0,1073741856;\n"
        f"or.b32 i3,i0,1610612784; @pc {cp} [$5],$6; @e {cp} [$3],$4;\n"
        "add.s64 xa0,$1,0; add.s64 xb0,$2,0; add.s64 xa1,$1,2; add.s64 xb1,$2,2;\n"
        "add.s64 xa2,$1,4; add.s64 xb2,$2,4; add.s64 xa3,$1,6; add.s64 xb3,$2,6;\n"
    )
    for k in range(4):
        code += f"@e {mma}.kind::mxf8f6f4.block_scale [$0],\n"
        code += f"xa{k},xb{k},i{k},[$3],[$5],{'pt' if k else 'p0'};\n"
    code += f"@pe {commit} [$9]; @pa {commit} [$10];\n}}"
    return code, constraints, tuple(t(v) for t, v in zip(types, args))


def launch(kernel, args, grid, block, cluster=None, pdl=True, **options):
    """Launch kernel(args[:-1]) on stream args[-1]; an int dimension means (n, 1, 1)."""
    dim = lambda d: d if isinstance(d, tuple) else (d, 1, 1)
    if cluster is not None:
        options["cluster"] = dim(cluster)
    kernel(args[:-1]).launch(
        grid=dim(grid), block=dim(block), stream=args[-1], use_pdl=pdl, **options
    )


@cute.jit
def scatter_reduce(values, lane, count: cutlass.Constexpr[int]):
    """Warp-reduce ``count`` values; lane groups end up owning one value each."""
    selected = Int32(0)
    for stage in cutlass.range_constexpr(5):
        offset = 16 >> stage
        remaining = max(count >> stage, 1)
        if cutlass.const_expr(remaining > 1):
            half = remaining // 2
            upper = (lane & offset) != 0
            for i in cutlass.range_constexpr(half):
                send = values[i] if upper else values[i + half]
                keep = values[i + half] if upper else values[i]
                values[i] = keep + cute.arch.shuffle_sync_bfly(send, offset)
            if upper:
                selected += half
        else:
            values[0] += cute.arch.shuffle_sync_bfly(values[0], offset)
    return values[0], selected
