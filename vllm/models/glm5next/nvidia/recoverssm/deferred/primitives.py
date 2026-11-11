# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL primitives of the KDA kernels."""

import math
from functools import partial, wraps

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def ptx(assembly, constraints, operands=(), result=None, *, effects, loc=None, ip=None):
    """One verbatim asm block; a tuple `result` (even a 1-tuple) returns a tuple."""
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
    """`f(*args) -> (code, constraints, operands[, result, pure])` as one inline-PTX
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


@asm(Float32, pure=True)
def fma(a, b, c, *, ftz=False):
    return f"fma.rn{'.ftz' if ftz else ''}.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, c)


def bits_float(bits):
    return Float32(llvm.bitcast(T.f32(), Uint32(bits).ir_value()))


def bf16(word, high=False):
    """FP32 of the low (or high) BF16 half of `word`."""
    return bits_float(word & 0xFFFF0000 if high else word << 16)


@asm((Float32, Float32), pure=True)
def fma2(a, b, c):
    """Lanewise a * b + c of FP32 pairs in one packed instruction (IEEE per lane)."""
    code = "{ .reg .b64 a,b,c,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    code += "mov.b64 c,{$6,$7}; fma.rn.f32x2 d,a,b,c; mov.b64 {$0,$1},d; }"
    return code, "=f,=f,f,f,f,f,f,f", (*a, *b, *c)


@cute.jit
def warp_sum(value: Float32, lanes: cutlass.Constexpr[int] = 32):
    for shift in cutlass.range_constexpr(lanes.bit_length() - 2, -1, -1):
        value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
    return value


@asm((Float32,) * 4, pure=True)
def mma(a, b, accumulator, *, fp8=False, bf16=False):
    """m16n8k16 (FP8: k32) mma.sync."""
    kind = "e4m3" if fp8 else "bf16" if bf16 else "f16"
    code = f"mma.sync.aligned.m16n8k{32 if fp8 else 16}.row.col.f32.{kind}.{kind}.f32 "
    code += "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};"
    return code, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", (*a, *b, *accumulator)


def mma_into(acc, a, b, **kwargs):
    """`acc += a @ b` with one `mma`; `acc` is a four-element register view."""
    result = mma(a, b, tuple(acc[k] for k in range(4)), **kwargs)
    for k in range(4):
        acc[k] = result[k]


def at(tensor, *coord):
    """Pointer to `tensor[coord]`."""
    return tensor.iterator + cute.crd2idx(coord, tensor.layout)


def zeros(shape, dtype):
    tensor = cute.make_rmem_tensor(shape, dtype)
    tensor.fill(0)
    return tensor


def alloc(allocator, dtype, shape, align=16):
    """Row-major `shape` tensor from a `SmemAllocator`."""
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
    """`count` words (a tuple); `scope` is a relaxed load at that scope.
    `mutable` (a side-effecting load) for data written during the step (activations,
    tables, refills). Only static weights use `evict_first`."""
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


load_f32x4 = partial(load_words, count=4, result_type=Float32, word_type="f32")


@asm()
def store_f32x4(pointer, values, when=None):
    """`when`: a predicate, so callers need no branch."""
    code = "st.global.v4.f32 [$0], {$1, $2, $3, $4};"
    if when is None:
        return code, "l,f,f,f,f", (pointer, *values)
    code = "{ .reg .pred p; setp.ne.b32 p,$5,0; @p " + code + " }"
    return code, "l,f,f,f,f,r", (pointer, *values, Int32(when))


@asm()
def push_float(pointer, value, barrier, peer, when=None):
    """DSMEM store of one FP32 `value` (or a 4-tuple) completing on `barrier`
    (`when`: predicated on it)."""
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
def copy_g2s(dst, src, size=16, barrier=None, *, evict_first=False):
    """16-byte cp.async, or a static/dynamic `size` bulk copy on `barrier`."""
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


@asm()
def prefetch_bytes(src, size):
    return "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r", (src, Int32(size))


@asm(Int32)
def try_wait(barrier, phase=0, *, test=False, scope="cluster"):
    """Acquire parity probe at `scope` ("cluster", or "cta" when every arrival is
    CTA-local); `test` is the non-blocking probe of phase 0."""
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
    """Warp-reduce `count` values; lane groups end up owning one value each."""
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
