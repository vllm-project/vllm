# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared device runtime of the CuTe DSL GLM-5.3 kernel families."""

import contextlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, make_dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar, dataclass_transform

import cutlass
import cutlass.cute as cute
import regex as re
from cutlass import (
    BFloat16,
    Float32,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, nvvm, scf
from cutlass.base_dsl.typing import GridConstant
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental.cuda.tensor_map import (
    TensorMap,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)

if TYPE_CHECKING:

    class Generated(type):
        """For type checkers: class attributes added at import (PTX tables, shared
        memory maps)."""

        def __getattr__(cls, name: str) -> Any: ...

else:
    Generated = type


def _constraint(t) -> str:
    if t in (Int64, Uint64):
        return "l"
    if t is Float32:
        return "f"
    if t in (Int16, Uint16):
        return "h"
    return "r"


def _is_int_const(value) -> bool:
    owner = value.owner
    op = getattr(owner, "operation", owner)
    return getattr(op, "name", "") == "arith.constant" and "i" in str(value.type)


HIDDEN, RANKS = 6144, 8
VOCAB, SHARD_ROWS = 154880, 19360
_FCMP = {"==": "eq", "!=": "neu", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}


def _c_intop(a, b, op, py):
    if isinstance(a, int) and isinstance(b, int):
        return py(a, b)
    if isinstance(a, (Int32, int)) and isinstance(b, (Int32, int)):
        return Int32(op(Int32(a).ir_value(), Int32(b).ir_value()))
    return py(a, b)


_SHARED_SUFFIX = {Float32: "f32", Uint32: "u32", Uint64: "u64", Uint16: "b16"}


_ASM_IR = {"r": T.i32, "f": T.f32, "l": T.i64, "h": T.i16}
_ASM_OUT = {"r": Uint32, "f": Float32, "l": Uint64, "h": Uint16}
_ASM_CAST = {"r": Int32, "f": Float32, "l": Int64, "h": Int16}


def _asm_operand(value, letter, loc, ip):
    if isinstance(value, cute.Pointer):
        value = value.toint()
    if letter == "r" and not isinstance(value, int):
        if getattr(type(value), "width", 32) != 32:
            value = Int32(value)
        return value.ir_value(loc=loc, ip=ip)
    return _ASM_CAST[letter](value).ir_value(loc=loc, ip=ip)


_MMA_BF16 = (
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, "
    "{$4, $5, $6, $7}, {$8, $9},"
    " {$10, $11, $12, $13};"
)


def _seq(x, n: int) -> list:
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise TypeError(f"expected {n} operands, got {len(x)}")
        return list(x)
    return [x[i] for i in range(n)]


def _list(x) -> list:
    if not isinstance(x, (list, tuple)):
        raise TypeError(
            f"pass a list or tuple of operands, not {type(x).__name__} (its size is "
            "not known)"
        )
    return list(x)


def _atom128(dtype):
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128
    )


def zeros(shape):
    """A Float32 register tensor of ``shape`` (an int or a 2-tuple) set to 0.0 element
    by element.
    """
    count = shape if isinstance(shape, int) else shape[0] * shape[1]
    t = cute.make_rmem_tensor(shape, Float32)
    for i in range(count):
        t[i] = Float32(0.0)
    return t


class Core:
    """Inline-PTX emitter and model constants."""

    @staticmethod
    @dsl_user_op
    def ptx(
        code: str,
        *ins,
        outs: tuple = (),
        volatile: bool = True,
        memory: bool | None = None,
        imm: bool = False,
        loc=None,
        ip=None,
    ):
        ins = [a if isinstance(a, cutlass.Numeric) else Int32(a) for a in ins]  # type: ignore[assignment]
        values = [a.ir_value(loc=loc, ip=ip) for a in ins]
        constraints = [f"={_constraint(t)}" for t in outs] + [
            "n" if imm and _is_int_const(v) else _constraint(type(a))
            for a, v in zip(ins, values)
        ]
        if volatile if memory is None else memory:
            constraints.append("~{memory}")
        if len(outs) == 0:
            result_type = None
        elif len(outs) == 1:
            result_type = outs[0].mlir_type
        else:
            result_type = llvm.StructType.get_literal([t.mlir_type for t in outs])
        joined = ",".join(constraints)
        result = llvm.inline_asm(
            result_type, values, code, joined, has_side_effects=volatile, loc=loc, ip=ip
        )
        if len(outs) == 0:
            return None
        if len(outs) == 1:
            return outs[0](result)
        return tuple(
            t(llvm.extractvalue(t.mlir_type, result, [i], loc=loc, ip=ip))
            for i, t in enumerate(outs)
        )

    @staticmethod
    def pure(code: str, outs, *ins, imm: bool = False):
        outs = outs if isinstance(outs, tuple) else (outs,)
        return Core.ptx(code, *ins, outs=outs, volatile=False, imm=imm)

    @staticmethod
    def vol(code: str, outs, *ins, imm: bool = False):
        outs = outs if isinstance(outs, tuple) else (outs,)
        return Core.ptx(code, *ins, outs=outs, memory=False, imm=imm)

    @staticmethod
    def mem(code: str, outs, *ins):
        return Core.ptx(code, *ins, outs=outs if isinstance(outs, tuple) else (outs,))

    HIDDEN, VOCAB, SHARD_ROWS, RANKS = HIDDEN, VOCAB, SHARD_ROWS, RANKS

    @staticmethod
    @dsl_user_op
    def asm(
        code: str,
        outs: str,
        ins: str,
        *args,
        v: bool = False,
        m: bool = False,
        types=None,
        loc=None,
        ip=None,
    ):
        out_l = [c.strip() for c in outs.split(",") if c.strip()]
        in_l = [c.strip() for c in ins.split(",") if c.strip()]
        tied = [i for i, c in enumerate(out_l) if c.startswith("+")]
        assert len(args) == len(tied) + len(in_l), (code, outs, ins, len(args))
        ir_in = [_asm_operand(a, c, loc, ip) for a, c in zip(args[len(tied) :], in_l)]
        ir_in += [
            _asm_operand(a, out_l[i][1], loc, ip)
            for a, i in zip(args[: len(tied)], tied)
        ]
        cons = ["=" + c[1] for c in out_l] + in_l + [str(i) for i in tied]
        if m:
            cons.append("~{memory}")
        text = re.sub(r"%(\d+)", r"$\1", code.replace("$", "$$")).replace("%%", "%")
        letters = [c[1] for c in out_l]
        types = [_ASM_OUT[c] for c in letters] if types is None else types
        if not out_l:
            llvm.inline_asm(
                None, ir_in, text, ",".join(cons), has_side_effects=v, loc=loc, ip=ip
            )
            return None
        assert len(out_l) == 1, "asm with several outputs goes through Core.ptx"
        out_type, joined = _ASM_IR[letters[0]](), ",".join(cons)
        r = llvm.inline_asm(
            out_type, ir_in, text, joined, has_side_effects=v, loc=loc, ip=ip
        )
        return types[0](r)


_PTX_ROW = re.compile(
    r"(\w+)\((.*)\)(?:\s*->\s*(\w+)(?:\*(\d+))?)?\s*\|\s*(\w+)(\+imm)?\s*\|\s*(.+)"
)
_PTX_TYPES = {
    t.__name__: t for t in (Float32, Int16, Int32, Int64, Uint16, Uint32, Uint64)
}


def _ptx_helper(row: str):
    name, params, out, count, mode, imm, code = _PTX_ROW.fullmatch(row.strip()).groups()
    outs = () if out is None else (_PTX_TYPES[out],) * int(count or 1)
    spec, sig = [], []
    for p in filter(None, (p.strip() for p in params.split(","))):
        pname, _, default = p.partition("=")
        pname, _, ty = pname.partition(":")
        width = re.fullmatch(r"\s*(\w+)(?:\[(\d+)\])?", ty) if ty else None
        if width:
            spec.append(
                (pname.strip(), _PTX_TYPES[width[1]], width[2] and int(width[2]))
            )
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        dflt = inspect.Parameter.empty if not default else default.strip() == "True"
        sig.append(inspect.Parameter(pname.strip(), kind, default=dflt))
    signature = inspect.Signature(sig)

    def helper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        memory = bound.arguments.get("memory", mode == "mem")
        vals = [bound.arguments[pname] for pname, _, _ in spec]
        if "%" in code:
            outs_l = ",".join("=" + _constraint(t) for t in outs)
            ins_l = ",".join(_constraint(t) for _, t, _ in spec)
            types = list(outs) or None
            return Core.asm(
                code, outs_l, ins_l, *vals, v=mode != "pure", m=memory, types=types
            )
        ins = []
        for v, (_, ty, n) in zip(vals, spec):
            ins += [ty(x) for x in _seq(v, n)] if n else [ty(v)]
        return Core.ptx(
            code, *ins, outs=outs, volatile=mode != "pure", memory=memory, imm=bool(imm)
        )

    helper.__name__ = helper.__qualname__ = name
    helper.__signature__ = signature  # type: ignore[attr-defined]
    helper.__doc__ = f"``{code}``"
    return name, helper


def ptx_table(cls):
    """Attach a static method for every row of ``cls.PTX``: ``name(param: Type, ...) ->
    Out[*n] | mode | template``.
    """
    for row in filter(str.strip, cls.PTX.splitlines()):
        name, helper = _ptx_helper(row)
        setattr(cls, name, staticmethod(helper))
    return cls


@ptx_table
class FastMath(metaclass=Generated):
    """Fast-math f32 arithmetic and C integer division."""

    PTX = """
    add(a: Float32, b: Float32) -> Float32              | pure    | add.ftz.f32 $0, $1, $2;
    sub(a: Float32, b: Float32) -> Float32              | pure    | sub.ftz.f32 $0, $1, $2;
    mul(a: Float32, b: Float32) -> Float32              | pure    | mul.ftz.f32 $0, $1, $2;
    fma(a: Float32, b: Float32, c: Float32) -> Float32  | pure    | fma.rn.ftz.f32 $0, $1, $2, $3;
    div_rn(a: Float32, b: Float32) -> Float32           | pure    | div.rn.ftz.f32 $0, $1, $2;
    rsqrt(a: Float32) -> Float32                        | pure    | rsqrt.approx.ftz.f32 $0, $1;
    fmax(a: Float32, b: Float32) -> Float32             | pure    | max.ftz.f32 $0, $1, $2;
    rcp_rn(a: Float32) -> Float32                       | pure    | rcp.rn.ftz.f32 $0, $1;
    max3(a: Float32, b: Float32, c: Float32) -> Float32 | vol     | max.f32 $0, $1, $2, $3;
    max_noftz(a: Float32, b: Float32) -> Float32        | pure    | max.f32 $0, $1, $2;
    abs_ftz(x: Float32) -> Float32                      | vol+imm | abs.ftz.f32 $0, $1;
    lg2_approx(x: Float32) -> Float32                   | vol     | lg2.approx.f32 $0, $1;
    """  # noqa: E501

    @staticmethod
    def fcmp(op: str, a, b):
        code = (
            "{\n\t.reg .pred p;\n\tsetp."
            + _FCMP[op]
            + ".ftz.f32 p, $1, $2;\n\tselp.u32 $0, 1, 0, p;\n\t}"
        )
        flag = Core.pure(code, Int32, Float32(a), Float32(b))
        return flag != 0

    @staticmethod
    def cdiv(a, b):
        return _c_intop(a, b, arith.divsi, lambda x, y: x // y)

    @staticmethod
    def cmod(a, b):
        return _c_intop(a, b, arith.remsi, lambda x, y: x % y)

    @staticmethod
    def fma2(a, b, c) -> tuple:
        code = (
            "{\n\t.reg .b64 a, b, c, d;\n\tmov.b64 a, {$2, $3};\n\t"
            "mov.b64 b, {$4, $5};\n\tmov.b64 c, {$6, $7};\n\t"
            "fma.rn.ftz.f32x2 d, a, b, c;\n\tmov.b64 {$0, $1}, d;\n\t}"
        )
        return Core.pure(
            code,
            (Float32, Float32),
            *[Float32(x) for x in (a[0], a[1], b[0], b[1], c[0], c[1])],
        )

    @staticmethod
    def mul2(a, b) -> tuple:
        code = (
            "{\n\t.reg .b64 a, b, d;\n\tmov.b64 a, {$2, $3};\n\t"
            "mov.b64 b, {$4, $5};\n\tmul.rn.ftz.f32x2 d, a, b;\n\t"
            "mov.b64 {$0, $1}, d;\n\t}"
        )
        return Core.pure(
            code, (Float32, Float32), *[Float32(x) for x in (a[0], a[1], b[0], b[1])]
        )

    @staticmethod
    @dsl_user_op
    def cdiv64(a, b, *, loc=None, ip=None) -> Int64:
        return Int64(
            arith.divsi(
                Int64(a).ir_value(loc=loc, ip=ip),
                Int64(b).ir_value(loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
        )

    @staticmethod
    @dsl_user_op
    def select_i32(cond, a, b, *, loc=None, ip=None) -> Int32:
        return Int32(
            arith.select(
                cutlass.Boolean(cond).ir_value(loc=loc, ip=ip),
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
        )

    @staticmethod
    def min_typed(a, b):
        if isinstance(b, int):
            b = type(a)(b)
        return type(a)(cutlass.select_(a < b, a, b))

    @staticmethod
    def max_typed(a, b):
        return type(a)(cutlass.select_(a > b, a, b))


@ptx_table
class Convert(metaclass=Generated):
    """Bit casts, bf16 conversions and E4M3 -> bf16 MMA A fragments."""

    PTX = """
    f32x2_to_bf16x2(lo: Float32, hi: Float32) -> Uint32              | pure    | cvt.rn.bf16x2.f32 $0, $2, $1;
    f32_to_bf16(x: Float32) -> Uint16                                | pure    | cvt.rn.bf16.f32 $0, $1;
    bf16_to_f32(h: Uint16) -> Float32                                | pure    | cvt.f32.bf16 $0, $1;
    e4m3x2_to_bf16x2(h: Uint16) -> Uint32                            | vol+imm | cvt.rn.bf16x2.e4m3x2 $0, $1;
    unpack_u16x4(x: Uint64) -> Uint16*4                              | vol+imm | mov.b64 {$0, $1, $2, $3}, $4;
    ffs(x: Uint32) -> Int32                                          | vol+imm | { .reg .b32 t; brev.b32 t, $1; bfind.shiftamt.u32 t, t; add.s32 $0, t, 1; }
    as_u32(f: Float32) -> Uint32                                     | pure    | mov.b32 $0, $1;
    e4m3x4(a: Float32, b: Float32, c: Float32, d: Float32) -> Uint32 | vol+imm | { .reg .b16 l, h; cvt.rn.satfinite.e4m3x2.f32 l, $2, $1; cvt.rn.satfinite.e4m3x2.f32 h, $4, $3; mov.b32 $0, {l, h}; }
    bf16_round(x: Float32) -> Float32                                | vol+imm | { .reg .b16 h, z; .reg .b32 w; cvt.rn.bf16.f32 h, $1; mov.b16 z, 0; mov.b32 w, {z, h}; mov.b32 $0, w; }
    f32x2_to_e4m3x2(hi: Float32, lo: Float32) -> Uint16              | pure    | cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;
    """  # noqa: E501

    @staticmethod
    def f32_bits(x) -> Uint32:
        return Uint32(llvm.bitcast(T.i32(), Float32(x).ir_value()))

    @staticmethod
    def bits_f32(x) -> Float32:
        return Float32(llvm.bitcast(T.f32(), Uint32(x).ir_value()))

    @staticmethod
    def bf16x2_to_f32x2(word) -> tuple[Float32, Float32]:
        code = "{\n\tshl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xffff0000;\n\t}\n"
        return Core.vol(code, (Float32, Float32), Uint32(word))

    @staticmethod
    def bf16x8_to_f32(words) -> list:
        return [v for w in _list(words) for v in Convert.bf16x2_to_f32x2(w)]

    @staticmethod
    @cute.jit
    def ld_u32x4(src: cute.Tensor) -> cute.Tensor:
        atom = _atom128(Uint32)
        raw = cute.make_rmem_tensor(4, Uint32)
        cute.copy(atom, src, raw)
        return raw

    @staticmethod
    @cute.jit
    def unpack8(raw: cute.Tensor) -> cute.Tensor:
        out = cute.make_rmem_tensor(8, Float32)
        for pair in cutlass.range_constexpr(4):
            out[2 * pair], out[2 * pair + 1] = Convert.bf16x2_to_f32x2(raw[pair])
        return out

    @staticmethod
    @cute.jit
    def ld_bf16x8_f32(src: cute.Tensor) -> cute.Tensor:
        return Convert.unpack8(Convert.ld_u32x4(cute.recast_tensor(src, Uint32)))

    @staticmethod
    @cute.jit
    def st_f32_bf16x8(vals: cute.Tensor, dst: cute.Tensor):
        atom = _atom128(Uint32)
        packed = cute.make_rmem_tensor(4, Uint32)
        for pair in cutlass.range_constexpr(4):
            packed[pair] = Convert.f32x2_to_bf16x2(vals[2 * pair], vals[2 * pair + 1])
        cute.copy(atom, packed, cute.recast_tensor(dst, Uint32))

    @staticmethod
    def a_frag_u32(lo, hi) -> tuple:
        code = (
            "{\n\t.reg .b64 p;\n\t.reg .b16 h0, h1, h2, h3;\n\tmov.b64 p, {$4, $5};\n\t"
            "mov.b64 {h0, h1, h2, h3}, p;\n\tcvt.rn.bf16x2.e4m3x2 $0, h0;\n\t"
            "cvt.rn.bf16x2.e4m3x2 $1, h2;\n\t"
            "cvt.rn.bf16x2.e4m3x2 $2, h1;\n\tcvt.rn.bf16x2.e4m3x2 $3, h3;\n\t}"
        )
        return Core.pure(code, (Uint32,) * 4, Uint32(lo), Uint32(hi))

    @staticmethod
    def a_frag_u64(row_a, row_b, k_half: int) -> tuple:
        i, j = 2 * k_half, 2 * k_half + 1
        code = (
            "{\n\t.reg .b16 a0, a1, a2, a3, b0, b1, b2, b3;\n\t"
            "mov.b64 {a0, a1, a2, a3}, $4;\n\t"
            f"mov.b64 {{b0, b1, b2, b3}}, $5;\n\tcvt.rn.bf16x2.e4m3x2 $0, a{i};\n\t"
            f"cvt.rn.bf16x2.e4m3x2 $1, b{i};\n\t"
            f"cvt.rn.bf16x2.e4m3x2 $2, a{j};\n\tcvt.rn.bf16x2.e4m3x2 $3, b{j};\n\t}}"
        )
        return Core.pure(code, (Uint32,) * 4, Uint64(row_a), Uint64(row_b))

    @staticmethod
    def ld_bf16x8_words(words, off) -> list:
        return Convert.bf16x8_to_f32([words[off + i] for i in range(4)])

    @staticmethod
    def prmt(a, b, sel: int) -> Uint32:
        return Core.pure(
            f"prmt.b32 $0, $1, $2, {sel:#x};", Uint32, Uint32(a), Uint32(b)
        )

    @staticmethod
    def e4m3_pair_to_a(w0, w1) -> list:
        halves = (w0, w1, w0 >> 16, w1 >> 16)
        return [Convert.e4m3x2_to_bf16x2(Uint16(h)) for h in halves]

    @staticmethod
    def bf16_low(x) -> Uint32:
        return Convert.f32x2_to_bf16x2(x, 0.0) & Uint32(65535)

    @staticmethod
    def words_to_f32(words) -> list:
        return [Convert.bits_f32(w) for w in _list(words)]

    @staticmethod
    def to_bf16(x):
        return Float32(x).to(cutlass.BFloat16)

    @staticmethod
    def widen_e4m3_pairs_shared(pred, own, partner) -> None:
        text = [
            "{ .reg .pred p; .reg .b64 d<8>; .reg .b16 h<32>; .reg .b32 w<32>;",
            "setp.ne.s32 p, %2, 0;",
        ]
        text += [
            f"@p ld.shared.v2.b64 {{d{2 * part}, d{2 * part + 1}}}, [%0+{part * 2048}];"
            for part in range(4)
        ]
        for part in range(4):
            for half, slot in ((0, "%0"), (1, "%1")):
                k = 2 * part + half
                h = [f"h{4 * k + i}" for i in range(4)]
                w = [f"w{4 * k + i}" for i in range(4)]
                text.append(f"mov.b64 {{{', '.join(h)}}}, d{k};")
                text.extend(f"cvt.rn.bf16x2.e4m3x2 {w[i]}, {h[i]};" for i in range(4))
                text.append(
                    f"@p st.shared.v4.b32 [{slot}+{part * 2048}], {{{', '.join(w)}}};"
                )
        text.append("}")
        Core.asm("\n\t".join(text), "", "r,r,r", own, partner, pred, v=True, m=True)


class WarpLeader:
    """``with WarpLeader(warp, n):`` runs its body in one elected lane of warp ``n``."""

    __dsl_trace_time_ctxmgr__ = True

    def __init__(self, warp, n: int) -> None:
        self.warp, self.n = warp, n
        self.regions: list[ir.InsertionPoint] = []

    def _enter(self, cond) -> None:
        if_op = scf.IfOp(cutlass.Boolean(cond).ir_value())
        region = ir.InsertionPoint(if_op.then_block)
        region.__enter__()
        self.regions.append(region)

    def __enter__(self) -> None:
        self._enter(self.warp == self.n)
        self._enter(Sync.elect_sync() != 0)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        while self.regions:
            if exc_type is None:
                scf.yield_([])
            self.regions.pop().__exit__(exc_type, exc_value, traceback)


@ptx_table
class Sync(metaclass=Generated):
    """Control flow, warp collectives and CTA synchronization (barriers, PDL, elect,
    shuffles).
    """

    PTX = """
    trap()                                   | mem     | trap;
    redux_max_nan_f32(x: Float32) -> Float32 | vol     | redux.sync.max.NaN.f32 $0, $1, 0xffffffff;
    redux_min_u32(x: Uint32) -> Uint32       | vol     | redux.sync.min.u32 $0, $1, 0xffffffff;
    redux_max_u32(x: Uint32) -> Uint32       | vol+imm | redux.sync.max.u32 $0, $1, 0xffffffff;
    redux_add_u32(x: Uint32) -> Uint32       | vol+imm | redux.sync.add.u32 $0, $1, 0xffffffff;
    syncwarp()                               | mem     | bar.warp.sync 0xffffffff;
    smid() -> Uint32                         | vol     | mov.u32 %0, %%smid;
    launch_dependents_asm()                  | mem     | griddepcontrol.launch_dependents;
    """  # noqa: E501

    @staticmethod
    def bar_sync(barrier_id: int, threads: int) -> None:
        Core.ptx(f"barrier.sync {barrier_id}, {threads};")

    @staticmethod
    def pdl_prologue() -> None:
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

    @staticmethod
    def warp_uniform(x) -> Int32:
        return Int32(
            Core.vol(
                "shfl.sync.idx.b32 $0, $1, 0, 0x1F, 0xFFFFFFFF;", Uint32, Uint32(x)
            )
        )

    @staticmethod
    def elect_sync() -> Int32:
        code = (
            "{\n\t.reg .pred px;\n\telect.sync _|px, $1;\n\tselp.u32 $0, 1, 0, px;\n\t}"
        )
        return Core.vol(code, Int32, Uint32(0xFFFFFFFF))

    @staticmethod
    def globaltimer() -> Uint64:
        return Uint64(cute.arch.globaltimer())

    @staticmethod
    def vote_all_asm(pred) -> Uint32:
        code = (
            "{\n\t.reg .pred p, q;\n\tsetp.ne.s32 p, $1, 0;\n\t"
            "vote.sync.all.pred q, p, 0xffffffff;\n\t"
            "selp.u32 $0, 1, 0, q;\n\t}"
        )
        return Core.mem(code, Uint32, Uint32(pred))

    @staticmethod
    def popc(x) -> Int32:
        return Int32(cute.arch.popc(Uint32(x)))

    @staticmethod
    def shfl_up(x, delta):
        return cute.arch.shuffle_sync_up(x, delta, mask_and_clamp=0)

    @staticmethod
    def vote_all(pred) -> Int32:
        code = (
            "{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, $1, 0;\n\t"
            "vote.sync.all.pred q, p, 0xffffffff;\n\t"
            "selp.s32 $0, 1, 0, q;\n\t}\n"
        )
        return Core.vol(code, Int32, Int32(pred))

    @staticmethod
    def vote_any(pred, mask=0xFFFFFFFF) -> Int32:
        code = (
            "{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, $1, 0;\n\t"
            "vote.sync.any.pred q, p, $2;\n\t"
            "selp.s32 $0, 1, 0, q;\n\t}\n"
        )
        return Core.vol(code, Int32, Int32(pred), Uint32(mask))

    @staticmethod
    def vote_ballot(pred) -> Uint32:
        code = (
            "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $1, 0;\n\t"
            "vote.sync.ballot.b32 $0, p, 0xffffffff;\n\t}\n"
        )
        return Core.vol(code, Uint32, Int32(pred))

    @staticmethod
    def ballot(pred) -> Uint32:
        return Uint32(cute.arch.vote_ballot_sync(pred))


@ptx_table
class Memory(metaclass=Generated):
    """Shared- and global-memory access, memory ordering, atomics and peer
    addressing.
    """

    PTX = """
    lds_v4(addr: Uint32) -> Uint32*4                                 | mem     | ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];
    sts_v4(addr: Uint32, words: Uint32[4])                           | mem     | st.shared.v4.b32 [$0], {$1,$2,$3,$4};
    ldg_nc_v4(addr: Uint64) -> Uint32*4                              | mem     | ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];
    ldg_nc_b16(addr: Uint64) -> Uint16                               | mem     | ld.global.nc.b16 $0, [$1];
    stg_b16(addr: Uint64, value: Uint16)                             | mem     | st.global.b16 [$0], $1;
    st_global_v4(addr: Int64, words: Uint32[4])                      | mem     | st.global.v4.b32 [$0], {$1,$2,$3,$4};
    ld_volatile_v4(address: Int64) -> Uint32*4                       | mem     | ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];
    st_volatile_v4(address: Int64, words: Uint32[4])                 | mem     | st.volatile.global.v4.b32 [$0], {$1, $2, $3, $4};
    st_relaxed_sys(address: Int64, word: Uint32)                     | mem     | st.relaxed.sys.u32 [$0], $1;
    ld_relaxed_gpu(address: Int64) -> Uint32                         | mem     | ld.relaxed.gpu.u32 $0, [$1];
    atom_add_u32(address: Uint64, value: Uint32) -> Uint32           | mem     | atom.global.add.u32 $0, [$1], $2;
    red_add_u32(address: Uint64, value: Uint32)                      | mem     | red.global.add.u32 [$0], $1;
    fence_release_gpu()                                              | mem     | fence.release.gpu;
    fence_acquire_gpu()                                              | mem     | fence.acquire.gpu;
    fence_proxy_async_shared()                                       | mem     | fence.proxy.async.shared::cta;
    st_shared_v2(addr: Uint32, a: Uint32, b: Uint32)                 | mem     | st.shared.v2.b32 [$0], {$1, $2};
    ld_cg_v4(gaddr: Int64, memory=False) -> Uint32*4                 | vol+imm | ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];
    ld_cg_v2(gaddr: Int64) -> Uint32*2                               | vol+imm | ld.global.cg.v2.b32 {$0, $1}, [$2];
    ld_cg_b32(gaddr: Int64) -> Uint32                                | vol+imm | ld.global.cg.b32 $0, [$1];
    red_async_release_gpu_add(gaddr: Int64, v: Uint32, memory=False) | vol+imm | red.async.release.global.gpu.add.u32 [$0], $1;
    st_volatile_u64(gaddr: Int64, v: Uint64, memory=False)           | vol+imm | st.volatile.u64 [$0], $1;
    ld_volatile_u64(gaddr: Int64, memory=False) -> Uint64            | vol+imm | ld.volatile.u64 $0, [$1];
    ld_global_v8_f32(gaddr: Int64) -> Float32*8                      | vol+imm | ld.global.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];
    ld_global_v4(gaddr: Int64) -> Uint32*4                           | vol+imm | ld.global.v4.b32 {$0, $1, $2, $3}, [$4];
    ld_global_v8(gaddr: Int64, memory=False) -> Uint32*8             | vol+imm | ld.global.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];
    st_global_b32(gaddr: Int64, v: Uint32)                           | vol+imm | st.global.b32 [$0], $1;
    ld_volatile_shared_u32(addr: Int32) -> Uint32                    | vol+imm | ld.volatile.shared.u32 $0, [$1];
    atom_add_acq_rel_gpu(gaddr: Int64, v: Uint32) -> Uint32          | vol+imm | atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;
    atom_add_shared(addr: Uint32, value: Uint32) -> Uint32           | mem     | atom.shared.add.u32 $0, [$1], $2;
    atom_max_shared(addr: Uint32, value: Uint32) -> Uint32           | mem     | atom.shared.max.u32 $0, [$1], $2;
    st_shared_v2_f32(addr: Uint32, f0: Float32, f1: Float32)         | mem     | st.shared.v2.f32 [$0], {$1,$2};
    ld_global_v4_f32(addr: Int64, memory=True) -> Float32*4          | vol     | ld.global.v4.f32 {$0, $1, $2, $3}, [$4];
    st_global_v2_f32(addr: Int64, a: Float32, b: Float32)            | mem     | st.global.v2.f32 [$0], {$1, $2};
    cvta_shared_u32(address: Int64) -> Int32                         | mem     | { .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }
    fence_proxy_async_global()                                       | mem     | fence.proxy.async.global;
    ldg_nc_u32(address: Int64) -> Uint32                             | pure    | ld.global.nc.u32 $0, [$1];
    """  # noqa: E501

    @staticmethod
    def lds(addr, ty=Float32):
        return Core.mem(f"ld.shared.{_SHARED_SUFFIX[ty]} $0, [$1];", ty, Uint32(addr))

    @staticmethod
    def sts(addr, value, ty=Float32) -> None:
        Core.ptx(f"st.shared.{_SHARED_SUFFIX[ty]} [$0], $1;", Uint32(addr), ty(value))

    @staticmethod
    def lds_b128(addr) -> tuple:
        code = (
            "{\n\t.reg .b128 t;\n\t.reg .b64 lo, hi;\n\tld.shared.b128 t, [$4];\n\t"
            "mov.b128 {lo, hi}, t;\n\t"
            "mov.b64 {$0, $1}, lo;\n\tmov.b64 {$2, $3}, hi;\n\t}"
        )
        return Core.vol(code, (Uint32,) * 4, Uint32(addr))

    @staticmethod
    def lds_weak_b64(addr) -> tuple:
        code = (
            "{\n\t.reg .b64 t;\n\tld.weak.shared::cta.b64 t, [$2];\n\t"
            "mov.b64 {$0, $1}, t;\n\t}"
        )
        return Core.mem(code, (Uint32,) * 2, Uint32(addr))

    @staticmethod
    def smem_view(base, offset: int, dtype, layout) -> cute.Tensor:
        return cute.make_tensor(cute.recast_ptr(base + offset, dtype=dtype), layout)

    @staticmethod
    @cute.jit
    def select_peer(index: Int32, bases: list) -> Int64:
        base = bases[0]
        for peer in cutlass.range_constexpr(1, 8):
            if index == peer:
                base = bases[peer]
        return base

    @staticmethod
    def gmem_ptr(dtype, address, align: int = 16):
        return cute.make_ptr(
            dtype, address, cute.AddressSpace.gmem, assumed_align=align
        )

    @staticmethod
    def smem_base_u32(smem_raw) -> Int32:
        return Sync.warp_uniform(Int32(smem_raw.toint()))

    @staticmethod
    def smem_views(smem, table) -> dict:
        views = {}
        for name, off, dtype, n in table:
            ptr = cute.make_ptr(
                cutlass.Uint8,
                smem + off,
                cute.AddressSpace.smem,
                assumed_align=min(16, off & -off),
            )
            views[name] = cute.make_tensor(
                cute.recast_ptr(ptr, dtype=dtype), cute.make_layout(n)
            )
            views[name + "_addr"] = smem + off
        return views

    @staticmethod
    def ga(tensor, elem_off) -> Int64:
        return tensor.iterator.toint() + Int64(elem_off) * (
            tensor.element_type.width // 8
        )

    @staticmethod
    def atomic_add_u32(tensor, off, value):
        return cute.arch.atomic_add(
            tensor.iterator + off, Uint32(value), sem="relaxed", scope="gpu"
        )

    @staticmethod
    def baddr(t, byte_off) -> Int64:
        return t.iterator.toint() + Int64(byte_off)

    @staticmethod
    def typed(tensor, dtype):
        return (
            TypedView(tensor, dtype)
            if dtype in (Uint8, Uint16, Uint32, Uint64)
            else tensor
        )

    @staticmethod
    def gview(dtype, address, n: int = 1 << 30, align: int = 4):
        ptr = cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=align)
        return Memory.typed(cute.make_tensor(ptr, cute.make_layout(n)), dtype)

    @staticmethod
    def smem_view_n(smem, off: int, dtype, n: int):
        align = min(16, off & -off) if off else 16
        ptr = cute.make_ptr(
            dtype, smem + off, cute.AddressSpace.smem, assumed_align=align
        )
        return Memory.typed(cute.make_tensor(ptr, cute.make_layout(n)), dtype)

    @staticmethod
    def rmem_tensor(n: int, dtype):
        return Memory.typed(cute.make_rmem_tensor(n, dtype), dtype)

    @staticmethod
    def put(t, off: int, vals) -> None:
        for i, v in enumerate(_list(vals)):
            t[off + i] = v

    @staticmethod
    def take(t, off: int, n: int) -> list:
        return [t[off + i] for i in range(n)]

    @staticmethod
    def rmem_fill(n: int, dtype, fill):
        t = cute.make_rmem_tensor(n, dtype)
        for i in range(n):
            t[i] = dtype(fill)
        return t

    @staticmethod
    def smem_ptr(smem, off: int, dtype):
        align = min(16, off & -off) if off else 16
        ptr = cute.make_ptr(
            Uint8, smem + off, cute.AddressSpace.smem, assumed_align=align
        )
        return cute.recast_ptr(ptr, dtype=dtype)

    @staticmethod
    def at(t, i):
        return t.iterator + i

    @staticmethod
    def atomic_max_u32(t, off, v) -> None:
        cute.arch.atomic_max(t.iterator + off, Uint32(v), sem="relaxed", scope="gpu")

    @staticmethod
    def stg_u32x4(t, elem_off, w) -> None:
        src = cute.make_rmem_tensor(4, Uint32)
        for i in range(4):
            src[i] = w[i]
        ptr = cute.make_ptr(
            Uint32,
            (t.iterator + elem_off).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        dst = cute.make_tensor(ptr, cute.make_layout(4))
        cute.copy(_atom128(Uint32), src, dst)


@ptx_table
class AsyncCopy(metaclass=Generated):
    """mbarriers, TMA, bulk copies and ``cp.async``."""

    PTX = """
    mbar_init(addr: Uint32, count: Uint32)                           | mem     | mbarrier.init.shared::cta.b64 [$0], $1;
    mbar_arrive(addr: Uint32)                                        | mem     | mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];
    mbar_expect_tx(addr: Uint32, nbytes: Uint32)                     | mem     | mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $1;
    tma_3d(dst: Uint32, tmap: Uint64, c0: Uint32, c1: Uint32, c2: Uint32, mbar: Uint32, hint: Uint64) | mem     | cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1, {$2, $3, $4}], [$5], $6;
    bulk_g2s(dst: Uint32, src: Uint64, nbytes: Uint32, mbar: Uint32) | mem     | cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];
    createpolicy_evict_first(src: Int64, primary: Uint32, total: Uint32) -> Uint64 | vol     | createpolicy.range.global.L2::evict_first.L2::evict_unchanged.b64 $0, [$1], $2, $3;
    bulk_g2s_hint(dst: Uint32, src: Int64, nbytes: Uint32, mbar: Uint32, policy: Uint64) | vol     | cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;
    tma_4d(dst: Uint32, tmap: Uint64, c0: Uint32, c1: Uint32, c2: Uint32, c3: Uint32, mbar: Uint32, hint: Uint64) | mem     | cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1, {$2, $3, $4, $5}], [$6], $7;
    cp_async_cg16(dst: Uint32, src: Int64)                           | vol     | cp.async.cg.shared::cta.global [$0], [$1], 16;
    cp_async_cg16_clobber(dst: Uint32, src: Uint64)                  | mem     | cp.async.cg.shared::cta.global [$0], [$1], 16;
    cp_async_commit_clobber()                                        | mem     | cp.async.commit_group;
    cp_async_cg16_pred_hint(pred: Int32, saddr: Int32, gaddr: Int64, policy: Uint64) | vol+imm | { .reg .pred p; setp.ne.b32 p, $0, 0; @p cp.async.cg.shared::cta.global.L2::cache_hint [$1], [$2], 16, $3; }
    cp_async_cg16_pred(pred: Int32, saddr: Int32, gaddr: Int64)      | vol+imm | { .reg .pred p; setp.ne.b32 p, $0, 0; @p cp.async.cg.shared::cta.global [$1], [$2], 16; }
    cp_async_mbarrier_arrive_noinc(mbar: Int32)                      | vol+imm | cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];
    gather4(dst: Uint32, tmap: Int64, col: Uint32, row0: Uint32, row1: Uint32, row2: Uint32, row3: Uint32, mbar: Uint32, hint: Uint64) | mem     | cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1, {$2, $3, $4, $5, $6}], [$7], $8;
    prefetch_tensormap(tmap: Int64)                                  | mem     | prefetch.tensormap [%0];
    prefetch_tile_2d(tmap: Int64, c0: Int32, c1: Int32)              | mem     | cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];
    bulk_prefetch_l2(src: Int64, nbytes: Int32)                      | vol     | cp.async.bulk.prefetch.L2.global [%0], %1;
    """  # noqa: E501

    @staticmethod
    def mbar_wait(addr, parity) -> None:
        code = (
            "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [$0], $1, $2;\n\t"
            "@P1 bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}\n"
        )
        Core.ptx(code, Uint32(addr), Uint32(parity), Uint32(0x989680))

    @staticmethod
    def bulk_g2s_evict_first(dst_smem, src, nbytes: int, mbar_smem, mbar_ptr) -> None:
        code = (
            "createpolicy.range.global.L2::evict_first.L2::evict_unchanged.b64 $0, "
            "[$1], $2, $3;"
        )
        policy = Core.mem(code, Int64, Int64(src), Int32(nbytes), Int32(nbytes))
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, nbytes)
        code = (
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            ".L2::cache_hint [$0], [$1], $2, [$3], $4;"
        )
        Core.ptx(
            code, Int32(dst_smem), Int64(src), Int32(nbytes), Int32(mbar_smem), policy
        )

    @staticmethod
    def cp_async_wait_clobber(n: int) -> None:
        Core.ptx(f"cp.async.wait_group {n};")

    @staticmethod
    def cp_async_cg16_zfill_if(pred, dst, src, src_size, policy) -> None:
        code = (
            "{ .reg .pred p; setp.ne.s32 p, %3, 0; "
            "@p cp.async.cg.shared::cta.global.L2::cache_hint [%0], [%1], 16, %2, "
            "%4; }"
        )
        Core.asm(code, "", "r,l,r,r,l", dst, src, src_size, pred, policy, v=True)


@ptx_table
class TensorCore(metaclass=Generated):
    """Warp-level ``mma.sync`` on bf16 fragments."""

    PTX = """
    tcgen05_commit(mbar: Int32)         | vol+imm | tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];
    ldsm_x4(addr: Uint32) -> Uint32*4   | mem     | ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];
    ldsm_x2(addr: Uint32) -> Uint32*2   | mem     | ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];
    ldsm_t_b8(addr: Uint32) -> Uint32*2 | mem     | ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];
    """  # noqa: E501

    @staticmethod
    def mma(acc, a, b, zero: bool) -> list:
        c = [0.0] * 4 if zero else acc
        a_regs, b_regs = (
            [Uint32(x) for x in _seq(a, 4)],
            [Uint32(x) for x in _seq(b, 2)],
        )
        c_regs = [Float32(x) for x in _seq(c, 4)]
        return list(
            Core.vol(_MMA_BF16 + "\n", (Float32,) * 4, *a_regs, *b_regs, *c_regs)
        )

    @staticmethod
    def mma_acc(acc: cute.Tensor, a, b, zero: bool) -> None:
        d = TensorCore.mma(
            [acc[i] for i in range(4)], [a[i] for i in range(4)], [b[0], b[1]], zero
        )
        for i in range(4):
            acc[i] = d[i]

    @staticmethod
    def mma_acc_at(d, o: int, a, b) -> None:
        Memory.put(d, o, TensorCore.mma(Memory.take(d, o, 4), a, b, False))

    @staticmethod
    def tmem_ld_32x32b_x64(taddr) -> tuple:
        regs = ", ".join(f"${i}" for i in range(64))
        code = "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" + regs + "}, [$64];"
        return Core.vol(code, (Float32,) * 64, Int32(taddr), imm=True)

    @staticmethod
    def umma_f8f6f4_k4(a_lo, b_lo, tmem, accumulate) -> None:
        step = (
            "add.u32 a_lo, a_lo, 2; add.u32 b_lo, b_lo, 2; mov.b64 da, {a_lo, adhi}; "
            "mov.b64 db, {b_lo, bdhi};"
            " tcgen05.mma.cta_group::1.kind::f8f6f4 [$2], da, db, id, p1; "
        )
        code = (
            "{ .reg .pred p0, p1; .reg .b32 adhi, bdhi, a_lo, b_lo, id;"
            " .reg .b64 da, db;"
            " setp.ne.b32 p0, $3, 0; setp.ne.b32 p1, 1, 0; mov.b32 adhi, 0x40004040; "
            "mov.b32 bdhi, 0x40004040;"
            " mov.b32 id, 136314896; mov.b32 a_lo, $0; mov.b32 b_lo, $1; mov.b64 da, "
            "{a_lo, adhi};"
            " mov.b64 db, {b_lo, bdhi};"
            " tcgen05.mma.cta_group::1.kind::f8f6f4 [$2], da, db, id, p0; "
            + step * 3
            + "}"
        )
        Core.vol(
            code, (), Int32(a_lo), Int32(b_lo), Int32(tmem), Int32(accumulate), imm=True
        )

    @staticmethod
    def tmem_dealloc(taddr, cols: int) -> None:
        code = f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {cols};"
        Core.vol(code, (), Uint32(taddr), imm=True)

    @staticmethod
    @dsl_user_op
    def fence_after_thread_sync(*, loc=None, ip=None) -> None:
        nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)

    @staticmethod
    @dsl_user_op
    def fence_before_thread_sync(*, loc=None, ip=None) -> None:
        nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


class TypedView:
    """A tensor whose element loads come back as ``dtype`` (CuTe pointers
    are signless).
    """

    def __init__(self, tensor, dtype):
        self._t = tensor
        self._dtype = dtype

    def __getitem__(self, index):
        return self._dtype(self._t[index])

    def __setitem__(self, index, value):
        self._t[index] = value

    def __getattr__(self, name):
        return getattr(self._t, name)

    def __extract_mlir_values__(self):
        return self._t.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        return type(self)(self._t.__new_from_mlir_values__(values), self._dtype)

    def __get_mlir_types__(self):
        return self._t.__get_mlir_types__()


class ReadOnlyView(TypedView):
    """A read-only kernel argument loaded with ``ld.global.nc``."""

    def __getitem__(self, index):
        word = Core.pure(
            "ld.global.nc.b32 $0, [$1];", Uint32, Int64(Memory.ga(self._t, index))
        )
        return Convert.bits_f32(word) if self._dtype is Float32 else self._dtype(word)


class Prologue:
    """Kernel prologue: thread ids, the dynamic shared-memory base, mbarrier init, role
    entry.
    """

    @staticmethod
    def thread_ids():
        tid, _, _ = cute.arch.thread_idx()
        warp = Sync.warp_uniform(FastMath.cdiv(tid, 32))
        lane = FastMath.cmod(tid, 32)
        bid, _, _ = cute.arch.block_idx()
        return tid, bid, warp, lane

    @staticmethod
    def smem(alignment: int = 1024):
        smem_raw = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=alignment)
        return smem_raw, Memory.smem_base_u32(smem_raw)

    @staticmethod
    def init_mbarriers(smem, groups, fence: bool = True) -> None:
        for off, n, arrivals in groups:
            for i in range(n):
                AsyncCopy.mbar_init(smem + off + 8 * i, arrivals)
        if fence:
            cute.arch.mbarrier_init_fence()

    @staticmethod
    def role_regs(increase: bool, count: int) -> None:
        if increase:
            cute.arch.setmaxregister_increase(count)
        else:
            cute.arch.setmaxregister_decrease(count)


PUN_SUFFIX = {
    Uint8: "bytes",
    BFloat16: "bf16",
    Uint32: "words",
    Int32: "i32",
    Float32: "f32",
    Uint64: "u64",
}


@dataclass
class Region:
    """One shared-memory array of a layout: byte ``offset``, element ``dtype`` and
    ``shape``.
    """

    offset: int
    dtype: type
    shape: tuple
    also: tuple = ()
    arrivals: int | None = None
    name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj, owner: type):
        return self if obj is None else obj.base + self.offset

    @property
    def size(self) -> int:
        n = 1
        for extent in self.shape:
            n *= extent
        return n

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.width // 8  # type: ignore[attr-defined]

    def views(self) -> tuple:
        return ((self.name, self.dtype, self.size),) + tuple(
            (f"{self.name}_{PUN_SUFFIX[t]}", t, self.nbytes * 8 // t.width)
            for t in self.also
        )


class SmemAlloc:
    """Trace-time bump allocator of a kernel's dynamic shared memory."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Scopes: ["seq", cursor] or ["union", start, top].
        self._frames: list[list] = [["seq", 0]]

    @property
    def cursor(self) -> int:
        frame = self._frames[-1]
        return frame[1] if frame[0] == "seq" else frame[2]

    def alloc(self, nbytes: int, align: int = 16) -> int:
        frame = self._frames[-1]
        if frame[0] == "union":
            assert frame[1] % align == 0, (frame[1], align)
            frame[2] = max(frame[2], frame[1] + nbytes)
            return frame[1]
        off = -(-frame[1] // align) * align
        frame[1] = off + nbytes
        assert frame[1] <= self.capacity, (off, nbytes, self.capacity)
        return off

    def array(
        self, dtype: type, shape, *, align: int | None = None, also: tuple = ()
    ) -> Region:
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        region = Region(0, dtype, shape, tuple(also))
        region.offset = self.alloc(region.nbytes, align or dtype.width // 8)  # type: ignore[attr-defined]
        return region

    def mbarriers(self, n: int, arrivals: int | None = None) -> Region:
        region = self.array(Uint64, n, align=8)
        region.arrivals = arrivals
        return region

    def pad(self, nbytes: int) -> None:
        self.alloc(nbytes, align=1)

    @contextlib.contextmanager
    def union(self, align: int = 16):
        start = self.alloc(0, align)
        self._frames.append(["union", start, start])
        yield
        top = self._frames.pop()[2]
        self.alloc(top - start, align=1)

    @contextlib.contextmanager
    def arm(self):
        union = self._frames[-1]
        assert union[0] == "union", "arm() is only valid inside union()"
        self._frames.append(["seq", union[1]])
        yield
        union[2] = max(union[2], self._frames.pop()[1])

    def end(self) -> int:
        assert len(self._frames) == 1 and self.cursor <= self.capacity, (
            self.cursor,
            self.capacity,
        )
        return self.cursor


class SmemLayout(metaclass=Generated):
    """A kernel's shared-memory map declared as a class body of :class:`SmemAlloc`
    calls.
    """

    REGIONS: tuple = ()

    def __init__(self, base) -> None:
        self.base = base

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        alloc = vars(cls).get("a")
        if isinstance(alloc, SmemAlloc):
            alloc.end()
            cls.CAPACITY = alloc.capacity  # type: ignore[attr-defined]
            del cls.a
        cls.REGIONS = (
            tuple(r for r in vars(cls).values() if isinstance(r, Region)) or cls.REGIONS
        )

    @classmethod
    def barriers(cls) -> tuple:
        return tuple(
            (r.offset, r.size, r.arrivals)
            for r in cls.REGIONS
            if r.arrivals is not None
        )

    @classmethod
    def views(cls, order=()) -> tuple:
        regions = sorted(
            (r for r in cls.REGIONS if r.arrivals is None), key=lambda r: r.offset
        )
        rank = {name: i for i, name in enumerate(order)}
        regions.sort(key=lambda r: rank.get(r.name, len(rank)))
        return tuple(
            (name, r.offset, dtype, n) for r in regions for name, dtype, n in r.views()
        )

    @classmethod
    def record(
        cls, base, *, ring="", order=(), to_end=False, typed=False, bases=(), **extra
    ):
        values = {"base": base}
        values.update(
            (r.name, base + r.offset) for r in cls.REGIONS if r.arrivals is not None
        )
        if ring:
            values[ring] = Pipeline.Ring(
                values["ring_full"], values["ring_empty"], cls.ring_full.size
            )
        table = cls.views(order)
        if to_end:
            table = tuple(
                (n, off, t, (cls.CAPACITY - off) * 8 // t.width)
                for n, off, t, _ in table
            )
        views = Memory.smem_views(base, table)
        for name, _, dtype, _ in table:
            values[name] = Memory.typed(views[name], dtype) if typed else views[name]
        values.update((name + "_addr", views[name + "_addr"]) for name, *_ in table)
        values.update(extra)
        fields = [(name, object, field(default=None)) for name in values]
        return make_dataclass(
            cls.__name__ + "Record", fields, bases=bases, frozen=True
        )(**values)


def ptr(dtype: str, align: int = 16):
    """A params-record pointer field to ``dtype`` elements (the kernel's view and the
    launch ABI).
    """
    return field(metadata={"dtype": dtype, "align": align})


_Record = TypeVar("_Record")


@dataclass_transform(frozen_default=True, field_specifiers=(field, ptr))
def record(cls: type[_Record]) -> type[_Record]:
    """``dataclass(frozen=True)``, which type checkers also see take :func:`ptr`
    fields (no default)."""
    return dataclass(frozen=True)(cls)


def byte_map(
    address, dims, strides, box, swizzle: bool = False, l2_256b: bool = False
) -> TensorMap:
    """Host wrapper: a ``uint8`` TMA descriptor over ``address`` (``dims``/``box``
    innermost first, ``strides`` in bytes).
    """
    return create_tensor_map_tiled(
        address,
        Uint8,
        dims,
        [stride // 16 for stride in strides],
        box,
        swizzle=TensorMapSwizzle.s128b if swizzle else None,
        l2_promotion=TensorMapL2Promotion.l2_256b if l2_256b else None,
    )


class Params:
    """Mixin for a params record: pointers as typed views or addresses; peer bases."""

    EXTENT = 1 << 30
    UNSIGNED: dict = {}
    READ_ONLY: dict = {}

    @classmethod
    def view(cls, ptr, name: str):
        t = cute.make_tensor(ptr, cute.make_layout(cls.EXTENT))
        if name in cls.READ_ONLY:
            return ReadOnlyView(t, cls.READ_ONLY[name])
        return Memory.typed(t, cls.UNSIGNED.get(name))

    def _pointers(self, fn):
        names = [f.name for f in fields(self) if f.type is cute.Pointer]  # type: ignore[arg-type]
        return replace(self, **{n: fn(getattr(self, n), n) for n in names})  # type: ignore[type-var]

    def views(self):
        return self._pointers(self.view)

    def addresses(self):
        return self._pointers(lambda ptr, _: ptr.toint())

    @property
    def peer_bases(self) -> list:
        return [getattr(self, f"peer_base{i}") for i in range(8)]


@dataclass(frozen=True)
class ThreadIds:
    """The thread's ids, read once by the kernel prologue: thread, block, warp-uniform
    warp, lane.
    """

    tid: Int32
    bid: Int32
    warp: Int32
    lane: Int32


@dataclass(frozen=True)
class Role:
    """One warp role of a warp-specialized kernel."""

    method: str
    first: int
    last: int
    regs: int = 0
    increase: bool = True


class DecodeKernel(ABC):
    """One served decode kernel built from a params record and a fixed skeleton."""

    Params: type
    GRID: int
    BLOCK: int = 384
    SMEM_BYTES: int = 0
    #: ``.minnctapersm`` (required with ``setmaxnreg``); None leaves it out.
    MIN_BLOCKS_PER_MP: int | None = 1
    #: Emit ``.maxntid`` (``max_number_threads``) as well as ``.reqntid``.
    MAX_THREADS: bool = False
    USE_PDL: bool = True
    #: Host-wrapper arguments after the params record fields: ``(name, DSL type)``.
    HOST_ARGS: tuple = ()

    def __init__(self, **constants: int | bool) -> None:
        for name, value in constants.items():
            if not hasattr(type(self), name):
                raise TypeError(
                    f"{type(self).__name__} has no compile-time knob {name}"
                )
            setattr(self, name, value)

    def smem(self, base: Int32):
        return None

    def mbarriers(self) -> tuple:
        return ()

    @cute.jit
    def prologue(self, p):
        tid, bid, warp, lane = Prologue.thread_ids()
        _, base = Prologue.smem()
        s = self.smem(base)
        if cutlass.const_expr(len(self.mbarriers()) > 0):
            with WarpLeader(warp, 0):
                Prologue.init_mbarriers(base, self.mbarriers())
            cute.arch.sync_threads()
        return ThreadIds(tid, bid, warp, lane), s

    @abstractmethod
    def run(self, p, s, ids: ThreadIds) -> None:
        raise NotImplementedError

    @cute.jit
    def epoch_ticket(self, warp: Int32, full_epoch, slot) -> None:
        with WarpLeader(warp, 9):
            ticket = cute.arch.atomic_add(full_epoch, Uint32(1))
            slot[0] = FastMath.cdiv(ticket, Uint32(self.GRID))

    def tensor_maps(self, p) -> tuple:
        return ()

    @cute.kernel
    def kernel(self, p):
        ids, s = self.prologue(p)
        self.run(p, s, ids)

    def _launch_bounds(self) -> dict:
        bounds = {}
        if self.SMEM_BYTES:
            bounds["smem"] = self.SMEM_BYTES
        if self.MAX_THREADS:
            bounds["max_number_threads"] = [self.BLOCK, 1, 1]  # type: ignore[assignment]
        if self.MIN_BLOCKS_PER_MP is not None:
            bounds["min_blocks_per_mp"] = self.MIN_BLOCKS_PER_MP
        return bounds

    @cute.jit
    def launch(self, *args):
        *buffers, stream = args
        p = self.Params(*buffers)
        kernel = self.kernel(p, *self.tensor_maps(p))
        grid, block, bounds = (
            (self.GRID, 1, 1),
            (self.BLOCK, 1, 1),
            self._launch_bounds(),
        )
        kernel.launch(
            grid=grid, block=block, stream=stream, use_pdl=self.USE_PDL, **bounds
        )


class RoleKernel(DecodeKernel):
    """A warp-specialized kernel: :meth:`run` hands each warp range of ``ROLES`` to its
    role method.
    """

    ROLES: tuple[Role, ...]

    @cute.jit
    def run(self, p, s, ids: ThreadIds) -> None:
        for i in cutlass.range_constexpr(len(self.ROLES)):
            role = self.ROLES[i]
            if cutlass.const_expr(role.first == 0):
                if ids.warp <= role.last:
                    getattr(self, role.method)(p, s, ids)
            else:
                if (ids.warp >= role.first) & (ids.warp <= role.last):
                    getattr(self, role.method)(p, s, ids)

    def set_regs(self, method: str) -> None:
        role = next(r for r in self.ROLES if r.method == method)
        Prologue.role_regs(role.increase, role.regs)

    @abstractmethod
    def compute(self, p, s, ids: ThreadIds) -> Int32 | None:
        raise NotImplementedError

    @abstractmethod
    def load(self, p, s, ids: ThreadIds) -> None:
        raise NotImplementedError


class UpDownMapKernel(RoleKernel):
    """A role kernel whose Up and Down weight pages arrive through two TMA descriptors
    (:meth:`tensor_maps`).
    """

    up_map = down_map = None

    @cute.kernel
    def kernel(
        self, p, UpMap: GridConstant[TensorMap], DownMap: GridConstant[TensorMap]
    ):
        self.up_map, self.down_map = UpMap.get_ptr().toint(), DownMap.get_ptr().toint()
        ids, s = self.prologue(p)
        self.run(p, s, ids)


class TaggedMailbox:
    """TP8 tagged mailboxes (flag-in-data exchange): 24-bit payload words tagged with
    the launch's generation.
    """

    @staticmethod
    def tag8(epoch) -> Uint32:
        return Uint32(128) | ((epoch + 1) & 127)

    @staticmethod
    def pack24(w, tag_high) -> tuple:
        return (
            (w[0] & 16777215) | tag_high,
            (w[0] >> 24) | ((w[1] & 65535) << 8) | tag_high,
            (w[1] >> 16) | ((w[2] & 255) << 16) | tag_high,
            (w[2] >> 8) | tag_high,
        )

    @staticmethod
    def unpack24(v) -> tuple:
        return (
            (v[0] & 16777215) | (v[1] << 24),
            ((v[1] >> 8) & 65535) | (v[2] << 16),
            ((v[2] >> 16) & 255) | (v[3] << 8),
        )

    @staticmethod
    def peer_base(index, bases) -> Int64:
        return Memory.select_peer(index, bases)

    @staticmethod
    def push(address, words) -> None:
        Memory.st_volatile_v4(address, words)

    @staticmethod
    def stale(words, tag) -> Uint32:
        bits = Uint32(0)
        for w in _list(words):
            bits = bits | ((w >> 24) ^ tag)
        return bits

    @staticmethod
    @cute.jit
    def poll(packets: cute.Tensor, addresses, tag):
        addresses = _list(addresses)
        fresh = Uint32(1)
        while fresh != 0:
            for p in cutlass.range_constexpr(len(addresses)):
                (
                    packets[4 * p],
                    packets[4 * p + 1],
                    packets[4 * p + 2],
                    packets[4 * p + 3],
                ) = Memory.ld_volatile_v4(addresses[p])
            fresh = Uint32(0)
            for i in cutlass.range_constexpr(4 * len(addresses)):
                fresh = fresh | ((packets[i] >> 24) ^ tag)


class Pipeline:
    """mbarrier page rings: a producer warp streams pages into shared-memory slots for
    consumers.
    """

    class Ring:
        """One ring: ``slots`` full/empty mbarrier pairs, 8 bytes apart, at 32-bit
        shared addresses.
        """

        def __init__(self, full_addr, empty_addr, slots: int):
            self.full_addr, self.empty_addr, self.slots = full_addr, empty_addr, slots

        def __extract_mlir_values__(self):
            return [
                v
                for a in (self.full_addr, self.empty_addr)
                for v in Int32(a).__extract_mlir_values__()
            ]

        def __new_from_mlir_values__(self, values):
            return Pipeline.Ring(Int32(values[0]), Int32(values[1]), self.slots)

        def slot(self, sequence):
            return FastMath.cmod(sequence, self.slots)

        def parity(self, sequence):
            return FastMath.cmod(FastMath.cdiv(sequence, self.slots), 2)

        def full(self, slot):
            return self.full_addr + slot * 8

        def empty(self, slot):
            return self.empty_addr + slot * 8

        def wait_full(self, sequence, slot=None) -> None:
            AsyncCopy.mbar_wait(
                self.full(self.slot(sequence) if slot is None else slot),
                self.parity(sequence),
            )

        def release(self, slot) -> None:
            Memory.fence_proxy_async_shared()
            AsyncCopy.mbar_arrive(self.empty(slot))

        def acquire(self, sequence, nbytes: int, slot=None) -> None:
            slot = self.slot(sequence) if slot is None else slot
            AsyncCopy.mbar_wait(self.empty(slot), self.parity(sequence) ^ 1)
            AsyncCopy.mbar_expect_tx(self.full(slot), nbytes)


class Scan:
    """Warp and CTA scans over the 8 compute warps."""

    @staticmethod
    @cute.jit
    def up(v, lane):
        for s in cutlass.range_constexpr(5):
            carried = Sync.shfl_up(v, 1 << s)
            if lane >= (1 << s):
                v = v + carried
        return v

    @staticmethod
    def cta(value, warp_eq, warp, lane):
        inclusive = Scan.up(value, lane)
        Scan.last_lane(warp_eq, warp, lane, inclusive)
        Sync.bar_sync(8, 256)
        before, total = Scan.block_prefix(warp_eq, warp, Int32(0))
        Sync.bar_sync(8, 256)
        return inclusive, before, total

    @staticmethod
    @cute.jit
    def last_lane(arr, warp, lane, value):
        if lane == 31:
            arr[warp] = value

    @staticmethod
    @cute.jit
    def block_prefix(arr, warp, init):
        before = init
        total = Int32(0)
        for w in cutlass.range_constexpr(8):
            c = arr[w]
            if warp > w:
                before = before + c
            total = total + c
        return before, total


class Reduce:
    """Reductions and RMSNorm in a fixed association order, with ``.ftz`` float math."""

    @staticmethod
    def sum_squares(values, start=None) -> Float32:
        acc = Float32(0.0) if start is None else start
        for v in _list(values):
            acc = FastMath.fma(v, v, acc)
        return acc

    @staticmethod
    def warp_sum_down(x, steps: int = 5) -> Float32:
        for step in range(steps):
            x = FastMath.add(x, cute.arch.shuffle_sync_down(x, 16 >> step))
        return x

    @staticmethod
    def ordered_sum(read, n: int, first: int = 0) -> Float32:
        total = read(first)
        for i in range(first + 1, first + n):
            total = FastMath.add(total, read(i))
        return total

    @staticmethod
    def strided_sum(t, idx, stride: int, n: int = 8) -> Float32:
        total = t[idx]
        for w in range(1, n):
            total = FastMath.add(total, t[w * stride + idx])
        return total

    @staticmethod
    def rms_inverse(total, n: float = 6144.0, eps: float = 1e-05) -> Float32:
        return FastMath.rsqrt(FastMath.add(FastMath.div_rn(total, n), eps))

    @staticmethod
    def norm_bf16x8(values, inverse, gamma) -> list:
        y = [FastMath.mul(FastMath.mul(values[i], inverse), gamma[i]) for i in range(8)]
        return [Convert.f32x2_to_bf16x2(y[2 * i], y[2 * i + 1]) for i in range(4)]

    @staticmethod
    @cute.jit
    def fold(vals, lane, bit, tail: cutlass.Constexpr):
        vals = _list(vals)
        for _ in cutlass.range_constexpr(len(vals).bit_length() - 1):
            half = len(vals) // 2
            out = []
            for i in cutlass.range_constexpr(half):
                kept = vals[i]
                sent = vals[half + i]
                if (lane & bit) != 0:
                    kept = vals[half + i]
                    sent = vals[i]
                out.append(FastMath.add(kept, cute.arch.shuffle_sync_bfly(sent, bit)))
            vals = out
            bit = bit >> 1
        last = vals[0]
        for b in cutlass.range_constexpr(len(tail)):
            last = FastMath.add(last, cute.arch.shuffle_sync_bfly(last, tail[b]))
        return last

    @staticmethod
    def argmax_better(ov, oi, v, i):
        return FastMath.fcmp(">", ov, v) | (FastMath.fcmp("==", ov, v) & (oi < i))

    @staticmethod
    @cute.jit
    def argmax_bfly(v: Float32, i: Int32, offsets: cutlass.Constexpr):
        for off in cutlass.range_constexpr(len(offsets)):
            ov = cute.arch.shuffle_sync_bfly(v, offsets[off])
            oi = cute.arch.shuffle_sync_bfly(i, offsets[off])
            if Reduce.argmax_better(ov, oi, v, i):
                v = ov
                i = oi
        return v, i


class Mma:
    """FP8 (E4M3) weight-page MMA steps of the GEMV families."""

    @staticmethod
    def gate_up_steps(accum, rows, b_word, gate_scale, up_scale, tiles=None) -> None:
        count = tiles or 1
        page_acc = cute.make_rmem_tensor(8 * count, Float32)
        for k in range(8):
            b = b_word(k)
            for half in range(2):
                frag = Convert.a_frag_u32(rows[16 * half + k], rows[16 * half + 8 + k])
                for t in range(count):
                    acc_tile = cute.local_tile(page_acc, (4,), (2 * t + half,))
                    TensorCore.mma_acc(acc_tile, frag, b[t] if tiles else b, k == 0)
        for t in range(count):
            for i in range(4):
                accum[8 * t + i] = FastMath.fma(
                    page_acc[8 * t + i], gate_scale, accum[8 * t + i]
                )
                accum[8 * t + 4 + i] = FastMath.fma(
                    page_acc[8 * t + 4 + i], up_scale, accum[8 * t + 4 + i]
                )

    @staticmethod
    def step_u64(acc, row_a, row_b, k_half: int, b, zero: bool) -> list:
        return TensorCore.mma(acc, Convert.a_frag_u64(row_a, row_b, k_half), b, zero)

    @staticmethod
    def scale_accumulate(total, partial, scale) -> list:
        partial = _list(partial)
        return [FastMath.fma(partial[i], scale, total[i]) for i in range(len(partial))]


# Short module-level names for the hottest helpers, so call sites read ``rt.cdiv(...)``.
cdiv, cmod, add, mul, fma = (
    FastMath.cdiv,
    FastMath.cmod,
    FastMath.add,
    FastMath.mul,
    FastMath.fma,
)
max_noftz = FastMath.max_noftz
bar_sync, elect_sync, syncwarp = Sync.bar_sync, Sync.elect_sync, Sync.syncwarp
warp_leader = WarpLeader
globaltimer = Sync.globaltimer
ga, baddr, rmem_tensor = Memory.ga, Memory.baddr, Memory.rmem_tensor
mbar_wait, mbar_arrive = AsyncCopy.mbar_wait, AsyncCopy.mbar_arrive
mbar_expect_tx = AsyncCopy.mbar_expect_tx
bulk_g2s, bulk_g2s_hint = AsyncCopy.bulk_g2s, AsyncCopy.bulk_g2s_hint
f32x2_to_bf16x2, bf16x8_to_f32 = Convert.f32x2_to_bf16x2, Convert.bf16x8_to_f32


class SliceExchange:
    """TP8 all-reduce of each CTA's 48-column output slice of ``rows`` token rows (MoE,
    dense FFN).
    """

    COLUMNS, TOKEN, CHUNK = 48, 128, 16
    #: Tokens one warp carries (lanes / 8 packets per token).
    GROUP_TOKENS = 4

    def __init__(self, rows: int, seam: bool = False) -> None:
        self.rows, self.seam = rows, seam
        #: Token groups, and 8-row MMA tiles, of the capacity.
        self.groups = -(-rows // SliceExchange.GROUP_TOKENS)
        self.tiles = max(1, -(-rows // 8))
        self.SLICE = SliceExchange.TOKEN * rows
        self.PARITY = 128 * self.SLICE
        self.SLOT = 2 * self.PARITY
        self.OWN = 7 * self.SLOT

    @cute.jit
    def reduce_rows(self, scratch, local_o, LocalO, total, tid, warp, lane, bid):
        rows, columns = self.rows, SliceExchange.COLUMNS
        stride = columns * rows + 1
        if cmod(lane, 4) < max(1, min(4, rows // 2)):
            base = warp * stride + cdiv(lane, 4)
            token = cmod(lane, 4) * 2
            if cutlass.const_expr(rows <= 4):
                token = cmod(token, 4)
            for t in cutlass.range_constexpr(self.tiles):
                for m in cutlass.range_constexpr(3):
                    for e in cutlass.range_constexpr(0, 4, 1 if rows >= 4 else 2):
                        column = token + 1 if e % 2 else token
                        column = column + 8 * t if t else column
                        scratch[base + column * columns + m * 16 + 8 * (e // 2)] = (
                            total[t * 12 + m * 4 + e]
                        )
        bar_sync(8, 256)
        # 256 threads sum 48 x rows outputs: one round up to five rows, more above.
        for index_round in cutlass.range_constexpr(-(-columns * rows // 256)):
            index = tid + 256 * index_round if index_round else tid
            if index < columns * rows:
                value = Reduce.strided_sum(scratch, index, stride)
                local_o[index] = value.to(BFloat16)
                LocalO[
                    cdiv(index, columns) * HIDDEN + bid * columns + cmod(index, columns)
                ] = value.to(BFloat16)
        bar_sync(8, 256)

    @cute.jit
    def push(self, words, epoch, slice_, lane, base, slot_offset, group: int = 0):
        token = cdiv(lane, 8)
        if group:
            token = token + SliceExchange.GROUP_TOKENS * group
        chunk = cmod(lane, 8)
        tag_high = (128 | epoch + 1 & 127) << 24
        w = [Uint32(words[token * 24 + chunk * 3 + i]) for i in range(3)]
        Memory.st_volatile_v4(
            base
            + Int64(
                slot_offset
                + Int32(cmod(epoch, 2)) * self.PARITY
                + slice_ * self.SLICE
                + token * SliceExchange.TOKEN
                + chunk * SliceExchange.CHUNK
            ),
            TaggedMailbox.pack24(w, tag_high),
        )

    @cute.jit
    def push_peers(self, p, words, epoch, slice_, tid, warp, lane):
        pushing = tid < 224
        if pushing:
            destination = cutlass.select_(warp < p.peer_rank, warp, warp + 1)
            source_slot = cutlass.select_(
                destination > p.peer_rank, p.peer_rank, p.peer_rank - 1
            )
            peer = Memory.select_peer(destination, p.peer_bases)
            for group in cutlass.range_constexpr(self.groups):
                self.push(
                    words, epoch, slice_, lane, peer, source_slot * self.SLOT, group
                )

    @cute.jit
    def combine(
        self,
        p,
        src,
        lane,
        slice_,
        epoch,
        own_words=None,
        own_wait: cutlass.Constexpr = False,
        group=0,
        live=None,
    ):
        seam, columns = self.seam, SliceExchange.COLUMNS
        peer_rank = p.peer_rank
        slots = 8 if seam else 7
        token = cdiv(lane, 8)
        # ``group`` is a trace-time int (0: the first four tokens) or a per-warp value.
        if cutlass.const_expr(not isinstance(group, int) or group != 0):
            token = token + SliceExchange.GROUP_TOKENS * group
        chunk = cmod(lane, 8)
        active = Int32(token < live) if live is not None else None
        row = token * HIDDEN + slice_ * columns + chunk * 6
        residual = [src[row + item].to(Float32) for item in range(6)]
        gamma = None
        own = None
        if cutlass.const_expr(seam):
            gamma = [
                p.GammaNext[slice_ * columns + chunk * 6 + item] for item in range(6)
            ]
        else:
            own = [Uint32(own_words[token * 24 + chunk * 3 + i]) for i in range(3)]
        mailbox = Memory.select_peer(peer_rank, p.peer_bases)
        tag8 = 128 | epoch + 1 & 127
        first = (
            Int32(cmod(epoch, 2)) * self.PARITY
            + slice_ * self.SLICE
            + token * SliceExchange.TOKEN
            + chunk * SliceExchange.CHUNK
        )
        if cutlass.const_expr(own_wait):
            own_waiting = Int32(1)
            while own_waiting != 0:
                v = Memory.ld_volatile_v4(mailbox + Int64(first + self.OWN))
                own_stale = TaggedMailbox.stale([v[k] for k in range(4)], tag8)
                if cute.arch.vote_all_sync(own_stale == 0):
                    own_waiting = Int32(0)
                else:
                    backoff = globaltimer()
                    spinning = Int32(1)
                    while spinning != 0:
                        if globaltimer() - backoff >= 1000:
                            spinning = Int32(0)
        incoming = cute.make_rmem_tensor(3 * slots, Uint32)
        ld = cute.make_rmem_tensor(4 * slots, Uint32)
        polling = Int32(1)
        while polling != 0:
            for j in cutlass.range_constexpr(slots):
                v = Memory.ld_volatile_v4(mailbox + Int64(first + self.SLOT * j))
                for k in cutlass.range_constexpr(4):
                    ld[4 * j + k] = v[k]
            stale = TaggedMailbox.stale([ld[i] for i in range(4 * slots)], tag8)
            fresh = stale == 0
            if cutlass.const_expr(live is not None):
                fresh = (stale == 0) | (active == 0)
            if cutlass.const_expr(seam):
                fresh = cute.arch.vote_all_sync(fresh)
            if fresh:
                for j in cutlass.range_constexpr(slots):
                    for i, word in enumerate(
                        TaggedMailbox.unpack24([ld[4 * j + k] for k in range(4)])
                    ):
                        incoming[3 * j + i] = word
                polling = Int32(0)
        sums = [Float32(0.0)] * 6
        words = cute.make_rmem_tensor(3, Uint32)
        if cutlass.const_expr(seam):
            own = [incoming[21 + i] for i in range(3)]
        for r in cutlass.range_constexpr(8):
            if r == peer_rank:
                for i in cutlass.range_constexpr(3):
                    words[i] = own[i]  # type: ignore[index]
            elif peer_rank > r:
                for i in cutlass.range_constexpr(3):
                    words[i] = incoming[min(3 * r, 18) + i]
            else:
                for i in cutlass.range_constexpr(3):
                    words[i] = incoming[max(3 * r - 3, 0) + i]
            vals = bf16x8_to_f32([words[i] for i in range(3)])
            sums = [add(sums[i], vals[i]) for i in range(6)]
        reduced = bf16x8_to_f32(
            [f32x2_to_bf16x2(sums[2 * i], sums[2 * i + 1]) for i in range(3)]
        )
        for item in cutlass.range_constexpr(6):
            x = add(reduced[item], residual[item]).to(BFloat16)
            if cutlass.const_expr(live is None):  # noqa: SIM114  constexpr and runtime conditions stay separate
                p.Final[row + item] = x
                if cutlass.const_expr(seam):
                    p.XG[row + item] = mul(x.to(Float32), gamma[item]).to(BFloat16)  # type: ignore[index]
            elif active != 0:
                p.Final[row + item] = x
                if cutlass.const_expr(seam):
                    p.XG[row + item] = mul(x.to(Float32), gamma[item]).to(BFloat16)  # type: ignore[index]


class OExchange:
    """TP8 O exchange of a CTA's 48-column slice of up to four rows (MLA worker, MTP
    preprocess).
    """

    @staticmethod
    def word(token, source, parity, slot, packet):
        return token * 57344 + source * 8192 + parity * 4096 + slot * 32 + packet * 4

    @staticmethod
    def push(
        local_o_words, bases, peer_rank, epoch, token, packed_peer, chunk, slot
    ) -> None:
        destination = FastMath.select_i32(
            packed_peer < peer_rank, packed_peer, packed_peer + 1
        )
        source_slot = FastMath.select_i32(
            destination > peer_rank, peer_rank, peer_rank - 1
        )
        payload = [local_o_words[token * 24 + chunk * 3 + i] for i in range(3)]
        words = TaggedMailbox.pack24(payload, TaggedMailbox.tag8(epoch) << 24)
        first = OExchange.word(token, source_slot, cmod(epoch + 1, 2), slot, chunk)
        TaggedMailbox.push(
            TaggedMailbox.peer_base(destination, bases) + Int64(first * 4), words
        )

    @staticmethod
    def reduce(
        local_o_words, local_o_addr, peer_rank, query_, lane, received: int = 128
    ) -> None:
        sums = [Float32(0.0)] * 4
        loaded = []
        for r in range(8):
            own = Int32(peer_rank == r)
            above = Int32(peer_rank < r)
            peer_word = received + query_ * 168 + (r - above) * 24 + lane * 2
            word = own * (query_ * 24 + lane * 2) + (1 - own) * peer_word
            loaded.append((local_o_words[word], local_o_words[word + 1]))
        for r in range(8):
            f = list(Convert.bf16x2_to_f32x2(loaded[r][0])) + list(
                Convert.bf16x2_to_f32x2(loaded[r][1])
            )
            sums = [add(sums[i], f[i]) for i in range(4)]
        out_at = local_o_addr + (query_ * 24 + lane * 2) * 4
        Memory.st_shared_v2(
            out_at, f32x2_to_bf16x2(sums[0], sums[1]), f32x2_to_bf16x2(sums[2], sums[3])
        )

    @staticmethod
    @cute.jit
    def receive(
        received_words, base, parity, tag, slot, lane_0, items, n: cutlass.Constexpr
    ):
        incoming = rmem_tensor(3 * len(items), Uint32)
        done = Int32(0)
        while done == 0:
            vs = [
                Memory.ld_volatile_v4(
                    base
                    + Int64(
                        cdiv(it, 56) * 57344
                        + cmod(cdiv(it, 8), 7) * 8192
                        + parity * 4096
                        + slot * 32
                        + cmod(it, 8) * 4
                    )
                    * 4
                )
                for it in items
            ]
            stale = Uint32(0)
            for w in cutlass.range_constexpr(4 * len(items)):
                stale = stale | ((vs[w // 4][w % 4] >> 24) ^ tag)
            if stale == 0:
                for h in cutlass.range_constexpr(len(items)):
                    vh = vs[h]
                    incoming[3 * h] = (vh[0] & 16777215) | (vh[1] << 24)
                    incoming[3 * h + 1] = ((vh[1] >> 8) & 65535) | (vh[2] << 16)
                    incoming[3 * h + 2] = ((vh[2] >> 16) & 255) | (vh[3] << 8)
                done = Int32(1)
        if cutlass.const_expr(len(items) > 1):
            syncwarp()
        for k in cutlass.range_constexpr(len(items)):
            it = lane_0 + 128 * k
            if it < n:
                word = cdiv(it, 8) * 24 + cmod(it, 8) * 3
                for i in cutlass.range_constexpr(3):
                    received_words[word + i] = incoming[3 * k + i]


class Poll:
    """Spin-polls of generation-tagged u64 mailbox records: the tag is the hi 32 bits,
    or (``tag40``) bits 40..
    """

    @staticmethod
    def tagged(word, want, tag40: bool) -> Int32:
        if tag40:
            return Int32(Uint32(word >> 40) == (want & Uint32(16777215)))
        return Int32(Uint32(word >> 32) == want)

    @staticmethod
    def bin_record(generation, pair):
        return (
            Uint64(generation & 16777215) << 40
            | Uint64(pair[1]) << 20
            | Uint64(pair[0])
        )

    @staticmethod
    @cute.jit
    def word(addr, want, tag40: cutlass.Constexpr, memory: cutlass.Constexpr = False):
        polled = Uint64(0)
        waiting = Int32(1)
        while waiting != 0:
            polled = Memory.ld_volatile_u64(addr, memory=memory)
            if Poll.tagged(polled, want, tag40) != 0:
                waiting = Int32(0)
        return polled

    @staticmethod
    def wait(addrs, want, live=(), tag40=False, backoff=False):
        words = cute.make_rmem_tensor(len(addrs), Uint64)
        tags = tag40 if isinstance(tag40, tuple) else (tag40,) * len(addrs)
        live = tuple(live) or (None,) * len(addrs)
        gated = sum(1 << i for i, v in enumerate(live) if v is not None)
        tagged40 = sum(1 << i for i, t in enumerate(tags) if t)
        preds = tuple(Int32(1 if v is None else v) for v in live)
        Poll.spin(words, tuple(addrs), want, preds, gated, tagged40, backoff)
        return words

    @staticmethod
    @cute.jit
    def spin(
        words,
        addrs,
        want,
        preds,
        gated: cutlass.Constexpr,
        tag40: cutlass.Constexpr,
        backoff: cutlass.Constexpr,
    ):
        waiting = Int32(1)
        while waiting != 0:
            ready = Int32(1)
            for i in cutlass.range_constexpr(len(addrs)):
                if cutlass.const_expr((gated >> i) & 1 == 0):
                    polled = Memory.ld_volatile_u64(addrs[i])
                    words[i] = polled
                    ready = ready & Poll.tagged(polled, want, (tag40 >> i) & 1 == 1)
                else:
                    words[i] = Uint64(0)
                    if preds[i] != 0:
                        live_word = Memory.ld_volatile_u64(addrs[i])
                        words[i] = live_word
                        ready = ready & Poll.tagged(
                            live_word, want, (tag40 >> i) & 1 == 1
                        )
            if cute.arch.vote_all_sync(ready != 0):
                waiting = Int32(0)
            else:
                if cutlass.const_expr(backoff):
                    Poll.backoff()

    @staticmethod
    @cute.jit
    def backoff():
        backoff_start = globaltimer()
        spinning = Int32(1)
        while spinning != 0:
            backoff_now = globaltimer()
            if cute.arch.vote_all_sync(backoff_now - backoff_start >= 256):
                spinning = Int32(0)


class Selection:
    """Threshold search over a CTA's cached (id, key) u32 pairs (worker refine CTAs,
    selector final CTAs).
    """

    @staticmethod
    def add(view, addr, index, value) -> None:
        if addr is None:
            cute.arch.atomic_add(view.iterator + index, Uint32(value))
        else:
            Memory.atom_add_shared(addr + index * 4, value)

    @staticmethod
    @cute.jit
    def threshold(
        cache,
        hist,
        hist_addr,
        control,
        warps,
        count,
        largest,
        need,
        ties,
        tid,
        warp,
        lane,
    ):
        lean_bits = Int32(20)
        lean_prefix = Uint32(largest >> 20)
        for lean_pass in cutlass.range(0, 2, 1, unroll=1):
            pass_shift = Uint32(lean_bits - 10)
            if lean_pass != 0:
                for part in cutlass.range_constexpr(4):
                    hist[part * 256 + tid] = Uint32(0)
                bar_sync(8, 256)
                for index in cutlass.range(tid, count, 256):
                    key = Uint32(cache[index * 2 + 1])
                    if key >> Uint32(lean_bits) == lean_prefix:
                        Selection.add(
                            hist, hist_addr, Int32(key >> pass_shift & 1023), 1
                        )
            bar_sync(8, 256)
            top = 1023 - tid * 4
            counts = []
            own = Int32(0)
            for item in cutlass.range_constexpr(4):
                counts.append(Int32(hist[top - item]))
                own = own + counts[item]
            inclusive, before, _ = Scan.cta(own, warps, warp, lane)
            if (need > before + inclusive - own) & (
                need <= before + inclusive - own + own
            ):
                running = before + inclusive - own
                found = Int32(0)
                for item in cutlass.range_constexpr(4):
                    running = running + counts[item]
                    if (found == 0) & (running >= need):
                        control[0] = top - item
                        control[1] = running - counts[item]
                        control[2] = counts[item]
                        found = Int32(1)
            bar_sync(8, 256)
            lean_digit = control[0]
            lean_above = control[1]
            ties = control[2]
            bar_sync(8, 256)
            lean_prefix = Uint32(Uint32(lean_prefix << 10) | Uint32(lean_digit))
            need = need - lean_above
            lean_bits = lean_bits - 10
        return lean_prefix, need, ties

    @staticmethod
    @cute.jit
    def tie_limit(
        cache,
        blocks,
        blocks_addr,
        bits,
        bits_addr,
        control,
        warps,
        count,
        n_blocks,
        key,
        need,
        tid,
        warp,
        lane,
    ):
        for block in cutlass.range(tid, n_blocks, 256):
            blocks[block] = Uint32(0)
        if tid < 4:
            bits[tid] = Uint32(0)
        bar_sync(8, 256)
        for index in cutlass.range(tid, count, 256):
            if Uint32(cache[index * 2 + 1]) == key:
                Selection.add(
                    blocks, blocks_addr, Int32(cdiv(Uint32(cache[index * 2]), 128)), 1
                )
        bar_sync(8, 256)
        span = cdiv(n_blocks + 256 - 1, 256)
        begin = n_blocks
        if (tid * span) < n_blocks:
            begin = tid * span
        end = n_blocks
        if (begin + span) < n_blocks:
            end = begin + span
        own = Int32(0)
        for block in cutlass.range(begin, end, 1):
            own = own + Int32(blocks[block])
        inclusive, before, _ = Scan.cta(own, warps, warp, lane)
        if (need > before + inclusive - own) & (need <= before + inclusive - own + own):
            running = before + inclusive - own
            found = Int32(0)
            for block in cutlass.range(begin, end, 1):
                block_count = Int32(blocks[block])
                if (found == 0) & (need <= running + block_count):
                    control[0] = block
                    control[1] = need - running
                    found = Int32(1)
                running = running + block_count
        bar_sync(8, 256)
        target = control[0]
        rank = control[1]
        for index in cutlass.range(tid, count, 256):
            tie_id = Int32(cache[index * 2])
            if (Uint32(cache[index * 2 + 1]) == key) & (cdiv(tie_id, 128) == target):
                bit = cmod(tie_id, 128)
                Selection.add(
                    bits, bits_addr, cdiv(bit, 32), Uint32(1) << Uint32(cmod(bit, 32))
                )
        bar_sync(8, 256)
        position = Int32(0)
        seen = Int32(0)
        for word in cutlass.range_constexpr(4):
            mask = Uint32(bits[word])
            for bit in cutlass.range(0, 32, 1, unroll=1):
                if (mask >> Uint32(bit) & 1) != 0:
                    seen = seen + 1
                    if seen == rank:
                        position = word * 32 + bit
        bar_sync(8, 256)
        return Uint32(target * 128 + position)


class LocalityProbe:
    """The SM recorder of a ``grid``-CTA, one-CTA-per-SM launch."""

    #: Dynamic shared memory that keeps one CTA per SM.
    SMEM_BYTES = 160 * 1024
    THREADS = 32

    def __init__(self, grid: int) -> None:
        self.grid = grid

    def _launch(self, kernel, stream) -> None:
        kernel.launch(
            grid=(self.grid, 1, 1),
            block=(self.THREADS, 1, 1),
            smem=self.SMEM_BYTES,
            stream=stream,
        )

    @cute.kernel
    def record_sm_kernel(self, sms: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        out = cute.make_tensor(sms, cute.make_layout(self.grid))
        if tid == 0:
            out[bid] = Uint32(Sync.smid())

    @cute.jit
    def record_sm(self, sms: cute.Pointer, stream):
        self._launch(self.record_sm_kernel(sms), stream)


class Umma:
    """``tcgen05.mma kind::f8f6f4`` of E4M3 operands: ``D[128, n] (+)= A[128, 128 K] x
    B[n, 128 K]^T`` per K block, with ``D`` FP32 in tensor memory (TMEM).
    """

    DESCRIPTOR_HIGH = 0x40004040

    @staticmethod
    def instruction_descriptor(n: int) -> int:
        assert n % 16 == 0 and 16 <= n <= 256, n
        return 16 | ((n >> 3) << 17) | (8 << 24)

    @staticmethod
    def operand_low(smem_addr):
        return (smem_addr >> 4) & 0x3FFF

    @staticmethod
    def mma_k128(a_low, b_low, tmem, n: int, accumulate) -> None:
        step = (
            "add.u32 a_lo, a_lo, 2; add.u32 b_lo, b_lo, 2; mov.b64 da, {a_lo, adhi}; "
            "mov.b64 db, {b_lo, bdhi};"
            " tcgen05.mma.cta_group::1.kind::f8f6f4 [$2], da, db, id, p1; "
        )
        code = (
            "{ .reg .pred p0, p1; .reg .b32 adhi, bdhi, a_lo, b_lo, id;"
            " .reg .b64 da, db;"
            " setp.ne.b32 p0, $3, 0; setp.ne.b32 p1, 1, 0; "
            f"mov.b32 adhi, {Umma.DESCRIPTOR_HIGH:#x};"
            f" mov.b32 bdhi, {Umma.DESCRIPTOR_HIGH:#x};"
            f" mov.b32 id, {Umma.instruction_descriptor(n)};"
            " mov.b32 a_lo, $0; mov.b32 b_lo, $1;"
            " mov.b64 da, {a_lo, adhi}; mov.b64 db, {b_lo, bdhi};"
            " tcgen05.mma.cta_group::1.kind::f8f6f4 [$2], da, db, id, p0; "
            + step * 3
            + "}"
        )
        Core.vol(
            code,
            (),
            Int32(a_low),
            Int32(b_low),
            Int32(tmem),
            Int32(accumulate),
            imm=True,
        )

    @staticmethod
    def tmem_ld(taddr, count: int) -> tuple:
        assert count in (8, 16, 32, 64), count
        regs = ", ".join(f"${i}" for i in range(count))
        code = f"tcgen05.ld.sync.aligned.32x32b.x{count}.b32 {{{regs}}}, [${count}];"
        return Core.vol(code, (Float32,) * count, Int32(taddr), imm=True)

    @staticmethod
    def tmem_ld_columns(taddr, count: int) -> list:
        values, done = [], 0
        for width in (64, 32, 16):
            while count - done >= width:
                values += list(Umma.tmem_ld(taddr + done, width))
                done += width
        assert done == count, count
        return values

    @staticmethod
    def wait_ld() -> None:
        cute.arch.fence_view_async_tmem_load()

    @staticmethod
    @cute.jit
    def alloc(base, hold_offset: int, columns: int, warp) -> Int32:
        hold = Memory.smem_ptr(base, hold_offset, Int32)
        if warp == 0:
            cute.arch.alloc_tmem(columns, hold)
            cute.arch.sync_warp()
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        TensorCore.fence_after_thread_sync()
        return cute.make_tensor(hold, cute.make_layout(1))[0]

    @staticmethod
    @cute.jit
    def free(base, hold_offset: int, columns: int, warp) -> None:
        if warp == 0:
            TensorCore.fence_after_thread_sync()
            TensorCore.tmem_dealloc(
                Memory.ld_volatile_shared_u32(base + hold_offset), columns
            )
