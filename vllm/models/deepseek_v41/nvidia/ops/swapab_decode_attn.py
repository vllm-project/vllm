# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from SGLang's swap-AB DSv4 decode attention (Apache-2.0):
# https://github.com/sgl-project/sglang/blob/91b3692a4c39453694f53ff54b67186ca4a7f34c/python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100.py
# https://github.com/sgl-project/sglang/blob/91b3692a4c39453694f53ff54b67186ca4a7f34c/python/sglang/kernels/ops/attention/dsv4/decode_attention_sm100_gluon.py
"""Swap-AB decode attention for DeepSeek V4.1's small decode steps (SM100).

FlashMLA's FP8 decode kernel pads the query heads to 64, so with 32 or 16 live
heads per rank half or three quarters of its MMA work is padding. This kernel
puts the KV slots on the tcgen05 MMA's M side and the live heads on N instead.
A token's SWA and compressed slot lists are split into 64- and 128-slot tiles,
one CTA each, and the tiles of a token form a thread-block cluster that merges
their partial results over DSMEM in a fixed order. It reads the mega kernel's
decode inputs and writes its MXFP8 output for the live heads.
"""

import math
import struct
from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
    Uint64,
    target_version,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.cute_utils import _tcgen05, cvt, recast_val
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.models.deepseek_v41.common.ops.fused_layout import WV_GROUP_SIZE
from vllm.models.deepseek_v41.nvidia.ops.fused_wo_a import _ue8m0_rp
from vllm.utils.math_utils import cdiv

_HEAD_DIM = 512
_ROPE_DIM = 64
_ROPE_START = _HEAD_DIM - _ROPE_DIM
_TILE = 128  # KV slots per compressed-list CTA = UMMA M
_SWA_ROWS = 64  # KV slots per SWA-list CTA (fp8 records are ~1.8x the NVFP4 bytes)
_THREADS = 512
# V4.1 KV records, each a token's data followed by its own scales.
_FP8_RECORD = 528  # 512 e4m3 + 16 ue8m0 (one per 32 dims)
_FP4_RECORD = 288  # 256 B e2m1 + 32 e4m3 (one per 16 dims)
_MAX_CLUSTER = 8  # portable thread-block cluster size
_LOG2E = 1.4426950408889634


def _asm(result, operands, text, constraints, *, side_effects=False, loc=None, ip=None):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, ty, loc, ip):
    return ty(x).ir_value(loc=loc, ip=ip)


def _fields(out, n, mlir_ty, wrap, loc, ip):
    return tuple(
        wrap(llvm.extractvalue(mlir_ty, out, [i], loc=loc, ip=ip)) for i in range(n)
    )


@dsl_user_op
def _ldg_s32(addr, pred, fill, *, loc=None, ip=None) -> Int32:
    """4-byte global load; ``fill`` where ``pred`` is 0 (no access then)."""
    out = _asm(
        T.i32(),
        [_v(addr, Int64, loc, ip), _v(pred, Int32, loc, ip), _v(fill, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, $3;\n\t"
        "@p ld.global.b32 $0, [$1];\n\t}",
        "=r,l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return Int32(out)


@dsl_user_op
def _ldg_v4(addr, pred, *, loc=None, ip=None):
    """16-byte global load without L1 allocation; zeros where ``pred`` is 0."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(addr, Int64, loc, ip), _v(pred, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t"
        "mov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "@p ld.global.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def _ldg_f32x4(addr, pred, fill: float, *, loc=None, ip=None):
    """16-byte fp32 global load; ``fill`` in every lane where ``pred`` is 0."""
    bits = f"0f{struct.unpack('>I', struct.pack('>f', fill))[0]:08X}"
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_v(addr, Int64, loc, ip), _v(pred, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t"
        f"mov.f32 $0, {bits};\n\tmov.f32 $1, {bits};\n\t"
        f"mov.f32 $2, {bits};\n\tmov.f32 $3, {bits};\n\t"
        "@p ld.global.v4.f32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=f,=f,=f,=f,l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def _stg_b32(addr, value, pred, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            _v(addr, Int64, loc, ip),
            _v(value, Uint32, loc, ip),
            _v(pred, Int32, loc, ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t"
        "@p st.global.b32 [$0], $1;\n\t}",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_b8(addr, value, pred, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            _v(addr, Int64, loc, ip),
            _v(value, Uint32, loc, ip),
            _v(pred, Int32, loc, ip),
        ],
        "{\n\t.reg .pred p;\n\t.reg .b16 t;\n\tsetp.ne.b32 p, $2, 0;\n\t"
        "cvt.u16.u32 t, $1;\n\t@p st.global.b8 [$0], t;\n\t}",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, Int32, loc, ip)] + [_v(w, Uint32, loc, ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4p(addr, w0, w1, w2, w3, pred, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, Int32, loc, ip)]
        + [_v(w, Uint32, loc, ip) for w in (w0, w1, w2, w3)]
        + [_v(pred, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t"
        "@p st.shared.v4.b32 [$0], {$1, $2, $3, $4};\n\t}",
        "r,r,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4f(addr, a0, a1, a2, a3, pred, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, Int32, loc, ip)]
        + [_v(a, Float32, loc, ip) for a in (a0, a1, a2, a3)]
        + [_v(pred, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t"
        "@p st.shared.v4.f32 [$0], {$1, $2, $3, $4};\n\t}",
        "r,f,f,f,f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4f(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_v(addr, Int32, loc, ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def _lds_v2f(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_v(addr, Int32, loc, ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 2, T.f32(), Float32, loc, ip)


@dsl_user_op
def _sts_v2(addr, w0, w1, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, Int32, loc, ip), _v(w0, Uint32, loc, ip), _v(w1, Uint32, loc, ip)],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32(addr, value, pred=1, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [
            _v(addr, Int32, loc, ip),
            _v(value, Float32, loc, ip),
            _v(pred, Int32, loc, ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t"
        "@p st.shared.f32 [$0], $1;\n\t}",
        "r,f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_s32(addr, value, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, Int32, loc, ip), _v(value, Int32, loc, ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(addr, Int32, loc, ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_s32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(addr, Int32, loc, ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_peer_f32(addr, rank, *, loc=None, ip=None) -> Float32:
    """Load an fp32 from cluster peer ``rank``'s SMEM at this CTA's ``addr``."""
    return Float32(
        _asm(
            T.f32(),
            [_v(addr, Int32, loc, ip), _v(rank, Int32, loc, ip)],
            "{\n\t.reg .b32 ra;\n\tmapa.shared::cluster.u32 ra, $1, $2;\n\t"
            "ld.shared::cluster.f32 $0, [ra];\n\t}",
            "=f,r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_peer_f32x4(addr, rank, pred, *, loc=None, ip=None):
    """16-byte DSMEM load from peer ``rank``; zeros where ``pred`` is 0."""
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_v(addr, Int32, loc, ip), _v(rank, Int32, loc, ip), _v(pred, Int32, loc, ip)],
        "{\n\t.reg .pred p;\n\t.reg .b32 ra;\n\tsetp.ne.b32 p, $6, 0;\n\t"
        "mapa.shared::cluster.u32 ra, $4, $5;\n\t"
        "mov.f32 $0, 0f00000000;\n\tmov.f32 $1, 0f00000000;\n\t"
        "mov.f32 $2, 0f00000000;\n\tmov.f32 $3, 0f00000000;\n\t"
        "@p ld.shared::cluster.v4.f32 {$0, $1, $2, $3}, [ra];\n\t}",
        "=f,=f,=f,=f,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(a, Float32, loc, ip), _v(b, Float32, loc, ip), _v(c, Float32, loc, ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(a, Float32, loc, ip), _v(b, Float32, loc, ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(x, Float32, loc, ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fp8x8_to_bf16x8_ue8m0(w0, w1, exponent, *, loc=None, ip=None):
    """Eight e4m3 bytes times 2^(exponent - 127) -> four bf16x2 words (exact)."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            _v(w0, Uint32, loc, ip),
            _v(w1, Uint32, loc, ip),
            _v(exponent, Uint32, loc, ip),
        ],
        "{\n\t.reg .b16 a0, a1, a2, a3, sf;\n\t.reg .b32 e2;\n\t"
        "mov.b32 {a0, a1}, $4;\n\tmov.b32 {a2, a3}, $5;\n\t"
        "shl.b32 e2, $6, 8;\n\tor.b32 e2, e2, $6;\n\tcvt.u16.u32 sf, e2;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 $0, a0, sf;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 $1, a1, sf;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 $2, a2, sf;\n\t"
        "cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2 $3, a3, sf;\n\t}",
        "=r,=r,=r,=r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def _e4m3_to_bf16x2(byte, *, loc=None, ip=None) -> Uint32:
    """One e4m3 byte -> the same bf16 value in both halves of a word (exact)."""
    b = Uint32(byte)
    return Uint32(
        _asm(
            T.i32(),
            [_v(b | (b << 8), Uint32, loc, ip)],
            "{\n\t.reg .b16 x;\n\tcvt.u16.u32 x, $1;\n\t"
            "cvt.rn.bf16x2.e4m3x2 $0, x;\n\t}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fp4x8_to_bf16x8(w, scale_bf16x2, *, loc=None, ip=None):
    """Eight e2m1 (even element in the low nibble) times a bf16 e4m3 scale.

    e2m1 -> bf16 is exact and e2m1 * e4m3 has at most 6 significant bits, so the
    bf16 product is exact: ``bf16(E2M1[code] * fp32(scale))``.
    """
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(w, Uint32, loc, ip), _v(scale_bf16x2, Uint32, loc, ip)],
        "{\n\t.reg .b8 b0, b1, b2, b3;\n\t.reg .b32 h0, h1, h2, h3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, $4;\n\t"
        "cvt.rn.bf16x2.e2m1x2 h0, b0;\n\tcvt.rn.bf16x2.e2m1x2 h1, b1;\n\t"
        "cvt.rn.bf16x2.e2m1x2 h2, b2;\n\tcvt.rn.bf16x2.e2m1x2 h3, b3;\n\t"
        "mul.rn.bf16x2 $0, h0, $5;\n\tmul.rn.bf16x2 $1, h1, $5;\n\t"
        "mul.rn.bf16x2 $2, h2, $5;\n\tmul.rn.bf16x2 $3, h3, $5;\n\t}",
        "=r,=r,=r,=r,r,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


def _make_sdesc_64B_swizzle(SBO: int):
    """``_tcgen05.make_sdesc_128B_swizzle`` for the 16-head P tile's 64 B rows."""
    return Uint64((SBO >> 4 << 32) | (1 << 46) | (4 << 61))


def _i32(cond):
    return Int32(cutlass.select_(cond, Int32(1), Int32(0)))


def _sel(cond, a, b, ty):
    return ty(cutlass.select_(cond, ty(a), ty(b)))


class SwapABDecodeKernel(VllmCuTeDSLJitKernel["SwapABDecodeKernel.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        heads: int  # live heads: 16 or 32
        swa_width: int  # columns of the SWA slot list
        extra_width: int  # columns of the compressed slot list
        extra_record: int  # compressed record bytes: 528 (fp8) or 288 (NVFP4)
        swa_page: int  # slots per SWA cache page
        extra_page: int  # slots per compressed cache page

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        H = compile_key.heads
        assert H in (16, 32), "live heads must be 16 or 32"
        NK, NE = compile_key.swa_width, compile_key.extra_width
        assert compile_key.extra_record in (_FP8_RECORD, _FP4_RECORD), (
            "compressed records must be V4.1 fp8 or NVFP4"
        )
        EXTRA_FP4 = compile_key.extra_record == _FP4_RECORD
        KP, EP = compile_key.swa_page, compile_key.extra_page
        for page in (KP, EP):
            assert page > 0 and page & (page - 1) == 0, "page sizes must be powers of 2"
        KT = cdiv(NK, _SWA_ROWS)
        ET = cdiv(NE, _TILE)
        C = KT + ET
        assert 1 <= C <= _MAX_CLUSTER, f"{C} KV tiles per token > {_MAX_CLUSTER}"

        NC = H // 4  # score / P columns per thread in the softmax
        S_COL = 8 * H  # TMEM: O hi|lo for 4 M-blocks first, then S
        TMEM_COLS = 256 if S_COL + H <= 256 else 512
        P_ROW = 4 * H  # bytes per P row: H bf16 hi + H bf16 lo
        Q_PIECES = H * _HEAD_DIM // 8  # 16-byte Q pieces
        QP = Q_PIECES // _THREADS
        assert QP * _THREADS == Q_PIECES
        U = (_HEAD_DIM // 32) * H  # (head, 32-dim chunk) output units
        UPC = cdiv(U, C)
        NPASS = cdiv(UPC, _THREADS // 8)
        STATS = C * 2 * H
        UNIT_BYTES = 32 * 4  # one (head, 32-dim chunk) unit of the fp32 partial

        IDESC_S = _tcgen05.make_bf16_idesc(_TILE, H)
        IDESC_O = _tcgen05.make_bf16_idesc(
            _TILE, 2 * H, transpose_A=True, transpose_B=True
        )
        DESC_KMAJ = _tcgen05.make_sdesc_128B_swizzle(0)
        DESC_KV_MN = _tcgen05.make_sdesc_128B_swizzle(16384)
        DESC_P = (
            _tcgen05.make_sdesc_128B_swizzle(0)
            if H == 32
            else _make_sdesc_64B_swizzle(8 * P_ROW)
        )
        KB_Q = H * 128  # bytes per 64-dim k-block of Q

        def stage_inputs(
            fmt_fp4: bool,
            cache,
            idx,
            lens,
            q,
            positions,
            cos_sin,
            b,
            tid,
            row0,
            width: int,
            page: int,
            a_kv,
            a_q,
            a_valid,
            a_cs,
            rows: int,
        ):
            """Issue every global load, then Q RoPE -> SMEM and KV dequant -> SMEM.

            ``rows`` (128 or 64) slots of the list start at ``row0``; with 64 the
            tile's upper half is left unwritten and flagged invalid. Returns 1 if
            the tile has any candidate slot (CTA-uniform).
            """
            page_shift = page.bit_length() - 1
            row_bytes = 256 if fmt_fp4 else 512
            sf_bytes = 32 if fmt_fp4 else 16
            record = row_bytes + sf_bytes
            # Gather mapping: TPR threads per row; lane % RPW = row within the warp,
            # lane // RPW = part j, which owns 16-byte chunks j, j + TPR, ... .
            TPR = _THREADS // rows
            RPW = 32 // TPR
            warp = tid // 32
            lane = tid % 32
            t = warp * RPW + lane % RPW
            j = lane // RPW

            # Slot ids do not wait for the length (one dependent round trip less).
            rowpos = row0 + t
            id_addr = (
                idx.iterator.toint()
                + (Int64(b) * Int64(idx.stride[0]) + Int64(rowpos)) * 4
            )
            sid = _ldg_s32(id_addr, _i32(rowpos < width), Int32(-1))
            limit = cutlass.min(lens[b], Int32(width))
            pos = positions[b]
            live = row0 < limit
            in_range = rowpos < limit

            # Q: 16-byte pieces of the chunk-interleaved layout
            # q[b, (d // 16) * HP * 16 + h * 16 + d % 16]; only the last piece of a
            # thread can reach the RoPE dims (static, see QP).
            hp = Int64(q.shape[1])
            q_base = q.iterator.toint() + Int64(b) * Int64(q.stride[0]) * 2
            qw = []
            qmeta = []
            for k in range(QP):
                i = tid + _THREADS * k
                c16 = i // (2 * H)
                rem = i % (2 * H)
                h = rem // 2
                half = rem % 2
                g = q_base + (Int64(c16) * hp * 16 + Int64(h * 16 + half * 8)) * 2
                qw.append(_ldg_v4(g, Int32(1)))
                qmeta.append((c16 * 16 + half * 8, h))
            # KV gather (predicated: invalid rows read nothing and stay zero).
            slots = Int64(cache.shape[0]) * page
            valid = in_range & (sid >= 0) & (Int64(sid) < slots)
            vp = _i32(valid)
            blk = sid >> page_shift
            slot = sid & (page - 1)
            base = cache.iterator.toint() + Int64(blk) * Int64(cache.stride[0])
            data = base + Int64(slot * record)
            sfa = data + row_bytes
            # 16-byte chunk c = TPR k + j of the row: a warp-wide load covers
            # TPR * 16 contiguous bytes of each of its rows.
            n_chunks = (row_bytes // 16) // TPR
            kv = []
            for k in range(n_chunks):
                kv.append(_ldg_v4(data + (Int64(TPR * k) + Int64(j)) * 16, vp))
            sfw = _ldg_v4(sfa, vp)
            if fmt_fp4:
                sfw = sfw + _ldg_v4(sfa + 16, vp)

            # cos | sin at the query position (after the KV loads: pos may arrive later
            # than the slot ids). The whole row also goes to SMEM for the epilogue.
            d_last = qmeta[QP - 1][0]
            rope = d_last >= _ROPE_START
            f0 = (d_last - _ROPE_START) // 2
            cs_row = (
                cos_sin.iterator.toint() + Int64(pos) * Int64(cos_sin.stride[0]) * 4
            )
            cosv = _ldg_f32x4(cs_row + Int64(f0) * 4, _i32(rope), 1.0)
            sinv = _ldg_f32x4(cs_row + Int64(f0 + _ROPE_DIM // 2) * 4, _i32(rope), 0.0)
            cs_lane = tid % 16
            cs_pred = _i32(tid < 16)
            csw = _ldg_v4(cs_row + Int64(cs_lane) * 16, cs_pred)

            # Q RoPE (fp32, GPT-J pairs) -> bf16 -> SMEM [H, 512] K-major SW128.
            for k in range(QP):
                d0, h = qmeta[k]
                words = list(qw[k])
                if k == QP - 1:
                    for w in range(4):
                        qe, qo = cvt.bf16x2_to_fp32x2(words[w])
                        c, s = cosv[w], sinv[w]
                        re = _fma(qe, c, -_mul(qo, s))
                        ro = _fma(qo, c, _mul(qe, s))
                        words[w] = _sel(
                            rope, cvt.fp32x2_to_bf16x2(re, ro), words[w], Uint32
                        )
                kb = d0 // 64
                phys = ((d0 % 64) // 8) ^ (h % 8)
                addr = a_q + kb * KB_Q + (h // 8) * 1024 + (h % 8) * 128 + phys * 16
                _sts_v4(addr, words[0], words[1], words[2], words[3])

            # KV dequant -> SMEM [128, 512] bf16, K-major SW128 (8 k-blocks of
            # 128 rows x 128 B; 16-byte chunk c of row r stored at c ^ (r % 8)).
            row_off = a_kv + (t // 8) * 1024 + (t % 8) * 128
            if fmt_fp4:
                # NVFP4: 16-byte chunk c = TPR k + j holds dims [32 c, 32 c + 32); its
                # e4m3 scales are bytes 2 c and 2 c + 1 of the record's 32 scale
                # bytes, i.e. word TPR / 2 * k + j // 2.
                for k in range(n_chunks):
                    c = TPR * k + j
                    kb_off = row_off + (c // 2) * 16384
                    lc0 = (c % 2) * 4
                    word = sfw[(TPR // 2) * k]
                    for jj in range(1, TPR // 2):
                        word = _sel(
                            j // 2 == jj, sfw[(TPR // 2) * k + jj], word, Uint32
                        )
                    sh = Uint32((j % 2) * 16)
                    s0 = _e4m3_to_bf16x2((word >> sh) & Uint32(0xFF))
                    s1 = _e4m3_to_bf16x2((word >> (sh + Uint32(8))) & Uint32(0xFF))
                    for w in range(4):
                        o = _fp4x8_to_bf16x8(kv[k][w], s0 if w < 2 else s1)
                        _sts_v4(
                            kb_off + ((lc0 + w) ^ (t % 8)) * 16, o[0], o[1], o[2], o[3]
                        )
            else:
                # fp8: 16-byte chunk c = TPR k + j holds dims [16 c, 16 c + 16) (k-block
                # c // 4, 16-byte SMEM chunks 2 (j % 4) + {0, 1}), scaled by ue8m0 byte
                # g = c // 2 = TPR / 2 * k + j // 2 of the record's 16 scale bytes.
                for k in range(n_chunks):
                    c = TPR * k + j
                    kb_off = row_off + (c // 4) * 16384
                    g0 = (TPR // 2) * k  # static part of g; j // 2 adds 0..TPR/2-1
                    e = (
                        sfw[g0 // 4] >> Uint32(8 * (g0 % 4)) >> Uint32(8 * (j // 2))
                    ) & Uint32(0xFF)
                    for w in range(2):
                        o = _fp8x8_to_bf16x8_ue8m0(kv[k][2 * w], kv[k][2 * w + 1], e)
                        _sts_v4(
                            kb_off + ((2 * (j % 4) + w) ^ (t % 8)) * 16,
                            o[0],
                            o[1],
                            o[2],
                            o[3],
                        )
            _sts_s32(a_valid + t * 4, vp)
            if rows < _TILE:  # the unwritten rows never take part
                _sts_s32(a_valid + (rows + tid % (_TILE - rows)) * 4, Int32(0))
            _sts_v4p(a_cs + cs_lane * 16, csw[0], csw[1], csw[2], csw[3], cs_pred)
            return _i32(live)

        @cute.kernel
        def device_kernel(
            q: cute.Tensor,
            swa: cute.Tensor,
            swa_idx: cute.Tensor,
            swa_len: cute.Tensor,
            ext: cute.Tensor,
            ext_idx: cute.Tensor,
            ext_len: cute.Tensor,
            positions: cute.Tensor,
            cos_sin: cute.Tensor,
            sink: cute.Tensor,
            out: cute.Tensor,
            out_sf: cute.Tensor,
            scale_log2: Float32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            rank, b, _ = cute.arch.block_idx()
            warp = cute.arch.make_warp_uniform(tid // 32)
            lane = tid % 32

            smem = utils.SmemAllocator()
            p_kv = smem.allocate(_TILE * _HEAD_DIM * 2, byte_alignment=1024)
            p_q = smem.allocate(H * _HEAD_DIM * 2, byte_alignment=1024)
            p_p = smem.allocate(_TILE * P_ROW, byte_alignment=1024)
            p_valid = smem.allocate(_TILE * 4, byte_alignment=16)
            p_red = smem.allocate(2 * 4 * H * 4, byte_alignment=16)
            p_stat_all = smem.allocate(STATS * 4, byte_alignment=16)
            p_cs = smem.allocate(_ROPE_DIM * 4, byte_alignment=16)
            mbar = smem.allocate_array(Int64, 2, byte_alignment=8)
            p_tmem = smem.allocate_array(Int32, 4, byte_alignment=16)
            a_kv = p_kv.toint()
            a_q = p_q.toint()
            a_p = p_p.toint()
            a_valid = p_valid.toint()
            a_rmax = p_red.toint()
            a_rsum = a_rmax + 4 * H * 4
            # Own stats (m, l per head) live in this CTA's slot of the all-stats array.
            a_stat = p_stat_all.toint() + rank * (2 * H * 4)
            a_stat_all = p_stat_all.toint()
            a_cs = p_cs.toint()

            if tid == 0:
                cute.arch.mbarrier_init(mbar, 1)
                cute.arch.mbarrier_init(mbar + 1, 1)
                cute.arch.mbarrier_init_fence()
            # Every global read below may depend on the PDL predecessor.
            cute.arch.griddepcontrol_wait()
            # TMEM after the PDL wait: a predecessor on this SM may hold it.
            if warp == 0:
                cute.arch.alloc_tmem(TMEM_COLS, p_tmem)
                cute.arch.relinquish_tmem_alloc_permit()
                _tcgen05.fence_before_thread_sync()

            # ---- 1-2: loads, Q RoPE, KV dequant into SMEM
            tile_live = Int32(0)
            if rank < KT:
                tile_live = stage_inputs(
                    False,
                    swa,
                    swa_idx,
                    swa_len,
                    q,
                    positions,
                    cos_sin,
                    b,
                    tid,
                    rank * _SWA_ROWS,
                    NK,
                    KP,
                    a_kv,
                    a_q,
                    a_valid,
                    a_cs,
                    _SWA_ROWS,
                )
            else:
                tile_live = stage_inputs(
                    EXTRA_FP4,
                    ext,
                    ext_idx,
                    ext_len,
                    q,
                    positions,
                    cos_sin,
                    b,
                    tid,
                    (rank - KT) * _TILE,
                    NE,
                    EP,
                    a_kv,
                    a_q,
                    a_valid,
                    a_cs,
                    _TILE,
                )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            _tcgen05.fence_after_thread_sync()
            tmem = _lds_s32(p_tmem.toint())

            if tile_live != 0:
                # ---- 3: S[128, H] = KV[128, 512] . Q[H, 512]^T
                if warp == 0:
                    _tcgen05.fence_after_thread_sync()
                    s_adesc = DESC_KMAJ | Uint64((a_kv >> 4) & 0x3FFF)
                    s_bdesc = DESC_KMAJ | Uint64((a_q >> 4) & 0x3FFF)
                    for s_kb in cutlass.range_constexpr(_HEAD_DIM // 64):
                        for s_ks in cutlass.range_constexpr(4):
                            _tcgen05.mma_f16(
                                tmem + S_COL,
                                s_adesc + (s_kb * 16384 + s_ks * 32) // 16,
                                s_bdesc + (s_kb * KB_Q + s_ks * 32) // 16,
                                IDESC_S,
                                s_kb > 0 or s_ks > 0,
                            )
                    _tcgen05.commit(mbar)

                # ---- 4: softmax over the tile's 128 rows (column-wise per head)
                quad = warp % 4
                cg = warp // 4
                t = quad * 32 + lane
                cute.arch.mbarrier_wait(mbar, 0)
                _tcgen05.fence_after_thread_sync()
                sv = cute.make_rmem_tensor(NC, Float32)
                sv.store(_tcgen05.ld(quad * 32, tmem + S_COL + cg * NC, "32x32b", NC))
                _tcgen05.wait_ld()
                row_ok = _lds_s32(a_valid + t * 4) != 0
                neg_inf = Float32(-math.inf)
                xs = []
                for i in cutlass.range_constexpr(NC):
                    xs.append(_sel(row_ok, sv[i] * scale_log2, neg_inf, Float32))
                # Column max over the 128 rows: butterfly over the 32 lanes (all of
                # this thread's columns in lockstep), then across the 4 lane quadrants.
                mxs = list(xs)
                for sh in cutlass.range_constexpr(5):
                    for i in cutlass.range_constexpr(NC):
                        mxs[i] = cute.arch.fmax(
                            mxs[i], cute.arch.shuffle_sync_bfly(mxs[i], offset=16 >> sh)
                        )
                for k4 in cutlass.range_constexpr(NC // 4):
                    _sts_v4f(
                        a_rmax + (quad * H + cg * NC + 4 * k4) * 4,
                        mxs[4 * k4],
                        mxs[4 * k4 + 1],
                        mxs[4 * k4 + 2],
                        mxs[4 * k4 + 3],
                        _i32(lane == 0),
                    )
                cute.arch.sync_threads()
                cmax = []
                for k4 in cutlass.range_constexpr(NC // 4):
                    v4 = list(_lds_v4f(a_rmax + (cg * NC + 4 * k4) * 4))
                    for qd in cutlass.range_constexpr(1, 4):
                        w4 = _lds_v4f(a_rmax + (qd * H + cg * NC + 4 * k4) * 4)
                        for e in cutlass.range_constexpr(4):
                            v4[e] = cute.arch.fmax(v4[e], w4[e])
                    for e in cutlass.range_constexpr(4):
                        cmax.append(
                            _sel(v4[e] == neg_inf, Float32(0.0), v4[e], Float32)
                        )
                probs = []
                for i in cutlass.range_constexpr(NC):
                    probs.append(_ex2(xs[i] - cmax[i]))
                # P = hi + lo (bf16 each): one [128, 2H] MN-major B tile.
                hi_w = []
                lo_w = []
                for i in cutlass.range_constexpr(NC // 2):
                    hw = cvt.fp32x2_to_bf16x2(probs[2 * i], probs[2 * i + 1])
                    h0, h1 = cvt.bf16x2_to_fp32x2(hw)
                    hi_w.append(hw)
                    lo_w.append(
                        cvt.fp32x2_to_bf16x2(probs[2 * i] - h0, probs[2 * i + 1] - h1)
                    )
                if cutlass.const_expr(H == 32):
                    prow = a_p + (t // 8) * 1024 + (t % 8) * 128
                    _sts_v4(prow + ((cg ^ (t % 8)) * 16), *hi_w)
                    _sts_v4(prow + (((4 + cg) ^ (t % 8)) * 16), *lo_w)
                else:
                    prow = a_p + (t // 8) * 512 + (t % 8) * 64 + (cg % 2) * 8
                    sw = (t % 8) // 2
                    _sts_v2(prow + (((cg // 2) ^ sw) * 16), *hi_w)
                    _sts_v2(prow + (((2 + cg // 2) ^ sw) * 16), *lo_w)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()

                # ---- 5: O[512, 2H] = KV^T[512, 128] . [P_hi | P_lo][128, 2H]
                if warp == 0:
                    _tcgen05.fence_after_thread_sync()
                    o_adesc = DESC_KV_MN | Uint64((a_kv >> 4) & 0x3FFF)
                    o_bdesc = DESC_P | Uint64((a_p >> 4) & 0x3FFF)
                    for o_mb in cutlass.range_constexpr(_HEAD_DIM // _TILE):
                        for o_ks in cutlass.range_constexpr(_SWA_ROWS // 16):
                            _tcgen05.mma_f16(
                                tmem + o_mb * 2 * H,
                                o_adesc + (o_mb * 2 * 16384 + o_ks * 2048) // 16,
                                o_bdesc + (o_ks * 16 * P_ROW) // 16,
                                IDESC_O,
                                o_ks > 0,
                            )
                    # Compressed tiles: the other 64 K rows (an SWA tile's upper
                    # half is stale SMEM and must not be read).
                    if rank >= KT:
                        for o_mb in cutlass.range_constexpr(_HEAD_DIM // _TILE):
                            for o_ks in cutlass.range_constexpr(
                                _SWA_ROWS // 16, _TILE // 16
                            ):
                                _tcgen05.mma_f16(
                                    tmem + o_mb * 2 * H,
                                    o_adesc + (o_mb * 2 * 16384 + o_ks * 2048) // 16,
                                    o_bdesc + (o_ks * 16 * P_ROW) // 16,
                                    IDESC_O,
                                    True,
                                )
                    _tcgen05.commit(mbar + 1)
                # Column sums (only the merge needs them) while MMA2 runs.
                sums = list(probs)
                for sh in cutlass.range_constexpr(5):
                    for i in cutlass.range_constexpr(NC):
                        sums[i] = sums[i] + cute.arch.shuffle_sync_bfly(
                            sums[i], offset=16 >> sh
                        )
                for k4 in cutlass.range_constexpr(NC // 4):
                    _sts_v4f(
                        a_rsum + (quad * H + cg * NC + 4 * k4) * 4,
                        sums[4 * k4],
                        sums[4 * k4 + 1],
                        sums[4 * k4 + 2],
                        sums[4 * k4 + 3],
                        _i32(lane == 0),
                    )
                cute.arch.sync_threads()
                if tid < H:
                    st_m = _lds_f32(a_rmax + tid * 4)
                    st_l = _lds_f32(a_rsum + tid * 4)
                    for st_q in cutlass.range_constexpr(1, 4):
                        st_m = cute.arch.fmax(
                            st_m, _lds_f32(a_rmax + (st_q * H + tid) * 4)
                        )
                        st_l = st_l + _lds_f32(a_rsum + (st_q * H + tid) * 4)
                    st_m = _sel(st_m == Float32(-math.inf), Float32(0.0), st_m, Float32)
                    _sts_f32(a_stat + tid * 4, st_m)
                    _sts_f32(a_stat + (H + tid) * 4, st_l)

                # ---- 6: O = hi + lo -> fp32 [H, 512] SMEM over the KV tile
                cute.arch.mbarrier_wait(mbar + 1, 0)
                _tcgen05.fence_after_thread_sync()
                mb = warp // 4
                ov = cute.make_rmem_tensor(2 * H, Float32)
                ov.store(_tcgen05.ld(quad * 32, tmem + mb * 2 * H, "32x32b", 2 * H))
                _tcgen05.wait_ld()
                d = mb * _TILE + quad * 32 + lane
                for h in cutlass.range_constexpr(H):
                    o = ov[h] + ov[H + h]
                    _sts_f32(a_kv + (h * _HEAD_DIM + d) * 4, o)
            else:
                if tid < H:
                    _sts_f32(a_stat + tid * 4, Float32(0.0))
                    _sts_f32(a_stat + (H + tid) * 4, Float32(0.0))
                # Peers pull every partial unconditionally: hand them zeros.
                zero = Uint32(0)
                for z in cutlass.range_constexpr(H * _HEAD_DIM // (4 * _THREADS)):
                    _sts_v4(a_kv + (tid + z * _THREADS) * 16, zero, zero, zero, zero)

            # ---- 7: every CTA's partial and stats are final.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            cute.arch.griddepcontrol_launch_dependents()
            # ---- 8: pull this CTA's (head, 32-dim chunk) units from every peer's
            # partial (fixed peer order: deterministic) while gathering their stats.
            u_begin = rank * UPC
            u_count = cutlass.min(Int32(UPC), Int32(U) - u_begin)
            sub = tid % 8
            pulls = []
            for ps in cutlass.range_constexpr(NPASS):
                uo = ps * (_THREADS // 8) + tid // 8
                u = u_begin + cutlass.min(uo, u_count - 1)
                src = a_kv + u * UNIT_BYTES + sub * 16
                row = []
                for qp in cutlass.range_constexpr(C):
                    row.append(_ld_peer_f32x4(src, Int32(qp), Int32(1)))
                pulls.append(row)
            for k in cutlass.range_constexpr(cdiv(STATS, _THREADS)):
                i = cutlass.min(tid + _THREADS * k, Int32(STATS - 1))
                peer = i // (2 * H)
                v = _ld_peer_f32(p_stat_all.toint() + i * 4, peer)
                _sts_f32(p_stat_all.toint() + i * 4, v, _i32(peer != rank))
            cute.arch.sync_threads()

            # ---- 9: merge this CTA's (head, 32-dim chunk) units (fixed peer order:
            # deterministic; sink folded once), then the epilogue: inverse RoPE,
            # MXFP8 cast, o-fused store.
            out_base = out.iterator.toint() + Int64(b) * Int64(out.stride[0])
            sf_base = out_sf.iterator.toint()
            inf = Float32(math.inf)
            for ps in cutlass.range_constexpr(NPASS):
                uo = ps * (_THREADS // 8) + tid // 8
                active = uo < u_count
                uoc = cutlass.min(uo, u_count - 1)
                u = u_begin + uoc
                hh = u // 16
                cc = u % 16
                sk = sink[hh]
                sk2 = sk * _LOG2E
                parts = pulls[ps]
                ms = []
                ls = []
                big = sk2
                for qp in cutlass.range_constexpr(C):
                    mq = _lds_f32(a_stat_all + (qp * 2 * H + hh) * 4)
                    lq = _lds_f32(a_stat_all + (qp * 2 * H + H + hh) * 4)
                    ms.append(mq)
                    ls.append(lq)
                    big = _sel(lq > Float32(0.0), cute.arch.fmax(big, mq), big, Float32)
                big = _sel((big == inf) | (big == -inf), Float32(0.0), big, Float32)
                den = _ex2(sk2 - big)
                acc = [Float32(0.0)] * 4
                for qp in cutlass.range_constexpr(C):
                    f = _sel(
                        ls[qp] > Float32(0.0), _ex2(ms[qp] - big), Float32(0.0), Float32
                    )
                    den = den + f * ls[qp]
                    for e in cutlass.range_constexpr(4):
                        acc[e] = acc[e] + parts[qp][e] * f
                ok = (den > Float32(0.0)) & (sk != inf)
                for e in cutlass.range_constexpr(4):
                    acc[e] = _sel(ok, acc[e] / den, Float32(0.0), Float32)
                if cutlass.const_expr(ps == NPASS - 1):
                    # Every pulled value is consumed: peers may leave once all of
                    # them got here too (split barrier, waited on before exit).
                    cute.arch.cluster_arrive_relaxed()
                # Inverse GPT-J RoPE on dims >= 448 (identity elsewhere).
                is_rope = cc >= _ROPE_START // 32
                f0 = cutlass.max((cc - _ROPE_START // 32) * 16 + sub * 2, Int32(0))
                cs2 = _lds_v2f(a_cs + f0 * 4)
                sn2 = _lds_v2f(a_cs + (f0 + _ROPE_DIM // 2) * 4)
                x = []
                for pr in cutlass.range_constexpr(2):
                    cv = _sel(is_rope, cs2[pr], Float32(1.0), Float32)
                    sv1 = _sel(is_rope, sn2[pr], Float32(0.0), Float32)
                    oe, oo = acc[2 * pr], acc[2 * pr + 1]
                    x.append(_fma(oe, cv, _mul(sv1, oo)))
                    x.append(_fma(oo, cv, -_mul(sv1, oe)))
                # MXFP8: ue8m0 = ceil(log2(amax / 448)) over the 32-dim chunk.
                amax = cute.arch.fmax(
                    cute.arch.fmax(
                        cute.arch.fmax(x[0], -x[0]), cute.arch.fmax(x[1], -x[1])
                    ),
                    cute.arch.fmax(
                        cute.arch.fmax(x[2], -x[2]), cute.arch.fmax(x[3], -x[3])
                    ),
                )
                for off in (4, 2, 1):
                    amax = cute.arch.fmax(
                        amax, cute.arch.shuffle_sync_bfly(amax, offset=off)
                    )
                amax = cute.arch.fmax(amax, Float32(1e-10))
                ex = _ue8m0_rp(amax * Float32(1.0 / 448.0))
                inv = recast_val((Uint32(254) - ex) << Uint32(23), Float32)
                qb = cvt.fp32x4_to_fp8x4(x[0] * inv, x[1] * inv, x[2] * inv, x[3] * inv)
                g = hh // WV_GROUP_SIZE
                hg = hh % WV_GROUP_SIZE
                ka = cc * WV_GROUP_SIZE + hg
                oa = (
                    out_base
                    + Int64(g) * Int64(out.stride[1])
                    + Int64(ka * 32 + sub * 4)
                )
                _stg_b32(oa, qb, _i32(active))
                sw_idx = (
                    Int64(b) * Int64(out_sf.stride[0])
                    + Int64(g) * Int64(out_sf.stride[1])
                    + Int64(ka // 4) * Int64(out_sf.stride[2])
                )
                _stg_b8(
                    sf_base + sw_idx * 4 + Int64(ka % 4), ex, _i32(active & (sub == 0))
                )

            # ---- 10: nobody leaves while a peer may still read its SMEM (the
            # matching arrive follows this CTA's last pull).
            cute.arch.cluster_wait()
            if warp == 0:
                cute.arch.dealloc_tmem(
                    cute.make_ptr(Float32, tmem, cute.AddressSpace.tmem), TMEM_COLS
                )

        @cute.jit
        def host_entrypoint(
            q: cute.Tensor,
            swa: cute.Tensor,
            swa_idx: cute.Tensor,
            swa_len: cute.Tensor,
            ext: cute.Tensor,
            ext_idx: cute.Tensor,
            ext_len: cute.Tensor,
            positions: cute.Tensor,
            cos_sin: cute.Tensor,
            sink: cute.Tensor,
            out: cute.Tensor,
            out_sf: cute.Tensor,
            scale_log2: Float32,
            stream: CUstream,
        ):
            device_kernel(
                q,
                swa,
                swa_idx,
                swa_len,
                ext,
                ext_idx,
                ext_len,
                positions,
                cos_sin,
                sink,
                out,
                out_sf,
                scale_log2,
            ).launch(
                grid=(C, q.shape[0], 1),
                block=(_THREADS, 1, 1),
                cluster=(C, 1, 1),
                stream=stream,
                use_pdl=True,
                # 150+ KiB of SMEM leaves room for one CTA per SM anyway; saying
                # so lets ptxas use up to 128 registers per thread.
                min_blocks_per_mp=1,
            )

        return host_entrypoint

    def dispatch(self, **compile_key_fields: int) -> CompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(self, **layer_shape: int) -> list[CompileKey]:
        # The token count is a grid dimension, so one executor per layer shape
        # serves every decode step.
        return self._trace_dispatch(self.dispatch)(**layer_shape)

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        def cache(page: int, record: int):
            return make_fake_tensor(
                Uint8,
                (cute.sym_int(), page, record),
                (cute.sym_int(), record, 1),
                assumed_align=16,
            )

        # Independent leading dims: metadata buffers may be longer than s_q.
        def slots(width: int):
            return make_fake_tensor(
                Int32, (cute.sym_int(), width), (cute.sym_int(), 1), assumed_align=4
            )

        def vec(dtype):
            return make_fake_tensor(dtype, (cute.sym_int(),), (1,), assumed_align=4)

        out_dim = WV_GROUP_SIZE * _HEAD_DIM
        return (
            make_fake_tensor(
                BFloat16,
                (cute.sym_int(), cute.sym_int(), _HEAD_DIM),
                (cute.sym_int(), _HEAD_DIM, 1),
                assumed_align=16,
            ),
            cache(compile_key.swa_page, _FP8_RECORD),
            slots(compile_key.swa_width),
            vec(Int32),
            cache(compile_key.extra_page, compile_key.extra_record),
            slots(compile_key.extra_width),
            vec(Int32),
            vec(Int32),
            make_fake_tensor(
                Float32,
                (cute.sym_int(), _ROPE_DIM),
                (cute.sym_int(), 1),
                assumed_align=16,
            ),
            vec(Float32),
            make_fake_tensor(
                Float8E4M3FN,
                (cute.sym_int(), cute.sym_int(), out_dim),
                (cute.sym_int(), out_dim, 1),
                assumed_align=16,
            ),
            make_fake_tensor(
                Int32,
                (cute.sym_int(), cute.sym_int(), out_dim // 128),
                (cute.sym_int(), cute.sym_int(), cute.sym_int()),
                assumed_align=4,
            ),
            Float32(0.0),
        )

    @kernel_launcher
    def __call__(
        self,
        *,
        q: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_idx: torch.Tensor,
        swa_len: torch.Tensor,
        extra_cache: torch.Tensor,
        extra_idx: torch.Tensor,
        extra_len: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        sink: torch.Tensor,
        heads: int,
        softmax_scale: float,
        out_data: torch.Tensor,
        out_scale: torch.Tensor,
    ) -> CuTeDSLLaunchSpec["SwapABDecodeKernel.CompileKey"]:
        """Decode attention of ``heads`` live heads, in place into ``out_*``.

        The arguments are those of the mega kernel's decode call: ``q`` is
        ``[s_q, padded_heads, 512]`` bf16 in its chunk-interleaved layout, not
        yet RoPEd; the caches are ``[num_blocks, page, record]`` uint8 views
        of V4.1 records (a token's data, then its scales) whose pages may be
        further apart than ``page * record`` bytes; the slot lists are
        ``[s_q, width]`` int32 slot ids (``block * page + offset``, ``-1``
        invalid) with ``[s_q]`` int32 lengths; ``positions`` is int32. Only
        the live heads' groups of ``out_data`` / ``out_scale`` are written.
        """
        assert q.shape[1] >= heads
        compile_key = self.dispatch(
            heads=heads,
            swa_width=swa_idx.shape[-1],
            extra_width=extra_idx.shape[-1],
            extra_record=extra_cache.shape[-1],
            swa_page=swa_cache.shape[1],
            extra_page=extra_cache.shape[1],
        )
        launch_args = (
            q,
            swa_cache,
            swa_idx,
            swa_len,
            extra_cache,
            extra_idx,
            extra_len,
            positions,
            cos_sin_cache,
            sink,
            out_data,
            out_scale,
            softmax_scale * _LOG2E,
        )
        return compile_key, launch_args


def swapab_decode_supported(heads: int, swa_width: int, extra_width: int) -> bool:
    """Whether the kernel serves these live heads and slot-list widths.

    A token's KV tiles have to fit one portable cluster, and the fp8 / fp4 ->
    bf16 conversions need PTX ISA 9.2, i.e. a CuTe DSL built with CUDA 13.2+.
    """
    tiles = cdiv(swa_width, _SWA_ROWS) + cdiv(extra_width, _TILE)
    return (
        heads in (16, 32)
        and 1 <= tiles <= _MAX_CLUSTER
        and target_version(min_version="13.2")
    )


_SWAPAB_DECODE_KERNEL = SwapABDecodeKernel()
