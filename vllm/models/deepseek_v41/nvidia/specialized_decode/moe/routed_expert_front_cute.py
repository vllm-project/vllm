# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# The MXFP8 quantization helpers below (abs_max_bf16x2 through
# bf16x4_to_e4m3x4) are adapted from FlashInfer
# (flashinfer/quantization/quantization_cute_dsl_utils.py and
# flashinfer/cute_dsl/fp4_common.py, flashinfer-python 0.7.1), which carry the
# notice below. Modified by the vLLM contributors: helpers are merged and
# renamed.
#
# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# FlashInfer's NOTICE: FlashInfer / Copyright 2025-2026 NVIDIA /
# Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
"""Fused MoE front of the DeepSeek-V4.1 low-batch routed experts in CuTe DSL.

Between the router GEMM and the first routed-expert GEMM, the stock path runs
three kernels; :class:`RoutedExpertFront` replaces them with one launch:

1. the DSv4 top-k (vLLM's Triton ``_dsv4_topk_kernel``, ``dsv4_topk.py``):
   per row, ``score = sqrt(softplus(logit))`` (softplus(x) = x above 20, else
   log(1 + exp(x))), ``biased = score + bias`` (``bias_vl`` for image sentinel
   tokens), the six largest biased scores (lowest expert id on exact ties, NaN
   ranked as -1e30), and ``weights`` = the unbiased scores times scale / sum
   (sum replaced by 1 unless positive); a CUDA graph padding row
   (``VLLM_MOE_SKIP_PADDING``'s mask) gets ids -1 and weights 0, so no expert
   computes it;
2. the routed input's MXFP8 quantization (FlashInfer's linear-layout
   ``mxfp8_quantize``): per 32-value block, a UE8M0 scale rounded up from
   amax / 448 and E4M3 values rounded to nearest with saturation;
3. the routing record (``routed_expert_route_cute.py``).

Bitwise contract. Every output equals that of the three kernels it replaces,
so the arithmetic is spelled out instruction by instruction in inline PTX:
the Triton kernel's exp is ``ex2.approx.f32`` of x * log2(e), its sqrt
``sqrt.approx.ftz.f32``, its division ``div.full.f32``, its weight sum the
butterfly order of a one-warp eight-lane reduction; the quantization is
FlashInfer's instruction sequence. The one exception is the log: like the
Triton kernel, the kernel calls the toolchain's ``logf`` (libdevice, through
``cute.math.log``), whose argument 1 + exp(x) is at least 1. No other
floating-point operation is left to the compiler, which could contract or
flush it.

Work split (1024 threads a CTA). The kernel is built for calls of up to 8,
16, 32, 64, 128 or 192 rows (``max_rows``); the first ``S = 8`` CTAs select
and form one thread block cluster, the rest quantize:

* Selecting CTA ``s`` takes rows ``s``, ``s + S``, ``s + 2 S``, ... (at most
  ``max_rows / S`` rows: 1, 2, 4 or 8 up to 64 rows, 16 at 128, 24 at 192).
  Four warps per row (two at 16 rows a CTA, one at 24) each take the top-6 of
  their share of the experts (3, 6 or 12 per lane); one warp per row merges
  the candidates and normalizes. The top-k is bound by instruction issue,
  so its time grows with the rows a CTA holds; the rows are therefore spread
  over all eight CTAs, even at 8 rows or fewer. The
  selection is a lexicographic (value, lowest id) maximum, so splitting it
  does not change the result. The merging warps store the selected ids to
  CTA 0's shared memory (through distributed shared memory) and release
  them with the cluster barrier, then CTA 0 builds the routing record from
  them. The ids and weights go to global memory only after that release,
  which would otherwise wait for those stores to reach L2.
* Quantizing CTA ``S + c``: rows ``3c .. 3c + 2`` of the BF16 input, two
  threads per 32-value block. The grid is rounded up to whole clusters.

PDL: every CTA triggers its dependents on entry (GEMM1's set-up overlaps this
kernel; GEMM1 reads the outputs only after its own grid-dependency wait). The
bias rows are loaded before this kernel's grid-dependency wait (they are
weights); the logits, input ids and activations after it.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ...ops.cute_dsl.primitives import load_global_u32x4
from .routed_expert_route_cute import (
    MAX_EXPERTS,
    build_record,
    clear_scratch,
    levels_for,
    mask_words_for,
    present_words,
    record_words,
    zero_record,
)

HIDDEN = 5120
EXPERTS = 384
TOP_K = 6
BLOCKS = HIDDEN // 32  # quantization blocks per row
FRONT_THREADS = 1024
ROW_THREADS = 2 * BLOCKS  # 320 threads quantize one row
QUANT_ROWS = FRONT_THREADS // ROW_THREADS  # rows per quantizing CTA
SELECT_CTAS = 8  # the selecting CTAs: a portable cluster
MAX_PARTS = 4  # the most warps per row: 4 x 6 candidates fit one merging warp
NAN_KEY_VALUE = 0xF149F2CA  # -1e30, Triton's rank of a NaN score

assert EXPERTS <= MAX_EXPERTS
assert EXPERTS % 32 == 0


def _asm(result, operands: list, text: str, constraints: str, loc, ip):
    """A side-effect-free PTX inline assembly operation."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _f32(value, loc, ip):
    """IR value of a Float32."""
    return Float32(value).ir_value(loc=loc, ip=ip)


def _u32(value, loc, ip):
    """IR value of a Uint32."""
    return Uint32(value).ir_value(loc=loc, ip=ip)


# ---- top-k

# 1 + exp(logit), the softplus logarithm's argument.
_SOFTPLUS_ARGUMENT = """{
.reg .f32 a;
mul.f32 a, $1, 0f3FB8AA3B;
ex2.approx.f32 a, a;
add.f32 $0, a, 0f3F800000;
}"""

# score = sqrt(softplus(logit)) from the logarithm ($3) of that argument, then
# the selection key of score + bias: an order-preserving unsigned key (larger
# value, larger key; -0 keys as +0; NaN as -1e30).
_SCORE_KEY = """{
.reg .pred p;
.reg .f32 l, s, v;
.reg .b32 b, n;
setp.gt.f32 p, $2, 0f41A00000;
selp.f32 l, $2, $3, p;
sqrt.approx.ftz.f32 s, l;
add.f32 v, s, $4;
setp.num.f32 p, v, v;
selp.f32 v, v, 0fF149F2CA, p;
setp.eq.f32 p, v, 0f00000000;
selp.f32 v, 0f00000000, v, p;
mov.b32 b, v;
setp.lt.s32 p, b, 0;
not.b32 n, b;
or.b32 b, b, 0x80000000;
selp.b32 $1, n, b, p;
mov.f32 $0, s;
}"""


@dsl_user_op
def score_and_key(logit, bias, *, loc=None, ip=None):
    """The unbiased score of ``logit`` and the selection key of score + bias."""
    argument = Float32(
        _asm(T.f32(), [_f32(logit, loc, ip)], _SOFTPLUS_ARGUMENT, "=f,f", loc, ip)
    )
    # The toolchain's logf (libdevice), as in the Triton kernel.
    log_argument = cute.math.log(argument, fastmath=False)
    result = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        [_f32(logit, loc, ip), _f32(log_argument, loc, ip), _f32(bias, loc, ip)],
        _SCORE_KEY,
        "=f,=r,f,f,f",
        loc,
        ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def add_f32(a, b, *, loc=None, ip=None) -> Float32:
    """``a + b``, round to nearest, no contraction."""
    operands = [_f32(a, loc, ip), _f32(b, loc, ip)]
    return Float32(_asm(T.f32(), operands, "add.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def normalized_weight(scale, total, weight, *, loc=None, ip=None) -> Float32:
    """``scale / (total if total > 0 else 1) * weight``, as Triton divides."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(scale, loc, ip), _f32(total, loc, ip), _f32(weight, loc, ip)],
            "{\n.reg .pred p;\n.reg .f32 d;\nsetp.gt.f32 p, $2, 0f00000000;\n"
            "selp.f32 d, $2, 0f3F800000, p;\ndiv.full.f32 d, $1, d;\n"
            "mul.f32 $0, d, $3;\n}",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@cute.jit
def lane_best(key: cute.Tensor):
    """The lane's best remaining entry: the largest key, then the lowest j."""
    best_key = Uint32(key[0])
    best_j = Int32(0)
    for j in cutlass.range_constexpr(1, cute.size(key)):
        if Uint32(key[j]) > best_key:
            best_key = Uint32(key[j])
            best_j = Int32(j)
    return best_key, best_j


@cute.jit
def part_top_k(logit: cute.Tensor, bias: cute.Tensor, base: Int32, lane: Int32):
    """One warp's top-6 of its share of the experts (expert = base + lane +
    32 j, ``j`` below the size of ``logit``), in
    selection order: each round takes the warp's largest key with one
    ``redux.sync`` max, then the lowest expert id holding it with one min
    (Triton's ``tl.max``, then the lowest index equal to it); only the winning
    lane rescans. Lane ``s < 6`` gets slot ``s``: its key, id and unbiased
    score."""
    per_lane = cute.size(logit)
    score = cute.make_rmem_tensor((per_lane,), Float32)
    key = cute.make_rmem_tensor((per_lane,), Uint32)
    for j in cutlass.range_constexpr(per_lane):
        score[j], key[j] = score_and_key(logit[j], bias[j])
    best_key, best_j = lane_best(key)
    out_key = Uint32(0)
    out_id = Int32(-1)
    out_score = Float32(0.0)
    for slot in cutlass.range_constexpr(TOP_K):
        top = Uint32(cute.arch.warp_redux_sync(best_key, "umax"))
        mine = Uint32(0xFFFFFFFF)
        if best_key == top:
            mine = Uint32(base + lane + 32 * best_j)
        winner = Int32(cute.arch.warp_redux_sync(mine, "umin"))
        won = Float32(0.0)
        if winner - base == lane + 32 * best_j:
            for j in cutlass.range_constexpr(per_lane):
                if best_j == j:
                    won = score[j]
                    key[j] = Uint32(0)  # below every value's key
            best_key, best_j = lane_best(key)
        weight = cute.arch.shuffle_sync(won, (winner - base) & 31)
        if lane == slot:
            out_key = top
            out_id = winner
            out_score = weight
    return out_key, out_id, out_score


@cute.jit
def merge_row(
    key: Uint32,
    expert: Int32,
    score: Float32,
    scale: Float32,
    lane: Int32,
    padding: Int32,
):
    """Merge a row's candidates (lane ``c`` holds candidate ``c``) into
    its top-6 and normalize as Triton does; lane ``s < 6`` returns slot
    ``s``'s expert id and weight. A padding row (``padding`` nonzero) gets
    id -1 and weight 0, as the Triton kernel's padding mask does."""
    slot_weight = Float32(0.0)  # lane s < 8 keeps slot s (Triton's 8-wide vector)
    slot_id = Int32(0)
    for slot in cutlass.range_constexpr(TOP_K):
        top = Uint32(cute.arch.warp_redux_sync(key, "umax"))
        mine = Uint32(0xFFFFFFFF)
        if key == top:
            mine = Uint32(expert)
        chosen = Int32(cute.arch.warp_redux_sync(mine, "umin"))
        holders = Uint32(cute.arch.vote_ballot_sync(expert == chosen))
        first = Int32(cute.arch.popc((holders & (Uint32(0) - holders)) - Uint32(1)))
        weight = cute.arch.shuffle_sync(score, first)
        if expert == chosen:
            key = Uint32(0)
        if lane == slot:
            slot_weight = weight
            slot_id = chosen
    # Triton's tl.sum over the eight-wide vector (slots 6 and 7 hold zero):
    # lane i holds element i, butterfly offsets 4, 2, 1.
    total = slot_weight
    for shift in cutlass.range_constexpr(2, -1, -1):
        total = add_f32(total, cute.arch.shuffle_sync_bfly(total, 1 << shift))
    weight = Float32(0.0)
    if lane < TOP_K:
        weight = normalized_weight(scale, total, slot_weight)
        if padding != 0:
            slot_id = Int32(-1)
            weight = Float32(0.0)
    return slot_id, weight


@dsl_user_op
def store_cta0_shared(pointer: cute.Pointer, value, *, loc=None, ip=None):
    """Store the 32-bit ``value`` at ``pointer``'s shared-memory offset in CTA 0
    of the cluster (distributed shared memory)."""
    llvm.inline_asm(
        None,
        [
            Int32(pointer.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .u32 a;\nmapa.shared::cluster.u32 a, $0, 0;\n"
        "st.shared::cluster.u32 [a], $1;\n}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---- MXFP8 quantization (FlashInfer's linear layout, two threads per block)


@dsl_user_op
def abs_max_bf16x2(a, b, *, loc=None, ip=None) -> Uint32:
    """Lane-wise ``max(|a|, |b|)`` of two BF16 pairs."""
    return Uint32(
        _asm(
            T.i32(),
            [_u32(a, loc, ip), _u32(b, loc, ip)],
            "{\n.reg .b32 x, y;\nand.b32 x, $1, 0x7FFF7FFF;\n"
            "and.b32 y, $2, 0x7FFF7FFF;\nmax.bf16x2 $0, x, y;\n}",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def max_bf16x2(a, b, *, loc=None, ip=None) -> Uint32:
    """Lane-wise max of two BF16 pairs."""
    operands = [_u32(a, loc, ip), _u32(b, loc, ip)]
    return Uint32(_asm(T.i32(), operands, "max.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def block_scale(amax, *, loc=None, ip=None) -> Uint32:
    """UE8M0 of amax / 448, rounded up (0 for a zero block)."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(amax, loc, ip)],
            "{\n.reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
            ".reg .u32 bits, exp_biased, mantissa, bump, result;\n"
            ".reg .f32 value;\nmul.f32 value, $1, 0f3B124925;\n"
            "setp.le.f32 p_zero, value, 0f00000000;\n"
            "mov.b32 bits, value;\nshr.b32 exp_biased, bits, 23;\n"
            "and.b32 exp_biased, exp_biased, 255;\n"
            "and.b32 mantissa, bits, 0x7FFFFF;\n"
            "setp.ne.u32 p_has_mant, mantissa, 0;\nselp.u32 bump, 1, 0, p_has_mant;\n"
            "setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
            "setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
            "and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
            "@p_tiny_sub mov.u32 bump, 0;\nadd.u32 result, exp_biased, bump;\n"
            "setp.gt.u32 p_ovf, result, 254;\nselp.u32 result, 254, result, p_ovf;\n"
            "selp.u32 $0, 0, result, p_zero;\n}",
            "=r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def bf16x2_max_to_f32(x, *, loc=None, ip=None) -> Float32:
    """The larger of a BF16 pair, as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [_u32(x, loc, ip)],
            "{\n.reg .b32 lo, hi;\n.reg .f32 f0, f1;\nand.b32 lo, $1, 0xFFFF;\n"
            "shr.b32 hi, $1, 16;\nshl.b32 lo, lo, 16;\nshl.b32 hi, hi, 16;\n"
            "mov.b32 f0, lo;\nmov.b32 f1, hi;\nmax.f32 $0, f0, f1;\n}",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def max_f32(a, b, *, loc=None, ip=None) -> Float32:
    """``max(a, b)`` (no flush)."""
    operands = [_f32(a, loc, ip), _f32(b, loc, ip)]
    return Float32(_asm(T.f32(), operands, "max.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def inverse_scale(code, *, loc=None, ip=None) -> Float32:
    """2^(127 - code), or 0 for code 0."""
    return Float32(
        _asm(
            T.f32(),
            [_u32(code, loc, ip)],
            "{\n.reg .s32 new_exp;\n.reg .b32 float_bits;\n.reg .pred p_zero;\n"
            "setp.eq.u32 p_zero, $1, 0;\nsub.s32 new_exp, 254, $1;\n"
            "max.s32 new_exp, new_exp, 0;\nshl.b32 float_bits, new_exp, 23;\n"
            "mov.b32 $0, float_bits;\n@p_zero mov.b32 $0, 0;\n}",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def bf16x4_to_e4m3x4(a, b, inverse, *, loc=None, ip=None) -> Uint32:
    """Two BF16 pairs times the inverse scale as four saturating E4M3 bytes
    (``a``'s pair first)."""
    convert = (
        "and.b32 lo, {0}, 0xFFFF;\nshr.b32 hi, {0}, 16;\nshl.b32 lo, lo, 16;\n"
        "shl.b32 hi, hi, 16;\nmov.b32 f0, lo;\nmov.b32 f1, hi;\n"
        "mul.f32 f0, f0, $3;\nmul.f32 f1, f1, $3;\n"
        "min.f32 f0, f0, 0f43E00000;\nmax.f32 f0, f0, 0fC3E00000;\n"
        "min.f32 f1, f1, 0f43E00000;\nmax.f32 f1, f1, 0fC3E00000;\n"
        "cvt.rn.satfinite.e4m3x2.f32 {1}, f1, f0;\n"
    )
    return Uint32(
        _asm(
            T.i32(),
            [_u32(a, loc, ip), _u32(b, loc, ip), _f32(inverse, loc, ip)],
            "{\n.reg .b32 lo, hi;\n.reg .f32 f0, f1;\n.reg .b16 p0, p1;\n"
            + convert.format("$1", "p0")
            + convert.format("$2", "p1")
            + "mov.b32 $0, {p0, p1};\n}",
            "=r,r,r,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def store_global_v4(pointer: cute.Pointer, a, b, c, d, *, loc=None, ip=None):
    """One 16-byte store of four words."""
    llvm.inline_asm(
        None,
        [pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
        + [_u32(v, loc, ip) for v in (a, b, c, d)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def quantize_row(
    x: cute.Tensor,
    row: Int32,
    thread: Int32,
    x_quant: cute.Tensor,
    x_scale: cute.Tensor,
):
    """Thread ``thread`` of a row's 320 quantizes half of block
    ``thread / 2``: the block amax over both halves, its UE8M0 scale and 16
    E4M3 values. ``x``: the BF16 input as words."""
    block = thread >> 1
    half = thread & 1
    first = row * HIDDEN + block * 32 + half * 16  # element index
    lo = load_global_u32x4(x.iterator + first // 2)
    hi = load_global_u32x4(x.iterator + first // 2 + 4)
    m01 = max_bf16x2(abs_max_bf16x2(lo[0], lo[1]), abs_max_bf16x2(lo[2], lo[3]))
    m23 = max_bf16x2(abs_max_bf16x2(hi[0], hi[1]), abs_max_bf16x2(hi[2], hi[3]))
    local = bf16x2_max_to_f32(max_bf16x2(m01, m23))
    amax = max_f32(local, cute.arch.shuffle_sync_bfly(local, 1))
    code = block_scale(amax)
    inverse = inverse_scale(code)
    store_global_v4(
        x_quant.iterator + first,
        bf16x4_to_e4m3x4(lo[0], lo[1], inverse),
        bf16x4_to_e4m3x4(lo[2], lo[3], inverse),
        bf16x4_to_e4m3x4(hi[0], hi[1], inverse),
        bf16x4_to_e4m3x4(hi[2], hi[3], inverse),
    )
    if half == 0:
        x_scale[row * BLOCKS + block] = Uint8(code)


class RoutedExpertFront:
    """The fused front (module docstring); ``image``: whether the router
    routes image sentinel tokens with ``bias_vl``; ``max_rows``: 8, 16, 32,
    64, 128 or 192, the most rows of a call."""

    def __init__(self, image: bool, max_rows: int = 8) -> None:
        assert max_rows in (8, 16, 32, 64, 128, 192)
        self.image = image
        self.max_rows = max_rows
        self.selecting = SELECT_CTAS  # CTAs, one cluster
        self.select_rows = max_rows // self.selecting  # 1-8, 16 at 128 rows, 24 at 192
        # Warps per row: four, or as many as fit above 8 rows a CTA.
        self.parts = min(FRONT_THREADS // 32 // self.select_rows, MAX_PARTS)
        self.per_lane = EXPERTS // 32 // self.parts  # experts per lane
        self.candidates = self.parts * TOP_K  # per row, merged by one warp
        self.levels = levels_for(max_rows)
        assert EXPERTS % (32 * self.parts) == 0 and self.candidates <= 32

    @cute.jit
    def __call__(
        self,
        logits_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        bias_vl_ptr: cute.Pointer,
        input_ids_ptr: cute.Pointer,
        ids_are_64bit: Int32,
        sentinel_lo: Int32,
        scale: Float32,
        is_padding_ptr: cute.Pointer,
        has_padding: Int32,
        x_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        x_quant_ptr: cute.Pointer,
        x_scale_ptr: cute.Pointer,
        permuted_ptr: cute.Pointer,
        record_ptr: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """``logits``: FP32 [rows, 384]; ``bias``, ``bias_vl``: FP32 [384];
        ``input_ids``: [rows] int32, or int64 if ``ids_are_64bit``, as Int32
        words (``bias_vl`` and ``input_ids`` are read only with image routing;
        pass any pointers otherwise); ``is_padding``: bool [rows], read only
        if ``has_padding``; ``x``: BF16 [rows, 5120] as Uint32
        words; outputs: ``topk_ids`` int32 / ``topk_weights`` FP32 [rows, 6],
        ``x_quant`` [rows, 5120] E4M3 bytes, ``x_scale`` [rows, 160] UE8M0,
        ``permuted`` [rows, 6] and the routing record. rows in [1, max_rows]."""
        slots = cute.make_layout(rows * TOP_K)
        selecting = self.selecting
        quantizing = (rows + QUANT_ROWS - 1) // QUANT_ROWS
        # Whole clusters: the last one's spare CTAs exit at once.
        quantizing = (quantizing + selecting - 1) // selecting * selecting
        self.kernel(
            cute.make_tensor(logits_ptr, cute.make_layout(rows * EXPERTS)),
            cute.make_tensor(bias_ptr, cute.make_layout(EXPERTS)),
            cute.make_tensor(bias_vl_ptr, cute.make_layout(EXPERTS)),
            cute.make_tensor(input_ids_ptr, cute.make_layout(2 * rows)),
            ids_are_64bit,
            sentinel_lo,
            scale,
            cute.make_tensor(is_padding_ptr, cute.make_layout(rows)),
            has_padding,
            cute.make_tensor(x_ptr, cute.make_layout(rows * HIDDEN // 2)),
            cute.make_tensor(topk_ids_ptr, slots),
            cute.make_tensor(topk_weights_ptr, slots),
            cute.make_tensor(x_quant_ptr, cute.make_layout(rows * HIDDEN)),
            cute.make_tensor(x_scale_ptr, cute.make_layout(rows * BLOCKS)),
            cute.make_tensor(permuted_ptr, slots),
            cute.make_tensor(
                record_ptr, cute.make_layout(record_words(rows, TOP_K, EXPERTS))
            ),
            rows,
        ).launch(
            grid=(selecting + quantizing, 1, 1),
            block=[FRONT_THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
            cluster=[selecting, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        bias_vl: cute.Tensor,
        input_ids: cute.Tensor,
        ids_are_64bit: Int32,
        sentinel_lo: Int32,
        scale: Float32,
        is_padding: cute.Tensor,
        has_padding: Int32,
        x: cute.Tensor,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        x_quant: cute.Tensor,
        x_scale: cute.Tensor,
        permuted: cute.Tensor,
        record: cute.Tensor,
        rows: Int32,
    ):
        """The first ``selecting`` CTAs select (CTA 0 also routes); the others
        quantize (module docstring)."""
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        if cta >= self.selecting:
            row = (cta - self.selecting) * QUANT_ROWS + thread // ROW_THREADS
            if thread < QUANT_ROWS * ROW_THREADS and row < rows:
                cute.arch.griddepcontrol_wait()
                quantize_row(x, row, thread % ROW_THREADS, x_quant, x_scale)
        else:
            self.select_and_route(
                logits,
                bias,
                bias_vl,
                input_ids,
                ids_are_64bit,
                sentinel_lo,
                scale,
                is_padding,
                has_padding,
                topk_ids,
                topk_weights,
                permuted,
                record,
                rows,
                cta,
                thread,
            )

    @cute.jit
    def select_and_route(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        bias_vl: cute.Tensor,
        input_ids: cute.Tensor,
        ids_are_64bit: Int32,
        sentinel_lo: Int32,
        scale: Float32,
        is_padding: cute.Tensor,
        has_padding: Int32,
        topk_ids: cute.Tensor,
        topk_weights: cute.Tensor,
        permuted: cute.Tensor,
        record: cute.Tensor,
        rows: Int32,
        rank: Int32,
        thread: Int32,
    ):
        """Selecting CTA ``rank``: top-k of its rows; then CTA 0 builds the
        routing record from every row's ids."""
        selecting, levels = self.selecting, self.levels
        bitmap_words = present_words(levels)
        mask_size = MAX_EXPERTS * mask_words_for(levels)
        selected = self.max_rows * TOP_K
        parts, per_lane, candidates = self.parts, self.per_lane, self.candidates
        held = self.select_rows * candidates

        @cute.struct
        class Shared:
            present: cute.struct.MemRange[Uint32, bitmap_words]
            rows_of: cute.struct.MemRange[Uint32, mask_size]
            ids: cute.struct.MemRange[Int32, selected]
            candidate_key: cute.struct.MemRange[Uint32, held]
            candidate_id: cute.struct.MemRange[Int32, held]
            candidate_score: cute.struct.MemRange[Float32, held]

        shared = cutlass.utils.SmemAllocator().allocate(Shared)

        def view(member, size):
            return cute.make_tensor(member.data_ptr(), cute.make_layout(size))

        present = view(shared.present, bitmap_words)
        rows_of = view(shared.rows_of, mask_size)
        ids_shared = view(shared.ids, selected)
        candidate_key = view(shared.candidate_key, held)
        candidate_id = view(shared.candidate_id, held)
        candidate_score = view(shared.candidate_score, held)
        # Every CTA of the cluster must have started before the stores to CTA
        # 0's shared memory; this phase completes during the top-k.
        cute.arch.cluster_arrive_relaxed()
        if rank == 0:
            clear_scratch(present, rows_of, thread, FRONT_THREADS, levels)
        warp = thread >> 5
        lane = thread & 31
        local_row = warp // parts
        part = warp % parts
        row = rank + selecting * local_row
        base = part * 32 * per_lane  # this warp's experts: base + lane + 32 j
        row_bias = cute.make_rmem_tensor((per_lane,), Float32)
        row_bias_vl = cute.make_rmem_tensor((per_lane,), Float32)
        if row < rows:
            for j in cutlass.range_constexpr(per_lane):
                row_bias[j] = bias[base + lane + 32 * j]
                if cutlass.const_expr(self.image):
                    row_bias_vl[j] = bias_vl[base + lane + 32 * j]
        cute.arch.griddepcontrol_wait()
        if rank == 0:
            zero_record(record, rows, TOP_K, EXPERTS, thread, FRONT_THREADS)
        if row < rows:
            logit = cute.make_rmem_tensor((per_lane,), Float32)
            for j in cutlass.range_constexpr(per_lane):
                logit[j] = logits[row * EXPERTS + base + lane + 32 * j]
            if cutlass.const_expr(self.image):
                token = Int64(input_ids[row])
                if ids_are_64bit != 0:
                    low_word = Int64(Uint32(input_ids[2 * row]))
                    token = (Int64(input_ids[2 * row + 1]) << 32) | low_word
                low = Int64(sentinel_lo)
                if token >= low and token < low + 5:
                    for j in cutlass.range_constexpr(per_lane):
                        row_bias[j] = row_bias_vl[j]
            key, expert, score = part_top_k(logit, row_bias, base, lane)
            if lane < TOP_K:
                slot = local_row * candidates + part * TOP_K + lane
                candidate_key[slot] = key
                candidate_id[slot] = expert
                candidate_score[slot] = score
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        slot_id = Int32(-1)
        slot_weight = Float32(0.0)
        if row < rows and part == 0:
            key = Uint32(0)
            expert = Int32(-1)
            score = Float32(0.0)
            if lane < candidates:
                key = Uint32(candidate_key[local_row * candidates + lane])
                expert = candidate_id[local_row * candidates + lane]
                score = candidate_score[local_row * candidates + lane]
            padding = Int32(0)
            if has_padding != 0:
                padding = Int32(is_padding[row])
            slot_id, slot_weight = merge_row(key, expert, score, scale, lane, padding)
            if lane < TOP_K:
                store_cta0_shared(ids_shared.iterator + row * TOP_K + lane, slot_id)
        # Release the stores to CTA 0; CTA 0 acquires them. The release orders
        # every earlier store of the thread, so the global stores follow it
        # rather than delay it by their round trip to L2 (about 0.2-0.4 us).
        cute.arch.cluster_arrive()
        if row < rows and part == 0 and lane < TOP_K:
            topk_ids[row * TOP_K + lane] = slot_id
            topk_weights[row * TOP_K + lane] = slot_weight
        cute.arch.cluster_wait()
        if rank == 0:
            build_record(
                present,
                rows_of,
                ids_shared,
                permuted,
                record,
                rows,
                Int32(TOP_K),
                Int32(EXPERTS),
                thread,
                FRONT_THREADS,
                levels,
            )
