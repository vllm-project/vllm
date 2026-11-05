# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Front kernel of the MiniMax-M3 low-batch FFN in CuTe DSL.

:class:`LowBatchFront` replaces the routing kernel and the routed input's
quantization of the stock chain with one launch (grid ``1 + rows``, 768
threads), built for calls of up to 8, 16 or 32 rows (the record's levels):

* CTA 0, warp ``w`` takes rows ``w`` and ``w + 24``: ``biased =
  sigmoid(logit) + bias`` in FP32 for the 128 experts, the four largest biased
  scores (lowest expert id on exact ties), ``s_k = biased_k - bias_k`` (the
  recovered score, as the routing policy computes it) and the weights ``scale *
  s_k / (sum(s) + 1e-20)`` in BF16; then the whole CTA builds the routing
  record (``routed_expert_route_cute.py``) from the selected ids;
* CTA ``1 + r``: row ``r`` of the BF16 input to per-token NVFP4, eight values
  per thread.

The record also gets one zeroed word per token tile after its own words: the
tiles' FC1 item counts (``gemm_cute.py``).

The quantization is per token, as the stock activation quantization does it
(its fast path): ``pt = amax(|row|) * (1 / 2688)``, ``encode = rcp(pt)``; per
16-value block ``sf = e4m3(encode * (amax(block) * rcp(6)))``, ``out =
rcp(f32(sf) * rcp(encode))``, ``code = e2m1(value * out)`` (``rcp`` is
``rcp.approx.ftz``, conversions round to nearest and saturate). The GEMMs
dequantize with ``pt``. The input's scales are stored linearly ([rows,
384]). FC1 quantizes its ``h`` rows for FC2 with the same helpers, the
scales in the 128x4 atom layout (``H_TILE_BYTES`` per 8-row tile).

PDL: the front triggers its dependents on entry; every read of a previous
kernel's output follows the grid-dependency wait (the routing bias, a weight,
is read before it).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    f32_to_bf16_bits,
    load_global_u32x4,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_front_cute import (  # noqa: E501
    abs_max_bf16x2,
    add_f32,
    bf16x2_max_to_f32,
    max_bf16x2,
    max_f32,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_route_cute import (  # noqa: E501
    MAX_EXPERTS,
    build_record,
    clear_scratch,
    levels_for,
    mask_words_for,
    max_tiles,
    present_words,
    record_words,
    tile_words,
    zero_record,
)

from .. import EXPERTS, HIDDEN, TOP_K
from .. import INTERMEDIATE_PER_RANK as INTERMEDIATE

MAX_ROWS = 32
SF_VEC = 16
THREADS = 768
PER_LANE = EXPERTS // 32
X_THREADS = HIDDEN // 8  # threads per input row
H_TILE_BYTES = 128 * INTERMEDIATE // SF_VEC  # one tile's h scales

assert X_THREADS == THREADS
assert EXPERTS <= MAX_EXPERTS and MAX_ROWS * TOP_K <= THREADS


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
    return Float32(value).ir_value(loc=loc, ip=ip)


def _u32(value, loc, ip):
    return Uint32(value).ir_value(loc=loc, ip=ip)


# ---- per-token NVFP4

# Eight BF16 values ($2-$5) of a block with amax $6, under encode scale $7:
# the eight E2M1 codes ($0, value 0 in the low nibble) and the block's E4M3
# scale byte ($1).
_QUANTIZE_X8 = """{
.reg .pred p;
.reg .b16 sb, s_lo, s_hi;
.reg .b32 sw, w;
.reg .f32 r6, sf, sv, ie, os, a, b;
.reg .b8 q0, q1, q2, q3;
rcp.approx.ftz.f32 r6, 0f40C00000;
mul.f32 sf, $6, r6;
mul.f32 sf, $7, sf;
cvt.rn.satfinite.e4m3x2.f32 sb, 0f00000000, sf;
cvt.rn.f16x2.e4m3x2 sw, sb;
mov.b32 {s_lo, s_hi}, sw;
cvt.f32.f16 sv, s_lo;
rcp.approx.ftz.f32 ie, $7;
mul.f32 sv, sv, ie;
rcp.approx.ftz.f32 os, sv;
setp.ne.f32 p, $6, 0f00000000;
selp.f32 os, os, 0f00000000, p;
shl.b32 w, $2, 16;
mov.b32 a, w;
and.b32 w, $2, 0xFFFF0000;
mov.b32 b, w;
mul.f32 a, a, os;
mul.f32 b, b, os;
cvt.rn.satfinite.e2m1x2.f32 q0, b, a;
shl.b32 w, $3, 16;
mov.b32 a, w;
and.b32 w, $3, 0xFFFF0000;
mov.b32 b, w;
mul.f32 a, a, os;
mul.f32 b, b, os;
cvt.rn.satfinite.e2m1x2.f32 q1, b, a;
shl.b32 w, $4, 16;
mov.b32 a, w;
and.b32 w, $4, 0xFFFF0000;
mov.b32 b, w;
mul.f32 a, a, os;
mul.f32 b, b, os;
cvt.rn.satfinite.e2m1x2.f32 q2, b, a;
shl.b32 w, $5, 16;
mov.b32 a, w;
and.b32 w, $5, 0xFFFF0000;
mov.b32 b, w;
mul.f32 a, a, os;
mul.f32 b, b, os;
cvt.rn.satfinite.e2m1x2.f32 q3, b, a;
mov.b32 $0, {q0, q1, q2, q3};
cvt.u32.u16 $1, sb;
}"""


@dsl_user_op
def quantize_x8(words, block_amax, encode, *, loc=None, ip=None):
    """Eight BF16 values (four words) of one block to E2M1 codes and the
    block's E4M3 scale (module docstring)."""
    result = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [_u32(words[i], loc, ip) for i in range(4)]
        + [_f32(block_amax, loc, ip), _f32(encode, loc, ip)],
        _QUANTIZE_X8,
        "=r,=r,r,r,r,r,f,f",
        loc,
        ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def per_token_scales(amax, *, loc=None, ip=None):
    """``pt = amax / 2688`` and ``encode = rcp(pt)`` (0 for a zero row)."""
    result = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_f32(amax, loc, ip)],
        "{\n.reg .pred p;\n.reg .f32 e;\nmul.f32 $0, $2, 0f39C30C31;\n"
        "rcp.approx.ftz.f32 e, $0;\nsetp.ne.f32 p, $0, 0f00000000;\n"
        "selp.f32 $1, e, 0f00000000, p;\n}",
        "=f,=f,f",
        loc,
        ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@cute.jit
def local_amax(words) -> Float32:
    """Amax of eight BF16 values."""
    pair = max_bf16x2(
        abs_max_bf16x2(words[0], words[1]), abs_max_bf16x2(words[2], words[3])
    )
    return bf16x2_max_to_f32(pair)


@cute.jit
def warp_max(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = max_f32(value, cute.arch.shuffle_sync_bfly(value, 1 << shift))
    return value


# ---- routing


@dsl_user_op
def sigmoid_biased(logit, bias, *, loc=None, ip=None):
    """``biased = 1 / (1 + exp(-logit)) + bias`` and its selection key (an
    order-preserving unsigned key, larger value, larger key)."""
    e = cute.math.exp(-Float32(logit), fastmath=False)
    result = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        [_f32(e, loc, ip), _f32(bias, loc, ip)],
        "{\n.reg .pred p;\n.reg .f32 d;\n.reg .b32 b, n;\n"
        "add.f32 d, $2, 0f3F800000;\nrcp.rn.f32 d, d;\nadd.f32 $0, d, $3;\n"
        "mov.b32 b, $0;\nsetp.lt.s32 p, b, 0;\nnot.b32 n, b;\n"
        "or.b32 b, b, 0x80000000;\nselp.b32 $1, n, b, p;\n}",
        "=f,=r,f,f",
        loc,
        ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def sub_f32(a, b, *, loc=None, ip=None) -> Float32:
    operands = [_f32(a, loc, ip), _f32(b, loc, ip)]
    return Float32(_asm(T.f32(), operands, "sub.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def routing_weight(score, scale, total, *, loc=None, ip=None) -> Float32:
    """``score * scale / (total + 1e-20)``, round to nearest."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(score, loc, ip), _f32(scale, loc, ip), _f32(total, loc, ip)],
            "{\n.reg .f32 n, d;\nmul.f32 n, $1, $2;\nadd.f32 d, $3, 0f1E3CE508;\n"
            "div.rn.f32 $0, n, d;\n}",
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
    for j in cutlass.range_constexpr(1, PER_LANE):
        if Uint32(key[j]) > best_key:
            best_key = Uint32(key[j])
            best_j = Int32(j)
    return best_key, best_j


@cute.jit
def select_row(logit: cute.Tensor, bias: cute.Tensor, lane: Int32):
    """The top-4 of one row (expert = lane + 32 j), in selection order: each
    round takes the warp's largest key with one ``redux.sync`` max, then the
    lowest expert id holding it with one min; only the winning lane rescans.
    Lane ``k < 4`` gets slot ``k``: its expert and recovered score."""
    biased = cute.make_rmem_tensor((PER_LANE,), Float32)
    key = cute.make_rmem_tensor((PER_LANE,), Uint32)
    for j in cutlass.range_constexpr(PER_LANE):
        biased[j], key[j] = sigmoid_biased(logit[j], bias[j])
    best_key, best_j = lane_best(key)
    out_id = Int32(-1)
    out_score = Float32(0.0)
    for slot in cutlass.range_constexpr(TOP_K):
        top = Uint32(cute.arch.warp_redux_sync(best_key, "umax"))
        mine = Uint32(0xFFFFFFFF)
        if best_key == top:
            mine = Uint32(lane + 32 * best_j)
        winner = Int32(cute.arch.warp_redux_sync(mine, "umin"))
        won = Float32(0.0)
        if winner == lane + 32 * best_j:
            for j in cutlass.range_constexpr(PER_LANE):
                if best_j == j:
                    won = sub_f32(biased[j], bias[j])
                    key[j] = Uint32(0)  # below every value's key
            best_key, best_j = lane_best(key)
        score = cute.arch.shuffle_sync(won, winner & 31)
        if lane == slot:
            out_id = winner
            out_score = score
    return out_id, out_score


class LowBatchFront:
    """Routing, routing record and per-token NVFP4 input (module docstring),
    for calls of up to ``max_rows`` rows."""

    def __init__(self, max_rows: int) -> None:
        assert max_rows <= MAX_ROWS
        self.max_rows = max_rows
        self.levels = levels_for(max_rows)

    @cute.jit
    def __call__(
        self,
        logits_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        scale: Float32,
        x_ptr: cute.Pointer,
        weights_ptr: cute.Pointer,
        x_quant_ptr: cute.Pointer,
        x_scale_ptr: cute.Pointer,
        x_pt_ptr: cute.Pointer,
        permuted_ptr: cute.Pointer,
        record_ptr: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """``logits``: FP32 [rows, 128]; ``bias``: FP32 [128]; ``x``: BF16
        [rows, 6144] as words; outputs: ``weights`` BF16 bits [rows, 4] (as
        Uint16), ``x_quant`` [rows, 3072] packed E2M1 as words, ``x_scale``
        [rows, 384] E4M3, ``x_pt`` FP32 [rows], ``permuted`` int32 [rows, 4]
        and the record. rows in [1, max_rows]."""
        slots = cute.make_layout(rows * TOP_K)
        self.kernel(
            cute.make_tensor(logits_ptr, cute.make_layout(rows * EXPERTS)),
            cute.make_tensor(bias_ptr, cute.make_layout(EXPERTS)),
            scale,
            cute.make_tensor(x_ptr, cute.make_layout(rows * HIDDEN // 2)),
            cute.make_tensor(weights_ptr, slots),
            cute.make_tensor(x_quant_ptr, cute.make_layout(rows * HIDDEN // 8)),
            cute.make_tensor(x_scale_ptr, cute.make_layout(rows * HIDDEN // SF_VEC)),
            cute.make_tensor(x_pt_ptr, cute.make_layout(rows)),
            cute.make_tensor(permuted_ptr, slots),
            cute.make_tensor(
                record_ptr,
                cute.make_layout(
                    record_words(rows, TOP_K, EXPERTS)
                    + tile_words(max_tiles(rows, TOP_K, EXPERTS))
                ),
            ),
            rows,
        ).launch(
            grid=(1 + rows, 1, 1), block=[THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,
        bias: cute.Tensor,
        scale: Float32,
        x: cute.Tensor,
        weights: cute.Tensor,
        x_quant: cute.Tensor,
        x_scale: cute.Tensor,
        x_pt: cute.Tensor,
        permuted: cute.Tensor,
        record: cute.Tensor,
        rows: Int32,
    ):
        """CTA 0 routes; CTA ``1 + r`` quantizes row ``r``."""
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()

        levels = self.levels
        bitmap_words = present_words(levels)
        mask_size = MAX_EXPERTS * mask_words_for(levels)
        selected = self.max_rows * TOP_K

        @cute.struct
        class Shared:
            present: cute.struct.MemRange[Uint32, bitmap_words]
            rows_of: cute.struct.MemRange[Uint32, mask_size]
            ids: cute.struct.MemRange[Int32, selected]
            warp_amax: cute.struct.MemRange[Float32, THREADS // 32]

        shared = cutlass.utils.SmemAllocator().allocate(Shared)
        present = _view(shared.present, bitmap_words)
        rows_of = _view(shared.rows_of, mask_size)
        ids = _view(shared.ids, selected)
        warp_amax = _view(shared.warp_amax, THREADS // 32)
        if cta != 0:
            cute.arch.griddepcontrol_wait()
            quantize_row(x, cta - 1, thread, x_quant, x_scale, x_pt, warp_amax, HIDDEN)
        else:
            clear_scratch(present, rows_of, thread, THREADS, levels)
            row_bias = cute.make_rmem_tensor((PER_LANE,), Float32)
            if thread // 32 < rows:
                for j in cutlass.range_constexpr(PER_LANE):
                    row_bias[j] = bias[thread % 32 + 32 * j]
            cute.arch.griddepcontrol_wait()
            zero_record(record, rows, TOP_K, EXPERTS, thread, THREADS)
            # The tiles' FC1 item counts, after the record.
            counts = record_words(rows, TOP_K, EXPERTS)
            for word in cutlass.range(
                thread, tile_words(max_tiles(rows, TOP_K, EXPERTS)), THREADS, unroll=1
            ):
                record[counts + word] = Int32(0)
            for row in cutlass.range(thread // 32, rows, THREADS // 32, unroll=1):
                route_row(logits, row_bias, scale, weights, ids, row, thread % 32)
            cute.arch.sync_threads()
            build_record(
                present,
                rows_of,
                ids,
                permuted,
                record,
                rows,
                Int32(TOP_K),
                Int32(EXPERTS),
                thread,
                THREADS,
                levels,
            )


def _view(member, size: int) -> cute.Tensor:
    return cute.make_tensor(member.data_ptr(), cute.make_layout(size))


@cute.jit
def route_row(
    logits: cute.Tensor,
    bias: cute.Tensor,
    scale: Float32,
    weights: cute.Tensor,
    ids: cute.Tensor,
    row: Int32,
    lane: Int32,
):
    """One warp: the top-4 and BF16 weights of ``row``, the ids also to
    shared memory for the routing record."""
    logit = cute.make_rmem_tensor((PER_LANE,), Float32)
    for j in cutlass.range_constexpr(PER_LANE):
        logit[j] = logits[row * EXPERTS + lane + 32 * j]
    expert, score = select_row(logit, bias, lane)
    total = score
    if lane >= TOP_K:
        total = Float32(0.0)
    for shift in cutlass.range_constexpr(2):
        total = add_f32(total, cute.arch.shuffle_sync_bfly(total, 1 << shift))
    weight = cutlass.Uint16(f32_to_bf16_bits(routing_weight(score, scale, total)))
    if lane < TOP_K:
        weights[row * TOP_K + lane] = weight
        ids[row * TOP_K + lane] = expert


@cute.jit
def quantize_row(
    values: cute.Tensor,
    row: Int32,
    thread: Int32,
    codes_out: cute.Tensor,
    scales_out: cute.Tensor,
    pt_out: cute.Tensor,
    warp_amax: cute.Tensor,
    width: cutlass.Constexpr[int],
):
    """Per-token NVFP4 of BF16 row ``row`` (``values`` as words) by a CTA of
    ``width / 8`` threads; linear scales."""
    words = load_global_u32x4(values.iterator + row * (width // 2) + thread * 4)
    amax = local_amax(words)
    row_amax = warp_max(amax)
    if cute.arch.lane_idx() == 0:
        warp_amax[thread // 32] = row_amax
    cute.arch.sync_threads()
    row_amax = Float32(warp_amax[0])
    for w in cutlass.range_constexpr(1, width // 8 // 32):
        row_amax = max_f32(row_amax, warp_amax[w])
    pt, encode = per_token_scales(row_amax)
    block_amax = max_f32(amax, cute.arch.shuffle_sync_bfly(amax, 1))
    codes, sf = quantize_x8(words, block_amax, encode)
    codes_out[row * (width // 8) + thread] = codes
    if thread % 2 == 0:
        scales_out[row * (width // SF_VEC) + thread // 2] = Uint8(sf)
    if thread == 0:
        pt_out[row] = pt
