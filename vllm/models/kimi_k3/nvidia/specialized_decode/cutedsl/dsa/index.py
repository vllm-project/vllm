# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer inputs of one decode step, one 256-thread CTA per token:

- the indexer query [32, 128] (the BF16 wq_b projection of the normalized q_c) in FP8
  with one power-of-two scale per head, the scale folded into the head weights
  wv * scale / sqrt(32 * 128) as indexer_q_quant_fold_weights computes them, in the
  BF16 SM100's MQA logits take (mqa_logits_mx_inputs);
- the shared indexer key: LayerNorm (FP32) of the wk projection, BF16, then FP8 with
  a power-of-two scale into the indexer cache, as indexer_k_norm_quant_insert writes
  it: per block, the FP8 keys of its tokens, then their scales: FP32, or (`mx`: the
  SM100 MQA logits kernels' MX scales, mqa_logits_use_mx_scales) the scale's
  exponent as four equal packed UE8M0 bytes.

A half-warp owns one head (8 columns per lane); warp 0's low half also owns the key.

MXFP4 indexer cache (68 B/token: `cache.shape[2] == FP4_ENTRY`, configured with
indexer_kv_dtype='mxfp4', kimi_k3_indexer_uses_fp4): the query and the key are MXFP4
as the writers of nvidia/ops/dsa_indexer_mxfp4.py make them (_mxfp4_quant: one
UE8M0 scale per 32 values, E2M1 round to nearest even with saturation), the head
weights only take the indexer weight scale (the MX scales stay in the logits MMA),
and the cache block is [64 x 64 B E2M1 | 64 x 4 UE8M0]. The FP4 query lives in the
FP8 query's rows (FP4_QUERY_ROW bytes per row): [32 heads x 64 B E2M1 | 32 heads x
4 UE8M0] (fp4_query_views).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Uint32

from ..common.primitives import (
    asm,
    at,
    float_bits,
    launch,
    load_bf16x8,
    pack_e2m1,
    pack_fp8,
    store_words,
)
from ..mla.query import half_warp_max, pow2_ceil

HEADS, DIM = 32, 128
FP4_ENTRY = DIM // 2 + DIM // 32  # MXFP4 indexer cache bytes per token
FP4_SCALES = HEADS * DIM // 2  # byte offset of an FP4 query row's scales
FP4_WEIGHT = (HEADS * DIM) ** -0.5  # stock's weight scale (exact power of two)


def is_fp4_cache(cache) -> bool:
    """Whether an indexer cache (torch or static cute shape) holds MXFP4 keys."""
    return cache.shape[-1] == FP4_ENTRY


def fp4_query_views(q_rows: torch.Tensor, rows: int):
    """The MXFP4 query in the FP8 query buffer `q_rows` [>= rows, 32, 128] uint8:
    (E2M1 [rows, 32, 64] uint8, UE8M0 x4 [rows, 32] int32), row stride 4 KiB."""
    pitch = HEADS * DIM
    base = q_rows.reshape(-1)
    packed = torch.as_strided(base, (rows, HEADS, DIM // 2), (pitch, DIM // 2, 1))
    words = base.view(torch.int32)
    scales = torch.as_strided(
        words,
        (rows, HEADS),
        (pitch // 4, 1),
        storage_offset=words.storage_offset() + FP4_SCALES // 4,
    )
    return packed, scales


@asm((Float32, Uint32), pure=True)
def mxfp4_scale(amax: Float32):
    """(1 / scale, UE8M0 byte) of an MXFP4 block with absolute maximum `amax`:
    scale = 2^clamp(ceil(log2(max(amax, 6 * 2^-126) / 6)), -127, 127), by the
    instructions Triton emits for the MXFP4 writer (ops/dsa_indexer_mxfp4._mxfp4_quant):
    div.full, libdevice log2f, cvt.rpi, ex2.approx, div.full. Bit-exact with it,
    rounding of the log2 near powers of two included."""
    code = (
        "{ .reg .f32 a,d,t,x,e,c,l,m,p,q; .reg .b32 xi,ki,si; .reg .pred p1,p2,p3; "
        "max.f32 a,$2,0f01C00000; div.full.f32 d,a,0f40C00000; "
        "setp.lt.f32 p1,d,0f00800000; mul.f32 t,d,0f4B000000; "
        "selp.f32 x,t,d,p1; selp.f32 e,0fC1B80000,0f00000000,p1; "
        "mov.b32 xi,x; add.s32 ki,xi,-1060439283; and.b32 ki,ki,-8388608; "
        "sub.s32 si,xi,ki; cvt.rn.f32.s32 c,ki; fma.rn.ftz.f32 l,c,0f34000000,e; "
        "mov.b32 m,si; add.f32 m,m,0fBF800000; "
        "fma.rn.ftz.f32 p,0f3DC6B27F,m,0fBE2C7F30; "
        "fma.rn.ftz.f32 p,p,m,0f3E2FCF2A; fma.rn.ftz.f32 p,p,m,0fBE374E43; "
        "fma.rn.ftz.f32 p,p,m,0f3E520BF4; fma.rn.ftz.f32 p,p,m,0fBE763C8B; "
        "fma.rn.ftz.f32 p,p,m,0f3E93BF99; fma.rn.ftz.f32 p,p,m,0fBEB8AA49; "
        "fma.rn.ftz.f32 p,p,m,0f3EF6384A; fma.rn.ftz.f32 p,p,m,0fBF38AA3B; "
        "mul.f32 p,m,p; mul.f32 p,m,p; fma.rn.ftz.f32 p,m,0f3FB8AA3B,p; "
        "add.f32 l,l,p; fma.rn.ftz.f32 q,x,0f7F800000,0f7F800000; "
        "setp.lt.u32 p2,xi,2139095040; selp.f32 l,l,q,p2; "
        "setp.eq.f32 p3,x,0f00000000; selp.f32 l,0fFF800000,l,p3; "
        "cvt.rpi.ftz.f32.f32 l,l; max.f32 l,l,0fC2FE0000; min.f32 l,l,0f42FE0000; "
        "ex2.approx.ftz.f32 t,l; div.full.f32 $0,0f3F800000,t; "
        "add.f32 l,l,0f42FE0000; cvt.rzi.u32.f32 $1,l; }"
    )
    return code, "=f,=r,f", (amax,)


@asm(Uint32, pure=True)
def e2m1x2(low: Float32, high: Float32):
    """One byte of two E2M1 (round to nearest even, satfinite), `low` in the low
    nibble."""
    code = "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }"
    return code, "=r,f,f", (low, high)


@asm()
def store_b16(pointer, value):
    code = "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.b16 [$0], t; }"
    return code, "l,r,~{memory}", (pointer, Uint32(value))


@asm()
def store_b8(pointer, value):
    code = "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.b8 [$0], t; }"
    return code, "l,r,~{memory}", (pointer, Uint32(value))


@cute.jit
def group_max(value: Float32, lanes: cutlass.Constexpr[int]):
    """Max over aligned groups of `lanes` lanes."""
    for shift in cutlass.range_constexpr(lanes.bit_length() - 2, -1, -1):
        value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, 1 << shift))
    return value


@cute.jit
def gather_bytes(byte: Uint32, lane, stride: cutlass.Constexpr[int]) -> Uint32:
    """The word of the bytes of lanes lane, lane + stride, + 2 stride, + 3 stride
    (the first lane's in the low byte; valid in that first lane)."""
    word = byte
    for i in cutlass.range_constexpr(1, 4):
        word |= cute.arch.shuffle_sync(byte, lane + i * stride) << Uint32(8 * i)
    return word


@cute.jit
def half_warp_sum(value: Float32):
    for shift in cutlass.range_constexpr(3, -1, -1):
        value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
    return value


class DsaIndex:
    """`key`/`weight`: column offsets of the wk and wv projections in the packed
    projection row; `mx`: key scales as packed UE8M0 (MX) bytes."""

    def __init__(self, key: int, weight: int, mx: bool = False):
        self.key, self.weight, self.mx = key, weight, mx

    @cute.jit
    def __call__(self, *args):
        batch = args[1].shape[0]
        launch(self.kernel, args, (batch, 1, 1), 256)

    @cute.kernel
    def kernel(self, args):
        projected, iq, norm_w, norm_b, cache, slots, q_out, w_out, eps = args
        tid, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        half, column = lane // 16, (lane % 16) * 8
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(is_fp4_cache(cache)):
            row = at(q_out, token, 0, 0)
            for p in cutlass.range_constexpr(2):
                head = p * 16 + warp * 2 + half
                values = load_bf16x8(at(iq, token, head * DIM + column), mutable=True)
                quantize_fp4_query(values, 8, row, head, lane % 16, lane)
                if lane % 16 == 0:
                    wv = Float32(projected[token, self.weight + head])
                    w_out[token, head] = BFloat16(wv * Float32(FP4_WEIGHT))
        else:
            self.fp8_query(projected, iq, q_out, w_out, token, warp, lane)
        if warp == 0:
            key = projected, self.key, norm_w, norm_b, cache, slots, token, eps
            insert_key(*key, self.mx, lane)

    @cute.jit
    def fp8_query(self, projected, iq, q_out, w_out, token, warp, lane):
        half, column = lane // 16, (lane % 16) * 8
        for p in cutlass.range_constexpr(2):
            head = p * 16 + warp * 2 + half
            values = load_bf16x8(at(iq, token, head * DIM + column), mutable=True)
            amax = Float32(1e-10)
            for i in cutlass.range_constexpr(8):
                amax = cute.arch.fmax(amax, abs(values[i]))
            amax = half_warp_max(amax)
            scale = pow2_ceil(amax / Float32(448))  # amax >= 1e-10
            inverse = Float32(1) / scale
            words = [
                pack_fp8(tuple(values[4 * w + i] * inverse for i in range(4)))
                for w in range(2)
            ]
            store_words(at(q_out, token, head, column), tuple(words))
            if lane % 16 == 0:
                wv = Float32(projected[token, self.weight + head])
                fold = (wv * scale) * Float32((HEADS * DIM) ** -0.5)
                w_out[token, head] = BFloat16(fold)


@cute.jit
def quantize_fp4_query(values, per_lane: cutlass.Constexpr[int], row, head, part, lane):
    """MXFP4 of `per_lane` (4 or 8) consecutive columns of one head, columns
    `part * per_lane` on, into the FP4 query row at pointer `row`: E2M1 at
    head * 64, the head's four UE8M0 bytes at FP4_SCALES + head * 4 (written by
    the lane of part 0; the head's parts are consecutive aligned lanes)."""
    group = 32 // per_lane  # lanes of one 32-value block
    amax = Float32(0)
    for i in cutlass.range_constexpr(per_lane):
        amax = cute.arch.fmax(amax, abs(values[i]))
    amax = group_max(amax, group)
    inverse, byte = mxfp4_scale(amax)
    scaled = [values[i] * inverse for i in range(per_lane)]
    scaled += [Float32(0)] * (8 - per_lane)
    word = pack_e2m1(tuple(scaled))
    data = row + head * (DIM // 2) + part * (per_lane // 2)
    if cutlass.const_expr(per_lane == 8):
        store_words(data, (word,))
    else:
        store_b16(data, word)
    word = gather_bytes(byte, lane, group)
    if part == 0:
        store_words(row + FP4_SCALES + head * 4, (word,))


@cute.jit
def insert_key(
    projected,
    key: cutlass.Constexpr[int],
    norm_w,
    norm_b,
    cache,
    slots,
    token,
    eps,
    mx: cutlass.Constexpr[bool],
    lane,
):
    """One warp: LayerNorm (FP32) of row `token`'s wk projection (at column
    `key`), BF16, FP8 with a power-of-two scale into its indexer cache slot (the
    low half-warp writes; `mx`: packed UE8M0 scales). Nothing for a slot < 0."""
    column = (lane % 16) * 8
    slot = slots[token]
    if slot >= 0:
        x = load_bf16x8(at(projected, token, key + column), mutable=True)
        total = Float32(0)
        for i in cutlass.range_constexpr(8):
            total += x[i]
        mean = half_warp_sum(total) / Float32(DIM)
        square = Float32(0)
        for i in cutlass.range_constexpr(8):
            square += (x[i] - mean) * (x[i] - mean)
        variance = half_warp_sum(square) / Float32(DIM)
        rstd = cute.math.rsqrt(variance + eps, fastmath=False)
        normed = cute.make_rmem_tensor(8, Float32)
        amax = Float32(0)
        for i in cutlass.range_constexpr(8):
            y = (x[i] - mean) * rstd * norm_w[column + i] + norm_b[column + i]
            normed[i] = Float32(BFloat16(y))
            amax = cute.arch.fmax(amax, abs(normed[i]))
        if cutlass.const_expr(is_fp4_cache(cache)):
            insert_fp4_key(normed, amax, cache, slot, lane)
        else:
            insert_fp8_key(normed, amax, cache, slot, lane, mx)


@cute.jit
def insert_fp4_key(normed, amax, cache, slot, lane):
    """MXFP4 of the key's 8 columns per lane into its slot (the low half-warp
    writes): E2M1 at token * 64, the four UE8M0 bytes at 64 * 64 + token * 4."""
    half = lane // 16
    amax = group_max(amax, 4)
    inverse, byte = mxfp4_scale(amax)
    word = pack_e2m1(tuple(normed[i] * inverse for i in range(8)))
    tokens = cache.shape[1]
    page, offset = slot // tokens, slot % tokens
    base = at(cache, page, 0, 0)
    scales = gather_bytes(byte, lane, 4)
    if half == 0:
        store_words(base + offset * (DIM // 2) + (lane % 16) * 4, (word,))
        if lane == 0:
            store_words(base + tokens * (DIM // 2) + offset * 4, (scales,))


@cute.jit
def insert_fp8_key(normed, amax, cache, slot, lane, mx: cutlass.Constexpr[bool]):
    half, column = lane // 16, (lane % 16) * 8
    amax = half_warp_max(amax)
    scale = pow2_ceil(cute.arch.fmax(amax, Float32(1e-4)) / Float32(448))
    inverse = Float32(1) / scale
    tokens = cache.shape[1]
    page, offset = slot // tokens, slot % tokens
    base = at(cache, page, 0, 0)
    if half == 0:
        words = [
            pack_fp8(tuple(normed[4 * w + i] * inverse for i in range(4)))
            for w in range(2)
        ]
        store_words(base + offset * DIM + column, tuple(words))
        if lane == 0:
            scales = base + tokens * DIM
            word = float_bits(scale)
            if cutlass.const_expr(mx):
                word = ((word >> 23) & 0xFF) * 0x01010101
            store_words(scales + offset * 4, (word,))
