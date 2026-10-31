# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DeepseekV4-MLA Q-norm/RoPE + KV RoPE/quant/insert, CuTe DSL port.

CUDA port of csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu (still built
for ROCm; the _dispatch module routes between the two). One warp owns
one (token, head_slot) pair, 32 lanes x 16 elements = the 512-wide head:

  slot < num_heads_q                     -> live Q  (optional RMSNorm + RoPE,
                                             zero-padded to q_head_padded)
  num_heads_q <= slot < q_head_padded    -> pad Q   (zero fill)
  slot == q_head_padded                  -> KV      (RoPE + UE8M0 FP8 quant +
                                             paged-cache insert)

RoPE is GPT-J interleaved-pair on dims [448, 512), all math fp32, cos_sin
layout [max_pos, 64] = cos || sin. Q output optionally lands in FlashMLA's
chunk-interleaved layout (32 chunks of [num_heads, 16]).

Paged-cache record layout per block of bs tokens, selected by kv_mxfp8:
  V4    (584 B/token): [0, bs*576) token data (448 fp8 + 128 bf16 each),
         then [bs*576, bs*584) UE8M0 scales, 7 real + 1 pad per token.
  V4.1  (528 B/token, kv_mxfp8): per-token inline records,
         [i*528, i*528+512) fp8 data + [i*528+512, (i+1)*528) 16 UE8M0
         scales, one per 32 dims; RoPE dims are quantized too.

The full-cache variants write a contiguous 512-wide token row in the cache's
own dtype (bf16, or per-tensor-scaled e4m3) with no Q padding.

Bit-exactness notes carried over from the .cu: the KV branch double-rounds
through bf16 before the absmax, the warp reductions use the same shuffle
order (fp addition is not associative), and ceil(log2(absmax/448)) is
computed from the float's exponent field, which is exact for normal inputs.
Scalar min/max/abs/rsqrt go through cute.arch/cute.math, which lower to the
same hardware instructions as the .cu's fmaxf/fminf/fabsf/rsqrtf.

Memory access matches the .cu's width too: every per-lane group moves as
128-bit vector ops (the .cu's uint4 loads/stores and float4 cos/sin loads).
Host-side views expose each lane's group as a statically-shaped trailing
mode, and the kernels move them with CopyG2ROp/CopyR2GOp copy atoms, so a
512-wide head is 2 loads + 2 stores per lane instead of 16 + 16 scalars.
This is why the wrappers reject inputs the .cu could not have handled
either: row strides and base pointers must be 16-byte aligned where the
.cu's uint4/float4 accesses required it.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.cute_utils import recast_val
from vllm.cute_utils.cvt import (
    bf16x2_to_fp32x2,
    fp32x2_to_bf16x2,
    fp32x4_to_fp8x4,
)
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)
from vllm.platforms import current_platform

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2
ELEMS_PER_LANE = HEAD_DIM // 32
BF16_WORDS_PER_LANE = ELEMS_PER_LANE // 2
FP8_WORDS_PER_LANE = ELEMS_PER_LANE // 4
FP8_MAX = 448.0

V4_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2
V4_DATA_WORDS = V4_DATA_BYTES // 4
V4_ROPE_WORD_BASE = NOPE_DIM // 4
V4_SCALE_BYTES = NOPE_DIM // 64 + 1
V4_BYTES_PER_TOKEN = V4_DATA_BYTES + V4_SCALE_BYTES
MXFP8_DATA_BYTES = HEAD_DIM
MXFP8_DATA_WORDS = MXFP8_DATA_BYTES // 4
MXFP8_SCALE_BYTES = HEAD_DIM // 32
MXFP8_BYTES_PER_TOKEN = MXFP8_DATA_BYTES + MXFP8_SCALE_BYTES

BLOCK_SIZE = 128
WARPS_PER_BLOCK = BLOCK_SIZE // 32
# Full-cache CTA size, chosen per store dtype and launch size. Kernel-only
# microseconds on H200 (16 q heads, CUDA-graph replay):
#
#   T     bf16/128  bf16/256  fp8/128  fp8/256  .cu bf16  .cu fp8
#   128      2.32      2.45     2.54     2.40     2.87     2.73
#   512      5.53      5.64     5.03     4.50     5.70     5.05
#   1024     8.61      8.58     7.67     6.74     9.03      -
#   2048    15.11     15.33    15.76    16.99    17.20      -
#   4096    37.22     37.87    29.69    31.27    47.26    39.88
#
# bf16 rows prefer 128-thread CTAs at every size. fp8 rows, half the bytes,
# prefer 256-thread CTAs while the working set still fits in L2 (latency
# bound) and 128-thread CTAs once the launch spills to DRAM.
FULL_CACHE_BLOCK_SIZE = 128
FULL_CACHE_FP8_L2_BLOCK_SIZE = 256
FULL_CACHE_FP8_L2_TOKENS = 2048

_DUMMY_TENSORS: dict[tuple[Any, ...], torch.Tensor] = {}


def _dummy_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    tensor = _DUMMY_TENSORS.get(key)
    if tensor is None:
        tensor = torch.empty((1,), dtype=dtype, device=device)
        _DUMMY_TENSORS[key] = tensor
    return tensor


def _dummy_q_words(
    device: torch.device, heads: int, interleaved: bool, group_words: int
) -> torch.Tensor:
    # Stand-in for an unused per-lane word view. The head count is real so
    # the JIT's static strides match the compiled specialization; a
    # zero-head launch is floored at one head exactly like its warmup fake.
    h = max(heads, 1)
    key = (device, "qw", heads, interleaved, group_words)
    tensor = _DUMMY_TENSORS.get(key)
    if tensor is None:
        if interleaved:
            tensor = torch.empty(
                (1, 32, h, group_words), dtype=torch.uint32, device=device
            )
        else:
            tensor = torch.empty(
                (1, h, 32, group_words), dtype=torch.uint32, device=device
            )
        _DUMMY_TENSORS[key] = tensor
    return tensor


def _cos_sin_word_view(cos_sin: torch.Tensor) -> torch.Tensor:
    # (max_pos, 16, 4): one float4 row per cos/sin quarter.
    return cos_sin.reshape(cos_sin.shape[0], ROPE_DIM // 4, 4)


@cute.jit
def _clamp_fp8(x: Float32) -> Float32:
    return cute.arch.fmin(cute.arch.fmax(x, Float32(-FP8_MAX)), Float32(FP8_MAX))


@cute.jit
def _warp_sum(val: Float32) -> Float32:
    # Offsets descend 16, 8, 4, 2, 1 like the .cu's warpSum; fp addition is
    # not associative, so the shuffle order is part of the numerical contract.
    for s in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, 16 >> s)
    return val


@cute.jit
def _warp4_max_abs(val: Float32) -> Float32:
    peer = cute.arch.shuffle_sync_bfly(val, 1)
    val = cute.arch.fmax(val, peer)
    peer = cute.arch.shuffle_sync_bfly(val, 2)
    return cute.arch.fmax(val, peer)


@cute.jit
def _warp2_max_abs(val: Float32) -> Float32:
    return cute.arch.fmax(val, cute.arch.shuffle_sync_bfly(val, 1))


@cute.jit
def _decode_words(words: cute.Tensor, elems: cute.Tensor) -> None:
    # bf16x2_to_fp32x2 returns the low half first: (even element, odd element).
    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
        even, odd = bf16x2_to_fp32x2(words[w])
        elems[2 * w] = even
        elems[2 * w + 1] = odd


@dsl_user_op
def _q_tail_store_bf16(
    q: cute.Tensor,
    token: Int32,
    slot: Int32,
    lane: Int32,
    elems: cute.Tensor,
    row_words: int,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Q-path bf16 tail: encode this lane's 16 floats and store both 128-bit
    halves in a single inline-asm block.

    The eight ``cvt.rn.bf16x2.f32`` are F2FP ops; with cute.copy the store
    address folds at trace time, so nothing overlaps the converts and they
    bunch up at the warp tail. Interleaving the dependent store-address
    chain among the converts gives ptxas the same latency-hiding structure
    the CUDA kernel gets from its runtime-division address math. The
    converts and stores are the same instructions cute.copy would emit
    (verified bit-exact), just scheduled around the address math.

    q has the (rows, heads, 32, 8) word layout from _q_word_views; the byte
    address of (token, slot, lane, word 0) is
    (token * row_words + slot * 256 + lane * 8) * 4, where row_words is the
    token stride in words (HQ * 256). In ``cvt.rn.bf16x2.f32 d, a, b`` a
    lands in the upper 16 bits and b in the lower, so word w packs
    elems[2w+1] high and elems[2w] low.
    """
    ptr = q.iterator.to_llvm_ptr(loc=loc, ip=ip)
    addr = llvm.ptrtoint(T.i64(), ptr, loc=loc, ip=ip)
    ins = [
        addr,
        token.ir_value(loc=loc, ip=ip),
        slot.ir_value(loc=loc, ip=ip),
        lane.ir_value(loc=loc, ip=ip),
    ]
    ins += [elems[i].ir_value(loc=loc, ip=ip) for i in range(ELEMS_PER_LANE)]
    asm = (
        "{\n\t"
        ".reg .b32 t0, t1, w0, w1, w2, w3, w4, w5, w6, w7;\n\t"
        ".reg .b64 d0, d1;\n\t"
        "shl.b32 t0, $3, 8;\n\t"
        "cvt.rn.bf16x2.f32 w0, $6, $5;\n\t"
        "cvt.u64.u32 d0, t0;\n\t"
        "cvt.rn.bf16x2.f32 w1, $8, $7;\n\t"
        f"mad.wide.u32 d1, $2, {row_words}, d0;\n\t"
        "cvt.rn.bf16x2.f32 w2, $10, $9;\n\t"
        "shl.b32 t1, $4, 3;\n\t"
        "cvt.rn.bf16x2.f32 w3, $12, $11;\n\t"
        "cvt.u64.u32 d0, t1;\n\t"
        "cvt.rn.bf16x2.f32 w4, $14, $13;\n\t"
        "add.u64 d1, d1, d0;\n\t"
        "cvt.rn.bf16x2.f32 w5, $16, $15;\n\t"
        "shl.b64 d1, d1, 2;\n\t"
        "cvt.rn.bf16x2.f32 w6, $18, $17;\n\t"
        "add.u64 d1, d1, $1;\n\t"
        "cvt.rn.bf16x2.f32 w7, $20, $19;\n\t"
        "st.global.v4.b32 [d1], {w0, w1, w2, w3};\n\t"
        "st.global.v4.b32 [d1+16], {w4, w5, w6, w7};\n\t"
        "mov.u32 $0, w0;\n\t"
        "}\n"
    )
    out = llvm.inline_asm(
        T.i32(),
        ins,
        asm,
        "=r,l,r,r,r" + ",f" * ELEMS_PER_LANE,
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@cute.jit
def _encode_words(elems: cute.Tensor, words: cute.Tensor) -> None:
    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
        words[w] = fp32x2_to_bf16x2(elems[2 * w], elems[2 * w + 1])


@cute.jit
def _round_to_bf16(elems: cute.Tensor) -> None:
    # KV quant must read the bf16-rounded values the unfused norm-then-insert
    # pipeline would have written and read back: two roundings, not one.
    for i in cutlass.range_constexpr(ELEMS_PER_LANE):
        w = fp32x2_to_bf16x2(elems[i], elems[i])
        _, lo = bf16x2_to_fp32x2(w)
        elems[i] = lo


@cute.jit
def _apply_q_norm(elems: cute.Tensor, eps: Float32) -> None:
    sum_sq = Float32(0.0)
    for i in cutlass.range_constexpr(ELEMS_PER_LANE):
        sum_sq = sum_sq + elems[i] * elems[i]
    sum_sq = _warp_sum(sum_sq)
    rcp = cute.math.rsqrt(sum_sq / Float32(HEAD_DIM) + eps, fastmath=True)
    for i in cutlass.range_constexpr(ELEMS_PER_LANE):
        elems[i] = elems[i] * rcp


@cute.jit
def _apply_kv_norm(
    elems: cute.Tensor, dim_base: Int32, eps: Float32, norm_w: cute.Tensor
) -> None:
    sum_sq = Float32(0.0)
    for i in cutlass.range_constexpr(ELEMS_PER_LANE):
        sum_sq = sum_sq + elems[i] * elems[i]
    sum_sq = _warp_sum(sum_sq)
    inv_rms = Float32(1.0) / cute.math.sqrt(sum_sq / Float32(HEAD_DIM) + eps)
    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
        even_w, odd_w = bf16x2_to_fp32x2(norm_w[dim_base // 2 + w])
        elems[2 * w] = elems[2 * w] * inv_rms * even_w
        elems[2 * w + 1] = elems[2 * w + 1] * inv_rms * odd_w
    _round_to_bf16(elems)


@cute.jit
def _load_cos_sin(
    cos_sin: cute.Tensor,
    pos: Int64,
    dim_base: Int32,
    cs_atom: cute.CopyAtom,
    cos_v: cute.Tensor,
    sin_v: cute.Tensor,
) -> None:
    # cos_sin is viewed (max_pos, 16, 4): rows 0..7 hold cos, 8..15 sin, one
    # float4 per row. On rope lanes half_base is a multiple of 8, so this
    # lane's 8 cos and 8 sin values are exactly two rows each — the .cu's
    # four float4 loads. `& 7` is the identity on those lanes and parks every
    # other lane on a valid stand-in row, so all 32 lanes can issue the loads
    # up front — before the q/kv load and the RMSNorm reduction — instead of
    # paying the latency inside the rope branch.
    half_base = (dim_base - NOPE_DIM) >> 1
    hb4 = (half_base >> 2) & Int32(7)
    for j in cutlass.range_constexpr(2):
        cute.copy(cs_atom, cos_sin[pos, Int32(hb4 + j), None], cos_v[j, None])
        cute.copy(
            cs_atom,
            cos_sin[pos, Int32(HALF_ROPE // 4 + hb4 + j), None],
            sin_v[j, None],
        )


@cute.jit
def _apply_rope(elems: cute.Tensor, cos_v: cute.Tensor, sin_v: cute.Tensor) -> None:
    # cos_v/sin_v carry this lane's 8 cos and 8 sin values, preloaded by
    # _load_cos_sin; only rope lanes call this.
    for p in cutlass.range_constexpr(ELEMS_PER_LANE // 2):
        c = cos_v[p // 4, p % 4]
        s = sin_v[p // 4, p % 4]
        x_even = elems[2 * p]
        x_odd = elems[2 * p + 1]
        elems[2 * p] = x_even * c - x_odd * s
        elems[2 * p + 1] = x_even * s + x_odd * c


@cute.jit
def _ue8m0_scale(absmax: Float32) -> tuple[Int32, Float32]:
    # ceil(log2(absmax / FP8_MAX)) without libm: for normal y > 0 the biased
    # exponent is exactly floor(log2(y)), so the ceil is that plus one iff the
    # mantissa is nonzero. exp2 of an integer exponent is then an exact power
    # of two via bitcast. absmax is floored at 1e-4 upstream, so y stays
    # normal.
    y = absmax / Float32(FP8_MAX)
    bits = recast_val(y, Uint32)
    exponent = Int32((bits >> 23) & 0xFF) - 127
    if (bits & 0x7FFFFF) != Uint32(0):
        exponent = exponent + 1
    inv_scale = recast_val(Int32((127 - exponent) << 23), Float32)
    return exponent, inv_scale


def _paged_cache_views(
    k_cache: torch.Tensor, cache_block_size: int, kv_mxfp8: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a uint8 paged cache into its u32 data and u8 scale views.

    Accepts [num_blocks, block_bytes] 2D or [num_blocks, bs, row_bytes] 3D
    (rows must be packed: stride(1) == size(2)); the .cu addressed tokens at
    fixed byte offsets within a block, so anything else was never a valid
    input. Returns (data, scale):

      V4:    data [B, bs, 144] u32 stride (S0//4, 144, 1),
             scale [B, bs, 8] u8 stride (S0, 8, 1)    (tail after bs*576)
      mxfp8: data [B, bs, 128] u32 stride (S0//4, 132, 1),
             scale [B, bs, 16] u8 stride (S0, 528, 1) (inline per token)
    """
    row_bytes = MXFP8_BYTES_PER_TOKEN if kv_mxfp8 else V4_BYTES_PER_TOKEN
    block_bytes = cache_block_size * row_bytes
    if k_cache.dim() == 2:
        flat = k_cache
        if flat.stride(1) != 1:
            raise ValueError("k_cache rows must be byte-contiguous (stride 1)")
    elif k_cache.dim() == 3:
        if k_cache.stride(2) != 1:
            raise ValueError("k_cache rows must be byte-contiguous (stride 1)")
        if k_cache.stride(1) != k_cache.shape[2]:
            raise ValueError(
                "k_cache rows must be packed (stride(1) == size(2)); the "
                "record layout is addressed at fixed offsets within a block"
            )
        flat = k_cache.view(k_cache.shape[0], -1)
    else:
        raise ValueError(f"k_cache must be 2D or 3D, got {k_cache.dim()}D")
    if flat.shape[1] < block_bytes:
        raise ValueError(
            f"k_cache row width {flat.shape[1]} is smaller than one "
            f"{cache_block_size}-token block ({block_bytes} bytes)"
        )
    if flat.stride(0) % 16 != 0:
        # The .cu's uint4 stores (and this port's 128-bit copies) require it.
        raise ValueError("k_cache block stride must be 16-byte aligned")
    num_blocks = flat.shape[0]
    if kv_mxfp8:
        records = flat[:, :block_bytes].view(
            num_blocks, cache_block_size, MXFP8_BYTES_PER_TOKEN
        )
        data = records[:, :, :MXFP8_DATA_BYTES].view(torch.uint32)
        scale = records[:, :, MXFP8_DATA_BYTES:]
    else:
        data = (
            flat[:, : cache_block_size * V4_DATA_BYTES]
            .view(num_blocks, cache_block_size, V4_DATA_BYTES)
            .view(torch.uint32)
        )
        scale = flat[
            :,
            cache_block_size * V4_DATA_BYTES : cache_block_size * V4_BYTES_PER_TOKEN,
        ].view(num_blocks, cache_block_size, V4_SCALE_BYTES)
    return data, scale


def _cache_word_views(
    data: torch.Tensor, kv_mxfp8: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a (blocks, bs, words) u32 cache-data view into the per-lane
    group views the kernel copies through: 4-word groups for the fp8 stores
    and 8-word groups for the V4 bf16 rope tail. Under mxfp8 the 8-word view
    is a stand-in that no store ever targets."""
    num_blocks, block_slots = data.shape[0], data.shape[1]
    stride0 = data.stride(0)
    if kv_mxfp8:
        fp8 = data.as_strided(
            (num_blocks, block_slots, MXFP8_DATA_WORDS // 4, 4),
            (stride0, MXFP8_BYTES_PER_TOKEN // 4, 4, 1),
        )
        bf16 = data.as_strided(
            (num_blocks, block_slots, MXFP8_DATA_WORDS // 8, 8),
            (stride0, MXFP8_DATA_WORDS, 8, 1),
        )
    else:
        fp8 = data.as_strided(
            (num_blocks, block_slots, V4_DATA_WORDS // 4, 4),
            (stride0, V4_DATA_WORDS, 4, 1),
        )
        bf16 = data.as_strided(
            (num_blocks, block_slots, V4_DATA_WORDS // 8, 8),
            (stride0, V4_DATA_WORDS, 8, 1),
        )
    return fp8, bf16


def _q_word_views(t: torch.Tensor, heads: int, interleaved: bool) -> torch.Tensor:
    """View a [rows, heads, 512] bf16 tensor as u32 words grouped per lane:
    (rows, heads, 32, 8), or (rows, 32, heads, 8) for the FlashMLA
    chunk-interleaved layout. The trailing (8,) mode is one lane's group,
    32 bytes at a 32-byte boundary."""
    rows = t.shape[0]
    words = t.view(torch.uint32)
    if interleaved:
        return words.as_strided((rows, 32, heads, 8), (heads * 256, heads * 8, 8, 1))
    return words.reshape(rows, heads, 32, 8)


def _kv_word_view(kv: torch.Tensor) -> torch.Tensor:
    words = kv.view(torch.uint32)
    return words.as_strided((words.shape[0], 32, 8), (words.stride(0), 8, 1))


def _full_cache_word_views(
    cache_u32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(blocks, bs, 512-wide rows) u32 cache -> 8-word and 4-word group
    views, each exactly covering the row. Only the one selected by the
    store dtype is ever sliced."""
    num_blocks, block_slots = cache_u32.shape[0], cache_u32.shape[1]
    stride0, row = cache_u32.stride(0), cache_u32.stride(1)
    row_words = cache_u32.shape[2]
    bf16 = cache_u32.as_strided(
        (num_blocks, block_slots, row_words // 8, 8), (stride0, row, 8, 1)
    )
    fp8 = cache_u32.as_strided(
        (num_blocks, block_slots, row_words // 4, 4), (stride0, row, 4, 1)
    )
    return fp8, bf16


def _check_common(
    op_name: str,
    kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> None:
    if not current_platform.is_cuda():
        raise RuntimeError(
            f"{op_name}: the CuTe DSL port is CUDA-only; this platform has "
            "no implementation"
        )
    if not kv.is_cuda:
        raise ValueError(f"{op_name}: kv must be a CUDA tensor")
    if kv.dtype == torch.float16:
        raise NotImplementedError(
            f"{op_name}: float16 is not ported to the CuTe DSL kernel"
        )
    if kv.dtype != torch.bfloat16:
        raise ValueError(f"{op_name}: kv must be bfloat16, got {kv.dtype}")
    if kv.dim() != 2 or kv.shape[1] != HEAD_DIM or kv.stride(1) != 1:
        raise ValueError(f"{op_name}: kv must be [tokens, {HEAD_DIM}] row-major")
    if kv.stride(0) % 8 != 0:
        # 16 bytes: the .cu's uint4 loads require it, and so do this
        # port's 128-bit vector loads.
        raise ValueError(f"{op_name}: kv row stride must be 16-byte aligned")
    if kv.data_ptr() % 16 != 0:
        raise ValueError(f"{op_name}: kv must be 16-byte aligned")
    for name, tensor in (
        ("slot_mapping", slot_mapping),
        ("position_ids", position_ids),
    ):
        if (
            tensor.dtype != torch.int64
            or tensor.dim() != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{op_name}: {name} must be contiguous 1D int64")
        if tensor.device != kv.device:
            raise ValueError(f"{op_name}: {name} must be on the KV device")
    if position_ids.shape[0] != kv.shape[0]:
        raise ValueError(
            f"{op_name}: positions must match KV rows "
            f"({position_ids.shape[0]} vs {kv.shape[0]})"
        )
    if slot_mapping.shape[0] > kv.shape[0]:
        raise ValueError(
            f"{op_name}: slot_mapping must not exceed KV rows "
            f"({slot_mapping.shape[0]} vs {kv.shape[0]})"
        )
    if (
        cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[1] != ROPE_DIM
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError(
            f"{op_name}: cos_sin_cache must be contiguous float32 [max_pos, {ROPE_DIM}]"
        )
    if cos_sin_cache.data_ptr() % 16 != 0:
        raise ValueError(f"{op_name}: cos_sin_cache must be 16-byte aligned")
    if cos_sin_cache.device != kv.device:
        raise ValueError(f"{op_name}: cos_sin_cache must be on the KV device")


def _check_scalar(
    op_name: str, name: str, scale: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError(f"{op_name}: {name} must be one float32 scalar")
    if scale.device != device:
        raise ValueError(f"{op_name}: {name} must be on the KV device")
    return scale.reshape(1)


@dataclass(frozen=True)
class PagedCompileKey:
    num_heads_q: int
    q_head_padded: int
    apply_q_norm: bool
    apply_q_rope: bool
    is_q_interleaved: bool
    kv_mxfp8: bool
    has_kv_norm: bool
    eps: float


class _PagedInsertKernel(VllmCuTeDSLJitKernel[PagedCompileKey]):
    """Paged-record variant: V4/V4.1 UE8M0 quantizing insert."""

    CompileKey = PagedCompileKey

    @staticmethod
    def kernel(compile_key: PagedCompileKey) -> Any:
        HQ = compile_key.num_heads_q
        QP = compile_key.q_head_padded
        APPLY_Q_NORM = compile_key.apply_q_norm
        APPLY_Q_ROPE = compile_key.apply_q_rope
        Q_INTERLEAVED = compile_key.is_q_interleaved
        KV_MXFP8 = compile_key.kv_mxfp8
        HAS_KV_NORM = compile_key.has_kv_norm
        EPS = Float32(compile_key.eps)
        USE_PDL = current_platform.is_arch_support_pdl()

        @cute.jit
        def process_slot(
            token: Int32,
            slot: Int32,
            lane: Int32,
            dim_base: Int32,
            q_in: cute.Tensor,
            q_out: cute.Tensor,
            kv: cute.Tensor,
            cache_fp8: cute.Tensor,
            cache_bf16: cute.Tensor,
            cache_scale: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            load_atom: cute.CopyAtom,
            store_atom: cute.CopyAtom,
            cs_atom: cute.CopyAtom,
            n_insert: Int32,
            block_size: Int32,
        ) -> None:
            if slot == Int32(QP):
                # KV slot: RoPE + quant + insert, only for the first
                # n_insert rows (with DP padding, tail rows have no slot).
                if token < n_insert:
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_wait()
                    cs_layout = cute.make_layout((2, 4), stride=(4, 1))
                    cos_v = cute.make_rmem_tensor(cs_layout, Float32)
                    sin_v = cute.make_rmem_tensor(cs_layout, Float32)
                    # Only rope lanes (dim_base >= NOPE_DIM, 4 of 32) read
                    # cos_v/sin_v; skip the 128-bit cos/sin loads elsewhere,
                    # matching the .cu's is_rope_lane guard.
                    if dim_base >= Int32(NOPE_DIM):
                        pos = pos_ids[token]
                        _load_cos_sin(cos_sin, pos, dim_base, cs_atom, cos_v, sin_v)
                    elems = cute.make_rmem_tensor((ELEMS_PER_LANE,), Float32)
                    words = cute.make_rmem_tensor((BF16_WORDS_PER_LANE,), Uint32)
                    cute.copy(load_atom, kv[token, Int32(lane), None], words[None])
                    _decode_words(words, elems)
                    if cutlass.const_expr(HAS_KV_NORM):
                        _apply_kv_norm(elems, dim_base, EPS, norm_w)
                    if dim_base >= Int32(NOPE_DIM):
                        _apply_rope(elems, cos_v, sin_v)
                    sid = slots[token]
                    if sid >= Int64(0):
                        block = sid // Int64(block_size)
                        pib = sid % Int64(block_size)
                        _round_to_bf16(elems)
                        local_max = Float32(0.0)
                        for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                            local_max = cute.arch.fmax(
                                local_max, cute.math.abs(elems[i])
                            )
                        if cutlass.const_expr(KV_MXFP8):
                            absmax = cute.arch.fmax(
                                _warp2_max_abs(local_max), Float32(1e-4)
                            )
                        else:
                            absmax = cute.arch.fmax(
                                _warp4_max_abs(local_max), Float32(1e-4)
                            )
                        exponent, inv_scale = _ue8m0_scale(absmax)
                        if cutlass.const_expr(KV_MXFP8):
                            store_fp8 = True
                        else:
                            store_fp8 = dim_base < Int32(NOPE_DIM)
                        if store_fp8:
                            packed = cute.make_rmem_tensor(
                                (FP8_WORDS_PER_LANE,), Uint32
                            )
                            for w in cutlass.range_constexpr(FP8_WORDS_PER_LANE):
                                v0 = _clamp_fp8(elems[4 * w] * inv_scale)
                                v1 = _clamp_fp8(elems[4 * w + 1] * inv_scale)
                                v2 = _clamp_fp8(elems[4 * w + 2] * inv_scale)
                                v3 = _clamp_fp8(elems[4 * w + 3] * inv_scale)
                                packed[w] = fp32x4_to_fp8x4(v0, v1, v2, v3)
                            cute.copy(
                                store_atom,
                                packed[None],
                                cache_fp8[block, pib, Int32(lane), None],
                            )
                            encoded = exponent + 127
                            if encoded > 255:
                                encoded = 255
                            if encoded < 0:
                                encoded = 0
                            if cutlass.const_expr(KV_MXFP8):
                                # One scale per 32 dims = one per lane pair.
                                if (lane & 1) == 0:
                                    cache_scale[block, pib, Int64(lane >> 1)] = Uint8(
                                        encoded
                                    )
                            else:
                                if (lane & 3) == 0:
                                    cache_scale[block, pib, Int64(lane >> 2)] = Uint8(
                                        encoded
                                    )
                                if lane == 0:
                                    # Pad byte the reader expects at index 7.
                                    cache_scale[
                                        block, pib, Int64(V4_SCALE_BYTES - 1)
                                    ] = Uint8(0)
                        else:
                            _encode_words(elems, words)
                            # The rope tail lands at words [8L-112, 8L-104)
                            # of the token row = tile L-14 of the 8-word view.
                            cute.copy(
                                store_atom,
                                words[None],
                                cache_bf16[block, pib, Int32(lane - 14), None],
                            )
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_launch_dependents()
            else:
                # Q slot: pad slots only zero-fill; live slots decode,
                # optionally norm and rope, then re-encode.
                if cutlass.const_expr(USE_PDL):
                    cute.arch.griddepcontrol_wait()
                if slot < Int32(HQ):
                    if cutlass.const_expr(APPLY_Q_ROPE):
                        cs_layout = cute.make_layout((2, 4), stride=(4, 1))
                        cos_v = cute.make_rmem_tensor(cs_layout, Float32)
                        sin_v = cute.make_rmem_tensor(cs_layout, Float32)
                        # Same rope-lane guard as the KV slot above.
                        if dim_base >= Int32(NOPE_DIM):
                            pos = pos_ids[token]
                            _load_cos_sin(cos_sin, pos, dim_base, cs_atom, cos_v, sin_v)
                    elems = cute.make_rmem_tensor((ELEMS_PER_LANE,), Float32)
                    words = cute.make_rmem_tensor((BF16_WORDS_PER_LANE,), Uint32)
                    if cutlass.const_expr(Q_INTERLEAVED):
                        cute.copy(
                            load_atom,
                            q_in[token, Int32(lane), slot, None],
                            words[None],
                        )
                    else:
                        cute.copy(
                            load_atom,
                            q_in[token, slot, Int32(lane), None],
                            words[None],
                        )
                    _decode_words(words, elems)
                    if cutlass.const_expr(APPLY_Q_NORM):
                        _apply_q_norm(elems, EPS)
                    if cutlass.const_expr(APPLY_Q_ROPE):  # noqa: SIM102
                        if dim_base >= Int32(NOPE_DIM):
                            _apply_rope(elems, cos_v, sin_v)
                    _encode_words(elems, words)
                    if cutlass.const_expr(Q_INTERLEAVED):
                        cute.copy(
                            store_atom,
                            words[None],
                            q_out[token, Int32(lane), slot, None],
                        )
                    else:
                        cute.copy(
                            store_atom,
                            words[None],
                            q_out[token, slot, Int32(lane), None],
                        )
                else:
                    # FlashMLA reads the pad slots; bf16 +0.0 is 0x0000.
                    words = cute.make_rmem_tensor((BF16_WORDS_PER_LANE,), Uint32)
                    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                        words[w] = Uint32(0)
                    if cutlass.const_expr(Q_INTERLEAVED):
                        cute.copy(
                            store_atom,
                            words[None],
                            q_out[token, Int32(lane), slot, None],
                        )
                    else:
                        cute.copy(
                            store_atom,
                            words[None],
                            q_out[token, slot, Int32(lane), None],
                        )
                if cutlass.const_expr(USE_PDL):
                    cute.arch.griddepcontrol_launch_dependents()

        @cute.kernel
        def device_kernel(
            q_in: cute.Tensor,
            q_out: cute.Tensor,
            kv: cute.Tensor,
            cache_fp8: cute.Tensor,
            cache_bf16: cute.Tensor,
            cache_scale: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            load_atom: cute.CopyAtom,
            store_atom: cute.CopyAtom,
            cs_atom: cute.CopyAtom,
            n_full: Int32,
            n_insert: Int32,
            block_size: Int32,
        ):
            bidx, _, _ = cute.arch.block_idx()
            warp = cute.arch.warp_idx()
            lane = cute.arch.lane_idx()
            gw = bidx * WARPS_PER_BLOCK + warp
            token = gw // Int32(QP + 1)
            slot = gw % Int32(QP + 1)
            dim_base = lane * ELEMS_PER_LANE
            if token < n_full:
                process_slot(
                    token,
                    slot,
                    lane,
                    dim_base,
                    q_in,
                    q_out,
                    kv,
                    cache_fp8,
                    cache_bf16,
                    cache_scale,
                    slots,
                    pos_ids,
                    cos_sin,
                    norm_w,
                    load_atom,
                    store_atom,
                    cs_atom,
                    n_insert,
                    block_size,
                )

        @cute.jit
        def host_entrypoint(
            q_in: cute.Tensor,
            q_out: cute.Tensor,
            kv: cute.Tensor,
            cache_fp8: cute.Tensor,
            cache_bf16: cute.Tensor,
            cache_scale: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            stream: CUstream,
        ):
            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), Uint32, num_bits_per_copy=128
            )
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(), Uint32, num_bits_per_copy=128
            )
            cs_atom = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), Float32, num_bits_per_copy=128
            )
            n = kv.shape[0]
            ni = slots.shape[0]
            bs = cache_fp8.shape[1]
            device_kernel(
                q_in,
                q_out,
                kv,
                cache_fp8,
                cache_bf16,
                cache_scale,
                slots,
                pos_ids,
                cos_sin,
                norm_w,
                load_atom,
                store_atom,
                cs_atom,
                n,
                ni,
                bs,
            ).launch(
                grid=(
                    (n * (QP + 1) + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK,
                    1,
                    1,
                ),
                block=(BLOCK_SIZE, 1, 1),
                stream=stream,
                use_pdl=USE_PDL,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        **compile_key_fields: Any,
    ) -> PagedCompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(self, *args: Any, **kwargs: Any) -> list[PagedCompileKey]:
        return []

    def warmup_inputs(self, compile_key: PagedCompileKey) -> tuple[Any, ...]:
        # Head counts are compile-key constants, so the q views' strides are
        # static; a stand-in for an unused q view is floored at one head.
        hq = max(compile_key.num_heads_q, 1)
        qp = max(compile_key.q_head_padded, 1)
        n = cute.sym_int()
        i = cute.sym_int()
        p = cute.sym_int()
        b = cute.sym_int()
        bs = cute.sym_int()
        qt = cute.sym_int()
        qot = cute.sym_int()
        wt = cute.sym_int()
        data_stride0 = cute.sym_int64(divisibility=4)
        scale_stride0 = cute.sym_int64(divisibility=1)
        kv_row = cute.sym_int64(divisibility=4)
        if compile_key.kv_mxfp8:
            fp8_tiles, fp8_row = MXFP8_DATA_WORDS // 4, MXFP8_BYTES_PER_TOKEN // 4
            # The bf16-group view is only sliced by the V4 rope-tail store;
            # under mxfp8 it stands in, never accessed.
            bf16_tiles, bf16_row = MXFP8_DATA_WORDS // 8, MXFP8_DATA_WORDS
            scale_bytes = MXFP8_SCALE_BYTES
            scale_row = MXFP8_BYTES_PER_TOKEN
        else:
            fp8_tiles, fp8_row = V4_DATA_WORDS // 4, V4_DATA_WORDS
            bf16_tiles, bf16_row = V4_DATA_WORDS // 8, V4_DATA_WORDS
            scale_bytes = V4_SCALE_BYTES
            scale_row = V4_SCALE_BYTES
        if compile_key.is_q_interleaved:
            q_in = cute.runtime.make_fake_tensor(
                Uint32,
                (qt, 32, hq, 8),
                stride=(hq * 256, hq * 8, 8, 1),
                assumed_align=16,
            )
            q_out = cute.runtime.make_fake_tensor(
                Uint32,
                (qot, 32, qp, 8),
                stride=(qp * 256, qp * 8, 8, 1),
                assumed_align=16,
            )
        else:
            q_in = cute.runtime.make_fake_tensor(
                Uint32, (qt, hq, 32, 8), stride=(hq * 256, 256, 8, 1), assumed_align=16
            )
            q_out = cute.runtime.make_fake_tensor(
                Uint32, (qot, qp, 32, 8), stride=(qp * 256, 256, 8, 1), assumed_align=16
            )
        kv = cute.runtime.make_fake_tensor(
            Uint32,
            (n, 32, 8),
            stride=(kv_row, 8, 1),
            assumed_align=16,
        )
        cache_fp8 = cute.runtime.make_fake_tensor(
            Uint32,
            (b, bs, fp8_tiles, 4),
            stride=(data_stride0, fp8_row, 4, 1),
            assumed_align=16,
        )
        cache_bf16 = cute.runtime.make_fake_tensor(
            Uint32,
            (b, bs, bf16_tiles, 8),
            stride=(data_stride0, bf16_row, 8, 1),
            assumed_align=16,
        )
        cache_scale = cute.runtime.make_fake_tensor(
            Uint8,
            (b, bs, scale_bytes),
            stride=(scale_stride0, scale_row, 1),
            assumed_align=4,
        )
        slots = cute.runtime.make_fake_tensor(Int64, (i,), stride=(1,), assumed_align=8)
        pos_ids = cute.runtime.make_fake_tensor(
            Int64, (n,), stride=(1,), assumed_align=8
        )
        cos_sin = cute.runtime.make_fake_tensor(
            Float32, (p, ROPE_DIM // 4, 4), stride=(ROPE_DIM, 4, 1), assumed_align=16
        )
        norm_w = cute.runtime.make_fake_tensor(
            Uint32, (wt,), stride=(1,), assumed_align=4
        )
        return (
            q_in,
            q_out,
            kv,
            cache_fp8,
            cache_bf16,
            cache_scale,
            slots,
            pos_ids,
            cos_sin,
            norm_w,
        )

    @kernel_launcher
    def __call__(
        self,
        q_in: torch.Tensor,
        q_out: torch.Tensor,
        kv: torch.Tensor,
        cache_fp8: torch.Tensor,
        cache_bf16: torch.Tensor,
        cache_scale: torch.Tensor,
        slots: torch.Tensor,
        pos_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        norm_w: torch.Tensor,
        *,
        num_heads_q: int,
        q_head_padded: int,
        apply_q_norm: bool,
        apply_q_rope: bool,
        is_q_interleaved: bool,
        kv_mxfp8: bool,
        has_kv_norm: bool,
        eps: float,
    ) -> CuTeDSLLaunchSpec[PagedCompileKey]:
        compile_key = self.dispatch(
            num_heads_q=num_heads_q,
            q_head_padded=q_head_padded,
            apply_q_norm=apply_q_norm,
            apply_q_rope=apply_q_rope,
            is_q_interleaved=is_q_interleaved,
            kv_mxfp8=kv_mxfp8,
            has_kv_norm=has_kv_norm,
            eps=eps,
        )
        return compile_key, (
            q_in,
            q_out,
            kv,
            cache_fp8,
            cache_bf16,
            cache_scale,
            slots,
            pos_ids,
            cos_sin,
            norm_w,
        )


@dataclass(frozen=True)
class FullCacheCompileKey:
    num_heads_q: int
    apply_q_norm: bool
    has_kv_norm: bool
    store_q_fp8: bool
    store_kv_fp8: bool
    eps: float
    block_size: int = FULL_CACHE_BLOCK_SIZE


class _FullCacheInsertKernel(VllmCuTeDSLJitKernel[FullCacheCompileKey]):
    """Contiguous 512-wide cache-row variant (FlashInfer full cache)."""

    CompileKey = FullCacheCompileKey

    @staticmethod
    def kernel(compile_key: FullCacheCompileKey) -> Any:
        HQ = compile_key.num_heads_q
        APPLY_Q_NORM = compile_key.apply_q_norm
        HAS_KV_NORM = compile_key.has_kv_norm
        STORE_Q_FP8 = compile_key.store_q_fp8
        STORE_KV_FP8 = compile_key.store_kv_fp8
        FC_BLOCK = compile_key.block_size
        FC_WARPS = FC_BLOCK // 32
        EPS = Float32(compile_key.eps)
        USE_PDL = current_platform.is_arch_support_pdl()

        @cute.kernel
        def device_kernel(
            q: cute.Tensor,
            q_fp8: cute.Tensor,
            kv: cute.Tensor,
            cache_fp8: cute.Tensor,
            cache_bf16: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            kv_scale: cute.Tensor,
            q_scale_inv: cute.Tensor,
            load_atom: cute.CopyAtom,
            store_atom: cute.CopyAtom,
            cs_atom: cute.CopyAtom,
            n_full: Int32,
            n_insert: Int32,
            block_size: Int32,
        ):
            bidx, _, _ = cute.arch.block_idx()
            warp = cute.arch.warp_idx()
            lane = cute.arch.lane_idx()
            gw = bidx * FC_WARPS + warp
            token = gw // Int32(HQ + 1)
            slot = gw % Int32(HQ + 1)
            dim_base = lane * ELEMS_PER_LANE
            if token < n_full:
                if slot == Int32(HQ):
                    if token < n_insert:
                        if cutlass.const_expr(USE_PDL):
                            cute.arch.griddepcontrol_wait()
                        pos = pos_ids[token]
                        cs_layout = cute.make_layout((2, 4), stride=(4, 1))
                        cos_v = cute.make_rmem_tensor(cs_layout, Float32)
                        sin_v = cute.make_rmem_tensor(cs_layout, Float32)
                        _load_cos_sin(cos_sin, pos, dim_base, cs_atom, cos_v, sin_v)
                        elems = cute.make_rmem_tensor((ELEMS_PER_LANE,), Float32)
                        words = cute.make_rmem_tensor((BF16_WORDS_PER_LANE,), Uint32)
                        cute.copy(load_atom, kv[token, Int32(lane), None], words[None])
                        _decode_words(words, elems)
                        if cutlass.const_expr(HAS_KV_NORM):
                            _apply_kv_norm(elems, dim_base, EPS, norm_w)
                        # Full-cache rows rope the KV unconditionally; this
                        # variant has no apply_q_rope knob.
                        if dim_base >= Int32(NOPE_DIM):
                            _apply_rope(elems, cos_v, sin_v)
                        sid = slots[token]
                        if sid >= Int64(0):
                            block = sid // Int64(block_size)
                            pib = sid % Int64(block_size)
                            if cutlass.const_expr(STORE_KV_FP8):
                                inv = Float32(1.0) / kv_scale[0]
                                packed = cute.make_rmem_tensor(
                                    (FP8_WORDS_PER_LANE,), Uint32
                                )
                                for w in cutlass.range_constexpr(FP8_WORDS_PER_LANE):
                                    v0 = _clamp_fp8(elems[4 * w] * inv)
                                    v1 = _clamp_fp8(elems[4 * w + 1] * inv)
                                    v2 = _clamp_fp8(elems[4 * w + 2] * inv)
                                    v3 = _clamp_fp8(elems[4 * w + 3] * inv)
                                    packed[w] = fp32x4_to_fp8x4(v0, v1, v2, v3)
                                cute.copy(
                                    store_atom,
                                    packed[None],
                                    cache_fp8[block, pib, Int32(lane), None],
                                )
                            else:
                                _encode_words(elems, words)
                                cute.copy(
                                    store_atom,
                                    words[None],
                                    cache_bf16[block, pib, Int32(lane), None],
                                )
                        if cutlass.const_expr(USE_PDL):
                            cute.arch.griddepcontrol_launch_dependents()
                else:
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_wait()
                    pos = pos_ids[token]
                    cs_layout = cute.make_layout((2, 4), stride=(4, 1))
                    cos_v = cute.make_rmem_tensor(cs_layout, Float32)
                    sin_v = cute.make_rmem_tensor(cs_layout, Float32)
                    _load_cos_sin(cos_sin, pos, dim_base, cs_atom, cos_v, sin_v)
                    elems = cute.make_rmem_tensor((ELEMS_PER_LANE,), Float32)
                    words = cute.make_rmem_tensor((BF16_WORDS_PER_LANE,), Uint32)
                    cute.copy(load_atom, q[token, slot, Int32(lane), None], words[None])
                    _decode_words(words, elems)
                    if cutlass.const_expr(APPLY_Q_NORM):
                        _apply_q_norm(elems, EPS)
                    if dim_base >= Int32(NOPE_DIM):
                        _apply_rope(elems, cos_v, sin_v)
                    if cutlass.const_expr(STORE_Q_FP8):
                        sinv = q_scale_inv[0]
                        packed = cute.make_rmem_tensor((FP8_WORDS_PER_LANE,), Uint32)
                        for w in cutlass.range_constexpr(FP8_WORDS_PER_LANE):
                            v0 = _clamp_fp8(elems[4 * w] * sinv)
                            v1 = _clamp_fp8(elems[4 * w + 1] * sinv)
                            v2 = _clamp_fp8(elems[4 * w + 2] * sinv)
                            v3 = _clamp_fp8(elems[4 * w + 3] * sinv)
                            packed[w] = fp32x4_to_fp8x4(v0, v1, v2, v3)
                        cute.copy(
                            store_atom,
                            packed[None],
                            q_fp8[token, slot, Int32(lane), None],
                        )
                    else:
                        _q_tail_store_bf16(q, token, slot, Int32(lane), elems, HQ * 256)
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_launch_dependents()

        @cute.jit
        def host_entrypoint(
            q: cute.Tensor,
            q_fp8: cute.Tensor,
            kv: cute.Tensor,
            cache_fp8: cute.Tensor,
            cache_bf16: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            kv_scale: cute.Tensor,
            q_scale_inv: cute.Tensor,
            stream: CUstream,
        ):
            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), Uint32, num_bits_per_copy=128
            )
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyR2GOp(), Uint32, num_bits_per_copy=128
            )
            cs_atom = cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), Float32, num_bits_per_copy=128
            )
            n = kv.shape[0]
            ni = slots.shape[0]
            bs = cache_fp8.shape[1]
            device_kernel(
                q,
                q_fp8,
                kv,
                cache_fp8,
                cache_bf16,
                slots,
                pos_ids,
                cos_sin,
                norm_w,
                kv_scale,
                q_scale_inv,
                load_atom,
                store_atom,
                cs_atom,
                n,
                ni,
                bs,
            ).launch(
                grid=(
                    (n * (HQ + 1) + FC_WARPS - 1) // FC_WARPS,
                    1,
                    1,
                ),
                block=(FC_BLOCK, 1, 1),
                stream=stream,
                use_pdl=USE_PDL,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        **compile_key_fields: Any,
    ) -> FullCacheCompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(self, *args: Any, **kwargs: Any) -> list[FullCacheCompileKey]:
        return []

    def warmup_inputs(self, compile_key: FullCacheCompileKey) -> tuple[Any, ...]:
        n = cute.sym_int()
        i = cute.sym_int()
        p = cute.sym_int()
        b = cute.sym_int()
        bs = cute.sym_int()
        qn = cute.sym_int()
        qfn = cute.sym_int()
        wt = cute.sym_int()
        s1 = cute.sym_int()
        s2 = cute.sym_int()
        kv_row = cute.sym_int64(divisibility=4)
        cache_stride0 = cute.sym_int64(divisibility=4)
        cache_row = cute.sym_int64(divisibility=4)
        # A Q-less launch compiles with num_heads_q == 0, but a zero shape
        # mode is not a legal CuTe layout; floor the fake at one head. The
        # grid never dispatches a Q slot when num_heads_q == 0, so the
        # stand-in is never read.
        heads = max(compile_key.num_heads_q, 1)
        q = cute.runtime.make_fake_tensor(
            Uint32,
            (qn, heads, 32, 8),
            stride=(heads * (HEAD_DIM // 2), HEAD_DIM // 2, 8, 1),
            assumed_align=16,
        )
        q_fp8 = cute.runtime.make_fake_tensor(
            Uint32,
            (qfn, heads, 32, 4),
            stride=(heads * (HEAD_DIM // 4), HEAD_DIM // 4, 4, 1),
            assumed_align=16,
        )
        kv = cute.runtime.make_fake_tensor(
            Uint32,
            (n, 32, 8),
            stride=(kv_row, 8, 1),
            assumed_align=16,
        )
        # Both cache group views exactly cover a 512-wide row in its store
        # dtype: 8-word groups for bf16 stores, 4-word groups for fp8. Only
        # the one selected by store_kv_fp8 is ever sliced.
        row_words = HEAD_DIM // 4 if compile_key.store_kv_fp8 else HEAD_DIM // 2
        cache_bf16 = cute.runtime.make_fake_tensor(
            Uint32,
            (b, bs, row_words // 8, 8),
            stride=(cache_stride0, cache_row, 8, 1),
            assumed_align=16,
        )
        cache_fp8 = cute.runtime.make_fake_tensor(
            Uint32,
            (b, bs, row_words // 4, 4),
            stride=(cache_stride0, cache_row, 4, 1),
            assumed_align=16,
        )
        slots = cute.runtime.make_fake_tensor(Int64, (i,), stride=(1,), assumed_align=8)
        pos_ids = cute.runtime.make_fake_tensor(
            Int64, (n,), stride=(1,), assumed_align=8
        )
        cos_sin = cute.runtime.make_fake_tensor(
            Float32, (p, ROPE_DIM // 4, 4), stride=(ROPE_DIM, 4, 1), assumed_align=16
        )
        norm_w = cute.runtime.make_fake_tensor(
            Uint32, (wt,), stride=(1,), assumed_align=4
        )
        kv_scale = cute.runtime.make_fake_tensor(
            Float32, (s1,), stride=(1,), assumed_align=4
        )
        q_scale_inv = cute.runtime.make_fake_tensor(
            Float32, (s2,), stride=(1,), assumed_align=4
        )
        return (
            q,
            q_fp8,
            kv,
            cache_fp8,
            cache_bf16,
            slots,
            pos_ids,
            cos_sin,
            norm_w,
            kv_scale,
            q_scale_inv,
        )

    @kernel_launcher
    def __call__(
        self,
        q: torch.Tensor,
        q_fp8: torch.Tensor,
        kv: torch.Tensor,
        cache_fp8: torch.Tensor,
        cache_bf16: torch.Tensor,
        slots: torch.Tensor,
        pos_ids: torch.Tensor,
        cos_sin: torch.Tensor,
        norm_w: torch.Tensor,
        kv_scale: torch.Tensor,
        q_scale_inv: torch.Tensor,
        *,
        num_heads_q: int,
        apply_q_norm: bool,
        has_kv_norm: bool,
        store_q_fp8: bool,
        store_kv_fp8: bool,
        eps: float,
        block_size: int = FULL_CACHE_BLOCK_SIZE,
    ) -> CuTeDSLLaunchSpec[FullCacheCompileKey]:
        compile_key = self.dispatch(
            num_heads_q=num_heads_q,
            apply_q_norm=apply_q_norm,
            has_kv_norm=has_kv_norm,
            store_q_fp8=store_q_fp8,
            store_kv_fp8=store_kv_fp8,
            eps=eps,
            block_size=block_size,
        )
        return compile_key, (
            q,
            q_fp8,
            kv,
            cache_fp8,
            cache_bf16,
            slots,
            pos_ids,
            cos_sin,
            norm_w,
            kv_scale,
            q_scale_inv,
        )


_PAGED_INSERT_KERNEL = _PagedInsertKernel()
_FULL_CACHE_INSERT_KERNEL = _FullCacheInsertKernel()


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q_in: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_head_padded: int,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
    kv_mxfp8: bool = False,
    apply_q_rope: bool = True,
    is_q_interleaved: bool = False,
) -> torch.Tensor:
    """Q side: optional RMSNorm + RoPE + zero-pad to q_head_padded.

    KV side: RoPE + UE8M0 FP8 quant + paged-cache insert. Returns the padded
    q tensor (empty when q_head_padded == 0). With DP padding, q/kv/positions
    may have more rows than slot_mapping; Q covers all rows, the KV insert
    only the first slot_mapping rows.
    """
    op_name = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    _check_common(op_name, kv, slot_mapping, position_ids, cos_sin_cache)
    if q_in.dim() != 3 or q_in.shape[2] != HEAD_DIM:
        raise ValueError(f"{op_name}: q_in shape [N, num_heads_q, {HEAD_DIM}]")
    if q_in.dtype != kv.dtype:
        raise ValueError(f"{op_name}: q_in and kv dtype must match")
    if q_in.device != kv.device:
        raise ValueError(f"{op_name}: q_in must be on the KV device")
    if not q_in.is_contiguous() or not kv.is_contiguous():
        raise ValueError(f"{op_name}: q_in and kv must be contiguous")
    if q_in.shape[0] != kv.shape[0]:
        raise ValueError(
            f"{op_name}: q/kv row counts must match ({q_in.shape[0]} vs {kv.shape[0]})"
        )
    if q_head_padded != 0 and q_head_padded < q_in.shape[1]:
        raise ValueError(
            f"{op_name}: q_head_padded must be 0 or >= num_heads_q "
            f"({q_head_padded} vs {q_in.shape[1]})"
        )
    if k_cache.dtype != torch.uint8:
        raise ValueError(f"{op_name}: k_cache must be uint8")
    if k_cache.device != kv.device:
        raise ValueError(f"{op_name}: k_cache must be on the KV device")
    if q_in.data_ptr() % 16 != 0:
        raise ValueError(f"{op_name}: q_in must be 16-byte aligned")
    num_tokens_full = q_in.shape[0]
    num_heads_q = q_in.shape[1]
    if q_head_padded == 0:
        q_out = torch.empty((0,), dtype=q_in.dtype, device=q_in.device)
        q_in_v = _dummy_q_words(q_in.device, num_heads_q, is_q_interleaved, 8)
        q_out_v = _dummy_q_words(q_in.device, 0, is_q_interleaved, 8)
    else:
        q_out = torch.empty(
            (num_tokens_full, q_head_padded, HEAD_DIM),
            dtype=q_in.dtype,
            device=q_in.device,
        )
        q_in_v = _q_word_views(q_in, num_heads_q, is_q_interleaved)
        q_out_v = _q_word_views(q_out, q_head_padded, is_q_interleaved)
    if num_tokens_full == 0:
        return q_out
    cache_data, cache_scale = _paged_cache_views(k_cache, cache_block_size, kv_mxfp8)
    cache_fp8_v, cache_bf16_v = _cache_word_views(cache_data, kv_mxfp8)
    _PAGED_INSERT_KERNEL(
        q_in_v,
        q_out_v,
        _kv_word_view(kv),
        cache_fp8_v,
        cache_bf16_v,
        cache_scale,
        slot_mapping,
        position_ids,
        _cos_sin_word_view(cos_sin_cache),
        _dummy_tensor(q_in.device, torch.uint32),
        num_heads_q=num_heads_q,
        q_head_padded=q_head_padded,
        apply_q_norm=apply_q_norm,
        apply_q_rope=apply_q_rope,
        is_q_interleaved=is_q_interleaved,
        kv_mxfp8=kv_mxfp8,
        has_kv_norm=False,
        eps=eps if apply_q_norm else 0.0,
    )
    return q_out


def fused_deepseek_v4_kv_rope_insert(
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_block_size: int,
    fp8_scale: torch.Tensor | None = None,
    kv_mxfp8: bool = False,
    kv_norm_weight: torch.Tensor | None = None,
    kv_norm_eps: float = 1e-6,
) -> None:
    """KV half alone: RoPE + quant + insert, no query tensors touched."""
    op_name = "fused_deepseek_v4_kv_rope_insert"
    _check_common(op_name, kv, slot_mapping, position_ids, cos_sin_cache)
    has_kv_norm = kv_norm_weight is not None
    if kv_norm_weight is not None and (
        kv_norm_weight.dtype != torch.bfloat16
        or kv_norm_weight.dim() != 1
        or kv_norm_weight.shape[0] != HEAD_DIM
        or not kv_norm_weight.is_contiguous()
        or kv_norm_weight.device != kv.device
    ):
        raise ValueError(
            f"{op_name}: kv_norm_weight must be contiguous bfloat16 "
            f"[{HEAD_DIM}] on the KV device"
        )
    if k_cache.device != kv.device:
        raise ValueError(f"{op_name}: k_cache must be on the KV device")
    if k_cache.dim() != 3 or k_cache.shape[1] != cache_block_size:
        raise ValueError(
            f"{op_name}: k_cache must be [num_blocks, {cache_block_size}, ...]"
        )
    normed_scale: torch.Tensor | None = None
    if k_cache.dtype == torch.uint8:
        row_bytes = MXFP8_BYTES_PER_TOKEN if kv_mxfp8 else V4_BYTES_PER_TOKEN
        if k_cache.shape[2] != row_bytes:
            raise ValueError(
                f"{op_name}: cache row width {k_cache.shape[2]} does not "
                f"match {row_bytes} for kv_mxfp8={kv_mxfp8}"
            )
    elif k_cache.dtype == torch.float8_e4m3fn:
        if k_cache.shape[2] != HEAD_DIM:
            raise ValueError(f"{op_name}: fp8 cache row width must be {HEAD_DIM}")
        if k_cache.stride(0) % 16 != 0 or k_cache.stride(1) % 16 != 0:
            raise ValueError(f"{op_name}: fp8 cache strides must be 16-byte aligned")
        if k_cache.data_ptr() % 16 != 0:
            raise ValueError(f"{op_name}: fp8 cache must be 16-byte aligned")
        if fp8_scale is None:
            raise ValueError(f"{op_name}: fp8 cache requires one float32 KV scale")
        normed_scale = _check_scalar(op_name, "fp8_scale", fp8_scale, kv.device)
    elif k_cache.dtype == torch.bfloat16:
        if k_cache.shape[2] != HEAD_DIM:
            raise ValueError(f"{op_name}: bf16 cache row width must be {HEAD_DIM}")
        if k_cache.stride(0) % 8 != 0 or k_cache.stride(1) % 8 != 0:
            raise ValueError(f"{op_name}: bf16 cache strides must be 16-byte aligned")
        if k_cache.data_ptr() % 16 != 0:
            raise ValueError(f"{op_name}: bf16 cache must be 16-byte aligned")
    else:
        raise ValueError(
            f"{op_name}: cache must be uint8, bfloat16 or float8_e4m3fn, "
            f"got {k_cache.dtype}"
        )
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    eps = kv_norm_eps if has_kv_norm else 0.0
    norm_w_u32 = (
        kv_norm_weight.view(torch.uint32).reshape(-1)
        if kv_norm_weight is not None
        else _dummy_tensor(kv.device, torch.uint32)
    )
    if k_cache.dtype == torch.uint8:
        cache_data, cache_scale = _paged_cache_views(
            k_cache, cache_block_size, kv_mxfp8
        )
        cache_fp8_v, cache_bf16_v = _cache_word_views(cache_data, kv_mxfp8)
        _PAGED_INSERT_KERNEL(
            _dummy_q_words(kv.device, 0, False, 8),
            _dummy_q_words(kv.device, 0, False, 8),
            _kv_word_view(kv),
            cache_fp8_v,
            cache_bf16_v,
            cache_scale,
            slot_mapping,
            position_ids,
            _cos_sin_word_view(cos_sin_cache),
            norm_w_u32,
            num_heads_q=0,
            q_head_padded=0,
            apply_q_norm=False,
            apply_q_rope=True,
            is_q_interleaved=False,
            kv_mxfp8=kv_mxfp8,
            has_kv_norm=has_kv_norm,
            eps=eps,
        )
    elif k_cache.dtype == torch.float8_e4m3fn:
        cache_fp8_v, cache_bf16_v = _full_cache_word_views(k_cache.view(torch.uint32))
        _FULL_CACHE_INSERT_KERNEL(
            _dummy_q_words(kv.device, 0, False, 8),
            _dummy_q_words(kv.device, 0, False, 4),
            _kv_word_view(kv),
            cache_fp8_v,
            cache_bf16_v,
            slot_mapping,
            position_ids,
            _cos_sin_word_view(cos_sin_cache),
            norm_w_u32,
            normed_scale,
            _dummy_tensor(kv.device, torch.float32),
            num_heads_q=0,
            apply_q_norm=False,
            has_kv_norm=has_kv_norm,
            store_q_fp8=False,
            store_kv_fp8=True,
            eps=eps,
        )
    else:
        cache_fp8_v, cache_bf16_v = _full_cache_word_views(k_cache.view(torch.uint32))
        _FULL_CACHE_INSERT_KERNEL(
            _dummy_q_words(kv.device, 0, False, 8),
            _dummy_q_words(kv.device, 0, False, 4),
            _kv_word_view(kv),
            cache_fp8_v,
            cache_bf16_v,
            slot_mapping,
            position_ids,
            _cos_sin_word_view(cos_sin_cache),
            norm_w_u32,
            _dummy_tensor(kv.device, torch.float32),
            _dummy_tensor(kv.device, torch.float32),
            num_heads_q=0,
            apply_q_norm=False,
            has_kv_norm=has_kv_norm,
            store_q_fp8=False,
            store_kv_fp8=False,
            eps=eps,
        )


def _full_cache_checks(
    op_name: str,
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cache_block_size: int,
) -> None:
    _check_common(op_name, kv, slot_mapping, position_ids, cos_sin_cache)
    if q.dim() != 3 or q.shape[2] != HEAD_DIM:
        raise ValueError(f"{op_name}: q shape [N, H, {HEAD_DIM}]")
    if q.dtype != kv.dtype:
        raise ValueError(f"{op_name}: q and kv dtype must match")
    if q.device != kv.device:
        raise ValueError(f"{op_name}: q must be on the KV device")
    if not q.is_contiguous() or not kv.is_contiguous():
        raise ValueError(f"{op_name}: q and kv must be contiguous")
    if q.data_ptr() % 16 != 0:
        raise ValueError(f"{op_name}: q must be 16-byte aligned")
    if q.shape[0] != kv.shape[0]:
        raise ValueError(
            f"{op_name}: q/kv row counts must match ({q.shape[0]} vs {kv.shape[0]})"
        )
    if k_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            f"{op_name}: k_cache must be bfloat16 or float8_e4m3fn, got {k_cache.dtype}"
        )
    if k_cache.device != kv.device:
        raise ValueError(f"{op_name}: k_cache must be on the KV device")
    if (
        k_cache.dim() != 3
        or k_cache.shape[1] != cache_block_size
        or k_cache.shape[2] != HEAD_DIM
        or k_cache.stride(2) != 1
    ):
        raise ValueError(
            f"{op_name}: k_cache shape [num_blocks, {cache_block_size}, {HEAD_DIM}]"
        )


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
) -> None:
    """Rewrite q in place (RMSNorm + RoPE) and insert the bf16 KV row."""
    op_name = "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert"
    _full_cache_checks(
        op_name,
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        cache_block_size,
    )
    if k_cache.stride(0) % 8 != 0 or k_cache.stride(1) % 8 != 0:
        raise ValueError(f"{op_name}: bf16 cache strides must be 16-byte aligned")
    if k_cache.data_ptr() % 16 != 0:
        raise ValueError(f"{op_name}: bf16 cache must be 16-byte aligned")
    if q.shape[0] == 0:
        return
    cache_fp8_v, cache_bf16_v = _full_cache_word_views(k_cache.view(torch.uint32))
    _FULL_CACHE_INSERT_KERNEL(
        _q_word_views(q, q.shape[1], False),
        _dummy_q_words(q.device, q.shape[1], False, 4),
        _kv_word_view(kv),
        cache_fp8_v,
        cache_bf16_v,
        slot_mapping,
        position_ids,
        _cos_sin_word_view(cos_sin_cache),
        _dummy_tensor(q.device, torch.uint32),
        _dummy_tensor(q.device, torch.float32),
        _dummy_tensor(q.device, torch.float32),
        num_heads_q=q.shape[1],
        apply_q_norm=apply_q_norm,
        has_kv_norm=False,
        store_q_fp8=False,
        store_kv_fp8=False,
        eps=eps if apply_q_norm else 0.0,
    )


def fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_fp8: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    fp8_scale: torch.Tensor,
    q_fp8_scale_inv: torch.Tensor,
    eps: float,
    cache_block_size: int,
    apply_q_norm: bool = True,
) -> None:
    """Quantize q and the KV row to per-tensor-scaled e4m3."""
    op_name = "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert"
    _full_cache_checks(
        op_name,
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        cache_block_size,
    )
    if (
        q_fp8.dtype != torch.float8_e4m3fn
        or q_fp8.shape != q.shape
        or not q_fp8.is_contiguous()
        or q_fp8.device != q.device
    ):
        raise ValueError(
            f"{op_name}: q_fp8 must be a contiguous float8_e4m3fn tensor "
            "matching q's shape on the KV device"
        )
    if k_cache.stride(0) % 16 != 0 or k_cache.stride(1) % 16 != 0:
        raise ValueError(f"{op_name}: fp8 cache strides must be 16-byte aligned")
    if k_cache.data_ptr() % 16 != 0:
        raise ValueError(f"{op_name}: fp8 cache must be 16-byte aligned")
    if q_fp8.data_ptr() % 16 != 0:
        raise ValueError(f"{op_name}: q_fp8 must be 16-byte aligned")
    kv_scale = _check_scalar(op_name, "fp8_scale", fp8_scale, q.device)
    q_scale_inv = _check_scalar(op_name, "q_fp8_scale_inv", q_fp8_scale_inv, q.device)
    if q.shape[0] == 0:
        return
    cache_fp8_v, cache_bf16_v = _full_cache_word_views(k_cache.view(torch.uint32))
    _FULL_CACHE_INSERT_KERNEL(
        _q_word_views(q, q.shape[1], False),
        q_fp8.view(torch.uint32).reshape(q.shape[0], q.shape[1], 32, 4),
        _kv_word_view(kv),
        cache_fp8_v,
        cache_bf16_v,
        slot_mapping,
        position_ids,
        _cos_sin_word_view(cos_sin_cache),
        _dummy_tensor(q.device, torch.uint32),
        kv_scale,
        q_scale_inv,
        num_heads_q=q.shape[1],
        apply_q_norm=apply_q_norm,
        has_kv_norm=False,
        store_q_fp8=True,
        store_kv_fp8=True,
        eps=eps if apply_q_norm else 0.0,
        block_size=(
            FULL_CACHE_FP8_L2_BLOCK_SIZE
            if q.shape[0] < FULL_CACHE_FP8_L2_TOKENS
            else FULL_CACHE_BLOCK_SIZE
        ),
    )
