# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DeepseekV4-MLA Q-norm/RoPE + KV RoPE/quant/insert, CuTe DSL port.

Replaces csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu. One warp owns
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
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, Uint8, Uint32

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

BLOCK_SIZE = 256
WARPS_PER_BLOCK = BLOCK_SIZE // 32
NUM_TOKEN_CUTOFF = 1024

_DUMMY_TENSORS: dict[tuple[Any, ...], torch.Tensor] = {}


def _dummy_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    tensor = _DUMMY_TENSORS.get(key)
    if tensor is None:
        tensor = torch.empty((1,), dtype=dtype, device=device)
        _DUMMY_TENSORS[key] = tensor
    return tensor


def _dummy_3d(device: torch.device, dtype: torch.dtype, width: int) -> torch.Tensor:
    key = (device, dtype, "3d", width)
    tensor = _DUMMY_TENSORS.get(key)
    if tensor is None:
        tensor = torch.empty((1, 0, width), dtype=dtype, device=device)
        _DUMMY_TENSORS[key] = tensor
    return tensor


@cutlass.cutlass_dsl.dsl_user_op
def _rsqrt_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        cutlass._mlir.dialects.llvm.inline_asm(
            cutlass.cutlass_dsl.T.f32(),
            [x.ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.cutlass_dsl.dsl_user_op
def _sqrt_rn(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        cutlass._mlir.dialects.llvm.inline_asm(
            cutlass.cutlass_dsl.T.f32(),
            [x.ir_value(loc=loc, ip=ip)],
            "sqrt.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _fmaxf(a: Float32, b: Float32) -> Float32:
    out = a
    if b > a:
        out = b
    return out


@cute.jit
def _fminf(a: Float32, b: Float32) -> Float32:
    out = a
    if b < a:
        out = b
    return out


@cute.jit
def _fabsf(x: Float32) -> Float32:
    out = x
    if x < Float32(0.0):
        out = -x
    return out


@cute.jit
def _clamp_fp8(x: Float32) -> Float32:
    return _fminf(_fmaxf(x, Float32(-FP8_MAX)), Float32(FP8_MAX))


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
    val = _fmaxf(val, peer)
    peer = cute.arch.shuffle_sync_bfly(val, 2)
    return _fmaxf(val, peer)


@cute.jit
def _warp2_max_abs(val: Float32) -> Float32:
    return _fmaxf(val, cute.arch.shuffle_sync_bfly(val, 1))


@cute.jit
def _decode_words(words: cute.Tensor, elems: cute.Tensor) -> None:
    # bf16x2_to_fp32x2 returns (high half, low half); the low half is the
    # even element of the pair.
    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
        hi, lo = bf16x2_to_fp32x2(words[w])
        elems[2 * w] = lo
        elems[2 * w + 1] = hi


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
    rcp = _rsqrt_approx(sum_sq / Float32(HEAD_DIM) + eps)
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
    inv_rms = Float32(1.0) / _sqrt_rn(sum_sq / Float32(HEAD_DIM) + eps)
    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
        hi, lo = bf16x2_to_fp32x2(norm_w[dim_base // 2 + w])
        elems[2 * w] = elems[2 * w] * inv_rms * lo
        elems[2 * w + 1] = elems[2 * w + 1] * inv_rms * hi
    _round_to_bf16(elems)


@cute.jit
def _apply_rope(
    elems: cute.Tensor, pos: Int64, dim_base: Int32, cos_sin: cute.Tensor
) -> None:
    half_base = (dim_base - NOPE_DIM) >> 1
    for p in cutlass.range_constexpr(ELEMS_PER_LANE // 2):
        c = cos_sin[pos, Int64(half_base + p)]
        s = cos_sin[pos, Int64(HALF_ROPE + half_base + p)]
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
    if flat.stride(0) % 4 != 0:
        raise ValueError("k_cache block stride must be 4-byte aligned")
    num_blocks = flat.shape[0]
    if kv_mxfp8:
        records = flat[:, :block_bytes].view(
            num_blocks, cache_block_size, MXFP8_BYTES_PER_TOKEN
        )
        data = records[:, :, :MXFP8_DATA_BYTES].view(torch.uint32)
        scale = records[:, :, MXFP8_DATA_BYTES:]
    else:
        data = (
            flat[:, :cache_block_size * V4_DATA_BYTES]
            .view(num_blocks, cache_block_size, V4_DATA_BYTES)
            .view(torch.uint32)
        )
        scale = flat[
            :,
            cache_block_size * V4_DATA_BYTES : cache_block_size * V4_BYTES_PER_TOKEN,
        ].view(num_blocks, cache_block_size, V4_SCALE_BYTES)
    return data, scale


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
    if kv.stride(0) % 2 != 0:
        raise ValueError(f"{op_name}: kv row stride must be even")
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
            f"{op_name}: cos_sin_cache must be contiguous float32 "
            f"[max_pos, {ROPE_DIM}]"
        )
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
    reduced_grid: bool
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
        REDUCED = compile_key.reduced_grid
        EPS = Float32(compile_key.eps)
        USE_PDL = current_platform.is_arch_support_pdl()

        def q_in_off(token: Int32, slot: Int32, lane: Int32, w: int) -> Int64:
            if Q_INTERLEAVED:
                return (
                    Int64(token) * Int64(HQ * (HEAD_DIM // 2))
                    + Int64(lane * HQ * 8 + slot * 8 + w)
                )
            return (Int64(token) * Int64(HQ) + Int64(slot)) * 256 + Int64(
                8 * lane + w
            )

        def q_out_off(token: Int32, slot: Int32, lane: Int32, w: int) -> Int64:
            if Q_INTERLEAVED:
                return (
                    Int64(token) * Int64(QP * (HEAD_DIM // 2))
                    + Int64(lane * QP * 8 + slot * 8 + w)
                )
            return (Int64(token) * Int64(QP) + Int64(slot)) * 256 + Int64(
                8 * lane + w
            )

        def process_slot(
            token: Int32,
            slot: Int32,
            lane: Int32,
            dim_base: Int32,
            q_in: cute.Tensor,
            q_out: cute.Tensor,
            kv: cute.Tensor,
            cache_data: cute.Tensor,
            cache_scale: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            n_insert: Int32,
            block_size: Int32,
        ) -> None:
            if slot == Int32(QP):
                # KV slot: RoPE + quant + insert, only for the first
                # n_insert rows (with DP padding, tail rows have no slot).
                if token < n_insert:
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_wait()
                    elems = cute.make_rmem_tensor((ELEMS_PER_LANE,), Float32)
                    words = cute.make_rmem_tensor((BF16_WORDS_PER_LANE,), Uint32)
                    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                        words[w] = kv[token, 8 * lane + w]
                    _decode_words(words, elems)
                    if cutlass.const_expr(HAS_KV_NORM):
                        _apply_kv_norm(elems, dim_base, EPS, norm_w)
                    if dim_base >= Int32(NOPE_DIM):
                        _apply_rope(elems, pos_ids[token], dim_base, cos_sin)
                    sid = slots[token]
                    if sid >= Int64(0):
                        block = sid // Int64(block_size)
                        pib = sid % Int64(block_size)
                        _round_to_bf16(elems)
                        local_max = Float32(0.0)
                        for i in cutlass.range_constexpr(ELEMS_PER_LANE):
                            local_max = _fmaxf(local_max, _fabsf(elems[i]))
                        if cutlass.const_expr(KV_MXFP8):
                            absmax = _fmaxf(
                                _warp2_max_abs(local_max), Float32(1e-4)
                            )
                        else:
                            absmax = _fmaxf(
                                _warp4_max_abs(local_max), Float32(1e-4)
                            )
                        exponent, inv_scale = _ue8m0_scale(absmax)
                        if cutlass.const_expr(KV_MXFP8):
                            store_fp8 = True
                        else:
                            store_fp8 = dim_base < Int32(NOPE_DIM)
                        if store_fp8:
                            for w in cutlass.range_constexpr(FP8_WORDS_PER_LANE):
                                v0 = _clamp_fp8(elems[4 * w] * inv_scale)
                                v1 = _clamp_fp8(elems[4 * w + 1] * inv_scale)
                                v2 = _clamp_fp8(elems[4 * w + 2] * inv_scale)
                                v3 = _clamp_fp8(elems[4 * w + 3] * inv_scale)
                                packed = fp32x4_to_fp8x4(v0, v1, v2, v3)
                                cache_data[block, pib, Int64(dim_base // 4 + w)] = (
                                    packed
                                )
                            encoded = exponent + 127
                            if encoded > 255:
                                encoded = 255
                            if encoded < 0:
                                encoded = 0
                            if cutlass.const_expr(KV_MXFP8):
                                # One scale per 32 dims = one per lane pair.
                                if (lane & 1) == 0:
                                    cache_scale[block, pib, Int64(lane >> 1)] = (
                                        Uint8(encoded)
                                    )
                            else:
                                if (lane & 3) == 0:
                                    cache_scale[block, pib, Int64(lane >> 2)] = (
                                        Uint8(encoded)
                                    )
                                if lane == 0:
                                    # Pad byte the reader expects at index 7.
                                    cache_scale[
                                        block, pib, Int64(V4_SCALE_BYTES - 1)
                                    ] = Uint8(0)
                        else:
                            _encode_words(elems, words)
                            rope_word = V4_ROPE_WORD_BASE + (dim_base - NOPE_DIM) // 4
                            for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                                cache_data[block, pib, Int64(rope_word + w)] = (
                                    words[w]
                                )
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_launch_dependents()
            else:
                # Q slot: pad slots only zero-fill; live slots decode,
                # optionally norm and rope, then re-encode.
                if cutlass.const_expr(USE_PDL):
                    cute.arch.griddepcontrol_wait()
                if slot < Int32(HQ):
                    elems = cute.make_rmem_tensor((ELEMS_PER_LANE,), Float32)
                    words = cute.make_rmem_tensor((BF16_WORDS_PER_LANE,), Uint32)
                    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                        words[w] = q_in[q_in_off(token, slot, lane, w)]
                    _decode_words(words, elems)
                    if cutlass.const_expr(APPLY_Q_NORM):
                        _apply_q_norm(elems, EPS)
                    if cutlass.const_expr(APPLY_Q_ROPE):  # noqa: SIM102
                        if dim_base >= Int32(NOPE_DIM):
                            _apply_rope(elems, pos_ids[token], dim_base, cos_sin)
                    _encode_words(elems, words)
                    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                        q_out[q_out_off(token, slot, lane, w)] = words[w]
                else:
                    # FlashMLA reads the pad slots; bf16 +0.0 is 0x0000.
                    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                        q_out[q_out_off(token, slot, lane, w)] = Uint32(0)
                if cutlass.const_expr(USE_PDL):
                    cute.arch.griddepcontrol_launch_dependents()

        @cute.kernel
        def device_kernel(
            q_in: cute.Tensor,
            q_out: cute.Tensor,
            kv: cute.Tensor,
            cache_data: cute.Tensor,
            cache_scale: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
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
                    cache_data,
                    cache_scale,
                    slots,
                    pos_ids,
                    cos_sin,
                    norm_w,
                    n_insert,
                    block_size,
                )

        @cute.kernel
        def device_kernel_reduced(
            q_in: cute.Tensor,
            q_out: cute.Tensor,
            kv: cute.Tensor,
            cache_data: cute.Tensor,
            cache_scale: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            n_full: Int32,
            n_insert: Int32,
            block_size: Int32,
        ):
            # One CTA per token; each warp walks slots strided by
            # WARPS_PER_BLOCK. Pays off above NUM_TOKEN_CUTOFF tokens in a
            # single insert, where it is worth ~2x kernel-isolated.
            bidx, _, _ = cute.arch.block_idx()
            warp = cute.arch.warp_idx()
            lane = cute.arch.lane_idx()
            dim_base = lane * ELEMS_PER_LANE
            token = bidx
            if token < n_full:
                slot_end = Int32(QP)
                if token < n_insert:
                    slot_end = Int32(QP + 1)
                if warp < slot_end:
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_wait()
                    slot = warp
                    while slot < slot_end:
                        process_slot(
                            token,
                            slot,
                            lane,
                            dim_base,
                            q_in,
                            q_out,
                            kv,
                            cache_data,
                            cache_scale,
                            slots,
                            pos_ids,
                            cos_sin,
                            norm_w,
                            n_insert,
                            block_size,
                        )
                        slot = slot + WARPS_PER_BLOCK
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_launch_dependents()

        @cute.jit
        def host_entrypoint(
            q_in: cute.Tensor,
            q_out: cute.Tensor,
            kv: cute.Tensor,
            cache_data: cute.Tensor,
            cache_scale: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            stream: CUstream,
        ):
            n = kv.shape[0]
            ni = slots.shape[0]
            bs = cache_data.shape[1]
            if REDUCED:
                device_kernel_reduced(
                    q_in,
                    q_out,
                    kv,
                    cache_data,
                    cache_scale,
                    slots,
                    pos_ids,
                    cos_sin,
                    norm_w,
                    n,
                    ni,
                    bs,
                ).launch(
                    grid=(n, 1, 1),
                    block=(BLOCK_SIZE, 1, 1),
                    stream=stream,
                    use_pdl=USE_PDL,
                )
            else:
                device_kernel(
                    q_in,
                    q_out,
                    kv,
                    cache_data,
                    cache_scale,
                    slots,
                    pos_ids,
                    cos_sin,
                    norm_w,
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
        n = cute.sym_int()
        i = cute.sym_int()
        p = cute.sym_int()
        b = cute.sym_int()
        bs = cute.sym_int()
        qt = cute.sym_int()
        qot = cute.sym_int()
        wt = cute.sym_int()
        data_stride0 = cute.sym_int64(divisibility=1)
        scale_stride0 = cute.sym_int64(divisibility=1)
        kv_row = cute.sym_int64(divisibility=1)
        if compile_key.kv_mxfp8:
            data_words = MXFP8_DATA_WORDS
            data_row = MXFP8_BYTES_PER_TOKEN // 4
            scale_bytes = MXFP8_SCALE_BYTES
            scale_row = MXFP8_BYTES_PER_TOKEN
        else:
            data_words = V4_DATA_WORDS
            data_row = V4_DATA_WORDS
            scale_bytes = V4_SCALE_BYTES
            scale_row = V4_SCALE_BYTES
        q_in = cute.runtime.make_fake_tensor(
            Uint32, (qt,), stride=(1,), assumed_align=4
        )
        q_out = cute.runtime.make_fake_tensor(
            Uint32, (qot,), stride=(1,), assumed_align=4
        )
        kv = cute.runtime.make_fake_tensor(
            Uint32,
            (n, HEAD_DIM // 2),
            stride=(kv_row, 1),
            assumed_align=4,
        )
        cache_data = cute.runtime.make_fake_tensor(
            Uint32,
            (b, bs, data_words),
            stride=(data_stride0, data_row, 1),
            assumed_align=4,
        )
        cache_scale = cute.runtime.make_fake_tensor(
            Uint8,
            (b, bs, scale_bytes),
            stride=(scale_stride0, scale_row, 1),
            assumed_align=4,
        )
        slots = cute.runtime.make_fake_tensor(
            Int64, (i,), stride=(1,), assumed_align=8
        )
        pos_ids = cute.runtime.make_fake_tensor(
            Int64, (n,), stride=(1,), assumed_align=8
        )
        cos_sin = cute.runtime.make_fake_tensor(
            Float32, (p, ROPE_DIM), stride=(ROPE_DIM, 1), assumed_align=4
        )
        norm_w = cute.runtime.make_fake_tensor(
            Uint32, (wt,), stride=(1,), assumed_align=4
        )
        return (
            q_in,
            q_out,
            kv,
            cache_data,
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
        cache_data: torch.Tensor,
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
        reduced_grid: bool,
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
            reduced_grid=reduced_grid,
            eps=eps,
        )
        return compile_key, (
            q_in,
            q_out,
            kv,
            cache_data,
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
        EPS = Float32(compile_key.eps)
        USE_PDL = current_platform.is_arch_support_pdl()

        @cute.kernel
        def device_kernel(
            q: cute.Tensor,
            q_fp8: cute.Tensor,
            kv: cute.Tensor,
            cache_data: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            kv_scale: cute.Tensor,
            q_scale_inv: cute.Tensor,
            n_full: Int32,
            n_insert: Int32,
            block_size: Int32,
        ):
            bidx, _, _ = cute.arch.block_idx()
            warp = cute.arch.warp_idx()
            lane = cute.arch.lane_idx()
            gw = bidx * WARPS_PER_BLOCK + warp
            token = gw // Int32(HQ + 1)
            slot = gw % Int32(HQ + 1)
            dim_base = lane * ELEMS_PER_LANE
            if token < n_full:
                if slot == Int32(HQ):
                    if token < n_insert:
                        if cutlass.const_expr(USE_PDL):
                            cute.arch.griddepcontrol_wait()
                        elems = cute.make_rmem_tensor((ELEMS_PER_LANE,), Float32)
                        words = cute.make_rmem_tensor(
                            (BF16_WORDS_PER_LANE,), Uint32
                        )
                        for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                            words[w] = kv[token, 8 * lane + w]
                        _decode_words(words, elems)
                        if cutlass.const_expr(HAS_KV_NORM):
                            _apply_kv_norm(elems, dim_base, EPS, norm_w)
                        # Full-cache rows rope the KV unconditionally; this
                        # variant has no apply_q_rope knob.
                        if dim_base >= Int32(NOPE_DIM):
                            _apply_rope(elems, pos_ids[token], dim_base, cos_sin)
                        sid = slots[token]
                        if sid >= Int64(0):
                            block = sid // Int64(block_size)
                            pib = sid % Int64(block_size)
                            if cutlass.const_expr(STORE_KV_FP8):
                                inv = Float32(1.0) / kv_scale[0]
                                for w in cutlass.range_constexpr(FP8_WORDS_PER_LANE):
                                    v0 = _clamp_fp8(elems[4 * w] * inv)
                                    v1 = _clamp_fp8(elems[4 * w + 1] * inv)
                                    v2 = _clamp_fp8(elems[4 * w + 2] * inv)
                                    v3 = _clamp_fp8(elems[4 * w + 3] * inv)
                                    packed = fp32x4_to_fp8x4(v0, v1, v2, v3)
                                    cache_data[block, pib, Int64(dim_base // 4 + w)] = (
                                        packed
                                    )
                            else:
                                _encode_words(elems, words)
                                for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                                    cache_data[block, pib, Int64(dim_base // 2 + w)] = (
                                        words[w]
                                    )
                        if cutlass.const_expr(USE_PDL):
                            cute.arch.griddepcontrol_launch_dependents()
                else:
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_wait()
                    elems = cute.make_rmem_tensor((ELEMS_PER_LANE,), Float32)
                    words = cute.make_rmem_tensor((BF16_WORDS_PER_LANE,), Uint32)
                    for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                        words[w] = q[token, slot, 8 * lane + w]
                    _decode_words(words, elems)
                    if cutlass.const_expr(APPLY_Q_NORM):
                        _apply_q_norm(elems, EPS)
                    if dim_base >= Int32(NOPE_DIM):
                        _apply_rope(elems, pos_ids[token], dim_base, cos_sin)
                    if cutlass.const_expr(STORE_Q_FP8):
                        sinv = q_scale_inv[0]
                        for w in cutlass.range_constexpr(FP8_WORDS_PER_LANE):
                            v0 = _clamp_fp8(elems[4 * w] * sinv)
                            v1 = _clamp_fp8(elems[4 * w + 1] * sinv)
                            v2 = _clamp_fp8(elems[4 * w + 2] * sinv)
                            v3 = _clamp_fp8(elems[4 * w + 3] * sinv)
                            q_fp8[token, slot, 4 * lane + w] = fp32x4_to_fp8x4(
                                v0, v1, v2, v3
                            )
                    else:
                        _encode_words(elems, words)
                        for w in cutlass.range_constexpr(BF16_WORDS_PER_LANE):
                            q[token, slot, 8 * lane + w] = words[w]
                    if cutlass.const_expr(USE_PDL):
                        cute.arch.griddepcontrol_launch_dependents()

        @cute.jit
        def host_entrypoint(
            q: cute.Tensor,
            q_fp8: cute.Tensor,
            kv: cute.Tensor,
            cache_data: cute.Tensor,
            slots: cute.Tensor,
            pos_ids: cute.Tensor,
            cos_sin: cute.Tensor,
            norm_w: cute.Tensor,
            kv_scale: cute.Tensor,
            q_scale_inv: cute.Tensor,
            stream: CUstream,
        ):
            n = kv.shape[0]
            ni = slots.shape[0]
            bs = cache_data.shape[1]
            device_kernel(
                q,
                q_fp8,
                kv,
                cache_data,
                slots,
                pos_ids,
                cos_sin,
                norm_w,
                kv_scale,
                q_scale_inv,
                n,
                ni,
                bs,
            ).launch(
                grid=(
                    (n * (HQ + 1) + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK,
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
    ) -> FullCacheCompileKey:
        return self.CompileKey(**compile_key_fields)

    def get_warmup_keys(
        self, *args: Any, **kwargs: Any
    ) -> list[FullCacheCompileKey]:
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
        kv_row = cute.sym_int64(divisibility=1)
        cache_stride0 = cute.sym_int64(divisibility=1)
        cache_row = cute.sym_int64(divisibility=1)
        data_words = HEAD_DIM // 4 if compile_key.store_kv_fp8 else HEAD_DIM // 2
        q = cute.runtime.make_fake_tensor(
            Uint32,
            (qn, compile_key.num_heads_q, HEAD_DIM // 2),
            stride=(compile_key.num_heads_q * (HEAD_DIM // 2), HEAD_DIM // 2, 1),
            assumed_align=4,
        )
        q_fp8 = cute.runtime.make_fake_tensor(
            Uint32,
            (qfn, compile_key.num_heads_q, HEAD_DIM // 4),
            stride=(compile_key.num_heads_q * (HEAD_DIM // 4), HEAD_DIM // 4, 1),
            assumed_align=4,
        )
        kv = cute.runtime.make_fake_tensor(
            Uint32,
            (n, HEAD_DIM // 2),
            stride=(kv_row, 1),
            assumed_align=4,
        )
        cache_data = cute.runtime.make_fake_tensor(
            Uint32,
            (b, bs, data_words),
            stride=(cache_stride0, cache_row, 1),
            assumed_align=4,
        )
        slots = cute.runtime.make_fake_tensor(
            Int64, (i,), stride=(1,), assumed_align=8
        )
        pos_ids = cute.runtime.make_fake_tensor(
            Int64, (n,), stride=(1,), assumed_align=8
        )
        cos_sin = cute.runtime.make_fake_tensor(
            Float32, (p, ROPE_DIM), stride=(ROPE_DIM, 1), assumed_align=4
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
            cache_data,
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
        cache_data: torch.Tensor,
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
    ) -> CuTeDSLLaunchSpec[FullCacheCompileKey]:
        compile_key = self.dispatch(
            num_heads_q=num_heads_q,
            apply_q_norm=apply_q_norm,
            has_kv_norm=has_kv_norm,
            store_q_fp8=store_q_fp8,
            store_kv_fp8=store_kv_fp8,
            eps=eps,
        )
        return compile_key, (
            q,
            q_fp8,
            kv,
            cache_data,
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
            f"{op_name}: q/kv row counts must match "
            f"({q_in.shape[0]} vs {kv.shape[0]})"
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
    num_tokens_full = q_in.shape[0]
    num_heads_q = q_in.shape[1]
    if q_head_padded == 0:
        q_out = torch.empty((0,), dtype=q_in.dtype, device=q_in.device)
        q_in_u32 = _dummy_tensor(q_in.device, torch.uint32)
        q_out_u32 = q_in_u32
    else:
        q_out = torch.empty(
            (num_tokens_full, q_head_padded, HEAD_DIM),
            dtype=q_in.dtype,
            device=q_in.device,
        )
        q_in_u32 = q_in.view(torch.uint32).reshape(-1)
        q_out_u32 = q_out.view(torch.uint32).reshape(-1)
    if num_tokens_full == 0:
        return q_out
    kv_u32 = kv.view(torch.uint32)
    cache_data, cache_scale = _paged_cache_views(
        k_cache, cache_block_size, kv_mxfp8
    )
    _PAGED_INSERT_KERNEL(
        q_in_u32,
        q_out_u32,
        kv_u32,
        cache_data,
        cache_scale,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        _dummy_tensor(q_in.device, torch.uint32),
        num_heads_q=num_heads_q,
        q_head_padded=q_head_padded,
        apply_q_norm=apply_q_norm,
        apply_q_rope=apply_q_rope,
        is_q_interleaved=is_q_interleaved,
        kv_mxfp8=kv_mxfp8,
        has_kv_norm=False,
        reduced_grid=num_tokens_full >= NUM_TOKEN_CUTOFF and q_head_padded > 0,
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
    if has_kv_norm and (
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
            raise ValueError(
                f"{op_name}: fp8 cache row width must be {HEAD_DIM}"
            )
        if k_cache.stride(0) % 4 != 0 or k_cache.stride(1) % 4 != 0:
            raise ValueError(f"{op_name}: fp8 cache strides must be 4-aligned")
        if fp8_scale is None:
            raise ValueError(
                f"{op_name}: fp8 cache requires one float32 KV scale"
            )
        normed_scale = _check_scalar(op_name, "fp8_scale", fp8_scale, kv.device)
    elif k_cache.dtype == torch.bfloat16:
        if k_cache.shape[2] != HEAD_DIM:
            raise ValueError(
                f"{op_name}: bf16 cache row width must be {HEAD_DIM}"
            )
        if k_cache.stride(0) % 2 != 0 or k_cache.stride(1) % 2 != 0:
            raise ValueError(f"{op_name}: bf16 cache strides must be even")
    else:
        raise ValueError(
            f"{op_name}: cache must be uint8, bfloat16 or float8_e4m3fn, "
            f"got {k_cache.dtype}"
        )
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    kv_u32 = kv.view(torch.uint32)
    eps = kv_norm_eps if has_kv_norm else 0.0
    norm_w_u32 = (
        kv_norm_weight.view(torch.uint32).reshape(-1)
        if has_kv_norm
        else _dummy_tensor(kv.device, torch.uint32)
    )
    if k_cache.dtype == torch.uint8:
        cache_data, cache_scale = _paged_cache_views(
            k_cache, cache_block_size, kv_mxfp8
        )
        _PAGED_INSERT_KERNEL(
            _dummy_tensor(kv.device, torch.uint32),
            _dummy_tensor(kv.device, torch.uint32),
            kv_u32,
            cache_data,
            cache_scale,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            norm_w_u32,
            num_heads_q=0,
            q_head_padded=0,
            apply_q_norm=False,
            apply_q_rope=True,
            is_q_interleaved=False,
            kv_mxfp8=kv_mxfp8,
            has_kv_norm=has_kv_norm,
            reduced_grid=False,
            eps=eps,
        )
    elif k_cache.dtype == torch.float8_e4m3fn:
        _FULL_CACHE_INSERT_KERNEL(
            _dummy_3d(kv.device, torch.uint32, HEAD_DIM // 2),
            _dummy_3d(kv.device, torch.uint32, HEAD_DIM // 4),
            kv_u32,
            k_cache.view(torch.uint32),
            slot_mapping,
            position_ids,
            cos_sin_cache,
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
        _FULL_CACHE_INSERT_KERNEL(
            _dummy_3d(kv.device, torch.uint32, HEAD_DIM // 2),
            _dummy_3d(kv.device, torch.uint32, HEAD_DIM // 4),
            kv_u32,
            k_cache.view(torch.uint32),
            slot_mapping,
            position_ids,
            cos_sin_cache,
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
    if q.shape[0] != kv.shape[0]:
        raise ValueError(
            f"{op_name}: q/kv row counts must match "
            f"({q.shape[0]} vs {kv.shape[0]})"
        )
    if k_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            f"{op_name}: k_cache must be bfloat16 or float8_e4m3fn, "
            f"got {k_cache.dtype}"
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
            f"{op_name}: k_cache shape "
            f"[num_blocks, {cache_block_size}, {HEAD_DIM}]"
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
    if k_cache.stride(0) % 2 != 0 or k_cache.stride(1) % 2 != 0:
        raise ValueError(f"{op_name}: bf16 cache strides must be even")
    if q.shape[0] == 0:
        return
    _FULL_CACHE_INSERT_KERNEL(
        q.view(torch.uint32),
        _dummy_3d(q.device, torch.uint32, HEAD_DIM // 4),
        kv.view(torch.uint32),
        k_cache.view(torch.uint32),
        slot_mapping,
        position_ids,
        cos_sin_cache,
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
    if k_cache.stride(0) % 4 != 0 or k_cache.stride(1) % 4 != 0:
        raise ValueError(f"{op_name}: fp8 cache strides must be 4-aligned")
    kv_scale = _check_scalar(op_name, "fp8_scale", fp8_scale, q.device)
    q_scale_inv = _check_scalar(
        op_name, "q_fp8_scale_inv", q_fp8_scale_inv, q.device
    )
    if q.shape[0] == 0:
        return
    _FULL_CACHE_INSERT_KERNEL(
        q.view(torch.uint32),
        q_fp8.view(torch.uint32),
        kv.view(torch.uint32),
        k_cache.view(torch.uint32),
        slot_mapping,
        position_ids,
        cos_sin_cache,
        _dummy_tensor(q.device, torch.uint32),
        kv_scale,
        q_scale_inv,
        num_heads_q=q.shape[1],
        apply_q_norm=apply_q_norm,
        has_kv_norm=False,
        store_q_fp8=True,
        store_kv_fp8=True,
        eps=eps if apply_q_norm else 0.0,
    )
