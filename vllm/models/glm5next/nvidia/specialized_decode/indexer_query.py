# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The indexer's per-token epilogue in one kernel: key LayerNorm, query
Hadamard rotation and FP8 quant, head-weight scaling, and the zero padding of
the query heads and weights that DeepGEMM's MQA logits need (stock: five
kernels).

One warp per query head, four values per lane: the Hadamard stages of strides 1
and 2 stay in a lane, the others pair lanes ``l`` and ``l ^ (stride / 4)``. The
butterflies, the rounding and the power-of-two scale (the stock kernel's
accurate log2, read off the exponent bits where that is exact) follow
``fwht128_quant_fp8`` bit for bit.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    asm,
    bf16x2,
    bits_float,
    float_bits,
    load_words,
    pack_fp8,
    store_words,
    unpack_bf16,
    warp_max_abs,
    warp_sum,
)

from .mega_moe import _store_bf16
from .shared_expert import _round_bf16

DIM = 128
WARPS = 4
INV_SQRT_DIM = 0.08838834764831845  # 1 / sqrt(128), exact in FP32


@asm(Float32, pure=True)
def _ceil(value):
    return "cvt.rpi.ftz.f32.f32 $0, $1;", "=f,f", (value,)


@cute.jit
def _exponent(x: Float32):
    """``ceil(log2(x))`` with the stock kernel's accurate log2, for positive x.

    From the bits unless the mantissa is barely above a power of two (where the
    rounded log2 can land on the exponent itself) or x is not normal.
    """
    bits = Int32(float_bits(x))
    biased = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    exponent = biased - 127 + Int32(mantissa != 0)
    if (mantissa != 0 and mantissa < 4096) or biased == 0 or biased == 255:
        exponent = Int32(_ceil(cute.math.log2(x, fastmath=False)))
    return exponent


@cute.jit
def hadamard_quant(got, lane):
    """One query head, a warp's four values per lane (two BF16x2 words), as
    ``fwht128_quant_fp8``: (the lane's FP8x4 word, the head's power-of-two scale).
    """
    x = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
    for i in cutlass.range_constexpr(2):
        x[2 * i], x[2 * i + 1] = unpack_bf16(got[i])
    # Strides 1 and 2 inside the lane: (a, b) -> (a + b, a - b).
    for stride in (1, 2):
        for i in cutlass.range_constexpr(4):
            if cutlass.const_expr(i % (2 * stride) < stride):
                a, b = x[i], x[i + stride]
                x[i], x[i + stride] = a + b, a - b
    # Strides 4..64 pair lanes l and l ^ (stride / 4); the lower lane holds a.
    for shift in cutlass.range_constexpr(5):
        upper = (lane >> shift) & 1
        for i in cutlass.range_constexpr(4):
            other = cute.arch.shuffle_sync_bfly(x[i], offset=1 << shift)
            if upper == 1:
                x[i] = other - x[i]
            else:
                x[i] = x[i] + other
    amax = Float32(0.0)
    for i in cutlass.range_constexpr(4):
        x[i] = _round_bf16(x[i] * Float32(INV_SQRT_DIM))
        amax = cute.arch.fmax(amax, cute.arch.fmax(x[i], -x[i]))
    amax = cute.arch.fmax(warp_max_abs(amax), Float32(1e-4))
    exponent = _exponent(amax * Float32(1.0 / 448.0))
    q_scale = bits_float(Uint32(exponent + 127) << 23)
    inverse = bits_float(Uint32(127 - exponent) << 23)
    y = [
        cute.arch.fmax(cute.arch.fmin(x[i] * inverse, Float32(448.0)), Float32(-448.0))
        for i in range(4)
    ]
    return Int32(pack_fp8(tuple(y))), q_scale


@cute.jit
def store_weight(w_out, t, head, value, padded: cutlass.Constexpr, bf16):
    """``w_out[t, head] = value``, rounded to BF16 (as ``.to(torch.bfloat16)``)
    when ``bf16``."""
    if cutlass.const_expr(bf16):
        _store_bf16(w_out.iterator + t * padded + head, value)
    else:
        w_out[t, head] = value


class _QueryKernel:
    def __init__(self, heads: int, padded: int, bf16: bool):
        assert heads % WARPS == 0 and padded >= heads
        self.heads, self.padded, self.bf16 = heads, padded, bf16
        self.groups = heads // WARPS  # CTAs per token for the heads

    @cute.jit
    def __call__(
        self, q: cute.Tensor, k: cute.Tensor, weights: cute.Tensor,
        norm_w: cute.Tensor, norm_b: cute.Tensor, q_out: cute.Tensor,
        w_out: cute.Tensor, k_out: cute.Tensor, scale: Float32, eps: Float32,
        tokens: Int32, st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            q, k, weights, norm_w, norm_b, q_out, w_out, k_out, scale, eps
        ).launch(
            grid=(tokens, self.groups + 1, 1),
            block=(32 * WARPS, 1, 1),
            stream=st,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, q: cute.Tensor, k: cute.Tensor, weights: cute.Tensor,
        norm_w: cute.Tensor, norm_b: cute.Tensor, q_out: cute.Tensor,
        w_out: cute.Tensor, k_out: cute.Tensor, scale: Float32, eps: Float32,
    ):  # fmt: skip
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        t, group, _ = cute.arch.block_idx()
        gamma = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        beta = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        for e in cutlass.range_constexpr(4):
            gamma[e] = norm_w[lane * 4 + e]
            beta[e] = norm_b[lane * 4 + e]
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if group < self.groups:
            self.head(q, weights, q_out, w_out, scale, t, group * WARPS + warp, lane)
        else:
            if warp == 0:
                self.key(k, k_out, gamma, beta, eps, t, lane)
            for head in cutlass.range(self.heads + warp, self.padded, WARPS, unroll=1):
                q_out[t, head, lane] = Int32(0)
                if lane == 0:
                    store_weight(w_out, t, head, Float32(0.0), self.padded, self.bf16)

    @cute.jit
    def head(self, q, weights, q_out, w_out, scale, t, head, lane):
        got = load_words(
            q.iterator + (t * self.heads + head) * DIM + lane * 4,
            count=2, word_type="u32", mutable=True,
        )  # fmt: skip
        weight = weights[t, head]  # issued beside the query load
        word, q_scale = hadamard_quant(got, lane)
        q_out[t, head, lane] = word
        if lane == 0:
            value = weight * q_scale * scale
            store_weight(w_out, t, head, value, self.padded, self.bf16)

    @cute.jit
    def key(self, k, k_out, gamma, beta, eps, t, lane):
        values = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        got = load_words(
            k.iterator + t * DIM + lane * 4, count=2, word_type="u32", mutable=True
        )
        for i in cutlass.range_constexpr(2):
            values[2 * i], values[2 * i + 1] = unpack_bf16(got[i])
        mean = warp_sum(values[0] + values[1] + values[2] + values[3]) / Float32(DIM)
        squares = Float32(0.0)
        for i in cutlass.range_constexpr(4):
            values[i] = values[i] - mean
            squares = squares + values[i] * values[i]
        inv = cute.math.rsqrt(warp_sum(squares) / Float32(DIM) + eps)
        for i in cutlass.range_constexpr(4):
            values[i] = values[i] * inv * gamma[i] + beta[i]
        store_words(
            k_out.iterator + t * DIM + lane * 4,
            (bf16x2(values[0], values[1]), bf16x2(values[2], values[3])),
        )


_compiled: dict = {}


def indexer_query(
    q, k, weights, norm, scale: float, padded: int, weights_dtype=torch.float32
):
    """Returns (q_fp8 [M, padded, 128], weights [M, padded], key [M, 128]).

    Args:
        q: [M, heads * 128] BF16 indexer query.
        k: [M, 128] BF16 indexer key before its LayerNorm.
        weights: [M, heads] FP32 head weights.
        norm: the key LayerNorm module (FP32 weight and bias).
        scale: the folded softmax and head-count scale.
        padded: heads after zero padding.
        weights_dtype: FP32, or BF16 for consumers that round them to BF16.

    """
    m, dim = k.shape
    heads = weights.shape[1]
    assert dim == DIM and q.is_contiguous() and k.is_contiguous()
    q_out = torch.empty((m, padded, dim), dtype=torch.float8_e4m3fn, device=q.device)
    w_out = torch.empty((m, padded), dtype=weights_dtype, device=q.device)
    k_out = torch.empty_like(k)
    words = q_out.view(torch.int32).view(m, padded, dim // 4)
    args = (q, k, weights, norm.weight, norm.bias, words, w_out, k_out)
    st = stream(q.device)
    bf16 = weights_dtype == torch.bfloat16
    key = (heads, padded, bf16, q.device.index)
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _QueryKernel(heads, padded, bf16), *(to_cute(t, 16) for t in args),
            Float32(scale), Float32(norm.eps), Int32(m), st,
        )  # fmt: skip
    _compiled[key](
        *(to_cute(t, 16) for t in args),
        Float32(scale), Float32(norm.eps), Int32(m), st,
    )  # fmt: skip
    return q_out, w_out, k_out
