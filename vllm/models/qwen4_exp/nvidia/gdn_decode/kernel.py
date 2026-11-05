# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated SM120 GDN RMS128/sigmoid/BF16/MXFP8 experiment; no model hooks.

Norm arithmetic follows vLLM layernorm_guard.py. Scale conversion and layout
follow FlashInfer0.7.2.post1 quantization_cute_dsl_utils.py (Apache-2.0).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _norm_sigmoid_mxfp8(
    X,
    Z,
    W,
    Q,
    SF,
    NORM,
    X_TOKEN_STRIDE: tl.constexpr,
    Z_TOKEN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    BATCH: tl.constexpr,
    STORE_NORM: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    columns = tl.arange(0, 128)[None, :]
    x = tl.load(X + token * X_TOKEN_STRIDE + head * 128 + columns).to(tl.float32)
    z = tl.load(Z + token * Z_TOKEN_STRIDE + head * 128 + columns).to(tl.float32)
    weight = tl.load(W + columns).to(tl.float32)
    variance = tl.sum(x * x, axis=1) / 128
    rstd = tl.rsqrt(variance + EPS)
    normalized = x * rstd[:, None]
    normalized = normalized * weight
    normalized = normalized * tl.sigmoid(z)
    # Native norm writes BF16 before the separate MXFP8 quantizer reads it.
    rounded = normalized.to(tl.bfloat16)
    if STORE_NORM:
        tl.store(NORM + token * 6144 + head * 128 + columns, rounded)
    values = tl.reshape(rounded.to(tl.float32), (4, 32))
    maximum = tl.max(tl.abs(values), axis=1)
    normalized_max = maximum * (1.0 / 448.0)
    bits = normalized_max.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = ((mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))).to(tl.uint32)
    scale = tl.minimum(exponent + bump, 254)
    scale = tl.where(normalized_max <= 0, 0, scale)
    inverse_bits = tl.where(scale == 0, 0, (254 - scale) << 23).to(tl.uint32)
    inverse = inverse_bits.to(tl.float32, bitcast=True)
    scaled = values * inverse[:, None]
    # Match native min.f32/max.f32 plus saturating round-to-nearest E4M3.
    clipped = tl.inline_asm_elementwise(
        "min.f32 $0, $1, 0f43E00000; max.f32 $0, $0, 0fC3E00000;",
        constraints="=f,f",
        args=[scaled],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    quantized = clipped.to(tl.float8e4nv)
    tl.store(
        Q + token * 6144 + head * 128 + tl.arange(0, 128), tl.reshape(quantized, (128,))
    )
    tl.store(SF + head * 512 + token * 16 + tl.arange(0, 4), scale.to(tl.uint8))
    # Each head owns one 512-byte scale tile. Only token0 writes its padding;
    # the mask excludes every live token's scale bytes, avoiding cross-CTA races.
    if token == 0:
        offsets = tl.arange(0, 512)
        logical_row = offsets // 16 + ((offsets % 16) // 4) * 32
        tl.store(SF + head * 512 + offsets, 0, mask=logical_row >= BATCH)


class GdnNormQuant:
    """Own outputs for exactly B1/B2 ×48 heads ×128 dimensions, one CUDA stream."""

    def __init__(self, batch, *, device):
        if batch not in (1, 2):
            raise ValueError("This isolated prototype supports only B1/B2")
        self.batch = batch
        self.quantized = torch.empty(
            (batch, 6144), dtype=torch.float8_e4m3fn, device=device
        )
        self.scales = torch.empty((128 * 192,), dtype=torch.uint8, device=device)
        self.normalized = torch.empty(
            (batch, 48, 128), dtype=torch.bfloat16, device=device
        )

    def run(self, x, z, weight, *, eps=1e-6, store_norm=False):
        if x.shape != (self.batch, 48, 128) or z.shape != x.shape:
            raise ValueError("Expected x and z shapes [B,48,128]")
        if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
            raise TypeError("Expected BF16 x/z")
        if weight.shape != (128,) or weight.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("Expected BF16/FP32 norm weight[128]")
        if (
            x.stride()[-2:] != (128, 1)
            or z.stride()[-2:] != (128, 1)
            or weight.stride() != (1,)
        ):
            raise ValueError("Expected contiguous head dimensions and norm weight")
        if any(value.device != self.quantized.device for value in (x, z, weight)):
            raise ValueError("All inputs must share the instance device")
        if not 0 < eps < 1:
            raise ValueError("Expected finite positive norm epsilon less than1")
        _norm_sigmoid_mxfp8[(self.batch, 48)](
            x,
            z,
            weight,
            self.quantized,
            self.scales,
            self.normalized,
            X_TOKEN_STRIDE=x.stride(0),
            Z_TOKEN_STRIDE=z.stride(0),
            EPS=eps,
            BATCH=self.batch,
            STORE_NORM=store_norm,
            num_warps=1,
        )
        return self.quantized, self.scales
