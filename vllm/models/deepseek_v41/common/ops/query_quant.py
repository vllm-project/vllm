# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _norm_row(x, weight, row, stride, eps, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    cols = tl.arange(0, BLOCK)
    w = tl.load(weight + cols, cols < SIZE, 0).to(tl.float32)
    values = tl.load(x + row * stride + cols, cols < SIZE, 0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(values * values, 0) / SIZE + eps)
    return values * rrms * w


@triton.jit(do_not_specialize=["num_tokens"])
def _q_kv_norm_quant_kernel(
    q,
    kv,
    qw,
    kvw,
    qo,
    kvo,
    scales,
    num_tokens,
    q_stride,
    kv_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if tl.program_id(1) == 0:
        groups = tl.arange(0, BLOCK // 32)
        if row < num_tokens:
            y = _norm_row(q, qw, row, q_stride, eps, Q_SIZE, BLOCK)
            # Preserve the materialized normalization output's rounding boundary.
            y = y.to(q.dtype.element_ty).to(tl.float32)
            grouped = tl.reshape(y, (BLOCK // 32, 32))
            amax = tl.max(tl.abs(grouped), 1)
            normalized = amax * (1.0 / 448.0)
            bits = normalized.to(tl.uint32, bitcast=True)
            exponent = (bits >> 23) & 255
            mantissa = bits & 0x7FFFFF
            bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
            sf = tl.minimum(exponent + bump, 254)
            sf = tl.where(normalized <= 0, 0, sf)
            # Match FlashInfer's UE8M0 conversion, including zero/subnormal scales.
            inv_bits = tl.where(sf == 0, 0, (254 - sf) << 23)
            inv_scale = inv_bits.to(tl.float32, bitcast=True)
            quantized = tl.reshape(grouped * inv_scale[:, None], (BLOCK,))
            tl.store(qo + row * Q_SIZE + cols, quantized, cols < Q_SIZE)
        else:
            sf = tl.full((BLOCK // 32,), 0, tl.uint32)
        padded_groups: tl.constexpr = triton.cdiv(Q_SIZE // 32, 4) * 4
        sf = tl.where(groups < Q_SIZE // 32, sf, 0)
        # F8_128x4: [row/128, group/4, row%32, row%128/32, group%4].
        offsets = (
            row // 128 * (128 * padded_groups)
            + groups // 4 * 512
            + row % 32 * 16
            + row % 128 // 32 * 4
            + groups % 4
        )
        tl.store(scales + offsets, sf, groups < padded_groups)
    elif row < num_tokens:
        y = _norm_row(kv, kvw, row, kv_stride, eps, KV_SIZE, BLOCK)
        tl.store(kvo + row * KV_SIZE + cols, y, cols < KV_SIZE)


def fused_q_kv_rmsnorm_quant(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Normalize Q/KV and quantize Q with FlashInfer's swizzled MXFP8 scales."""
    assert qr.ndim == kv.ndim == 2 and qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    assert qr.shape[1] % 32 == 0
    tokens, q_size = qr.shape
    kv_size = kv.shape[1]
    qo = torch.empty(qr.shape, dtype=torch.float8_e4m3fn, device=qr.device)
    kvo = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    padded_tokens = triton.cdiv(tokens, 128) * 128
    padded_groups = triton.cdiv(q_size // 32, 4) * 4
    scales = torch.empty(
        padded_tokens * padded_groups, dtype=torch.uint8, device=qr.device
    )
    if tokens:
        block = triton.next_power_of_2(max(q_size, kv_size))
        _q_kv_norm_quant_kernel[(padded_tokens, 2)](
            qr,
            kv,
            q_weight,
            kv_weight,
            qo,
            kvo,
            scales,
            tokens,
            qr.stride(0),
            kv.stride(0),
            eps,
            q_size,
            kv_size,
            block,
            current_platform.is_arch_support_pdl(),
            num_warps=8 if block >= 2048 else 4,
        )
    return QuantizedActivation(qo, scales, qr.dtype, qr.shape, kMxfp8Dynamic), kvo


def can_fuse_query_quant(linears: list[torch.nn.Module]) -> bool:
    """Require both local projections to use the same MXFP8 activation ABI."""
    if not current_platform.is_cuda():
        return False
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )

    # QuantKey does not encode scale layout; restrict this producer to the
    # consumers that accept F8_128x4 swizzled scales.
    return all(
        getattr(linear, "input_quant_key", None) == kMxfp8Dynamic
        and type(getattr(getattr(linear, "quant_method", None), "kernel", None))
        in (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel)
        for linear in linears
    )
