# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
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


@triton.jit
def mxfp8_scale_offset(row, group, GROUPS: tl.constexpr):
    """F8_128x4: [row/128, group/4, row%32, row%128/32, group%4]."""
    offset = (row // 128 * triton.cdiv(GROUPS, 4) + group // 4) * 512
    return offset + row % 32 * 16 + row % 128 // 32 * 4 + group % 4


def mxfp8_scale_bytes(tokens: int, width: int) -> int:
    """Bytes of F8_128x4 scales for `tokens` rows of `width` MXFP8 values."""
    return triton.cdiv(tokens, 128) * 128 * triton.cdiv(width // 32, 4) * 4


@triton.jit
def _row_major_mxfp8(y, FP8_MAX: tl.constexpr, TINY: tl.constexpr, BLOCK: tl.constexpr):
    """Quantize one padded row. Same arithmetic as `_mxfp8_quant_triton_kernel`:
    per-32-block amax, biased E8M0 scale, reciprocal rescale, round to e4m3.
    The reciprocal (not `exp2(sb - 127)`) stays normal when `sb == 0`, which
    would otherwise flush to zero on CDNA and turn an all-zero block into NaN.
    """
    grouped = tl.reshape(y, (BLOCK // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(grouped), 1), TINY)
    sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
    sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
    rescale = tl.exp2(127.0 - sb)
    quantized = tl.reshape(grouped * rescale[:, None], (BLOCK,))
    return quantized, sb


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
    TRIGGER_FIRST: tl.constexpr,
    ROW_MAJOR_SCALES: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if LAUNCH_PDL:
        # Dependents read the outputs only after their own grid-dependency
        # wait, so triggering first only lets them launch sooner: a decode
        # step's query projection then loads its weights while this kernel
        # waits for its input.
        if TRIGGER_FIRST:
            tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
        if not TRIGGER_FIRST:
            tl.extra.cuda.gdc_launch_dependents()
    if tl.program_id(1) == 0:
        if ROW_MAJOR_SCALES:
            # [M, K / 32] E8M0, the layout RocmDotScaledMxfp8LinearKernel reads.
            # No 128-row padding: that exists only for the swizzled layout.
            if row < num_tokens:
                y = _norm_row(q, qw, row, q_stride, eps, Q_SIZE, BLOCK)
                # Same bf16 rounding the separate RMSNorm materializes before quant.
                y = y.to(q.dtype.element_ty).to(tl.float32)
                quantized, sb = _row_major_mxfp8(y, FP8_MAX, TINY, BLOCK)
                tl.store(
                    qo + row * Q_SIZE + cols,
                    quantized.to(qo.dtype.element_ty),
                    cols < Q_SIZE,
                )
                groups = tl.arange(0, BLOCK // 32)
                tl.store(
                    scales + row * (Q_SIZE // 32) + groups,
                    sb.to(tl.uint8),
                    groups < Q_SIZE // 32,
                )
        else:
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
            offsets = mxfp8_scale_offset(row, groups, Q_SIZE // 32)
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
    early_trigger: bool = False,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Normalize Q/KV and quantize Q with FlashInfer's swizzled MXFP8 scales.
    With `early_trigger`, the kernel triggers its PDL dependents before its
    own grid-dependency wait, for a caller whose query projection loads its
    weights before its wait.
    CUDA writes FlashInfer's swizzled F8_128x4 scales. ROCm writes row-major
    `[M, K / 32]` E8M0 scales, which is what the CDNA block-scaled GEMM reads,
    and skips the 128-row scale padding that layout needs
    """
    assert qr.ndim == kv.ndim == 2 and qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    assert qr.shape[1] % 32 == 0
    tokens, q_size = qr.shape
    kv_size = kv.shape[1]
    qo = torch.empty(qr.shape, dtype=torch.float8_e4m3fn, device=qr.device)
    kvo = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    # Swizzled scales are addressed in 128-row tiles, so the grid covers the
    # padding. Row-major scales are one byte per 32 values and need no padding.
    row_major = current_platform.is_rocm()
    grid_tokens = tokens if row_major else triton.cdiv(tokens, 128) * 128
    scales = (
        torch.empty((tokens, q_size // 32), dtype=torch.uint8, device=qr.device)
        if row_major
        else torch.empty(
            mxfp8_scale_bytes(tokens, q_size), dtype=torch.uint8, device=qr.device
        )
    )
    if tokens:
        block = triton.next_power_of_2(max(q_size, kv_size))
        launch_pdl = current_platform.is_arch_support_pdl()
        # The AMD Triton backend rejects the launch_pdl keyword, even when False.
        launch = {"launch_pdl": True} if launch_pdl else {}
        _q_kv_norm_quant_kernel[(grid_tokens, 2)](
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
            launch_pdl,
            launch_pdl and early_trigger,
            row_major,
            float(torch.finfo(torch.float8_e4m3fn).max),
            float(torch.finfo(torch.float32).tiny),
            num_warps=8 if block >= 2048 else 4,
            **launch,
        )
    return QuantizedActivation(qo, scales, qr.dtype, qr.shape, kMxfp8Dynamic), kvo


def can_fuse_query_quant(
    linears: list[torch.nn.Module], *, row_major: bool = False
) -> bool:
    """Require every linear to consume the MXFP8 activation this producer emits.

    The default producer writes FlashInfer's F8_128x4 swizzled scales. With
    `row_major` it writes the `[M, K / 32]` E8M0 scales the ROCm GEMM
    reads. QuantKey does not encode that layout, so each mode is limited to
    the kernels that accept it. Engram keeps the default: it writes swizzled
    scales from its own kernel.
    """
    if row_major:
        if not current_platform.is_rocm():
            return False
        from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
            RocmDotScaledMxfp8LinearKernel,
        )

        return all(
            get_input_quant_key(linear) == kMxfp8Dynamic
            and isinstance(
                kernel := getattr(
                    getattr(linear, "quant_method", None), "kernel", None
                ),
                RocmDotScaledMxfp8LinearKernel,
            )
            and kernel.supports_quantized_input(linear)
            for linear in linears
        )
    if not current_platform.is_cuda():
        return False
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.pytorch import (
        TorchMxFp8LinearKernel,
    )

    # QuantKey does not encode scale layout; restrict this producer to the
    # consumers that accept F8_128x4 swizzled scales.
    return all(
        get_input_quant_key(linear) == kMxfp8Dynamic
        and type(getattr(getattr(linear, "quant_method", None), "kernel", None))
        in (
            FlashInferCutedslMxfp8LinearKernel,
            FlashInferCutlassMxfp8LinearKernel,
            TorchMxFp8LinearKernel,
        )
        for linear in linears
    )
