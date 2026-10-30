# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

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


# The KV's GPT-J RoPE dims (its last ones) and a cos || sin row's width.
ROPE_DIM = tl.constexpr(64)


class KVCacheInsert(NamedTuple):
    """A per-tensor FP8 paged cache that ``fused_q_kv_rmsnorm_quant`` also
    writes the normalized KV into, byte for byte as
    ``torch.ops._C.fused_deepseek_v4_kv_rope_insert`` writes it: the GPT-J RoPE
    of the last 64 dims, then E4M3 of the KV over ``scale``."""

    # [blocks, block_size, KV width] float8_e4m3fn.
    cache: torch.Tensor
    # [<= tokens] int64, -1 for none; the tokens past it insert nothing.
    slot_mapping: torch.Tensor
    # [tokens] int64.
    positions: torch.Tensor
    # [max_position, 64] FP32: each position's 32 cos, then its 32 sin.
    cos_sin_cache: torch.Tensor
    block_size: int
    # [1] FP32.
    scale: torch.Tensor


@triton.jit
def _gptj_rope(even, odd, cos, sin):
    """``even * cos - odd * sin`` and ``even * sin + odd * cos`` contracted as
    nvcc contracts them in ``fused_deepseek_v4_kv_rope_insert``:
    fma(even, cos, -(odd * sin)) and fma(even, sin, odd * cos)."""
    return tl.inline_asm_elementwise(
        "{\n.reg .f32 t;\n"
        "mul.rn.f32 t, $3, $5;\nneg.f32 t, t;\nfma.rn.f32 $0, $2, $4, t;\n"
        "mul.rn.f32 t, $3, $4;\nfma.rn.f32 $1, $2, $5, t;\n}",
        "=f,=f,f,f,f,f",
        [even, odd, cos, sin],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit(do_not_specialize=["num_tokens", "num_slots"])
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
    cache=None,
    slots=None,
    positions=None,
    cos_sin=None,
    kv_scale=None,
    num_slots=0,
    block_stride=0,
    token_stride=0,
    CACHE_BLOCK: tl.constexpr = 1,
    INSERT_KV: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    if LAUNCH_PDL and TRIGGER_FIRST:
        # Dependents read the outputs only after their own grid-dependency
        # wait, so triggering first only lets them launch sooner: a decode
        # step's query projection then loads its weights while this kernel
        # waits for its input.
        tl.extra.cuda.gdc_launch_dependents()
    if INSERT_KV:
        # Step inputs and buffers set at load time, which no kernel of the
        # step writes: loaded before the grid-dependency wait, but after the
        # trigger, as the cos/sin addresses stall on the position load.
        insert = (tl.program_id(1) == 1) & (row < num_slots)
        slot = tl.load(slots + row, mask=insert, other=-1)
        position = tl.load(positions + row, mask=insert, other=0)
        pair = tl.arange(0, BLOCK // 2) - (KV_SIZE - ROPE_DIM) // 2
        rope = insert & (pair >= 0) & (pair < ROPE_DIM // 2)
        cos = tl.load(cos_sin + position * ROPE_DIM + pair, mask=rope, other=0.0)
        sin = tl.load(
            cos_sin + position * ROPE_DIM + ROPE_DIM // 2 + pair, mask=rope, other=0.0
        )
        kv_inv_scale = tl.math.div_rn(1.0, tl.load(kv_scale))
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        if not TRIGGER_FIRST:
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
        offsets = mxfp8_scale_offset(row, groups, Q_SIZE // 32)
        tl.store(scales + offsets, sf, groups < padded_groups)
    elif row < num_tokens:
        y = _norm_row(kv, kvw, row, kv_stride, eps, KV_SIZE, BLOCK)
        tl.store(kvo + row * KV_SIZE + cols, y, cols < KV_SIZE)
        if INSERT_KV:
            # What fused_deepseek_v4_kv_rope_insert computes from kvo.
            y = y.to(kvo.dtype.element_ty).to(tl.float32)
            even, odd = tl.split(tl.reshape(y, (BLOCK // 2, 2)))
            roped_even, roped_odd = _gptj_rope(even, odd, cos, sin)
            even = tl.where(rope, roped_even, even)
            odd = tl.where(rope, roped_odd, odd)
            y = tl.reshape(tl.join(even, odd), (BLOCK,))
            y = tl.minimum(tl.maximum(y * kv_inv_scale, -448.0), 448.0)
            block = slot // CACHE_BLOCK
            dst = (
                cache
                + block * block_stride
                + (slot - block * CACHE_BLOCK) * token_stride
            )
            tl.store(dst + cols, y, (cols < KV_SIZE) & (slot >= 0))


def fused_q_kv_rmsnorm_quant(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    early_trigger: bool = False,
    kv_insert: KVCacheInsert | None = None,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Normalize Q/KV and quantize Q with FlashInfer's swizzled MXFP8 scales.
    With ``early_trigger``, the kernel triggers its PDL dependents before its
    own grid-dependency wait, for a caller whose query projection loads its
    weights before its wait. With ``kv_insert``, it also inserts the
    normalized KV into that cache."""
    assert qr.ndim == kv.ndim == 2 and qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    assert qr.shape[1] % 32 == 0
    tokens, q_size = qr.shape
    kv_size = kv.shape[1]
    qo = torch.empty(qr.shape, dtype=torch.float8_e4m3fn, device=qr.device)
    kvo = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    padded_tokens = triton.cdiv(tokens, 128) * 128
    scales = torch.empty(
        mxfp8_scale_bytes(tokens, q_size), dtype=torch.uint8, device=qr.device
    )
    if tokens:
        block = triton.next_power_of_2(max(q_size, kv_size))
        launch_pdl = current_platform.is_arch_support_pdl()
        insert: dict[str, object] = {}
        if kv_insert is not None:
            cache = kv_insert.cache
            insert = dict(
                cache=cache,
                slots=kv_insert.slot_mapping,
                positions=kv_insert.positions,
                cos_sin=kv_insert.cos_sin_cache,
                kv_scale=kv_insert.scale,
                num_slots=kv_insert.slot_mapping.shape[0],
                block_stride=cache.stride(0),
                token_stride=cache.stride(1),
                CACHE_BLOCK=kv_insert.block_size,
                INSERT_KV=True,
            )
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
            launch_pdl,
            launch_pdl and early_trigger,
            **insert,
            num_warps=8 if block >= 2048 else 4,
            launch_pdl=launch_pdl,
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
        get_input_quant_key(linear) == kMxfp8Dynamic
        and type(getattr(getattr(linear, "quant_method", None), "kernel", None))
        in (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel)
        for linear in linears
    )
