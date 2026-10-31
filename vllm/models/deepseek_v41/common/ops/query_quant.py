# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _norm_rows(
    x, weight, rows, live, stride, eps, SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    cols = tl.arange(0, BLOCK)
    w = tl.load(weight + cols, cols < SIZE, 0).to(tl.float32)
    mask = live[:, None] & (cols < SIZE)[None, :]
    values = tl.load(x + rows[:, None] * stride + cols[None, :], mask, 0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(values * values, 1) / SIZE + eps)
    return values * rrms[:, None] * w[None, :]


@triton.jit
def mxfp8_scale_offset(row, group, GROUPS: tl.constexpr):
    """F8_128x4: [row/128, group/4, row%32, row%128/32, group%4]."""
    offset = (row // 128 * triton.cdiv(GROUPS, 4) + group // 4) * 512
    return offset + row % 32 * 16 + row % 128 // 32 * 4 + group % 4


def mxfp8_scale_bytes(tokens: int, width: int) -> int:
    """Bytes of F8_128x4 scales for `tokens` rows of `width` MXFP8 values."""
    return triton.cdiv(tokens, 128) * 128 * triton.cdiv(width // 32, 4) * 4


class Mxfp8KVInsert(NamedTuple):
    """A packed MXFP8 SWA cache insert, byte for byte as
    ``torch.ops._C.fused_deepseek_v4_kv_rope_insert(..., kv_mxfp8=True)``."""

    # [blocks, block bytes] uint8.
    cache: torch.Tensor
    # [<= tokens] int64, -1 for none; tokens past it insert nothing.
    slot_mapping: torch.Tensor
    # [tokens] int64.
    positions: torch.Tensor
    # [max_position, 64] FP32: each position's 32 cos, then its 32 sin.
    cos_sin_cache: torch.Tensor
    block_size: int


@triton.jit
def _gptj_rope(even, odd, cos, sin):
    """GPT-J RoPE with the CUDA insert kernel's nvcc contraction:
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


@triton.jit
def _mxfp8_kv_insert(
    y,
    rows,
    cache,
    slots,
    positions,
    cos_sin,
    num_slots,
    block_stride,
    KV_SIZE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
):
    """Insert the [ROWS, KV_SIZE] FP32 normalized KV rows ``y`` (rounded to
    BF16 here, as the KV tensor is) into the packed MXFP8 SWA cache."""
    ROWS: tl.constexpr = y.shape[0]
    PAIRS: tl.constexpr = KV_SIZE // 2
    ROPE_PAIRS: tl.constexpr = 32
    insert = rows < num_slots
    slot = tl.load(slots + rows, insert, other=-1)
    position = tl.load(positions + rows, insert, other=0)
    pair = tl.arange(0, PAIRS) - (PAIRS - ROPE_PAIRS)
    rope = (pair >= 0)[None, :] & insert[:, None]
    cs = cos_sin + position[:, None] * (2 * ROPE_PAIRS) + pair[None, :]
    cos = tl.load(cs, rope, other=1.0)
    sin = tl.load(cs + ROPE_PAIRS, rope, other=0.0)
    y = y.to(tl.bfloat16).to(tl.float32)
    even, odd = tl.split(tl.reshape(y, (ROWS, PAIRS, 2)))
    roped_even, roped_odd = _gptj_rope(even, odd, cos, sin)
    even = tl.where(rope, roped_even, even)
    odd = tl.where(rope, roped_odd, odd)
    y = tl.reshape(tl.join(even, odd), (ROWS, KV_SIZE))
    y = y.to(tl.bfloat16).to(tl.float32)
    GROUPS: tl.constexpr = KV_SIZE // 32
    grouped = tl.reshape(y, (ROWS, GROUPS, 32))
    absmax = tl.maximum(tl.max(tl.abs(grouped), 2), 1e-4)
    exponent = tl.math.ceil(tl.extra.cuda.libdevice.log2(tl.math.div_rn(absmax, 448.0)))
    inv_scale = tl.extra.cuda.libdevice.exp2(-exponent)
    scaled = tl.minimum(tl.maximum(grouped * inv_scale[:, :, None], -448.0), 448.0)
    data = (
        tl.reshape(scaled, (ROWS, KV_SIZE)).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    )
    sf = tl.minimum(tl.maximum(exponent + 127.0, 0.0), 255.0).to(tl.uint8)
    block = slot // CACHE_BLOCK
    record = (
        cache + block * block_stride + (slot - block * CACHE_BLOCK) * (KV_SIZE + GROUPS)
    )
    live = slot >= 0
    c = tl.arange(0, KV_SIZE)
    tl.store(record[:, None] + c[None, :], data, live[:, None])
    g = tl.arange(0, GROUPS)
    tl.store(record[:, None] + KV_SIZE + g[None, :], sf, live[:, None])


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
    Q_BLOCK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
    cache=None,
    slots=None,
    positions=None,
    cos_sin=None,
    num_slots=0,
    block_stride=0,
    CACHE_BLOCK: tl.constexpr = 1,
    INSERT_KV: tl.constexpr = False,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = rows < num_tokens
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if tl.program_id(1) == 0:
        cols = tl.arange(0, Q_BLOCK)
        groups = tl.arange(0, Q_BLOCK // 32)
        y = _norm_rows(q, qw, rows, live, q_stride, eps, Q_SIZE, Q_BLOCK)
        # Preserve the materialized normalization output's rounding boundary.
        y = y.to(q.dtype.element_ty).to(tl.float32)
        grouped = tl.reshape(y, (ROWS, Q_BLOCK // 32, 32))
        amax = tl.max(tl.abs(grouped), 2)
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
        quantized = tl.reshape(grouped * inv_scale[:, :, None], (ROWS, Q_BLOCK))
        mask = live[:, None] & (cols < Q_SIZE)[None, :]
        tl.store(qo + rows[:, None] * Q_SIZE + cols[None, :], quantized, mask)
        # Rows past num_tokens up to the 128-row scale tile get zero scales.
        padded_groups: tl.constexpr = triton.cdiv(Q_SIZE // 32, 4) * 4
        sf = tl.where(live[:, None] & (groups < Q_SIZE // 32)[None, :], sf, 0)
        offsets = mxfp8_scale_offset(rows[:, None], groups[None, :], Q_SIZE // 32)
        tl.store(scales + offsets, sf, (groups < padded_groups)[None, :])
    else:
        kv_y = _norm_rows(kv, kvw, rows, live, kv_stride, eps, KV_SIZE, KV_BLOCK)
        if INSERT_KV:
            _mxfp8_kv_insert(
                kv_y,
                rows,
                cache,
                slots,
                positions,
                cos_sin,
                num_slots,
                block_stride,
                KV_SIZE,
                CACHE_BLOCK,
            )
        else:
            kv_cols = tl.arange(0, KV_BLOCK)
            kv_mask = live[:, None] & (kv_cols < KV_SIZE)[None, :]
            tl.store(kvo + rows[:, None] * KV_SIZE + kv_cols[None, :], kv_y, kv_mask)


def fused_q_kv_rmsnorm_quant(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    kv_insert: Mxfp8KVInsert | None = None,
) -> tuple[QuantizedActivation, torch.Tensor | None]:
    """Normalize Q/KV and quantize Q with FlashInfer's swizzled MXFP8 scales.
    With ``kv_insert`` the KV goes straight into the SWA cache instead."""
    assert qr.ndim == kv.ndim == 2 and qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()
    assert qr.shape[1] % 32 == 0
    tokens, q_size = qr.shape
    kv_size = kv.shape[1]
    qo = torch.empty(qr.shape, dtype=torch.float8_e4m3fn, device=qr.device)
    kvo = None
    if kv_insert is None:
        kvo = torch.empty(kv.shape, dtype=kv.dtype, device=kv.device)
    padded_tokens = triton.cdiv(tokens, 128) * 128
    scales = torch.empty(
        mxfp8_scale_bytes(tokens, q_size), dtype=torch.uint8, device=qr.device
    )
    if tokens:
        insert: dict[str, object] = {}
        if kv_insert is not None:
            assert kv_insert.positions.shape[0] == tokens
            insert = dict(
                cache=kv_insert.cache,
                slots=kv_insert.slot_mapping,
                positions=kv_insert.positions,
                cos_sin=kv_insert.cos_sin_cache,
                num_slots=kv_insert.slot_mapping.shape[0],
                block_stride=kv_insert.cache.stride(0),
                CACHE_BLOCK=kv_insert.block_size,
                INSERT_KV=True,
            )
        # Small batches take one row per program (8 warps below 512 tokens),
        # prefill-sized ones 4 rows; batch invariance keeps one launch shape.
        rows = 4 if tokens >= 1024 and not envs.VLLM_BATCH_INVARIANT else 1
        num_warps = 8 if tokens < 512 or envs.VLLM_BATCH_INVARIANT else 4
        launch_pdl = current_platform.is_arch_support_pdl()
        _q_kv_norm_quant_kernel[(padded_tokens // rows, 2)](
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
            triton.next_power_of_2(q_size),
            triton.next_power_of_2(kv_size),
            rows,
            launch_pdl,
            **insert,
            num_warps=num_warps,
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
