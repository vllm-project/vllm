# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Q/K normalization, position transforms, and attention output gating."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice


@triton.jit(do_not_specialize=["SIZE"])
def _output_gate_kernel(
    X,
    G,
    B,
    Y,
    SIZE,
    G_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    FLOOR: tl.constexpr,
    TEMPERATURE: tl.constexpr,
    BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    dtype = G.dtype.element_ty
    g_idx = i // (HEADS * DIM) * G_STRIDE + i % (HEADS * DIM)
    g = tl.load(G + g_idx, i < SIZE, other=0).to(tl.float32)
    if BIAS:
        b = tl.load(B + i // DIM % HEADS).to(tl.float32)
        g = (g + b).to(dtype).to(tl.float32)
    g = (g / TEMPERATURE).to(dtype).to(tl.float32)
    sigmoid = (1.0 / (1.0 + tl.exp(-g))).to(dtype).to(tl.float32)
    multiplier = (sigmoid * (1.0 - FLOOR)).to(dtype).to(tl.float32)
    multiplier = (multiplier + FLOOR).to(dtype).to(tl.float32)
    x = tl.load(X + i, i < SIZE, other=0).to(tl.float32)
    tl.store(Y + i, x * multiplier, i < SIZE)


def beam_output_gate(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    gate_bias: torch.Tensor | None,
    *,
    num_heads: int,
    head_dim: int,
    floor: float,
    temperature: float,
) -> torch.Tensor:
    """Fuse gate operations while retaining activation-dtype rounding."""
    output = torch.empty_like(attn_output)
    if output.numel():
        if (
            attn_output.dtype == gate.dtype == torch.bfloat16
            and (gate_bias is None or gate_bias.dtype == torch.bfloat16)
            and current_platform.is_device_capability_family(100)
        ):
            from .attention_ops_cutedsl import run

            run(
                attn_output,
                gate,
                gate_bias,
                num_heads,
                head_dim,
                floor,
                temperature,
                output,
            )
            return output
        _output_gate_kernel[(triton.cdiv(output.numel(), 256),)](
            attn_output,
            gate,
            gate_bias,
            output,
            output.numel(),
            gate.stride(0),
            num_heads,
            head_dim,
            floor,
            temperature,
            gate_bias is not None,
            256,
            enable_fp_fusion=False,
        )
    return output


@triton.jit(
    do_not_specialize=["M", "NUM_SLOTS"], do_not_specialize_on_alignment=["POS"]
)
def _qk_norm_rope_kernel(
    Q,
    K,
    QW,
    KW,
    POS,
    CACHE,
    OQ,
    OK,
    M,
    V,
    KV_CACHE,
    SLOTS,
    NUM_SLOTS,
    WRITE_CACHE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_HEAD_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOKENS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    Q_WEIGHT: tl.constexpr,
    K_WEIGHT: tl.constexpr,
    ROTARY: tl.constexpr,
    NEOX: tl.constexpr,
    SSMAX: tl.constexpr,
    OFFSET: tl.constexpr,
    LOG_DIVISOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64) * TOKENS + tl.arange(0, TOKENS)
    head = tl.program_id(1)
    c = tl.arange(0, BLOCK)
    mask = (token[:, None] < M) & (c[None, :] < DIM)
    if head < Q_HEADS:
        x = tl.load(
            Q + token[:, None] * Q_STRIDE + head * DIM + c[None, :], mask, 0
        ).to(tl.float32)
        if BLOCK == 128:
            squares = (x * x).reshape((TOKENS, 4, 32))
            total = tl.sum(tl.sum(squares, 2), 1)
        else:
            total = tl.sum(x * x, 1)
        y = x * tl.rsqrt(total / DIM + EPS)[:, None]
        if Q_WEIGHT:
            y = y * tl.load(QW + c, c < DIM, 0).to(tl.float32)[None, :]
        out = OQ + (token[:, None] * Q_HEADS + head) * DIM
    else:
        h = head - Q_HEADS
        x = tl.load(K + token[:, None] * K_STRIDE + h * DIM + c[None, :], mask, 0).to(
            tl.float32
        )
        if BLOCK == 128:
            squares = (x * x).reshape((TOKENS, 4, 32))
            total = tl.sum(tl.sum(squares, 2), 1)
        else:
            total = tl.sum(x * x, 1)
        y = x * tl.rsqrt(total / DIM + EPS)[:, None]
        if K_WEIGHT:
            y = y * tl.load(KW + c, c < DIM, 0).to(tl.float32)[None, :]
        out = OK + (token[:, None] * K_HEADS + h) * DIM
    y = y.to(Q.dtype.element_ty).to(tl.float32)
    if ROTARY:
        half: tl.constexpr = ROTARY // 2
        if NEOX:
            partner = tl.where(c < half, c + half, c - half)
            second = c >= half
            offset = c % half
        else:
            partner = c ^ 1
            second = c % 2 != 0
            offset = c // 2
        partner = tl.minimum(tl.maximum(partner, 0), BLOCK - 1)
        other = tl.gather(y, tl.broadcast_to(partner[None, :], (TOKENS, BLOCK)), axis=1)
        pos = tl.load(POS + token, token < M, 0).to(tl.int64)
        rope_mask = (token[:, None] < M) & (c[None, :] < ROTARY)
        cos = tl.load(CACHE + pos[:, None] * ROTARY + offset[None, :], rope_mask, 0)
        sin = tl.load(
            CACHE + pos[:, None] * ROTARY + half + offset[None, :], rope_mask, 0
        )
        # Match the FP32 multiply/FMA ordering of the CUDA RoPE kernel.
        rotated = tl.where(
            second[None, :],
            tl.fma(other, sin, y * cos),
            tl.fma(y, cos, -other * sin),
        )
        y = tl.where(c[None, :] < ROTARY, rotated, y)
    elif SSMAX != "none":
        if head < Q_HEADS:
            p = tl.load(POS + token, token < M, 0).to(tl.float32) + OFFSET
            if SSMAX == "default":
                scale = libdevice.log(p) / LOG_DIVISOR
            elif SSMAX == "root10":
                scale = libdevice.exp(0.1 * libdevice.log1p(p / 2048.0))
            else:
                scale = tl.maximum(
                    130.0 * libdevice.expm1(0.001 * libdevice.log(p)), 1.0
                )
            y = y * scale[:, None]
    tl.store(out + c[None, :], y, mask)
    if WRITE_CACHE and head >= Q_HEADS:
        slot = tl.load(SLOTS + token, token < NUM_SLOTS, -1).to(tl.int64)
        cache_mask = mask & (token[:, None] < NUM_SLOTS) & (slot[:, None] >= 0)
        address = (
            slot[:, None] // PAGE_SIZE * CACHE_BLOCK_STRIDE
            + slot[:, None] % PAGE_SIZE * CACHE_TOKEN_STRIDE
            + (head - Q_HEADS) * CACHE_HEAD_STRIDE
            + c[None, :]
        )
        tl.store(KV_CACHE + address, y, cache_mask)
        value = tl.load(
            V + token[:, None] * V_STRIDE + (head - Q_HEADS) * DIM + c[None, :],
            cache_mask,
            0,
        )
        tl.store(KV_CACHE + address + DIM, value, cache_mask)


def beam_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    head_dim: int,
    rotary_dim: int = 0,
    is_neox_style: bool = True,
    ssmax_mode: str = "none",
    ssmax_offset: int = 1,
    ssmax_log_divisor: float = 1.0,
    *,
    value: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round norm before position transforms and optionally insert BF16 K/V."""
    if kv_cache is not None:
        assert value is not None and slot_mapping is not None
        assert kv_cache.dtype == q.dtype == torch.bfloat16
        assert kv_cache.ndim == 4 and kv_cache.stride(-1) == 1
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    q_heads, k_heads = q.shape[-1] // head_dim, k.shape[-1] // head_dim
    if q.shape[0]:
        if rotary_dim or q.shape[0] < 512:
            tile, warps = 4, 4
        elif q.shape[0] <= 2048:
            tile, warps = 16, 8
        else:
            tile, warps = 32, 4
        _qk_norm_rope_kernel[(triton.cdiv(q.shape[0], tile), q_heads + k_heads)](
            q,
            k,
            q_weight,
            k_weight,
            positions,
            cos_sin_cache,
            q_out,
            k_out,
            q.shape[0],
            value,
            kv_cache,
            slot_mapping,
            slot_mapping.numel() if slot_mapping is not None else 0,
            kv_cache is not None,
            value.stride(0) if value is not None else 0,
            kv_cache.stride(0) if kv_cache is not None else 0,
            kv_cache.stride(1) if kv_cache is not None else 0,
            kv_cache.stride(2) if kv_cache is not None else 0,
            kv_cache.shape[2] if kv_cache is not None else 1,
            tile,
            q_heads,
            k_heads,
            head_dim,
            q.stride(0),
            k.stride(0),
            eps,
            q_weight is not None,
            k_weight is not None,
            rotary_dim,
            is_neox_style,
            ssmax_mode,
            ssmax_offset,
            ssmax_log_divisor,
            triton.next_power_of_2(head_dim),
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return q_out, k_out
