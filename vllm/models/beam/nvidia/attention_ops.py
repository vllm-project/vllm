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
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    dtype = G.dtype.element_ty
    g_idx = i // (HEADS * DIM) * G_STRIDE + i % (HEADS * DIM)
    g = tl.load(G + g_idx, i < SIZE, other=0).to(tl.float32)
    b = tl.load(B + i // DIM % HEADS).to(tl.float32)
    g = (g + b).to(dtype).to(tl.float32)
    sigmoid = (1.0 / (1.0 + tl.exp(-g))).to(dtype).to(tl.float32)
    x = tl.load(X + i, i < SIZE, other=0).to(tl.float32)
    tl.store(Y + i, x * sigmoid, i < SIZE)


def beam_output_gate(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    gate_bias: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """``attn_output * sigmoid(gate + gate_bias)`` with activation-dtype rounding
    after the bias add and the sigmoid."""
    output = torch.empty_like(attn_output)
    if output.numel():
        if (
            attn_output.dtype == gate.dtype == gate_bias.dtype == torch.bfloat16
            and current_platform.is_device_capability_family(100)
        ):
            from .attention_ops_cutedsl import run

            run(attn_output, gate, gate_bias, num_heads, head_dim, output)
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
    Q_SCALE,
    K_SCALE,
    V_SCALE,
    WRITE_CACHE: tl.constexpr,
    FP8_Q: tl.constexpr,
    FP8_KV: tl.constexpr,
    FP8_MAX: tl.constexpr,
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
    ROTARY: tl.constexpr,
    NEOX: tl.constexpr,
    SSMAX: tl.constexpr,
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
        y = y * tl.load(QW + c, c < DIM, 0).to(tl.float32)[None, :]
        out = (token[:, None] * Q_HEADS + head) * DIM
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
        y = y * tl.load(KW + c, c < DIM, 0).to(tl.float32)[None, :]
        out = (token[:, None] * K_HEADS + h) * DIM
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
    elif SSMAX:
        if head < Q_HEADS:
            # (1 + (position + 1) / 2048) ** 0.1
            p = tl.load(POS + token, token < M, 0).to(tl.float32) + 1
            scale = libdevice.exp(0.1 * libdevice.log1p(p / 2048.0))
            y = y * scale[:, None]
    if head < Q_HEADS:
        if FP8_Q:
            q = y.to(Q.dtype.element_ty).to(tl.float32)
            q = q * tl.div_rn(1.0, tl.load(Q_SCALE))
            q = tl.maximum(tl.minimum(q, FP8_MAX), -FP8_MAX)
            tl.store(OQ + out + c[None, :], q, mask)
        else:
            tl.store(OQ + out + c[None, :], y, mask)
    else:
        tl.store(OK + out + c[None, :], y, mask)
    if WRITE_CACHE and head >= Q_HEADS:
        slot = tl.load(SLOTS + token, token < NUM_SLOTS, -1).to(tl.int64)
        cache_mask = mask & (token[:, None] < NUM_SLOTS) & (slot[:, None] >= 0)
        address = (
            slot[:, None] // PAGE_SIZE * CACHE_BLOCK_STRIDE
            + slot[:, None] % PAGE_SIZE * CACHE_TOKEN_STRIDE
            + (head - Q_HEADS) * CACHE_HEAD_STRIDE
            + c[None, :]
        )
        value = tl.load(
            V + token[:, None] * V_STRIDE + (head - Q_HEADS) * DIM + c[None, :],
            cache_mask,
            0,
        )
        if FP8_KV:
            y = y.to(V.dtype.element_ty).to(tl.float32)
            y = tl.div_rn(y, tl.load(K_SCALE))
            y = tl.clamp(y, -FP8_MAX, FP8_MAX, propagate_nan=tl.PropagateNan.ALL)
            value = tl.div_rn(value.to(tl.float32), tl.load(V_SCALE))
            value = tl.clamp(
                value, -FP8_MAX, FP8_MAX, propagate_nan=tl.PropagateNan.ALL
            )
        tl.store(KV_CACHE + address, y, cache_mask)
        tl.store(KV_CACHE + address + DIM, value, cache_mask)


def beam_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    head_dim: int,
    rotary_dim: int = 0,
    is_neox_style: bool = True,
    *,
    ssmax: bool = False,
    value: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round norm before position transforms; optionally insert K/V and quantize Q."""
    fp8_kv = kv_cache is not None and kv_cache.dtype == torch.float8_e4m3fn
    if kv_cache is not None:
        assert value is not None and slot_mapping is not None
        assert kv_cache.dtype in (q.dtype, torch.float8_e4m3fn)
        assert q.dtype == value.dtype == torch.bfloat16
        assert kv_cache.ndim == 4 and kv_cache.stride(-1) == 1
        assert not fp8_kv or (k_scale is not None and v_scale is not None)
    assert q_scale is None or q.dtype == torch.bfloat16
    q_out = torch.empty_like(
        q, dtype=torch.float8_e4m3fn if q_scale is not None else q.dtype
    )
    k_out = torch.empty_like(k)
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
            q_scale,
            k_scale,
            v_scale,
            kv_cache is not None,
            q_scale is not None,
            fp8_kv,
            torch.finfo(torch.float8_e4m3fn).max,
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
            rotary_dim,
            is_neox_style,
            ssmax,
            triton.next_power_of_2(head_dim),
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return q_out, k_out


def supports_fused_kv_insert(attn: torch.nn.Module) -> bool:
    """Whether the fused kernel can write this KV cache: BF16 or per-tensor FP8."""
    if attn.dtype != torch.bfloat16:
        return False
    if attn.kv_cache_torch_dtype == torch.bfloat16:
        return True
    if attn.kv_cache_dtype not in ("fp8", "fp8_e4m3"):
        return False
    scales = [attn._k_scale, attn._v_scale]
    if attn.query_quant is not None:
        scales.append(attn._q_scale)
    return all(scale.numel() == 1 for scale in scales)


def fused_kv_insert_kwargs(
    attn: torch.nn.Module, slot_mapping: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Cache arguments for the fused kernel, with scales when the cache is FP8."""
    kwargs = {"kv_cache": attn.kv_cache, "slot_mapping": slot_mapping}
    if attn.kv_cache_dtype in ("fp8", "fp8_e4m3"):
        kwargs["kv_cache"] = attn.kv_cache.view(torch.float8_e4m3fn)
        kwargs["k_scale"] = attn._k_scale
        kwargs["v_scale"] = attn._v_scale
        if attn.query_quant is not None:
            kwargs["q_scale"] = attn._q_scale
    return kwargs
