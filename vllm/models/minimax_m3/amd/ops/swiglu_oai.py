# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SwiGLU-OAI activation (split layout) for AMD ROCm via Triton.

SwiGLU-OAI on a ``[*, 2I]`` split-layout input (gate = first half, up = second
half):

    gate = clamp(gate, max=limit)
    up   = clamp(up, -limit, +limit)
    out  = gate * sigmoid(alpha * gate) * (up + beta)

On ROCm the dense MLP and the native MXFP8 MoE (between its two GEMMs) fell back
to a chain of elementwise PyTorch ops with fp32 intermediates: vLLM's shared
``SiluAndMulWithClamp`` blanket-routes ROCm to ``forward_native``, and the MoE
applies the activation inline in PyTorch. This Triton kernel collapses that into
a single pass producing the ``[*, I]`` output directly, and computes in fp32
(rel ~1e-6 vs reference).

Note: the vectorized ``torch.ops._C.silu_and_mul_with_clamp`` op IS built on
ROCm and is ~1.2-2.2x faster in isolation, but the win is launch overhead that
HIP graphs already eliminate — measured end-to-end throughput is identical
(within noise), so we keep the fp32-accurate Triton kernel.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _swiglu_oai_kernel(
    g_ptr,
    out_ptr,
    n_inter,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    alpha,
    beta,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    pid_i = tl.program_id(1)
    cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < n_inter
    gate = tl.load(g_ptr + row * stride_gm + cols * stride_gn, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        g_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _swiglu_oai_quant_kernel(
    g_ptr,
    aq_ptr,
    as_ptr,
    M,
    n_inter,
    stride_gm,
    stride_gn,
    stride_qm,
    stride_qn,
    stride_sm,
    stride_sk,
    alpha,
    beta,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """SwiGLU-OAI (split layout) fused with per-32-block MXFP8 (E4M3 + E8M0)
    quant. Each program handles ``[BLOCK_M, 32]`` of the ``[M, I]`` output (one
    MX block): it reads the matching gate/up columns from ``g1`` (``[M, 2I]``),
    computes the SwiGLU in fp32, then derives the block E8M0 scale and emits the
    FP8 values + scale in a single pass — no bf16 ``act`` round-trip to HBM.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)  # which 32-element block along I
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_b * 32 + tl.arange(0, 32)
    m_mask = offs_m < M
    gate = tl.load(
        g_ptr + offs_m[:, None] * stride_gm + offs_c[None, :] * stride_gn,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        g_ptr + offs_m[:, None] * stride_gm + (n_inter + offs_c)[None, :] * stride_gn,
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    act = gate * tl.sigmoid(alpha * gate) * (up + beta)  # [BLOCK_M, 32] fp32
    amax = tl.maximum(tl.max(tl.abs(act), axis=1), 1e-30)  # [BLOCK_M]
    sb = tl.minimum(tl.maximum(tl.floor(tl.log2(amax)) + 127.0, 0.0), 254.0)
    descale = tl.exp2(sb - 127.0)
    aq = (act / descale[:, None]).to(aq_ptr.dtype.element_ty)
    tl.store(
        aq_ptr + offs_m[:, None] * stride_qm + offs_c[None, :] * stride_qn,
        aq,
        mask=m_mask[:, None],
    )
    tl.store(as_ptr + offs_m * stride_sm + pid_b * stride_sk, sb.to(tl.uint8), mask=m_mask)


def swiglu_oai_quantize_mxfp8(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    block_m: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU-OAI on split-layout ``[M, 2I]`` fused with MXFP8 activation-quant.

    Returns ``(act_q [M, I] float8_e4m3fn, act_scale [M, I//32] uint8 E8M0)``,
    identical to ``mxfp8_e4m3_quantize(swiglu_oai_split(gate_up))`` but in a
    single Triton pass (no bf16 intermediate). Used between the two GEMMs of the
    native MXFP8 MoE. Numerically equivalent to the unfused chain (bit-exact on
    measured MoE shapes); marginally more accurate (fp32 act, no bf16 round-trip).
    """
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
        MXFP8_SCALE_DTYPE,
        MXFP8_VALUE_DTYPE,
    )

    two_i = gate_up.shape[-1]
    n_inter = two_i // 2
    assert n_inter % MXFP8_BLOCK_SIZE == 0, (
        f"fused swiglu+quant needs I % {MXFP8_BLOCK_SIZE} == 0, got I={n_inter}"
    )
    g1 = gate_up.reshape(-1, two_i).contiguous()
    M = g1.shape[0]
    aq = torch.empty((M, n_inter), dtype=MXFP8_VALUE_DTYPE, device=g1.device)
    asc = torch.empty(
        (M, n_inter // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=g1.device
    )
    grid = (triton.cdiv(M, block_m), n_inter // MXFP8_BLOCK_SIZE)
    _swiglu_oai_quant_kernel[grid](
        g1,
        aq,
        asc,
        M,
        n_inter,
        g1.stride(0),
        g1.stride(1),
        aq.stride(0),
        aq.stride(1),
        asc.stride(0),
        asc.stride(1),
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        HAS_LIMIT=limit is not None,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return aq, asc


def swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """SwiGLU-OAI on a split-layout ``[*, 2I]`` tensor -> ``[*, I]``."""
    orig_shape = gate_up.shape
    two_i = orig_shape[-1]
    n_inter = two_i // 2
    x2 = gate_up.reshape(-1, two_i)
    m = x2.shape[0]
    dt = out_dtype if out_dtype is not None else gate_up.dtype
    out = torch.empty((m, n_inter), dtype=dt, device=gate_up.device)
    # Tile tuned on gfx950. The SwiGLU intermediate is sharded across tensor
    # parallel ranks (per-rank n_inter = I / tp: dense I=12288, MoE I=3072), and
    # a 512-wide tile (4 warps, ~2 elems/lane) only helps once the per-rank slice
    # is large enough to be bandwidth-bound — at TP=1 prefill that is ~1.25-1.35x
    # faster than 256. For small sharded slices (high TP) the kernel is launch-
    # bound (~12us) and a wide tile can slightly regress, so fall back to 256.
    # Decode is launch-bound at every TP. num_warps=8 underfills this tile, so it
    # is pinned to 4.
    block_i = 512 if n_inter >= 2048 else 256
    grid = (m, triton.cdiv(n_inter, block_i))
    _swiglu_oai_kernel[grid](
        x2,
        out,
        n_inter,
        x2.stride(0),
        x2.stride(1),
        out.stride(0),
        out.stride(1),
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        HAS_LIMIT=limit is not None,
        BLOCK_I=block_i,
        num_warps=4,
    )
    return out.reshape(*orig_shape[:-1], n_inter)
