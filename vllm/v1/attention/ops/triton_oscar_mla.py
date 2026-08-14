# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for OSCAR INT2 quantization of the MLA shared latent.

Follows the same split as the dense OSCAR path in ``triton_oscar_store.py``:
the dense rotation (``c_kv @ R``) is done externally with cuBLAS, and these
kernels handle quantize/pack and the fused dequant-in-decode.

Unlike the per-head dense path, MLA has a single latent shared by every head,
so there is one quantization group set per *token* rather than per
(token, head):

    storage   codes  uint8 [T, kv_lora_rank // 4]     (4 crumbs per byte)
              sb     fp32  [T, n_groups, 2]           (scale, bias) per group
              k_pe   bf16  [T, qk_rope_head_dim]      left unquantized

For GLM-5.2 (``kv_lora_rank=512``, ``group=128``, ``qk_rope_head_dim=64``)
that is 128 B of codes + 32 B of metadata = 160 B/token for ``c_kv``, against
1024 B in BF16 (6.4x).

The quantizer is Lloyd-Max for N(0, 1) and is kept **bit-exact** with
``mla_latent.fake_quant_int2_groupwise(lloyd_max=True)`` so the real-storage
path and the fake-quant measurement path agree.
"""

import torch

from vllm.triton_utils import tl, triton

# Lloyd-Max INT2 constants. These MUST match
# vllm/model_executor/layers/quantization/oscar/mla_latent.py.
_LM_T0 = -0.9810652732849121
_LM_T1 = 0.0
_LM_T2 = 0.9810652732849121
_LM_C0 = -1.5095585584640503
_LM_SPAN = 3.0191171169281006  # C3 - C0
_LM_RATIO = 1.16


@triton.jit
def _pack_int2_kernel(
    x_ptr,  # [T, D] fp32/bf16 — already rotated c_kv
    codes_ptr,  # [T, D // 4] uint8
    sb_ptr,  # [T, G_N, 2] fp32 (scale, bias)
    D: tl.constexpr,
    G: tl.constexpr,  # group size along D
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    C0: tl.constexpr,
    SPAN3: tl.constexpr,  # _LM_SPAN / 3
    RATIO: tl.constexpr,
):
    """Lloyd-Max bucketize + 2-bit pack, one program per token."""
    t = tl.program_id(0)
    n_g: tl.constexpr = D // G
    for g in tl.static_range(n_g):
        offs = t * D + g * G + tl.arange(0, G)
        x = tl.load(x_ptr + offs).to(tl.float32)
        mean = tl.sum(x, axis=0) / G
        diff = x - mean
        std = tl.sqrt(tl.sum(diff * diff, axis=0) / G + 1e-8)
        z = diff / std
        q = (z >= T0).to(tl.uint8) + (z >= T1).to(tl.uint8) + (z >= T2).to(tl.uint8)
        # Store the affine dequant pair directly so decode needs no constants.
        scale = SPAN3 * RATIO * std
        bias = mean + C0 * RATIO * std
        tl.store(sb_ptr + (t * n_g + g) * 2 + 0, scale)
        tl.store(sb_ptr + (t * n_g + g) * 2 + 1, bias)
        # Byte b holds dims 4b..4b+3, crumb i at bits 2i..2i+1.
        qb = tl.reshape(q, (G // 4, 4))
        sh = tl.arange(0, 4) * 2
        packed = tl.sum(qb.to(tl.int32) << sh[None, :], axis=1).to(tl.uint8)
        tl.store(
            codes_ptr + t * (D // 4) + g * (G // 4) + tl.arange(0, G // 4),
            packed,
        )


@triton.jit
def _mla_decode_int2_kernel(
    qL_ptr,  # [H, D] query, already absorbed into latent space
    qpe_ptr,  # [H, DP] rope part of the query
    codes_ptr,  # [T, D // 4] uint8
    sb_ptr,  # [T, G_N, 2] fp32
    kpe_ptr,  # [T, DP]
    m_ptr,  # [H, SPLITS] running max
    l_ptr,  # [H, SPLITS] running sum
    acc_ptr,  # [H, SPLITS, D]
    T,
    SM_SCALE,
    D: tl.constexpr,
    DP: tl.constexpr,
    G: tl.constexpr,
    BT: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Fused MQA-absorb decode with inline INT2 dequant.

    One program per (head, split). Each t-block is dequantized once and the
    result feeds both the score dot and the p@V accumulate, so the structure
    matches a BF16 latent kernel exactly -- only the load path differs.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    n_g: tl.constexpr = D // G
    per = tl.cdiv(T, SPLITS)
    t_lo = s * per
    t_hi = tl.minimum(t_lo + per, T)

    offs_d = tl.arange(0, D)
    byte_idx = offs_d // 4
    shift = (offs_d % 4) * 2
    qL = tl.load(qL_ptr + h * D + offs_d).to(tl.float32)
    qpe = tl.load(qpe_ptr + h * DP + tl.arange(0, DP)).to(tl.float32)

    m_i = -1e30
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    for t0 in range(t_lo, t_hi, BT):
        offs_t = t0 + tl.arange(0, BT)
        mask_t = offs_t < t_hi
        raw = tl.load(
            codes_ptr + offs_t[:, None] * (D // 4) + byte_idx[None, :],
            mask=mask_t[:, None],
            other=0,
        )
        q = ((raw >> shift[None, :]) & 3).to(tl.float32)
        # Broadcast the per-group (scale, bias) across the full width.
        scale_f = tl.zeros((BT, D), dtype=tl.float32)
        bias_f = tl.zeros((BT, D), dtype=tl.float32)
        for g in tl.static_range(n_g):
            sc_g = tl.load(sb_ptr + (offs_t * n_g + g) * 2 + 0, mask=mask_t, other=1.0)
            bi_g = tl.load(sb_ptr + (offs_t * n_g + g) * 2 + 1, mask=mask_t, other=0.0)
            in_g = (offs_d[None, :] // G) == g
            scale_f = tl.where(in_g, sc_g[:, None], scale_f)
            bias_f = tl.where(in_g, bi_g[:, None], bias_f)
        c = q * scale_f + bias_f

        kpe = tl.load(
            kpe_ptr + offs_t[:, None] * DP + tl.arange(0, DP)[None, :],
            mask=mask_t[:, None],
            other=0.0,
        ).to(tl.float32)

        sc = tl.sum(c * qL[None, :], axis=1) + tl.sum(kpe * qpe[None, :], axis=1)
        sc = sc * SM_SCALE
        sc = tl.where(mask_t, sc, -1e30)

        m_new = tl.maximum(m_i, tl.max(sc, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(sc - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * c, axis=0)
        m_i = m_new

    tl.store(m_ptr + h * SPLITS + s, m_i)
    tl.store(l_ptr + h * SPLITS + s, l_i)
    tl.store(acc_ptr + (h * SPLITS + s) * D + tl.arange(0, D), acc)


def oscar_mla_pack_int2(
    c_kv: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and pack a rotated latent to INT2 codes.

    Args:
        c_kv: ``[T, kv_lora_rank]``, already rotated by ``R``.
        group_size: Quantization group along the latent dim; must divide it.

    Returns:
        ``(codes, sb)`` with ``codes`` ``uint8 [T, D // 4]`` and ``sb``
        ``fp32 [T, D // group_size, 2]`` holding ``(scale, bias)``.
    """
    assert c_kv.dim() == 2, f"expected [T, D], got {tuple(c_kv.shape)}"
    t, d = c_kv.shape
    assert d % group_size == 0, f"group_size {group_size} must divide D={d}"
    c_kv = c_kv.contiguous()
    codes = torch.empty((t, d // 4), dtype=torch.uint8, device=c_kv.device)
    sb = torch.empty((t, d // group_size, 2), dtype=torch.float32, device=c_kv.device)
    _pack_int2_kernel[(t,)](
        c_kv,
        codes,
        sb,
        D=d,
        G=group_size,
        T0=_LM_T0,
        T1=_LM_T1,
        T2=_LM_T2,
        C0=_LM_C0,
        SPAN3=_LM_SPAN / 3.0,
        RATIO=_LM_RATIO,
    )
    return codes, sb


def oscar_mla_decode_int2(
    q_latent: torch.Tensor,
    q_pe: torch.Tensor,
    codes: torch.Tensor,
    sb: torch.Tensor,
    k_pe: torch.Tensor,
    sm_scale: float,
    group_size: int = 128,
    block_t: int = 16,
    splits: int = 8,
) -> torch.Tensor:
    """MQA-absorb decode against an INT2-packed latent cache.

    Args:
        q_latent: ``[H, kv_lora_rank]`` query absorbed into latent space.
        q_pe: ``[H, qk_rope_head_dim]`` rope part of the query.
        codes: ``uint8 [T, kv_lora_rank // 4]`` from ``oscar_mla_pack_int2``.
        sb: ``fp32 [T, n_groups, 2]`` scale/bias pairs.
        k_pe: ``[T, qk_rope_head_dim]`` unquantized rope keys.
        sm_scale: Softmax scale.

    Returns:
        ``[H, kv_lora_rank]`` fp32 attention output in latent space.
    """
    h, d = q_latent.shape
    t = codes.shape[0]
    dp = q_pe.shape[1]
    dev = q_latent.device
    m = torch.empty((h, splits), dtype=torch.float32, device=dev)
    ln = torch.empty((h, splits), dtype=torch.float32, device=dev)
    acc = torch.empty((h, splits, d), dtype=torch.float32, device=dev)
    _mla_decode_int2_kernel[(h, splits)](
        q_latent.contiguous(),
        q_pe.contiguous(),
        codes.contiguous(),
        sb.contiguous(),
        k_pe.contiguous(),
        m,
        ln,
        acc,
        t,
        sm_scale,
        D=d,
        DP=dp,
        G=group_size,
        BT=block_t,
        SPLITS=splits,
    )
    # Standard flash-decode split reduction.
    m_g = m.max(dim=1, keepdim=True).values
    w = (m - m_g).exp()
    return (acc * w.unsqueeze(-1)).sum(1) / (ln * w).sum(1, keepdim=True)


def oscar_mla_dequant_int2(
    codes: torch.Tensor,
    sb: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Reference dequant of packed codes, for tests and the prefill path."""
    t, packed = codes.shape
    d = packed * 4
    shifts = torch.arange(4, device=codes.device, dtype=torch.uint8) * 2
    q = (codes.unsqueeze(-1) >> shifts) & 3  # [T, D // 4, 4]
    q = q.reshape(t, d).to(torch.float32)
    scale = sb[..., 0].repeat_interleave(group_size, dim=1)
    bias = sb[..., 1].repeat_interleave(group_size, dim=1)
    return q * scale + bias
