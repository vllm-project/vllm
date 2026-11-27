# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 attention input glue in one Triton kernel.

Between the QKV (or Q) projection and the attention kernel every layer runs
q/k/v RMSNorm, neox RoPE on q and k, the per-tensor FP8 quantization of q and
the FP8 write of k and v into the paged KV cache. This kernel does all of it in
one pass over the projection output:

  * q: RMSNorm * weight -> RoPE -> / q_scale -> FP8, written to a new tensor;
  * k: RMSNorm * weight -> RoPE -> / k_scale -> FP8, written at slot_mapping;
  * v: weightless RMSNorm -> / v_scale -> FP8, written at slot_mapping.

The bf16 rounding points of the compiled unfused chain are kept (the norm
output x * rsqrt * w, the RoPE output), so the FP8 bytes agree with it up to
rare one-step differences from the fp32 reduction order. KV-shared layers only run the q part.
"""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

logger = init_logger(__name__)


@triton.jit
def _norm_rope_rows(
    x_ptr,
    row_off,
    w_ptr,
    cs_row_ptr,
    eps,
    offs,
    HALF: tl.constexpr,
    ROT: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    # x_ptr + row_off[:, None] + offs: [R, HALF] tiles of the two neox halves.
    p1 = x_ptr + row_off[:, None] + offs[None, :]
    x1 = tl.load(p1).to(tl.float32)
    x2 = tl.load(p1 + HALF).to(tl.float32)
    ss = tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)
    r = tl.rsqrt(ss * (1.0 / (2 * HALF)) + eps)
    n1 = x1 * r[:, None]
    n2 = x2 * r[:, None]
    if HAS_W:
        w1 = tl.load(w_ptr + offs).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + offs).to(tl.float32)
        n1 = n1 * w1[None, :]
        n2 = n2 * w2[None, :]
    n1 = n1.to(tl.bfloat16)
    n2 = n2.to(tl.bfloat16)
    if HAS_ROPE:
        rot = offs < ROT
        c = tl.load(cs_row_ptr + offs, mask=rot, other=1.0).to(tl.float32)
        s = tl.load(cs_row_ptr + HALF + offs, mask=rot, other=0.0).to(tl.float32)
        a = n1.to(tl.float32)
        b = n2.to(tl.float32)
        n1 = (a * c[None, :] - b * s[None, :]).to(tl.bfloat16)
        n2 = (b * c[None, :] + a * s[None, :]).to(tl.bfloat16)
    return n1.to(tl.float32), n2.to(tl.float32)


@triton.jit
def _fp8(x, scale, FP8_MAX: tl.constexpr, dtype: tl.constexpr):
    y = x * (1.0 / scale)
    return tl.minimum(tl.maximum(y, -FP8_MAX), FP8_MAX).to(dtype)


@triton.jit
def _e2m1_nibble(x, scale):
    # nearest OCP e2m1 code of x / scale (ties go to the smaller magnitude), sign in bit 3
    y = x / scale
    a = tl.abs(y)
    c = (
        (a > 0.25).to(tl.int32)
        + (a > 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a > 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a > 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return c | ((y < 0).to(tl.int32) << 3)


@triton.jit
def _qkv_glue_kernel(
    x_ptr,
    stride_x,
    pos_ptr,
    cs_ptr,
    stride_cs,
    qw_ptr,
    kw_ptr,
    qo_ptr,
    stride_qo,
    kc_ptr,
    vc_ptr,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_vb,
    stride_vs,
    stride_vh,
    slot_ptr,
    num_slots,
    block_size,
    qs_ptr,
    ks_ptr,
    vs_ptr,
    eps,
    FP8_MAX: tl.constexpr,
    NQ: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    ROT: tl.constexpr,
    RQ: tl.constexpr,
    RKV: tl.constexpr,
    WRITE_KV: tl.constexpr,
    KV4: tl.constexpr,
):
    HALF: tl.constexpr = HD // 2
    NQP: tl.constexpr = NQ // RQ
    NKP: tl.constexpr = NKV // RKV
    if WRITE_KV:
        NP: tl.constexpr = NQP + 2 * NKP
    else:
        NP: tl.constexpr = NQP
    pid = tl.program_id(0).to(tl.int64)
    t = pid // NP
    part = (pid % NP).to(tl.int32)
    offs = tl.arange(0, HALF)
    pos = tl.load(pos_ptr + t)
    cs_row = cs_ptr + pos * stride_cs
    fp8_ty: tl.constexpr = qo_ptr.dtype.element_ty
    x_row = x_ptr + t * stride_x

    if part < NQP:
        rbq = tl.arange(0, RQ)
        row_off = (part * RQ + rbq) * HD
        n1, n2 = _norm_rope_rows(x_row, row_off, qw_ptr, cs_row, eps, offs, HALF, ROT, True, True)
        qs = tl.load(qs_ptr)
        o = qo_ptr + t * stride_qo + row_off[:, None] + offs[None, :]
        tl.store(o, _fp8(n1, qs, FP8_MAX, fp8_ty))
        tl.store(o + HALF, _fp8(n2, qs, FP8_MAX, fp8_ty))
    elif WRITE_KV:
        rbk = tl.arange(0, RKV)
        if t < num_slots:
            slot = tl.load(slot_ptr + t)
            if slot >= 0:
                blk = slot // block_size
                off = slot % block_size
                if part < NQP + NKP:
                    h = (part - NQP) * RKV + rbk
                    n1, n2 = _norm_rope_rows(
                        x_row, NQ * HD + h * HD, kw_ptr, cs_row, eps, offs, HALF, ROT, True, True
                    )
                    ks = tl.load(ks_ptr)
                    o = kc_ptr + blk * stride_kb + off * stride_ks + h[:, None] * stride_kh + offs[None, :]
                    if KV4:
                        tl.store(o, (_e2m1_nibble(n1, ks) | (_e2m1_nibble(n2, ks) << 4)).to(tl.uint8))
                    else:
                        tl.store(o, _fp8(n1, ks, FP8_MAX, fp8_ty))
                        tl.store(o + HALF, _fp8(n2, ks, FP8_MAX, fp8_ty))
                else:
                    h = (part - NQP - NKP) * RKV + rbk
                    n1, n2 = _norm_rope_rows(
                        x_row, (NQ + NKV) * HD + h * HD, kw_ptr, cs_row, eps, offs, HALF, ROT, False, False
                    )
                    vs = tl.load(vs_ptr)
                    o = vc_ptr + blk * stride_vb + off * stride_vs + h[:, None] * stride_vh + offs[None, :]
                    if KV4:
                        tl.store(o, (_e2m1_nibble(n1, vs) | (_e2m1_nibble(n2, vs) << 4)).to(tl.uint8))
                    else:
                        tl.store(o, _fp8(n1, vs, FP8_MAX, fp8_ty))
                        tl.store(o + HALF, _fp8(n2, vs, FP8_MAX, fp8_ty))


_FP8_MAX = float(torch.finfo(current_platform.fp8_dtype()).max)


def qkv_glue(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor | None,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    key_cache: torch.Tensor | None,
    value_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rot_pairs: int,
    eps: float,
    num_warps: int = 1,
    rq: int | None = None,
    rkv: int | None = None,
    kv4: bool = False,
) -> torch.Tensor:
    rq = rq or num_heads
    rkv = rkv or num_kv_heads
    num_tokens = qkv.shape[0]
    out = torch.empty(
        (num_tokens, num_heads * head_dim),
        dtype=current_platform.fp8_dtype(),
        device=qkv.device,
    )
    if num_tokens == 0:
        return out
    write_kv = key_cache is not None and slot_mapping is not None
    nparts = num_heads // rq + (2 * (num_kv_heads // rkv) if write_kv else 0)
    grid = (num_tokens * nparts,)
    if write_kv:
        strides = (*key_cache.stride()[:3], *value_cache.stride()[:3])
        block_size = key_cache.shape[1]
        slots, nslots = slot_mapping, slot_mapping.shape[0]
    else:
        strides = (0,) * 6
        block_size, slots, nslots = 1, positions, 0
        key_cache = value_cache = out
        k_weight = q_weight
    _qkv_glue_kernel[grid](
        qkv,
        qkv.stride(0),
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_weight,
        k_weight,
        out,
        out.stride(0),
        key_cache,
        value_cache,
        *strides,
        slots,
        nslots,
        block_size,
        q_scale,
        k_scale,
        v_scale,
        eps,
        FP8_MAX=_FP8_MAX,
        NQ=num_heads,
        NKV=num_kv_heads,
        HD=head_dim,
        ROT=rot_pairs,
        RQ=rq,
        RKV=rkv,
        WRITE_KV=write_kv,
        KV4=kv4,
        num_warps=num_warps,
    )
    return out


def _gemma4_qkv_glue_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    layer_name: LayerNameType,
    write_kv: bool,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rot_pairs: int,
    eps: float,
) -> torch.Tensor:
    from vllm.model_executor.layers.attention.attention import get_attention_context

    logger.info_once("gemma4_qkv_glue kernel running (first call)")
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(
        _resolve_layer_name(layer_name)
    )
    key_cache = value_cache = None
    k_scale, v_scale = attn_layer._k_scale, attn_layer._v_scale
    impl = attn_layer.impl
    kv4 = bool(getattr(impl, "kv4", False))
    if write_kv and layer_slot_mapping is not None and kv_cache.numel() > 0:
        key_cache, value_cache = impl._split_kv_cache(kv_cache)
        dt = torch.uint8 if kv4 else impl.fp8_dtype
        key_cache = key_cache.view(dt)
        value_cache = value_cache.view(dt)
        if kv4:
            from vllm.v1.attention.ops.rocm_kv4_attention import scale_tensor

            k_scale = scale_tensor(impl.kv4_k_scale, qkv.device)
            v_scale = scale_tensor(impl.kv4_v_scale, qkv.device)
    return qkv_glue(
        qkv,
        positions,
        cos_sin_cache,
        q_weight,
        k_weight,
        attn_layer._q_scale,
        k_scale,
        v_scale,
        key_cache,
        value_cache,
        layer_slot_mapping if key_cache is not None else None,
        num_heads,
        num_kv_heads,
        head_dim,
        rot_pairs,
        eps,
        kv4=kv4,
    )


def _gemma4_qkv_glue_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    layer_name: LayerNameType,
    write_kv: bool,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rot_pairs: int,
    eps: float,
) -> torch.Tensor:
    return torch.empty(
        (qkv.shape[0], num_heads * head_dim),
        dtype=current_platform.fp8_dtype(),
        device=qkv.device,
    )


direct_register_custom_op(
    op_name="gemma4_qkv_glue",
    op_func=_gemma4_qkv_glue_impl,
    mutates_args=[],
    fake_impl=_gemma4_qkv_glue_fake,
)


def gemma4_qkv_glue(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    layer_name: str,
    write_kv: bool,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rot_pairs: int,
    eps: float,
) -> torch.Tensor:
    """Returns the FP8 query; the FP8 k/v of write_kv layers land in the cache."""
    return torch.ops.vllm.gemma4_qkv_glue(
        qkv,
        positions,
        cos_sin_cache,
        q_weight,
        k_weight,
        _encode_layer_name(layer_name),
        write_kv,
        num_heads,
        num_kv_heads,
        head_dim,
        rot_pairs,
        eps,
    )
