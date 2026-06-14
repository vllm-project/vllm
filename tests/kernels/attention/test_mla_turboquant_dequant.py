# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from tools/test_mla_tq_fused_dequant.py
"""Numerical equivalence sweep for fused MLA TurboQuant decode.

Compares ``fused_mla_tq_decode_stage1 + _decode_softmax_reducev_fwd`` against
a pure PyTorch oracle (``torch_ref_dequant`` materializing a bf16 KV cache,
then ``decode_attention_fwd_grouped`` over it) across a 48-case parameter
sweep.
"""

import math
from collections import namedtuple

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.mla.triton_mla_tq import (
    _pack_bits_rows,
    _unpack_bits_rows,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_decode_attention import (
    _decode_softmax_reducev_fwd,
)
from vllm.v1.attention.ops.triton_turboquant_mla_decode import (
    fused_mla_tq_decode_stage1,
)


def make_synth_cache(
    n_active,
    block_size,
    L,
    R,
    mse_bits,
    mse_bytes,
    kv_c_bytes,
    packed_bytes,
    device,
    kpe_fp8=False,
):
    """Construct a quantized cache with random indices/norms/k_pe.

    When kpe_fp8=True, the k_pe segment uses [R fp8 bytes | 2-byte fp16 scale]
    layout instead of 2*R bf16 bytes.
    """
    torch.manual_seed(0)
    _FP8_DTYPE = torch.float8_e4m3fn
    _FP8_MAX = 448.0
    idx = torch.randint(
        0, 1 << mse_bits, (n_active * block_size, L), device=device, dtype=torch.int64
    )
    packed_idx = _pack_bits_rows(idx, bits=mse_bits)
    norms_fp16 = (0.5 + 1.5 * torch.rand(n_active * block_size, device=device)).to(
        torch.float16
    )
    norms_bytes = norms_fp16.view(torch.uint8).view(-1, 2)
    kpe_bf = torch.randn(n_active * block_size, R, device=device, dtype=torch.bfloat16)
    if kpe_fp8:
        kpe_f32 = kpe_bf.to(torch.float32)
        max_abs = kpe_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_abs / _FP8_MAX
        x = (kpe_f32 / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
        kpe_data_bytes = x.view(torch.uint8).view(-1, R)
        scale_fp16 = scale.squeeze(-1).to(torch.float16)
        scale_bytes = scale_fp16.view(torch.uint8).view(-1, 2)
        kpe_bytes = torch.cat([kpe_data_bytes, scale_bytes], dim=-1)
        kpe_dec = (x.to(torch.float32) * scale).to(torch.bfloat16)
        kpe_ref = kpe_dec
    else:
        kpe_bytes = kpe_bf.view(torch.uint8).view(-1, 2 * R)
        kpe_ref = kpe_bf
    row = torch.cat([packed_idx, norms_bytes, kpe_bytes], dim=-1)
    assert row.shape[-1] == packed_bytes, (row.shape, packed_bytes)
    return (
        row.view(n_active, block_size, packed_bytes).contiguous(),
        idx,
        norms_fp16,
        kpe_ref,
    )


def torch_ref_dequant(
    cache,
    centroids_bf16,
    Pi_bf16,
    L,
    R,
    mse_bits,
    mse_bytes,
    kv_c_bytes,
    norm_correction,
    kpe_fp8=False,
):
    """Pure PyTorch reference dequant.

    Returns a bf16 ``(nb, bs, L+R)`` tensor whose ``[..., :L]`` slice is
    already ``@ Pi``-rotated (i.e. in the rotated frame the fused kernel
    operates in on the q-side).  This matches the contract used by the
    fused-path oracle: q is pre-rotated with ``Pi`` and the K cache used by
    the oracle attention call is also pre-rotated, so the bilinear form
    ``q^T k`` produces the same scores as the fused kernel.
    """
    _FP8_DTYPE = torch.float8_e4m3fn
    nb, bs, _ = cache.shape
    idx_bytes = cache[..., :mse_bytes].contiguous()
    norms_bytes = cache[..., mse_bytes : mse_bytes + 2].contiguous()

    idx = _unpack_bits_rows(idx_bytes, bits=mse_bits, D=L).to(torch.int32)
    y_hat = centroids_bf16[idx.to(torch.int64)]
    if norm_correction:
        y32 = y_hat.to(torch.float32)
        c_norm = y32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        y_hat = (y32 / c_norm).to(torch.bfloat16)
    vec_norm = norms_bytes.view(torch.float16).to(torch.bfloat16)
    kv_c_pre_rot = y_hat * vec_norm
    kv_c = kv_c_pre_rot.view(nb * bs, L) @ Pi_bf16
    out = torch.empty(nb, bs, L + R, dtype=torch.bfloat16, device=cache.device)
    out[..., :L] = kv_c.view(nb, bs, L)
    if kpe_fp8:
        kpe_fp8_bytes = cache[..., kv_c_bytes : kv_c_bytes + R].contiguous()
        kpe_fp8_t = kpe_fp8_bytes.view(_FP8_DTYPE)
        scale_bytes = cache[..., kv_c_bytes + R : kv_c_bytes + R + 2].contiguous()
        scale_fp16 = scale_bytes.view(torch.float16)
        scale_bf = scale_fp16.to(torch.bfloat16)
        out[..., L:] = kpe_fp8_t.to(torch.bfloat16) * scale_bf
    else:
        out[..., L:] = cache[..., kv_c_bytes:].contiguous().view(torch.bfloat16)
    return out


Tol = namedtuple("Tol", "atol rtol min_cos")


def _tolerance_for(mse_bits, norm_correction, kpe_fp8):
    base_atol = 5e-3 if mse_bits == 4 else 8e-3
    if kpe_fp8:
        base_atol += 2e-3
    min_cos = 0.999 if mse_bits == 4 else 0.997
    return Tol(atol=base_atol, rtol=5e-3, min_cos=min_cos)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("L", [128, 256, 512])
@pytest.mark.parametrize("R", [32, 64])
@pytest.mark.parametrize("mse_bits", [3, 4])
@pytest.mark.parametrize("kpe_fp8", [False, True])
@pytest.mark.parametrize("norm_correction", [False, True])
@torch.inference_mode()
def test_fused_mla_tq_decode_matches_torch_oracle(
    L,
    R,
    mse_bits,
    kpe_fp8,
    norm_correction,
):
    device = torch.device("cuda")
    B = 2
    H_q = 16
    num_kv_splits = 4
    page_size = 64
    num_pages_per_seq = 4
    seqlen = page_size * num_pages_per_seq
    n_active = B * num_pages_per_seq
    sm_scale = 1.0 / math.sqrt(L + R)

    mse_bytes = math.ceil(L * mse_bits / 8)
    kv_c_bytes = mse_bytes + 2
    kpe_bytes = (R + 2) if kpe_fp8 else 2 * R
    packed_bytes = kv_c_bytes + kpe_bytes

    cache, _, _, _ = make_synth_cache(
        n_active=n_active,
        block_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        packed_bytes=packed_bytes,
        device=device,
        kpe_fp8=kpe_fp8,
    )

    centroids = get_centroids(L, mse_bits).to(device=device, dtype=torch.bfloat16)
    Pi = _build_hadamard(L, str(device)).to(torch.bfloat16)

    req_to_tokens = torch.arange(n_active, device=device, dtype=torch.int32).view(
        B, num_pages_per_seq
    )
    b_seqlen = torch.full((B,), seqlen, device=device, dtype=torch.int32)

    torch.manual_seed(1)
    q = torch.randn(B, H_q, L + R, device=device, dtype=torch.bfloat16)
    q_rot = q.clone()
    q_rot[..., :L] = (q[..., :L].reshape(-1, L) @ Pi).view(B, H_q, L)

    attn_logits = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    fused_mla_tq_decode_stage1(
        q_rot,
        cache,
        centroids,
        attn_logits,
        req_to_tokens,
        b_seqlen,
        sm_scale=sm_scale,
        page_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=kpe_fp8,
        num_kv_splits=num_kv_splits,
    )
    o_fused_rot = torch.empty(B, H_q, L, device=device, dtype=torch.bfloat16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=torch.bfloat16)
    v_shape_holder = torch.empty((1, L), device=device, dtype=torch.bfloat16)
    _decode_softmax_reducev_fwd(
        attn_logits,
        q_rot,
        o_fused_rot,
        lse_fused,
        v_shape_holder,
        b_seqlen,
        num_kv_splits,
    )
    o_fused = (o_fused_rot.reshape(-1, L) @ Pi).view(B, H_q, L).contiguous()

    kv_full = torch_ref_dequant(
        cache,
        centroids,
        Pi,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=kpe_fp8,
    )
    # PyTorch attention oracle.  ``kv_full[..., :L]`` is the original-frame
    # ``kv_c`` (i.e. (y_hat*vec_norm) @ Pi); ``kv_full[..., L:]`` is k_pe.
    # The fused kernel computes scores as (q@Pi)·(y*vn) and reduces V over
    # the same un-rotated (y*vn), then the caller un-rotates with @ Pi.  To
    # match that, the oracle uses the *original* q against the
    # original-frame kv cache: score = q · kv_c_orig = (q@Pi) · (y*vn),
    # and o_ref = sum_s p_s · kv_c_orig (already un-rotated, no @ Pi needed).
    kv_seq = kv_full.view(B, seqlen, L + R)
    k_seq = kv_seq
    v_seq = kv_seq[..., :L]
    q_f32 = q.to(torch.float32)
    k_f32 = k_seq.to(torch.float32)
    v_f32 = v_seq.to(torch.float32)
    scores = torch.einsum("bhd,bsd->bhs", q_f32, k_f32) * sm_scale
    p = torch.softmax(scores, dim=-1)
    o_ref = torch.einsum("bhs,bsd->bhd", p, v_f32).to(torch.bfloat16).contiguous()

    tol = _tolerance_for(mse_bits, norm_correction, kpe_fp8)
    torch.testing.assert_close(o_fused, o_ref, atol=tol.atol, rtol=tol.rtol)
    cos = torch.nn.functional.cosine_similarity(
        o_fused.flatten().float(),
        o_ref.flatten().float(),
        dim=0,
    ).item()
    assert cos >= tol.min_cos, f"cosine={cos} < {tol.min_cos}"


# ---------------------------------------------------------------------------
# FP8 key path (turboquant_k8v4)
# ---------------------------------------------------------------------------

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0


def make_synth_cache_fp8(
    n_active,
    block_size,
    L,
    R,
    k_pe_bytes,
    packed_bytes,
    device,
    kpe_fp8=False,
):
    """Construct an FP8 key cache with random kv_c / k_pe.

    Returns (cache_uint8, k_scale_tensor, kpe_ref_bf16).
    """
    torch.manual_seed(42)
    kv_c_bf = torch.randn(n_active * block_size, L, device=device, dtype=torch.bfloat16)
    # Per-tensor k_scale (layer-global).
    max_abs = kv_c_bf.abs().amax().clamp(min=1e-8)
    k_scale = max_abs / _FP8_MAX
    kv_c_fp8 = (
        (kv_c_bf.to(torch.float32) / k_scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    )
    kv_c_bytes = kv_c_fp8.view(torch.uint8).view(n_active * block_size, L)

    kpe_bf = torch.randn(n_active * block_size, R, device=device, dtype=torch.bfloat16)
    if kpe_fp8:
        kpe_f32 = kpe_bf.to(torch.float32)
        kpe_max = kpe_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = kpe_max / _FP8_MAX
        x = (kpe_f32 / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
        kpe_data = x.view(torch.uint8).view(n_active * block_size, R)
        scale_fp16 = scale.squeeze(-1).to(torch.float16)
        scale_bytes = scale_fp16.view(torch.uint8).view(n_active * block_size, 2)
        kpe_row = torch.cat([kpe_data, scale_bytes], dim=-1)
        kpe_ref = (x.to(torch.float32) * scale).to(torch.bfloat16)
    else:
        kpe_row = kpe_bf.view(torch.uint8).view(n_active * block_size, 2 * R)
        kpe_ref = kpe_bf

    row = torch.cat([kv_c_bytes, kpe_row], dim=-1)
    assert row.shape[-1] == packed_bytes, (row.shape, packed_bytes)
    return (
        row.view(n_active, block_size, packed_bytes).contiguous(),
        k_scale,
        kpe_ref,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("L", [128, 256, 512])
@pytest.mark.parametrize("R", [32, 64])
@pytest.mark.parametrize("kpe_fp8", [False, True])
@torch.inference_mode()
def test_fused_mla_tq_decode_fp8_keys(L, R, kpe_fp8):
    """FP8 key path (turboquant_k8v4): fused decode must match PyTorch oracle."""
    device = torch.device("cuda")
    B = 2
    H_q = 16
    num_kv_splits = 4
    page_size = 64
    num_pages_per_seq = 4
    seqlen = page_size * num_pages_per_seq
    n_active = B * num_pages_per_seq
    sm_scale = 1.0 / math.sqrt(L + R)

    kv_c_bytes = L  # 1 byte per fp8 element
    kpe_bytes_count = (R + 2) if kpe_fp8 else 2 * R
    packed_bytes = kv_c_bytes + kpe_bytes_count

    cache, k_scale, kpe_ref = make_synth_cache_fp8(
        n_active=n_active,
        block_size=page_size,
        L=L,
        R=R,
        k_pe_bytes=kpe_bytes_count,
        packed_bytes=packed_bytes,
        device=device,
        kpe_fp8=kpe_fp8,
    )

    # k_scale as device scalar tensor (matching production code).
    k_scale_tensor = k_scale.to(torch.float32).reshape(())

    req_to_tokens = torch.arange(n_active, device=device, dtype=torch.int32).view(
        B, num_pages_per_seq
    )
    b_seqlen = torch.full((B,), seqlen, device=device, dtype=torch.int32)

    torch.manual_seed(1)
    q = torch.randn(B, H_q, L + R, device=device, dtype=torch.bfloat16)

    # FP8 path: no Hadamard rotation on q.
    attn_logits = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    # centroids unused in FP8 mode — pass empty placeholder.
    centroids_placeholder = torch.empty(0, device=device, dtype=torch.bfloat16)
    fused_mla_tq_decode_stage1(
        q,
        cache,
        centroids_placeholder,
        attn_logits,
        req_to_tokens,
        b_seqlen,
        sm_scale=sm_scale,
        page_size=page_size,
        L=L,
        R=R,
        mse_bits=0,
        mse_bytes=0,
        kv_c_bytes=kv_c_bytes,
        norm_correction=False,
        kpe_fp8=kpe_fp8,
        key_fp8=True,
        k_scale=k_scale_tensor,
        num_kv_splits=num_kv_splits,
    )
    o_fused = torch.empty(B, H_q, L, device=device, dtype=torch.bfloat16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=torch.bfloat16)
    v_shape_holder = torch.empty((1, L), device=device, dtype=torch.bfloat16)
    _decode_softmax_reducev_fwd(
        attn_logits,
        q,
        o_fused,
        lse_fused,
        v_shape_holder,
        b_seqlen,
        num_kv_splits,
    )

    # PyTorch oracle: dequant fp8 → bf16, then standard attention.
    kv_c_fp8_view = cache[..., :L].contiguous().view(_FP8_DTYPE)
    kv_c_deq = (kv_c_fp8_view.to(torch.float32) * k_scale_tensor.to(torch.float32)).to(
        torch.bfloat16
    )
    if kpe_fp8:
        kpe_fp8_view = (
            cache[..., kv_c_bytes : kv_c_bytes + R].contiguous().view(_FP8_DTYPE)
        )
        scale_bytes = cache[..., kv_c_bytes + R : kv_c_bytes + R + 2].contiguous()
        scale_fp16 = scale_bytes.view(torch.float16).to(torch.bfloat16)
        kpe_deq = kpe_fp8_view.to(torch.bfloat16) * scale_fp16
    else:
        kpe_deq = cache[..., kv_c_bytes:].contiguous().view(torch.bfloat16)

    kv_full = torch.cat(
        [kv_c_deq.view(n_active, page_size, L), kpe_deq.view(n_active, page_size, R)],
        dim=-1,
    )
    kv_seq = kv_full.view(B, seqlen, L + R)
    # In MLA: K == V for the kv_c slice.
    k_seq = kv_seq
    v_seq = kv_seq[..., :L]
    q_f32 = q.to(torch.float32)
    k_f32 = k_seq.to(torch.float32)
    v_f32 = v_seq.to(torch.float32)
    scores = torch.einsum("bhd,bsd->bhs", q_f32, k_f32) * sm_scale
    p = torch.softmax(scores, dim=-1)
    o_ref = torch.einsum("bhs,bsd->bhd", p, v_f32).to(torch.bfloat16).contiguous()

    torch.testing.assert_close(o_fused, o_ref, atol=5e-3, rtol=5e-3)
    cos = torch.nn.functional.cosine_similarity(
        o_fused.flatten().float(), o_ref.flatten().float(), dim=0
    ).item()
    assert cos >= 0.999, f"cosine={cos}"


# ---------------------------------------------------------------------------
# End-to-end MSE roundtrip: quantize → packed cache → fused decode
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("L", [128, 512])
@pytest.mark.parametrize("mse_bits", [3, 4])
@pytest.mark.parametrize("norm_correction", [False, True])
@torch.inference_mode()
def test_mse_quant_store_decode_roundtrip(L, mse_bits, norm_correction):
    """End-to-end: quantize real kv_c → pack into uint8 cache → fused
    decode must match bf16 oracle that uses the original un-quantized kv_c.

    This tests the full pipeline (Hadamard rotation, Lloyd-Max bucketize,
    bit-pack, fused bit-unpack + centroid gather + vec_norm) against a
    ground truth built from the *original* bf16 kv_c, not from synthetic
    random indices.
    """
    R = 64
    B = 2
    H_q = 16
    page_size = 64
    num_pages_per_seq = 4
    seqlen = page_size * num_pages_per_seq
    n_active = B * num_pages_per_seq
    num_kv_splits = 4
    sm_scale = 1.0 / math.sqrt(L + R)
    device = torch.device("cuda")

    torch.manual_seed(99)
    # Generate latent kv_c (what the model would produce).
    n_tokens = n_active * page_size
    kv_c_orig = torch.randn(n_tokens, L, device=device, dtype=torch.bfloat16)
    k_pe_orig = torch.randn(n_tokens, R, device=device, dtype=torch.bfloat16)

    # Build the packed TQ cache.
    Pi = _build_hadamard(L, str(device))
    Pi_bf16 = Pi.to(torch.bfloat16)
    centroids = get_centroids(L, mse_bits).to(device=device, dtype=torch.bfloat16)

    # Quantize: same logic as TritonMLATurboQuantImpl._quantize_kv_c_mse.
    kv_c_f = kv_c_orig.to(torch.float32)
    norms = kv_c_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
    x_hat = kv_c_f / norms
    y = x_hat @ Pi.to(torch.float32)  # Pi is symmetric
    c_sorted, _ = centroids.to(torch.float32).sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    idx = torch.bucketize(y.contiguous(), midpoints).clamp(max=(1 << mse_bits) - 1)
    packed_idx = _pack_bits_rows(idx, bits=mse_bits)
    norms_fp16 = norms.squeeze(1).to(torch.float16)
    norms_bytes = norms_fp16.view(torch.uint8).view(-1, 2)
    k_pe_bytes = k_pe_orig.view(torch.uint8).view(-1, 2 * R)
    row = torch.cat([packed_idx, norms_bytes, k_pe_bytes], dim=-1)

    mse_bytes = math.ceil(L * mse_bits / 8)
    kv_c_bytes = mse_bytes + 2
    packed_bytes = kv_c_bytes + 2 * R
    assert row.shape[-1] == packed_bytes, (row.shape, packed_bytes)

    cache = row.view(n_active, page_size, packed_bytes).contiguous()

    # Build q (already rotated by Pi).
    q = torch.randn(B, H_q, L + R, device=device, dtype=torch.bfloat16)
    q_rot = q.clone()
    q_rot[..., :L] = (q[..., :L].reshape(-1, L) @ Pi_bf16).view(B, H_q, L)

    req_to_tokens = torch.arange(n_active, device=device, dtype=torch.int32).view(
        B, num_pages_per_seq
    )
    b_seqlen = torch.full((B,), seqlen, device=device, dtype=torch.int32)

    # Fused decode.
    attn_logits = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    fused_mla_tq_decode_stage1(
        q_rot,
        cache,
        centroids,
        attn_logits,
        req_to_tokens,
        b_seqlen,
        sm_scale=sm_scale,
        page_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=False,
        num_kv_splits=num_kv_splits,
    )
    o_fused_rot = torch.empty(B, H_q, L, device=device, dtype=torch.bfloat16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=torch.bfloat16)
    v_shape_holder = torch.empty((1, L), device=device, dtype=torch.bfloat16)
    _decode_softmax_reducev_fwd(
        attn_logits,
        q_rot,
        o_fused_rot,
        lse_fused,
        v_shape_holder,
        b_seqlen,
        num_kv_splits,
    )
    o_fused = (o_fused_rot.reshape(-1, L) @ Pi_bf16).view(B, H_q, L).contiguous()

    # Oracle: bf16 attention over the original kv_c + k_pe.
    kv_full = torch.cat([kv_c_orig, k_pe_orig], dim=-1)
    kv_seq = kv_full.view(B, seqlen, L + R)
    q_f32 = q.to(torch.float32)
    k_f32 = kv_seq.to(torch.float32)
    v_f32 = kv_seq[..., :L].to(torch.float32)
    scores = torch.einsum("bhd,bsd->bhs", q_f32, k_f32) * sm_scale
    p = torch.softmax(scores, dim=-1)
    o_ref = torch.einsum("bhs,bsd->bhd", p, v_f32).to(torch.bfloat16).contiguous()

    # Relaxed tolerance since quantization introduces systematic error.
    # 3-bit has higher quantization distortion; cosine similarity is the
    # primary quality metric.
    base_atol = 0.15 if mse_bits == 4 else 0.3
    tol = Tol(atol=base_atol, rtol=0.15, min_cos=0.93 if mse_bits == 3 else 0.96)
    torch.testing.assert_close(o_fused, o_ref, atol=tol.atol, rtol=tol.rtol)
    cos = torch.nn.functional.cosine_similarity(
        o_fused.flatten().float(), o_ref.flatten().float(), dim=0
    ).item()
    assert cos >= tol.min_cos, f"cosine={cos}"


# ---------------------------------------------------------------------------
# Store kernel tests: _mla_fused_store_fp8 and _mla_fused_store_mse
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("L", [128, 512])
@pytest.mark.parametrize("R", [32, 64])
@pytest.mark.parametrize("kpe_fp8", [False, True])
@torch.inference_mode()
def test_mla_fused_store_fp8(L, R, kpe_fp8):
    """Verify _mla_fused_store_fp8 writes correct FP8 kv_c + k_pe bytes."""
    from vllm.v1.attention.ops.triton_turboquant_store import _mla_fused_store_fp8

    _FP8_DTYPE = torch.float8_e4m3fn
    _FP8_MAX = 448.0
    device = torch.device("cuda")

    N = 8
    block_size = 16
    num_blocks = 2
    k_pe_bytes = R + 2 if kpe_fp8 else 2 * R
    packed_bytes = L + k_pe_bytes

    torch.manual_seed(42)
    kv_c = torch.randn(N, L, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(N, R, device=device, dtype=torch.bfloat16)
    k_scale = torch.tensor(1.5, device=device, dtype=torch.float32)

    # slot_mapping: map each of N tokens to a slot in the cache.
    slot_mapping = torch.arange(N, device=device, dtype=torch.int64)

    kv_cache = torch.zeros(
        num_blocks, block_size, packed_bytes, device=device, dtype=torch.uint8
    )

    _mla_fused_store_fp8(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        k_scale,
        kv_lora_rank=L,
        qk_rope_head_dim=R,
        k_pe_fp8=kpe_fp8,
    )

    # Verify kv_c bytes: kv_c / k_scale → clamp → fp8 → uint8.
    inv_scale = 1.0 / k_scale.item()
    kv_c_f32 = kv_c.to(torch.float32) * inv_scale
    kv_c_clamped = kv_c_f32.clamp(-_FP8_MAX, _FP8_MAX)
    kv_c_fp8_ref = kv_c_clamped.to(_FP8_DTYPE)
    kv_c_bytes_ref = kv_c_fp8_ref.view(torch.uint8)

    flat_cache = kv_cache.view(-1, packed_bytes)
    torch.testing.assert_close(flat_cache[:N, :L], kv_c_bytes_ref)

    # Verify k_pe bytes.
    if not kpe_fp8:
        # bf16 path: k_pe stored as 2*R bytes.
        k_pe_bf16_ref = k_pe.to(torch.bfloat16)
        k_pe_bytes_ref = k_pe_bf16_ref.view(torch.uint8)
        torch.testing.assert_close(flat_cache[:N, L:], k_pe_bytes_ref)
    else:
        # fp8 path: k_pe stored as R fp8 bytes + 2-byte fp16 scale.
        kpe_f32 = k_pe.to(torch.float32)
        absmax = kpe_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / _FP8_MAX
        kpe_scaled = kpe_f32 / scale
        kpe_clamped = kpe_scaled.clamp(-_FP8_MAX, _FP8_MAX)
        kpe_fp8_ref = kpe_clamped.to(_FP8_DTYPE)
        kpe_data_ref = kpe_fp8_ref.view(torch.uint8)
        scale_fp16_ref = scale.squeeze(-1).to(torch.float16)
        scale_bytes_ref = scale_fp16_ref.view(torch.uint8).view(-1, 2)

        torch.testing.assert_close(flat_cache[:N, L : L + R], kpe_data_ref)
        torch.testing.assert_close(flat_cache[:N, L + R : L + R + 2], scale_bytes_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("L", [128, 512])
@pytest.mark.parametrize("R", [32, 64])
@pytest.mark.parametrize("mse_bits", [3, 4])
@pytest.mark.parametrize("kpe_fp8", [False, True])
@torch.inference_mode()
def test_mla_fused_store_mse(L, R, mse_bits, kpe_fp8):
    """Verify _mla_fused_store_mse writes correct packed MSE indices + norm
    + k_pe bytes. Compares against a pure-Python reference implementation.
    """
    from vllm.model_executor.layers.quantization.turboquant.centroids import (
        get_centroids,
    )
    from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
    from vllm.v1.attention.ops.triton_turboquant_store import _mla_fused_store_mse

    _FP8_DTYPE = torch.float8_e4m3fn
    _FP8_MAX = 448.0
    device = torch.device("cuda")

    N = 8
    block_size = 16
    num_blocks = 2
    mse_bytes = math.ceil(L * mse_bits / 8)
    kv_c_bytes = mse_bytes + 2
    k_pe_bytes = R + 2 if kpe_fp8 else 2 * R
    packed_bytes = kv_c_bytes + k_pe_bytes

    torch.manual_seed(123)
    kv_c = torch.randn(N, L, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(N, R, device=device, dtype=torch.bfloat16)

    Pi = _build_hadamard(L, str(device))
    PiT = Pi.to(torch.float32)
    cents = get_centroids(L, mse_bits).to(device=device, dtype=torch.float32)
    c_sorted, _ = cents.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    slot_mapping = torch.arange(N, device=device, dtype=torch.int64)
    kv_cache = torch.zeros(
        num_blocks, block_size, packed_bytes, device=device, dtype=torch.uint8
    )

    _mla_fused_store_mse(
        kv_c,
        k_pe,
        kv_cache,
        slot_mapping,
        PiT,
        midpoints,
        mse_bits=mse_bits,
        kv_lora_rank=L,
        qk_rope_head_dim=R,
        k_pe_fp8=kpe_fp8,
    )

    # Python reference: normalize + rotate + bucketize + pack.
    kv_c_f = kv_c.to(torch.float32)
    norms = kv_c_f.norm(dim=1)  # (N,)
    x_hat = kv_c_f / (norms.unsqueeze(1) + 1e-8)
    y = x_hat @ PiT  # (N, L) rotated
    idx = torch.bucketize(y.contiguous(), midpoints).clamp(max=(1 << mse_bits) - 1)
    packed_idx_ref = _pack_bits_rows(idx, bits=mse_bits)
    norms_fp16_ref = norms.to(torch.float16)
    norms_bytes_ref = norms_fp16_ref.view(torch.uint8).view(-1, 2)

    flat_cache = kv_cache.view(-1, packed_bytes)

    # Check MSE packed indices.
    torch.testing.assert_close(flat_cache[:N, :mse_bytes], packed_idx_ref)

    # Check vec_norm bytes.
    torch.testing.assert_close(
        flat_cache[:N, mse_bytes : mse_bytes + 2], norms_bytes_ref
    )

    # Check k_pe bytes.
    kpe_offset = kv_c_bytes
    if not kpe_fp8:
        k_pe_bf16_ref = k_pe.to(torch.bfloat16)
        k_pe_bytes_ref = k_pe_bf16_ref.view(torch.uint8)
        torch.testing.assert_close(flat_cache[:N, kpe_offset:], k_pe_bytes_ref)
    else:
        kpe_f32 = k_pe.to(torch.float32)
        absmax = kpe_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / _FP8_MAX
        kpe_scaled = kpe_f32 / scale
        kpe_clamped = kpe_scaled.clamp(-_FP8_MAX, _FP8_MAX)
        kpe_fp8_ref = kpe_clamped.to(_FP8_DTYPE)
        kpe_data_ref = kpe_fp8_ref.view(torch.uint8)
        scale_fp16_ref = scale.squeeze(-1).to(torch.float16)
        scale_bytes_ref = scale_fp16_ref.view(torch.uint8).view(-1, 2)

        torch.testing.assert_close(
            flat_cache[:N, kpe_offset : kpe_offset + R], kpe_data_ref
        )
        torch.testing.assert_close(
            flat_cache[:N, kpe_offset + R : kpe_offset + R + 2],
            scale_bytes_ref,
        )


# ---------------------------------------------------------------------------
# Boundary condition tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("mse_bits", [3, 4])
@torch.inference_mode()
def test_dequant_L1024(mse_bits):
    """L=1024 stress test: maximum shared-memory pressure on the kernel."""
    L = 1024
    R = 64
    kpe_fp8 = False
    norm_correction = True
    device = torch.device("cuda")
    B = 2
    H_q = 16
    page_size = 64
    num_pages_per_seq = 4
    seqlen = page_size * num_pages_per_seq
    n_active = B * num_pages_per_seq
    num_kv_splits = 4
    sm_scale = 1.0 / math.sqrt(L + R)

    mse_bytes = math.ceil(L * mse_bits / 8)
    kv_c_bytes = mse_bytes + 2
    kpe_bytes = 2 * R
    packed_bytes = kv_c_bytes + kpe_bytes

    cache, _, _, _ = make_synth_cache(
        n_active=n_active,
        block_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        packed_bytes=packed_bytes,
        device=device,
        kpe_fp8=kpe_fp8,
    )

    centroids = get_centroids(L, mse_bits).to(device=device, dtype=torch.bfloat16)
    Pi = _build_hadamard(L, str(device)).to(torch.bfloat16)

    req_to_tokens = torch.arange(n_active, device=device, dtype=torch.int32).view(
        B, num_pages_per_seq
    )
    b_seqlen = torch.full((B,), seqlen, device=device, dtype=torch.int32)

    torch.manual_seed(1)
    q = torch.randn(B, H_q, L + R, device=device, dtype=torch.bfloat16)
    q_rot = q.clone()
    q_rot[..., :L] = (q[..., :L].reshape(-1, L) @ Pi).view(B, H_q, L)

    attn_logits = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    fused_mla_tq_decode_stage1(
        q_rot,
        cache,
        centroids,
        attn_logits,
        req_to_tokens,
        b_seqlen,
        sm_scale=sm_scale,
        page_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=kpe_fp8,
        num_kv_splits=num_kv_splits,
    )

    o_fused_rot = torch.empty(B, H_q, L, device=device, dtype=torch.bfloat16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=torch.bfloat16)
    v_shape_holder = torch.empty((1, L), device=device, dtype=torch.bfloat16)
    _decode_softmax_reducev_fwd(
        attn_logits,
        q_rot,
        o_fused_rot,
        lse_fused,
        v_shape_holder,
        b_seqlen,
        num_kv_splits,
    )
    o_fused = (o_fused_rot.reshape(-1, L) @ Pi).view(B, H_q, L).contiguous()

    kv_full = torch_ref_dequant(
        cache,
        centroids,
        Pi,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=kpe_fp8,
    )
    kv_seq = kv_full.view(B, seqlen, L + R)
    kv_seq_f32 = kv_seq.to(torch.float32)
    q_f32 = q.to(torch.float32)
    scores = torch.einsum("bhd,bsd->bhs", q_f32, kv_seq_f32) * sm_scale
    p = torch.softmax(scores, dim=-1)
    o_ref = (
        torch.einsum("bhs,bsd->bhd", p, kv_seq_f32[..., :L])
        .to(torch.bfloat16)
        .contiguous()
    )

    tol = _tolerance_for(mse_bits, norm_correction, kpe_fp8)
    torch.testing.assert_close(o_fused, o_ref, atol=tol.atol, rtol=tol.rtol)
    cos = torch.nn.functional.cosine_similarity(
        o_fused.flatten().float(), o_ref.flatten().float(), dim=0
    ).item()
    assert cos >= tol.min_cos, f"cosine={cos}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("page_size", [16, 32])
@torch.inference_mode()
def test_dequant_small_page_sizes(page_size):
    """Test minimum (16) and small (32) page sizes to verify addressing."""
    L = 128
    R = 32
    mse_bits = 4
    kpe_fp8 = False
    norm_correction = False
    device = torch.device("cuda")
    B = 2
    H_q = 8
    num_pages_per_seq = 8
    seqlen = page_size * num_pages_per_seq
    n_active = B * num_pages_per_seq
    num_kv_splits = 4
    sm_scale = 1.0 / math.sqrt(L + R)

    mse_bytes = math.ceil(L * mse_bits / 8)
    kv_c_bytes = mse_bytes + 2
    kpe_bytes = 2 * R
    packed_bytes = kv_c_bytes + kpe_bytes

    cache, _, _, _ = make_synth_cache(
        n_active=n_active,
        block_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        packed_bytes=packed_bytes,
        device=device,
        kpe_fp8=kpe_fp8,
    )

    centroids = get_centroids(L, mse_bits).to(device=device, dtype=torch.bfloat16)
    Pi = _build_hadamard(L, str(device)).to(torch.bfloat16)

    req_to_tokens = torch.arange(n_active, device=device, dtype=torch.int32).view(
        B, num_pages_per_seq
    )
    b_seqlen = torch.full((B,), seqlen, device=device, dtype=torch.int32)

    torch.manual_seed(7)
    q = torch.randn(B, H_q, L + R, device=device, dtype=torch.bfloat16)
    q_rot = q.clone()
    q_rot[..., :L] = (q[..., :L].reshape(-1, L) @ Pi).view(B, H_q, L)

    attn_logits = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    fused_mla_tq_decode_stage1(
        q_rot,
        cache,
        centroids,
        attn_logits,
        req_to_tokens,
        b_seqlen,
        sm_scale=sm_scale,
        page_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=kpe_fp8,
        num_kv_splits=num_kv_splits,
    )

    o_fused_rot = torch.empty(B, H_q, L, device=device, dtype=torch.bfloat16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=torch.bfloat16)
    v_shape_holder = torch.empty((1, L), device=device, dtype=torch.bfloat16)
    _decode_softmax_reducev_fwd(
        attn_logits,
        q_rot,
        o_fused_rot,
        lse_fused,
        v_shape_holder,
        b_seqlen,
        num_kv_splits,
    )
    o_fused = (o_fused_rot.reshape(-1, L) @ Pi).view(B, H_q, L).contiguous()

    kv_full = torch_ref_dequant(
        cache,
        centroids,
        Pi,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=kpe_fp8,
    )
    kv_seq = kv_full.view(B, seqlen, L + R)
    kv_seq_f32 = kv_seq.to(torch.float32)
    q_f32 = q.to(torch.float32)
    scores = torch.einsum("bhd,bsd->bhs", q_f32, kv_seq_f32) * sm_scale
    p = torch.softmax(scores, dim=-1)
    o_ref = (
        torch.einsum("bhs,bsd->bhd", p, kv_seq_f32[..., :L])
        .to(torch.bfloat16)
        .contiguous()
    )

    tol = _tolerance_for(mse_bits, norm_correction, kpe_fp8)
    torch.testing.assert_close(o_fused, o_ref, atol=tol.atol, rtol=tol.rtol)
    cos = torch.nn.functional.cosine_similarity(
        o_fused.flatten().float(), o_ref.flatten().float(), dim=0
    ).item()
    assert cos >= tol.min_cos, f"cosine={cos}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@torch.inference_mode()
def test_dequant_scattered_page_table():
    """Non-contiguous page table: pages are allocated in random order,
    which is the realistic scenario during serving.
    """
    L = 256
    R = 64
    mse_bits = 4
    kpe_fp8 = False
    norm_correction = True
    device = torch.device("cuda")
    B = 2
    H_q = 8
    page_size = 16
    num_pages_per_seq = 4
    seqlen = page_size * num_pages_per_seq
    n_active = B * num_pages_per_seq
    num_kv_splits = 4
    sm_scale = 1.0 / math.sqrt(L + R)

    mse_bytes = math.ceil(L * mse_bits / 8)
    kv_c_bytes = mse_bytes + 2
    kpe_bytes = 2 * R
    packed_bytes = kv_c_bytes + kpe_bytes

    # Build cache with extra "decoy" blocks to scatter into.
    total_blocks = n_active + 10
    cache, _, _, _ = make_synth_cache(
        n_active=total_blocks,
        block_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        packed_bytes=packed_bytes,
        device=device,
        kpe_fp8=kpe_fp8,
    )

    centroids = get_centroids(L, mse_bits).to(device=device, dtype=torch.bfloat16)
    Pi = _build_hadamard(L, str(device)).to(torch.bfloat16)

    # Scattered page table: random permutation of block indices.
    torch.manual_seed(55)
    perm = torch.randperm(total_blocks, device=device, dtype=torch.int32)
    req_to_tokens = perm[:n_active].view(B, num_pages_per_seq)
    b_seqlen = torch.full((B,), seqlen, device=device, dtype=torch.int32)

    q = torch.randn(B, H_q, L + R, device=device, dtype=torch.bfloat16)
    q_rot = q.clone()
    q_rot[..., :L] = (q[..., :L].reshape(-1, L) @ Pi).view(B, H_q, L)

    attn_logits = torch.empty(
        B, H_q, num_kv_splits, L + 1, device=device, dtype=torch.float32
    )
    fused_mla_tq_decode_stage1(
        q_rot,
        cache,
        centroids,
        attn_logits,
        req_to_tokens,
        b_seqlen,
        sm_scale=sm_scale,
        page_size=page_size,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=kpe_fp8,
        num_kv_splits=num_kv_splits,
    )

    o_fused_rot = torch.empty(B, H_q, L, device=device, dtype=torch.bfloat16)
    lse_fused = torch.empty(B, H_q, device=device, dtype=torch.bfloat16)
    v_shape_holder = torch.empty((1, L), device=device, dtype=torch.bfloat16)
    _decode_softmax_reducev_fwd(
        attn_logits,
        q_rot,
        o_fused_rot,
        lse_fused,
        v_shape_holder,
        b_seqlen,
        num_kv_splits,
    )
    o_fused = (o_fused_rot.reshape(-1, L) @ Pi).view(B, H_q, L).contiguous()

    # Build oracle from the scattered cache pages.
    kv_full = torch_ref_dequant(
        cache,
        centroids,
        Pi,
        L=L,
        R=R,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        kv_c_bytes=kv_c_bytes,
        norm_correction=norm_correction,
        kpe_fp8=kpe_fp8,
    )
    # Gather pages in scattered order.
    gathered_pages = kv_full[req_to_tokens.long()]  # (B, P, bs, L+R)
    kv_seq = gathered_pages.reshape(B, seqlen, L + R)
    kv_seq_f32 = kv_seq.to(torch.float32)
    q_f32 = q.to(torch.float32)
    scores = torch.einsum("bhd,bsd->bhs", q_f32, kv_seq_f32) * sm_scale
    p = torch.softmax(scores, dim=-1)
    o_ref = (
        torch.einsum("bhs,bsd->bhd", p, kv_seq_f32[..., :L])
        .to(torch.bfloat16)
        .contiguous()
    )

    tol = _tolerance_for(mse_bits, norm_correction, kpe_fp8)
    torch.testing.assert_close(o_fused, o_ref, atol=tol.atol, rtol=tol.rtol)
    cos = torch.nn.functional.cosine_similarity(
        o_fused.flatten().float(), o_ref.flatten().float(), dim=0
    ).item()
    assert cos >= tol.min_cos, f"cosine={cos}"
