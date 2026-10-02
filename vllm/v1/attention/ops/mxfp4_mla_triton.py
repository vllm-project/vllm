# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA attention over the MXFP4 cache, in Triton, on gfx950.

Same arithmetic as the HIP kernel (mxfp4_mla_native.hip), selected with
GLM53_MXFP4_NATIVE_IMPL=triton. It ties HIP at decode sizes and is ~2x slower
from 512 rows up, so HIP is the default.

  QK   tl.dot_scaled(K codes E2M1 + E8M0 scales, Q E4M3)
       -> v_mfma_scale_f32_16x16x128_f8f6f4, K's scales applied by the core

PV has two modes (PV_MODE):

  "fp4"  (default) V stays 4-bit. V's scales run along the latent, not the
         token axis PV sums over, so they cannot use the scale slot; they are
         folded into P per 32-wide latent group: P'_g = E4M3(p * s_V[:, g]).
         Each group is one tl.dot_scaled(P'_g E4M3, V_g codes E2M1 packed
         along N) -> v_mfma_f32_16x16x128_f8f6f4. Same arithmetic as the HIP
         kernel.
  "bf16" The FlyDSL UltraQuant pathway: V dequantized by
         v_cvt_scalef32_pk_bf16_fp4 (group scale fused into the convert), P
         bf16, PV as a bf16 tl.dot that Triton feeds with ds_read_b64_tr_b16.
         4x the matrix work of "fp4" plus a bf16 LDS round trip: 3.57 vs 1.54
         ms per layer at 8192 rows against the HIP kernel.

Work split: each row's topk is cut into BLOCK_N-token chunks; a program walks
CHUNKS_PER_SPLIT of them with an online softmax and writes a normalized fp32
partial plus its log-sum-exp; the split-KV combine kernel merges them. The
split count adapts to the batch (many splits at decode sizes, one at prefill).
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

HEADS = 16
LATENT = 512
ROW_BYTES = 272
SCALE_OFF = 256
GROUP = 32
E4M3_MAX = 448.0
PV_MODE = "fp4"  # "fp4" or "bf16" (the FlyDSL pathway)
# Tokens per chunk, tuned per mode (fp4 PV needs a multiple of the MFMA's K=128).
BLOCK_N_BY_MODE = {"fp4": 256, "bf16": 64}
BLOCK_N = BLOCK_N_BY_MODE[PV_MODE]
NUM_WARPS = 4
NUM_STAGES = 1  # 2 stages: more LDS, fewer workgroups per CU; slower or equal in sweeps
TARGET_PROGRAMS = 2048
MAX_ROW_SPLITS_PER_LAUNCH = 4096  # caps fp32 partials at 128 MB

# v_cvt_scalef32_pk_bf16_fp4 dst, src, scale: two E2M1 values from src -> two
# bf16 in dst, each multiplied by the f32 scale. op_sel picks the nibble pair.
_CVT01 = tl.constexpr("v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2 op_sel:[0,0,0]")
_CVT23 = tl.constexpr("v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2 op_sel:[1,0,0]")
_CVT45 = tl.constexpr("v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2 op_sel:[0,1,0]")
_CVT67 = tl.constexpr("v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2 op_sel:[1,1,0]")


@triton.jit
def _cvt(words, scale, ASM: tl.constexpr):
    return tl.inline_asm_elementwise(
        asm=ASM,
        constraints="=v,v,v",
        args=[words, scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pair(p):
    # A bf16 is the top half of the matching f32: low half shifted up, high masked.
    return (p << 16).to(tl.float32, bitcast=True), (p & -65536).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _dequant_v(words, scale_f32, BLOCK: tl.constexpr, LAT: tl.constexpr):
    """[BLOCK, LAT//8] int32 words + per-word f32 scale -> bf16 [BLOCK, LAT].

    Value j of word w is latent 8w + j; all 8 share one E8M0 group (32 values
    = 4 words), so the scale is per word and the convert applies it.
    """
    v0, v1 = _pair(_cvt(words, scale_f32, _CVT01))
    v2, v3 = _pair(_cvt(words, scale_f32, _CVT23))
    v4, v5 = _pair(_cvt(words, scale_f32, _CVT45))
    v6, v7 = _pair(_cvt(words, scale_f32, _CVT67))
    even = tl.interleave(tl.interleave(v0, v4), tl.interleave(v2, v6))
    odd = tl.interleave(tl.interleave(v1, v5), tl.interleave(v3, v7))
    return tl.interleave(even, odd).to(tl.bfloat16)


@triton.jit
def _pv_group(p, sg, vg):
    """One latent group of the 4-bit PV: [H, 32] = E4M3(p * s_V[:, g])^T @ V codes.

    sg: [BLOCK] E8M0 bytes of this group; vg: [BLOCK, 16] bytes = 32 E2M1 codes
    packed along the latent. s_V is a power of two, so the fold is exact up to
    E4M3 range; masked tokens have p == 0 exactly and are selected to 0, so a
    stray scale byte cannot make 0 * inf.
    """
    sgf = (sg.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    w = tl.where(p == 0.0, 0.0, tl.minimum(p * sgf[:, None], 448.0))
    p8 = tl.trans(w).to(tl.float8e4nv)  # [H, BLOCK]
    return tl.dot_scaled(p8, None, "e4m3", vg, None, "e2m1", rhs_k_pack=False)


@triton.jit
def _cat(a, b, H: tl.constexpr, W: tl.constexpr):
    """[H, W] ++ [H, W] -> [H, 2W] along the last axis, in order."""
    return tl.reshape(tl.permute(tl.join(a, b), (0, 2, 1)), (H, 2 * W))


@triton.jit
def _pv_fp4(p, kv_ptr, base, valid, scl, BLOCK: tl.constexpr, H: tl.constexpr):
    """All 16 latent groups: [H, 512].

    V's bytes are read once more (a cache hit: QK just read the row) straight
    into a [token, byte, group] shape with the group axis innermost, so the
    four tl.split steps that carve out the groups divide an axis each thread
    already holds. Two slower alternatives were measured at 8192 rows: 16
    per-group re-reads plus scale reads (2.1 of 3.0 ms) and carving the groups
    out of the QK tile with a permute (3.3 ms total). Written out because
    Triton kernels take neither list comprehensions nor tuple().
    """
    byte = tl.arange(0, 16)
    grp = tl.arange(0, 16)
    v = tl.load(
        kv_ptr + base[:, None, None] + byte[None, :, None] + 16 * grp[None, None, :],
        mask=valid[:, None, None],
        other=0,
    )  # [token, byte, group]
    v = tl.reshape(v, (BLOCK, 16, 2, 2, 2, 2))
    sc = tl.reshape(scl, (BLOCK, 2, 2, 2, 2))
    v_0, v_1 = tl.split(v)
    v_0_0, v_0_1 = tl.split(v_0)
    v_1_0, v_1_1 = tl.split(v_1)
    v_0_0_0, v_0_0_1 = tl.split(v_0_0)
    v_0_1_0, v_0_1_1 = tl.split(v_0_1)
    v_1_0_0, v_1_0_1 = tl.split(v_1_0)
    v_1_1_0, v_1_1_1 = tl.split(v_1_1)
    v_0_0_0_0, v_0_0_0_1 = tl.split(v_0_0_0)
    v_0_0_1_0, v_0_0_1_1 = tl.split(v_0_0_1)
    v_0_1_0_0, v_0_1_0_1 = tl.split(v_0_1_0)
    v_0_1_1_0, v_0_1_1_1 = tl.split(v_0_1_1)
    v_1_0_0_0, v_1_0_0_1 = tl.split(v_1_0_0)
    v_1_0_1_0, v_1_0_1_1 = tl.split(v_1_0_1)
    v_1_1_0_0, v_1_1_0_1 = tl.split(v_1_1_0)
    v_1_1_1_0, v_1_1_1_1 = tl.split(v_1_1_1)
    v0 = v_0_0_0_0
    v1 = v_1_0_0_0
    v2 = v_0_1_0_0
    v3 = v_1_1_0_0
    v4 = v_0_0_1_0
    v5 = v_1_0_1_0
    v6 = v_0_1_1_0
    v7 = v_1_1_1_0
    v8 = v_0_0_0_1
    v9 = v_1_0_0_1
    v10 = v_0_1_0_1
    v11 = v_1_1_0_1
    v12 = v_0_0_1_1
    v13 = v_1_0_1_1
    v14 = v_0_1_1_1
    v15 = v_1_1_1_1
    s_0, s_1 = tl.split(sc)
    s_0_0, s_0_1 = tl.split(s_0)
    s_1_0, s_1_1 = tl.split(s_1)
    s_0_0_0, s_0_0_1 = tl.split(s_0_0)
    s_0_1_0, s_0_1_1 = tl.split(s_0_1)
    s_1_0_0, s_1_0_1 = tl.split(s_1_0)
    s_1_1_0, s_1_1_1 = tl.split(s_1_1)
    s_0_0_0_0, s_0_0_0_1 = tl.split(s_0_0_0)
    s_0_0_1_0, s_0_0_1_1 = tl.split(s_0_0_1)
    s_0_1_0_0, s_0_1_0_1 = tl.split(s_0_1_0)
    s_0_1_1_0, s_0_1_1_1 = tl.split(s_0_1_1)
    s_1_0_0_0, s_1_0_0_1 = tl.split(s_1_0_0)
    s_1_0_1_0, s_1_0_1_1 = tl.split(s_1_0_1)
    s_1_1_0_0, s_1_1_0_1 = tl.split(s_1_1_0)
    s_1_1_1_0, s_1_1_1_1 = tl.split(s_1_1_1)
    s0 = s_0_0_0_0
    s1 = s_1_0_0_0
    s2 = s_0_1_0_0
    s3 = s_1_1_0_0
    s4 = s_0_0_1_0
    s5 = s_1_0_1_0
    s6 = s_0_1_1_0
    s7 = s_1_1_1_0
    s8 = s_0_0_0_1
    s9 = s_1_0_0_1
    s10 = s_0_1_0_1
    s11 = s_1_1_0_1
    s12 = s_0_0_1_1
    s13 = s_1_0_1_1
    s14 = s_0_1_1_1
    s15 = s_1_1_1_1
    pv0 = _pv_group(p, s0, v0)
    pv1 = _pv_group(p, s1, v1)
    pv2 = _pv_group(p, s2, v2)
    pv3 = _pv_group(p, s3, v3)
    pv4 = _pv_group(p, s4, v4)
    pv5 = _pv_group(p, s5, v5)
    pv6 = _pv_group(p, s6, v6)
    pv7 = _pv_group(p, s7, v7)
    pv8 = _pv_group(p, s8, v8)
    pv9 = _pv_group(p, s9, v9)
    pv10 = _pv_group(p, s10, v10)
    pv11 = _pv_group(p, s11, v11)
    pv12 = _pv_group(p, s12, v12)
    pv13 = _pv_group(p, s13, v13)
    pv14 = _pv_group(p, s14, v14)
    pv15 = _pv_group(p, s15, v15)
    a0 = _cat(pv0, pv1, H, 32)
    a1 = _cat(pv2, pv3, H, 32)
    a2 = _cat(pv4, pv5, H, 32)
    a3 = _cat(pv6, pv7, H, 32)
    a4 = _cat(pv8, pv9, H, 32)
    a5 = _cat(pv10, pv11, H, 32)
    a6 = _cat(pv12, pv13, H, 32)
    a7 = _cat(pv14, pv15, H, 32)
    b0 = _cat(a0, a1, H, 64)
    b1 = _cat(a2, a3, H, 64)
    b2 = _cat(a4, a5, H, 64)
    b3 = _cat(a6, a7, H, 64)
    return _cat(_cat(b0, b1, H, 128), _cat(b2, b3, H, 128), H, 256)


@triton.jit
def _quantize_q_kernel(
    q_ptr, q_st, q_sh, q8_ptr, qs_ptr, H: tl.constexpr, LAT: tl.constexpr
):
    """Q -> E4M3 with a per-head absmax scale, bit-identical to quantize_q:
    scale = amax * fp32(1/448) (torch's tensor/scalar is a reciprocal
    multiply), elements divided with correct rounding (torch's tensor/tensor).
    """
    row = tl.program_id(0)
    h = tl.arange(0, H)
    d = tl.arange(0, LAT)
    x = tl.load(q_ptr + row * q_st + h[:, None] * q_sh + d[None, :]).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.where(amax > 0, amax * (1.0 / 448.0), 1.0)
    y = tl.div_rn(x, scale[:, None])
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)
    q8 = y.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(q8_ptr + (row * H + h[:, None]) * LAT + d[None, :], q8)
    tl.store(qs_ptr + row * H + h, scale)


@triton.jit
def _split_kernel(
    q8_ptr,
    qs_ptr,
    kv_ptr,
    idx_ptr,
    indptr_ptr,
    part_ptr,
    lse_ptr,
    num_slots,
    sm_scale,
    chunks_per_split,
    NUM_SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    H: tl.constexpr,
    LAT: tl.constexpr,
    ROW: tl.constexpr,
    SOFF: tl.constexpr,
    GRP: tl.constexpr,
    PV_FP4: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // NUM_SPLITS
    split = pid % NUM_SPLITS
    h = tl.arange(0, H)
    d = tl.arange(0, LAT)

    kv_start = tl.load(indptr_ptr + row)
    kv_len = tl.load(indptr_ptr + row + 1) - kv_start
    first = split * chunks_per_split * BLOCK
    n_chunks = tl.minimum(
        chunks_per_split, tl.cdiv(tl.maximum(kv_len - first, 0), BLOCK)
    )

    # Q as the rhs of dot_scaled: [LAT, H] E4M3, loaded once.
    q8 = tl.load(q8_ptr + (row * H + h[None, :]) * LAT + d[:, None]).to(
        tl.float8e4nv, bitcast=True
    )
    qs = tl.load(qs_ptr + row * H + h) * sm_scale

    m_i = tl.full((H,), float("-inf"), tl.float32)
    l_i = tl.zeros((H,), tl.float32)
    acc = tl.zeros((H, LAT), tl.float32)
    n = tl.arange(0, BLOCK)
    for c in range(n_chunks):
        pos = first + c * BLOCK + n
        valid = pos < kv_len
        slot = tl.load(idx_ptr + kv_start + pos, mask=valid, other=-1)
        valid = valid & (slot >= 0) & (slot < num_slots)
        base = tl.where(valid, slot, 0).to(tl.int64) * ROW

        # One gather of the packed rows, read two ways: bytes for QK, words for V.
        codes = tl.load(
            kv_ptr + base[:, None] + tl.arange(0, LAT // 2)[None, :],
            mask=valid[:, None],
            other=0,
        )
        scl = tl.load(
            kv_ptr + base[:, None] + SOFF + tl.arange(0, LAT // GRP)[None, :],
            mask=valid[:, None],
            other=127,
        )

        # QK on the scaled FP4 x FP8 MFMA: [BLOCK, H]
        s = tl.dot_scaled(codes, scl, "e2m1", q8, None, "e4m3")
        s = tl.where(valid[:, None], s * qs[None, :], float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(valid[:, None], tl.exp(s - m_new[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        if PV_FP4:
            pv = _pv_fp4(p, kv_ptr, base, valid, scl, BLOCK, H)
        else:
            # V: hardware convert with the group scale fused, then a bf16 dot.
            words = tl.load(
                (kv_ptr + base[:, None]).to(tl.pointer_type(tl.int32))
                + tl.arange(0, LAT // 8)[None, :],
                mask=valid[:, None],
                other=0,
            )
            wscale = tl.broadcast_to(
                scl[:, :, None], (BLOCK, LAT // GRP, GRP // 8)
            ).reshape(BLOCK, LAT // 8)
            wscale = (wscale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
            v = _dequant_v(words, wscale, BLOCK, LAT)
            pv = tl.dot(tl.trans(p.to(tl.bfloat16)), v)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    has = l_i > 0.0
    out = tl.where(has[:, None], acc / tl.where(has, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(has, m_i + tl.log(tl.where(has, l_i, 1.0)), float("-inf"))
    tl.store(part_ptr + (pid * H + h[:, None]) * LAT + d[None, :], out)
    tl.store(lse_ptr + pid * H + h, lse)


@triton.jit
def _combine_split_kv_kernel(
    part_ptr, lse_ptr, out_ptr, SPLITS: tl.constexpr, H: tl.constexpr, D: tl.constexpr
):
    """out[q, h] = sum_s e^(lse_s - LSE) part_s / sum_s e^(lse_s - LSE)."""
    q = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, D)
    lse = tl.load(lse_ptr + (q * SPLITS + s) * H + h)
    m = tl.max(lse, axis=0)
    w = tl.where(lse > float("-inf"), tl.exp(lse - m), 0.0)
    part = tl.load(part_ptr + ((q * SPLITS + s)[:, None] * H + h) * D + d[None, :])
    acc = tl.sum(w[:, None] * part, axis=0) / tl.maximum(tl.sum(w, axis=0), 1e-30)
    tl.store(out_ptr + (q * H + h) * D + d, acc.to(tl.bfloat16))


def choose_splits(nq: int, chunks: int) -> int:
    """One chunk per split, halved while the grid keeps TARGET_PROGRAMS."""
    splits = chunks
    while splits > 1 and nq * (splits // 2) >= TARGET_PROGRAMS:
        splits //= 2
    return splits


def supported(q: torch.Tensor, kv: torch.Tensor) -> bool:
    return (
        q.dim() == 3
        and q.shape[1] == HEADS
        and q.shape[2] == LATENT
        and kv.dtype == torch.uint8
        and kv.shape[-1] == ROW_BYTES
    )


def mxfp4_mla_triton(
    q: torch.Tensor,  # [nq, 16, 512] bf16
    kv: torch.Tensor,  # [num_slots, 272] uint8
    indices: torch.Tensor,  # ragged int32
    indptr: torch.Tensor,  # [nq + 1] int32
    sm_scale: float,
    max_topk: int,
    out: torch.Tensor,  # [nq, 16, 512] bf16
    num_splits: int | None = None,
) -> torch.Tensor:
    assert supported(q, kv) and q.dtype == torch.bfloat16 and q.stride(-1) == 1
    kv = kv.reshape(-1, ROW_BYTES)
    indices = indices.to(torch.int32).contiguous()
    indptr = indptr.to(torch.int32).contiguous()
    nq = q.shape[0]
    block = BLOCK_N_BY_MODE[PV_MODE]
    chunks = triton.cdiv(max_topk, block)
    # The combine indexes splits with tl.arange, which needs a power of two.
    splits = num_splits or choose_splits(nq, chunks)
    splits = triton.next_power_of_2(splits)
    cps = triton.cdiv(chunks, splits)
    rows_per_launch = max(1, MAX_ROW_SPLITS_PER_LAUNCH // splits)

    q8 = torch.empty(nq, HEADS, LATENT, dtype=torch.uint8, device=q.device)
    qs = torch.empty(nq, HEADS, dtype=torch.float32, device=q.device)
    _quantize_q_kernel[(nq,)](q, q.stride(0), q.stride(1), q8, qs, H=HEADS, LAT=LATENT)

    for r0 in range(0, nq, rows_per_launch):
        n = min(rows_per_launch, nq - r0)
        part = torch.empty(
            n * splits, HEADS, LATENT, dtype=torch.float32, device=q.device
        )
        lse = torch.empty(n * splits, HEADS, dtype=torch.float32, device=q.device)
        _split_kernel[(n * splits,)](
            q8[r0:],
            qs[r0:],
            kv,
            indices,
            indptr[r0:],
            part,
            lse,
            kv.shape[0],
            float(sm_scale),
            cps,
            NUM_SPLITS=splits,
            BLOCK=block,
            H=HEADS,
            LAT=LATENT,
            ROW=ROW_BYTES,
            SOFF=SCALE_OFF,
            GRP=GROUP,
            PV_FP4=PV_MODE == "fp4",
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        sub = out[r0 : r0 + n]
        tmp = sub if sub.is_contiguous() else torch.empty_like(sub)
        _combine_split_kv_kernel[(n, HEADS)](
            part, lse, tmp, SPLITS=splits, H=HEADS, D=LATENT
        )
        if tmp is not sub:
            sub.copy_(tmp)
    return out
