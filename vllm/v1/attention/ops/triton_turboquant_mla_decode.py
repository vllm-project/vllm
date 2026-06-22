# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused TurboQuant MSE dequant for MLA decode.

This module provides a single Triton kernel `_tq_mla_dequant_mse` that
collapses the per-layer hot path:
    {bit-unpack → centroid gather → optional norm-correction → vec_norm mul}
into one launch, writing the un-rotated reconstruction (`y_hat * vec_norm`)
plus the k_pe bf16 slice into a dense workspace.

The final inverse-Hadamard rotation `y_normed @ Pi` is left as a single
cuBLAS bf16 GEMM in Python — that one is already a single fast kernel
and would only complicate the Triton kernel.

Reference for the unpack + centroid gather pattern:
    `vllm/v1/attention/ops/triton_turboquant_decode.py::_tq_full_dequant_kv`
(non-MLA backend; we mirror its K-dequant branch).

Numerical equivalence with `_dequant_kv_c_mse` (PyTorch reference in
`triton_mla_tq.py`) is bit-for-bit at fp32 round-trip; bf16 ULP-level
deviations are allowed.
"""

import torch

from vllm.triton_utils import tl, triton

_BF16 = torch.bfloat16


@triton.jit
def _tq_mla_dequant_mse(
    Cache_ptr,  # uint8: (n_active, block_size, packed_bytes)
    Centroids_ptr,  # bf16: (2**bits,) sorted Lloyd-Max codebook
    Out_ptr,  # bf16: (n_active, block_size, L+R)
    # Strides (in elements; cache stride in bytes since uint8)
    stride_cache_n,
    stride_cache_p,
    stride_out_n,
    stride_out_p,
    # Compile-time constants
    L: tl.constexpr,  # kv_lora_rank, e.g. 512
    R: tl.constexpr,  # qk_rope_head_dim, e.g. 64
    BLOCK_SIZE: tl.constexpr,  # cache block_size (page count)
    BLOCK_D: tl.constexpr,  # next_pow2(L)
    BLOCK_R: tl.constexpr,  # next_pow2(R)
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(L * MSE_BITS / 8)
    KV_C_BYTES: tl.constexpr,  # MSE_BYTES + 2 (vec_norm fp16)
    NORM_CORRECTION: tl.constexpr,  # 0/1
    KPE_FP8: tl.constexpr,  # 0=bf16, 1=fp8 e4m3 + per-token fp16 scale
):
    """One program = one (active_block, page).

    Writes to Out_ptr[active_idx, page, :L+R]:
      out[:L] = y_hat_normed * vec_norm   (un-rotated; caller applies @ Pi)
      out[L:] = k_pe (bf16 reinterpret of cache[..., KV_C_BYTES:])
    """
    n_idx = tl.program_id(0)  # active block index
    p_idx = tl.program_id(1)  # page within block

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < L

    # ----- Address of this slot's packed bytes in the cache -----
    slot_base = n_idx * stride_cache_n + p_idx * stride_cache_p

    # ----- Bit-unpack indices (mirrors _unpack_bits_rows) -----
    bit_off = d_offs * MSE_BITS
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    raw0 = tl.load(Cache_ptr + slot_base + byte_idx, mask=d_mask, other=0).to(tl.int32)
    raw1 = tl.load(Cache_ptr + slot_base + byte_idx + 1, mask=d_mask, other=0).to(
        tl.int32
    )
    raw16 = raw0 | (raw1 << 8)
    idx = (raw16 >> bit_shift) & umask

    # ----- Centroid gather (bf16) -----
    # Cast to fp32 immediately for stable norm computation.
    # .ca hint: codebook is tiny (8/16 bf16); cache at all levels.
    y_hat_bf = tl.load(
        Centroids_ptr + idx, mask=d_mask, other=0.0, cache_modifier=".ca"
    )
    y_hat = y_hat_bf.to(tl.float32)

    # ----- Optional norm correction: re-normalize y_hat to unit norm -----
    if NORM_CORRECTION:
        sq = tl.where(d_mask, y_hat * y_hat, 0.0)
        c_norm_sq = tl.sum(sq, axis=0)
        inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-8)
        y_hat = y_hat * inv_norm

    # ----- Recover per-token vec_norm from the 2 fp16 bytes -----
    n_lo = tl.load(Cache_ptr + slot_base + MSE_BYTES).to(tl.uint16)
    n_hi = tl.load(Cache_ptr + slot_base + MSE_BYTES + 1).to(tl.uint16)
    vec_norm_u16 = n_lo | (n_hi << 8)
    vec_norm_f32 = vec_norm_u16.to(tl.float16, bitcast=True).to(tl.float32)

    # ----- Final scalar multiply, cast back to bf16 -----
    out_kvc = (y_hat * vec_norm_f32).to(tl.bfloat16)

    # ----- Store y_hat * vec_norm into Out[..., :L] -----
    out_base = n_idx * stride_out_n + p_idx * stride_out_p
    tl.store(Out_ptr + out_base + d_offs, out_kvc, mask=d_mask)

    # ----- Inline copy k_pe -----
    # bf16 layout: cache[KV_C_BYTES : KV_C_BYTES + 2*R] = R bf16 elems.
    # fp8 layout:  cache[KV_C_BYTES : KV_C_BYTES + R]   = R fp8 e4m3 elems,
    #              cache[KV_C_BYTES + R : KV_C_BYTES + R + 2] = fp16 scale.
    r_offs = tl.arange(0, BLOCK_R)
    r_mask = r_offs < R
    if KPE_FP8:
        # Reload fp8 byte and bitcast to fp8e4m3 → fp32 → bf16.
        # Then multiply per-token scale.
        fp8_byte = tl.load(
            Cache_ptr + slot_base + KV_C_BYTES + r_offs,
            mask=r_mask,
            other=0,
        ).to(tl.uint8)
        # bitcast uint8 → fp8 e4m3
        kpe_fp8 = fp8_byte.to(tl.float8e4nv, bitcast=True)
        kpe_f32 = kpe_fp8.to(tl.float32)
        # Per-token fp16 scale at cache[KV_C_BYTES + R : KV_C_BYTES + R + 2]
        s_lo = tl.load(Cache_ptr + slot_base + KV_C_BYTES + R).to(tl.uint16)
        s_hi = tl.load(Cache_ptr + slot_base + KV_C_BYTES + R + 1).to(tl.uint16)
        scale_u16 = s_lo | (s_hi << 8)
        scale_f32 = scale_u16.to(tl.float16, bitcast=True).to(tl.float32)
        kpe_bf = (kpe_f32 * scale_f32).to(tl.bfloat16)
    else:
        # bf16 path: reinterpret 2 consecutive uint8 as one bf16.
        kpe_lo = tl.load(
            Cache_ptr + slot_base + KV_C_BYTES + r_offs * 2,
            mask=r_mask,
            other=0,
        ).to(tl.uint16)
        kpe_hi = tl.load(
            Cache_ptr + slot_base + KV_C_BYTES + r_offs * 2 + 1,
            mask=r_mask,
            other=0,
        ).to(tl.uint16)
        kpe_u16 = kpe_lo | (kpe_hi << 8)
        kpe_bf = kpe_u16.to(tl.bfloat16, bitcast=True)
    tl.store(Out_ptr + out_base + L + r_offs, kpe_bf, mask=r_mask)


def fused_mla_dequant_mse(
    cache: torch.Tensor,  # (n_active, block_size, packed_bytes) uint8
    centroids_bf16: torch.Tensor,  # (2**bits,) bf16
    out: torch.Tensor,  # (n_active, block_size, L+R) bf16 (un-rotated)
    *,
    L: int,
    R: int,
    mse_bits: int,
    mse_bytes: int,
    kv_c_bytes: int,
    norm_correction: bool,
    kpe_fp8: bool = False,
) -> None:
    """Launch the fused MSE dequant kernel.

    `out[..., :L]` receives `y_hat_normed * vec_norm` — the caller must
    apply `out[..., :L] @ Pi` (cuBLAS bf16 GEMM) to finish the inverse
    Hadamard rotation. `out[..., L:]` receives `k_pe`.
    """
    assert cache.dtype == torch.uint8
    assert out.dtype == _BF16
    assert centroids_bf16.dtype == _BF16
    n_active, block_size, _packed = cache.shape
    assert out.shape == (n_active, block_size, L + R), (
        f"out shape mismatch: {out.shape} vs ({n_active}, {block_size}, {L + R})"
    )
    if n_active == 0:
        return

    BLOCK_D = triton.next_power_of_2(L)
    BLOCK_R = triton.next_power_of_2(R)
    grid = (n_active, block_size)
    _tq_mla_dequant_mse[grid](
        cache,
        centroids_bf16,
        out,
        cache.stride(0),
        cache.stride(1),
        out.stride(0),
        out.stride(1),
        L=L,
        R=R,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        BLOCK_R=BLOCK_R,
        MSE_BITS=mse_bits,
        MSE_BYTES=mse_bytes,
        KV_C_BYTES=kv_c_bytes,
        NORM_CORRECTION=1 if norm_correction else 0,
        KPE_FP8=1 if kpe_fp8 else 0,
        num_warps=4,
        num_stages=2,
    )


# =============================================================================
# Fused decode stage1: dequant + grouped attention in a single kernel.
#
# Mirrors `_fwd_grouped_kernel_stage1` in
# `vllm/v1/attention/ops/triton_decode_attention.py` (L261-443) but consumes
# a packed-uint8 paged TurboQuant cache directly. At each K-load site we
# inline {bit-unpack → centroid gather → vec_norm scale} instead of reading
# bf16, eliminating the per-layer bf16 dequant workspace. The query is
# expected to be pre-rotated (Πq) on the caller side — see
# `tests/kernels/attention/test_mla_turboquant_qside_rotation.py` for the
# math identity.
#
# Cache slot layout (per token in `(num_blocks, block_size, packed_bytes)`):
#     bytes [0 ... MSE_BYTES)              packed kv_c indices (MSE_BITS each)
#     bytes [MSE_BYTES ... KV_C_BYTES)     fp16 vec_norm  (MSE_BYTES + 2)
#     bytes [KV_C_BYTES ... )              k_pe (bf16) or k_pe (fp8 + fp16 scale)
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2, maxnreg=192),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2, maxnreg=192),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3, maxnreg=192),
        triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=2, maxnreg=192),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=3, maxnreg=192),
    ],
    key=["L", "R", "MSE_BITS", "KEY_FP8", "PAGE_SIZE"],
)
@triton.jit
def _fwd_grouped_kernel_stage1_tq(
    Q,  # bf16: (batch, q_head_num, Lk) Lk=L+R; q is Π-rotated on the L slice
    K_Cache,  # uint8: (num_blocks, block_size, packed_bytes)
    Centroids_ptr,  # bf16: (2**MSE_BITS,) Lloyd-Max codebook
    sm_scale,
    Req_to_tokens,  # int32: (batch, max_kv_pages) -> page index
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_cache_n,  # bytes per page block (block_size * packed_bytes)
    stride_cache_p,  # bytes per token slot (packed_bytes)
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # next_pow2(L)
    BLOCK_DPE: tl.constexpr,  # next_pow2(R)
    BLOCK_DV: tl.constexpr,  # = BLOCK_DMODEL (MLA: V is L slice of kv_c)
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    L: tl.constexpr,  # kv_lora_rank
    R: tl.constexpr,  # qk_rope_head_dim
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KV_C_BYTES: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    KPE_FP8: tl.constexpr,
    KEY_FP8: tl.constexpr,
    K_SCALE_ptr,  # fp32 device scalar: layer-global k_scale
):
    """One program = (batch, head_block, kv_split). Heads are grouped via BLOCK_H
    just like the upstream stage1; KV is reduced one BLOCK_N tile at a time.

    KEY_FP8=1: kv_c bytes are L fp8 e4m3 elems (no Hadamard, no vec_norm,
    no centroid). The layer-global K_SCALE compile-time constant is multiplied
    into the bf16 K tile (and therefore into V via the MLA K==V identity)
    so output magnitude matches the bf16-workspace baseline that pre-scales
    the cache.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # MLA: kv_group_num == q_head_num (single shared kv_c).
    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < q_head_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < L
    offs_dpe = L + tl.arange(0, BLOCK_DPE)
    mask_dpe = offs_dpe < (L + R)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    # Load the (already Π-rotated) query.
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(
        Q + offs_q,
        mask=mask_h[:, None] & mask_d[None, :],
        other=0.0,
        cache_modifier=".ca",
    )
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(
        Q + off_qpe,
        mask=mask_h[:, None] & mask_dpe[None, :],
        other=0.0,
        cache_modifier=".ca",
    )

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    # Bit-unpack constants.
    bit_off = offs_d * MSE_BITS  # (BLOCK_DMODEL,)
    byte_idx = bit_off // 8
    bit_shift = bit_off % 8
    umask = (1 << MSE_BITS) - 1

    if split_kv_end > split_kv_start:
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_kv_end

            kv_page = tl.load(
                Req_to_tokens
                + stride_req_to_tokens_b * cur_batch
                + offs_n // PAGE_SIZE,
                mask=n_mask,
                other=0,
                cache_modifier=".ca",
            )
            kv_in_page = offs_n % PAGE_SIZE
            slot_base = kv_page * stride_cache_n + kv_in_page * stride_cache_p
            # slot_base shape (BLOCK_N,); we need (BLOCK_DMODEL, BLOCK_N).
            sb = slot_base[None, :]
            bi = byte_idx[:, None]
            bs = bit_shift[:, None]
            d_full_mask = mask_d[:, None] & n_mask[None, :]

            if KEY_FP8:
                # K3: FP8 keys — load fp8 byte, bitcast to fp8e4nv,
                # cast directly to bf16 (skip fp32 intermediate), multiply
                # k_scale in bf16. Saves one F2FP conversion step.
                k_scale = tl.load(K_SCALE_ptr).to(tl.bfloat16)
                fp8_k = tl.load(
                    K_Cache + sb + offs_d[:, None],
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint8)
                k_fp8 = fp8_k.to(tl.float8e4nv, bitcast=True)
                k = k_fp8.to(tl.bfloat16) * k_scale
                qk = tl.dot(q, k)
            else:
                # ----- Bit-unpack across (BLOCK_DMODEL, BLOCK_N) -----
                raw0 = tl.load(
                    K_Cache + sb + bi,
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                raw1 = tl.load(
                    K_Cache + sb + bi + 1,
                    mask=d_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.int32)
                raw16 = raw0 | (raw1 << 8)
                idx = (raw16 >> bs) & umask

                # ----- Centroid gather (bf16 → fp32 for stable downstream) -----
                # .ca hint: codebook is tiny (8/16 bf16 values); cache at all
                # levels so subsequent tiles hit L1 instead of global memory.
                y_hat = tl.load(
                    Centroids_ptr + idx,
                    mask=d_full_mask,
                    other=0.0,
                    cache_modifier=".ca",
                ).to(tl.float32)

                if NORM_CORRECTION:
                    sq = tl.where(d_full_mask, y_hat * y_hat, 0.0)
                    c_norm_sq = tl.sum(sq, axis=0)  # (BLOCK_N,)
                    inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-8)
                    y_hat = y_hat * inv_norm[None, :]

                # ----- Per-token vec_norm (fp16, 2 bytes) -----
                vn_lo = tl.load(
                    K_Cache + slot_base + MSE_BYTES, mask=n_mask, other=0
                ).to(tl.uint16)
                vn_hi = tl.load(
                    K_Cache + slot_base + MSE_BYTES + 1, mask=n_mask, other=0
                ).to(tl.uint16)
                vn_u16 = vn_lo | (vn_hi << 8)
                vec_norm = vn_u16.to(tl.float16, bitcast=True).to(tl.float32)

                # K1-a: scale qk by per-token vec_norm AFTER the dot, instead of
                # scaling the [BLOCK_DMODEL, BLOCK_N] k tile before the cast. This
                # avoids materializing a separately-scaled bf16 K tile.
                k = y_hat.to(q.dtype)

                qk = tl.dot(q, k)
                qk = qk * vec_norm[None, :]

            # ----- k_pe path -----
            r_offs = tl.arange(0, BLOCK_DPE)
            r_mask = r_offs < R
            r_full_mask = r_mask[:, None] & n_mask[None, :]
            if KPE_FP8:
                fp8_byte = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None],
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint8)
                kpe_f32 = fp8_byte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                s_lo = tl.load(
                    K_Cache + slot_base + KV_C_BYTES + R, mask=n_mask, other=0
                ).to(tl.uint16)
                s_hi = tl.load(
                    K_Cache + slot_base + KV_C_BYTES + R + 1, mask=n_mask, other=0
                ).to(tl.uint16)
                scale = (s_lo | (s_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                kpe = (kpe_f32 * scale[None, :]).to(qpe.dtype)
            else:
                kpe_lo = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None] * 2,
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint16)
                kpe_hi = tl.load(
                    K_Cache + sb + KV_C_BYTES + r_offs[:, None] * 2 + 1,
                    mask=r_full_mask,
                    other=0,
                    cache_modifier=".cg",
                ).to(tl.uint16)
                kpe_u16 = kpe_lo | (kpe_hi << 8)
                kpe = kpe_u16.to(tl.bfloat16, bitcast=True).to(qpe.dtype)

            qk += tl.dot(qpe, kpe)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tl.math.tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & n_mask[None, :], qk, float("-inf"))

            # MLA reuses k as v (transposed).
            v = tl.trans(k)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            # K1-a value-path correction: in the MSE branch K is stored as
            # un-scaled centroid `y_hat` (vec_norm is folded into qk *after*
            # the dot to save a (BLOCK_DMODEL, BLOCK_N) multiply). Since
            # `v = trans(k) = trans(y_hat)` is also missing vec_norm, the
            # accumulator must apply vec_norm here — mathematically:
            #   sum_n p_n * (y_hat_n * vn_n) = sum_n (p_n * vn_n) * y_hat_n
            # so we scale p (shape (BLOCK_H, BLOCK_N), small) instead of v.
            # KEY_FP8 path bakes K_SCALE into y_hat before the cast, so v
            # already carries the correct scale and no correction is needed.
            p_for_v = p.to(v.dtype) if KEY_FP8 else (p * vec_norm[None, :]).to(v.dtype)
            acc += tl.dot(p_for_v, v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_d[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=mask_h[:, None] & mask_d[None, :],
        )
        offs_lse = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + L
        )
        tl.store(Att_Out + offs_lse, e_max + tl.log(e_sum), mask=mask_h)


def fused_mla_tq_decode_stage1(
    q: torch.Tensor,  # bf16: (batch, q_head_num, L+R), Π-rotated on [:L]
    cache: torch.Tensor,  # uint8: (num_blocks, block_size, packed_bytes)
    centroids_bf16: torch.Tensor,  # bf16: (2**MSE_BITS,) or empty when key_fp8
    att_out: torch.Tensor,  # fp32: (batch, q_head_num, NUM_KV_SPLITS, L+1)
    req_to_tokens: torch.Tensor,  # int32: (batch, max_kv_pages)
    b_seqlen: torch.Tensor,  # int32: (batch,)
    *,
    sm_scale: float,
    page_size: int,
    L: int,
    R: int,
    mse_bits: int,
    mse_bytes: int,
    kv_c_bytes: int,
    norm_correction: bool,
    kpe_fp8: bool,
    key_fp8: bool = False,
    k_scale: torch.Tensor | None = None,  # fp32 scalar on device
    num_kv_splits: int = 4,
    logit_cap: float = 0.0,
) -> None:
    """Launch the fused TurboQuant MLA decode stage1.

    MSE keys (default): `q` must be the standard MLA decode query with the
    Hadamard rotation applied to its first L (kv_lora_rank) elements; the
    kernel does no rotation internally.

    FP8 keys (`key_fp8=True`): cache layout is `[L bytes fp8 e4m3 | k_pe...]`.
    No Hadamard rotation; q is consumed as-is. The layer-global `k_scale`
    is passed as a device scalar and multiplied into the bf16 K tile
    inside the kernel (so V = trans(K) inherits the same scale, preserving
    MLA K==V semantics). centroids_bf16 is unused in this mode (pass any
    bf16 1-D tensor; e.g. an empty one).
    """
    assert cache.dtype == torch.uint8
    assert q.dtype == torch.bfloat16
    assert centroids_bf16.dtype == torch.bfloat16
    batch, q_head_num, head_dim = q.shape
    assert head_dim == L + R, f"q head_dim {head_dim} != L+R {L + R}"

    BLOCK_DMODEL = triton.next_power_of_2(L)
    BLOCK_DPE = triton.next_power_of_2(R)
    BLOCK_DV = BLOCK_DMODEL
    BLOCK_H = 16

    grid = (
        batch,
        triton.cdiv(q_head_num, BLOCK_H),
        num_kv_splits,
    )
    # k_scale is a device scalar (fp32, shape ()). For the MSE path the
    # kernel never reads it; pass a placeholder that satisfies Triton's
    # signature requirements.
    if k_scale is None:
        k_scale_tensor = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    else:
        assert k_scale.dim() == 0 and k_scale.dtype == torch.float32, (
            f"k_scale must be a scalar fp32 tensor, got shape={k_scale.shape} "
            f"dtype={k_scale.dtype}"
        )
        k_scale_tensor = k_scale

    _fwd_grouped_kernel_stage1_tq[grid](
        q,
        cache,
        centroids_bf16,
        sm_scale,
        req_to_tokens,
        b_seqlen,
        att_out,
        req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        cache.stride(0),
        cache.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        q_head_num=q_head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        L=L,
        R=R,
        MSE_BITS=mse_bits,
        MSE_BYTES=mse_bytes,
        KV_C_BYTES=kv_c_bytes,
        NORM_CORRECTION=1 if norm_correction else 0,
        KPE_FP8=1 if kpe_fp8 else 0,
        KEY_FP8=1 if key_fp8 else 0,
        K_SCALE_ptr=k_scale_tensor,
    )
