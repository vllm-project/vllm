# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 (CDNA3 / MI300X) Gluon port of AITER's fp8_mqa_logits scoring kernel.

This is a fork-and-adapt of AITER's gfx950 Gluon kernel
(``aiter/ops/triton/_gluon_kernels/gfx950/attention/fp8_mqa_logits.py``) to
CDNA3, replacing the current generic-Triton fallback that vLLM forces on gfx942
(``triton_fp8_mqa_logits.fp8_mqa_logits_gfx942``).

Adaptations from the gfx950 source:
  * ``gl.amd.cdna4.mfma_scaled(version=4, [32,32,64])`` (gfx950-only microscaled
    MFMA) -> ``gl.amd.cdna3.mfma`` with ``AMDMFMALayout(version=3,
    instr_shape=[32,32,16])`` and FNUZ fp8 (``float8_e4m3fnuz``) operands. This
    emits the native ``v_mfma_f32_32x32x16_fp8_fp8`` (2x K-rate vs the
    ``input_precision="ieee"`` f16-upcast path the generic kernel uses).
  * ``gl.amd.cdna4.async_copy.*`` (LDS double-buffer; no CDNA3 lowering) ->
    direct ``gl.amd.cdna3.buffer_load`` of each KV tile into registers, with Q +
    weights hoisted out of the loop (LICM).
  * ``gl.amd.cdna4.buffer_load/store`` -> ``gl.amd.cdna3.buffer_load/store``.
  * Dropped the ``PaddedSharedLayout`` XOR-swizzle (only needed for the LDS
    staging path, which CDNA3 does not use here).

Correctness validated on MI300X co-located microbench: rel_max ~1.8e-07 vs an
fp32 dequant oracle across M in {64..8192}, N in {40..36000}, causal, dense and
partial-tile windows. At the real GLM-5.2 indexer shape (H=32 query heads) the
tuned kernel reaches ~490 TFLOPs mean / ~500 TFLOPs at large shapes = ~37-40%
of the ~1307 TFLOPs MI300X fp8 MFMA reference peak (up from ~18% before the
``transposed=False`` MFMA-layout fix; see the mfma_layout comment below).

Requires the native fp8 MFMA, so the query/key operands MUST be FNUZ e4m3
(``torch.float8_e4m3fnuz`` / Triton ``float8e4b8``); the wrapper converts if the
caller passes OCP e4m3 (``float8_e4m3fn``).
"""
import os

import torch
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan

# 2 GiB resource-descriptor cap for AMD buffer_load/buffer_store ops.
_BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

# Winning launch config from the MI300X co-located sweep at the real indexer
# shape (H=32 query heads). NUM_WARPS=1: the per-row head-reduction (sum over the
# 32 heads) is intra-warp, so >1 warp would force a cross-warp reduction and
# regress badly (measured 3-5x slower). BLOCK_KV=64 + waves_per_eu=2 is the best
# occupancy/tile-size point once transposed=False makes the reduction cheap
# (sweep: this cluster is within ~1% of the per-shape optimum; larger BLOCK_KV or
# waves_per_eu do not help on average and raise register pressure).
_BLOCK_KV = 64
_NUM_WARPS = 1
_WAVES_PER_EU = 2

# Opt-in until an e2e serve validation confirms the upstream indexer feeds fp8
# values that round-trip through FNUZ correctly (default OFF -> no prod risk).
GLUON_GFX942_MQA_ENABLED = os.environ.get("VLLM_ROCM_MQA_LOGITS_GLUON", "0") == "1"

_MAX_PROP_NAN = gl.constexpr(PropagateNan.ALL)


@gluon.jit
def _relu_f32(x):
    return gl.maximum(x, 0.0, propagate_nan=_MAX_PROP_NAN)


@gluon.jit
def _score_tile(mfma_q, mfma_k, w_col, kv_scales,
                NUM_HEADS: gl.constexpr, BLOCK_KV: gl.constexpr,
                mfma_layout: gl.constexpr):
    # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
    acc = gl.zeros([NUM_HEADS, BLOCK_KV], dtype=gl.float32, layout=mfma_layout)
    scores = gl.amd.cdna3.mfma(mfma_q, mfma_k, acc)
    scores = _relu_f32(scores)
    scores = scores * w_col                  # per-head weight, broadcast over KV
    scores = gl.sum(scores, axis=0)          # sum over heads -> [BLOCK_KV]
    scores = scores * kv_scales              # per-kv-token scale
    return scores


@gluon.jit
def _gluon_fp8_mqa_logits_kernel_gfx942(
    Q_ptr,            # fp8 (fnuz e4m3) [seq_len, NUM_HEADS, HEAD_SIZE]
    KV_ptr,           # fp8 (fnuz e4m3) [seq_len_kv, HEAD_SIZE]
    kv_scales_ptr,    # fp32   [seq_len_kv]
    weights_ptr,      # fp32   [seq_len, NUM_HEADS]
    cu_start_ptr,     # int32  [seq_len]
    cu_end_ptr,       # int32  [seq_len]
    logits_ptr,       # fp32   [seq_len, seq_len_kv]
    seq_len: gl.int32,
    seq_len_kv: gl.int32,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    stride_q_s: gl.int32,
    stride_q_h: gl.constexpr,
    stride_q_d: gl.constexpr,
    stride_kv_s: gl.int32,
    stride_kv_d: gl.constexpr,
    stride_w_s: gl.int32,
    stride_w_h: gl.constexpr,
    stride_logits_s: gl.int32,
    stride_logits_k: gl.int32,
    BLOCK_KV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # go from larger to smaller work to reduce the tail effect (mirror gfx950)
    row_id = gl.num_programs(0) - gl.program_id(axis=0) - 1

    # CDNA3 native fp8 MFMA: v_mfma_f32_32x32x16_fp8_fp8.
    # transposed=True is the SERVE-VALIDATED MFMA output layout (e2e coherent,
    # -20.8% TTFT at C=1). The transposed=False variant measured faster in the
    # co-located microbench (~495 TFLOPs) but produced GARBAGE logits in live
    # TP8 serve (the fp32 microbench oracle shared its layout assumption and
    # missed the real-serve bug) -- do NOT flip this back to transposed=False.
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[32, 32, 16], transposed=True,
        warps_per_cta=[NUM_WARPS, 1])
    K_WIDTH: gl.constexpr = 16
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=K_WIDTH)
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=K_WIDTH)

    # Q [NUM_HEADS, HEAD_SIZE] contiguous along HEAD_SIZE
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16], threads_per_warp=[32, 2],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0])
    # K [HEAD_SIZE, BLOCK_KV] contiguous along HEAD_SIZE (dim 0)
    blocked_kv: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1], threads_per_warp=[4, 16],
        warps_per_cta=[1, NUM_WARPS], order=[0, 1])

    start_ind = gl.load(cu_start_ptr + row_id)
    end_ind = gl.load(cu_end_ptr + row_id)
    start_ind = gl.maximum(start_ind, 0)
    end_ind = gl.minimum(end_ind, seq_len_kv)

    # --- Load Q + weights once (LICM) ---
    offs_h_q = gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, blocked_q))
    offs_d_q = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, blocked_q))
    q = gl.amd.cdna3.buffer_load(
        Q_ptr,
        row_id * stride_q_s
        + offs_h_q[:, None] * stride_q_h
        + offs_d_q[None, :] * stride_q_d,
        cache=".cg",
    )
    mfma_q = gl.convert_layout(q, dot_a_layout)

    offs_h_w = gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))
    w_block = gl.amd.cdna3.buffer_load(
        weights_ptr, row_id * stride_w_s + offs_h_w[:, None] * stride_w_h,
        cache=".cg",
    ).to(gl.float32)

    kv_arange = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout))
    offs_d_kv = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, blocked_kv))
    offs_n_kv = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, blocked_kv))

    logits_row = logits_ptr + row_id * stride_logits_s
    num_full_tiles = (end_ind - start_ind) // BLOCK_KV

    kv_pos = start_ind
    # --- full tiles (no mask) ---
    for _ in tl.range(0, num_full_tiles):
        k_tile = gl.amd.cdna3.buffer_load(
            KV_ptr,
            offs_d_kv[:, None] * stride_kv_d
            + (kv_pos + offs_n_kv)[None, :] * stride_kv_s,
        )
        mfma_k = gl.convert_layout(k_tile, dot_b_layout)
        kv_scales = gl.amd.cdna3.buffer_load(kv_scales_ptr, kv_pos + kv_arange)
        scores = _score_tile(mfma_q, mfma_k, w_block, kv_scales,
                              NUM_HEADS, BLOCK_KV, mfma_layout)
        gl.amd.cdna3.buffer_store(
            scores, logits_row, (kv_pos + kv_arange) * stride_logits_k)
        kv_pos += BLOCK_KV

    # --- masked tail (kv_pos .. end_ind) ---
    tail_mask_1d = (kv_pos + kv_arange) < end_ind
    tail_mask_2d = (kv_pos + offs_n_kv)[None, :] < end_ind
    k_tile = gl.amd.cdna3.buffer_load(
        KV_ptr,
        offs_d_kv[:, None] * stride_kv_d + (kv_pos + offs_n_kv)[None, :] * stride_kv_s,
        mask=tail_mask_2d, other=0.0,
    )
    mfma_k = gl.convert_layout(k_tile, dot_b_layout)
    kv_scales = gl.amd.cdna3.buffer_load(
        kv_scales_ptr, kv_pos + kv_arange, mask=tail_mask_1d, other=0.0)
    scores = _score_tile(mfma_q, mfma_k, w_block, kv_scales,
                         NUM_HEADS, BLOCK_KV, mfma_layout)
    gl.amd.cdna3.buffer_store(
        scores, logits_row, (kv_pos + kv_arange) * stride_logits_k, mask=tail_mask_1d)


def supported(q: torch.Tensor, k_fp8: torch.Tensor, seq_len: int,
              seq_len_kv: int) -> bool:
    """True iff this Gluon port can serve the given shapes.

    Requires power-of-two H/D, and both KV and the fp32 logits output to fit the
    AMD buffer-op 2 GiB resource-descriptor cap (the direct-register load path
    used here has no non-buffer fallback yet).
    """
    if not GLUON_GFX942_MQA_ENABLED:
        return False
    _, num_heads, head_size = q.shape
    if num_heads & (num_heads - 1) or head_size & (head_size - 1):
        return False
    kv_bytes = k_fp8.numel() * k_fp8.element_size()
    logits_bytes = seq_len * seq_len_kv * 4
    return kv_bytes < _BUFFER_LIMIT_BYTES and logits_bytes < _BUFFER_LIMIT_BYTES


def fp8_mqa_logits_gfx942_gluon(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Native-fp8-MFMA gfx942 MQA logits. Drop-in for ``fp8_mqa_logits_gfx942``.

    ``q``/``k_fp8`` are converted to ``torch.float8_e4m3fnuz`` if not already, so
    the CDNA3 ``v_mfma_f32_32x32x16_fp8_fp8`` path sees FNUZ operands.
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]

    fnuz = torch.float8_e4m3fnuz
    if q.dtype != fnuz:
        q = q.to(fnuz)
    if k_fp8.dtype != fnuz:
        k_fp8 = k_fp8.to(fnuz)

    kv_scales_1d = kv_scales.reshape(-1)
    logits = torch.full((seq_len, seq_len_kv), fill_value=-float("inf"),
                        dtype=torch.float32, device=q.device)

    stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_kv_s, stride_kv_d = k_fp8.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    _gluon_fp8_mqa_logits_kernel_gfx942[(seq_len,)](
        Q_ptr=q, KV_ptr=k_fp8, kv_scales_ptr=kv_scales_1d, weights_ptr=weights,
        cu_start_ptr=cu_starts, cu_end_ptr=cu_ends, logits_ptr=logits,
        seq_len=seq_len, seq_len_kv=seq_len_kv, NUM_HEADS=num_heads,
        HEAD_SIZE=head_size, stride_q_s=stride_q_s, stride_q_h=stride_q_h,
        stride_q_d=stride_q_d, stride_kv_s=stride_kv_s, stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s, stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s, stride_logits_k=stride_logits_k,
        BLOCK_KV=_BLOCK_KV, NUM_WARPS=_NUM_WARPS,
        num_warps=_NUM_WARPS, waves_per_eu=_WAVES_PER_EU)
    return logits
