# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drop-in Triton replacements for the native FlashMLA *sparse* CUDA ops.

These functions reproduce, on consumer Blackwell (sm_121 / SM12x) where the
compiled ``vllm._flashmla_C`` extension does not exist, the exact behaviour of
the two native ops that the V3.2 sparse-MLA attention backend
(``vllm/v1/attention/backends/mla/flashmla_sparse.py`` ::
``FlashMLASparseImpl``) calls:

  * ``flash_mla_sparse_fwd``        -- BF16 sparse *prefill*  kernel
  * ``flash_mla_with_kvcache``      -- FP8  sparse *decode*   kernel
       (``fp8_ds_mla`` 656-byte cache format)

The heavy lifting is done by the portable Triton kernels in
``sparse_mla_kernels.py`` (copied verbatim from jasl/vllm's deepseek_v4 path).
This module only adapts the *layouts* expected by the V3.2 backend to those
kernels.

----------------------------------------------------------------------------
Native semantics being reproduced (from FlashMLA/flash_mla_interface.py)
----------------------------------------------------------------------------

flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None,
                     topk_length=None) -> (output, max_logits, lse)
    q       : [s_q, h_q, d_qk]  bfloat16   (d_qk = 576 = 512 NoPE + 64 RoPE)
    kv      : [s_kv, h_kv, d_qk] bfloat16  (h_kv = 1)
    indices : [s_q, h_kv, topk] int32      (-1 or >= s_kv == invalid)
    score   = (q . kv) * sm_scale          over the full d_qk (576)
    softmax over valid candidates; output = sum(p * kv[:, :d_v])
    attn_sink (optional, [h_q] f32): output *= exp(lse) / (exp(lse)+exp(sink))
    topk_length (optional, [s_q] int32): per-query cap on #candidates.
    returns output [s_q, h_q, d_v] bf16, max_logits [s_q, h_q] f32,
            lse [s_q, h_q] f32.

flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
                       tile_scheduler_metadata, num_splits=None,
                       softmax_scale=None, causal=False, is_fp8_kvcache=False,
                       indices=None, attn_sink=None, extra_k_cache=None,
                       extra_indices_in_kvcache=None, topk_length=None,
                       extra_topk_length=None) -> (out, softmax_lse)
    For the V3.2 fp8 sparse decode call site (see ``_fp8_flash_mla_kernel`` in
    flashmla_sparse.py):
        q       : [b, s_q, h_q(padded 64/128), 576] bf16
        k_cache : [num_blocks, block_size, 1, 656] uint8  (fp8_ds_mla)
        indices : [b, s_q, topk] int32  (already converted to GLOBAL slot ids
                                         by triton_convert_req_index_to_global_index;
                                         -1 / >= total_slots == invalid)
        head_dim_v = 512, is_fp8_kvcache = True
        attn_sink / topk_length: NOT passed by V3.2 (None).
    returns out [b, s_q, h_q, 512] bf16, lse [b, h_q, s_q] f32.

The fp8_ds_mla 656-byte per-token cache layout (kv_lora_rank=512, rope=64):
    bytes [  0:512]  : 512 x float8_e4m3   (quantized NoPE latent)
    bytes [512:528]  : 4   x float32        (scale, one per 128-elem tile)
    bytes [528:656]  : 64  x bfloat16       (RoPE, not quantized)
    dequant_nope[i] = float(fp8[i]) * scale[i // 128]   (direct fp32 multiply)
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_indexed_sparse_mla_attention_chunk,
    finish_sparse_mla_attention_with_sink,
)

logger = init_logger(__name__)

# fp8_ds_mla 656-byte layout constants (DeepSeek V3.2 main MLA).
_FP8DS_NOPE_DIM = 512          # number of fp8 NoPE values
_FP8DS_ROPE_DIM = 64           # number of bf16 RoPE values
_FP8DS_QUANT_TILE = 128        # one fp32 scale per 128 fp8 values
_FP8DS_NUM_SCALES = _FP8DS_NOPE_DIM // _FP8DS_QUANT_TILE  # 4
_FP8DS_ENTRY_BYTES = (
    _FP8DS_NOPE_DIM + 4 * _FP8DS_NUM_SCALES + 2 * _FP8DS_ROPE_DIM
)  # 512 + 16 + 128 = 656
_DQK = _FP8DS_NOPE_DIM + _FP8DS_ROPE_DIM  # 576
_DV = _FP8DS_NOPE_DIM  # 512


def _profile_enabled() -> bool:
    return os.getenv("GLM52_PREFILL_PROFILE", "0").lower() not in {
        "",
        "0",
        "false",
        "no",
    }


def _cuda_capture_active() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing and is_capturing())


class _CudaProfileRegion:
    def __init__(self, region: str, **fields: object) -> None:
        self.region = region
        self.fields = fields
        self.enabled = (
            _profile_enabled()
            and torch.cuda.is_available()
            and not _cuda_capture_active()
        )
        self.start_event: torch.cuda.Event | None = None
        self.end_event: torch.cuda.Event | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        torch.cuda.nvtx.range_push(f"glm52:{self.region}")
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def stop(self) -> None:
        if not self.enabled or self.start_event is None or self.end_event is None:
            return
        self.end_event.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        details = " ".join(f"{key}={value}" for key, value in self.fields.items())
        logger.info(
            "GLM52_PREFILL_PROFILE region=%s elapsed_ms=%.3f %s",
            self.region,
            self.start_event.elapsed_time(self.end_event),
            details,
        )


# ---------------------------------------------------------------------------
# BF16 sparse prefill : flash_mla_sparse_fwd
# ---------------------------------------------------------------------------
def flash_mla_sparse_fwd_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton BF16 sparse-MLA prefill, drop-in for ``flash_mla_sparse_fwd``.

    Args mirror the native op exactly:
        q       : [s_q, h_q, d_qk]   bf16
        kv      : [s_kv, h_kv, d_qk] bf16   (h_kv == 1)
        indices : [s_q, h_kv, topk]  int32  (-1 or >= s_kv invalid)
        sm_scale: float
        d_v     : value dim (512)
        attn_sink: optional [h_q] f32
        topk_length: optional [s_q] int32 per-query candidate cap
        out     : optional pre-allocated output [s_q, h_q, d_v]
    Returns (output [s_q, h_q, d_v] bf16, max_logits [s_q, h_q] f32,
             lse [s_q, h_q] f32).

    Implementation note on the d_qk vs d_v split: the indexed accumulate kernel
    scores ``q . kv`` over the *full* head_dim (= d_qk = 576, including RoPE)
    and accumulates the value as the full ``kv`` row.  Because the value
    accumulation is per-dimension independent, the first ``d_v`` (512) columns
    of the accumulator are *exactly* the NoPE-part value reduction we want; we
    simply slice them off.  The RoPE columns (512:576) of the accumulator are
    discarded (wasted compute, correct result).
    """
    assert q.dim() == 3, f"expected q [s_q, h_q, d_qk], got {q.shape}"
    assert kv.dim() == 3, f"expected kv [s_kv, h_kv, d_qk], got {kv.shape}"
    assert indices.dim() == 3, f"expected indices [s_q, h_kv, topk], got {indices.shape}"
    assert kv.shape[1] == 1, "sparse MLA requires h_kv == 1"
    assert indices.shape[1] == 1, "sparse MLA requires h_kv == 1 on indices"
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[2]
    assert kv.shape[2] == d_qk
    assert indices.shape[0] == s_q

    device = q.device
    kv_flat = kv.reshape(s_kv, d_qk)              # [s_kv, d_qk]
    idx2d = indices.reshape(s_q, topk).contiguous()  # [s_q, topk] int32

    # Per-query candidate count (lens). The accumulate kernel iterates
    # ``min(num_candidates, max(lens - candidate_offset, 0))`` and still
    # individually skips negative sentinels, so:
    #   * If topk_length is given, lens = topk_length (matches native cap).
    #   * Otherwise lens = topk (full width); -1 entries are skipped per-cell.
    if topk_length is not None:
        lens = topk_length.to(device=device, dtype=torch.int32).contiguous()
        assert lens.shape[0] == s_q
    else:
        lens = torch.full((s_q,), topk, dtype=torch.int32, device=device)

    profile = _CudaProfileRegion(
        "attention.bf16_sparse_prefill",
        q_shape=tuple(q.shape),
        kv_shape=tuple(kv.shape),
        topk=topk,
        topk_length="set" if topk_length is not None else "none",
    )
    profile.start()

    # State buffers for the online-softmax accumulation (fp32).
    max_score = torch.full((s_q, h_q), float("-inf"), dtype=torch.float32, device=device)
    denom = torch.zeros((s_q, h_q), dtype=torch.float32, device=device)
    acc = torch.zeros((s_q, h_q, d_qk), dtype=torch.float32, device=device)

    # Single chunk over all candidates (correctness-first; chunking is a
    # perf-only refinement that the V4 wrapper does for very large topk).
    accumulate_indexed_sparse_mla_attention_chunk(
        q=q,
        kv_flat=kv_flat,
        indices=idx2d,
        lens=lens,
        scale=float(sm_scale),
        max_score=max_score,
        denom=denom,
        acc=acc,
        candidate_offset=0,
    )

    # max_logits / lse are computed from the online-softmax state.
    # max_logits[t, h] = running_max (over valid candidates), -inf if none.
    # lse[t, h]        = log(denom) + running_max.
    max_logits = max_score.clone()
    safe_max = torch.where(denom > 0, max_score, torch.zeros_like(max_score))
    lse = torch.where(
        denom > 0,
        torch.log(denom.clamp_min(1e-30)) + safe_max,
        torch.full_like(max_score, float("-inf")),
    )

    if out is None:
        out = torch.empty((s_q, h_q, _DV), dtype=q.dtype, device=device)
    assert out.shape[0] == s_q and out.shape[2] == _DV

    # attn_sink handling: finish_sparse_mla_attention_with_sink applies the
    #   output *= exp(running_max - merge_max) / (denom*... + exp(sink-...))
    # which is algebraically identical to the documented
    #   output *= exp(lse) / (exp(lse) + exp(attn_sink)).
    # When attn_sink is None we pass a -inf sink (no effect) per the native
    # docstring ("-inf has no effect").
    if attn_sink is None:
        sink = torch.full((h_q,), float("-inf"), dtype=torch.float32, device=device)
    else:
        sink = attn_sink.to(device=device, dtype=torch.float32).contiguous()
        assert sink.shape[0] >= h_q

    # finish reads acc over its last dim; slice to d_v (NoPE value part).
    acc_v = acc[:, :, :_DV].contiguous()
    finish_sparse_mla_attention_with_sink(
        max_score,
        denom,
        acc_v,
        sink,
        output=out,
    )
    profile.stop()
    return out, max_logits, lse


# ---------------------------------------------------------------------------
# fp8_ds_mla dequant-gather kernel (for the FP8 sparse decode path)
# ---------------------------------------------------------------------------
@triton.jit
def _gather_dequant_fp8ds_kernel(
    cache_ptr,            # uint8 [total_slots, 656] (or strided view)
    indices_ptr,          # int32 [T, K] global slot ids, -1 invalid
    out_ptr,              # bf16  [T, K, 576] gathered dequantized kv
    stride_cache_s,
    stride_idx_t: tl.constexpr,
    stride_idx_k: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_k: tl.constexpr,
    stride_out_d: tl.constexpr,
    total_slots,
    nope_dim: tl.constexpr,       # 512
    rope_dim: tl.constexpr,       # 64
    quant_tile: tl.constexpr,     # 128
    scale_byte_off: tl.constexpr,  # 512
    rope_byte_off: tl.constexpr,   # 528
    BLOCK_D: tl.constexpr,        # >= 576 (head_dim)
):
    # int64 program ids: the gathered-KV ``out`` tensor is [T, K, 576] and for
    # the cluster's mixed-batch prefill (T=2048, K=2048) its per-token stride
    # (K*576) times ``t`` overflows int32 (>2^31) around t~1821, wrapping the
    # store address negative -> illegal global write. Promote the row/col ids to
    # int64 so every ``t*stride`` / ``k*stride`` offset below is computed in
    # int64. (stride_idx_* are small but the out_ptr offset is the one that
    # overflows; keep both 64-bit for safety.)
    t = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1).to(tl.int64)

    slot = tl.load(indices_ptr + t * stride_idx_t + k * stride_idx_k)
    is_valid = (slot >= 0) & (slot < total_slots)

    dim_offsets = tl.arange(0, BLOCK_D)
    head_dim = nope_dim + rope_dim
    dim_mask = dim_offsets < head_dim
    nope_mask = dim_offsets < nope_dim
    rope_mask = (dim_offsets >= nope_dim) & dim_mask

    base = cache_ptr + slot.to(tl.int64) * stride_cache_s

    # ---- NoPE: fp8_e4m3 bytes [0:512] dequantized by per-128 float32 scale ----
    fp8_bytes = tl.load(
        base + dim_offsets,
        mask=nope_mask & is_valid,
        other=0,
    )
    fp8_vals = fp8_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    # scales: 4 float32 at byte offset 512; tile index = dim // 128
    scale_tile = dim_offsets // quant_tile  # 0..3 within nope region
    scale_byte = scale_byte_off + scale_tile * 4
    # load each of the 4 scale-bytes as a float32 (little-endian assemble)
    b0 = tl.load(base + scale_byte + 0, mask=nope_mask & is_valid, other=0).to(tl.int32)
    b1 = tl.load(base + scale_byte + 1, mask=nope_mask & is_valid, other=0).to(tl.int32)
    b2 = tl.load(base + scale_byte + 2, mask=nope_mask & is_valid, other=0).to(tl.int32)
    b3 = tl.load(base + scale_byte + 3, mask=nope_mask & is_valid, other=0).to(tl.int32)
    scale_bits = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24)
    scale_f32 = scale_bits.to(tl.float32, bitcast=True)
    nope = fp8_vals * scale_f32

    # ---- RoPE: bf16 values [528:656] -> elements [0:64] over nope:nope+64 ----
    rope_elem = tl.maximum(dim_offsets - nope_dim, 0)  # 0..63 for rope region
    rope_ptr = (base + rope_byte_off).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(rope_ptr + rope_elem, mask=rope_mask & is_valid, other=0.0).to(
        tl.float32
    )

    kv = tl.where(nope_mask, nope, rope)
    kv = tl.where(dim_mask & is_valid, kv, 0.0)

    tl.store(
        out_ptr + t * stride_out_t + k * stride_out_k + dim_offsets * stride_out_d,
        kv.to(out_ptr.dtype.element_ty),
        mask=dim_mask,
    )


def _gather_dequant_fp8ds(
    cache_flat: torch.Tensor,   # uint8 [total_slots, 656]
    indices: torch.Tensor,      # int32 [T, K] global slot ids
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather + dequantize fp8_ds_mla cache rows selected by ``indices``.

    Returns (kv [T, K, 576] bf16, valid [T, K] bool).
    Invalid (-1 / out-of-range) rows are zero-filled and marked invalid.
    """
    assert cache_flat.dtype == torch.uint8
    assert cache_flat.dim() == 2 and cache_flat.shape[1] == _FP8DS_ENTRY_BYTES, (
        f"expected fp8_ds_mla cache [total_slots, {_FP8DS_ENTRY_BYTES}], "
        f"got {tuple(cache_flat.shape)}"
    )
    assert indices.dim() == 2
    T, K = indices.shape
    total_slots = cache_flat.shape[0]
    device = cache_flat.device

    out = torch.empty((T, K, _DQK), dtype=torch.bfloat16, device=device)
    idx_c = indices.contiguous()
    # valid mask is a pure function of the (already global) indices; compute on
    # the host to avoid scalar-indexing inside the kernel.
    valid = (idx_c >= 0) & (idx_c < total_slots)
    block_d = max(64, next_power_of_2(_DQK))  # 1024 (>= 576)
    grid = (T, K)
    _gather_dequant_fp8ds_kernel[grid](
        cache_flat,
        idx_c,
        out,
        cache_flat.stride(0),
        idx_c.stride(0),
        idx_c.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        total_slots,
        _FP8DS_NOPE_DIM,
        _FP8DS_ROPE_DIM,
        _FP8DS_QUANT_TILE,
        _FP8DS_NOPE_DIM,            # scale_byte_off = 512
        _FP8DS_NOPE_DIM + 4 * _FP8DS_NUM_SCALES,  # rope_byte_off = 528
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out, valid


# ---------------------------------------------------------------------------
# FUSED gather-dequant + online-softmax attention (no [T,K,576] materialization)
# ---------------------------------------------------------------------------
# The materialized path (``_gather_dequant_fp8ds`` -> ``_materialized_attn_kernel``)
# writes a dense [T, K, 576] bf16 tensor (~4.83 GB at T=K=2048) then reads it
# back. Profiling on a 5090 (T=K=2048, H=24) showed the *attend* pass dominates
# (~90%), not the gather (~10%): the materialized attend kernel walks one
# candidate at a time with a scalar ``tl.sum(q*kv)`` reduction and a
# data-dependent ``if is_valid`` branch, which is both low-occupancy and
# memory-bound on the 4.83 GB re-read. The fused kernel below (a) never
# materializes the [T,K,576] tensor (dequant happens in-tile, in registers),
# eliminating that ~4.83 GB write+read of HBM traffic, and (b) tiles the
# candidate axis (BLOCK_N) so the QK score becomes a ``tl.dot`` (tensor-core)
# matmul and the per-candidate loop overhead is amortized by BLOCK_N.
#
# *** NoPE/RoPE D-split layout (2026-06-20 tuning) ***
# The head_dim is 576 = 512 NoPE + 64 RoPE. A naive single-tile layout has to
# round 576 up to the next power of two (1024) for ``tl.arange``/``tl.dot``,
# which (a) dots over 1024 columns instead of 576 (~1.78x wasted tensor-core
# FLOPs on QK) and (b) makes the per-candidate KV tile 1024*2 B wide, so at
# num_warps=4 the 99 KB opt-in shared-mem budget caps BLOCK_N at 16. By handling
# NoPE (512, already pow2) and RoPE (64, pow2) as two *separate* sub-blocks the
# QK score becomes ``q_nope.kv_nopeᵀ + q_rope.kv_ropeᵀ`` over exactly 512+64=576
# effective columns (no 1024 padding), the KV tile shrinks to (512+64)*2 B, and
# BLOCK_N=32 now fits in 99 KB. The value accumulation is over the NoPE block
# only (the RoPE columns are never part of the output value), so the PV dot is
# 512-wide instead of 1024-wide. Net: ~3.4-3.6x over the BLOCK_N=16 single-tile
# baseline at the cluster prefill shape (T=2048/4096, H=24). Pinned config (no
# runtime autotune): BLOCK_N=32, HEAD_BLOCK=16, num_warps=4, num_stages=1 — the
# only config that both fits 99 KB smem and maximizes throughput in the local
# sweep; num_stages>=2 overflows smem (~109 KB), BLOCK_N>=64 overflows, and
# HEAD_BLOCK=32 wastes lanes at H=24. See NOTES.md "Throughput tuning" section.
#
# Numerics are kept identical (within bf16 reduction-order tol) to the
# materialized path, verified against the fp64 reference (max err 1.5e-4 at the
# cluster shape, <=1.3e-3 across the suite, all << 2e-2 bf16 tol):
#   * the dequant math is copied verbatim from ``_gather_dequant_fp8ds_kernel``
#     (fp8 NoPE * per-128 fp32 scale; bf16 RoPE; -1/out-of-range -> zero row);
#   * the online-softmax recurrence is the same as ``_materialized_attn_kernel``,
#     just applied to a BLOCK_N tile at a time (a flash-style block reduction,
#     algebraically equivalent to the one-at-a-time recurrence);
#   * invalid (sentinel / OOB) candidates contribute a -inf score (zero weight),
#     exactly as the materialized path skips them via the ``valid`` mask.
# The int64 offset-promotion fix is preserved (slot*stride is computed in int64).
@triton.jit
def _fused_gather_dequant_attn_kernel(
    q_ptr,                # bf16 [T, H, D]
    cache_ptr,            # uint8 [total_slots, 656] (flattened paged cache)
    indices_ptr,          # int32 [T, K] global slot ids, -1 / >= total invalid
    lens_ptr,             # int32 [T] per-token valid-candidate count
    max_score_ptr,        # f32  [T, H] (pre-filled -inf)
    denom_ptr,            # f32  [T, H] (pre-zeroed)
    acc_ptr,              # f32  [T, H, D] (pre-zeroed; only [:, :, :512] written)
    stride_q_t,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_cache_s,
    stride_idx_t,
    stride_idx_k: tl.constexpr,
    stride_state_t,
    stride_state_h: tl.constexpr,
    stride_acc_t,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    total_slots,
    num_heads: tl.constexpr,
    nope_dim: tl.constexpr,        # 512 (pow2 -> BLOCK_NOPE)
    rope_dim: tl.constexpr,        # 64  (pow2 -> BLOCK_ROPE)
    quant_tile: tl.constexpr,      # 128
    scale_byte_off: tl.constexpr,  # 512
    rope_byte_off: tl.constexpr,   # 528
    num_candidates,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,      # 512
    BLOCK_ROPE: tl.constexpr,      # 64
):
    # int64 token id: matches the materialized kernels' offset-promotion fix.
    # (q/acc here are [T,H,D] so per-token offsets are small, but keep 64-bit
    # for parity and safety; the slot*stride below is the genuinely large one.)
    token_idx = tl.program_id(0).to(tl.int64)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    head_mask = head_offsets < num_heads
    nope_off = tl.arange(0, BLOCK_NOPE)            # 0..511
    rope_off = tl.arange(0, BLOCK_ROPE)            # 0..63
    nope_dmask = nope_off < nope_dim
    rope_dmask = rope_off < rope_dim
    scale_tile = nope_off // quant_tile            # per-128 scale index (0..3)

    # q split into NoPE [HEAD_BLOCK, 512] and RoPE [HEAD_BLOCK, 64] bf16 (tensor-
    # core dot operands), loaded once. Masked dims load as 0 (contribute nothing,
    # matching the materialized path which zero-pads q beyond head_dim).
    q_nope = tl.load(
        q_ptr + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h + nope_off[None, :] * stride_q_d,
        mask=head_mask[:, None] & nope_dmask[None, :], other=0.0,
    ).to(tl.bfloat16)
    q_rope = tl.load(
        q_ptr + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h + (nope_dim + rope_off[None, :]) * stride_q_d,
        mask=head_mask[:, None] & rope_dmask[None, :], other=0.0,
    ).to(tl.bfloat16)

    running_max = tl.full((HEAD_BLOCK,), -float("inf"), tl.float32)
    running_denom = tl.zeros((HEAD_BLOCK,), tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_NOPE), tl.float32)  # value = NoPE only

    # Per-token valid candidate count (matches the materialized path's
    # ``valid = valid & (col < lens)`` cap): only iterate tiles that can hold a
    # valid candidate. Per-cell sentinel/OOB masking still applied below.
    valid_len = tl.load(lens_ptr + token_idx)

    cand_base = tl.arange(0, BLOCK_N)
    for n0 in range(0, num_candidates, BLOCK_N):
        cand = n0 + cand_base                       # [BLOCK_N] candidate cols
        cand_in_range = cand < num_candidates
        slot = tl.load(
            indices_ptr + token_idx * stride_idx_t + cand * stride_idx_k,
            mask=cand_in_range,
            other=-1,
        )
        # Native semantics: invalid when slot<0 OR slot>=total_slots OR the
        # candidate is past the per-token valid length (topk_length cap).
        cand_valid = (slot >= 0) & (slot < total_slots) & (cand < valid_len)

        # int64 row offset (the overflow-prone term). base: [BLOCK_N, 1] ptrs.
        base = cache_ptr + slot.to(tl.int64)[:, None] * stride_cache_s

        # ---- NoPE [BLOCK_N, 512]: fp8_e4m3 bytes [0:512] * per-128 fp32 scale ----
        nmask = cand_valid[:, None] & nope_dmask[None, :]
        fp8_bytes = tl.load(base + nope_off[None, :], mask=nmask, other=0)
        fp8_vals = fp8_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_byte = scale_byte_off + scale_tile[None, :] * 4
        b0 = tl.load(base + scale_byte + 0, mask=nmask, other=0).to(tl.int32)
        b1 = tl.load(base + scale_byte + 1, mask=nmask, other=0).to(tl.int32)
        b2 = tl.load(base + scale_byte + 2, mask=nmask, other=0).to(tl.int32)
        b3 = tl.load(base + scale_byte + 3, mask=nmask, other=0).to(tl.int32)
        scale_bits = (
            (b0 & 0xFF)
            | ((b1 & 0xFF) << 8)
            | ((b2 & 0xFF) << 16)
            | ((b3 & 0xFF) << 24)
        )
        scale_f32 = scale_bits.to(tl.float32, bitcast=True)
        kv_nope = tl.where(nmask, fp8_vals * scale_f32, 0.0).to(tl.bfloat16)  # [BLOCK_N, 512]

        # ---- RoPE [BLOCK_N, 64]: bf16 values at byte offset 528 ----
        rmask = cand_valid[:, None] & rope_dmask[None, :]
        rope_ptr = (base + rope_byte_off).to(tl.pointer_type(tl.bfloat16))
        kv_rope = tl.load(rope_ptr + rope_off[None, :], mask=rmask, other=0.0)
        kv_rope = tl.where(rmask, kv_rope, 0.0).to(tl.bfloat16)               # [BLOCK_N, 64]

        # ---- scores: q_nope.kv_nopeᵀ + q_rope.kv_ropeᵀ over 512+64=576 ----
        # bf16xbf16 -> fp32 accumulate (matches the materialized path which
        # scored q.float() * kv.float(): the kv values are bf16 either way, so
        # the two bf16 dots reproduce the same 576-d inner product up to fp32
        # accumulation).
        scores = (
            tl.dot(q_nope, tl.trans(kv_nope))
            + tl.dot(q_rope, tl.trans(kv_rope))
        ).to(tl.float32) * scale                        # [HEAD_BLOCK, BLOCK_N]
        # Invalid candidates -> -inf so they get zero softmax weight (and a zero
        # kv row, so they also contribute nothing to the value accumulation).
        scores = tl.where(cand_valid[None, :], scores, -float("inf"))

        # ---- flash-style online-softmax block update over BLOCK_N ----
        tile_max = tl.max(scores, axis=1)               # [HEAD_BLOCK]
        next_max = tl.maximum(running_max, tile_max)
        # If a head has seen no valid candidate yet, next_max may be -inf; guard
        # exp(-inf - -inf)=exp(nan). When next_max is -inf the whole tile is
        # empty for this head, so previous_weight and p are forced to 0.
        safe_next = tl.where(next_max == -float("inf"), 0.0, next_max)
        previous_weight = tl.exp(running_max - safe_next)
        previous_weight = tl.where(running_max == -float("inf"), 0.0, previous_weight)
        p = tl.exp(scores - safe_next[:, None])         # [HEAD_BLOCK, BLOCK_N]
        p = tl.where(cand_valid[None, :], p, 0.0)
        tile_denom = tl.sum(p, axis=1)                  # [HEAD_BLOCK]
        # value accumulation over the NoPE block only (512); RoPE is not a value
        # dim, so the PV dot is 512-wide (was 1024-wide in the single-tile path).
        running_acc = running_acc * previous_weight[:, None] + tl.dot(
            p.to(tl.bfloat16), kv_nope
        ).to(tl.float32)
        running_denom = running_denom * previous_weight + tile_denom
        running_max = next_max

    state_base = token_idx * stride_state_t
    tl.store(
        max_score_ptr + state_base + head_offsets * stride_state_h,
        running_max,
        mask=head_mask,
    )
    tl.store(
        denom_ptr + state_base + head_offsets * stride_state_h,
        running_denom,
        mask=head_mask,
    )
    # store the value accumulator into the first 512 (NoPE) cols of the
    # [T, H, D=576] acc tensor; cols 512:576 stay zero (pre-zeroed) and are
    # discarded by the d_v=512 slice in the host wrapper, exactly as before.
    tl.store(
        acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + nope_off[None, :] * stride_acc_d,
        running_acc,
        mask=head_mask[:, None] & nope_dmask[None, :],
    )


# --- Pinned launch config for the fused kernel (local 5090 sm_120 sweep, valid
# on GB10 sm_121: identical 99 KB opt-in shared mem). This is a STATIC pin, NOT a
# runtime autotuner -- only this one (BLOCK_N, HEAD_BLOCK, num_warps, num_stages)
# is ever compiled, so the cluster warmup compiles exactly one kernel for the
# warmup shape (no autotune config explosion / re-benchmark stall on startup).
# BLOCK_N=32 is the largest candidate tile that fits 99 KB smem in the NoPE/RoPE
# split layout (BLOCK_N>=64 or num_stages>=2 overflow ~109 KB+). Override only for
# experiments via the kwargs below; production always uses the defaults.
_FUSED_BLOCK_N = 32
_FUSED_HEAD_BLOCK = 16
_FUSED_NUM_WARPS = 4
_FUSED_NUM_STAGES = 1


def _fused_gather_dequant_attend(
    q: torch.Tensor,         # [T, H, D] bf16
    cache_flat: torch.Tensor,  # [total_slots, 656] uint8
    indices: torch.Tensor,   # [T, K] int32 global slot ids
    lens: torch.Tensor,      # [T] int32 per-token valid candidate count
    scale: float,
    max_score: torch.Tensor,  # [T, H] f32 (pre-filled -inf)
    denom: torch.Tensor,      # [T, H] f32 (pre-zeroed)
    acc: torch.Tensor,        # [T, H, D] f32 (pre-zeroed)
    *,
    block_n: int = _FUSED_BLOCK_N,
    head_block: int | None = None,
    num_warps: int = _FUSED_NUM_WARPS,
    num_stages: int = _FUSED_NUM_STAGES,
) -> None:
    """Fused gather+dequant+online-softmax over the fp8_ds_mla paged cache.

    Equivalent (within bf16 tol) to ``_gather_dequant_fp8ds`` followed by
    ``_materialized_indexed_accumulate``, but never materializes the
    ``[T, K, 576]`` KV tensor. Uses the NoPE/RoPE D-split layout (no 1024-pad
    waste) with the pinned BLOCK_N=32 config; ~3.4-3.6x over the BLOCK_N=16
    single-tile path at the cluster prefill shape.
    """
    T, H, D = q.shape
    K = indices.shape[1]
    total_slots = cache_flat.shape[0]
    assert cache_flat.dtype == torch.uint8
    assert cache_flat.shape[1] == _FP8DS_ENTRY_BYTES
    assert D == _DQK, f"fused path expects head_dim {_DQK}, got {D}"
    assert max_score.shape == (T, H) and denom.shape == (T, H)
    assert acc.shape[:2] == (T, H)
    assert acc.shape[2] >= _DV
    if head_block is None:
        # tl.dot needs the M dim >= 16; pad small head counts up to 16.
        head_block = _FUSED_HEAD_BLOCK if H >= 16 else next_power_of_2(max(H, 1))
        head_block = max(head_block, 16)
    grid = (T, triton.cdiv(H, head_block))
    _fused_gather_dequant_attn_kernel[grid](
        q,
        cache_flat,
        indices,
        lens,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cache_flat.stride(0),
        indices.stride(0),
        indices.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        total_slots,
        H,
        _FP8DS_NOPE_DIM,                          # nope_dim = 512
        _FP8DS_ROPE_DIM,                          # rope_dim = 64
        _FP8DS_QUANT_TILE,
        _FP8DS_NOPE_DIM,                          # scale_byte_off = 512
        _FP8DS_NOPE_DIM + 4 * _FP8DS_NUM_SCALES,  # rope_byte_off = 528
        K,
        float(scale),
        HEAD_BLOCK=head_block,
        BLOCK_N=block_n,
        BLOCK_NOPE=_FP8DS_NOPE_DIM,               # 512 (pow2)
        BLOCK_ROPE=_FP8DS_ROPE_DIM,               # 64 (pow2)
        num_warps=num_warps,
        num_stages=num_stages,
    )


# Default to the fused path; set VLLM_SPARSE_MLA_FUSED=0 to fall back to the
# original materialized (gather-then-attend) path for debug/A-B comparison.
def _fused_enabled() -> bool:
    return os.getenv("VLLM_SPARSE_MLA_FUSED", "1") != "0"


# ---------------------------------------------------------------------------
# FP8 sparse decode : flash_mla_with_kvcache
# ---------------------------------------------------------------------------
def flash_mla_with_kvcache_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None = None,
    head_dim_v: int = 512,
    tile_scheduler_metadata=None,
    num_splits=None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton FP8 sparse-MLA decode, drop-in for ``flash_mla_with_kvcache``.

    Only the *sparse fp8* path used by the V3.2 backend is implemented:
      q       : [b, s_q, h_q, 576] bf16
      k_cache : [num_blocks, block_size, 1, 656] uint8  (fp8_ds_mla)
                (the V3.2 backend passes ``cache.view(uint8).unsqueeze(-2)``)
      indices : [b, s_q, topk] int32  GLOBAL slot ids, -1 / >= total invalid
      head_dim_v == 512, is_fp8_kvcache == True
    Returns (out [b, s_q, h_q, 512] bf16, lse [b, h_q, s_q] f32).

    extra_k_cache / extra_indices / *_topk_length are the DeepSeek-V4 SWA
    arguments; the V3.2 backend never sets them and they are asserted None.
    """
    assert indices is not None, "Triton flash_mla_with_kvcache only supports sparse mode"
    assert is_fp8_kvcache, "Triton decode path requires is_fp8_kvcache=True (fp8_ds_mla)"
    assert head_dim_v == _DV, f"head_dim_v must be {_DV}, got {head_dim_v}"
    assert extra_k_cache is None and extra_indices_in_kvcache is None, (
        "V3.2 sparse decode does not use extra_k_cache / extra_indices "
        "(those are DeepSeek-V4 SWA-only arguments)"
    )
    assert extra_topk_length is None
    assert not causal, "sparse attention requires causal=False"

    # q: [b, s_q, h_q, d_qk] -> flatten batch*seq to T tokens.
    assert q.dim() == 4, f"expected q [b, s_q, h_q, d_qk], got {q.shape}"
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == _DQK, f"expected d_qk == {_DQK}, got {d_qk}"
    q_flat = q.reshape(b * s_q, h_q, d_qk)

    # indices: [b, s_q, topk] -> [T, topk]
    assert indices.dim() == 3, f"expected indices [b, s_q, topk], got {indices.shape}"
    assert indices.shape[0] == b and indices.shape[1] == s_q
    topk = indices.shape[2]
    idx2d = indices.reshape(b * s_q, topk).contiguous()
    T = b * s_q

    if softmax_scale is None:
        softmax_scale = d_qk ** (-0.5)

    # k_cache may arrive as [num_blocks, block_size, 1, 656] (uint8) per the
    # V3.2 call site (``.view(uint8).unsqueeze(-2)``). Flatten to
    # [total_slots, 656]; global indices index into this flattened slot space
    # exactly as triton_convert_req_index_to_global_index produced them
    # (block_id * block_size + offset).
    kc = k_cache
    assert kc.dtype == torch.uint8, "fp8_ds_mla decode cache must be uint8"
    cache_flat = kc.reshape(-1, _FP8DS_ENTRY_BYTES)

    device = q.device
    # Per-token valid candidate count: topk_length when given (native cap),
    # else the full topk width (per-cell -1 / out-of-range masking still applies
    # in the kernels). Used by both the fused and the materialized paths.
    if topk_length is not None:
        lens = topk_length.reshape(-1).to(device=device, dtype=torch.int32).contiguous()
        assert lens.shape[0] == T
    else:
        lens = torch.full((T,), topk, dtype=torch.int32, device=device)

    fused_enabled = _fused_enabled()
    profile = _CudaProfileRegion(
        "attention.fp8_sparse_decode_or_mixed",
        q_shape=tuple(q.shape),
        k_cache_shape=tuple(k_cache.shape),
        topk=topk,
        fused=fused_enabled,
        topk_length="set" if topk_length is not None else "none",
    )
    profile.start()

    # State for online softmax over the gathered candidates (fp32).
    max_score = torch.full((T, h_q), float("-inf"), dtype=torch.float32, device=device)
    denom = torch.zeros((T, h_q), dtype=torch.float32, device=device)

    if fused_enabled:
        # FUSED path (default): gather+dequant+attend in-tile, no [T,K,576]
        # materialization. Only the 512 NoPE value dims are accumulated, which
        # avoids a 576-wide fp32 state tensor and a post-kernel contiguous copy.
        acc_v = torch.zeros((T, h_q, _DV), dtype=torch.float32, device=device)
        _fused_gather_dequant_attend(
            q_flat,
            cache_flat,
            idx2d,
            lens,
            float(softmax_scale),
            max_score,
            denom,
            acc_v,
        )
    else:
        # Legacy materialized path (VLLM_SPARSE_MLA_FUSED=0): gather the dense
        # [T, topk, 576] bf16 tensor then attend over it. Kept for debug / A-B.
        acc = torch.zeros((T, h_q, d_qk), dtype=torch.float32, device=device)
        kv_gathered, valid = _gather_dequant_fp8ds(cache_flat, idx2d)
        if topk_length is not None:
            col = torch.arange(topk, device=device, dtype=torch.int32)
            valid = valid & (col[None, :] < lens[:, None])
        _materialized_indexed_accumulate(
            q_flat, kv_gathered, valid, float(softmax_scale), max_score, denom, acc
        )
        acc_v = acc[:, :, :_DV].contiguous()

    # lse for return value (native returns lse [b, h_q, s_q]).
    safe_max = torch.where(denom > 0, max_score, torch.zeros_like(max_score))
    lse = torch.where(
        denom > 0,
        torch.log(denom.clamp_min(1e-30)) + safe_max,
        torch.full_like(max_score, float("-inf")),
    )

    if attn_sink is None:
        sink = torch.full((h_q,), float("-inf"), dtype=torch.float32, device=device)
    else:
        sink = attn_sink.to(device=device, dtype=torch.float32).contiguous()
        assert sink.shape[0] >= h_q

    if out is None:
        out = torch.empty((b, s_q, h_q, _DV), dtype=q.dtype, device=device)
    out_flat = out.reshape(T, h_q, _DV)

    finish_sparse_mla_attention_with_sink(
        max_score, denom, acc_v, sink, output=out_flat
    )

    # lse return shape: [b, h_q, s_q]
    lse_out = lse.reshape(b, s_q, h_q).permute(0, 2, 1).contiguous()
    profile.stop()
    return out, lse_out


@triton.jit
def _materialized_attn_kernel(
    q_ptr,           # bf16 [T, H, D]
    kv_ptr,          # bf16 [T, K, D] (per-query gathered)
    valid_ptr,       # bool [T, K]
    max_score_ptr,   # f32  [T, H]
    denom_ptr,       # f32  [T, H]
    acc_ptr,         # f32  [T, H, D]
    stride_q_t: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_t: tl.constexpr,
    stride_kv_k: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_valid_t: tl.constexpr,
    stride_valid_k: tl.constexpr,
    stride_state_t: tl.constexpr,
    stride_state_h: tl.constexpr,
    stride_acc_t: tl.constexpr,
    stride_acc_h: tl.constexpr,
    stride_acc_d: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_candidates,
    scale: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # int64 token id: the per-query gathered ``kv`` tensor is [T, K, D] and its
    # per-token stride (K*D) times ``token_idx`` overflows int32 for the
    # cluster's mixed-batch prefill (T=K=2048) exactly as in the gather kernel.
    # Promote so every ``token_idx*stride`` offset (q/kv/state/acc) is int64.
    token_idx = tl.program_id(0).to(tl.int64)
    head_block_idx = tl.program_id(1)
    head_offsets = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + token_idx * stride_q_t
        + head_offsets[:, None] * stride_q_h
        + dim_offsets[None, :] * stride_q_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((HEAD_BLOCK,), -float("inf"), tl.float32)
    running_denom = tl.zeros((HEAD_BLOCK,), tl.float32)
    running_acc = tl.zeros((HEAD_BLOCK, BLOCK_D), tl.float32)

    for candidate_idx in range(0, num_candidates):
        is_valid = tl.load(
            valid_ptr + token_idx * stride_valid_t + candidate_idx * stride_valid_k
        )
        if is_valid:
            kv = tl.load(
                kv_ptr
                + token_idx * stride_kv_t
                + candidate_idx * stride_kv_k
                + dim_offsets * stride_kv_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q * kv[None, :], axis=1) * scale
            next_max = tl.maximum(running_max, scores)
            previous_weight = tl.exp(running_max - next_max)
            candidate_weight = tl.exp(scores - next_max)
            running_acc = (
                running_acc * previous_weight[:, None]
                + kv[None, :] * candidate_weight[:, None]
            )
            running_denom = running_denom * previous_weight + candidate_weight
            running_max = next_max

    state_base = token_idx * stride_state_t
    tl.store(
        max_score_ptr + state_base + head_offsets * stride_state_h,
        running_max,
        mask=head_mask,
    )
    tl.store(
        denom_ptr + state_base + head_offsets * stride_state_h,
        running_denom,
        mask=head_mask,
    )
    tl.store(
        acc_ptr
        + token_idx * stride_acc_t
        + head_offsets[:, None] * stride_acc_h
        + dim_offsets[None, :] * stride_acc_d,
        running_acc,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def _materialized_indexed_accumulate(
    q: torch.Tensor,         # [T, H, D]
    kv: torch.Tensor,        # [T, K, D]  per-query gathered + dequantized
    valid: torch.Tensor,     # [T, K] bool
    scale: float,
    max_score: torch.Tensor,  # [T, H] f32 (pre-filled -inf)
    denom: torch.Tensor,      # [T, H] f32 (pre-zeroed)
    acc: torch.Tensor,        # [T, H, D] f32 (pre-zeroed)
) -> None:
    """Online-softmax accumulation over per-query materialized KV.

    Equivalent semantics to ``accumulate_indexed_sparse_mla_attention_chunk``
    but reads a *per-query* dequantized KV tensor (the decode case where each
    query attends to its own gathered candidate set) rather than a shared
    ``kv_flat`` indexed by global ids.
    """
    T, H, D = q.shape
    K = kv.shape[1]
    head_block = 8 if H >= 8 else 1
    block_d = min(1024, next_power_of_2(D))
    grid = (T, triton.cdiv(H, head_block))
    _materialized_attn_kernel[grid](
        q,
        kv,
        valid,
        max_score,
        denom,
        acc,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        valid.stride(0),
        valid.stride(1),
        max_score.stride(0),
        max_score.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        H,
        D,
        K,
        scale,
        HEAD_BLOCK=head_block,
        BLOCK_D=block_d,
        num_warps=4,
    )
