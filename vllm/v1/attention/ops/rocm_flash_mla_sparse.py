# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm fallback for DeepSeek-V4's FlashMLA sparse attention kernels.

The official FlashMLA kernels (``flash_mla_sparse_fwd`` for prefill and the
V4-extended ``flash_mla_with_kvcache`` for decode) are NVIDIA-only — they live
in the ``vllm._flashmla_C`` extension which is not built on ROCm. The wrapper in
``vllm/v1/attention/ops/flashmla.py`` raises ``RuntimeError`` for both calls on
non-CUDA platforms, which crashes DeepSeek-V4 inference at the first generation
step.

This module provides ROCm-friendly equivalents:

* ``flash_mla_sparse_fwd_rocm``  — sparse attention over a *bf16* KV pool. The
  V4 prefill path pre-dequantizes the FP8 cache via
  :func:`vllm.v1.attention.ops.deepseek_v4_ops.dequantize_and_gather_k_cache`
  (Triton, works on ROCm), then feeds bf16 ``kv`` into FlashMLA. We can run the
  same sparse softmax+gemm in chunked online-softmax form on top of the
  dequantized KV without needing the FP8-aware kernel.

* ``flash_mla_with_kvcache_rocm`` — decode path. Here FlashMLA reads the
  FP8 ``swa_cache`` (and optionally a global compressed ``extra_k_cache``)
  directly via ``is_fp8_kvcache=True``. We dequantize the requested slots on
  the fly with a small Triton kernel (mirroring
  ``_dequantize_and_gather_k_kernel`` but indexed by arbitrary global slot ids
  instead of a block table), then run the same chunked sparse attention.

* ``get_mla_metadata_rocm`` — returns an empty ``FlashMLASchedMeta`` stub so
  the V4 SWA metadata builder can populate ``tile_sched_*`` fields without
  crashing. The metadata is unused by our fallback path.

Both attention paths use *online softmax* with a bounded ``chunk_topk`` over
the candidate axis so peak intermediate memory stays manageable even with
many query tokens × thousands of selected positions.

Numerics notes
--------------
* The softmax includes the per-head ``attn_sink`` logit as an extra column
  whose value is dropped before the ``attn @ V`` reduction (matches FlashMLA
  semantics: sink mass affects the partition function only).
* Invalid ``indices == -1`` entries are masked with ``-inf`` so they never
  contribute, regardless of what we (safely) dequantize at slot 0.
* Rows where every candidate is invalid AND ``attn_sink == -inf`` produce a
  zero output (we trap the all-``-inf`` case to avoid NaNs from ``exp(0)/0``).
"""
from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Cache layout constants — must mirror
# vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py.
# ---------------------------------------------------------------------------
_FP8_DIM = 448
_BF16_DIM = 64
_SCALE_DIM = 8
_QUANT_BLOCK_SIZE = 64
_TOKEN_DATA_SIZE = _FP8_DIM + _BF16_DIM * 2  # 576
_HEAD_DIM = _FP8_DIM + _BF16_DIM  # 512
_N_QUANT_BLOCKS = 7  # 7 real (448 // 64), 1 pad slot at index 7

# Chunk size for online-softmax over the candidate axis. 128 keeps memory
# small (~64 MiB for T_q=512, head_dim=512, bf16) while letting the matmul
# inside torch see enough work to be efficient.
_DEFAULT_CHUNK_TOPK = 128


# ---------------------------------------------------------------------------
# FP8 slot dequantization (decode path).
# ---------------------------------------------------------------------------
if HAS_TRITON and current_platform.is_cuda_alike():

    @triton.jit
    def _gather_dequant_slots_kernel(
        out_ptr,  # (N, head_dim) bf16
        out_stride_n,
        indices_ptr,  # (N,) int32, -1 = invalid (still safely dequant slot 0)
        k_cache_ptr,  # uint8 byte buffer
        block_stride,  # bytes per block
        cache_block_size: tl.constexpr,
        fp8_dim: tl.constexpr,
        bf16_dim: tl.constexpr,
        scale_dim: tl.constexpr,
        quant_block: tl.constexpr,
        token_data_size: tl.constexpr,
        head_dim: tl.constexpr,
        n_quant_blocks: tl.constexpr,
        N,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        raw_slot = tl.load(indices_ptr + pid)
        # Always dequant slot >= 0 to keep the kernel branch-free; the
        # caller masks invalid indices in the attention softmax.
        slot = tl.maximum(raw_slot, 0)

        out_row_ptr = out_ptr + pid * out_stride_n

        block_idx = (slot // cache_block_size).to(tl.int64)
        pos_in_block = slot % cache_block_size

        cache_block_ptr = k_cache_ptr + block_idx * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim

        # Dequantize the 448 FP8 dims in 7 blocks of 64.
        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block
            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim
                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
                x_float = x_fp8.to(tl.float32)
                encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                exponent = encoded_scale.to(tl.float32) - 127.0
                scale = tl.exp2(exponent)
                x_dequant = x_float * scale
                tl.store(
                    out_row_ptr + offsets,
                    x_dequant.to(tl.bfloat16),
                    mask=mask,
                )

        # Copy the trailing 64 bf16 dims unchanged.
        bf16_out_ptr = out_row_ptr + fp8_dim
        bf16_cache_bf16_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_bf16_ptr + chunk_offsets)
            tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)
else:
    _gather_dequant_slots_kernel = None  # type: ignore[assignment]


def _gather_dequant_slots_triton(
    indices: torch.Tensor,  # (N,) int32 — global slot ids, -1 allowed
    k_cache: torch.Tensor,  # uint8 (num_blocks, ...) — byte buffer
    out: torch.Tensor,  # (N, head_dim) bf16 output buffer
) -> None:
    """Triton gather + UE8M0 FP8 dequant for arbitrary global slot ids."""
    assert _gather_dequant_slots_kernel is not None
    assert k_cache.dtype == torch.uint8, (
        f"k_cache must be uint8 byte buffer, got {k_cache.dtype}"
    )
    assert out.dtype == torch.bfloat16
    assert out.shape == (indices.shape[0], _HEAD_DIM)
    assert indices.is_contiguous()
    assert out.is_contiguous()

    block_stride = k_cache.stride(0)
    n = indices.shape[0]
    if n == 0:
        return

    # Block size in *tokens*. The cache is shaped (num_blocks, block_size, 584)
    # in the metadata, so dim 1 is the token count per block.
    if k_cache.dim() >= 2:
        cache_block_size = k_cache.shape[1]
    else:
        # 1D byte buffer; assume 64 (the default DeepSeek block size).
        cache_block_size = 64

    _gather_dequant_slots_kernel[(n,)](
        out,
        out.stride(0),
        indices,
        k_cache,
        block_stride,
        cache_block_size=cache_block_size,
        fp8_dim=_FP8_DIM,
        bf16_dim=_BF16_DIM,
        scale_dim=_SCALE_DIM,
        quant_block=_QUANT_BLOCK_SIZE,
        token_data_size=_TOKEN_DATA_SIZE,
        head_dim=_HEAD_DIM,
        n_quant_blocks=_N_QUANT_BLOCKS,
        N=n,
    )


def _gather_dequant_slots_torch(
    indices: torch.Tensor,
    k_cache: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Pure-torch reference for ``_gather_dequant_slots_triton``.

    Slow but correct — useful for environments without a Triton runtime and
    for unit-style sanity checks. Implements the same UE8M0 FP8 dequant + bf16
    copy as the Triton kernel.
    """
    assert k_cache.dtype == torch.uint8
    assert out.dtype == torch.bfloat16
    n = indices.shape[0]
    if n == 0:
        return

    block_stride = k_cache.stride(0)
    cache_block_size = k_cache.shape[1] if k_cache.dim() >= 2 else 64
    flat_cache = k_cache.view(torch.uint8).contiguous().view(-1)

    safe = indices.clamp(min=0).to(torch.int64)
    block_idx = safe // cache_block_size
    pos_in_block = safe % cache_block_size

    # Per-token base byte offsets for the data and scale regions.
    base = block_idx * block_stride
    data_base = base + pos_in_block * _TOKEN_DATA_SIZE  # (N,)
    scale_base = (
        base + cache_block_size * _TOKEN_DATA_SIZE + pos_in_block * _SCALE_DIM
    )  # (N,)

    # ---- FP8 NoPE (448 dims) ----
    fp8_offsets = data_base.unsqueeze(-1) + torch.arange(
        _FP8_DIM, device=indices.device, dtype=torch.int64
    )
    fp8_bytes = flat_cache[fp8_offsets.flatten()].view(n, _FP8_DIM)
    fp8_vals = fp8_bytes.view(torch.float8_e4m3fn).to(torch.float32)

    # 7 UE8M0 scales, 1 byte each.
    scale_offsets = scale_base.unsqueeze(-1) + torch.arange(
        _N_QUANT_BLOCKS, device=indices.device, dtype=torch.int64
    )
    scale_bytes = flat_cache[scale_offsets.flatten()].view(n, _N_QUANT_BLOCKS)
    exponents = scale_bytes.to(torch.float32) - 127.0
    scales = torch.exp2(exponents)  # (N, 7)
    # Repeat each scale across its 64-element block.
    scales_per_dim = scales.repeat_interleave(_QUANT_BLOCK_SIZE, dim=-1)
    nope = (fp8_vals * scales_per_dim).to(torch.bfloat16)

    # ---- BF16 RoPE (64 dims) ----
    bf16_byte_offsets = (
        data_base + _FP8_DIM
    ).unsqueeze(-1) + torch.arange(
        _BF16_DIM * 2, device=indices.device, dtype=torch.int64
    )
    bf16_bytes = flat_cache[bf16_byte_offsets.flatten()].view(n, _BF16_DIM * 2)
    rope = bf16_bytes.view(torch.bfloat16).view(n, _BF16_DIM)

    out.copy_(torch.cat([nope, rope], dim=-1))


def _gather_dequant_slots(
    indices: torch.Tensor,
    k_cache: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Dispatch to Triton when available, otherwise pure torch."""
    if _gather_dequant_slots_kernel is not None and indices.is_cuda:
        _gather_dequant_slots_triton(indices, k_cache, out)
    else:
        _gather_dequant_slots_torch(indices, k_cache, out)


# ---------------------------------------------------------------------------
# Sparse attention with online softmax (chunked over the candidate axis).
# ---------------------------------------------------------------------------
def _online_softmax_init(
    t_q: int,
    num_heads: int,
    head_dim_v: int,
    attn_sink: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Seed the (m, l, O) running state with the per-head ``attn_sink`` logit.

    The sink contributes mass exp(sink) to the partition function but no V
    contribution, so we initialize:
      m = sink (or -inf if no sink)
      l = exp(sink - m) = 1 (or 0 if sink == -inf)
      O = 0
    """
    if attn_sink is not None:
        sink = attn_sink.to(torch.float32).view(1, num_heads).expand(t_q, num_heads)
        m = sink.contiguous()
    else:
        m = torch.full((t_q, num_heads), float("-inf"), dtype=torch.float32, device=device)

    finite_sink = torch.isfinite(m)
    l = torch.where(finite_sink, torch.ones_like(m), torch.zeros_like(m))
    O = torch.zeros((t_q, num_heads, head_dim_v), dtype=torch.float32, device=device)
    return m, l, O


def _online_softmax_update(
    m: torch.Tensor,  # (T_q, H) running max
    l: torch.Tensor,  # (T_q, H) running denominator
    O: torch.Tensor,  # (T_q, H, head_dim_v) running output (fp32)
    scores: torch.Tensor,  # (T_q, H, c) new logits (fp32, -inf for invalid)
    V_chunk: torch.Tensor,  # (T_q, c, head_dim_v) bf16/fp32 V values
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One online-softmax step.

    Numerical care: when a row's running max stays ``-inf`` (no candidate yet
    finite) we keep ``O = 0`` and ``l = 0`` and just track the new max so the
    next chunk can rebase from it.
    """
    chunk_max = scores.amax(dim=-1)  # (T_q, H)
    new_m = torch.maximum(m, chunk_max)  # (T_q, H)

    # Avoid -inf - -inf = nan when both old and new max are still -inf.
    finite_old = torch.isfinite(m) & torch.isfinite(new_m)  # (T_q, H)
    scale_old = torch.where(
        finite_old,
        torch.exp(m - torch.where(finite_old, new_m, m)),
        torch.zeros_like(m),
    )  # (T_q, H)

    # Per-element diff: -inf - finite = -inf; finite - -inf would blow up so
    # only subtract when new_m is finite. Keep the 2D mask for building
    # ``safe_new_m`` (same shape as ``new_m``); unsqueeze separately for the
    # 3D mask used against ``scores``.
    finite_new_2d = torch.isfinite(new_m)  # (T_q, H)
    safe_new_m = torch.where(
        finite_new_2d, new_m, torch.zeros_like(new_m)
    ).unsqueeze(-1)  # (T_q, H, 1)
    finite_new_3d = finite_new_2d.unsqueeze(-1)  # (T_q, H, 1)
    e_scores = torch.where(
        finite_new_3d & torch.isfinite(scores),
        torch.exp(scores - safe_new_m),
        torch.zeros_like(scores),
    )  # (T_q, H, c)

    l_new = l * scale_old + e_scores.sum(dim=-1)  # (T_q, H)
    # O_new = scale_old * O + e_scores @ V_chunk
    O_new = O * scale_old.unsqueeze(-1) + torch.einsum(
        "thc,tcd->thd", e_scores, V_chunk.to(torch.float32)
    )  # (T_q, H, head_dim_v)
    return new_m, l_new, O_new


def _sparse_attn_chunked(
    q: torch.Tensor,  # (T_q, H, head_dim) bf16/fp32
    indices: torch.Tensor,  # (T_q, max_topk) int32, -1 for invalid
    K_provider,  # callable: (idx_chunk: (T_q, c) int32) -> (T_q, c, head_dim) bf16
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    head_dim_v: int,
    chunk_topk: int = _DEFAULT_CHUNK_TOPK,
) -> torch.Tensor:
    """Generic sparse attention with online softmax.

    ``K_provider`` is a callable that returns the dequantized K (bf16) for a
    chunk of candidate indices. This lets the same attention loop drive both
    the prefill path (already-dequantized bf16 KV pool, simple ``K_full[idx]``
    gather) and the decode path (per-slot Triton FP8 dequant).
    """
    t_q, num_heads, _ = q.shape
    max_topk = indices.shape[-1]
    device = q.device

    m, l, O = _online_softmax_init(t_q, num_heads, head_dim_v, attn_sink, device)
    q_f = q.to(torch.float32)

    for cs in range(0, max_topk, chunk_topk):
        ce = min(cs + chunk_topk, max_topk)
        idx_chunk = indices[:, cs:ce].contiguous()  # (T_q, c)
        valid = idx_chunk >= 0  # (T_q, c)
        if not valid.any():
            continue

        K_chunk = K_provider(idx_chunk)  # (T_q, c, head_dim) bf16

        scores = torch.einsum(
            "thd,tcd->thc", q_f, K_chunk.to(torch.float32)
        ) * sm_scale  # (T_q, H, c)
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

        V_chunk = K_chunk[..., :head_dim_v]
        m, l, O = _online_softmax_update(m, l, O, scores, V_chunk)

    # Finalize: divide by total partition function.
    finite_l = l > 0
    out_f = torch.where(
        finite_l.unsqueeze(-1), O / l.clamp_min(1e-30).unsqueeze(-1), torch.zeros_like(O)
    )
    return out_f


# ---------------------------------------------------------------------------
# Prefill: K is already dequantized to bf16 by the caller.
# ---------------------------------------------------------------------------
def flash_mla_sparse_fwd_rocm(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    head_dim_v: int | None = None,
    chunk_topk: int = _DEFAULT_CHUNK_TOPK,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """ROCm fallback for ``flash_mla_sparse_fwd``.

    Args:
        q: ``(s_q, h_q, d)`` bf16 query.
        kv: ``(s_kv, 1, d)`` bf16 KV pool (already dequantized + gathered).
        indices: ``(s_q, 1, topk)`` int32 with -1 sentinel for invalid slots.
        sm_scale: softmax scale factor.
        attn_sink: optional ``(h_q,)`` per-head sink logit (fp32).
        topk_length: kept for API parity; we use the -1 sentinel for masking.
        out: optional ``(s_q, h_q, d_v_or_d)`` bf16 output buffer.
        head_dim_v: V head dim (default = ``out.shape[-1]`` or ``d``).

    Returns ``(out, max_logits, lse)`` matching the upstream signature; the
    optional aux outputs are ``None`` since the caller only reads ``out``.
    """
    del topk_length  # unused: -1 sentinel masking is sufficient
    assert kv.dim() == 3 and kv.shape[1] == 1, (
        f"kv must be (s_kv, 1, d), got {kv.shape}"
    )
    assert indices.dim() == 3 and indices.shape[1] == 1, (
        f"indices must be (s_q, 1, topk), got {indices.shape}"
    )

    t_q, num_heads, head_dim = q.shape
    if head_dim_v is None:
        head_dim_v = out.shape[-1] if out is not None else head_dim
    head_dim_v = min(head_dim_v, head_dim)

    K = kv.squeeze(1)  # (N_kv, d)
    idx_2d = indices.squeeze(1)  # (T_q, max_topk)

    def K_provider(idx_chunk: torch.Tensor) -> torch.Tensor:
        safe = idx_chunk.clamp(min=0).to(torch.int64)
        return K[safe]

    out_f = _sparse_attn_chunked(
        q=q,
        indices=idx_2d,
        K_provider=K_provider,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        head_dim_v=head_dim_v,
        chunk_topk=chunk_topk,
    )

    if out is None:
        out = torch.empty(t_q, num_heads, head_dim_v, dtype=q.dtype, device=q.device)
    out[..., :head_dim_v].copy_(out_f.to(out.dtype))
    if out.shape[-1] > head_dim_v:
        out[..., head_dim_v:].zero_()
    return out, None, None


# ---------------------------------------------------------------------------
# Decode: K cache is FP8-packed; dequantize requested slots on the fly.
# ---------------------------------------------------------------------------
def _gather_chunk_to_bf16(
    idx_chunk: torch.Tensor,  # (T_q, c) int32
    k_cache: torch.Tensor,  # uint8 byte buffer
) -> torch.Tensor:
    """Dequantize `(T_q, c)` cache slots into a `(T_q, c, head_dim)` bf16
    tensor."""
    t_q, c = idx_chunk.shape
    flat_idx = idx_chunk.reshape(-1).to(torch.int32).contiguous()
    flat_out = torch.empty(
        (flat_idx.shape[0], _HEAD_DIM),
        dtype=torch.bfloat16,
        device=idx_chunk.device,
    )
    _gather_dequant_slots(flat_idx, k_cache, flat_out)
    return flat_out.view(t_q, c, _HEAD_DIM)


def flash_mla_with_kvcache_rocm(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None = None,
    head_dim_v: int = _HEAD_DIM,
    tile_scheduler_metadata: object | None = None,
    cache_seqlens: torch.Tensor | None = None,
    is_fp8_kvcache: bool = True,
    indices: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    causal: bool = False,
    chunk_topk: int = _DEFAULT_CHUNK_TOPK,
    **_unused_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """ROCm fallback for V4-extended ``flash_mla_with_kvcache``.

    Decodes one query token per batch position by sparse attention over up to
    two FP8-packed caches:

    * ``k_cache``  + ``indices`` / ``topk_length``                  (SWA)
    * ``extra_k_cache`` + ``extra_indices_in_kvcache`` / ``extra_topk_length``
      (global compressed cache, optional — only present on layers with
      ``compress_ratio > 1``)

    The two index sets are concatenated into a single virtual KV pool with a
    chunked online softmax that includes the per-head ``attn_sink``.

    Args mirror the V4 call site in ``deepseek_v4_attention._forward_decode``.
    Unused-on-ROCm kwargs (``tile_scheduler_metadata``, ``cache_seqlens``,
    ``num_splits``, ``causal``) are accepted for API compatibility.
    """
    del tile_scheduler_metadata, cache_seqlens, block_table, causal
    del topk_length, extra_topk_length  # -1 sentinel masking is sufficient

    assert is_fp8_kvcache, (
        "rocm flash_mla_with_kvcache fallback requires is_fp8_kvcache=True "
        "(DeepSeek-V4 always quantizes KV cache to UE8M0 FP8)"
    )
    assert indices is not None, "SWA indices must be provided for V4 decode"
    assert q.dim() == 4 and q.shape[1] == 1, (
        f"q must be (batch, 1, num_heads, head_dim), got {q.shape}"
    )
    assert indices.dim() == 3 and indices.shape[1] == 1, (
        f"indices must be (batch, 1, max_swa_topk), got {indices.shape}"
    )

    batch_size, _, num_heads, head_dim = q.shape
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    head_dim_v = min(head_dim_v, head_dim)

    q_2d = q.squeeze(1)  # (batch, H, head_dim)
    swa_idx = indices.squeeze(1)  # (batch, max_swa_topk)

    if extra_k_cache is not None:
        assert extra_indices_in_kvcache is not None
        assert extra_indices_in_kvcache.dim() == 3
        extra_idx = extra_indices_in_kvcache.squeeze(1)  # (batch, max_extra_topk)
    else:
        extra_idx = None

    # Concatenate SWA + extra index sets into one virtual pool. Each pool gets
    # its own dequantization closure; the index encoding tags which pool.
    swa_topk = swa_idx.shape[-1]
    extra_topk = extra_idx.shape[-1] if extra_idx is not None else 0
    total_topk = swa_topk + extra_topk

    # Build a single (batch, total_topk) index tensor where the second half is
    # offset by a sentinel so the dispatcher can route to the right cache.
    # Encoding: pool 0 = SWA (raw index), pool 1 = extra (index + 2^30).
    # We rely on a closure capturing the boundary instead of bit-twiddling
    # so torch.int32 stays clean.
    if extra_idx is None:
        combined_idx = swa_idx
    else:
        combined_idx = torch.cat([swa_idx, extra_idx], dim=-1)

    # Carry which slice maps to which cache by partitioning chunks at the
    # SWA/extra boundary inside the loop. To keep _sparse_attn_chunked
    # generic, do the chunking here and call the online-softmax primitives.
    device = q.device
    m, l, O = _online_softmax_init(batch_size, num_heads, head_dim_v, attn_sink, device)
    q_f = q_2d.to(torch.float32)

    def step(idx_chunk: torch.Tensor, cache: torch.Tensor) -> None:
        nonlocal m, l, O
        valid = idx_chunk >= 0
        if not valid.any():
            return
        K_chunk = _gather_chunk_to_bf16(idx_chunk, cache)
        scores = torch.einsum(
            "thd,tcd->thc", q_f, K_chunk.to(torch.float32)
        ) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        V_chunk = K_chunk[..., :head_dim_v]
        m, l, O = _online_softmax_update(m, l, O, scores, V_chunk)

    # Pool 0: SWA cache.
    for cs in range(0, swa_topk, chunk_topk):
        ce = min(cs + chunk_topk, swa_topk)
        step(swa_idx[:, cs:ce].contiguous(), k_cache)

    # Pool 1: extra (global compressed) cache.
    if extra_idx is not None:
        for cs in range(0, extra_topk, chunk_topk):
            ce = min(cs + chunk_topk, extra_topk)
            step(extra_idx[:, cs:ce].contiguous(), extra_k_cache)

    finite_l = l > 0
    out_f = torch.where(
        finite_l.unsqueeze(-1),
        O / l.clamp_min(1e-30).unsqueeze(-1),
        torch.zeros_like(O),
    )

    if out is None:
        out = torch.empty(
            (batch_size, 1, num_heads, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )
    out_view = out.squeeze(1)
    out_view[..., :head_dim_v].copy_(out_f.to(out.dtype))
    if out_view.shape[-1] > head_dim_v:
        out_view[..., head_dim_v:].zero_()

    # Upstream returns (out, softmax_lse). LSE isn't consumed by the V4 caller.
    return out, None


# ---------------------------------------------------------------------------
# Stubs for FlashMLA's planner-side helpers.
# ---------------------------------------------------------------------------
class _FlashMLASchedMetaStub:
    """Placeholder ``FlashMLASchedMeta`` for ROCm.

    The real CUDA struct holds tile-scheduler tensors that are populated by
    the in-kernel planner on first use. Our fallback ignores it but the V4
    metadata builder still allocates one per layer type.
    """

    have_initialized: bool = False
    tile_scheduler_metadata: torch.Tensor | None = None
    num_splits: torch.Tensor | None = None


def get_mla_metadata_rocm(*_args, **_kwargs) -> tuple[_FlashMLASchedMetaStub, None]:
    """ROCm stub for FlashMLA's ``get_mla_metadata``.

    Returns a fresh empty scheduler-metadata struct so the V4
    ``DeepseekSparseSWAMetadataBuilder.build_tile_scheduler`` can populate
    its per-layer-type cache without crashing on platforms without FlashMLA.
    """
    return _FlashMLASchedMetaStub(), None


__all__ = [
    "flash_mla_sparse_fwd_rocm",
    "flash_mla_with_kvcache_rocm",
    "get_mla_metadata_rocm",
]
