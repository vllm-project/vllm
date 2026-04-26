# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AITER-accelerated sparse MLA decode for DeepSeek V4 on ROCm (MI355X / gfx950).

Drop-in replacement for `DeepseekV4MLAAttention._ref_sparse_attn_decode`.
Uses `aiter.mla.mla_decode_fwd` with persistent-mode ASM kernels and FP8
inputs to get 2-3x speedup at high batch sizes.

Key design decisions (validated on MI355X, see benchmarks/dsv4_mi355/PLAN.md §12):
  - FP8/FP8 path only: gfx950 persistent-mode ASM kernels with return_lse=True
    exist ONLY for FP8/FP8 (not BF16). We need LSE for attn_sink correction.
  - Fixed-stride kv_indices: persistent-mode expects (total_q * topk) flat
    layout with -1 sentinels for invalid entries; NOT ragged.
  - Per-scope scratch: SWA and extra scopes have different topk, requiring
    independent AITER metadata buffers.
  - Cudagraph-safe: all per-step indexing tensors, the FP8 query buffer, the
    output buffer, and the constant scale tensors are preallocated in
    `AiterSparseScratch` and rewritten in-place so cudagraph capture sees
    stable memory layouts.
  - Per-layer-sized intermediates (the bf16 dequant of the paged K cache and
    its bf16->fp8 cast) are NOT owned by `AiterSparseScratch`; they are
    routed through `current_workspace_manager()` by the caller via the
    `kv_fp8_buf` / `extra_kv_fp8_buf` arguments. That lets a single pair of
    bf16+fp8 buffers be shared across all 61 DSv4 layers instead of
    growing the cudagraph memory pool by 61x worth of fresh allocations.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class AiterSparseScratch:
    """Cached per-step AITER persistent-mode scratch buffers (cudagraph-safe).

    Allocate once per `(total_q, nhead, topk, d_qk, d_v, dtype, kvtype)` key
    and reuse across all 61 DSv4 attention layers in the same decode step.
    Buffers fall into three groups:

      * AITER persistent metadata (`work_*`, `reduce_*`) — sized at rebuild
        from `get_mla_metadata_info_v1` (purely shape-determined), but rewritten
        in-place by `aiter.get_mla_metadata_v1` every step, because the work
        plan encodes the *actual* kv lengths and the persistent ASM kernel
        reads out-of-bounds if it is left stale.
      * Per-step indexing/IO buffers (`qo_indptr`, `kv_indptr`, `kv_indices_2d`,
        `kv_last_page_lens`, `valid_mask`, `valid_lens`, `col_arange`, `q_fp8`,
        `out_buf`) — written in-place each step.
      * Constant scale tensors (`q_scale`, `kv_scale`) — initialised once.

    The metadata buffers, the per-step buffers and the scale tensors all keep
    stable `data_ptr()`s across the entire lifetime of a shape key, so a
    HIP/CUDA graph captured around the second or later step replays correctly.
    """

    __slots__ = (
        # AITER persistent metadata buffers
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
        # Per-step indexing buffers
        "qo_indptr",
        "kv_indptr",
        "kv_indices_2d",
        "kv_last_page_lens",
        "valid_mask",
        "valid_lens",
        "col_arange",
        # FP8 query buffer + output buffer
        "q_fp8",
        "out_buf",
        # Constant scale tensors (always 1.0 for our quantization scheme)
        "q_scale",
        "kv_scale",
        # GQA ratios captured at rebuild time so per-step refresh can call
        # `get_mla_metadata_v1` with the same parameters every time.
        "_gqa_ratio",
        "_nhead_kv",
        "_page_size",
        "_topk",
        "_dtype",
        "_kvtype",
        "_max_split_per_batch",
        # Identity key for cache lookups
        "_key",
    )

    def __init__(self) -> None:
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._key = ()

    def matches(
        self,
        total_q: int,
        nhead: int,
        topk: int,
        d_qk: int,
        d_v: int,
        dtype: torch.dtype,
        kvtype: torch.dtype,
    ) -> bool:
        return self._key == (total_q, nhead, topk, d_qk, d_v, dtype, kvtype)

    def rebuild(
        self,
        *,
        total_q: int,
        nhead: int,
        nhead_kv: int,
        topk: int,
        d_qk: int,
        d_v: int,
        page_size: int,
        dtype: torch.dtype,
        kvtype: torch.dtype,
        device: torch.device,
        max_split_per_batch: int = 256,
    ) -> None:
        """Allocate every persistent buffer for the given shape key.

        Buffer sizes returned by `get_mla_metadata_info_v1` are determined by
        shapes and `max_split_per_batch` only, so they are large enough for
        any kv-length distribution. The actual work plan is computed on the
        per-step path by `refresh_metadata`, which writes these buffers
        in-place using the freshly populated `qo_indptr`/`kv_indptr`/
        `kv_last_page_lens` -- those pointers stay stable for the lifetime
        of this scratch.
        """
        import aiter

        # ---- AITER persistent metadata buffers (sizes only) ------------- #
        (
            (wmd_size, wmd_type),
            (wi_size, wi_type),
            (wis_size, wis_type),
            (ri_size, ri_type),
            (rfm_size, rfm_type),
            (rpm_size, rpm_type),
        ) = aiter.get_mla_metadata_info_v1(
            total_q,
            1,
            nhead,
            dtype,
            kvtype,
            is_sparse=True,
            fast_mode=True,
            num_kv_splits=max_split_per_batch,
        )
        self.work_meta_data = torch.empty(wmd_size, dtype=wmd_type, device=device)
        self.work_indptr = torch.empty(wi_size, dtype=wi_type, device=device)
        self.work_info_set = torch.empty(wis_size, dtype=wis_type, device=device)
        self.reduce_indptr = torch.empty(ri_size, dtype=ri_type, device=device)
        self.reduce_final_map = torch.empty(rfm_size, dtype=rfm_type, device=device)
        self.reduce_partial_map = torch.empty(rpm_size, dtype=rpm_type, device=device)

        # ---- Per-step indexing buffers ---------------------------------- #
        # qo_indptr is always [0, 1, 2, ..., total_q] for one query per token.
        self.qo_indptr = torch.arange(
            total_q + 1, dtype=torch.int32, device=device
        )
        self.kv_indptr = torch.zeros(
            total_q + 1, dtype=torch.int32, device=device
        )
        self.kv_indices_2d = torch.empty(
            (total_q, topk), dtype=torch.int32, device=device
        )
        # kv_last_page_lens is always all-ones for our 1-token-per-page layout.
        self.kv_last_page_lens = torch.ones(
            total_q, dtype=torch.int32, device=device
        )
        self.valid_mask = torch.empty(
            (total_q, topk), dtype=torch.bool, device=device
        )
        self.valid_lens = torch.empty(
            total_q, dtype=torch.int32, device=device
        )
        self.col_arange = torch.arange(topk, dtype=torch.int32, device=device)

        # ---- FP8 query + bf16 output buffers --------------------------- #
        self.q_fp8 = torch.empty(
            (total_q, nhead, d_qk), dtype=dtype, device=device
        )
        self.out_buf = torch.empty(
            (total_q, nhead, d_v), dtype=torch.bfloat16, device=device
        )

        # ---- Constant scale tensors (1.0 for our quant scheme) --------- #
        self.q_scale = torch.ones(1, dtype=torch.float32, device=device)
        self.kv_scale = torch.ones(1, dtype=torch.float32, device=device)

        # Cache parameters for `refresh_metadata`.
        self._gqa_ratio = nhead // nhead_kv
        self._nhead_kv = nhead_kv
        self._page_size = page_size
        self._topk = topk
        self._dtype = dtype
        self._kvtype = kvtype
        self._max_split_per_batch = max_split_per_batch

        self._key = (total_q, nhead, topk, d_qk, d_v, dtype, kvtype)

    def refresh_metadata(self) -> None:
        """Re-run `aiter.get_mla_metadata_v1` against the current
        `kv_indptr` / `kv_last_page_lens`, writing the new work plan into the
        same `work_*` / `reduce_*` buffers in-place.

        Must be called every step *after* `kv_indptr` is updated and *before*
        `aiter.mla.mla_decode_fwd`. The persistent ASM kernel reads
        out-of-bounds if it is left with a stale work plan, so this call is
        not optional even when shapes are unchanged.
        """
        import aiter

        aiter.get_mla_metadata_v1(
            self.qo_indptr,
            self.kv_indptr,
            self.kv_last_page_lens,
            self._gqa_ratio,
            self._nhead_kv,
            True,
            self.work_meta_data,
            self.work_info_set,
            self.work_indptr,
            self.reduce_indptr,
            self.reduce_final_map,
            self.reduce_partial_map,
            page_size=self._page_size,
            kv_granularity=max(self._page_size, 16),
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=True,
            max_split_per_batch=self._max_split_per_batch,
            topk=self._topk,
            dtype_q=self._dtype,
            dtype_kv=self._kvtype,
        )


def aiter_sparse_attn_decode(
    *,
    q: torch.Tensor,
    blocked_k: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    scale: float,
    head_dim: int,
    extra_blocked_k: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    scratch: AiterSparseScratch | None = None,
    extra_scratch: AiterSparseScratch | None = None,
    kv_fp8_buf: torch.Tensor | None = None,
    extra_kv_fp8_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """AITER-backed replacement for _ref_sparse_attn_decode.

    Args:
        q: (b, 1, h_q, d_qk) bf16 — unsqueezed decode query
        blocked_k: (n_blk, blk_sz, 1, d_qk) bf16 — dequantized SWA K cache
        indices_in_kvcache: (b, 1, topk_swa) int32
        topk_length: (b,) int32 or None
        attn_sink: (h_q,) fp32 or None
        scale: softmax scale (1/sqrt(d_qk))
        head_dim: kv_lora_rank (d_v, typically 512)
        extra_blocked_k: optional second scope K cache
        extra_indices_in_kvcache: optional second scope indices
        extra_topk_length: optional second scope lengths
        scratch: persistent scratch for SWA scope
        extra_scratch: persistent scratch for extra scope
        kv_fp8_buf: optional preallocated fp8 buffer (same shape as `blocked_k`)
            for the SWA scope's bf16->fp8 cast. Lets the caller route the cast
            through `current_workspace_manager()` so the buffer is shared
            across all 61 DSv4 layers; otherwise allocates fresh per call.
        extra_kv_fp8_buf: same idea for the extra scope.

    Returns: (b, h_q, d_v) bf16
    """
    b, s_q, h_q, d_qk = q.shape
    d_v = head_dim

    if scratch is None:
        scratch = AiterSparseScratch()
    if extra_scratch is None:
        extra_scratch = AiterSparseScratch()

    out_swa, lse_swa = _aiter_decode_one_scope(
        q=q,
        blocked_k=blocked_k,
        indices=indices_in_kvcache,
        lens=topk_length,
        sm_scale=scale,
        d_v=d_v,
        scratch=scratch,
        kv_fp8_buf=kv_fp8_buf,
    )

    if extra_blocked_k is not None:
        assert extra_indices_in_kvcache is not None
        out_ext, lse_ext = _aiter_decode_one_scope(
            q=q,
            blocked_k=extra_blocked_k,
            indices=extra_indices_in_kvcache,
            lens=extra_topk_length,
            sm_scale=scale,
            d_v=d_v,
            scratch=extra_scratch,
            kv_fp8_buf=extra_kv_fp8_buf,
        )
        lse_total = torch.logsumexp(
            torch.stack([lse_swa, lse_ext], dim=0), dim=0)
        w_swa = (lse_swa - lse_total).exp().unsqueeze(-1)
        w_ext = (lse_ext - lse_total).exp().unsqueeze(-1)
        out = w_swa * out_swa + w_ext * out_ext
        lse = lse_total
    else:
        out = out_swa
        lse = lse_swa

    if attn_sink is not None:
        sink = attn_sink.view(1, 1, h_q).to(lse.dtype)
        correction = 1.0 / (1.0 + (sink - lse).exp())
        out = out * correction.unsqueeze(-1).to(out.dtype)

    lonely = lse == float("-inf")
    if lonely.any():
        out = out.masked_fill(lonely.unsqueeze(-1), 0.0)

    return out.view(b, s_q, h_q, d_v).squeeze(1).to(torch.bfloat16)


def _aiter_decode_one_scope(
    *,
    q: torch.Tensor,
    blocked_k: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor | None,
    sm_scale: float,
    d_v: int,
    scratch: AiterSparseScratch,
    kv_fp8_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-scope AITER mla_decode_fwd call.

    Writes into `scratch`'s preallocated buffers in-place so the entire
    decode call is cudagraph-capture friendly.

    Args:
        q, blocked_k, indices, lens, sm_scale, d_v, scratch: see
            `aiter_sparse_attn_decode`.
        kv_fp8_buf: optional preallocated fp8 buffer same shape as `blocked_k`.
            When provided, the bf16->fp8 cast writes into this buffer in place
            instead of allocating a fresh one. Used by the caller to share one
            workspace-backed buffer across all DSv4 layers.

    Returns (output, lse) where:
      output: (total_q, h_q, d_v) bf16 — alias of `scratch.out_buf`
      lse:    (total_q, h_q) fp32 — allocated by AITER kernel
    """
    import aiter

    b, s_q, h_q, d_qk = q.shape
    device = q.device
    total_q = b * s_q

    indices_2d = indices.reshape(total_q, -1).contiguous()
    topk_max = indices_2d.size(-1)

    fp8_dtype = torch.float8_e4m3fn
    if not scratch.matches(
        total_q, h_q, topk_max, d_qk, d_v, fp8_dtype, fp8_dtype
    ):
        scratch.rebuild(
            total_q=total_q,
            nhead=h_q,
            nhead_kv=1,
            topk=topk_max,
            d_qk=d_qk,
            d_v=d_v,
            page_size=1,
            dtype=fp8_dtype,
            kvtype=fp8_dtype,
            device=device,
        )

    # ---- Build valid_mask + valid_lens directly into scratch ----------- #
    if lens is not None:
        if lens.numel() == b and s_q > 1:
            lens_per_tok = lens.repeat_interleave(s_q)
        else:
            lens_per_tok = lens.reshape(-1)
        # valid_mask[i, j] = (j < lens_per_tok[i]) AND (indices_2d[i, j] >= 0)
        torch.lt(
            scratch.col_arange.unsqueeze(0),
            lens_per_tok.unsqueeze(1),
            out=scratch.valid_mask,
        )
        scratch.valid_mask &= indices_2d >= 0
    else:
        torch.ge(indices_2d, 0, out=scratch.valid_mask)

    torch.sum(
        scratch.valid_mask, dim=-1, dtype=torch.int32, out=scratch.valid_lens
    )

    # ---- Compute kv_indptr in place (cumsum(min(valid_lens, topk))) ---- #
    scratch.kv_indptr[0] = 0
    torch.cumsum(
        scratch.valid_lens.clamp(max=topk_max),
        dim=0,
        out=scratch.kv_indptr[1:],
    )

    # ---- Fill kv_indices_2d in-place: keep valid, sentinel -1 elsewhere - #
    scratch.kv_indices_2d.copy_(indices_2d)
    scratch.kv_indices_2d.masked_fill_(~scratch.valid_mask, -1)

    # ---- Cast q to FP8 in-place into the preallocated buffer ----------- #
    scratch.q_fp8.copy_(q.reshape(total_q, h_q, d_qk))

    # ---- Cast blocked_k to FP8 ----------------------------------------- #
    # When the caller provides a preallocated `kv_fp8_buf` (typically backed
    # by `current_workspace_manager()` so it is shared across all 61 DSv4
    # layers), copy in place; otherwise allocate fresh. Either way the
    # AITER kernel sees a stable pointer for the duration of one call.
    if kv_fp8_buf is not None:
        assert kv_fp8_buf.shape == blocked_k.shape, (
            f"kv_fp8_buf shape {tuple(kv_fp8_buf.shape)} must match blocked_k "
            f"shape {tuple(blocked_k.shape)}"
        )
        assert kv_fp8_buf.dtype == fp8_dtype, (
            f"kv_fp8_buf dtype {kv_fp8_buf.dtype} must be {fp8_dtype}"
        )
        kv_fp8_buf.copy_(blocked_k)
        kv_view = kv_fp8_buf.view(-1, 1, 1, d_qk)
    else:
        kv_fp8 = blocked_k.to(fp8_dtype)
        kv_view = kv_fp8.view(-1, 1, 1, d_qk)

    # ---- Refresh AITER work plan against the current kv_indptr --------- #
    # The persistent ASM kernel encodes per-batch lengths into work_*; if we
    # leave that stale, the kernel reads out of bounds. Rewrite into the same
    # buffers in place so pointers stay stable for cudagraph capture.
    scratch.refresh_metadata()

    # ---- Persistent-mode FP8 mla_decode_fwd ---------------------------- #
    _, lse = aiter.mla.mla_decode_fwd(
        scratch.q_fp8,
        kv_view,
        scratch.out_buf,
        scratch.qo_indptr,
        scratch.kv_indptr,
        scratch.kv_indices_2d.view(-1),
        scratch.kv_last_page_lens,
        1,
        1,
        1,
        sm_scale,
        num_kv_splits=256,
        q_scale=scratch.q_scale,
        kv_scale=scratch.kv_scale,
        work_meta_data=scratch.work_meta_data,
        work_indptr=scratch.work_indptr,
        work_info_set=scratch.work_info_set,
        reduce_indptr=scratch.reduce_indptr,
        reduce_final_map=scratch.reduce_final_map,
        reduce_partial_map=scratch.reduce_partial_map,
        return_lse=True,
    )

    if lse is None:
        raise RuntimeError("aiter.mla.mla_decode_fwd returned no LSE")

    return scratch.out_buf, lse
