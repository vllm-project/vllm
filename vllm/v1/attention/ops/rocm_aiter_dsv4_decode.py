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
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class AiterSparseScratch:
    """Cached per-step AITER persistent-mode scratch buffers.

    Allocate once per metadata-build, reuse across all 61 DSv4 attn layers
    in the same decode step. Keyed by (batch_size, nhead, topk, dtype, kvtype).
    """

    __slots__ = (
        "work_meta_data", "work_indptr", "work_info_set",
        "reduce_indptr", "reduce_final_map", "reduce_partial_map",
        "_key",
    )

    def __init__(self) -> None:
        self.work_meta_data: torch.Tensor | None = None
        self.work_indptr: torch.Tensor | None = None
        self.work_info_set: torch.Tensor | None = None
        self.reduce_indptr: torch.Tensor | None = None
        self.reduce_final_map: torch.Tensor | None = None
        self.reduce_partial_map: torch.Tensor | None = None
        self._key: tuple = ()

    def matches(
        self,
        batch_size: int,
        nhead: int,
        topk: int,
        dtype: torch.dtype,
        kvtype: torch.dtype,
    ) -> bool:
        return self._key == (batch_size, nhead, topk, dtype, kvtype)

    def rebuild(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        nhead: int,
        nhead_kv: int,
        page_size: int,
        topk: int,
        dtype: torch.dtype,
        kvtype: torch.dtype,
        max_split_per_batch: int = 256,
    ) -> None:
        import aiter

        device = qo_indptr.device
        bs = qo_indptr.shape[0] - 1
        (
            (wmd_size, wmd_type),
            (wi_size, wi_type),
            (wis_size, wis_type),
            (ri_size, ri_type),
            (rfm_size, rfm_type),
            (rpm_size, rpm_type),
        ) = aiter.get_mla_metadata_info_v1(
            bs, 1, nhead, dtype, kvtype,
            is_sparse=True, fast_mode=True,
            num_kv_splits=max_split_per_batch,
        )
        self.work_meta_data = torch.empty(
            wmd_size, dtype=wmd_type, device=device)
        self.work_indptr = torch.empty(
            wi_size, dtype=wi_type, device=device)
        self.work_info_set = torch.empty(
            wis_size, dtype=wis_type, device=device)
        self.reduce_indptr = torch.empty(
            ri_size, dtype=ri_type, device=device)
        self.reduce_final_map = torch.empty(
            rfm_size, dtype=rfm_type, device=device)
        self.reduce_partial_map = torch.empty(
            rpm_size, dtype=rpm_type, device=device)

        aiter.get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_lens,
            nhead // nhead_kv, nhead_kv, True,
            self.work_meta_data, self.work_info_set, self.work_indptr,
            self.reduce_indptr, self.reduce_final_map,
            self.reduce_partial_map,
            page_size=page_size,
            kv_granularity=max(page_size, 16),
            max_seqlen_qo=1, uni_seqlen_qo=1,
            fast_mode=True,
            max_split_per_batch=max_split_per_batch,
            topk=topk,
            dtype_q=dtype, dtype_kv=kvtype,
        )
        self._key = (bs, nhead, topk, dtype, kvtype)


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

    Returns: (b, h_q, d_v) bf16
    """
    b, s_q, h_q, d_qk = q.shape
    d_v = head_dim
    device = q.device

    if scratch is None:
        scratch = AiterSparseScratch()
    if extra_scratch is None:
        extra_scratch = AiterSparseScratch()

    out_swa, lse_swa = _aiter_decode_one_scope(
        q=q, blocked_k=blocked_k,
        indices=indices_in_kvcache, lens=topk_length,
        sm_scale=scale, d_v=d_v, scratch=scratch,
    )

    if extra_blocked_k is not None:
        assert extra_indices_in_kvcache is not None
        out_ext, lse_ext = _aiter_decode_one_scope(
            q=q, blocked_k=extra_blocked_k,
            indices=extra_indices_in_kvcache, lens=extra_topk_length,
            sm_scale=scale, d_v=d_v, scratch=extra_scratch,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-scope AITER mla_decode_fwd call.

    Returns (output, lse) where:
      output: (total_q, h_q, d_v) bf16
      lse:    (total_q, h_q) fp32
    """
    import aiter

    b, s_q, h_q, d_qk = q.shape
    device = q.device
    total_q = b * s_q

    q_flat = q.reshape(total_q, h_q, d_qk).contiguous()

    indices_2d = indices.reshape(total_q, -1).contiguous()
    topk_max = indices_2d.size(-1)

    if lens is not None:
        if lens.numel() == b and s_q > 1:
            lens_per_tok = lens.repeat_interleave(s_q)
        else:
            lens_per_tok = lens.reshape(-1)
        col = torch.arange(topk_max, device=device).unsqueeze(0)
        valid_mask = (col < lens_per_tok.unsqueeze(1)) & (indices_2d >= 0)
    else:
        valid_mask = indices_2d >= 0

    valid_lens = valid_mask.sum(dim=-1).to(torch.int32)

    # Fixed-stride layout: keep per-row topk_max slots, invalid → -1
    indices_fs = torch.where(
        valid_mask, indices_2d, torch.full_like(indices_2d, -1)
    ).to(torch.int32).contiguous()
    kv_indices = indices_fs.view(-1)

    qo_indptr = torch.arange(total_q + 1, dtype=torch.int32, device=device)
    seq_lens_kv = valid_lens.clamp(max=topk_max)
    kv_indptr = torch.zeros(total_q + 1, dtype=torch.int32, device=device)
    torch.cumsum(seq_lens_kv, dim=0, out=kv_indptr[1:])
    kv_last_page_lens = torch.ones(total_q, dtype=torch.int32, device=device)

    # Cast to FP8 for the persistent ASM kernel (required for LSE on gfx950)
    fp8_dtype = torch.float8_e4m3fn
    q_fp8 = q_flat.to(fp8_dtype)
    kv_fp8 = blocked_k.to(fp8_dtype)

    if not scratch.matches(total_q, h_q, topk_max, fp8_dtype, fp8_dtype):
        scratch.rebuild(
            qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            kv_last_page_lens=kv_last_page_lens,
            nhead=h_q, nhead_kv=1, page_size=1,
            topk=topk_max, dtype=fp8_dtype, kvtype=fp8_dtype,
        )

    out_buf = torch.empty(
        (total_q, h_q, d_v), dtype=torch.bfloat16, device=device)

    kv_view = kv_fp8.view(-1, 1, 1, d_qk)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)
    kv_scale = torch.ones(1, dtype=torch.float32, device=device)

    _, lse = aiter.mla.mla_decode_fwd(
        q_fp8, kv_view, out_buf,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
        1, 1, 1, sm_scale,
        num_kv_splits=256,
        q_scale=q_scale, kv_scale=kv_scale,
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

    return out_buf, lse
