# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The levers on vLLM's ROCm MLA metadata builders.

``VLLM_K31_BEAT_DRAFTMETA``: ``AiterMLAMetadataBuilder._build_decode`` plans a
single uniform-qo request's persistent schedule with the closed-form
``mla_bs1_meta`` kernel instead of AITER's serial ``get_mla_metadata_v1``
(the bs1 draft's ~70 us at long context). AITER's planner is wrapped for the
duration of ``_build_decode`` only, and the wrapper takes just the call it
reproduces: the plain (no DCP round-robin, no asm DCP verify) schedule of one
request into the builder's own buffers, checked against AITER once per shape.
Only AITER MLA groups reach it: on Kimi-K3.1 that is the preview's dense MLA
draft. The release's GQA sliding-window draft decodes with
KIMI_K3_FLYDSL_SWA_GQA (or TRITON_ATTN), which plans no persistent schedule,
so the planner never runs for it (``levers.dspark.log_draft_attn`` says which
at startup).

``VLLM_K31_BEAT_EAGERPREP``: ``ROCMAiterMLASparseMetadataBuilder.build`` writes
the sparse paged-KV indptr (sparse seqlen, cumsum, tail fill) in one launch."""

import functools
import threading

import numpy as np
import torch

from vllm.models.kimi_k3.amd.specialized_decode.levers import (
    DRAFTMETA,
    EAGERPREP,
    lever_on,
    wrap,
)
from vllm.triton_utils import tl, triton

# The builder whose ``_build_decode`` runs on this thread.
_local = threading.local()

# get_mla_metadata_v1's keywords on the plain decode schedule.
_PLAIN_KWARGS = frozenset(
    (
        "page_size",
        "kv_granularity",
        "max_seqlen_qo",
        "uni_seqlen_qo",
        "fast_mode",
        "dtype_q",
        "dtype_kv",
    )
)


def install() -> None:
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseMetadataBuilder,
    )

    try:
        import aiter
    except ImportError:
        aiter = None
    if aiter is not None and hasattr(aiter, "get_mla_metadata_v1"):
        wrap(aiter, "get_mla_metadata_v1", _wrap_planner)
        wrap(AiterMLAMetadataBuilder, "_build_decode", _wrap_build_decode)
    wrap(ROCMAiterMLASparseMetadataBuilder, "build", _wrap_sparse_build)


def _wrap_build_decode(build_decode):
    @functools.wraps(build_decode)
    def _build_decode(self, *args, **kwargs):
        if not lever_on(DRAFTMETA) or self._asm_dcp_verify:
            return build_decode(self, *args, **kwargs)
        prev = getattr(_local, "builder", None)
        _local.builder = self
        try:
            return build_decode(self, *args, **kwargs)
        finally:
            _local.builder = prev

    return _build_decode


def _wrap_planner(aiter_planner):
    def get_mla_metadata_v1(*args, **kwargs):
        builder = getattr(_local, "builder", None)
        if builder is not None and use_bs1_planner(builder, args, kwargs):
            bs1_planner(builder).plan(args[1], int(kwargs["max_seqlen_qo"]))
            return None
        return aiter_planner(*args, **kwargs)

    # AITER's own, for ``AiterMlaBs1Planner.verify``.
    get_mla_metadata_v1.aiter_planner = aiter_planner  # type: ignore[attr-defined]
    return get_mla_metadata_v1


def bs1_planner(builder):
    """The builder's planner, bound to its persistent buffers."""
    planner = getattr(builder, "_sd_bs1_planner", None)
    if planner is None:
        from vllm.models.kimi_k3.amd.specialized_decode.levers.mla_bs1_meta import (
            AiterMlaBs1Planner,
        )

        planner = AiterMlaBs1Planner(
            builder._mla_work_meta_data,
            builder._mla_work_indptr,
            builder._mla_work_info_set,
            builder._mla_reduce_indptr,
            builder._mla_reduce_final_map,
            builder._mla_reduce_partial_map,
        )
        builder._sd_bs1_planner = planner
    return planner


def use_bs1_planner(builder, args: tuple, kwargs: dict) -> bool:
    """Whether this ``get_mla_metadata_v1`` call is the single uniform-qo request
    the closed-form planner reproduces bit-exactly (verified, or verifiable now:
    not under graph capture)."""
    if len(args) != 12 or kwargs.keys() != _PLAIN_KWARGS:
        # DCP round-robin or the asm DCP verify's split cap
        return False
    qo_indptr, kv_indptr, _, num_heads, num_kv_heads, is_causal, *buffers = args
    own = (
        builder._mla_work_meta_data,
        builder._mla_work_info_set,
        builder._mla_work_indptr,
        builder._mla_reduce_indptr,
        builder._mla_reduce_final_map,
        builder._mla_reduce_partial_map,
    )
    if (
        num_kv_heads != 1
        or not isinstance(is_causal, bool)
        or kwargs["page_size"] != 1
        or kwargs["kv_granularity"] != 16
        or kwargs["fast_mode"] is not True
        or qo_indptr.numel() != 2
        or kv_indptr.numel() != 2
        or any(a is not b for a, b in zip(buffers, own))
    ):
        return False
    qlen = int(kwargs["max_seqlen_qo"])
    if qlen < 2 or int(kwargs["uni_seqlen_qo"]) != qlen:
        return False
    planner = bs1_planner(builder)
    key = (qlen, num_heads, is_causal, kwargs["dtype_q"], kwargs["dtype_kv"])
    if planner.is_verified(*key):
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    return planner.verify(*key)


@triton.jit(do_not_specialize=["num_reqs", "num_tokens", "indptr_len"])
def _sparse_indptr_fused_kernel(
    seq_len_ptr,
    cu_query_lens_ptr,
    indptr_ptr,
    num_reqs,
    num_tokens,
    indptr_len,
    topk_token,
    BLOCK_T: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    """generate_sparse_seqlen + cumsum into indptr[1:num_tokens+1] + tail fill
    with indptr[num_tokens], one launch. Every program recomputes the (tiny)
    prefix sum; program 0 writes the head, every program writes one BLOCK_TAIL
    chunk of the tail."""
    pid = tl.program_id(0)
    tok = tl.arange(0, BLOCK_T)
    tmask = tok < num_tokens
    # Request owning each token (tokens are all inside query_start_loc).
    lo = tl.zeros([BLOCK_T], dtype=tl.int32)
    hi = tl.zeros([BLOCK_T], dtype=tl.int32) + num_reqs
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(
            cu_query_lens_ptr + mid + 1, mask=mid < num_reqs, other=0x7FFFFFFF
        )
        right = (lo < hi) & (end <= tok)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where((lo < hi) & ~right, mid, hi)
    req = tl.minimum(lo, num_reqs - 1)
    qs = tl.load(cu_query_lens_ptr + req, mask=tmask, other=0)
    qe = tl.load(cu_query_lens_ptr + req + 1, mask=tmask, other=0)
    seq_len = tl.load(seq_len_ptr + req, mask=tmask, other=0)
    sparse = seq_len - (qe - qs) + (tok - qs)
    val = tl.where(sparse + 1 < topk_token, sparse + 1, topk_token)
    val = tl.where(tmask & (seq_len != 0), val, 0)
    csum = tl.cumsum(val, 0)
    total = tl.sum(val, 0)
    if pid == 0:
        tl.store(indptr_ptr + 1 + tok, csum, mask=tmask)
    tail = num_tokens + 1 + pid * BLOCK_TAIL + tl.arange(0, BLOCK_TAIL)
    tl.store(
        indptr_ptr + tail,
        tl.zeros([BLOCK_TAIL], dtype=tl.int32) + total,
        mask=tail < indptr_len,
    )


_SPARSE_INDPTR_BLOCK_T = 256
_SPARSE_INDPTR_BLOCK_TAIL = 2048


def sparse_indptr_fused(
    seq_lens: torch.Tensor,
    cu_query_lens: torch.Tensor,
    indptr: torch.Tensor,
    topk_token: int,
    num_tokens: int,
) -> bool:
    """Exact one-launch replacement of generate_sparse_seqlen_triton + cumsum
    + tail copy. Returns False (nothing launched) when it does not apply."""
    num_reqs = cu_query_lens.size(0) - 1
    if (
        num_tokens <= 0
        or num_reqs <= 0
        or num_tokens > _SPARSE_INDPTR_BLOCK_T
        or indptr.dtype != torch.int32
        or not indptr.is_contiguous()
        or seq_lens.dtype != torch.int32
        or cu_query_lens.dtype != torch.int32
    ):
        return False
    indptr_len = indptr.numel()
    tail_len = max(indptr_len - num_tokens - 1, 0)
    grid = (max(1, triton.cdiv(tail_len, _SPARSE_INDPTR_BLOCK_TAIL)),)
    _sparse_indptr_fused_kernel[grid](
        seq_lens,
        cu_query_lens,
        indptr,
        num_reqs,
        num_tokens,
        indptr_len,
        topk_token,
        BLOCK_T=_SPARSE_INDPTR_BLOCK_T,
        BLOCK_TAIL=_SPARSE_INDPTR_BLOCK_TAIL,
        num_warps=4,
    )
    return True


def _wrap_sparse_build(build):
    @functools.wraps(build)
    def sparse_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # DCP attention builds its own CSR indices: no paged indptr to write.
        if (
            not lever_on(EAGERPREP)
            or self.vllm_config.parallel_config.decode_context_parallel_size > 1
        ):
            return build(
                self, common_prefix_len, common_attn_metadata, fast_build=fast_build
            )
        return _sparse_build_eagerprep(
            self, common_prefix_len, common_attn_metadata, fast_build
        )

    return sparse_build


def _sparse_build_eagerprep(self, common_prefix_len, common_attn_metadata, fast_build):
    """``ROCMAiterMLASparseMetadataBuilder.build`` without DCP, its paged indptr
    from ``sparse_indptr_fused`` when it applies."""
    from vllm.v1.attention.backends.mla import rocm_aiter_mla_sparse as sparse

    metadata = super(sparse.ROCMAiterMLASparseMetadataBuilder, self).build(
        common_prefix_len, common_attn_metadata, fast_build=fast_build
    )
    use_dense_prefill = metadata.prefill is not None and metadata.prefill.use_dense_mha
    num_reqs = metadata.num_decodes if use_dense_prefill else metadata.num_reqs
    num_tokens = int(common_attn_metadata.query_start_loc_cpu[num_reqs].item())
    num_decodes = metadata.num_decodes
    num_prefills = metadata.num_prefills
    num_decode_tokens = metadata.num_decode_tokens
    starts = np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
    seg_lengths = np.diff(starts[: num_reqs + 1])
    req_id_per_token = np.repeat(
        np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
    )
    # Only re-zero the shrink-tail (as the stock build).
    new_req_extent = int(req_id_per_token.shape[0])
    new_indices_extent = num_tokens * self.topk_tokens
    if self._prev_req_extent > new_req_extent:
        self.req_id_per_token_buffer[new_req_extent : self._prev_req_extent].fill_(0)
    if self._prev_indices_extent > new_indices_extent:
        self.paged_kv_indices[new_indices_extent : self._prev_indices_extent].fill_(0)
    self._prev_req_extent = new_req_extent
    self._prev_indices_extent = new_indices_extent
    self.req_id_per_token_buffer[:new_req_extent].copy_(
        sparse.np_to_pinned_tensor(req_id_per_token), non_blocking=True
    )
    query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
    seq_lens = common_attn_metadata.seq_lens[:num_reqs]
    if not sparse_indptr_fused(
        seq_lens, query_start_loc, self.paged_kv_indptr, self.topk_tokens, num_tokens
    ):
        sparse_seqlen = sparse.generate_sparse_seqlen_triton(
            seq_lens,
            query_start_loc,
            self.topk_tokens,
            num_tokens,
            common_attn_metadata.max_query_len,
        )
        torch.cumsum(sparse_seqlen, dim=0, out=self.paged_kv_indptr[1 : num_tokens + 1])
        # fill_() with a value that aliases the destination clones it first.
        indptr_tail = self.paged_kv_indptr[num_tokens + 1 :]
        indptr_tail.copy_(self.paged_kv_indptr[num_tokens].expand_as(indptr_tail))

    req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
    qo_indptr = self.qo_indptr[: num_tokens + 1]
    paged_kv_last_page_len = self.paged_kv_last_page_len[:num_tokens]
    paged_kv_indptr = self.paged_kv_indptr[: num_tokens + 1]
    paged_kv_indices = self.paged_kv_indices[: num_tokens * self.topk_tokens]

    # Persistent MLA metadata, as the stock build (fingerprinted, see there).
    head_size = self.mla_dims.kv_lora_rank + self.mla_dims.qk_rope_head_dim
    use_triton_sparse = sparse._use_rocm_sparse_triton(
        kv_cache_dtype=self.kv_cache_dtype,
        head_size=head_size,
        kv_lora_rank=self.mla_dims.kv_lora_rank,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        max_query_len=common_attn_metadata.max_query_len,
    )
    work_meta_data = None
    work_indptr = None
    work_info_set = None
    reduce_indptr = None
    reduce_final_map = None
    reduce_partial_map = None
    is_long_pure_bf16_prefill = sparse._is_long_bf16_prefill(
        self._mla_kv_dtype, metadata
    )
    if (
        self._use_persistent_metadata
        and not use_triton_sparse
        and not is_long_pure_bf16_prefill
        and num_tokens > 0
    ):
        with sparse.gpu_sync_allowed():
            seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].cpu().numpy()
        clamped_seq_lens = np.minimum(seq_lens_cpu, self.topk_tokens)
        clamped_context_lens = np.minimum(seq_lens_cpu - seg_lengths, self.topk_tokens)
        metadata_key = (
            num_tokens,
            int(common_attn_metadata.max_query_len),
            self._num_attention_heads,
            clamped_seq_lens.tobytes(),
            clamped_context_lens.tobytes(),
            seg_lengths.tobytes(),
        )
        if metadata_key != self._prev_metadata_key:
            from aiter import get_mla_metadata_v1

            max_split_per_batch = self._sparse_decode_max_split(
                int(common_attn_metadata.max_seq_len)
            )
            get_mla_metadata_v1(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                self._num_attention_heads,
                1,
                True,
                self._mla_work_meta_data,
                self._mla_work_info_set,
                self._mla_work_indptr,
                self._mla_reduce_indptr,
                self._mla_reduce_final_map,
                self._mla_reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=1,
                uni_seqlen_qo=1,
                fast_mode=True,
                max_split_per_batch=max_split_per_batch,
                dtype_q=self._mla_q_dtype,
                dtype_kv=self._mla_kv_dtype,
            )
            torch.cuda.current_stream(self.device).synchronize()
            self._prev_metadata_key = metadata_key
        work_meta_data = self._mla_work_meta_data
        work_indptr = self._mla_work_indptr
        work_info_set = self._mla_work_info_set
        reduce_indptr = self._mla_reduce_indptr
        reduce_final_map = self._mla_reduce_final_map
        reduce_partial_map = self._mla_reduce_partial_map

    metadata.attn_out_dtype = self.model_dtype
    metadata.req_id_per_token = req_id_per_token
    metadata.qo_indptr = qo_indptr
    metadata.paged_kv_last_page_len = paged_kv_last_page_len
    metadata.paged_kv_indices = paged_kv_indices
    metadata.paged_kv_indptr = paged_kv_indptr
    metadata.work_meta_data = work_meta_data
    metadata.work_indptr = work_indptr
    metadata.work_info_set = work_info_set
    metadata.reduce_indptr = reduce_indptr
    metadata.reduce_final_map = reduce_final_map
    metadata.reduce_partial_map = reduce_partial_map
    return metadata
