# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward interface for MiniMax M3 CuteDSL sparse prefill attention."""

import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from .common.cute_dsl_utils import to_cute_tensor as to_cute_tensor_kvouter
from .common.tma_utils import create_q_gather4_tma_desc, view_paged_kv_as_blocks
from .sm100.atten_fwd import SparseAttentionForwardSm100
from .sm100.combine import combine
from .sm100.prepare_k2q_csr import SparseAttentionSchedule

_compile_cache: dict = {}


def _to_cute_int32_metadata(t: torch.Tensor):
    return to_cute_tensor_kvouter(t, assumed_align=4)


def sparse_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    blk_kv: int = 128,
    causal: bool,
    softmax_scale: float,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    schedule: SparseAttentionSchedule,
    out: torch.Tensor,
):
    """Run SM100 sparse attention forward with prebuilt CSR schedule metadata."""
    head_kv = k.shape[-2]
    _sparse_atten_forward(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        topK,
        blk_kv,
        bool(causal),
        float(softmax_scale),
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        seqused_k,
        schedule,
        out,
        head_kv,
        max_seqlen_q,
    )


def _sparse_atten_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    topK: int,
    blk_kv: int,
    causal: bool,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    schedule: SparseAttentionSchedule,
    out: torch.Tensor,
    head_kv: int,
    max_seqlen_q: int,
):
    total_q, head_q, dim = q.shape
    max_num_kv_blocks = k2q_row_ptr.shape[1] - 1
    O_partial = torch.empty(
        topK, total_q, head_q, dim, dtype=torch.bfloat16, device=q.device
    )
    LSE_partial = torch.empty(
        topK, total_q, head_q, dtype=torch.float32, device=q.device
    )
    LSE_temperature_partial = None
    LSE_out = torch.empty(total_q, head_q, dtype=torch.float32, device=q.device)
    LSE_temperature_out = None
    k2q_qsplit_indices = schedule.qsplit_indices
    split_counts = schedule.split_counts
    _call_sparse_forward_sm100_csr_varlen(
        q,
        k,
        v,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_qsplit_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        seqused_k,
        O_partial,
        LSE_partial,
        LSE_temperature_partial,
        softmax_scale,
        max_num_kv_blocks,
        blk_kv,
        head_kv,
        max_seqlen_q,
        causal=causal,
        schedule=schedule,
    )
    combine(
        O_partial,
        LSE_partial,
        out,
        LSE_out,
        lse_temperature_partial=LSE_temperature_partial,
        lse_temperature_out=LSE_temperature_out,
        cu_seqlens=cu_seqlens_q,
        split_counts=split_counts,
        use_pdl=True,
    )


def _call_sparse_forward_sm100_csr_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k2q_row_ptr: torch.Tensor,
    k2q_q_indices: torch.Tensor,
    k2q_qsplit_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    O_partial: torch.Tensor,
    LSE_partial: torch.Tensor,
    LSE_temperature_partial: None,
    softmax_scale: float,
    max_num_kv_blocks: int,
    blk_kv: int,
    head_kv: int,
    max_seqlen_q: int,
    *,
    causal: bool = False,
    schedule: SparseAttentionSchedule,
) -> None:
    """Compile and launch the SM100 sparse forward K1 kernel on CSR metadata."""
    head_dim = q.shape[-1]
    dtype = q.dtype
    partial_dtype = O_partial.dtype
    head_q = q.shape[1]
    n_block_size = blk_kv
    qhead_per_kv = head_q // head_kv
    k_kernel = view_paged_kv_as_blocks(k, blk_kv=n_block_size)
    v_kernel = view_paged_kv_as_blocks(v, blk_kv=n_block_size)
    O_partial_flat = O_partial.view(-1, head_dim)
    Q_flat = q.view(-1, head_dim)
    Q_gather4_desc = (
        create_q_gather4_tma_desc(Q_flat) if qhead_per_kv in (1, 2, 4) else None
    )
    K_raw_desc = None
    V_raw_desc = None
    lse_temperature_inv_scale = 1.0

    scheduler_metadata = schedule.scheduler_metadata
    work_count = schedule.work_count
    work_capacity = schedule.work_capacity
    key = (
        "sparse_forward_sm100_csr_varlen",
        head_dim,
        n_block_size,
        qhead_per_kv,
        dtype,
        partial_dtype,
        bool(causal),
    )
    if key not in _compile_cache:
        kernel = SparseAttentionForwardSm100(
            head_dim=head_dim,
            qheadperkv=qhead_per_kv,
            n_block_size=n_block_size,
            paged_kv=True,
            page_size=n_block_size,
            has_seqused_k=True,
            causal=bool(causal),
        )
        _compile_cache[key] = cute.compile(
            kernel,
            to_cute_tensor_kvouter(k_kernel),
            to_cute_tensor_kvouter(v_kernel),
            to_cute_tensor_kvouter(k2q_q_indices),
            to_cute_tensor_kvouter(k2q_qsplit_indices),
            to_cute_tensor_kvouter(k2q_row_ptr),
            to_cute_tensor_kvouter(scheduler_metadata),
            to_cute_tensor_kvouter(work_count),
            to_cute_tensor_kvouter(O_partial_flat),
            to_cute_tensor_kvouter(LSE_partial),
            LSE_temperature_partial,
            to_cute_tensor_kvouter(Q_flat),
            None if Q_gather4_desc is None else to_cute_tensor_kvouter(Q_gather4_desc),
            K_raw_desc,
            V_raw_desc,
            _to_cute_int32_metadata(page_table),
            _to_cute_int32_metadata(seqused_k),
            _to_cute_int32_metadata(cu_seqlens_q),
            _to_cute_int32_metadata(cu_seqlens_k),
            Float32(softmax_scale),
            Float32(lse_temperature_inv_scale),
            Int32(max_num_kv_blocks),
            Int32(head_kv),
            Int32(max_seqlen_q),
            Int32(work_capacity),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    with torch.cuda.nvtx.range("Fwd_SparseAttn_Sm100_CsrVarlen"):
        _compile_cache[key](
            k_kernel,
            v_kernel,
            k2q_q_indices,
            k2q_qsplit_indices,
            k2q_row_ptr,
            scheduler_metadata,
            work_count,
            O_partial_flat,
            LSE_partial,
            LSE_temperature_partial,
            Q_flat,
            Q_gather4_desc,
            K_raw_desc,
            V_raw_desc,
            page_table,
            seqused_k,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            lse_temperature_inv_scale,
            max_num_kv_blocks,
            head_kv,
            max_seqlen_q,
            work_capacity,
        )
