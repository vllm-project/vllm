# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse k2q CSR op wrapper for SM100."""

from dataclasses import dataclass

import torch

from vllm.utils.math_utils import cdiv, round_up


@dataclass
class SparseAttentionSchedule:
    scheduler_metadata: torch.Tensor
    work_count: torch.Tensor
    qsplit_indices: torch.Tensor
    split_counts: torch.Tensor
    target_q_per_cta: int

    @property
    def work_capacity(self) -> int:
        return self.scheduler_metadata.shape[0]


def _target_q_per_cta(
    *,
    total_q: int,
    topk: int,
    head_kv: int,
    qhead_per_kv: int,
    device: torch.device,
) -> int:
    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    q_tokens_per_group = 128 // qhead_per_kv
    total_refs_upper = total_q * topk * head_kv
    desired_work_items = max(num_sm * 2, 1)
    total_groups_upper = cdiv(max(total_refs_upper, 1), q_tokens_per_group)
    target_groups_per_cta = min(
        512,
        max(1, cdiv(total_groups_upper, desired_work_items)),
    )
    return target_groups_per_cta * q_tokens_per_group


def _balanced_target_q_per_cta(
    *,
    total_q: int,
    topk: int,
    blk_kv: int,
    head_kv: int,
    qhead_per_kv: int,
    device: torch.device,
) -> int:
    q_tokens_per_group = 128 // qhead_per_kv
    occupancy_target = _target_q_per_cta(
        total_q=total_q,
        topk=topk,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        device=device,
    )
    sink_balance_cap = max(q_tokens_per_group, topk * blk_kv * 2)
    target = min(max(occupancy_target, q_tokens_per_group), sink_balance_cap)
    return round_up(target, q_tokens_per_group)


def _flat_schedule_capacity(
    *,
    total_rows: int,
    total_q: int,
    topk: int,
    head_kv: int,
    target_q_per_cta: int,
) -> int:
    row_upper = max(total_rows, 0) * max(head_kv, 1)
    refs_upper = max(total_q, 0) * max(topk, 1) * max(head_kv, 1)
    split_upper = cdiv(max(refs_upper, 1), max(target_q_per_cta, 1))
    return max(1, row_upper + split_upper)


def build_k2q_csr_with_schedule_sm100(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    blk_kv: int = 128,
    max_seqlen_k: int,
    max_seqlen_q: int,
    total_rows: int,
    qhead_per_kv: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, SparseAttentionSchedule]:
    """Allocate CSR/schedule buffers and dispatch to the SM100 AOT C++ op."""
    head_kv, total_q, topk = q2k_indices.shape
    batch = cu_seqlens_q.shape[0] - 1
    max_kv_blocks = cdiv(max(max_seqlen_k, blk_kv), blk_kv)
    nnz_upper_bound = total_q * topk
    device = q2k_indices.device
    k2q_row_ptr = torch.empty(
        (head_kv, total_rows + 1),
        dtype=torch.int32,
        device=device,
    )
    k2q_q_indices = torch.empty(
        (head_kv, nnz_upper_bound),
        dtype=torch.int32,
        device=device,
    )
    target_q_per_cta = _balanced_target_q_per_cta(
        total_q=total_q,
        topk=topk,
        blk_kv=blk_kv,
        head_kv=head_kv,
        qhead_per_kv=qhead_per_kv,
        device=device,
    )
    scheduler_metadata_capacity = _flat_schedule_capacity(
        total_rows=total_rows,
        total_q=total_q,
        topk=topk,
        head_kv=head_kv,
        target_q_per_cta=target_q_per_cta,
    )
    scheduler_metadata = torch.empty(
        (scheduler_metadata_capacity, 6), dtype=torch.int32, device=device
    )
    work_count = torch.empty((1,), dtype=torch.int32, device=device)
    qsplit_indices = torch.empty_like(k2q_q_indices)
    split_counts = torch.empty(
        (batch, max_seqlen_q, head_kv), dtype=torch.int32, device=device
    )
    schedule = SparseAttentionSchedule(
        scheduler_metadata=scheduler_metadata,
        work_count=work_count,
        qsplit_indices=qsplit_indices,
        split_counts=split_counts,
        target_q_per_cta=target_q_per_cta,
    )

    if total_rows == 0 or total_q == 0 or head_kv == 0 or topk == 0:
        k2q_row_ptr.zero_()
        k2q_q_indices.fill_(-1)
        schedule.work_count.zero_()
        schedule.split_counts.zero_()
        return k2q_row_ptr, k2q_q_indices, schedule

    import vllm._C  # noqa: F401

    with torch.cuda.nvtx.range("SparseK2qCsr_Pipeline"):
        torch.ops._C.minimax_m3_build_k2q_csr_with_schedule(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            k2q_row_ptr,
            k2q_q_indices,
            schedule.scheduler_metadata,
            schedule.work_count,
            schedule.qsplit_indices,
            schedule.split_counts,
            topk,
            blk_kv,
            total_rows,
            max_kv_blocks,
            schedule.target_q_per_cta,
            schedule.work_capacity,
            max_seqlen_q,
        )
    return k2q_row_ptr, k2q_q_indices, schedule
