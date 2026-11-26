# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels of the specialized decode's deferred RecoverSSM: commit-plan
steps that defer (`_defer_commit_kernel`) or flush (`_prepare_flush_plan_kernel`)
a request's pending replay, plus re-export of `kda_recoverssm_verify`."""

from vllm.models.kimi_k3.nvidia.ops.recoverssm import kda_recoverssm_verify
from vllm.triton_utils import tl, triton

__all__ = ["kda_recoverssm_verify"]


@triton.heuristics({"HAS_EAGER": lambda args: args["eager_ptr"] is not None})
@triton.jit
def _defer_commit_kernel(
    state_indices_ptr,
    commit_lens_ptr,
    record_rows_ptr,
    pending_ptr,
    eager_ptr,
    null_block_id,
    stride_state_indices,
    HAS_EAGER: tl.constexpr,
):
    """Deferred RecoverSSM after a plan: request b's commit length becomes the
    pending replay of its record slot (and no commit runs for it), unless the row is
    committed now (`eager`: pending 0, the plan kept). Rows without a state slot
    (padding, dummy batches) or a record slot leave every pending count."""
    spec_idx = tl.program_id(0)
    record_row = tl.load(record_rows_ptr + spec_idx)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices)
    if (record_row < 0) | (source_state_idx <= null_block_id):
        return
    commit_len = tl.load(commit_lens_ptr + spec_idx)
    eager = False
    if HAS_EAGER:
        eager = tl.load(eager_ptr + spec_idx) != 0
    if eager:
        tl.store(pending_ptr + record_row, 0)
    else:
        tl.store(pending_ptr + record_row, commit_len)
        tl.store(commit_lens_ptr + spec_idx, 0)


@triton.jit
def _prepare_flush_plan_kernel(
    state_indices_ptr,
    record_rows_ptr,
    pending_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_indices,
    SPEC_QUERY_LEN: tl.constexpr,
):
    """The commit plan folding each request's pending replay into its checkpoint
    in place (deferred RecoverSSM steps that the stock verify runs)."""
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices)
    record_row = tl.load(record_rows_ptr + spec_idx)
    commit_len = 0
    if record_row >= 0:
        commit_len = tl.load(pending_ptr + record_row)
    commit_len = tl.minimum(tl.maximum(commit_len, 0), SPEC_QUERY_LEN)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(final_state_indices_ptr + spec_idx, source_state_idx)
    tl.store(boundary_state_indices_ptr + spec_idx, null_block_id)
    tl.store(boundary_recovery_lens_ptr + spec_idx, 0)
