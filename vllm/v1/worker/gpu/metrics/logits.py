# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch._inductor.runtime.triton_helpers import libdevice

from vllm.triton_utils import tl, triton


@triton.jit
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=0
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.store(num_nans_ptr + req_idx, num_nans)


def get_num_nans(logits: torch.Tensor) -> torch.Tensor:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_nans = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _num_nans_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        num_nans,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return num_nans


def get_num_draft_nans(
    draft_logits: torch.Tensor,
    slot_per_row: torch.Tensor,
    step_per_row: torch.Tensor,
    vocab_size: int,
    temp_per_row: torch.Tensor,
) -> torch.Tensor:
    """[num_logits] NaN counts for the draft rows the verifier reads.

    ``draft_logits`` is the persistent [num_rows, num_steps, V] cache: only
    the cells the verifier reads are valid inputs, so the statistics are
    restricted to the verifier's read set. A full scan would report NaNs
    from inactive slots or unscheduled steps as current inputs:

    * active (slot, step) pairs only: row ``i`` reads the cache cell at
      ``[slot_per_row[i], step_per_row[i]]`` -- the same indices the
      verification kernels consume;
    * bonus rows (the last row of each request, whose ``step_per_row`` is
      ``num_steps``) read no draft: their gather index is clamped into
      range and their counts are zeroed, mirroring the verifier's bonus
      early-return;
    * padded vocab columns beyond ``vocab_size`` (the effective
      min(target, draft) width) are excluded, matching the verifier's
      column mask;
    * greedy rows (``temperature == 0``) are excluded: the verifier skips
      draft reads for them and accepts via target argmax.
    """
    step = step_per_row.clamp(max=draft_logits.shape[1] - 1)
    if step_per_row.shape[0] == 0:
        return torch.zeros(0, dtype=torch.int32, device=draft_logits.device)
    active = draft_logits[slot_per_row.long(), step.long()]
    if vocab_size < active.shape[1]:
        active = active[:, :vocab_size]
    counts = torch.isnan(active).sum(dim=-1)
    # A row is a bonus row iff the next row starts a new request (its
    # step resets to 0), mirroring the verifier's _is_bonus_logit check.
    next_step = torch.cat((step_per_row[1:], step_per_row.new_zeros(1)))
    counts = counts * (next_step != 0)
    return (counts * (temp_per_row != 0)).to(torch.int32)


def counts_rows_to_requests(
    counts: torch.Tensor,
    cu_num_logits: torch.Tensor,
) -> torch.Tensor:
    """[num_logits] per-row counts -> [num_reqs] per-request totals.

    ``cu_num_logits`` is the inclusive cumsum of rows per request. Rows of
    one request are contiguous, and requests scheduled zero rows contribute
    zero.
    """
    num_reqs = cu_num_logits.shape[0] - 1
    num_rows = counts.shape[0]
    if num_rows == 0:
        return torch.zeros(num_reqs, dtype=counts.dtype, device=counts.device)
    row_req = torch.searchsorted(
        cu_num_logits[1:].to(torch.int64),
        torch.arange(num_rows, device=counts.device),
        right=True,
    )
    totals = torch.zeros(num_reqs, dtype=counts.dtype, device=counts.device)
    return totals.index_add_(0, row_req.to(torch.int64), counts)
