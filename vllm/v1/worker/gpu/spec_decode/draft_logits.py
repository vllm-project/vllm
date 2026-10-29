# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.batch_shard import (
    shard_owner,
    shard_owners,
    shard_row,
    shard_rows,
)


@triton.jit
def draft_logits_row(req_state_idx, row_mapping_ptr):
    """Row of a request in this rank's draft logits cache, or -1 if no row.

    ``row_mapping_ptr`` is None for a replicated cache, which is indexed by request
    state. Otherwise it is `DraftLogitsCache.row_mapping`.
    """
    row = req_state_idx
    if row_mapping_ptr is not None:
        row = tl.load(
            row_mapping_ptr + req_state_idx, mask=req_state_idx >= 0, other=-1
        )
    return row.to(tl.int64)


@triton.jit(do_not_specialize=["num_reqs"])
def _assign_rows_kernel(
    idx_mapping_ptr,
    shard_mapping_ptr,
    row_mapping_ptr,
    inverse_row_mapping_ptr,
    num_reqs,
    NUM_SHARDS: tl.constexpr,
    SHARD_IDX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_block_idx = tl.program_id(0)
    req_block = req_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = req_block < num_reqs
    req_state_block = tl.load(idx_mapping_ptr + req_block, mask=mask, other=0)
    owners = shard_owner(req_block, NUM_SHARDS)
    rows = shard_row(req_block, NUM_SHARDS)
    is_local = owners == SHARD_IDX
    tl.store(shard_mapping_ptr + req_state_block, owners, mask=mask)
    tl.store(row_mapping_ptr + req_state_block, tl.where(is_local, rows, -1), mask=mask)
    tl.store(inverse_row_mapping_ptr + rows, req_state_block, mask=mask & is_local)


@triton.jit(do_not_specialize=["num_logits"])
def _stale_drafts_kernel(
    stale_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    row_mapping_ptr,
    inverse_row_mapping_ptr,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    logit_block_idx = tl.program_id(0)
    logit_block = logit_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = logit_block < num_logits
    req_state_block = tl.load(
        expanded_idx_mapping_ptr + logit_block, mask=mask, other=-1
    )
    local_pos = tl.load(expanded_local_pos_ptr + logit_block, mask=mask, other=0)
    rows = draft_logits_row(req_state_block, row_mapping_ptr)
    holders = tl.load(inverse_row_mapping_ptr + rows, mask=mask & (rows >= 0), other=-1)
    # Position 0 is the bonus token, which has no draft.
    stale = (local_pos > 0) & ((rows < 0) | (holders != req_state_block))
    tl.store(stale_ptr + logit_block, stale, mask=mask)


class DraftLogitsCache:
    """Pre-temperature draft logits, cached by a draft step for the next
    verification.

    A replicated cache holds every request on every TP rank, at row
    req_state_idx.

    A sharded cache (batch-sharded sampling) holds on each TP rank only the
    requests that rank verifies. Every draft step gives the request at batch
    position i to rank i % tp, at row i // tp of its cache, so no rank caches
    more than cdiv(num_reqs, tp) requests however the request slots are spread.

    Rows are reused every step. A request that the scheduler skips for a step
    keeps its drafts, but its row may since have been given to another
    request. `inverse_row_mapping` detects this, and `stale_drafts` marks those
    drafts invalid, so verification rejects them and samples from the target.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_steps: int,
        vocab_size: int,
        *,
        dtype: torch.dtype,
        fill: float,
        device: torch.device | str,
        sharded: bool = False,
    ):
        num_rows = max_num_reqs
        # [max_num_reqs] Each request's row in values, -1 if another rank
        # caches it. None for a replicated cache.
        self.row_mapping: torch.Tensor | None = None
        if sharded:
            tp_group = get_tp_group()
            self._num_shards = tp_group.world_size
            self._shard_idx = tp_group.rank_in_group
            num_rows = cdiv(max_num_reqs, self._num_shards)
            # Requests never drafted keep a slot-based owner. They have no
            # drafts, so any owner works as long as every rank agrees.
            slots = np.arange(max_num_reqs, dtype=np.int32)
            # [max_num_reqs] The TP rank that caches, and verifies, each request.
            self.shard_mapping_np = slots % self._num_shards
            self.shard_mapping = torch.from_numpy(self.shard_mapping_np).to(device)
            rows_np = np.where(
                self.shard_mapping_np == self._shard_idx,
                shard_rows(max_num_reqs, self._num_shards),
                -1,
            )
            self.row_mapping = torch.from_numpy(rows_np.astype(np.int32)).to(device)
            # [num_rows] The request last drafted into each row of this rank.
            self.inverse_row_mapping = torch.full(
                (num_rows,), -1, dtype=torch.int32, device=device
            )
        # [num_rows, num_steps, vocab_size]
        self.values = torch.full(
            (num_rows, num_steps, vocab_size), fill, dtype=dtype, device=device
        )

    def assign(self, idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray) -> None:
        """Give a sharded cache's rows out by batch position. Call after
        verification and before the draft step writes the draft logits."""
        num_reqs = idx_mapping_np.shape[0]
        if self.row_mapping is None or num_reqs == 0:
            return
        self.shard_mapping_np[idx_mapping_np] = shard_owners(num_reqs, self._num_shards)
        block_size = 1024
        _assign_rows_kernel[(triton.cdiv(num_reqs, block_size),)](
            idx_mapping,
            self.shard_mapping,
            self.row_mapping,
            self.inverse_row_mapping,
            num_reqs,
            NUM_SHARDS=self._num_shards,
            SHARD_IDX=self._shard_idx,
            BLOCK_SIZE=block_size,
        )

    def owners_of(
        self, idx_mapping_np: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor | None]:
        """The TP rank that verifies each of a batch's requests: the rank that
        cached its draft logits, or for a replicated cache, round-robin over
        batch positions.

        Returns:
            [num_reqs] each request's owner, and the [max_num_reqs] device owner
            table, or None for a replicated cache.

        """
        if self.row_mapping is None:
            num_reqs = idx_mapping_np.shape[0]
            return shard_owners(num_reqs, get_tp_group().world_size), None
        return self.shard_mapping_np[idx_mapping_np], self.shard_mapping

    def stale_drafts(self, input_batch: InputBatch) -> torch.Tensor | None:
        """Mark the drafts whose draft logits were overwritten since drafting.
        Only valid for requests this rank owns.

        Returns:
            [num_logits] True for each stale draft, or None for a replicated
            cache, whose rows are never reused.

        """
        if self.row_mapping is None:
            return None
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        num_logits = expanded_idx_mapping.shape[0]
        stale = expanded_idx_mapping.new_empty(num_logits, dtype=torch.bool)
        if num_logits == 0:
            return stale
        block_size = 1024
        _stale_drafts_kernel[(triton.cdiv(num_logits, block_size),)](
            stale,
            expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.row_mapping,
            self.inverse_row_mapping,
            num_logits,
            BLOCK_SIZE=block_size,
        )
        return stale
