# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layout a :class:`VerifyPlan` gives the target's verify forward.

Each request verifies the last ``count[b]`` rows of its query; row 0 of those is the
root. The runner computes the batch's positions and slot mappings as for a chain
(``num_computed_tokens + i``) and keeps the slot mappings: every row writes its KV at
the chain slot of its row index, and the commit compacts the accepted path afterwards.
For a plan that is not a chain, :meth:`PlanLayout.apply` then moves each row's
position to ``num_computed_tokens + depth`` (or ``+ VerifyPlan.positions``) and
computes its ancestor set (bit ``j`` set iff row ``j`` is the row or one of its
ancestors), which tree attention reads, and, for a layout with a mask buffer, the
rows it sees (``VerifyPlan.mask`` in the same bits, else the ancestor set), which
custom-mask attention reads.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.round import VerifyPlan

MAX_PLAN_ROWS = 64
"""Rows per request a plan can have: one bit per row in an int64 ancestor set."""


def chain_parents(
    max_num_reqs: int, max_rows: int, device: torch.device
) -> torch.Tensor:
    """The ``[R, C]`` parent table of a chain: ``parent[r, i] == i - 1``."""
    rows = torch.arange(-1, max_rows - 1, dtype=torch.int32, device=device)
    return rows.expand(max_num_reqs, max_rows).contiguous()


def chain_path(max_num_reqs: int, max_rows: int, device: torch.device) -> torch.Tensor:
    """The ``[R, C]`` accepted path of a chain: ``path[r, j] == j``."""
    rows = torch.arange(max_rows, dtype=torch.int32, device=device)
    return rows.expand(max_num_reqs, max_rows).contiguous()


@triton.jit
def _plan_layout_kernel(
    parent_ptr,  # [R, C] int32, by batch row
    stride_parent,
    count_ptr,  # [R] int32, by batch row
    tokens_ptr,  # [R, C] int32 candidate tokens, by batch row (unused unless TOKENS)
    stride_tokens,
    query_start_loc_ptr,  # [num_reqs + 1] int32
    positions_ptr,  # [num_tokens] int64: chain positions in, plan positions out
    input_ids_ptr,  # [num_tokens] int32: candidate rows take the plan's tokens
    tree_parents_ptr,  # [num_tokens] int32 out: parent row within the request
    tree_ancestors_ptr,  # [num_tokens] int64 out: rows seen, 0 = causal
    pos_delta_ptr,  # [num_tokens] int32 out: plan position - chain position
    plan_positions_ptr,  # [R, C] int32 position offsets (unused unless POSITIONS)
    stride_plan_positions,
    plan_mask_ptr,  # [R, C, C] bool rows seen (unused unless MASK)
    stride_mask_r,
    stride_mask_i,
    stride_mask_j,
    C: tl.constexpr,
    CP: tl.constexpr,
    TOKENS: tl.constexpr,
    CHAIN: tl.constexpr,
    POSITIONS: tl.constexpr,
    MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + b)
    end = tl.load(query_start_loc_ptr + b + 1)
    count = tl.load(count_ptr + b)
    start = end - count
    # Every token of the request first gets the chain layout (prefill tokens in
    # front of the plan rows keep it).
    for block in range(query_start, start, BLOCK):
        offs = block + tl.arange(0, BLOCK)
        in_query = offs < start
        tl.store(tree_parents_ptr + offs, offs - query_start - 1, mask=in_query)
        tl.store(tree_ancestors_ptr + offs, tl.zeros([BLOCK], tl.int64), mask=in_query)
        tl.store(pos_delta_ptr + offs, 0, mask=in_query)

    rows = tl.arange(0, CP)
    is_row = rows < count
    parents = tl.load(parent_ptr + b * stride_parent + rows, mask=rows < C, other=-1)

    # Rows are ordered parent first, so one pass in row order sees every parent done.
    depth = tl.zeros([CP], dtype=tl.int32)
    ancestors = tl.full([CP], 1, dtype=tl.int64)
    for r in range(1, C):
        parent = tl.sum(tl.where(rows == r, parents, 0))
        at_parent = rows == parent
        depth = tl.where(rows == r, tl.sum(tl.where(at_parent, depth, 0)) + 1, depth)
        own_bit = tl.cast(1, tl.int64) << r
        ancestors = tl.where(
            rows == r, tl.sum(tl.where(at_parent, ancestors, 0)) | own_bit, ancestors
        )
    # Chain position = context + row; plan position = context + depth, or context
    # + the plan's own offset.
    if POSITIONS:
        offset_pos = tl.load(
            plan_positions_ptr + b * stride_plan_positions + rows, mask=is_row, other=0
        )
    else:
        offset_pos = depth
    positions = tl.load(positions_ptr + start + rows, mask=is_row, other=0)
    tl.store(positions_ptr + start + rows, positions - rows + offset_pos, mask=is_row)
    tl.store(pos_delta_ptr + start + rows, offset_pos - rows, mask=is_row)
    # Parent as a row of the request's query: the root continues the token before
    # it (-1 when the root is the query's first token).
    offset = start - query_start
    local_parent = tl.where(parents >= 0, offset + parents, offset - 1)
    tl.store(tree_parents_ptr + start + rows, local_parent, mask=is_row)
    # A chain attends causally; a branching plan is the whole query (prefills
    # verify chains only), so its bits index the request's query rows.
    if CHAIN:
        seen = tl.zeros([CP], tl.int64)
    elif MASK:
        # Row i sees row j <= i where mask[b, i, j]; its own bit is always set
        # and later rows are never seen.
        cols = tl.arange(0, CP)
        lower = (cols[None, :] <= rows[:, None]) & (rows[:, None] < C)
        lower = lower & (cols[None, :] < C)
        sees = tl.load(
            plan_mask_ptr
            + b * stride_mask_r
            + rows[:, None] * stride_mask_i
            + cols[None, :] * stride_mask_j,
            mask=lower,
            other=0,
        )
        bits = tl.where((sees != 0) & lower, tl.cast(1, tl.int64) << cols[None, :], 0)
        own = tl.cast(1, tl.int64) << rows
        seen = tl.where(is_row, tl.sum(bits, axis=1) | own, 0)
    else:
        seen = ancestors
    tl.store(tree_ancestors_ptr + start + rows, seen, mask=is_row)
    if TOKENS:
        is_candidate = is_row & (rows > 0)
        tokens = tl.load(tokens_ptr + b * stride_tokens + rows, mask=is_candidate)
        tl.store(input_ids_ptr + start + rows, tokens, mask=is_candidate)


class PlanLayout:
    """The batch's verify rows laid out per query token, in buffers CUDA graphs
    replay, and the chain positions to put back after the forward.

    Per query token: ``tree_parents`` (the parent as a row of the request's query,
    -1 for the query's first token; a chain's ``i - 1`` for tokens that are not plan
    rows), ``tree_ancestors`` (the query rows the row sees, as bits: the plan's
    ``mask`` when it has one, else the row's ancestors; 0 for every token of a
    chain: attend causally) and ``pos_delta`` (plan position minus chain position).
    The first two are what ``CommonAttentionMetadata.tree_parents`` /
    ``tree_ancestors`` point at.
    """

    def __init__(
        self,
        max_num_tokens: int,
        max_rows: int,
        device: torch.device,
    ):
        if max_rows > MAX_PLAN_ROWS:
            raise ValueError(
                f"a verify plan has at most {MAX_PLAN_ROWS} rows per request, "
                f"got {max_rows}"
            )
        self.max_rows = max_rows
        self.tree_parents = torch.full(
            (max_num_tokens,), -1, dtype=torch.int32, device=device
        )
        self.tree_ancestors = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        )
        self.pos_delta = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.chain_positions = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        )
        # Positions currently hold plan positions that ``restore_positions`` undoes.
        self.applied = False

    def apply(
        self,
        plan: VerifyPlan,
        count: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        num_reqs: int,
        num_tokens: int,
        num_tokens_padded: int | None = None,
    ) -> None:
        """Turn ``positions`` from chain to plan positions, write the plan's
        candidate tokens into ``input_ids`` (a plan with ``tokens``), and fill the
        layout buffers for the first ``num_reqs`` batch rows. ``count`` is the
        plan's count, or the rows per request of a chain the runner lays out."""
        assert plan.parent.shape[1] <= self.max_rows
        self.chain_positions[:num_tokens].copy_(positions[:num_tokens])
        if num_tokens_padded is not None and num_tokens_padded > num_tokens:
            self.tree_parents[num_tokens:num_tokens_padded].fill_(-1)
            self.tree_ancestors[num_tokens:num_tokens_padded].zero_()
            self.pos_delta[num_tokens:num_tokens_padded].zero_()
        num_rows = plan.parent.shape[1]
        tokens = plan.tokens if plan.tokens is not None and not plan.is_chain else None
        assert tokens is None or input_ids is not None
        plan_positions, mask = plan.positions, plan.mask
        _plan_layout_kernel[(num_reqs,)](
            plan.parent,
            plan.parent.stride(0),
            count,
            tokens if tokens is not None else plan.parent,
            tokens.stride(0) if tokens is not None else 0,
            query_start_loc,
            positions,
            input_ids if input_ids is not None else positions,
            self.tree_parents,
            self.tree_ancestors,
            self.pos_delta,
            plan_positions if plan_positions is not None else plan.parent,
            plan_positions.stride(0) if plan_positions is not None else 0,
            mask if mask is not None else plan.parent,
            mask.stride(0) if mask is not None else 0,
            mask.stride(1) if mask is not None else 0,
            mask.stride(2) if mask is not None else 0,
            C=num_rows,
            CP=triton.next_power_of_2(num_rows),
            TOKENS=tokens is not None,
            CHAIN=plan.is_chain,
            POSITIONS=plan_positions is not None,
            MASK=mask is not None,
            BLOCK=256,
            num_warps=1,
        )
        self.applied = True

    def restore_positions(self, positions: torch.Tensor, num_tokens: int) -> None:
        """Put the chain positions back after the forward: the sampler and the
        drafter read them."""
        if self.applied:
            positions[:num_tokens].copy_(self.chain_positions[:num_tokens])
            self.applied = False
