# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layout a :class:`VerifyPlan` gives the target's verify forward: plan
positions (``num_computed_tokens + depth``) and each row's parent, over the chain
slot mappings, which stay as they are. Which rows a row attends to is the tree
attention backend's to derive from the parents."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.round import (
    AcceptResult,
    TargetTokens,
    VerifyPlan,
)


def arange_table(
    max_num_reqs: int, max_rows: int, device: torch.device, start: int = 0
) -> torch.Tensor:
    """``[R, C]`` int32 ``table[r, i] == start + i``: a chain's path, or with
    ``start=-1`` its parents."""
    rows = torch.arange(start, start + max_rows, dtype=torch.int32, device=device)
    return rows.expand(max_num_reqs, max_rows).contiguous()


@triton.jit
def _plan_layout_kernel(
    parent_ptr,  # [R, C] int32, by batch row
    stride_parent,
    rows_ptr,  # [R] int32: verify rows per request, by batch row
    tokens_ptr,  # [R, C] int32 candidate tokens, by batch row (unused unless TOKENS)
    stride_tokens,
    query_start_loc_ptr,  # [num_reqs + 1] int32
    positions_ptr,  # [num_tokens] int64: chain positions in, plan positions out
    input_ids_ptr,  # [num_tokens] int32: candidate rows take the plan's tokens
    tree_parents_ptr,  # [num_tokens] int32 out: parent row within the request
    pos_delta_ptr,  # [num_tokens] int32 out: plan position - chain position
    C: tl.constexpr,
    CP: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + b)
    end = tl.load(query_start_loc_ptr + b + 1)
    num_rows = tl.load(rows_ptr + b)
    start = end - num_rows
    # Every token of the request first gets the chain layout (prefill tokens in
    # front of the plan rows keep it).
    for block in range(query_start, start, BLOCK):
        offs = block + tl.arange(0, BLOCK)
        in_query = offs < start
        tl.store(tree_parents_ptr + offs, offs - query_start - 1, mask=in_query)
        tl.store(pos_delta_ptr + offs, 0, mask=in_query)

    rows = tl.arange(0, CP)
    is_row = rows < num_rows
    parents = tl.load(parent_ptr + b * stride_parent + rows, mask=rows < C, other=-1)
    # A parent that is not an earlier row is a chain step, so a bad plan stays
    # within its request.
    parents = tl.where((parents >= 0) & (parents < rows), parents, rows - 1)

    # Rows are ordered parent first, so one pass in row order sees every parent done.
    depth = tl.zeros([CP], dtype=tl.int32)
    for r in range(1, C):
        parent = tl.sum(tl.where(rows == r, parents, 0))
        parent_depth = tl.sum(tl.where(rows == parent, depth, 0))
        depth = tl.where(rows == r, parent_depth + 1, depth)
    # Chain position = context + row; plan position = context + depth.
    positions = tl.load(positions_ptr + start + rows, mask=is_row, other=0)
    tl.store(positions_ptr + start + rows, positions - rows + depth, mask=is_row)
    tl.store(pos_delta_ptr + start + rows, depth - rows, mask=is_row)
    # Parent as a row of the request's query: the root continues the token before
    # it (-1 when the root is the query's first token).
    offset = start - query_start
    local_parent = tl.where(parents >= 0, offset + parents, offset - 1)
    tl.store(tree_parents_ptr + start + rows, local_parent, mask=is_row)
    if TOKENS:
        is_candidate = is_row & (rows > 0)
        tokens = tl.load(tokens_ptr + b * stride_tokens + rows, mask=is_candidate)
        tl.store(input_ids_ptr + start + rows, tokens, mask=is_candidate)


class PlanLayout:
    """Per query token, in static buffers: ``tree_parents`` (parent row in the
    query, -1 for its first token) and ``pos_delta`` (plan minus chain
    position), plus the chain positions to put back after the forward."""

    def __init__(
        self,
        max_num_tokens: int,
        max_rows: int,
        device: torch.device,
    ):
        self.max_rows = max_rows
        self.tree_parents = torch.full(
            (max_num_tokens,), -1, dtype=torch.int32, device=device
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
        rows: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        num_reqs: int,
        num_tokens: int,
        num_tokens_padded: int | None = None,
    ) -> None:
        """Turn ``positions`` from chain to plan positions, write the plan's
        candidate tokens into ``input_ids`` (a plan with ``tokens``), and fill the
        layout buffers for the first ``num_reqs`` batch rows, whose requests
        verify the last ``rows[b]`` tokens of their queries."""
        assert plan.parent.shape[1] <= self.max_rows
        self.chain_positions[:num_tokens].copy_(positions[:num_tokens])
        if num_tokens_padded is not None and num_tokens_padded > num_tokens:
            self.tree_parents[num_tokens:num_tokens_padded].fill_(-1)
            self.pos_delta[num_tokens:num_tokens_padded].zero_()
        width = plan.parent.shape[1]
        tokens = plan.tokens
        assert tokens is None or input_ids is not None
        _plan_layout_kernel[(num_reqs,)](
            plan.parent,
            plan.parent.stride(0),
            rows,
            tokens if tokens is not None else plan.parent,
            tokens.stride(0) if tokens is not None else 0,
            query_start_loc,
            positions,
            input_ids if input_ids is not None else positions,
            self.tree_parents,
            self.pos_delta,
            C=width,
            CP=triton.next_power_of_2(width),
            TOKENS=tokens is not None,
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


@triton.jit
def _check_path_kernel(
    parent_ptr,  # [R, C] int32, by batch row
    stride_parent,
    row_start_ptr,  # [num_reqs + 1] int32
    path_ptr,  # [R, C] int32
    stride_path,
    path_len_ptr,  # [R] int32
    ok_ptr,  # [1] int32 out: 1 iff every request's path is in the plan
    num_reqs,
    C: tl.constexpr,
    CP: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cols = tl.arange(0, CP)[None, :]
    num_bad = 0
    for block in range(0, num_reqs, BLOCK_R):
        reqs = block + tl.arange(0, BLOCK_R)
        live = reqs < num_reqs
        rows = tl.load(row_start_ptr + reqs + 1, mask=live, other=1) - tl.load(
            row_start_ptr + reqs, mask=live, other=0
        )
        path_len = tl.load(path_len_ptr + reqs, mask=live, other=1)
        on_path = live[:, None] & (cols < path_len[:, None]) & (cols < C)
        path = tl.load(path_ptr + reqs[:, None] * stride_path + cols, mask=on_path)
        prev = tl.load(
            path_ptr + reqs[:, None] * stride_path + cols - 1,
            mask=on_path & (cols > 0),
            other=-1,
        )
        in_rows = (path >= 0) & (path < rows[:, None]) & (path < C)
        parent = tl.load(
            parent_ptr + reqs[:, None] * stride_parent + path,
            mask=on_path & in_rows,
            other=-2,
        )
        # Row 0 is the root; every later row is a child of the row before it.
        step_ok = ~on_path | tl.where(cols == 0, path == 0, in_rows & (parent == prev))
        req_ok = (path_len >= 1) & (path_len <= rows)
        req_ok = req_ok & (tl.min(step_ok.to(tl.int32), axis=1) == 1)
        num_bad += tl.sum((live & ~req_ok).to(tl.int32))
    tl.store(ok_ptr, (num_bad == 0).to(tl.int32))


def check_accept_result(
    plan: VerifyPlan, target: TargetTokens, result: AcceptResult
) -> None:
    """Assert on device, without a host sync, that ``result`` is a path of
    ``plan``: it starts at the root, each row's parent is the row before it,
    and ``path_len`` is in ``[1, rows]``. One kernel and one async assert."""
    num_reqs = target.row_start.shape[0] - 1
    width = plan.parent.shape[1]
    ok = torch.empty(1, dtype=torch.int32, device=result.path.device)
    _check_path_kernel[(1,)](
        plan.parent,
        plan.parent.stride(0),
        target.row_start,
        result.path,
        result.path.stride(0),
        result.path_len,
        ok,
        num_reqs,
        C=width,
        CP=triton.next_power_of_2(width),
        BLOCK_R=32,
        num_warps=1,
    )
    torch._assert_async(ok, "tree_accept returned a path that is not in the plan")
