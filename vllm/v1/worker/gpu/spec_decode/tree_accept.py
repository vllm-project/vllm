# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verification of a plan that is not a chain: the walk down the tree.

Every plan row gets the target's token: the argmax for a greedy request, else a
draw from the row's processed target distribution at its plan position. The walk
then goes from the root to the child whose token equals the current row's target
token, until a row has no such child. The accepted rows are the path (root
first); the committed tokens are the target tokens of the path rows, the last of
which is the bonus token.

Each committed token is the target's draw at its prefix, so the walk is lossless
whenever the drafted tokens do not depend on that draw (deterministic drafts,
such as a draft model's top-k).
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.round import AcceptResult, VerifyPlan


@dataclass
class VerifyOutput:
    """What the target computed for the rows of a request range, in plan order
    (every tensor indexed from the range's first row)."""

    logits: torch.Tensor
    """``[num_rows, vocab]`` logits after the sampling parameters."""
    target: torch.Tensor
    """``[num_rows]`` int64 the target's token at each row."""
    row_start: torch.Tensor
    """``[num_reqs + 1]`` int32 cumulative row offsets of each request."""
    drafted: torch.Tensor
    """``[num_rows]`` int64 token each row holds (row 0: the last committed
    token); -1 for a row that must not be accepted (a placeholder or a
    grammar-rejected draft)."""


@triton.jit
def _walk_tree_kernel(
    target_ptr,  # [num_rows] int64: the target's token at every row
    drafted_ptr,  # [num_rows] int64: each row's token, -1 never accepted
    row_start_ptr,  # [R + 1] int32
    parent_ptr,  # [R, C] int32
    stride_parent,
    sampled_ptr,  # [R, S] int64 out
    stride_sampled,
    path_ptr,  # [R, C] int32 out
    stride_path,
    path_len_ptr,  # [R] int32 out
    C: tl.constexpr,
    CP: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(row_start_ptr + b)
    num_rows = tl.load(row_start_ptr + b + 1) - start
    rows = tl.arange(0, CP)
    live = (rows > 0) & (rows < num_rows)
    parent = tl.load(parent_ptr + b * stride_parent + rows, mask=rows < C, other=-1)
    drafted = tl.load(drafted_ptr + start + rows, mask=live, other=-1)
    target = tl.load(target_ptr + start + rows, mask=rows < num_rows, other=-1)
    tl.store(path_ptr + b * stride_path + rows, rows, mask=rows < C)

    node = 0
    alive = num_rows > 0
    length = 0
    for step in range(0, C):
        token = tl.sum(tl.where(rows == node, target, 0))
        tl.store(sampled_ptr + b * stride_sampled + step, token, mask=alive)
        tl.store(path_ptr + b * stride_path + step, node, mask=alive)
        length += alive.to(tl.int32)
        child = tl.min(tl.where(live & (parent == node) & (drafted == token), rows, CP))
        alive = alive & (child < CP)
        node = tl.where(alive, child, node)
    tl.store(path_len_ptr + b, length)


def walk_tree(plan: VerifyPlan, output: VerifyOutput, width: int) -> AcceptResult:
    """The accepted path of each request: the walk on ``output.target``.
    ``sampled`` is ``[R, width]``; rows past a request's verified rows and rows
    whose ``output.drafted`` is -1 are never accepted."""
    num_reqs, num_cols = plan.parent.shape
    assert width >= num_cols
    device = output.target.device
    sampled = torch.zeros(num_reqs, width, dtype=torch.int64, device=device)
    path = torch.empty(num_reqs, num_cols, dtype=torch.int32, device=device)
    path_len = torch.empty(num_reqs, dtype=torch.int32, device=device)
    _walk_tree_kernel[(num_reqs,)](
        output.target,
        output.drafted,
        output.row_start,
        plan.parent,
        plan.parent.stride(0),
        sampled,
        sampled.stride(0),
        path,
        path.stride(0),
        path_len,
        C=num_cols,
        CP=triton.next_power_of_2(num_cols),
        num_warps=1,
    )
    return AcceptResult(path=path, path_len=path_len, sampled=sampled)
