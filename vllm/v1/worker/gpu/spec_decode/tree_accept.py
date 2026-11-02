# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verification of a plan that is not a chain: the walk down the tree.

Every plan row gets the target's token. The walk then goes from the root to the
child whose token equals the current row's target token, until a row has no such
child. The accepted rows are the path (root first); the committed tokens are the
target tokens of the path rows, the last of which is the bonus token.

The default ``accept`` for such a plan takes the argmax, which is exact for greedy
requests. A speculator that verifies sampled requests on trees overrides
``accept`` and passes its own per-row target tokens to :func:`walk_tree`.
"""

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.round import (
    AcceptResult,
    VerifyOutput,
    VerifyPlan,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.sample.sampler import Sampler


@triton.jit
def _walk_tree_kernel(
    target_ptr,  # [num_rows] int64: the target's token at every row
    drafted_ptr,  # [num_rows] int64: each row's token, -1 never accepted
    row_start_ptr,  # [R + 1] int32
    parent_ptr,  # [R, C] int32
    stride_parent,
    count_ptr,  # [R] int32
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
    num_rows = tl.minimum(
        tl.load(row_start_ptr + b + 1) - start, tl.load(count_ptr + b)
    )
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


def walk_tree(
    plan: VerifyPlan,
    output: VerifyOutput,
    target: torch.Tensor,
    width: int | None = None,
) -> AcceptResult:
    """The accepted path of each request given ``target [num_rows]``, the
    target's token at every row. ``sampled`` is ``[R, width]`` (default ``C``);
    rows past ``plan.count`` and rows whose ``output.drafted`` is -1 are never
    accepted."""
    assert plan.count is not None
    num_reqs, num_cols = plan.parent.shape
    width = num_cols if width is None else width
    assert width >= num_cols
    device = target.device
    sampled = torch.zeros(num_reqs, width, dtype=torch.int64, device=device)
    path = torch.empty(num_reqs, num_cols, dtype=torch.int32, device=device)
    path_len = torch.empty(num_reqs, dtype=torch.int32, device=device)
    _walk_tree_kernel[(num_reqs,)](
        target,
        output.drafted,
        output.row_start,
        plan.parent,
        plan.parent.stride(0),
        plan.count,
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


def accept_greedy_tree(
    plan: VerifyPlan, output: VerifyOutput, sampler: "Sampler", width: int
) -> AcceptResult:
    """The default ``accept`` of a plan that is not a chain: the walk on the
    argmax of each row's processed logits. Only greedy requests; a speculator
    that verifies sampled trees overrides ``accept``."""
    temperature = sampler.sampling_states.temperature.np[output.idx_mapping_np]
    if not bool(np.all(temperature == 0.0)):
        raise ValueError(
            "a plan that is not a chain verifies sampled requests only through "
            "the speculator's accept; the default accepts greedy requests"
        )
    processed = sampler.apply_sampling_params(
        output.logits,
        output.expanded_idx_mapping,
        output.idx_mapping,
        output.idx_mapping_np,
        output.positions,
        output.draft_tokens,
        output.expanded_local_pos,
        output.seq_lens_upper_bound_np,
    )
    result = walk_tree(plan, output, processed.argmax(dim=-1), width)
    if processed is not output.logits:
        result.processed_logits = processed
    return result
