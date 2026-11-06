# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _candidate_edges_kernel(
    PREDECESSORS,
    SUCCESSORS,
    IDS,
    UNARY,
    HIDDEN,
    ANCHORS,
    OUTPUT,
    ANCHOR_STRIDE: tl.constexpr,
    STEPS: tl.constexpr,
    TOP_K: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    position = tl.program_id(0)
    predecessor = tl.program_id(1)
    step = position % STEPS
    if step == 0:
        pred_id = tl.load(ANCHORS + position // STEPS * ANCHOR_STRIDE).to(tl.int64)
    else:
        pred_id = tl.load(IDS + (position - 1) * TOP_K + predecessor).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_K)
    pred = tl.load(PREDECESSORS + pred_id.to(tl.int64) * RANK + r, r < RANK, 0)
    hidden = tl.load(HIDDEN + position * RANK + r, r < RANK, 0)
    # Preserve the materialized product and GEMM's output rounding.
    weighted = (pred.to(tl.float32) * hidden.to(tl.float32)).to(hidden.dtype)
    ids = tl.load(IDS + position * TOP_K + c, c < TOP_K, 0).to(tl.int64)
    successors = tl.load(
        SUCCESSORS + ids[:, None] * RANK + r[None, :],
        (c[:, None] < TOP_K) & (r[None, :] < RANK),
        0,
    )
    scores = tl.sum(successors.to(tl.float32) * weighted[None, :].to(tl.float32), 1)
    scores = scores.to(hidden.dtype).to(tl.float32)
    unary = tl.load(UNARY + position * TOP_K + c, c < TOP_K, 0).to(tl.float32)
    tl.store(
        OUTPUT + (position * TOP_K + predecessor) * TOP_K + c, scores + unary, c < TOP_K
    )


def candidate_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Gather codebooks and score candidate edges without intermediate tensors."""
    batch, steps, _ = candidate_ids.shape
    rank = hidden.shape[-1]
    output = torch.empty(
        (batch, steps, top_k, top_k), dtype=unary_logits.dtype, device=hidden.device
    )
    if batch * steps:
        _candidate_edges_kernel[(batch * steps, top_k)](
            predecessor_table,
            successor_table,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            output,
            anchor_token_ids.stride(0),
            steps,
            top_k,
            rank,
            triton.next_power_of_2(top_k),
            triton.next_power_of_2(rank),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
