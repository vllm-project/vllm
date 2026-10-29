# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-known adaptive query lengths for PCP partitioning."""

import numpy as np

from vllm.distributed.parallel_state import get_pcp_group, get_tp_group


def select_pcp_drafts(
    survival: np.ndarray,
    scheduled_drafts: np.ndarray,
    non_draft_tokens: np.ndarray,
    cost_tables: tuple[np.ndarray, np.ndarray],
    max_draft_budget: int,
    pcp_size: int,
    shard_decode_requests: bool,
    has_prefill: bool,
) -> np.ndarray:
    """Price each candidate budget by its largest rank-local target batch."""
    num_reqs, num_steps = survival.shape
    valid = np.arange(num_steps)[None, :] < scheduled_drafts[:, None]
    order = np.argsort(-survival[valid], kind="stable")[:max_draft_budget]
    requests = np.broadcast_to(np.arange(num_reqs)[:, None], valid.shape)[valid][order]
    if has_prefill:
        # Keep the existing full-width verification policy for mixed batches.
        budget = len(order)
    else:
        owners = np.arange(num_reqs) % pcp_size
        if not shard_decode_requests:
            owners.fill(0)
        base_tokens = np.bincount(
            owners, weights=non_draft_tokens, minlength=pcp_size
        ).astype(np.int64)
        rank_drafts = np.zeros((len(order) + 1, pcp_size), dtype=np.int64)
        rank_drafts[1:] = np.cumsum(
            owners[requests, None] == np.arange(pcp_size), axis=0
        )
        local_tokens = (base_tokens + rank_drafts).max(axis=1)
        draft_cost, verify_cost = cost_tables
        costs = draft_cost[num_reqs] + verify_cost[local_tokens]
        accepted = np.concatenate(
            ([num_reqs], num_reqs + np.cumsum(survival[valid][order]))
        )
        budget = int(np.argmax(accepted / costs))
    return np.bincount(requests[:budget], minlength=num_reqs).astype(np.int32)


class PCPAdaptiveVerification:
    def __init__(self, pcp_size: int, shard_decode_requests: bool):
        self.pcp_size = pcp_size
        self.shard_decode_requests = shard_decode_requests
        self.admitted_drafts: np.ndarray | None = None
        self.num_tokens_per_req: np.ndarray | None = None

    def plan(
        self,
        survival: np.ndarray,
        scheduled_drafts: np.ndarray,
        non_draft_tokens: np.ndarray,
        cost_tables: tuple[np.ndarray, np.ndarray],
        max_draft_budget: int,
        has_prefill: bool,
    ) -> int:
        admitted = select_pcp_drafts(
            survival,
            scheduled_drafts,
            non_draft_tokens,
            cost_tables,
            max_draft_budget,
            self.pcp_size,
            self.shard_decode_requests,
            has_prefill,
        )
        # Both dimensions must agree on the complete layout, including ties.
        admitted = get_tp_group().broadcast_object(admitted, src=0)
        admitted = get_pcp_group().broadcast_object(admitted, src=0)
        self.admitted_drafts = admitted
        self.num_tokens_per_req = non_draft_tokens + admitted
        return int(admitted.sum())
