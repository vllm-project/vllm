# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_gather,
    batch_isend_irecv,
    get_global_rank,
)

from vllm.config.parallel import EPLBCommunicationConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RecvMetadata:
    """Metadata describing remote receives during EPLB rebalancing."""

    recv_primary_mask: np.ndarray
    """Mask of (num_local_experts,) indicating primary experts received."""
    recv_count: int
    """Number of received experts for the layer."""
    recv_expert_ids: np.ndarray
    """Expert ids (num_local_experts,) of remote primary experts."""
    recv_dst_rows: np.ndarray
    """Target expert indices (num_local_experts,) in local tensors to send."""


@dataclass
class CommunicationPlan:
    """Communication plan for P2P operations during EPLB rebalancing."""

    send_ops_per_round: list[list[P2POp]]
    """List of send operations for each communication round."""
    recv_ops_per_round: list[list[P2POp]]
    """List of receive operations for each communication round."""
    rank_to_global: dict[int, int]
    """Mapping from local EP rank to global rank."""
    num_rounds: int
    """Total number of communication rounds."""
    ops_per_expert: int
    """Number of P2P operations per expert (equals number of weight tensors)."""


# Type alias for the result of move_to_buffer or transfer_layer
MoveToBufferResult = tuple[np.ndarray, np.ndarray, RecvMetadata]


def get_ep_ranks_with_experts_batch(
    expert_ids: np.ndarray,
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Get the ranks of the experts that need to be exchanged.

    Args:
        expert_ids: 1D array of expert indices to query.
        num_local_experts: The number of local experts.
        old_indices: The old indices of the experts.
        new_indices: The new indices of the experts.

    Returns:
        A tuple of two dictionaries mapping expert_id to:
        - ranks_to_send: The ranks that have this expert and need to send.
        - ranks_to_recv: The ranks that need to receive this expert.
    """
    ranks_to_send_map: dict[int, list[int]] = {}
    ranks_to_recv_map: dict[int, list[int]] = {}

    # Fast path: if no experts, return empty dicts
    if expert_ids.size == 0:
        return ranks_to_send_map, ranks_to_recv_map

    unique_experts = np.unique(expert_ids)
    num_positions = len(old_indices)
    position_indices = np.arange(num_positions, dtype=np.int32)

    # Vectorized approach: find all positions matching any query expert in one pass
    # Use np.isin to get boolean masks for all relevant positions at once
    old_relevant_mask = np.isin(old_indices, unique_experts)
    new_relevant_mask = np.isin(new_indices, unique_experts)

    # Process old_indices (send ranks)
    if np.any(old_relevant_mask):
        old_relevant_positions = position_indices[old_relevant_mask]
        old_relevant_experts = old_indices[old_relevant_mask]
        old_relevant_ranks = old_relevant_positions // num_local_experts

        # Sort by expert first, then by position (to maintain first-appearance order)
        sort_order = np.lexsort((old_relevant_positions, old_relevant_experts))
        sorted_experts = old_relevant_experts[sort_order]
        sorted_ranks = old_relevant_ranks[sort_order]

        # Find boundaries where expert changes
        expert_boundaries = np.concatenate(
            [[0], np.where(np.diff(sorted_experts) != 0)[0] + 1, [len(sorted_experts)]]
        )

        # For each expert, extract unique ranks in order of first appearance
        for i in range(len(expert_boundaries) - 1):
            start, end = expert_boundaries[i], expert_boundaries[i + 1]
            expert = int(sorted_experts[start])
            expert_ranks = sorted_ranks[start:end]

            # Get unique ranks preserving order
            _, unique_idx = np.unique(expert_ranks, return_index=True)
            unique_ranks = expert_ranks[np.sort(unique_idx)]
            ranks_to_send_map[expert] = unique_ranks.tolist()

    # Process new_indices (recv ranks)
    if np.any(new_relevant_mask):
        new_relevant_positions = position_indices[new_relevant_mask]
        new_relevant_experts = new_indices[new_relevant_mask]
        new_relevant_ranks = new_relevant_positions // num_local_experts

        # Sort by expert first, then by position
        sort_order = np.lexsort((new_relevant_positions, new_relevant_experts))
        sorted_experts = new_relevant_experts[sort_order]
        sorted_ranks = new_relevant_ranks[sort_order]

        # Find boundaries where expert changes
        expert_boundaries = np.concatenate(
            [[0], np.where(np.diff(sorted_experts) != 0)[0] + 1, [len(sorted_experts)]]
        )

        # For each expert, extract unique ranks and exclude local copies
        for i in range(len(expert_boundaries) - 1):
            start, end = expert_boundaries[i], expert_boundaries[i + 1]
            expert = int(sorted_experts[start])
            expert_ranks = sorted_ranks[start:end]

            # Get unique ranks preserving order
            _, unique_idx = np.unique(expert_ranks, return_index=True)
            unique_ranks = expert_ranks[np.sort(unique_idx)]

            # Remove ranks that have local copies (in send map)
            send_ranks_set = set(ranks_to_send_map.get(expert, []))
            recv_ranks_actual = [
                int(r) for r in unique_ranks if r not in send_ranks_set
            ]
            ranks_to_recv_map[expert] = recv_ranks_actual

    # Handle experts that only appear in old (send only) or new (recv only)
    for expert in unique_experts:
        expert = int(expert)
        if expert not in ranks_to_send_map:
            ranks_to_send_map[expert] = []
        if expert not in ranks_to_recv_map:
            ranks_to_recv_map[expert] = []

    return ranks_to_send_map, ranks_to_recv_map


def _get_communication_round(rank1: int, rank2: int, group_size: int) -> int:
    """
    Determine which communication round a pair of ranks belongs to.

    Uses hierarchical grouping: ranks are divided into groups of size
    `group_size`.
    - Round 0: Same group (intra-group)
    - Round i (i>0): Communication between groups with XOR = i

    Returns the round index based on the XOR of group IDs.
    """
    group1 = rank1 // group_size
    group2 = rank2 // group_size
    return group1 ^ group2


def _build_communication_plan(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    expert_weights: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    communication_config: EPLBCommunicationConfig,
    send_expert_ids: np.ndarray,
    send_src_rows: np.ndarray,
    send_count: int,
    recv_expert_ids: np.ndarray,
    recv_dst_rows: np.ndarray,
    recv_count: int,
    expert_weights_buffers: Sequence[torch.Tensor],
) -> CommunicationPlan:
    """
    Build a communication plan for P2P operations during EPLB rebalancing.

    Args:
        num_local_experts: Number of local experts.
        old_indices: (num_experts_total,) ndarray of current expert assignments.
        new_indices: (num_experts_total,) ndarray of desired expert assignments.
        expert_weights: Original expert weights for the layer.
        ep_group: Distributed process group for expert parallel comms.
        communication_config: Communication configuration for P2P operations.
        send_expert_ids: Expert IDs to send (num_local_experts,).
        send_src_rows: Source rows for sends (num_local_experts,).
        send_count: Number of valid send entries.
        recv_expert_ids: Expert IDs to receive (num_local_experts,).
        recv_dst_rows: Destination rows for receives (num_local_experts,).
        recv_count: Number of valid receive entries.
        expert_weights_buffers: Intermediate buffers (one per tensor).

    Returns:
        CommunicationPlan containing send/recv operations organized by rounds.
    """
    ep_rank = ep_group.rank()
    ep_size = ep_group.size()

    # Pre-compute global ranks mapping
    rank_to_global = {rank: get_global_rank(ep_group, rank) for rank in range(ep_size)}

    # Use the already-validated configuration from EPLBConfig
    num_groups = communication_config.num_groups
    group_size = ep_size // num_groups

    # Calculate number of communication rounds
    max_group_id = (ep_size - 1) // group_size
    num_rounds = 1 << max_group_id.bit_length()

    # Calculate operations per expert (one operation per weight tensor)
    ops_per_expert = len(expert_weights)

    # Create separate lists for sends and recvs for each round
    send_ops_per_round: list[list[P2POp]] = [[] for _ in range(num_rounds)]
    recv_ops_per_round: list[list[P2POp]] = [[] for _ in range(num_rounds)]

    # Build send operations
    if send_count > 0:
        experts = send_expert_ids[:send_count]
        srcs = send_src_rows[:send_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        srcs = srcs[order]

        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts,
            num_local_experts,
            old_indices,
            new_indices,
        )

        for expert, src in zip(experts.tolist(), srcs.tolist()):
            ranks_to_send = send_map[expert]
            ranks_to_recv = recv_map[expert]
            if not ranks_to_send or not ranks_to_recv:
                continue
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            sender_pos = ranks_to_send.index(ep_rank)
            recv_begin = sender_pos * num_dst_per_sender
            recv_end = recv_begin + num_dst_per_sender
            recv_ranks = ranks_to_recv[recv_begin:recv_end]
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            recver_pos = remainder_start + sender_pos
            if recver_pos < len(ranks_to_recv):
                recv_ranks.append(ranks_to_recv[recver_pos])
            for dst in recv_ranks:
                dst_global = rank_to_global[dst]
                send_ops = [
                    P2POp(
                        torch.distributed.isend,
                        w[src],
                        dst_global,
                    )
                    for w in expert_weights
                ]
                # Determine which round based on sender and receiver ranks
                round_idx = _get_communication_round(ep_rank, dst, group_size)
                send_ops_per_round[round_idx] += send_ops

    # Build receive operations
    if recv_count > 0:
        experts = recv_expert_ids[:recv_count]
        dsts = recv_dst_rows[:recv_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        dsts = dsts[order]

        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts,
            num_local_experts,
            old_indices,
            new_indices,
        )

        for expert, dst in zip(experts.tolist(), dsts.tolist()):
            ranks_to_send = send_map[expert]
            ranks_to_recv = recv_map[expert]
            if not ranks_to_send or not ranks_to_recv:
                continue
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            recver_pos = ranks_to_recv.index(ep_rank)
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            if recver_pos < remainder_start:
                src = ranks_to_send[recver_pos // num_dst_per_sender]
            else:
                src = ranks_to_send[recver_pos - remainder_start]
            src_global = rank_to_global[src]
            recv_ops = [
                P2POp(
                    torch.distributed.irecv,
                    b[dst],
                    src_global,
                )
                for b in expert_weights_buffers
            ]
            # Determine which round based on sender and receiver ranks
            round_idx = _get_communication_round(ep_rank, src, group_size)
            recv_ops_per_round[round_idx] += recv_ops

    return CommunicationPlan(
        send_ops_per_round=send_ops_per_round,
        recv_ops_per_round=recv_ops_per_round,
        rank_to_global=rank_to_global,
        num_rounds=num_rounds,
        ops_per_expert=ops_per_expert,
    )


def _execute_batch(ops: list[P2POp], cuda_stream: torch.cuda.Stream | None) -> None:
    """Execute a batch of P2P operations and wait for completion."""
    if cuda_stream is not None:
        with torch.cuda.stream(cuda_stream):
            reqs = batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
    else:
        reqs = batch_isend_irecv(ops)
        for req in reqs:
            req.wait()


def _execute_p2p_ops_round(
    send_ops: list[P2POp],
    recv_ops: list[P2POp],
    *,
    op_batch_size: int | None,
    ep_rank: int,
    rank_to_global: dict[int, int],
    cuda_stream: torch.cuda.Stream | None,
) -> None:
    """
    Execute P2P operations for a communication round.

    When op_batch_size is set, uses rank ordering to avoid deadlocks:
    - Phase 1: lower rank sends (batches if needed), higher rank receives
    - Phase 2: lower rank receives (batches if needed), higher rank sends
    Otherwise, merges sends and recvs and executes them together.

    Args:
        op_batch_size: Number of P2P operations to process per batch.
    """
    if not send_ops and not recv_ops:
        return
    if op_batch_size is None:
        _execute_batch(send_ops + recv_ops, cuda_stream)
        return

    # When op_batch_size is set, each rank communicates with only one peer per round.
    # We need special handling to avoid deadlocks.
    # Validate that all operations are for the same peer.
    peer_ranks = set()
    for op in send_ops:
        peer_ranks.add(op.peer)
    for op in recv_ops:
        peer_ranks.add(op.peer)

    assert len(peer_ranks) == 1, (
        f"Rank {ep_rank}: All send and recv operations in a round "
        f"must be for the same peer. Found {len(peer_ranks)} "
        f"different peers: {peer_ranks}"
    )

    # Find the peer rank - should be the same for all ops in this round.
    my_global_rank = rank_to_global[ep_rank]
    peer_rank = peer_ranks.pop()

    # Determine if we should send first based on having smaller rank.
    send_first = my_global_rank < peer_rank

    # Phase 1: Lower rank sends, higher rank receives.
    if send_first:
        # We are the lower rank - send first.
        for i in range(0, len(send_ops), op_batch_size):
            _execute_batch(send_ops[i : i + op_batch_size], cuda_stream)
    else:
        # We are the higher rank - receive first.
        for i in range(0, len(recv_ops), op_batch_size):
            _execute_batch(recv_ops[i : i + op_batch_size], cuda_stream)

    # Phase 2: Lower rank receives, higher rank sends.
    if send_first:
        # We are the lower rank - receive second.
        for i in range(0, len(recv_ops), op_batch_size):
            _execute_batch(recv_ops[i : i + op_batch_size], cuda_stream)
    else:
        # We are the higher rank - send second.
        for i in range(0, len(send_ops), op_batch_size):
            _execute_batch(send_ops[i : i + op_batch_size], cuda_stream)


def move_to_buffer(
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffers: Sequence[torch.Tensor],
    cuda_stream: torch.cuda.Stream | None,
    ep_group: ProcessGroup,
    communication_config: EPLBCommunicationConfig,
) -> MoveToBufferResult:
    """
    Rearranges expert weights during EPLB rebalancing.

    Args:
        num_local_experts: Number of local experts.
        old_indices: (num_experts_total,) ndarray of current (old)
            global-to-local expert assignments.
        new_indices: (num_experts_total,) ndarray of desired (new)
            global-to-local assignments after rebalance.
        expert_weights: Original expert weights for the layer.
        expert_weights_buffers: Intermediate buffers (one per tensor).
        cuda_stream: CUDA stream for async copies (can be None for sync mode).
        ep_group: Distributed process group for expert parallel comms.
        communication_config: Communication configuration for P2P operations.

    Returns:
        is_unchanged (np.ndarray): (num_local_experts,), True where an expert row
            is unchanged after rebalance.
        is_received_locally (np.ndarray): (num_local_experts,), True where a row
            can be updated from local data.
        RecvMetadata: Metadata needed for completing remote weight transfers.
    """
    assert old_indices.shape == new_indices.shape
    ep_rank = ep_group.rank()

    recv_primary_mask = np.zeros((num_local_experts,), dtype=np.bool_)
    send_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    send_src_rows = np.full((num_local_experts,), -1, dtype=np.int32)
    recv_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    recv_dst_rows = np.full((num_local_experts,), -1, dtype=np.int32)

    base = ep_rank * num_local_experts
    local_rows = np.arange(num_local_experts, dtype=np.int32)
    local_global = base + local_rows

    old_local_expert_ids = old_indices[local_global]
    new_local_expert_ids = new_indices[local_global]

    # Unchanged mask
    is_unchanged = old_local_expert_ids == new_local_expert_ids

    # Local receive eligibility
    new_valid = new_local_expert_ids != -1
    can_recv_local = np.isin(
        new_local_expert_ids, old_local_expert_ids, assume_unique=False
    )
    is_received_locally = np.logical_or(
        is_unchanged, np.logical_and(new_valid, can_recv_local)
    )

    # Send map: first src row per unique expert present locally in old mapping
    send_count = 0
    valid_old = old_local_expert_ids != -1
    if np.any(valid_old):
        uniq_experts, first_idx = np.unique(
            old_local_expert_ids[valid_old], return_index=True
        )
        filtered_rows = local_rows[valid_old]
        src_rows = filtered_rows[first_idx]
        send_count = int(uniq_experts.shape[0])
        send_expert_ids[:send_count] = uniq_experts
        send_src_rows[:send_count] = src_rows

    # Recv map: primary dst per unique expert needed remotely
    recv_count = 0
    need_recv_mask = np.logical_and(~is_received_locally, new_valid)
    if np.any(need_recv_mask):
        desired_experts = new_local_expert_ids[need_recv_mask]
        desired_dsts = local_rows[need_recv_mask]
        uniq_recv_experts, uniq_indices = np.unique(desired_experts, return_index=True)
        dst_rows = desired_dsts[uniq_indices]
        recv_count = int(uniq_recv_experts.shape[0])
        recv_expert_ids[:recv_count] = uniq_recv_experts
        recv_dst_rows[:recv_count] = dst_rows
        recv_primary_mask[dst_rows] = True

    eligible_local_buffer_mask = np.logical_and(~is_unchanged, is_received_locally)

    # 1. Local moves into tmp buffers
    if bool(eligible_local_buffer_mask.any()) and send_count > 0:
        dest_indices = np.nonzero(eligible_local_buffer_mask)[0].tolist()
        expert_to_src_map = dict(
            zip(send_expert_ids[:send_count], send_src_rows[:send_count])
        )
        for dst in dest_indices:
            expert = new_local_expert_ids[dst]
            src_local = expert_to_src_map.get(expert, -1)
            if src_local != -1:
                for w, b in zip(expert_weights, expert_weights_buffers):
                    b[dst].copy_(w[src_local], non_blocking=True)

    # 2. Build communication plan (routing)
    comm_plan = _build_communication_plan(
        num_local_experts=num_local_experts,
        old_indices=old_indices,
        new_indices=new_indices,
        expert_weights=expert_weights,
        ep_group=ep_group,
        communication_config=communication_config,
        send_expert_ids=send_expert_ids,
        send_src_rows=send_src_rows,
        send_count=send_count,
        recv_expert_ids=recv_expert_ids,
        recv_dst_rows=recv_dst_rows,
        recv_count=recv_count,
        expert_weights_buffers=expert_weights_buffers,
    )

    # 3. Execute the P2P operations in multiple rounds
    # Each round handles communication between groups where
    # (group1 XOR group2) == round_idx
    # Round 0: Intra-group (same group)
    # Round i (i>0): Inter-group with XOR = i

    # Convert expert batch size to operation batch size
    experts_batch_size = communication_config.experts_batch_size
    op_batch_size = (
        experts_batch_size * comm_plan.ops_per_expert
        if experts_batch_size is not None
        else None
    )

    for round_idx in range(comm_plan.num_rounds):
        _execute_p2p_ops_round(
            comm_plan.send_ops_per_round[round_idx],
            comm_plan.recv_ops_per_round[round_idx],
            op_batch_size=op_batch_size,
            ep_rank=ep_rank,
            rank_to_global=comm_plan.rank_to_global,
            cuda_stream=cuda_stream,
        )
        # Barrier to ensure all ranks complete this round before proceeding
        # This prevents ranks from getting out of sync across rounds
        torch.distributed.barrier(group=ep_group)

    # wait for the communication to finish
    return (
        is_unchanged,
        is_received_locally,
        RecvMetadata(
            recv_primary_mask=recv_primary_mask,
            recv_count=recv_count,
            recv_expert_ids=recv_expert_ids,
            recv_dst_rows=recv_dst_rows,
        ),
    )


def move_from_buffer(
    expert_weights: Sequence[torch.Tensor],
    expert_weights_buffers: list[torch.Tensor],
    is_unchanged: np.ndarray,
    is_received_locally: np.ndarray,
    recv_metadata: RecvMetadata,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None:
    """
    Copies expert weights from communication buffers back to the target weight tensors
    after EPLB rebalancing.

    Args:
        expert_weights: List of the actual MoE layer weights used in the execution.
        expert_weights_buffers: Intermediate buffers containing the experts weights
            after the transfer is completed.
        is_unchanged: (num_local_experts,), True where an expert row is unchanged.
        is_received_locally: (num_local_experts,), True where a row is updated locally.
        recv_metadata: RecvMetadata containing remote receive metadata.
        new_indices: (num_experts_total,) mapping from local rows to desired
            (possibly global) expert id, after rebalance.
        ep_rank: Rank of the process in the expert parallel group.
    """
    recv_primary_mask = recv_metadata.recv_primary_mask
    recv_count = recv_metadata.recv_count
    recv_expert_ids = recv_metadata.recv_expert_ids
    recv_dst_rows = recv_metadata.recv_dst_rows
    num_local_experts = is_unchanged.shape[0]

    # Mask for rows to copy back from buffers:
    # copy if locally received OR remote primary recv
    copy_mask = np.logical_or(is_received_locally, recv_primary_mask)
    dest_mask_np = np.logical_and(~is_unchanged, copy_mask)
    if bool(dest_mask_np.any()):
        dest_indices = np.nonzero(dest_mask_np)[0].tolist()
        for dst in dest_indices:
            for w, b in zip(expert_weights, expert_weights_buffers):
                w[dst].copy_(b[dst], non_blocking=True)

    if recv_count == 0:
        return

    # Duplicate remote received rows to non-primary duplicate dsts
    base = ep_rank * num_local_experts
    local_experts = new_indices[base + np.arange(num_local_experts, dtype=np.int32)]
    duplicate_mask = np.logical_and(
        np.logical_and(~is_unchanged, ~is_received_locally),
        np.logical_and(~recv_primary_mask, local_experts != -1),
    )
    # All received experts are unique in the destination, so no need to copy duplicates
    if not bool(duplicate_mask.any()):
        return

    dup_dst_rows = np.nonzero(duplicate_mask)[0]
    dup_experts = local_experts[dup_dst_rows]

    prim_experts = recv_expert_ids[:recv_count]
    prim_dsts = recv_dst_rows[:recv_count]
    order = np.argsort(prim_experts, kind="stable")
    prim_experts_sorted = prim_experts[order]
    prim_dsts_sorted = prim_dsts[order]
    pos = np.searchsorted(prim_experts_sorted, dup_experts)
    valid = np.logical_and(
        pos < prim_experts_sorted.shape[0],
        prim_experts_sorted[np.minimum(pos, prim_experts_sorted.shape[0] - 1)]
        == dup_experts,
    )
    if not bool(valid.any()):
        return

    matched_dst_rows = dup_dst_rows[valid]
    matched_src_rows = prim_dsts_sorted[pos[valid]]

    for dst, src in zip(matched_dst_rows.tolist(), matched_src_rows.tolist()):
        for w in expert_weights:
            w[dst].copy_(w[src], non_blocking=True)


async def transfer_layer(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    communication_config: EPLBCommunicationConfig,
    is_profile: bool = False,
    layer: int = 0,
    cuda_stream: torch.cuda.Stream | None = None,
    rank_mapping: dict[int, int] | None = None,
) -> MoveToBufferResult:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        expert_weights: A sequence of shape (num_moe_layers)(weight_count)
            of tensors of shape (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection,
            so weight_count = 2. Each weight's hidden size can be different.
        ep_group: The device process group for expert parallelism.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.

    Returns:
        is_unchanged (np.ndarray): (1, num_local_experts), True where expert
            is left unchanged.
        is_received_locally (np.ndarray): (1, num_local_experts), True where expert
            can be received locally.
        RecvMetadata: Metadata needed for completing remote weight transfers.
    """
    ep_size = ep_group.size()
    if rank_mapping is not None:
        if len(rank_mapping) == ep_group.size():
            # scale down
            new_global_expert_indices = _map_new_expert_indices_with_rank_mapping(
                new_global_expert_indices,
                rank_mapping,
            )
        else:
            # scale up
            old_global_expert_indices = _map_old_expert_indices_with_rank_mapping(
                old_global_expert_indices,
                rank_mapping,
                ep_group.size(),
            )

    assert old_global_expert_indices.shape[1] == new_global_expert_indices.shape[1]
    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers
    assert len(expert_weights[0]) >= 1
    num_local_physical_experts = expert_weights[0][0].shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers, num_physical_experts)
    assert num_physical_experts == ep_size * num_local_physical_experts

    old_global_expert_indices_np = old_global_expert_indices.cpu().numpy()
    new_global_expert_indices_np = new_global_expert_indices.cpu().numpy()

    is_unchanged, is_received_locally, recv_metadata = move_to_buffer(
        num_local_experts=num_local_physical_experts,
        old_indices=old_global_expert_indices_np[layer],
        new_indices=new_global_expert_indices_np[layer],
        expert_weights=expert_weights[layer],
        expert_weights_buffers=expert_weights_buffer,
        cuda_stream=cuda_stream,
        ep_group=ep_group,
        communication_config=communication_config,
    )
    return is_unchanged, is_received_locally, recv_metadata


def rearrange_expert_weights_inplace(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    ep_group: ProcessGroup,
    communication_config: EPLBCommunicationConfig,
    is_profile: bool = False,
    rank_mapping: dict[int, int] | None = None,
) -> None:
    """
    Rearranges the expert weights in place according to the new expert indices.

    The value of the indices arguments are logical indices of the experts,
    while keys are physical.

    Args:
        old_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        new_global_expert_indices: Shape (num_moe_layers, num_physical_experts).
        expert_weights: A sequence of shape (num_moe_layers)(weight_count)
            of tensors of shape (num_local_physical_experts, hidden_size_i).
            For example, a linear layer may have up and down projection,
            so weight_count = 2. Each weight's hidden size can be different.
        ep_group: The device process group for expert parallelism.
        is_profile (bool): If `True`, do not perform any actual weight copy.
            This is used during profile run, where we only perform dummy
            communications to reserve enough memory for the buffers.
        rank_mapping: A dictionary mapping old rank to new rank.
        communication_config: Communication configuration for P2P operations.
    """
    if rank_mapping is not None:
        if len(rank_mapping) == ep_group.size():
            # scale down
            new_global_expert_indices = _map_new_expert_indices_with_rank_mapping(
                new_global_expert_indices,
                rank_mapping,
            )
        else:
            # scale up
            old_global_expert_indices = _map_old_expert_indices_with_rank_mapping(
                old_global_expert_indices,
                rank_mapping,
                ep_group.size(),
            )

    assert old_global_expert_indices.shape[1] == new_global_expert_indices.shape[1]

    num_moe_layers, num_physical_experts = old_global_expert_indices.shape
    assert len(expert_weights) == num_moe_layers
    assert len(expert_weights[0]) >= 1

    num_local_physical_experts = expert_weights[0][0].shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers, num_physical_experts)

    ep_size = ep_group.size()
    assert num_physical_experts == ep_size * num_local_physical_experts

    first_layer_weights = list(expert_weights[0])
    # Buffers to hold the expert weights during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.
    weights_buffer: list[torch.Tensor] = [
        torch.empty_like(w) for w in first_layer_weights
    ]
    if is_profile:
        # Reserve communication buffers via a minimal dummy all_gather on first layer
        for weight, buffer in zip(expert_weights[0], weights_buffer):
            dummy_recv_buffer = [buffer for _ in range(ep_size)]
            torch.distributed.barrier()
            all_gather(
                dummy_recv_buffer,
                weight,
                group=ep_group,
            )
        return

    # NOTE(bowen): We need this synchronize to run, but I don't know why.
    # If you figure out the reason, please let me know -- thank you!
    torch.cuda.synchronize()

    old_global_expert_indices_cpu = old_global_expert_indices.cpu().numpy()
    new_global_expert_indices_cpu = new_global_expert_indices.cpu().numpy()

    for layer_idx in range(num_moe_layers):
        is_unchanged, is_received_locally, recv_metadata = move_to_buffer(
            num_local_experts=num_local_physical_experts,
            old_indices=old_global_expert_indices_cpu[layer_idx],
            new_indices=new_global_expert_indices_cpu[layer_idx],
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=weights_buffer,
            cuda_stream=None,
            ep_group=ep_group,
            communication_config=communication_config,
        )

        move_from_buffer(
            expert_weights=expert_weights[layer_idx],
            expert_weights_buffers=weights_buffer,
            is_unchanged=is_unchanged,
            is_received_locally=is_received_locally,
            recv_metadata=recv_metadata,
            new_indices=new_global_expert_indices_cpu[layer_idx],
            ep_rank=ep_group.rank(),
        )


def _map_old_expert_indices_with_rank_mapping(
    old_global_expert_indices: torch.Tensor,
    rank_mapping: dict[int, int],
    new_ep_size: int,
) -> torch.Tensor:
    """
    Map the old global expert indices to the new global expert indices.

    Args:
        old_global_expert_indices:
            Shape (num_layers, old_ep_size * num_local_physical_experts).
        rank_mapping: Mapping from old rank to new rank.
        new_ep_size: New expert parallelism size.

    Returns:
        Mapped expert indices with shape
        (num_layers, new_ep_size * num_local_physical_experts).
    """
    num_layers, old_num_physical_experts = old_global_expert_indices.shape
    assert rank_mapping, "Rank mapping is required"

    # Get sizes from parameters and rank_mapping
    old_ep_size = len(rank_mapping)
    num_local_physical_experts = old_num_physical_experts // old_ep_size
    new_num_physical_experts = new_ep_size * num_local_physical_experts

    # Create mapped tensor with new shape, initialized to -1
    mapped_expert_indices = torch.full(
        (num_layers, new_num_physical_experts),
        fill_value=-1,
        dtype=old_global_expert_indices.dtype,
        device=old_global_expert_indices.device,
    )

    # Handle rank mapping (scale up/down with rank changes)
    for old_rank in range(old_ep_size):
        new_rank = rank_mapping.get(old_rank)
        if new_rank is not None and new_rank >= 0 and new_rank < new_ep_size:
            # This old rank exists in the new configuration
            old_start_idx = old_rank * num_local_physical_experts
            old_end_idx = (old_rank + 1) * num_local_physical_experts
            new_start_idx = new_rank * num_local_physical_experts
            new_end_idx = (new_rank + 1) * num_local_physical_experts

            mapped_expert_indices[:, new_start_idx:new_end_idx] = (
                old_global_expert_indices[:, old_start_idx:old_end_idx]
            )
        # If new_rank is None or >= new_ep_size, the experts remain -1
        # (scale down case)

    return mapped_expert_indices


def _map_new_expert_indices_with_rank_mapping(
    new_global_expert_indices: torch.Tensor,
    rank_mapping: dict[int, int],
) -> torch.Tensor:
    num_layers, new_num_physical_experts = new_global_expert_indices.shape
    assert rank_mapping, "Rank mapping is required"

    # Get sizes from parameters and rank_mapping
    old_ep_size = len(rank_mapping)
    new_ep_size = sum(new_rank != -1 for new_rank in rank_mapping.values())
    num_local_physical_experts = new_num_physical_experts // new_ep_size
    old_num_physical_experts = old_ep_size * num_local_physical_experts

    mapped_expert_indices = torch.full(
        (num_layers, old_num_physical_experts),
        fill_value=-1,
        dtype=new_global_expert_indices.dtype,
        device=new_global_expert_indices.device,
    )

    for old_rank in range(old_ep_size):
        new_rank = rank_mapping[old_rank]
        if new_rank >= 0 and new_rank < new_ep_size:
            old_start_idx = old_rank * num_local_physical_experts
            old_end_idx = (old_rank + 1) * num_local_physical_experts
            new_start_idx = new_rank * num_local_physical_experts
            new_end_idx = (new_rank + 1) * num_local_physical_experts

            mapped_expert_indices[:, old_start_idx:old_end_idx] = (
                new_global_expert_indices[:, new_start_idx:new_end_idx]
            )

    return mapped_expert_indices


__all__ = ["transfer_layer", "move_from_buffer", "RecvMetadata"]
