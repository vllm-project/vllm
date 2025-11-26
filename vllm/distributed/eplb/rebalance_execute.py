# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_gather,
    batch_isend_irecv,
    get_global_rank,
)

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


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


def move_to_buffer(
    num_local_experts: int,
    old_indices_group: np.ndarray,
    new_indices_group: np.ndarray,
    expert_weights_group: Sequence[Iterable[torch.Tensor]],
    buffers_group: Sequence[Sequence[torch.Tensor]],
    cuda_stream: torch.cuda.Stream | None,
    ep_group: ProcessGroup,
) -> tuple[
    np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Perform expert weights rearrangement of a group of layers.
    """
    assert len(old_indices_group) == len(new_indices_group) == len(expert_weights_group)
    group_size = len(old_indices_group)
    ep_rank = ep_group.rank()

    # Pre-allocate per-layer compact maps/masks (numpy)
    is_unchanged = np.zeros((group_size, num_local_experts), dtype=np.bool_)
    is_received_locally = np.zeros((group_size, num_local_experts), dtype=np.bool_)
    recv_primary_mask = np.zeros((group_size, num_local_experts), dtype=np.bool_)
    # Cache desired new expert ids per local row, for all layers
    new_local_expert_ids_mat = np.full(
        (group_size, num_local_experts), -1, dtype=np.int64
    )
    send_counts = np.zeros(group_size, dtype=np.int32)
    send_expert_ids = np.full((group_size, num_local_experts), -1, dtype=np.int64)
    send_src_rows = np.full((group_size, num_local_experts), -1, dtype=np.int32)
    recv_counts = np.zeros(group_size, dtype=np.int32)
    recv_expert_ids = np.full((group_size, num_local_experts), -1, dtype=np.int64)
    recv_dst_rows = np.full((group_size, num_local_experts), -1, dtype=np.int32)
    base = ep_rank * num_local_experts
    local_rows = np.arange(num_local_experts, dtype=np.int32)
    local_global = base + local_rows

    # Build masks and expert maps per layer
    for layer_idx in range(group_size):
        old_indices = old_indices_group[layer_idx]
        layer_new_indices = new_indices_group[layer_idx]

        old_local_expert_ids = old_indices[local_global]
        new_local_expert_ids = layer_new_indices[local_global]
        new_local_expert_ids_mat[layer_idx, :] = new_local_expert_ids

        # Unchanged per-dst mask
        unchanged_mask = old_local_expert_ids == new_local_expert_ids
        is_unchanged[layer_idx, :] = unchanged_mask

        # Local receive eligibility
        new_valid = new_local_expert_ids != -1
        can_recv_local = np.isin(
            new_local_expert_ids, old_local_expert_ids, assume_unique=False
        )
        is_local_recv = np.logical_or(
            unchanged_mask, np.logical_and(new_valid, can_recv_local)
        )
        is_received_locally[layer_idx, :] = is_local_recv

        # Send map: first src row per unique expert present locally in old mapping
        valid_old = old_local_expert_ids != -1
        if np.any(valid_old):
            uniq_experts, first_idx = np.unique(
                old_local_expert_ids[valid_old], return_index=True
            )
            filtered_rows = local_rows[valid_old]
            src_rows = filtered_rows[first_idx]
            layer_send_count = int(uniq_experts.shape[0])
            send_counts[layer_idx] = layer_send_count
            send_expert_ids[layer_idx, :layer_send_count] = uniq_experts
            send_src_rows[layer_idx, :layer_send_count] = src_rows
        else:
            send_counts[layer_idx] = 0

        # Recv map: primary dst per unique expert needed remotely
        need_recv_mask = np.logical_and(~is_local_recv, new_valid)
        if np.any(need_recv_mask):
            desired_experts = new_local_expert_ids[need_recv_mask]
            desired_dsts = local_rows[need_recv_mask]
            uniq_recv_experts, uniq_indices = np.unique(
                desired_experts, return_index=True
            )
            dst_rows = desired_dsts[uniq_indices]
            layer_send_count = int(uniq_recv_experts.shape[0])
            recv_counts[layer_idx] = layer_send_count
            recv_expert_ids[layer_idx, :layer_send_count] = uniq_recv_experts
            recv_dst_rows[layer_idx, :layer_send_count] = dst_rows
            recv_primary_mask[layer_idx, dst_rows] = True
        else:
            recv_counts[layer_idx] = 0

    # Precompute per-layer destination mask that actually needs local buffering:
    # need change, received locally, and valid target expert id
    eligible_local_buffer_mask = np.logical_and(
        np.logical_and(~is_unchanged, is_received_locally),
        new_local_expert_ids_mat != -1,
    )

    # 1. Local moves into tmp buffers
    for layer_idx in range(group_size):
        layer_send_count = int(send_counts[layer_idx])
        if layer_send_count <= 0:
            continue

        layer_send_experts = send_expert_ids[layer_idx, :layer_send_count]
        layer_send_srcs = send_src_rows[layer_idx, :layer_send_count]
        layer_weights_list = list(expert_weights_group[layer_idx])
        layer_buffers_list = list(buffers_group[layer_idx])
        new_local_expert_ids = new_local_expert_ids_mat[layer_idx, :]

        # Only consider destination rows that are eligible for local buffering
        eligible_mask = eligible_local_buffer_mask[layer_idx, :]
        if not bool(eligible_mask.any()):
            continue

        dest_indices = np.nonzero(eligible_mask)[0].tolist()
        # Build a map from expert_id to its source row.
        expert_to_src_map = dict(zip(layer_send_experts, layer_send_srcs))

        for dst in dest_indices:
            expert = new_local_expert_ids[dst]
            src_local = expert_to_src_map.get(expert, -1)
            if src_local != -1:
                for w, b in zip(layer_weights_list, layer_buffers_list):
                    b[dst].copy_(w[src_local])

    p2p_ops: list[P2POp] = []

    # Pre-compute global ranks mapping
    ep_size = ep_group.size()
    rank_to_global = {rank: get_global_rank(ep_group, rank) for rank in range(ep_size)}

    # 2. Post sends per layer
    for layer_idx in range(group_size):
        old_indices = old_indices_group[layer_idx]
        layer_new_indices = new_indices_group[layer_idx]
        layer_weights_list = list(expert_weights_group[layer_idx])
        layer_send_count = int(send_counts[layer_idx])
        if layer_send_count == 0:
            continue
        experts = send_expert_ids[layer_idx, :layer_send_count]
        srcs = send_src_rows[layer_idx, :layer_send_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        srcs = srcs[order]

        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts,
            num_local_experts,
            old_indices,
            layer_new_indices,
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
                p2p_ops += [
                    P2POp(
                        torch.distributed.isend,
                        w[src],
                        dst_global,
                    )
                    for w in layer_weights_list
                ]

    # 3. Post recvs per layer
    for layer_idx in range(group_size):
        old_indices = old_indices_group[layer_idx]
        layer_new_indices = new_indices_group[layer_idx]
        layer_buffers_list = list(buffers_group[layer_idx])
        layer_recv_count = int(recv_counts[layer_idx])
        if layer_recv_count == 0:
            continue
        experts = recv_expert_ids[layer_idx, :layer_recv_count]
        dsts = recv_dst_rows[layer_idx, :layer_recv_count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        dsts = dsts[order]

        # Batch query all experts for this layer
        send_map, recv_map = get_ep_ranks_with_experts_batch(
            experts,
            num_local_experts,
            old_indices,
            layer_new_indices,
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
            p2p_ops += [
                P2POp(
                    torch.distributed.irecv,
                    b[dst],
                    src_global,
                )
                for b in layer_buffers_list
            ]

    # 4. Execute the P2P operations. The real communication happens here.
    if p2p_ops and cuda_stream is not None:
        with torch.cuda.stream(cuda_stream):
            reqs = batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()
    elif p2p_ops:
        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()
    # wait for the communication to finish
    return (
        is_unchanged,
        is_received_locally,
        (recv_primary_mask, recv_counts, recv_expert_ids, recv_dst_rows),
    )


def move_from_buffer(
    weights_group: Sequence[Iterable[torch.Tensor]],
    buffers_group: Sequence[Sequence[torch.Tensor]],
    is_unchanged: np.ndarray,
    is_received_locally: np.ndarray,
    recv_metadata: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    new_indices_group: np.ndarray,
    ep_group: ProcessGroup,
) -> None:
    assert (
        len(weights_group)
        == len(buffers_group)
        == len(is_unchanged)
        == len(is_received_locally)
        == len(recv_metadata[0])
        == len(new_indices_group)
    ), "Unmatching layer group size"
    ep_rank = ep_group.rank()
    group_size = len(is_unchanged)
    recv_primary_mask, recv_counts, recv_expert_ids, recv_dst_rows = recv_metadata
    num_local_experts = is_unchanged.shape[1]
    # Mask for rows to copy back from buffers:
    # copy if locally received OR remote primary recv
    copy_mask = np.logical_or(is_received_locally, recv_primary_mask)
    # Copy back local buffered rows into destination weights
    for layer_idx in range(group_size):
        layer_is_unchanged = is_unchanged[layer_idx, :]
        layer_copy_mask = copy_mask[layer_idx, :]
        weights_list = list(weights_group[layer_idx])
        buffers_list = list(buffers_group[layer_idx])
        # rows to copy = (~unchanged) & copy_mask
        dest_mask_np = np.logical_and(~layer_is_unchanged, layer_copy_mask)
        if not bool(dest_mask_np.any()):
            continue
        dest_indices = np.nonzero(dest_mask_np)[0].tolist()
        for dst in dest_indices:
            for w, b in zip(weights_list, buffers_list):
                w[dst].copy_(b[dst])

    # Duplicate remote received rows to non-primary duplicate dsts
    for layer_idx in range(group_size):
        layer_is_unchanged = is_unchanged[layer_idx, :]
        layer_is_received_locally = is_received_locally[layer_idx, :]
        new_indices = new_indices_group[layer_idx]
        weights_list = list(weights_group[layer_idx])

        count_recv = int(recv_counts[layer_idx])
        if count_recv == 0:
            # No remote primaries on this layer → no remote duplicates to materialize
            continue
        # Local view of desired expert ids per local row
        base = ep_rank * num_local_experts
        local_experts = new_indices[base + np.arange(num_local_experts, dtype=np.int32)]
        # Duplicate rows mask: need remote, not primary, and valid expert id
        duplicate_mask = np.logical_and(
            np.logical_and(~layer_is_unchanged, ~layer_is_received_locally),
            np.logical_and(~recv_primary_mask[layer_idx, :], local_experts != -1),
        )
        if not bool(duplicate_mask.any()):
            continue
        dup_dst_rows = np.nonzero(duplicate_mask)[0]
        dup_experts = local_experts[dup_dst_rows]

        # Build primary mapping arrays (expert -> primary dst) and vector-match
        prim_experts = recv_expert_ids[layer_idx, :count_recv]
        prim_dsts = recv_dst_rows[layer_idx, :count_recv]
        order = np.argsort(prim_experts, kind="stable")
        prim_experts_sorted = prim_experts[order]
        prim_dsts_sorted = prim_dsts[order]
        pos = np.searchsorted(prim_experts_sorted, dup_experts)
        # Filter to experts that have a matching primary entry
        valid = np.logical_and(
            pos < prim_experts_sorted.shape[0],
            prim_experts_sorted[np.minimum(pos, prim_experts_sorted.shape[0] - 1)]
            == dup_experts,
        )
        if not bool(valid.any()):
            continue
        matched_dst_rows = dup_dst_rows[valid]
        matched_src_rows = prim_dsts_sorted[pos[valid]]

        # Perform row copies per (dst, src) pair without tensor indexing
        for dst, src in zip(matched_dst_rows.tolist(), matched_src_rows.tolist()):
            for w in weights_list:
                w[dst].copy_(w[src])


async def transfer_layer(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    is_profile: bool = False,
    layer: int = 0,
    cuda_stream: torch.cuda.Stream | None = None,
    rank_mapping: dict[int, int] | None = None,
) -> tuple[
    np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
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
    num_local_physical_experts = next(iter(expert_weights[0])).shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers, num_physical_experts)
    assert num_physical_experts == ep_size * num_local_physical_experts

    old_global_expert_indices_np = old_global_expert_indices.cpu().numpy()
    new_global_expert_indices_np = new_global_expert_indices.cpu().numpy()

    is_unchanged, is_received_locally, recv_metadata = move_to_buffer(
        num_local_experts=num_local_physical_experts,
        old_indices_group=old_global_expert_indices_np[layer : layer + 1],
        new_indices_group=new_global_expert_indices_np[layer : layer + 1],
        expert_weights_group=[expert_weights[layer]],
        buffers_group=[expert_weights_buffer],
        cuda_stream=cuda_stream,
        ep_group=ep_group,
    )
    return is_unchanged, is_received_locally, recv_metadata


def rearrange_expert_weights_inplace(
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    expert_weights: Sequence[Iterable[torch.Tensor]],
    ep_group: ProcessGroup,
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

    num_local_physical_experts = next(iter(expert_weights[0])).shape[0]
    assert new_global_expert_indices.shape == (num_moe_layers, num_physical_experts)

    ep_size = ep_group.size()
    assert num_physical_experts == ep_size * num_local_physical_experts

    # Max number of layers to group for communication
    max_group_layers = envs.VLLM_EPLB_SYNC_MAX_GROUPED_LAYERS
    max_group_layers = max(min(max_group_layers, num_moe_layers), 1)

    first_layer_weights = list(expert_weights[0])
    # Buffers to hold the expert weights during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.
    weights_buffers: list[list[torch.Tensor]] = [
        [torch.empty_like(w) for w in first_layer_weights]
        for _ in range(max_group_layers)
    ]
    if is_profile:
        # Reserve communication buffers via a minimal dummy all_gather on first layer
        for layer_idx in range(max_group_layers):
            for weight, buffer in zip(expert_weights[0], weights_buffers[layer_idx]):
                dummy_recv_buffer = [buffer for _ in range(ep_size)]
                torch.distributed.barrier()
                all_gather(
                    dummy_recv_buffer,
                    weight,
                    group=ep_group,
                )
        return
    logger.info_once(
        f"EPLB Sync: rearrange max_group_layers: {max_group_layers}", scope="global"
    )

    # NOTE(bowen): We need this synchronize to run, but I don't know why.
    # If you figure out the reason, please let me know -- thank you!
    torch.cuda.synchronize()

    old_global_expert_indices_cpu = old_global_expert_indices.cpu().numpy()
    new_global_expert_indices_cpu = new_global_expert_indices.cpu().numpy()

    start = 0
    while start < num_moe_layers:
        end = min(start + max_group_layers, num_moe_layers)
        old_group = old_global_expert_indices_cpu[start:end]
        new_group = new_global_expert_indices_cpu[start:end]
        weights_group = [expert_weights[i] for i in range(start, end)]
        buffers_group = weights_buffers[: (end - start)]

        is_unchanged, is_received_locally, recv_metadata = move_to_buffer(
            num_local_experts=num_local_physical_experts,
            old_indices_group=old_group,
            new_indices_group=new_group,
            expert_weights_group=weights_group,
            buffers_group=buffers_group,
            cuda_stream=None,
            ep_group=ep_group,
        )

        move_from_buffer(
            weights_group=weights_group,
            buffers_group=buffers_group,
            is_unchanged=is_unchanged,
            is_received_locally=is_received_locally,
            recv_metadata=recv_metadata,
            new_indices_group=new_group,
            ep_group=ep_group,
        )
        start = end


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


__all__ = ["transfer_layer", "move_from_buffer"]
