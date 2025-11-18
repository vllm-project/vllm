# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

import os
from collections.abc import Iterable, MutableSequence, Sequence
from functools import partial

import numpy as np
import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_gather,
    batch_isend_irecv,
    get_global_rank,
)

from vllm.logger import init_logger

logger = init_logger(__name__)


def idx_local_to_global(
    local_idx: int,
    local_cnt: int,
    ep_rank: int,
) -> int:
    """
    Convert a local expert index to a global expert index.
    """
    return ep_rank * local_cnt + local_idx


def idx_global_to_local(
    global_idx: int,
    local_cnt: int,
    ep_rank: int,
) -> int:
    """
    Convert a global expert index to a local expert index.
    """
    return global_idx - ep_rank * local_cnt


def global_idx_to_rank(
    global_idx: int,
    local_cnt: int,
) -> int:
    """
    Convert a global expert index to a rank index.
    """
    return global_idx // local_cnt


def get_ep_ranks_with_expert(
    idx: int,
    num_local_experts: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
) -> tuple[MutableSequence[int], MutableSequence[int]]:
    """
    Get the ranks of the experts that need to be exchanged.

    Args:
        idx: The index of the expert.
        num_local_experts: The number of local experts.
        old_indices: The old indices of the experts.
        new_indices: The new indices of the experts.

    Returns:
        A tuple of two lists:
        - The ranks of the experts that need to be sent.
        - The ranks of the experts that need to be received.
    """
    global2rank = partial(
        global_idx_to_rank,
        local_cnt=num_local_experts,
    )

    ranks_to_send: list[int] = []
    ranks_to_recv: list[int] = []

    for i, e in enumerate(old_indices):
        if e == idx:
            rank = global2rank(i)
            if not ranks_to_send or ranks_to_send[-1] != rank:
                ranks_to_send.append(rank)

    for i, e in enumerate(new_indices):
        if e == idx:
            rank = global2rank(i)
            if not ranks_to_recv or ranks_to_recv[-1] != rank:
                ranks_to_recv.append(rank)

    # Remove those ranks that can get this expert locally.
    ranks_to_send_set = set(ranks_to_send)
    ranks_to_recv_actual = [
        rank for rank in ranks_to_recv if rank not in ranks_to_send_set
    ]

    return ranks_to_send, ranks_to_recv_actual


def old_shuffle_layer_np(
    num_local_experts: int,
    ep_rank: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
) -> None:
    """
    Numpy-based single-layer expert weight exchange.
    Mirrors old_shuffle_layer but uses numpy for index/mask computations
    and get_ep_ranks_with_expert_np for rank sets.
    """
    base = ep_rank * num_local_experts
    local_rows = np.arange(num_local_experts, dtype=np.int32)
    local_global = base + local_rows

    old_local_expert_ids = old_indices[local_global]
    new_local_expert_ids = new_indices[local_global]

    # 0) Unchanged per-dst mask
    is_unchanged = old_local_expert_ids == new_local_expert_ids

    # 1) Local receive eligibility and staging into buffers
    new_valid = new_local_expert_ids != -1
    can_recv_local = np.isin(
        new_local_expert_ids, old_local_expert_ids, assume_unique=False
    )
    is_received_locally = np.logical_or(
        is_unchanged, np.logical_and(new_valid, can_recv_local)
    )

    # Map expert -> first src row locally present in old mapping
    experts_send_loc: dict[int, int] = {}
    valid_old = old_local_expert_ids != -1
    if np.any(valid_old):
        uniq_experts, first_idx = np.unique(
            old_local_expert_ids[valid_old], return_index=True
        )
        filtered_rows = local_rows[valid_old]
        src_rows = filtered_rows[first_idx]
        for e, s in zip(uniq_experts.tolist(), src_rows.tolist()):
            experts_send_loc[int(e)] = int(s)

    # Stage local copies into buffers for eligible dsts (excluding unchanged)
    weights_list = list(expert_weights)
    buffers_list = list(expert_weights_buffer)
    for dst in range(num_local_experts):
        if is_unchanged[dst] or not is_received_locally[dst]:
            continue
        expert = int(new_local_expert_ids[dst])
        if expert == -1:
            continue
        src_local = experts_send_loc.get(expert)
        if src_local is None:
            continue
        for w, b in zip(weights_list, buffers_list):
            b[dst].copy_(w[src_local])

    # 2) Build P2P ops
    p2p_ops: list[P2POp] = []

    # 2a) Sends: for each local unique expert, determine peers to send to
    for expert, src in sorted(experts_send_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert_np(
            expert,
            num_local_experts,
            old_indices,
            new_indices,
        )
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
        for dst_rank in recv_ranks:
            dst_global = get_global_rank(ep_group, dst_rank)
            p2p_ops += [
                P2POp(
                    torch.distributed.isend,
                    w[src],
                    dst_global,
                )
                for w in weights_list
            ]

    # 3) Recvs: choose primary dst per expert that needs remote
    experts_recv_loc: dict[int, int] = {}
    for dst in range(num_local_experts):
        if is_received_locally[dst]:
            continue
        expert = int(new_local_expert_ids[dst])
        if expert == -1:
            continue
        if expert not in experts_recv_loc:
            experts_recv_loc[expert] = dst

    for expert, dst in sorted(experts_recv_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert_np(
            expert,
            num_local_experts,
            old_indices,
            new_indices,
        )
        if not ranks_to_send or not ranks_to_recv:
            continue
        num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
        recver_pos = ranks_to_recv.index(ep_rank)
        remainder_start = len(ranks_to_send) * num_dst_per_sender
        if recver_pos < remainder_start:
            src_rank = ranks_to_send[recver_pos // num_dst_per_sender]
        else:
            src_rank = ranks_to_send[recver_pos - remainder_start]
        src_global = get_global_rank(ep_group, src_rank)
        p2p_ops += [
            P2POp(
                torch.distributed.irecv,
                b[dst],
                src_global,
            )
            for b in buffers_list
        ]

    # 4) Execute P2P
    if p2p_ops:
        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    # 5) Copy back from buffers to weights
    for dst in range(num_local_experts):
        if is_unchanged[dst]:
            continue
        if is_received_locally[dst]:
            for w, b in zip(weights_list, buffers_list):
                w[dst].copy_(b[dst])
        else:
            expert = int(new_local_expert_ids[dst])
            if expert == -1:
                continue
            src_local = experts_recv_loc.get(expert)
            if src_local is None:
                continue
            for w, b in zip(weights_list, buffers_list):
                w[dst].copy_(b[src_local])


def get_ep_ranks_with_expert_np(
    idx: int,
    num_local_experts: int,
    old_indices: np.ndarray,
    new_indices: np.ndarray,
) -> tuple[list[int], list[int]]:
    """
    Get the ranks of the experts that need to be exchanged.

    Args:
        idx: The index of the expert.
        num_local_experts: The number of local experts.
        old_indices: The old indices of the experts.
        new_indices: The new indices of the experts.

    Returns:
        A tuple of two lists:
        - The ranks of the experts that need to be sent.
        - The ranks of the experts that need to be received.
    """
    # Indices where expert idx appears
    old_pos = np.nonzero(old_indices == idx)[0]
    new_pos = np.nonzero(new_indices == idx)[0]
    # Map positions to ranks
    if old_pos.size > 0:
        old_ranks = old_pos // num_local_experts
        uniq_send, first_idx_send = np.unique(old_ranks, return_index=True)
        order_send = np.argsort(first_idx_send)
        ranks_to_send = uniq_send[order_send].astype(int).tolist()
    else:
        ranks_to_send = []
    if new_pos.size > 0:
        new_ranks = new_pos // num_local_experts
        uniq_recv, first_idx_recv = np.unique(new_ranks, return_index=True)
        order_recv = np.argsort(first_idx_recv)
        ranks_to_recv = uniq_recv[order_recv].astype(int).tolist()
    else:
        ranks_to_recv = []
    # Remove ranks that have local copies to avoid unnecessary recv
    ranks_to_send_set = set(ranks_to_send)
    ranks_to_recv_actual = [r for r in ranks_to_recv if r not in ranks_to_send_set]
    return ranks_to_send, ranks_to_recv_actual


def old_shuffle_layer(
    num_local_experts: int,
    ep_rank: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
) -> None:
    """
    Perform expert weights rearrangement of one layer.
    """
    local2global = partial(
        idx_local_to_global,
        local_cnt=num_local_experts,
        ep_rank=ep_rank,
    )

    # 0. Do nothing for experts that did not change.
    is_unchanged = [
        old_indices[local2global(i)] == new_indices[local2global(i)]
        for i in range(num_local_experts)
    ]

    # 1. Perform weight copy inside the local rank.
    is_received_locally = is_unchanged[:]
    for src in range(num_local_experts):
        src_global = local2global(src)
        for dst in range(num_local_experts):
            dst_global = local2global(dst)
            if is_received_locally[dst]:
                continue
            if old_indices[src_global] == -1 or new_indices[dst_global] == -1:
                continue
            if old_indices[src_global] == new_indices[dst_global]:
                is_received_locally[dst] = True
                for weight, buffer in zip(expert_weights, expert_weights_buffer):
                    buffer[dst].copy_(weight[src])

    p2p_ops: list[P2POp] = []

    # 2. Initiate sending of weights.
    experts_send_loc: dict[int, int] = {}
    for src in range(num_local_experts):
        expert = old_indices[local2global(src)]
        if expert == -1:
            continue
        if expert in experts_send_loc:
            continue
        experts_send_loc[expert] = src

    # We need to sort here to match send/recv
    for expert, src in sorted(experts_send_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
            expert,
            num_local_experts,
            old_indices,
            new_indices,
        )

        # Calculate the ranks to send by this rank
        num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
        sender_pos = ranks_to_send.index(ep_rank)
        recv_begin = sender_pos * num_dst_per_sender
        recv_end = recv_begin + num_dst_per_sender
        recv_ranks = ranks_to_recv[recv_begin:recv_end]

        # Tackle remainders
        remainder_start = len(ranks_to_send) * num_dst_per_sender
        recver_pos = remainder_start + sender_pos
        if recver_pos < len(ranks_to_recv):
            recv_ranks.append(ranks_to_recv[recver_pos])

        for dst in recv_ranks:
            dst_global = get_global_rank(ep_group, dst)
            p2p_ops += [
                P2POp(
                    torch.distributed.isend,
                    weight[src],
                    dst_global,
                )
                for weight in expert_weights
            ]

    # 3. Initiate receiving of weights.
    experts_recv_loc: dict[int, int] = {}
    for dst in range(num_local_experts):
        if is_received_locally[dst]:
            continue
        expert = new_indices[local2global(dst)]
        if expert == -1:
            continue
        if expert in experts_recv_loc:
            continue
        experts_recv_loc[expert] = dst

    # We need to sort here to match send/recv
    for expert, dst in sorted(experts_recv_loc.items()):
        ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
            expert,
            num_local_experts,
            old_indices,
            new_indices,
        )

        # Calculate the rank to recv by this rank
        num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
        recver_pos = ranks_to_recv.index(ep_rank)
        remainder_start = len(ranks_to_send) * num_dst_per_sender
        if recver_pos < remainder_start:
            src = ranks_to_send[recver_pos // num_dst_per_sender]
        else:
            src = ranks_to_send[recver_pos - remainder_start]

        src_global = get_global_rank(ep_group, src)
        p2p_ops += [
            P2POp(
                torch.distributed.irecv,
                weight[dst],
                src_global,
            )
            for weight in expert_weights_buffer
        ]

    # 4. Execute the P2P operations. The real communication happens here.
    if p2p_ops:
        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    # 5. Copy the weights from the buffer back to the original weights.
    for dst in range(num_local_experts):
        if is_unchanged[dst]:
            continue
        if is_received_locally[dst]:
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[dst])
        else:
            expert = new_indices[local2global(dst)]
            if expert == -1:
                continue
            src = experts_recv_loc[expert]
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[src])


def old_shuffle_layers_grouped_np(
    num_local_experts: int,
    ep_rank: int,
    old_indices_group: np.ndarray,
    new_indices_group: np.ndarray,
    expert_weights_group: Sequence[Iterable[torch.Tensor]],
    buffers_group: Sequence[Sequence[torch.Tensor]],
    ep_group: ProcessGroup,
) -> None:
    """
    Grouped version using numpy inputs for indices. Prefers numpy operations
    for computing per-layer masks and expert mappings, then posts P2P ops.
    """
    assert len(old_indices_group) == len(new_indices_group) == len(expert_weights_group)
    group_size = len(old_indices_group)

    # Pre-allocate per-layer compact maps/masks (numpy)
    layer_is_unchanged = np.zeros((group_size, num_local_experts), dtype=np.bool_)
    layer_is_received_locally = np.zeros(
        (group_size, num_local_experts), dtype=np.bool_
    )
    recv_primary_mask = np.zeros((group_size, num_local_experts), dtype=np.bool_)
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
        new_indices = new_indices_group[layer_idx]

        old_local_expert_ids = old_indices[local_global]
        new_local_expert_ids = new_indices[local_global]

        # Unchanged per-dst mask
        unchanged_mask = old_local_expert_ids == new_local_expert_ids
        layer_is_unchanged[layer_idx, :] = unchanged_mask

        # Local receive eligibility
        new_valid = new_local_expert_ids != -1
        can_recv_local = np.isin(
            new_local_expert_ids, old_local_expert_ids, assume_unique=False
        )
        is_local_recv = np.logical_or(
            unchanged_mask, np.logical_and(new_valid, can_recv_local)
        )
        layer_is_received_locally[layer_idx, :] = is_local_recv

        # Send map: first src row per unique expert present locally in old mapping
        valid_old = old_local_expert_ids != -1
        if np.any(valid_old):
            uniq_experts, first_idx = np.unique(
                old_local_expert_ids[valid_old], return_index=True
            )
            filtered_rows = local_rows[valid_old]
            src_rows = filtered_rows[first_idx]
            count = int(uniq_experts.shape[0])
            send_counts[layer_idx] = count
            send_expert_ids[layer_idx, :count] = uniq_experts
            send_src_rows[layer_idx, :count] = src_rows
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
            count = int(uniq_recv_experts.shape[0])
            recv_counts[layer_idx] = count
            recv_expert_ids[layer_idx, :count] = uniq_recv_experts
            recv_dst_rows[layer_idx, :count] = dst_rows
            recv_primary_mask[layer_idx, dst_rows] = True
        else:
            recv_counts[layer_idx] = 0

    # Stage local moves into buffers
    for layer_idx in range(group_size):
        is_unchanged = layer_is_unchanged[layer_idx, :]
        is_received_locally = layer_is_received_locally[layer_idx, :]
        new_indices = new_indices_group[layer_idx]
        count = int(send_counts[layer_idx])
        layer_send_experts = send_expert_ids[layer_idx, :count]
        layer_send_srcs = send_src_rows[layer_idx, :count]
        local2global = partial(
            idx_local_to_global,
            local_cnt=num_local_experts,
            ep_rank=ep_rank,
        )
        weights_list = list(expert_weights_group[layer_idx])
        buffers_list = list(buffers_group[layer_idx])
        for dst in range(num_local_experts):
            if is_unchanged[dst] or not is_received_locally[dst]:
                continue
            dst_global = local2global(dst)
            expert = new_indices[dst_global]
            if expert == -1:
                continue
            matches = np.nonzero(layer_send_experts == expert)[0]
            if matches.size == 0:
                continue
            src_local = int(layer_send_srcs[matches[0]])
            for w, b in zip(weights_list, buffers_list):
                b[dst].copy_(w[src_local])

    # Build P2P ops across layers
    p2p_ops: list[P2POp] = []

    # Post sends per layer
    for layer_idx in range(group_size):
        old_indices = old_indices_group[layer_idx]
        new_indices = new_indices_group[layer_idx]
        weights_list = list(expert_weights_group[layer_idx])
        count = int(send_counts[layer_idx])
        if count == 0:
            continue
        experts = send_expert_ids[layer_idx, :count]
        srcs = send_src_rows[layer_idx, :count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        srcs = srcs[order]
        for expert, src in zip(experts.tolist(), srcs.tolist()):
            ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert_np(
                expert,
                num_local_experts,
                old_indices,
                new_indices,
            )
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
                dst_global = get_global_rank(ep_group, dst)
                p2p_ops += [
                    P2POp(
                        torch.distributed.isend,
                        w[src],
                        dst_global,
                    )
                    for w in weights_list
                ]

    # Post recvs per layer (primary destinations)
    for layer_idx in range(group_size):
        old_indices = old_indices_group[layer_idx]
        new_indices = new_indices_group[layer_idx]
        buffers_list = list(buffers_group[layer_idx])
        count = int(recv_counts[layer_idx])
        if count == 0:
            continue
        experts = recv_expert_ids[layer_idx, :count]
        dsts = recv_dst_rows[layer_idx, :count]
        order = np.argsort(experts, kind="stable")
        experts = experts[order]
        dsts = dsts[order]
        for expert, dst in zip(experts.tolist(), dsts.tolist()):
            ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert_np(
                expert,
                num_local_experts,
                old_indices,
                new_indices,
            )
            if not ranks_to_send or not ranks_to_recv:
                continue
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            recver_pos = ranks_to_recv.index(ep_rank)
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            if recver_pos < remainder_start:
                src = ranks_to_send[recver_pos // num_dst_per_sender]
            else:
                src = ranks_to_send[recver_pos - remainder_start]
            src_global = get_global_rank(ep_group, src)
            p2p_ops += [
                P2POp(
                    torch.distributed.irecv,
                    b[dst],
                    src_global,
                )
                for b in buffers_list
            ]

    # Execute merged P2P
    if p2p_ops:
        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    # Copy back from buffers for locals and primary remote recvs
    for layer_idx in range(group_size):
        is_unchanged = layer_is_unchanged[layer_idx, :]
        copy_mask = np.logical_or(
            layer_is_received_locally[layer_idx, :], recv_primary_mask[layer_idx, :]
        )
        weights_list = list(expert_weights_group[layer_idx])
        buffers_list = list(buffers_group[layer_idx])
        for dst in range(num_local_experts):
            if is_unchanged[dst] or not copy_mask[dst]:
                continue
            for w, b in zip(weights_list, buffers_list):
                w[dst].copy_(b[dst])


# def old_shuffle_layers_grouped(
#     num_local_experts: int,
#     ep_rank: int,
#     old_indices_group: Sequence[Sequence[int]],
#     new_indices_group: Sequence[Sequence[int]],
#     expert_weights_group: Sequence[Iterable[torch.Tensor]],
#     buffers_group: Sequence[Sequence[torch.Tensor]],
#     ep_group: ProcessGroup,
# ) -> None:
#     """
#     Multi-layer version of old_shuffle_layer that merges P2POps across
#     several layers and receives directly into destination rows.
#     This variant avoids numpy and mirrors the logic of old_shuffle_layer,
#     only batching send/recv ops across layers.
#     """
#     assert len(old_indices_group) == len(new_indices_group)
#     == len(expert_weights_group)
#     group_size = len(old_indices_group)

#     # Per-layer state
#     layer_is_unchanged: list[list[bool]] = []
#     layer_is_received_locally: list[list[bool]] = []
#     layer_experts_send_loc: list[dict[int, int]] = []
#     layer_experts_recv_loc: list[dict[int, int]] = []

#     # 0-1) Build unchanged and local-recv maps and stage local copies into buffers
#     for layer_idx in range(group_size):
#         old_indices = old_indices_group[layer_idx]
#         new_indices = new_indices_group[layer_idx]
#         weights_list = list(expert_weights_group[layer_idx])
#         buffers_list = list(buffers_group[layer_idx])

#         local2global = partial(
#             idx_local_to_global,
#             local_cnt=num_local_experts,
#             ep_rank=ep_rank,
#         )

#         # Unchanged mask
#         is_unchanged = [
#             old_indices[local2global(i)] == new_indices[local2global(i)]
#             for i in range(num_local_experts)
#         ]
#         layer_is_unchanged.append(is_unchanged)

#         # Local send candidates (unique expert -> source row)
#         experts_send_loc: dict[int, int] = {}
#         for src in range(num_local_experts):
#             expert = old_indices[local2global(src)]
#             if expert == -1:
#                 continue
#             if expert not in experts_send_loc:
#                 experts_send_loc[expert] = src
#         layer_experts_send_loc.append(experts_send_loc)

#         # Determine which dst rows can be satisfied locally
#         is_received_locally = is_unchanged[:]
#         for src in range(num_local_experts):
#             src_global = local2global(src)
#             for dst in range(num_local_experts):
#                 dst_global = local2global(dst)
#                 if is_received_locally[dst]:
#                     continue
#                 if old_indices[src_global] == -1 or new_indices[dst_global] == -1:
#                     continue
#                 if old_indices[src_global] == new_indices[dst_global]:
#                     is_received_locally[dst] = True
#                     # Stage local copy into buffers for the dst row
#                     for w, b in zip(weights_list, buffers_list):
#                         b[dst].copy_(w[src])
#         layer_is_received_locally.append(is_received_locally)

#         # Primary remote receive destinations (expert -> dst row)
#         experts_recv_loc: dict[int, int] = {}
#         for dst in range(num_local_experts):
#             if is_received_locally[dst]:
#                 continue
#             expert = new_indices[local2global(dst)]
#             if expert == -1:
#                 continue
#             if expert not in experts_recv_loc:
#                 experts_recv_loc[expert] = dst
#         layer_experts_recv_loc.append(experts_recv_loc)

#     # 2-3) Post all P2P ops across layers
#     p2p_ops: list[P2POp] = []

#     # 2) Sends per layer
#     for layer_idx in range(group_size):
#         old_indices = old_indices_group[layer_idx]
#         new_indices = new_indices_group[layer_idx]
#         weights_list = list(expert_weights_group[layer_idx])
#         experts_send_loc = layer_experts_send_loc[layer_idx]
#         for expert in sorted(experts_send_loc.keys()):
#             src = experts_send_loc[expert]
#             ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
#                 expert,
#                 num_local_experts,
#                 old_indices,
#                 new_indices,
#             )
#             if not ranks_to_send or not ranks_to_recv:
#                 continue
#             # Calculate the ranks to send by this rank
#             num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
#             sender_pos = ranks_to_send.index(ep_rank)
#             recv_begin = sender_pos * num_dst_per_sender
#             recv_end = recv_begin + num_dst_per_sender
#             recv_ranks = ranks_to_recv[recv_begin:recv_end]
#             # Tackle remainders
#             remainder_start = len(ranks_to_send) * num_dst_per_sender
#             recver_pos = remainder_start + sender_pos
#             if recver_pos < len(ranks_to_recv):
#                 recv_ranks.append(ranks_to_recv[recver_pos])
#             for dst_rank in recv_ranks:
#                 dst_global = get_global_rank(ep_group, dst_rank)
#                 p2p_ops += [
#                     P2POp(
#                         torch.distributed.isend,
#                         w[src],
#                         dst_global,
#                     )
#                     for w in weights_list
#                 ]

#     # 3) Recvs per layer
#     for layer_idx in range(group_size):
#         old_indices = old_indices_group[layer_idx]
#         new_indices = new_indices_group[layer_idx]
#         buffers_list = list(buffers_group[layer_idx])
#         experts_recv_loc = layer_experts_recv_loc[layer_idx]
#         for expert in sorted(experts_recv_loc.keys()):
#             dst = experts_recv_loc[expert]
#             ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
#                 expert,
#                 num_local_experts,
#                 old_indices,
#                 new_indices,
#             )
#             if not ranks_to_send or not ranks_to_recv:
#                 continue
#             # Calculate the rank to recv by this rank
#             num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
#             recver_pos = ranks_to_recv.index(ep_rank)
#             remainder_start = len(ranks_to_send) * num_dst_per_sender
#             if recver_pos < remainder_start:
#                 src_rank = ranks_to_send[recver_pos // num_dst_per_sender]
#             else:
#                 src_rank = ranks_to_send[recver_pos - remainder_start]
#             src_global = get_global_rank(ep_group, src_rank)
#             p2p_ops += [
#                 P2POp(
#                     torch.distributed.irecv,
#                     b[dst],
#                     src_global,
#                 )
#                 for b in buffers_list
#             ]

#     # 4) Execute merged P2P operations
#     if p2p_ops:
#         reqs = batch_isend_irecv(p2p_ops)
#         for req in reqs:
#             req.wait()

#     # 5) Copy from buffers into expert weights
#     for layer_idx in range(group_size):
#         old_indices = old_indices_group[layer_idx]
#         new_indices = new_indices_group[layer_idx]
#         weights_list = list(expert_weights_group[layer_idx])
#         buffers_list = list(buffers_group[layer_idx])
#         is_unchanged = layer_is_unchanged[layer_idx]
#         is_received_locally = layer_is_received_locally[layer_idx]
#         experts_recv_loc = layer_experts_recv_loc[layer_idx]

#         local2global = partial(
#             idx_local_to_global,
#             local_cnt=num_local_experts,
#             ep_rank=ep_rank,
#         )
#         for dst in range(num_local_experts):
#             if is_unchanged[dst]:
#                 continue
#             if is_received_locally[dst]:
#                 for w, b in zip(weights_list, buffers_list):
#                     w[dst].copy_(b[dst])
#             else:
#                 expert = new_indices[local2global(dst)]
#                 if expert == -1:
#                     continue
#                 src_local = experts_recv_loc.get(expert)
#                 if src_local is None:
#                     continue
#                 for w, b in zip(weights_list, buffers_list):
#                     w[dst].copy_(b[src_local])


''' LEGACY DISABLED PATH
def shuffle_layer_pack(
    num_local_experts: int,
    ep_rank: int,
    old_indices_group: Sequence[Sequence[int]],
    new_indices_group: Sequence[Sequence[int]],
    expert_weights_group: Sequence[Iterable[torch.Tensor]],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    peer_send_buffers: dict[int, dict[torch.dtype, torch.Tensor]],
    peer_recv_buffers: dict[int, dict[torch.dtype, torch.Tensor]],
    layer_elem_offsets: dict[torch.dtype, list[int]],
) -> None:
    """
    Perform expert weights rearrangement for a group of layers in one
    batched communication.
    Steps:
      1) Build per-peer send/recv plans across all layers in the group.
      2) Pack send buffers per peer, start non-blocking P2P communication.
      3) Perform intra-rank local moves using expert_weights_buffer for all layers.
      4) Wait for all P2P, then unpack received buffers directly into expert weights.
    """
    assert len(old_indices_group) == len(new_indices_group) == len(expert_weights_group)
    group_size = len(old_indices_group)

    first_weights_list = list(expert_weights_group[0])
    device = first_weights_list[0].device

    # Precompute per-dtype expert sizes (elements) using provided offsets
    elems_per_expert_by_dtype: dict[torch.dtype, int] = {
        d: offsets[-1] for d, offsets in layer_elem_offsets.items()
    }
    elems_per_weight_by_dtype: dict[torch.dtype, list[int]] = {
        d: [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]
        for d, offsets in layer_elem_offsets.items()
    }

    # Helper mapping per layer
    def build_local_maps_for_layer(
        old_indices: Sequence[int],
        new_indices: Sequence[int],
    ) -> tuple[list[bool], dict[int, int], dict[int, int]]:
        local2global = partial(
            idx_local_to_global,
            local_cnt=num_local_experts,
            ep_rank=ep_rank,
        )
        is_unchanged = [
            old_indices[local2global(i)] == new_indices[local2global(i)]
            for i in range(num_local_experts)
        ]
        experts_send_loc: dict[int, int] = {}
        experts_recv_loc: dict[int, int] = {}
        # Local send candidates
        for src in range(num_local_experts):
            expert = old_indices[local2global(src)]
            if expert == -1:
                continue
            if expert not in experts_send_loc:
                experts_send_loc[expert] = src
        # Identify local moves to avoid including them in experts_recv_loc
        is_received_locally = is_unchanged[:]
        for src in range(num_local_experts):
            src_global = local2global(src)
            for dst in range(num_local_experts):
                dst_global = local2global(dst)
                if is_received_locally[dst]:
                    continue
                if old_indices[src_global] == -1 or new_indices[dst_global] == -1:
                    continue
                if old_indices[src_global] == new_indices[dst_global]:
                    is_received_locally[dst] = True
        for dst in range(num_local_experts):
            if is_received_locally[dst]:
                continue
            expert = new_indices[local2global(dst)]
            if expert == -1:
                continue
            if expert not in experts_recv_loc:
                experts_recv_loc[expert] = dst
        return is_unchanged, experts_send_loc, experts_recv_loc

    # Build per-peer expert lists across layers
    send_peers_group: dict[
        int, list[tuple[int, int]]
    ] = {}  # peer -> list[(layer_idx, expert)]
    recv_peers_group: dict[int, list[tuple[int, int]]] = {}
    layer_is_unchanged: list[list[bool]] = []
    layer_experts_send_loc: list[dict[int, int]] = []
    layer_experts_recv_loc: list[dict[int, int]] = []

    for layer_idx in range(group_size):
        old_indices = old_indices_group[layer_idx]
        new_indices = new_indices_group[layer_idx]
        is_unchanged, experts_send_loc, experts_recv_loc = build_local_maps_for_layer(
            old_indices, new_indices
        )
        layer_is_unchanged.append(is_unchanged)
        layer_experts_send_loc.append(experts_send_loc)
        layer_experts_recv_loc.append(experts_recv_loc)

        # Build per-peer routing for this layer and merge into group-level maps
        for expert, _src in sorted(experts_send_loc.items()):
            ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
                expert,
                num_local_experts,
                old_indices,
                new_indices,
            )
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            sender_pos = ranks_to_send.index(ep_rank)
            recv_begin = sender_pos * num_dst_per_sender
            recv_end = recv_begin + num_dst_per_sender
            recv_ranks = ranks_to_recv[recv_begin:recv_end]
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            recver_pos = remainder_start + sender_pos
            if recver_pos < len(ranks_to_recv):
                recv_ranks.append(ranks_to_recv[recver_pos])
            for dst_rank in recv_ranks:
                send_peers_group.setdefault(dst_rank, []).append((layer_idx, expert))

        for expert, _dst in sorted(experts_recv_loc.items()):
            ranks_to_send, ranks_to_recv = get_ep_ranks_with_expert(
                expert,
                num_local_experts,
                old_indices,
                new_indices,
            )
            num_dst_per_sender = len(ranks_to_recv) // len(ranks_to_send)
            recver_pos = ranks_to_recv.index(ep_rank)
            remainder_start = len(ranks_to_send) * num_dst_per_sender
            if recver_pos < remainder_start:
                src_rank = ranks_to_send[recver_pos // num_dst_per_sender]
            else:
                src_rank = ranks_to_send[recver_pos - remainder_start]
            recv_peers_group.setdefault(src_rank, []).append((layer_idx, expert))

    # Prepare per-peer buffers and post irecvs first
    p2p_ops: list[P2POp] = []
    recv_buffers: dict[int, dict[torch.dtype, torch.Tensor]] = {}
    recv_orders: dict[int, list[tuple[int, int]]] = {}

    for peer_rank, items in sorted(recv_peers_group.items()):
        experts_order = sorted(items)  # sort by (layer_idx, expert) for determinism
        if not experts_order:
            continue
        need_rows = len(experts_order)
        dtype_to_buf_recv: dict[torch.dtype, torch.Tensor] = {}
        for d, offsets in layer_elem_offsets.items():
            elems_per_expert = elems_per_expert_by_dtype[d]
            pre_buf_d = peer_recv_buffers.get(peer_rank, {}).get(d)
            if (
                pre_buf_d is not None
                and need_rows <= pre_buf_d.shape[0]
                and pre_buf_d.shape[1] == elems_per_expert
            ):
                recv_buf_d = pre_buf_d[:need_rows].reshape(-1)
            else:
                recv_buf_d = torch.empty(
                    need_rows * elems_per_expert, dtype=d, device=device
                )
            dtype_to_buf_recv[d] = recv_buf_d
        src_global = get_global_rank(ep_group, peer_rank)
        for d, buf in dtype_to_buf_recv.items():
            p2p_ops.append(P2POp(torch.distributed.irecv, buf, src_global))
        recv_buffers[peer_rank] = dtype_to_buf_recv
        recv_orders[peer_rank] = experts_order

    # Pack and post sends
    send_buffers: dict[int, dict[torch.dtype, torch.Tensor]] = {}
    for peer_rank, items in sorted(send_peers_group.items()):
        experts_order = sorted(items)
        if not experts_order:
            continue
        need_rows = len(experts_order)
        dtype_to_buf_send: dict[torch.dtype, torch.Tensor] = {}
        for d, offsets in layer_elem_offsets.items():
            elems_per_expert = elems_per_expert_by_dtype[d]
            pre_buf_d = peer_send_buffers.get(peer_rank, {}).get(d)
            if (
                pre_buf_d is not None
                and need_rows <= pre_buf_d.shape[0]
                and pre_buf_d.shape[1] == elems_per_expert
            ):
                send_buf_d = pre_buf_d[:need_rows].reshape(-1)
            else:
                send_buf_d = torch.empty(
                    need_rows * elems_per_expert, dtype=d, device=device
                )
            dtype_to_buf_send[d] = send_buf_d

        # Pack across layers
        for i, (layer_idx, expert) in enumerate(experts_order):
            weights_list = list(expert_weights_group[layer_idx])
            src = layer_experts_send_loc[layer_idx][expert]
            # Group weights by dtype in this layer to map to dtype-specific offsets
            weights_idx_by_dtype_send: dict[torch.dtype, list[int]] = {}
            for k, w in enumerate(weights_list):
                weights_idx_by_dtype_send.setdefault(w.dtype, []).append(k)
            for d, idx_list in weights_idx_by_dtype_send.items():
                send_buf_d = dtype_to_buf_send[d]
                offsets = layer_elem_offsets[d]
                elems_per_expert = elems_per_expert_by_dtype[d]
                base = i * elems_per_expert
                for j, k in enumerate(idx_list):
                    w = weights_list[k]
                    vec = w[src].reshape(-1)
                    start = base + offsets[j]
                    send_buf_d.narrow(0, start, vec.numel()).copy_(vec)

        dst_global = get_global_rank(ep_group, peer_rank)
        for d, buf in dtype_to_buf_send.items():
            p2p_ops.append(P2POp(torch.distributed.isend, buf, dst_global))
        send_buffers[peer_rank] = dtype_to_buf_send

    # Start all P2P ops
    reqs = batch_isend_irecv(p2p_ops) if p2p_ops else []
    local2global = partial(
        idx_local_to_global,
        local_cnt=num_local_experts,
        ep_rank=ep_rank,
    )

    # Perform local moves.
    # (TODO) this has to be cleaned up after postprocessing of the rebalance.
    # Doesn't make sense to move an expert within a rank. It has to be unchanged.
    for layer_idx in range(group_size):
        is_unchanged = layer_is_unchanged[layer_idx]
        weights_list = list(expert_weights_group[layer_idx])
        buffers_list = list(expert_weights_buffer)
        old_indices = old_indices_group[layer_idx]
        new_indices = new_indices_group[layer_idx]
        # Stage local moves into tmp buffer using expert -> src mapping
        experts_send_loc = layer_experts_send_loc[layer_idx]
        for dst in range(num_local_experts):
            if is_unchanged[dst]:
                continue
            dst_global = local2global(dst)
            expert = new_indices[dst_global]
            if expert == -1:
                continue
            src_local = experts_send_loc.get(expert)
            if src_local is None:
                continue
            for w, b in zip(weights_list, buffers_list):
                b[dst].copy_(w[src_local])
        # Move from tmp buffer to expert weights
        for dst in range(num_local_experts):
            if is_unchanged[dst]:
                continue
            dst_global = local2global(dst)
            expert = new_indices[dst_global]
            if expert == -1:
                continue
            src_local = experts_send_loc.get(expert)
            if src_local is None:
                continue
            for w, b in zip(weights_list, buffers_list):
                w[dst].copy_(b[dst])

    # Wait for P2P requests
    for req in reqs:
        req.wait()

    # Unpack received buffers directly into expert weights
    for peer_rank, experts_order in recv_orders.items():
        recv_bufs_by_dtype = recv_buffers[peer_rank]
        for i, (layer_idx, expert) in enumerate(experts_order):
            weights_list = list(expert_weights_group[layer_idx])
            dst = layer_experts_recv_loc[layer_idx][expert]
            # Group weights by dtype to unpack with dtype-specific buffers
            weights_idx_by_dtype_recv: dict[torch.dtype, list[int]] = {}
            for k, w in enumerate(weights_list):
                weights_idx_by_dtype_recv.setdefault(w.dtype, []).append(k)
            for d, idx_list in weights_idx_by_dtype_recv.items():
                recv_buf_d = recv_bufs_by_dtype[d]
                offsets = layer_elem_offsets[d]
                elems_per_expert = elems_per_expert_by_dtype[d]
                base = i * elems_per_expert
                elems_per_weight = elems_per_weight_by_dtype[d]
                for j, k in enumerate(idx_list):
                    w = weights_list[k]
                    num = elems_per_weight[j]
                    slice_view = recv_buf_d.narrow(0, base + offsets[j], num)
                    w[dst].copy_(slice_view.view_as(w[dst]))

    # After unpacking, duplicate experts to additional local destinations if needed
    # (TODO) this has to be cleaned up after postprocessing of the rebalance.
    # Doesn't make sense to have a copy of an expert on the same rank.
    for layer_idx in range(group_size):
        is_unchanged = layer_is_unchanged[layer_idx]
        weights_list = list(expert_weights_group[layer_idx])
        experts_recv_loc = layer_experts_recv_loc[layer_idx]
        new_indices = new_indices_group[layer_idx]
        for expert, primary_dst in experts_recv_loc.items():
            for dst in range(num_local_experts):
                if dst == primary_dst:
                    continue
                if is_unchanged[dst]:
                    continue
                dst_global = local2global(dst)
                if new_indices[dst_global] != expert:
                    continue
                for w in weights_list:
                    w[dst].copy_(w[primary_dst])
'''


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

    ep_rank = ep_group.rank()
    ep_size = ep_group.size()
    assert num_physical_experts == ep_size * num_local_physical_experts

    # Max number of layers to group for communication
    max_group_layers = int(os.environ.get("VLLM_EPLB_MAX_GROUPED_LAYERS", "1"))
    max_group_layers = max(min(max_group_layers, num_moe_layers), 1)

    # A buffer to hold the expert weights in one layer during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.
    # expert_weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]
    # Compute layer size and pre-allocate per-peer grouped communication buffers
    first_layer_weights = list(expert_weights[0])
    # (
    #     max_group_layers,
    #     peer_send_buffers,
    #     peer_recv_buffers,
    #     layer_elem_offsets,
    #     layer_elems_by_dtype,
    #     capacity_rows,
    # ) = _allocate_peer_group_buffers(
    #     first_layer_weights,
    #     num_moe_layers,
    #     ep_group,
    #     is_profile=is_profile,
    # )

    # Pre-allocate reusable buffers for up to max_group_layers layers.
    weights_buffers: list[list[torch.Tensor]] = [
        [torch.empty_like(w) for w in first_layer_weights]
        for _ in range(max_group_layers)
    ]

    if is_profile:
        # if max_group_layers == 0:
        #     # Maximum send size is to send all local experts to all ranks,
        #     # So we use a dummy `all_gather` to reserve enough communication buffer
        #     for weight, buffer in zip(expert_weights[0], expert_weights_buffer):
        #         # A `/dev/null`-like buffer to avoid real memory allocation
        #         dummy_recv_buffer = [buffer for _ in range(ep_size)]
        #         # NOTE(bowen): Needed this barrier to avoid OOM during actual
        #         # execution. I'm not very sure why this is needed
        #         torch.distributed.barrier()
        #         all_gather(
        #             dummy_recv_buffer,
        #             weight,
        #             group=ep_group,
        #         )
        #     return
        # else:
        #     # Do per-dtype all_gather using computed sizes
        #     # without allocating peer buffers
        #     device = first_layer_weights[0].device
        #     for d, elems_per_expert in layer_elems_by_dtype.items():
        #         tmp = torch.empty(capacity_rows * elems_per_expert,
        #    dtype=d, device=device)
        #         dummy_recv_buffer = [tmp for _ in range(ep_size)]
        #         torch.distributed.barrier()
        #         all_gather(
        #             dummy_recv_buffer,
        #             tmp,
        #             group=ep_group,
        #         )
        #     return
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

    # old_global_expert_indices_cpu = old_global_expert_indices.cpu()
    # new_global_expert_indices_cpu = new_global_expert_indices.cpu()

    # NOTE(bowen): We need this synchronize to run, but I don't know why.
    # If you figure out the reason, please let me know -- thank you!
    torch.cuda.synchronize()

    # for layer in range(num_moe_layers):
    #     old_shuffle_layer(
    #         num_local_physical_experts,
    #         ep_rank,
    #         old_global_expert_indices_cpu[layer].tolist(),
    #         new_global_expert_indices_cpu[layer].tolist(),
    #         expert_weights[layer],
    #         weights_buffers[0],
    #         ep_group,
    #     )
    # return

    # start = 0
    # while start < num_moe_layers:
    #     end = min(start + max_group_layers, num_moe_layers)
    #     old_group = old_global_expert_indices_cpu[start:end].tolist()
    #     new_group = new_global_expert_indices_cpu[start:end].tolist()
    #     weights_group = [expert_weights[i] for i in range(start, end)]
    #     buffers_group = weights_buffers[: (end - start)]
    #     old_shuffle_layers_grouped(
    #         num_local_physical_experts,
    #         ep_rank,
    #         old_group,
    #         new_group,
    #         weights_group,
    #         buffers_group,
    #         ep_group,
    #     )
    #     start = end

    old_global_expert_indices_cpu = old_global_expert_indices.cpu().numpy()
    new_global_expert_indices_cpu = new_global_expert_indices.cpu().numpy()

    # for layer in range(num_moe_layers):
    #     old_shuffle_layer_np(
    #         num_local_physical_experts,
    #         ep_rank,
    #         old_global_expert_indices_cpu[layer],
    #         new_global_expert_indices_cpu[layer],
    #         expert_weights[layer],
    #         weights_buffers[0],
    #         ep_group,
    #     )
    # return

    start = 0
    while start < num_moe_layers:
        end = min(start + max_group_layers, num_moe_layers)
        old_group = old_global_expert_indices_cpu[start:end]
        new_group = new_global_expert_indices_cpu[start:end]
        weights_group = [expert_weights[i] for i in range(start, end)]
        buffers_group = weights_buffers[: (end - start)]
        old_shuffle_layers_grouped_np(
            num_local_physical_experts,
            ep_rank,
            old_group,
            new_group,
            weights_group,
            buffers_group,
            ep_group,
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


__all__ = ["rearrange_expert_weights_inplace"]
