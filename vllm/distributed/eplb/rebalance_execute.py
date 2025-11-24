# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

import os
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_gather,
    all_to_all_single,
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


class AllToAllRowExchangePlanner:
    """
    Collects per-peer row sends and receives for a given weight index, then
    executes a single all_to_all_single to transfer rows and scatters them
    into their target positions.
    """

    def __init__(
        self,
        ep_group: ProcessGroup,
        ep_size: int,
    ) -> None:
        self.ep_group = ep_group
        self.ep_size = ep_size
        # Indexed by weight_idx -> peer_rank -> list[TensorRowView]
        self._sends_per_weight: list[dict[int, list[torch.Tensor]]] = []
        self._recvs_per_weight: list[dict[int, list[torch.Tensor]]] = []

    def _ensure_weight_slot(self, weight_idx: int) -> None:
        while len(self._sends_per_weight) <= weight_idx:
            self._sends_per_weight.append({})
            self._recvs_per_weight.append({})

    def add_send_row(
        self,
        peer_rank: int,
        weight_idx: int,
        row_tensor: torch.Tensor,
    ) -> None:
        self._ensure_weight_slot(weight_idx)
        self._sends_per_weight[weight_idx].setdefault(peer_rank, []).append(row_tensor)

    def add_recv_target(
        self,
        peer_rank: int,
        weight_idx: int,
        target_row: torch.Tensor,
    ) -> None:
        self._ensure_weight_slot(weight_idx)
        self._recvs_per_weight[weight_idx].setdefault(peer_rank, []).append(target_row)

    def execute(self) -> None:
        # Execute one all_to_all_single per weight index
        for weight_idx, (send_map, recv_map) in enumerate(
            zip(self._sends_per_weight, self._recvs_per_weight)
        ):
            # If neither sends nor recvs exist, skip
            has_any = any(send_map.values()) or any(recv_map.values())
            if not has_any:
                continue

            # Determine dtype/device/hidden_size from any available tensor
            sample_tensor: torch.Tensor | None = None
            if send_map:
                for rows in send_map.values():
                    if rows:
                        sample_tensor = rows[0]
                        break
            if sample_tensor is None and recv_map:
                for rows in recv_map.values():
                    if rows:
                        sample_tensor = rows[0]
                        break
            assert sample_tensor is not None
            device = sample_tensor.device
            dtype = sample_tensor.dtype
            # Treat rows as [1, hidden] slices; infer hidden size
            hidden_shape = sample_tensor.shape[1:]

            # Build per-peer concatenated send blocks
            per_peer_send_blocks: list[torch.Tensor] = []
            input_split_sizes: list[int] = []
            for peer in range(self.ep_size):
                rows = send_map.get(peer, [])
                if rows:
                    block = torch.cat(rows, dim=0).contiguous()
                else:
                    # Create an empty (0, hidden) block
                    empty_shape = (0,) + hidden_shape
                    block = torch.empty(empty_shape, device=device, dtype=dtype)
                per_peer_send_blocks.append(block)
                input_split_sizes.append(block.shape[0])

            # Flatten into a single input tensor along dim 0
            input_tensor: torch.Tensor
            if any(sz > 0 for sz in input_split_sizes):
                input_tensor = torch.cat(per_peer_send_blocks, dim=0)
            else:
                input_tensor = torch.empty(
                    (0,) + hidden_shape, device=device, dtype=dtype
                )

            # Expected receives per peer (counts of rows)
            output_split_sizes: list[int] = []
            for peer in range(self.ep_size):
                output_split_sizes.append(len(recv_map.get(peer, [])))

            total_recv_rows = sum(output_split_sizes)
            output_tensor = torch.empty(
                (total_recv_rows,) + hidden_shape, device=device, dtype=dtype
            )

            # If nothing to send/recv, skip the collective
            if input_tensor.numel() == 0 and output_tensor.numel() == 0:
                continue

            all_to_all_single(
                output_tensor,
                input_tensor,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=self.ep_group,
            )

            # Scatter received rows into their targets in the same per-peer order
            offset = 0
            for peer in range(self.ep_size):
                recv_rows = recv_map.get(peer, [])
                count = len(recv_rows)
                if count == 0:
                    continue
                block = output_tensor[offset : offset + count]
                # Row-wise copy into target views
                for i, target in enumerate(recv_rows):
                    target.copy_(block[i])
                offset += count
        # Clear after execution
        self._sends_per_weight.clear()
        self._recvs_per_weight.clear()


def shuffle_layers_grouped_sendrecv(
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

    # Local moves into tmp buffers
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


def shuffle_layers_grouped_all2all(
    num_local_experts: int,
    ep_rank: int,
    old_indices_group: np.ndarray,
    new_indices_group: np.ndarray,
    expert_weights_group: Sequence[Iterable[torch.Tensor]],
    buffers_group: Sequence[Sequence[torch.Tensor]],
    ep_group: ProcessGroup,
) -> None:
    """
    All-to-all optimized version using numpy indices planning and a single
    all_to_all_single per weight to exchange remote rows.
    - Local rows are first copied into temporary buffers.
    - Remote receives write directly into the original expert weights.
    - After communication, local buffered rows are copied back to weights.
    - Remote duplicates are materialized by copying from the primary dst row.
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

    # 1) Stage local copies into temporary buffers
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

    # 2) Plan remote exchanges per peer using deterministic expert ordering
    planner = AllToAllRowExchangePlanner(
        ep_group=ep_group,
        ep_size=ep_group.size(),
    )
    # Adds are ordered by sorted expert ids to ensure matching order between peers
    for layer_idx in range(group_size):
        old_indices = old_indices_group[layer_idx]
        new_indices = new_indices_group[layer_idx]
        weights_list = list(expert_weights_group[layer_idx])

        # Sends
        count_send = int(send_counts[layer_idx])
        if count_send > 0:
            experts = send_expert_ids[layer_idx, :count_send]
            srcs = send_src_rows[layer_idx, :count_send]
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
                for dst_rank in recv_ranks:
                    for weight_idx, w in enumerate(weights_list):
                        planner.add_send_row(dst_rank, weight_idx, w[src : src + 1])

        # Recvs (primary only)
        count_recv = int(recv_counts[layer_idx])
        if count_recv > 0:
            experts = recv_expert_ids[layer_idx, :count_recv]
            dsts = recv_dst_rows[layer_idx, :count_recv]
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
                    src_rank = ranks_to_send[recver_pos // num_dst_per_sender]
                else:
                    src_rank = ranks_to_send[recver_pos - remainder_start]
                for weight_idx, w in enumerate(weights_list):
                    planner.add_recv_target(src_rank, weight_idx, w[dst : dst + 1])

    # 3) Execute all-to-all exchanges and scatter results
    planner.execute()

    # 4) Copy back local buffered rows into destination weights
    for layer_idx in range(group_size):
        is_unchanged = layer_is_unchanged[layer_idx, :]
        is_received_locally = layer_is_received_locally[layer_idx, :]
        weights_list = list(expert_weights_group[layer_idx])
        buffers_list = list(buffers_group[layer_idx])
        for dst in range(num_local_experts):
            if is_unchanged[dst] or not is_received_locally[dst]:
                continue
            for w, b in zip(weights_list, buffers_list):
                w[dst].copy_(b[dst])

    # 5) Duplicate remote received rows to non-primary duplicate dsts
    for layer_idx in range(group_size):
        is_unchanged = layer_is_unchanged[layer_idx, :]
        is_received_locally = layer_is_received_locally[layer_idx, :]
        new_indices = new_indices_group[layer_idx]
        weights_list = list(expert_weights_group[layer_idx])

        count_recv = int(recv_counts[layer_idx])
        primary_map: dict[int, int] = {}
        if count_recv > 0:
            experts = recv_expert_ids[layer_idx, :count_recv]
            dsts = recv_dst_rows[layer_idx, :count_recv]
            for e, d in zip(experts.tolist(), dsts.tolist()):
                primary_map[int(e)] = int(d)

        for dst in range(num_local_experts):
            if is_unchanged[dst] or is_received_locally[dst]:
                continue
            expert = int(new_indices[ep_rank * num_local_experts + dst])
            if expert == -1:
                continue
            if recv_primary_mask[layer_idx, dst]:
                continue
            primary_dst = primary_map.get(expert)
            if primary_dst is None:
                continue
            for w in weights_list:
                w[dst].copy_(w[primary_dst])


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
        # shuffle_layers_grouped_sendrecv(
        shuffle_layers_grouped_all2all(
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
