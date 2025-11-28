# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The actual execution of the rearrangement.

This involves the exchange of expert weights between GPUs.
"""

from collections.abc import Iterable, MutableSequence, Sequence
from functools import partial

import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    all_gather,
    batch_isend_irecv,
    get_global_rank,
)


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


def move_to_buffer(
    num_local_experts: int,
    old_indices: Sequence[int],
    new_indices: Sequence[int],
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffer: Sequence[torch.Tensor],
    cuda_stream: torch.cuda.Stream | None,
    ep_group: ProcessGroup,
) -> tuple[list[bool], list[bool], dict[int, int]]:
    """
    Perform expert weights rearrangement of one layer.
    """
    ep_rank = ep_group.rank()
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
                    with torch.cuda.stream(cuda_stream):
                        buffer[dst].copy_(weight[src], non_blocking=True)

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
    return is_unchanged, is_received_locally, experts_recv_loc


def move_from_buffer(
    expert_weights: Iterable[torch.Tensor],
    expert_weights_buffer: list[torch.Tensor],
    is_unchanged: list[bool],
    is_received_locally: list[bool],
    experts_recv_loc: dict[int, int],
    new_indices: Sequence[int],
    ep_group: ProcessGroup,
) -> None:
    ep_rank = ep_group.rank()
    num_local_experts = len(is_unchanged)

    local2global = partial(
        idx_local_to_global, local_cnt=num_local_experts, ep_rank=ep_rank
    )

    for dst in range(num_local_experts):
        if is_unchanged[dst]:
            continue
        if is_received_locally[dst]:
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[dst], non_blocking=True)
        else:
            expert = new_indices[local2global(dst)]
            if expert == -1:
                continue
            src = experts_recv_loc[expert]
            for weight, buffer in zip(expert_weights, expert_weights_buffer):
                weight[dst].copy_(buffer[src], non_blocking=True)


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
) -> tuple[list[bool], list[bool], dict[int, int]]:
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
    # A buffer to hold the expert weights in one layer during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.

    is_unchanged, is_received_locally, experts_recv_loc = move_to_buffer(
        num_local_experts=num_local_physical_experts,
        old_indices=old_global_expert_indices[layer].tolist(),
        new_indices=new_global_expert_indices[layer].tolist(),
        expert_weights=expert_weights[layer],
        expert_weights_buffer=expert_weights_buffer,
        cuda_stream=cuda_stream,
        ep_group=ep_group,
    )
    return is_unchanged, is_received_locally, experts_recv_loc


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

    # A buffer to hold the expert weights in one layer during the exchange.
    # NOTE: Currently we assume the same weights across different layers
    # have the same shape.
    expert_weights_buffer = [torch.empty_like(w) for w in expert_weights[0]]

    if is_profile:
        # Maximum send size is to send all local experts to all ranks,
        # So we use a dummy `all_gather` to reserve enough communication buffer
        for weight, buffer in zip(expert_weights[0], expert_weights_buffer):
            # A `/dev/null`-like buffer to avoid real memory allocation
            dummy_recv_buffer = [buffer for _ in range(ep_size)]
            # NOTE(bowen): Needed this barrier to avoid OOM during actual
            # execution. I'm not very sure why this is needed
            torch.distributed.barrier()
            all_gather(
                dummy_recv_buffer,
                weight,
                group=ep_group,
            )
        return

    old_global_expert_indices_cpu = old_global_expert_indices.cpu()
    new_global_expert_indices_cpu = new_global_expert_indices.cpu()

    # NOTE(bowen): We need this synchronize to run, but I don't know why.
    # If you figure out the reason, please let me know -- thank you!
    torch.cuda.synchronize()

    for layer in range(num_moe_layers):
        is_unchanged, is_received_locally, experts_recv_loc = move_to_buffer(
            num_local_experts=num_local_physical_experts,
            old_indices=old_global_expert_indices_cpu[layer].tolist(),
            new_indices=new_global_expert_indices_cpu[layer].tolist(),
            expert_weights=expert_weights[layer],
            expert_weights_buffer=expert_weights_buffer,
            cuda_stream=None,
            ep_group=ep_group,
        )

        move_from_buffer(
            expert_weights=expert_weights[layer],
            expert_weights_buffer=expert_weights_buffer,
            is_unchanged=is_unchanged,
            is_received_locally=is_received_locally,
            experts_recv_loc=experts_recv_loc,
            new_indices=new_global_expert_indices[layer].tolist(),
            ep_group=ep_group,
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


__all__ = ["transfer_layer", "move_from_buffer"]
