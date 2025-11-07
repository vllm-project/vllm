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


def _allocate_peer_group_buffers(
    first_layer_weights: Sequence[torch.Tensor],
    num_moe_layers: int,
    ep_group: ProcessGroup,
) -> tuple[int, int, dict[int, torch.Tensor], list[int] | None]:
    """
    Allocate a single contiguous buffer per peer rank for grouped-layer comms.
    Returns:
        layer_bytes: The size of the layer in bytes.
        max_group_layers: The maximum number of layers that can be grouped.
        peer_buffers: A dictionary of peer buffers.
        elem_offsets: A list of element offsets per weight.
          - elem_offsets gives cumulative element offsets per weight for later packing.
    """
    assert first_layer_weights, "first_layer_weights must be non-empty"
    device = first_layer_weights[0].device
    assert device.type == "cuda", "Device must be CUDA"

    # Ensure all dtypes match so we can use a single dtype buffer
    dtypes = {w.dtype for w in first_layer_weights}
    assert len(dtypes) == 1, "All expert weights in a layer must share dtype"
    dtype = first_layer_weights[0].dtype

    layer_elems = 0
    layer_bytes = 0
    elem_offsets: list[int] = [0]
    for w in first_layer_weights:
        layer_elems += int(w.numel())
        layer_bytes += int(w.numel() * w.element_size())
        elem_offsets.append(layer_elems)

    free_bytes, _ = torch.cuda.mem_get_info(device)
    target_total_bytes = max(0, free_bytes // 2)

    world_size = ep_group.size()
    rank = ep_group.rank()
    num_peers = max(1, world_size - 1)
    per_peer_target_bytes = target_total_bytes // num_peers

    max_group_layers = min(num_moe_layers, per_peer_target_bytes // layer_bytes)
    if max_group_layers <= 0:
        return layer_bytes, 0, {}, elem_offsets

    peer_buffers: dict[int, torch.Tensor] = {}
    for peer in range(world_size):
        if peer == rank:
            continue
        peer_buffers[peer] = torch.empty(
            (max_group_layers, layer_elems),
            dtype=dtype,
            device=device,
        )

    return layer_bytes, max_group_layers, peer_buffers, elem_offsets


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


def shuffle_layer(
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

    # Prepare per-peer batched communication buffers
    p2p_ops: list[P2POp] = []

    # Map expert id -> local src (for send) and expert id -> local dst (for recv)
    experts_send_loc: dict[int, int] = {}
    for src in range(num_local_experts):
        expert = old_indices[local2global(src)]
        if expert == -1:
            continue
        if expert in experts_send_loc:
            continue
        experts_send_loc[expert] = src

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

    # Build per-peer expert lists for send/recv
    send_peers: dict[int, list[int]] = {}
    for expert, src in sorted(experts_send_loc.items()):
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
            send_peers.setdefault(dst_rank, []).append(expert)

    recv_peers: dict[int, list[int]] = {}
    for expert, dst in sorted(experts_recv_loc.items()):
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
        recv_peers.setdefault(src_rank, []).append(expert)

    # Validate dtypes and compute element layout per expert across weights
    weights_list = list(expert_weights)
    dtypes = {w.dtype for w in weights_list}
    assert len(dtypes) == 1, "All expert weights in a layer must share dtype"
    dtype = weights_list[0].dtype
    device = weights_list[0].device

    elems_per_weight: list[int] = [int(w.shape[1]) for w in weights_list]
    elem_offsets: list[int] = [0]
    for n in elems_per_weight:
        elem_offsets.append(elem_offsets[-1] + n)
    elems_per_expert = elem_offsets[-1]

    # Allocate per-peer send/recv buffers and enqueue P2P ops
    send_buffers: dict[int, torch.Tensor] = {}
    recv_buffers: dict[int, torch.Tensor] = {}
    recv_orders: dict[int, list[int]] = {}

    # Post receives first
    for peer_rank, experts in sorted(recv_peers.items()):
        experts_sorted = sorted(experts)
        if not experts_sorted:
            continue
        recv_buf = torch.empty(
            len(experts_sorted) * elems_per_expert, dtype=dtype, device=device
        )
        src_global = get_global_rank(ep_group, peer_rank)
        p2p_ops.append(P2POp(torch.distributed.irecv, recv_buf, src_global))
        recv_buffers[peer_rank] = recv_buf
        recv_orders[peer_rank] = experts_sorted

    # Then posts sends
    for peer_rank, experts in sorted(send_peers.items()):
        experts_sorted = sorted(experts)
        if not experts_sorted:
            continue
        send_buf = torch.empty(
            len(experts_sorted) * elems_per_expert, dtype=dtype, device=device
        )
        # pack
        for i, expert in enumerate(experts_sorted):
            src = experts_send_loc[expert]
            base = i * elems_per_expert
            for k, w in enumerate(weights_list):
                vec = w[src].reshape(-1)
                start = base + elem_offsets[k]
                send_buf.narrow(0, start, vec.numel()).copy_(vec)
        dst_global = get_global_rank(ep_group, peer_rank)
        p2p_ops.append(P2POp(torch.distributed.isend, send_buf, dst_global))
        send_buffers[peer_rank] = send_buf

    # 4. Execute the P2P operations. The real communication happens here.
    if p2p_ops:
        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    # Unpack received buffers into expert_weights_buffer
    for peer_rank, experts_sorted in recv_orders.items():
        recv_buf = recv_buffers[peer_rank]
        for i, expert in enumerate(experts_sorted):
            dst = experts_recv_loc[expert]
            base = i * elems_per_expert
            for k, buf in enumerate(expert_weights_buffer):
                num = elems_per_weight[k]
                slice_view = recv_buf.narrow(0, base + elem_offsets[k], num)
                buf[dst].copy_(slice_view.view_as(buf[dst]))

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

    # Compute layer size and pre-allocate per-peer grouped communication buffers
    first_layer_weights = list(expert_weights[0])
    layer_bytes, max_group_layers, communication_buffers, layer_elem_offsets = (
        _allocate_peer_group_buffers(
            first_layer_weights,
            num_moe_layers,
            ep_group,
        )
    )

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
    torch.cuda.synchronize()
    for layer in range(num_moe_layers):
        shuffle_layer(
            num_local_physical_experts,
            ep_rank,
            old_global_expert_indices_cpu[layer].tolist(),
            new_global_expert_indices_cpu[layer].tolist(),
            expert_weights[layer],
            expert_weights_buffer,
            ep_group,
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


__all__ = ["rearrange_expert_weights_inplace"]
