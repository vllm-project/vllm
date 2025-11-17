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

from vllm.logger import init_logger

logger = init_logger(__name__)


def _allocate_peer_group_buffers(
    first_layer_weights: Sequence[torch.Tensor],
    num_moe_layers: int,
    ep_group: ProcessGroup,
) -> tuple[
    int,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    list[int],
]:
    """
    Allocate two contiguous buffers per peer rank (send and recv)
    into half of the free memory for grouped-layer comms.
    Returns:
        layer_bytes: The size of the layer in bytes.
        max_group_layers: The maximum number of layers that can be grouped
        so that we can fit auxiliary buffers into half of the free memory.
        peer_send_buffers: A dictionary mapping peer -> send buffer.
        peer_recv_buffers: A dictionary mapping peer -> recv buffer.
        elem_offsets: A list of element offsets per weight.
          - elem_offsets gives cumulative element offsets per weight for later packing.
    """
    device = first_layer_weights[0].device
    dtypes = [w.dtype for w in first_layer_weights]
    if len(set(dtypes)) > 1:
        logger.warning("EPLB: Different dtypes found in MoE weight: %s", dtypes)
        return 0, {}, {}, []
    dtype = dtypes[0]

    # Number of local experts on this rank (rows in expert weight tensors)
    assert first_layer_weights, "Expected non-empty first_layer_weights"
    num_local_experts = int(first_layer_weights[0].shape[0])

    layer_elems = 0
    layer_bytes = 0
    elem_offsets: list[int] = [0]
    for w in first_layer_weights:
        assert w.dim() >= 2, "Expected expert weight with [num_local_experts, ...]"
        expert_width = int(w.shape[1])
        layer_elems += expert_width
        layer_bytes += int(expert_width * w.element_size())
        elem_offsets.append(layer_elems)

    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Fit auxiliary buffers into half of the free memory.
    # Another half could be used for nccl internal buffers.
    target_total_bytes = max(0, free_bytes // 2)
    world_size = ep_group.size()
    rank = ep_group.rank()
    num_peers = max(1, world_size - 1)
    # Each peer needs to allocate two contiguous buffers (send and recv).
    # Worst case: a single peer may receive all local experts from a layer.
    # Account for that by multiplying per-layer bytes by num_local_experts.
    bytes_per_row = layer_bytes  # one expert across all weights
    bytes_per_layer_per_peer = num_local_experts * bytes_per_row
    # Subtract one layer worth to account for the auxiliary buffers.
    per_peer_capacity = target_total_bytes // (2 * num_peers)
    per_peer_target_bytes = per_peer_capacity - bytes_per_layer_per_peer

    # Fit as many layers as possible into the target bytes.
    max_group_layers = min(
        num_moe_layers,
        max(0, per_peer_target_bytes) // max(1, bytes_per_layer_per_peer),
    )
    if max_group_layers <= 0:
        logger.warning(
            "Not enough free memory for EPLB peer buffers. "
            "layer_bytes(per-expert): %d, num_local_experts: %d, "
            "bytes_per_layer_per_peer: %d, max_group_layers: %d, "
            "free_bytes: %d (%.2f GiB), target_total_bytes: %d (%.2f GiB)",
            layer_bytes,
            num_local_experts,
            bytes_per_layer_per_peer,
            max_group_layers,
            free_bytes,
            free_bytes / 1024**3,
            target_total_bytes,
            target_total_bytes / 1024**3,
        )
        return 0, {}, {}, []
    if ep_group.rank() == 0:
        logger.debug(
            "EPLB: target_total_bytes=%.2fGiB, layer_bytes=%.2fKiB, "
            "local_experts=%d, free_bytes=%.2fGiB, max_group_layers=%d/%d",
            target_total_bytes / (1024**3),
            layer_bytes / 1024,
            num_local_experts,
            free_bytes / (1024**3),
            max_group_layers,
            num_moe_layers,
        )

    peer_send_buffers: dict[int, torch.Tensor] = {}
    peer_recv_buffers: dict[int, torch.Tensor] = {}
    # Preallocate enough rows to handle the worst case per layer
    capacity_rows = max_group_layers * num_local_experts
    for peer in range(world_size):
        if peer == rank:
            continue
        peer_send_buffers[peer] = torch.empty(
            (capacity_rows, layer_elems),
            dtype=dtype,
            device=device,
        )
        peer_recv_buffers[peer] = torch.empty(
            (capacity_rows, layer_elems),
            dtype=dtype,
            device=device,
        )

    return (
        max_group_layers,
        peer_send_buffers,
        peer_recv_buffers,
        elem_offsets,
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


def shuffle_layer_pack(
    num_local_experts: int,
    ep_rank: int,
    old_indices_group: Sequence[Sequence[int]],
    new_indices_group: Sequence[Sequence[int]],
    expert_weights_group: Sequence[Iterable[torch.Tensor]],
    expert_weights_buffer: Sequence[torch.Tensor],
    ep_group: ProcessGroup,
    peer_send_buffers: dict[int, torch.Tensor],
    peer_recv_buffers: dict[int, torch.Tensor],
    layer_elem_offsets: list[int],
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

    # Validate dtype/device consistency and prepare layout info
    first_weights_list = list(expert_weights_group[0])
    dtypes = {w.dtype for w in first_weights_list}
    assert len(dtypes) == 1, "All expert weights in a layer must share dtype"
    dtype = first_weights_list[0].dtype
    device = first_weights_list[0].device

    elem_offsets = layer_elem_offsets
    elems_per_weight = [
        elem_offsets[i + 1] - elem_offsets[i] for i in range(len(elem_offsets) - 1)
    ]
    elems_per_expert = elem_offsets[-1]

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
    recv_buffers: dict[int, torch.Tensor] = {}
    recv_orders: dict[int, list[tuple[int, int]]] = {}

    for peer_rank, items in sorted(recv_peers_group.items()):
        experts_order = sorted(items)  # sort by (layer_idx, expert) for determinism
        if not experts_order:
            continue
        need_rows = len(experts_order)
        pre_buf = peer_recv_buffers.get(peer_rank)
        if (
            pre_buf is not None
            and need_rows <= pre_buf.shape[0]
            and pre_buf.shape[1] == elems_per_expert
        ):
            recv_buf = pre_buf[:need_rows].reshape(-1)
        else:
            recv_buf = torch.empty(
                need_rows * elems_per_expert, dtype=dtype, device=device
            )
        src_global = get_global_rank(ep_group, peer_rank)
        p2p_ops.append(P2POp(torch.distributed.irecv, recv_buf, src_global))
        recv_buffers[peer_rank] = recv_buf
        recv_orders[peer_rank] = experts_order

    # Pack and post sends
    send_buffers: dict[int, torch.Tensor] = {}
    for peer_rank, items in sorted(send_peers_group.items()):
        experts_order = sorted(items)
        if not experts_order:
            continue
        need_rows = len(experts_order)
        pre_buf = peer_send_buffers.get(peer_rank)
        if (
            pre_buf is not None
            and need_rows <= pre_buf.shape[0]
            and pre_buf.shape[1] == elems_per_expert
        ):
            send_buf = pre_buf[:need_rows].reshape(-1)
        else:
            send_buf = torch.empty(
                need_rows * elems_per_expert, dtype=dtype, device=device
            )

        # Pack across layers
        for i, (layer_idx, expert) in enumerate(experts_order):
            weights_list = list(expert_weights_group[layer_idx])
            src = layer_experts_send_loc[layer_idx][expert]
            base = i * elems_per_expert
            for k, w in enumerate(weights_list):
                vec = w[src].reshape(-1)
                start = base + elem_offsets[k]
                send_buf.narrow(0, start, vec.numel()).copy_(vec)

        dst_global = get_global_rank(ep_group, peer_rank)
        p2p_ops.append(P2POp(torch.distributed.isend, send_buf, dst_global))
        send_buffers[peer_rank] = send_buf

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
        recv_buf = recv_buffers[peer_rank]
        for i, (layer_idx, expert) in enumerate(experts_order):
            weights_list = list(expert_weights_group[layer_idx])
            dst = layer_experts_recv_loc[layer_idx][expert]
            base = i * elems_per_expert
            for k, w in enumerate(weights_list):
                num = elems_per_weight[k]
                slice_view = recv_buf.narrow(0, base + elem_offsets[k], num)
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

    # Compute layer size and pre-allocate per-peer grouped communication buffers
    first_layer_weights = list(expert_weights[0])
    (
        max_group_layers,
        peer_send_buffers,
        peer_recv_buffers,
        layer_elem_offsets,
    ) = _allocate_peer_group_buffers(
        first_layer_weights,
        num_moe_layers,
        ep_group,
    )

    old_global_expert_indices_cpu = old_global_expert_indices.cpu()
    new_global_expert_indices_cpu = new_global_expert_indices.cpu()

    # NOTE(bowen): We need this synchronize to run, but I don't know why.
    # If you figure out the reason, please let me know -- thank you!
    torch.cuda.synchronize()
    if max_group_layers == 0:
        logger.warning(
            "max_group_layers is 0, performing per-layer Expert Weight Exchange"
        )
        for layer in range(num_moe_layers):
            old_shuffle_layer(
                num_local_physical_experts,
                ep_rank,
                old_global_expert_indices_cpu[layer].tolist(),
                new_global_expert_indices_cpu[layer].tolist(),
                expert_weights[layer],
                expert_weights_buffer,
                ep_group,
            )

    # Group layers into batches of up to max_group_layers and perform grouped shuffle
    start = 0
    while start < num_moe_layers:
        end = min(start + max_group_layers, num_moe_layers)
        old_group = [
            old_global_expert_indices_cpu[i].tolist() for i in range(start, end)
        ]
        new_group = [
            new_global_expert_indices_cpu[i].tolist() for i in range(start, end)
        ]
        weights_group = [expert_weights[i] for i in range(start, end)]
        shuffle_layer_pack(
            num_local_physical_experts,
            ep_rank,
            old_group,
            new_group,
            weights_group,
            expert_weights_buffer,
            ep_group,
            peer_send_buffers,
            peer_recv_buffers,
            layer_elem_offsets,
        )
        start = end
    del peer_send_buffers, peer_recv_buffers
    del expert_weights_buffer
    torch.cuda.empty_cache()


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
