# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
from overrides import override
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed import P2POp, Work, batch_isend_irecv, get_global_rank

from vllm.distributed.eplb.eplb_adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm.distributed.eplb.eplb_transfer.abstract_transfer import BaseTransfer
from vllm.distributed.eplb.eplb_utils.eplb_utils import (
    get_ep_ranks_with_expert,
    idx_local_to_global,
)


class EplbWeightTransfer(BaseTransfer):
    """
    A concrete implementation of BaseTransfer for Expert Parallel Load
    Balancing (EPLB).

    This class is responsible for managing the transfer and update of
    expert weights across different ranks during expert rearrangement.
    """

    def __init__(self, eplb_adaptor: VllmEplbAdaptor):
        """
        Initializes the EplbWeightTransfer.

        Args:
            eplb_adaptor: An adaptor to interact with the vLLM model's expert
                          parameters and buffer management.
        """
        self.reqs: list[Work] = []
        self.eplb_adaptor = eplb_adaptor
        self.layer_id: int = -1
        self.recv_expert_list: list[tuple[int, int]] = []
        self.comm_op_list: list[P2POp] = []
        self.updated_expert_map: Optional[torch.Tensor] = None
        self.updated_log2phy_map: Optional[torch.Tensor] = None

    def shuffle_layer(
        self,
        num_local_experts: int,
        ep_rank: int,
        old_indices: Sequence[int],
        new_indices: Sequence[int],
        expert_weights: Iterable[torch.Tensor],
        expert_weights_buffer: Sequence[torch.Tensor],
        ep_group: ProcessGroup,
    ) -> None:
        """
        Performs expert weights rearrangement for a single MoE layer.
        This method orchestrates the entire shuffling process including
        local copies and inter-rank P2P communications.

        Args:
            num_local_experts: The number of experts managed
                               by the current EP rank.
            ep_rank: The current rank within the expert parallel group.
            old_indices: Global indices of experts before rearrangement.
                         Shape: (num_global_experts,)
            new_indices: Global indices of experts after rearrangement.
                         Shape: (num_global_experts,)
            expert_weights: An iterable of torch.Tensor, representing
                            the actual expert weights (e.g., [w1, w2, ...]).
                            Each w_i is (num_local_experts, expert_dim).
            expert_weights_buffer: An iterable of torch.Tensor,
                                   representing buffers for receiving expert
                                   weights. Same structure as expert_weights.
            ep_group: The PyTorch distributed process group for
                      expert parallelism.
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
        p2p_ops = self.prepare_send_p2p_ops(
            ep_group,
            ep_rank,
            expert_weights,
            local2global,
            new_indices,
            num_local_experts,
            old_indices,
            p2p_ops,
        )

        # 3. Initiate receiving of weights.
        experts_recv_loc, p2p_ops = self.prepare_recv_p2p_ops(
            ep_group,
            ep_rank,
            expert_weights_buffer,
            is_received_locally,
            local2global,
            new_indices,
            num_local_experts,
            old_indices,
            p2p_ops,
        )

        # 4. Execute the P2P operations. The real communication happens here.
        self.send_recv(p2p_ops)

        # 5. Copy the weights from the buffer back to the original weights.
        self.update_weight(
            expert_weights,
            expert_weights_buffer,
            experts_recv_loc,
            is_received_locally,
            is_unchanged,
            local2global,
            new_indices,
            num_local_experts,
        )

    def update_weight(
        self,
        expert_weights,
        expert_weights_buffer,
        experts_recv_loc,
        is_received_locally,
        is_unchanged,
        local2global,
        new_indices,
        num_local_experts,
    ):
        """
        Updates the actual expert weights from the buffer after communication.
        This is part of the `shuffle_layer` process.

        Args:
            expert_weights: The actual expert weight tensors to be updated.
            expert_weights_buffer: Buffers containing received or
                                   locally copied weights.
            experts_recv_loc: A dictionary mapping global expert ID to
                              the local buffer index where it was received.
            is_received_locally: Boolean list indicating if
                                 an expert was copied locally.
            is_unchanged: Boolean list indicating if an expert
                          remained in its original slot.
            local2global: Partial function to convert
                          local index to global expert ID.
            new_local_expert_indices: The new global indices
                                      mapping for experts.
            num_local_experts: Number of local experts.
        """
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

    @override
    def send_recv(self, p2p_ops: list[P2POp]) -> None:
        """
        Executes a batch of P2P communication operations.

        Args:
            p2p_ops: A list of P2POp objects (isend/irecv operations).
        """
        if p2p_ops:
            reqs: list[Work] = batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

    def prepare_recv_p2p_ops(
        self,
        ep_group,
        ep_rank,
        expert_weights_buffer,
        is_received_locally,
        local2global,
        new_indices,
        num_local_experts,
        old_indices,
        p2p_ops,
    ):
        """
        Prepares irecv operations for experts thatneed to be received
        from other ranks as part of the `shuffle_layer` process.

        Args:
            ep_group: The PyTorch distributed process group.
            ep_rank: Current rank in the EP group.
            expert_weights_buffer: Buffers for receiving weights.
            is_received_locally: Boolean list indicating if an
                                 expertwas copied locally.
            local2global: Partial function for local to
                        global index conversion.
            new_indices: The new global expert indices mapping.
            num_local_experts: Number of local experts.
            old_indices: The old global expert indices mapping.
            p2p_ops: List to append P2POp objects.

        Returns:
            A tuple containing:
            - experts_recv_loc: A dictionary mapping global expert ID to the
                                local buffer index where it will be received.
            - p2p_ops: The updated list of P2POp objects.
        """
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
            if not ranks_to_send:
                raise ValueError(f"Expert {expert} is needed but no rank has it.")
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
        return experts_recv_loc, p2p_ops

    def prepare_send_p2p_ops(
        self,
        ep_group,
        ep_rank,
        expert_weights,
        local2global,
        new_indices,
        num_local_experts,
        old_indices,
        p2p_ops,
    ):
        """
        Prepares isend operations for experts that needto be sent
        to other ranks as part of the `shuffle_layer` process.

        Args:
            ep_group: The PyTorch distributed process group.
            ep_rank: Current rank in the EP group.
            expert_weights: The actual expert weight tensors to be sent.
            local2global: Partial function for local
                          to global index conversion.
            new_indices: The new global expert indices mapping.
            num_local_experts: Number of local experts.
            old_indices: The old global expert indices mapping.
            p2p_ops: List to append P2POp objects.

        Returns:
            The updated list of P2POp objects.
        """
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
        return p2p_ops

    @override
    def prepare_send(self, expert_send_info, layer_id):
        """
        Prepares asynchronous send tasks (isend) for expert weights.
        This method is intended to be called when
        setting up communication for a specific layer.

        Args:
            expert_send_info: A list of tuples, where each tuple is
                              (destination_rank, global_expert_id_to_send).
            layer_id: The ID of the MoE layer for which experts are being sent.
        """
        for dst_rank, global_expert_id_to_send in expert_send_info:
            local_expert_id = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id][
                global_expert_id_to_send
            ].item()
            for src_tensor in self.eplb_adaptor.expert_param_per_layer[layer_id][
                local_expert_id
            ]:
                self.comm_op_list.append(dist.P2POp(dist.isend, src_tensor, dst_rank))

    @override
    def prepare_recv(self, expert_recv_info, updated_expert_map):
        """
        Prepares asynchronous receive tasks (irecv) for expert weights.
        This method is intended to be called when setting up communication
        for a specific layer.

        Args:
            expert_recv_info: A list of tuples, where each tuple is
                (source_rank, global_expert_id_to_recv).
            updated_expert_map: The new expert map
                (global_expert_id -> local_expert_id) for the current layer.
                This is used to determine which local slot the received
                expert will occupy.
        """
        for buffer_tensor_id, (
            recv_rank,
            global_expert_id_to_recv,
        ) in enumerate(expert_recv_info):
            for buffer_tensor in self.eplb_adaptor.buffer_tensor_list[buffer_tensor_id]:
                self.comm_op_list.append(
                    dist.P2POp(dist.irecv, buffer_tensor, recv_rank)
                )
            local_expert_to_replace = updated_expert_map[
                global_expert_id_to_recv
            ].item()
            self.recv_expert_list.append((local_expert_to_replace, buffer_tensor_id))

    def generate_expert_d2d_transfer_task(
        self, expert_send_info, expert_recv_info, updated_expert_map, layer_id
    ):
        """
        Generates the expert data-to-data transfer tasks(send and
        receiveoperations). for a given layer based on the provided
        send/receive information and the new expert map.

        Args:
            expert_send_info: List of (destination_rank, global_expert_id)
                             for experts to send.
            expert_recv_info: List of (source_rank, global_expert_id)
                              for experts to receive.
            updated_expert_map: The new expert map for the layer.
            layer_id: The ID of the MoE layer.
        """
        if not (expert_send_info or expert_recv_info):
            return
        self.updated_expert_map = updated_expert_map
        self.layer_id = layer_id
        self.comm_op_list = []
        self.prepare_send(expert_send_info, layer_id)
        self.prepare_recv(expert_recv_info, updated_expert_map)

    def async_expert_weight_transfer(self) -> None:
        """
        Initiates the asynchronous expert weight transfer by executing the
        prepared P2P communication operations.

        Args:
            reqs: A list to which the communication requests will be appended.
                  These requests can then be waited upon later.
        """
        # set asynchronous stream for d2d expert weight transfer
        if self.comm_op_list:
            ret_list: list[Work] = dist.batch_isend_irecv(self.comm_op_list)
            self.reqs.extend(ret_list)

    @override
    def update_expert_map_and_weight(self):
        """
        Waits for all pending communication requests to complete, then updates
        the expert map, logical-to-physical map, and the expert weights based
        on the received data.

        Args:
            reqs: A list of communication requests to wait for.
        """
        # Waiting for send/recv tasks finish
        for req in self.reqs:
            req.wait()

        # update expert_map
        # decouple adaptor and transfer
        self.eplb_adaptor.do_update_expert_map(self.layer_id, self.updated_expert_map)

        # update log2phy_map
        self.eplb_adaptor.do_update_log2phy_map(self.layer_id, self.updated_log2phy_map)

        # update expert weight
        buffer_tensor_id = 0
        for recv_expert_info in self.recv_expert_list:
            local_expert_to_replace, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(
                self.layer_id, local_expert_to_replace, buffer_tensor_id
            )

        self.clear_update_data()

    def clear_update_data(self):
        """
        Clears the internal lists and temporary data used for the current
        update cycle. This prepares the transfer for the next rearrangement
        cycle.
        """
        if self.comm_op_list is not None:
            self.comm_op_list.clear()
        self.recv_expert_list.clear()
        self.reqs.clear()
        self.updated_expert_map = None
        self.layer_id = -1

    def set_log2phy_map(self, log2phy_map_this_rank):
        """
        Sets the logical-to-physical expert map for the current rank.
        This map is used during the update phase.

        Args:
            log2phy_map_this_rank: The logical-to-physical map
                                   tensor specific to this rank.
        """
        self.updated_log2phy_map = log2phy_map_this_rank
