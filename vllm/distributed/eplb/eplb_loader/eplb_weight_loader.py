from functools import partial
from typing import Sequence, Iterable

import torch
from overrides import override
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed import P2POp, batch_isend_irecv, get_global_rank

from vllm.distributed.eplb.eplb_loader.abstract_loader import BaseLoader
from vllm.distributed.eplb.eplb_utils.eplb_utils import idx_local_to_global, get_ep_ranks_with_expert


class EplbWeightLoader(BaseLoader):

    def shuffle_layer(self,
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
                    for weight, buffer in zip(expert_weights,
                                              expert_weights_buffer):
                        buffer[dst].copy_(weight[src])

        p2p_ops: list[P2POp] = []

        # 2. Initiate sending of weights.
        p2p_ops = self.prepare_send(ep_group, ep_rank, expert_weights, local2global, new_indices, num_local_experts,
                                    old_indices,
                                    p2p_ops)

        # 3. Initiate receiving of weights.
        experts_recv_loc, p2p_ops = self.prepare_recv(ep_group, ep_rank, expert_weights_buffer, is_received_locally,
                                                      local2global, new_indices, num_local_experts, old_indices,
                                                      p2p_ops)

        # 4. Execute the P2P operations. The real communication happens here.
        self.send_recv(p2p_ops)

        # 5. Copy the weights from the buffer back to the original weights.
        self.update_weight(expert_weights, expert_weights_buffer, experts_recv_loc, is_received_locally, is_unchanged,
                           local2global, new_indices, num_local_experts)

    @override
    def update_weight(self, expert_weights, expert_weights_buffer, experts_recv_loc, is_received_locally, is_unchanged,
                      local2global, new_indices, num_local_experts):
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
    def send_recv(self, p2p_ops):
        if p2p_ops:
            reqs = batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

    @override
    def prepare_recv(self, ep_group, ep_rank, expert_weights_buffer, is_received_locally, local2global, new_indices,
                     num_local_experts, old_indices, p2p_ops):
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
                ) for weight in expert_weights_buffer
            ]
        return experts_recv_loc, p2p_ops

    @override
    def prepare_send(self, ep_group, ep_rank, expert_weights, local2global, new_indices, num_local_experts, old_indices,
                     p2p_ops):
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
                    ) for weight in expert_weights
                ]
        return p2p_ops
