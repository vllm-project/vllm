# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.forward_context import get_forward_context


class All2AllBase:

    def __init__(self, cpu_group, model):
        self.cpu_group = cpu_group

        # compute some common properties
        from vllm.distributed.parallel_state import (get_dp_group,
                                                     get_ep_group,
                                                     get_tp_group,
                                                     in_the_same_node_as)

        # all2all lives in ep group, which is merged from dp and tp group
        self.dp_group = get_dp_group()
        self.tp_group = get_tp_group()
        self.ep_group = get_ep_group()
        self.dp_rank = self.dp_group.rank_in_group
        self.dp_world_size = self.dp_group.world_size

        # all2all communication often has separate implementations for
        # intra-node and inter-node communication
        self.intranode = in_the_same_node_as(cpu_group, source_rank=0)
        self.internode = not self.intranode

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        pass


class NaiveAll2All(All2AllBase):
    """
    A naive implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group, model):
        super().__init__(cpu_group, model)

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
        assert (len(x.shape) == 2)
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)

        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]
        buffer[start:end, :].copy_(x)
        for idx in range(self.dp_world_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            self.dp_group.broadcast(buffer[start:end, :], idx)

        return buffer

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu

        hidden_states = self.naive_multicast(hidden_states,
                                             cu_tokens_across_dp_cpu)
        router_logits = self.naive_multicast(router_logits,
                                             cu_tokens_across_dp_cpu)
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu
        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]

        all_hidden_states = self.dp_group.all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states

    def destroy(self):
        pass
