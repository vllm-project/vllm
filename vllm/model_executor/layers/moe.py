from typing import Tuple

import torch

from torch import nn
import torch.nn.functional as F

import triton
import triton.language as tl

from vllm.model_executor.layers.linear import (ReplicatedLinear,
                                               ColumnParallelLinear)
                                        
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)


class MoE(nn.Module):

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate = ReplicatedLinear(self.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     linear_method=None)

        self.w1s = nn.Parameter(torch.empty(self.num_total_experts,
                                            self.hidden_size,
                                            self.intermediate_size))
        self.w2s = nn.Parameter(torch.empty(self.num_total_experts,
                                            self.intermediate_size,
                                            self.hidden_size))
        self.w3s = nn.Parameter(torch.empty(self.num_total_experts,
                                            self.hidden_size,
                                            self.intermediate_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expanded_hidden_states, experts_range, expanded_weights = \
            self.expand_and_permutate_hidden_states(
                hidden_states, selected_experts, routing_weights)

        expanded_hidden_states = self.grouped_mlp(expanded_hidden_states, 
                                             experts_range, self.w1s.weight, 
                                             self.w2s.weight, self.w3s.weight)
        
        expanded_hidden_states.mul_(expanded_weights)

        tensor_model_parallel_all_reduce(expanded_hidden_states)

        return self.merge_expert_outputs(expanded_hidden_states, selected_experts, experts_range)


    def expand_and_permutate_hidden_states(
            self,
            hidden_states: torch.Tensor, # [batch_size, hidden_size]
            selected_experts: torch.Tensor, # [batch_size, top_k_experts]
            routing_weights: torch.Tensor, # [batch_size, top_k_experts]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cum_experts_range = torch.zeros(self.num_total_experts + 1, dtype=torch.int32, device=hidden_states.device)
        num_rows_per_expert = torch.bincount(selected_experts.view(-1), minlength=self.num_total_experts)
        torch.cumsum(num_rows_per_expert, dim=0, out=cum_experts_range[1:])
        experts_indices = torch.argsort(selected_experts.view(-1), dim=-1)
        expanded_weights = routing_weights.view(-1)[experts_indices]
        return hidden_states[experts_indices.div_(self.top_k, rounding_mode="floor")], cum_experts_range, expanded_weights


    def grouped_mlp(
            self,
            expanded_hidden_states: torch.Tensor, # [batch_size * top_k_experts, hidden_size]
            experts_range: torch.Tensor, # [num_experts, 2]
            w1s: torch.Tensor, # [num_experts, hidden_size, ffn_dim]
            w2s: torch.Tensor, # [num_experts, ffn_dim, hidden_size]
            w3s: torch.Tensor, # [num_experts, hidden_size, ffn_dim]
        ) -> torch.Tensor: # [batch_size * top_k_experts, hidden_size]
        pass

    def merge_expert_outputs(
            self,
            expanded_hidden_states: torch.Tensor, # [batch_size * top_k_experts, hidden_size]
            selected_experts: torch.Tensor, # [batch_size, top_k_experts]
            experts_range: torch.Tensor, # [num_experts, 2]
        ) -> torch.Tensor:
        pass