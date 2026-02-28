# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Token-Routed MLP (I64) - Deterministic expert routing for vLLM.

INL Innovation: Routes tokens to specialized experts based on token ID.
    expert_id = token_id % num_experts

This is NOT standard MoE:
- No learned router, no softmax, no topk
- Deterministic I64 modulo routing = stable, 100% parallel
- Mu-guided routing: INL Dynamics mu vector biases expert selection
- SwiGLU experts with fused gate+up projection

Parallelism:
- TP (Tensor Parallel): each expert sharded on intermediate dimension
- EP (Expert Parallel): experts distributed across ranks, all-to-all dispatch
- TP + EP can be combined
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.token_routed_i64.fused_experts import (
    fused_token_routed_forward,
)


class TokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP - Deterministic I64 expert routing with TP + EP.

    Routes tokens to experts based on: expert_id = token_id % num_experts
    Mu-guided routing: mu from INL Dynamics can bias expert selection.

    Parallelism modes:
        TP only:  All experts on all ranks, intermediate dim sharded
        EP only:  Experts distributed across ranks, all-to-all dispatch
        TP + EP:  Both combined

    Weight shapes (per rank):
        gate_up_proj: [local_num_experts, hidden_size, 2 * intermediate_per_tp]
        down_proj:    [local_num_experts, intermediate_per_tp, hidden_size]
    """

    # Scale factor for base expert logits in mu-guided routing.
    # High value ensures deterministic I64 routing dominates unless
    # mu provides a strong override signal.
    _BASE_ROUTING_SCALE = 10.0

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        vocab_size: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        # TP sharding
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # EP sharding
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.use_ep = ep_size > 1

        if self.use_ep:
            assert num_experts % ep_size == 0, (
                f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
            )
            self.local_num_experts = num_experts // ep_size
            self.expert_offset = ep_rank * self.local_num_experts
        else:
            self.local_num_experts = num_experts
            self.expert_offset = 0

        # Expert dimensions
        self.expert_intermediate_size = intermediate_size // num_experts
        assert self.expert_intermediate_size % self.tp_size == 0, (
            f"expert_intermediate_size ({self.expert_intermediate_size}) "
            f"must be divisible by tp_size ({self.tp_size})"
        )
        self.intermediate_per_tp = self.expert_intermediate_size // self.tp_size

        # Expert weights - only LOCAL experts, sharded on intermediate
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.local_num_experts, hidden_size, 2 * self.intermediate_per_tp
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.local_num_experts, self.intermediate_per_tp, hidden_size)
        )

        # Mu-guided routing (replicated - small tensor)
        self.mu_router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.mu_router.weight)

        # Deterministic I64 token -> expert mapping
        self.register_buffer(
            "token_to_expert",
            torch.arange(vocab_size, dtype=torch.long) % num_experts,
        )

        # Init weights
        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj, a=5**0.5)

    def _route_tokens(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor | None,
        mu: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute expert IDs for each token (global expert IDs)."""
        num_tokens = x.shape[0]

        if token_ids is None:
            return torch.zeros(num_tokens, dtype=torch.long, device=x.device)

        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        base_expert_ids = self.token_to_expert[token_ids_clamped]

        # Mu-guided bias (INL innovation)
        if mu is not None:
            mu_logits = self.mu_router(mu)
            base_one_hot = F.one_hot(base_expert_ids, self.num_experts).float()
            combined_logits = base_one_hot * self._BASE_ROUTING_SCALE + mu_logits
            return combined_logits.argmax(dim=-1)

        return base_expert_ids

    def _forward_local(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process tokens through LOCAL experts only.
        expert_ids are local (0..local_num_experts-1).

        Uses fused dispatch: BMM for small batches (decode),
        chunked sort-and-matmul for large batches (prefill).
        """
        return fused_token_routed_forward(
            x,
            self.gate_up_proj,
            self.down_proj,
            expert_ids,
            self.local_num_experts,
            self.intermediate_per_tp,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with I64 deterministic routing.

        Args:
            x: [num_tokens, hidden_size]
            token_ids: [num_tokens] - I64 token IDs for routing
            mu: [num_tokens, hidden_size] - mu from INL Dynamics
        """
        # === I64 Routing (global expert IDs) ===
        expert_ids = self._route_tokens(x, token_ids, mu)

        if self.use_ep:
            output = self._forward_ep(x, expert_ids)
        else:
            output = self._forward_local(x, expert_ids)

        # === TP all_reduce (RowParallel equivalent) ===
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)

        return output

    def _forward_ep(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expert Parallel forward: all-to-all dispatch.

        1. Each rank determines which tokens go to which EP rank
        2. All-to-all exchange: send tokens to the rank that owns the expert
        3. Each rank processes its local tokens
        4. All-to-all exchange: send results back
        """
        num_tokens = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Which EP rank owns each expert
        ep_rank_for_token = expert_ids // self.local_num_experts

        # Count tokens going to each EP rank
        send_counts = torch.bincount(ep_rank_for_token, minlength=self.ep_size)

        # Gather counts from all ranks (all-to-all metadata)
        recv_counts = torch.empty_like(send_counts)
        torch.distributed.all_to_all_single(
            recv_counts,
            send_counts,
            group=self._get_ep_group(),
        )

        # Sort tokens by destination rank
        rank_order = ep_rank_for_token.argsort(stable=True)
        sorted_x = x[rank_order]
        sorted_expert_ids = expert_ids[rank_order]

        # All-to-all: send tokens to owning ranks
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        total_recv = sum(recv_splits)

        recv_x = torch.empty(total_recv, self.hidden_size, device=device, dtype=dtype)
        recv_expert_ids = torch.empty(total_recv, dtype=torch.long, device=device)

        torch.distributed.all_to_all_single(
            recv_x,
            sorted_x,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self._get_ep_group(),
        )
        torch.distributed.all_to_all_single(
            recv_expert_ids,
            sorted_expert_ids,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self._get_ep_group(),
        )

        # Convert global expert IDs to local
        local_expert_ids = recv_expert_ids - self.expert_offset

        # Process local experts
        local_output = self._forward_local(recv_x, local_expert_ids)

        # All-to-all: send results back
        # (reverse direction: recv_splits become send, send_splits become recv)
        result_sorted = torch.empty(
            num_tokens, self.hidden_size, device=device, dtype=dtype
        )
        torch.distributed.all_to_all_single(
            result_sorted,
            local_output,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self._get_ep_group(),
        )

        # Unsort to original token order
        output = torch.empty_like(result_sorted)
        output[rank_order] = result_sorted

        return output

    def _get_ep_group(self):
        """Get the EP process group. Lazy import to avoid circular deps."""
        from vllm.distributed import get_ep_group

        return get_ep_group().device_group

    def load_tp_weight(
        self,
        param_name: str,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> bool:
        """
        Load a checkpoint weight with TP + EP sharding.

        Checkpoint stores all experts at full intermediate size.
        We narrow to:
        - Our EP rank's local experts
        - Our TP rank's intermediate slice

        Args:
            param_name: "gate_up_proj" or "down_proj"
            param: the parameter to load into
            loaded_weight: full checkpoint weight [num_experts, ...]

        Returns:
            True if loaded successfully
        """
        # EP: select only our local experts
        if self.use_ep:
            end = self.expert_offset + self.local_num_experts
            loaded_weight = loaded_weight[self.expert_offset : end]

        if param_name == "gate_up_proj":
            # Checkpoint: [local_E, hidden, 2 * full_intermediate]
            full_inter = loaded_weight.shape[2] // 2
            per_tp = full_inter // self.tp_size
            offset = self.tp_rank * per_tp

            gate_full = loaded_weight[:, :, :full_inter]
            up_full = loaded_weight[:, :, full_inter:]

            gate_shard = gate_full[:, :, offset : offset + per_tp].contiguous()
            up_shard = up_full[:, :, offset : offset + per_tp].contiguous()

            with torch.no_grad():
                param.copy_(torch.cat([gate_shard, up_shard], dim=2))
            return True

        elif param_name == "down_proj":
            # Checkpoint: [local_E, full_intermediate, hidden]
            per_tp = loaded_weight.shape[1] // self.tp_size
            offset = self.tp_rank * per_tp

            shard = loaded_weight[:, offset : offset + per_tp, :].contiguous()

            with torch.no_grad():
                param.copy_(shard)
            return True

        return False
