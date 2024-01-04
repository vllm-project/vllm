from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm._C import ops
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.utils import set_weight_attrs


class MoE(nn.Module):
    """a tensor-parallel MOE implementation that shards each expert across
    all ranks.

    Each expert's weights are sharded across all ranks. The forward pass
    will first expand and group the hidden states by experts, then compute
    the per-rank MLP output of each expert using grouped gemm, and finally
    reduce the output across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // tp_size

        self.gate = ReplicatedLinear(self.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     linear_method=None)

        self.w1s = nn.Parameter(
            torch.empty(self.num_total_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        device="cuda"))
        self.w2s = nn.Parameter(
            torch.empty(self.num_total_experts,
                        self.intermediate_size,
                        self.hidden_size,
                        device="cuda"))
        self.w3s = nn.Parameter(
            torch.empty(self.num_total_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        device="cuda"))

        set_weight_attrs(self.w1s, {
            "weight_loader": self.weight_loader,
            "tp_type": "column"
        })
        set_weight_attrs(self.w2s, {
            "weight_loader": self.weight_loader,
            "tp_type": "row"
        })
        set_weight_attrs(self.w3s, {
            "weight_loader": self.weight_loader,
            "tp_type": "column"
        })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        loaded_weight = loaded_weight.t()
        # The parallel dimension is 1 for column-parallel, and 0 for
        # row-parallel.
        parallel_dim = 1 if getattr(param, "tp_type", None) == "column" else 0
        param_data = param.data
        shard_size = param_data.shape[parallel_dim + 1]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(parallel_dim, start_idx,
                                             shard_size)
        assert param_data[expert_id].shape == loaded_weight.shape
        param_data[expert_id].copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Step 1: expand and permute hidden states and routing weights to group
        #        hidden states by experts.
        expanded_hidden_states, experts_range, expanded_weights, reverse_indices = \
            self.expand_and_permutate_hidden_states(
                hidden_states, selected_experts, routing_weights)

        # Step 2: compute the output of each expert.
        expanded_hidden_states = self.apply_experts_ffn(
            expanded_hidden_states, experts_range, self.w1s.data,
            self.w2s.data, self.w3s.data)

        # Step 3: apply weights to the output of each expert, and reduce
        # across ranks.
        expanded_hidden_states.mul_(expanded_weights.unsqueeze(-1))
        tensor_model_parallel_all_reduce(expanded_hidden_states)

        # Step 4: merge the output of each expert, according to the indices.
        return self.merge_expert_outputs(expanded_hidden_states,
                                         reverse_indices).view(
                                             batch_size, sequence_length,
                                             hidden_size)

    def expand_and_permutate_hidden_states(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand and group hidden states and routing weights according
        to the selected experts.

        Args:
            hidden_states (torch.Tensor): [batch_size, hidden_size]
                hidden states.
            selected_experts (torch.Tensor): [batch_size, top_k_experts]
                the indices of the selected experts.
            routing_weights (torch.Tensor): [batch_size, top_k_experts]
                the routing weights of the selected experts.

        Returns:
            expanded_hidden_states: [batch_size * top_k_experts, hidden_size]
                expanded hidden states that rows are grouped by experts.
            cum_experts_range: [num_experts + 1] the cumulative range of the
                experts in expanded_hidden_states, in the first dimension.
            expanded_weights: [batch_size * top_k_experts] the expanded
                expert weights for each row in expanded_hidden_states.
            reverse_indices: [batch_size * top_k_experts] the indices of each
                row in expanded_hidden_states which maps back to the original
                hidden states.
        """
        reverse_indices = torch.argsort(selected_experts.view(-1), dim=-1)
        cum_experts_range = torch.zeros(self.num_total_experts + 1,
                                        dtype=torch.int32,
                                        device=hidden_states.device)
        num_rows_per_expert = torch.zeros(self.num_total_experts,
                                          dtype=torch.int32,
                                          device=hidden_states.device)
        ops.bincount(selected_experts.view(-1), num_rows_per_expert)
        torch.cumsum(num_rows_per_expert, dim=0, out=cum_experts_range[1:])
        expanded_weights = routing_weights.view(-1)[reverse_indices]
        reverse_indices.div_(self.top_k, rounding_mode="floor")
        return hidden_states[
            reverse_indices], cum_experts_range, expanded_weights, reverse_indices

    def apply_experts_ffn(
        self,
        expanded_hidden_states: torch.
        Tensor,  # [batch_size * top_k_experts, hidden_size]
        cum_experts_range: torch.Tensor,  # [num_experts + 1]
        w1s: torch.Tensor,  # [num_experts, hidden_size, ffn_dim]
        w2s: torch.Tensor,  # [num_experts, ffn_dim, hidden_size]
        w3s: torch.Tensor,  # [num_experts, hidden_size, ffn_dim]
    ) -> torch.Tensor:  # [batch_size * top_k_experts, hidden_size]
        grouped_w1_out = grouped_matmul(expanded_hidden_states,
                                        cum_experts_range, w1s, "silu")
        grouped_w3_out = grouped_matmul(expanded_hidden_states,
                                        cum_experts_range, w3s)
        grouped_w1_out.mul_(grouped_w3_out)
        return grouped_matmul(grouped_w1_out, cum_experts_range, w2s)

    def merge_expert_outputs(
            self,
            expanded_hidden_states: torch.
        Tensor,  # [batch_size * top_k_experts, hidden_size]
            reverse_indices,  # [batch_size * top_k_experts]
    ) -> torch.Tensor:
        out = torch.zeros(expanded_hidden_states.shape[0] // self.top_k,
                          self.hidden_size,
                          device=expanded_hidden_states.device,
                          dtype=expanded_hidden_states.dtype)
        out.index_add_(0, reverse_indices, expanded_hidden_states)
        return out


# The following code is adapted from
# https://github.com/openai/triton/blob/main/python/tutorials/11-grouped-gemm.py
@triton.jit
def grouped_matmul_kernel(
    # [batch_size, k], where each group are stored compactly in the batch
    # dimension. The range of each group is specified in cumulative_m_range.
    group_a_ptr,
    # [num_groups, k, n]
    group_b_ptr,
    # [batch_size, n], where each group are stored compactly in the batch
    # dimension. The range of each group is specified in cumulative_m_range.
    group_c_ptr,
    # num of gemm problems
    group_size,
    # for each gemm problem with size <m, n, k>, m is stored in
    # cumulative_m_range[i + i] - cumulative_m_range[i].
    # n and k are the same for all problems.
    cumulative_m_range,
    n,
    k,
    # group_a_ptr.stride(0)
    stride_a0,
    # group_b_ptr.stride(1)
    stride_b1,
    # group_c_ptr.stride(0)
    stride_c0,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        a_offset = tl.load(cumulative_m_range + g)
        gm = tl.load(cumulative_m_range + g + 1) - a_offset
        gn = n
        gk = k
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end
               and tile_idx < last_problem_end + num_tiles):

            # pick up a tile from the current gemm problem
            k = gk
            a_ptr = group_a_ptr + a_offset * stride_a0
            b_ptr = group_b_ptr + g * k * n
            c_ptr = group_c_ptr + a_offset * stride_c0
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * stride_a0 + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * stride_b1 + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])

                a = tl.load(a_ptrs,
                            mask=(offs_k[None, :] < k - kk * BLOCK_SIZE_K) &
                            (offs_am[:, None] < gm),
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=(offs_k[:, None] < k - kk * BLOCK_SIZE_K) &
                            (offs_bn[None, :] < gn),
                            other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * stride_b1

            if ACTIVATION == "silu":
                accumulator = silu(accumulator)
            c = accumulator.to(group_c_ptr.dtype.element_ty)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_c0 * offs_cm[:, None] + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < gm) & (offs_cn[None, :] < gn)

            tl.store(c_ptrs, c, mask=c_mask)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


def grouped_matmul(input: torch.Tensor,
                   cumulative_group_range: torch.Tensor,
                   group_b_ptr: torch.Tensor,
                   activation: str = ""):
    """Performs a grouped matrix-matrix product of matrices stored in input
    and group_b_ptr.

    input is a tensor of shape [batch_size, k] where each group are stored
    compactly in the batch dimension. The range of each group is specified
    in cumulative_group_range. This allows the input to have fixed shape
    regardless of the group sizes.

    Args:
        input (torch.Tensor): [batch_size, k] compact input.
        cumulative_group_range (torch.Tensor): [num_groups + 1] the cumulative
            range of the groups in input.
        group_b_ptr (torch.Tensor): [num_groups, k, n] the second matrix.
        activation (str, optional): "" or "silu". Defaults to "".

    Returns:
        torch.Tensor: [batch_size, n] compact output where groups
            are stored compactly in the batch dimension.
    """
    device = torch.device('cuda')
    assert cumulative_group_range.shape[0] == group_b_ptr.shape[0] + 1
    group_size = cumulative_group_range.shape[0] - 1
    output = torch.zeros(input.shape[0],
                         group_b_ptr.shape[2],
                         device=device,
                         dtype=input.dtype)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    num_warps = 2
    NUM_SM = 128
    num_stages = 5
    # hand tune the block size for different problem sizes.
    if input.shape[0] >= 8:
        num_warps = 4
        BLOCK_SIZE_N = 128
    if input.shape[0] >= 32:
        num_warps = 4
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 128
    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], )
    grouped_matmul_kernel[grid](group_a_ptr=input,
                                group_b_ptr=group_b_ptr,
                                group_c_ptr=output,
                                group_size=group_size,
                                cumulative_m_range=cumulative_group_range,
                                n=group_b_ptr.shape[2],
                                k=group_b_ptr.shape[1],
                                stride_a0=input.stride(0),
                                stride_b1=group_b_ptr.stride(1),
                                stride_c0=output.stride(0),
                                ACTIVATION=activation,
                                BLOCK_SIZE_M=BLOCK_SIZE_M,
                                BLOCK_SIZE_N=BLOCK_SIZE_N,
                                BLOCK_SIZE_K=BLOCK_SIZE_K,
                                NUM_SM=NUM_SM,
                                num_warps=num_warps,
                                num_stages=num_stages),

    return output
