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

        self.ws = nn.Parameter(
            torch.empty(self.num_total_experts,
                        2 * self.intermediate_size,
                        self.hidden_size,
                        device="cuda"))
        self.w2s = nn.Parameter(
            torch.empty(self.num_total_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        device="cuda"))

        set_weight_attrs(self.ws, {
            "weight_loader": self.weight_loader,
        })
        set_weight_attrs(self.w2s, {
            "weight_loader": self.weight_loader,
        })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank+1) * shard_size)
        if weight_name.endswith("w1.weight"):
            param_data[expert_id,0:shard_size,:] = loaded_weight[shard,:]
        if weight_name.endswith("w3.weight"):
            param_data[expert_id,shard_size:2*shard_size,:] = loaded_weight[shard,:]
        if weight_name.endswith("w2.weight"):
            param_data[expert_id,:,:] = loaded_weight[:,shard]


    def fused_moe_infer(self, hidden_states: torch.Tensor,
                        selected_experts: torch.Tensor,
                        routing_weights: torch.Tensor) -> torch.Tensor:
        return fused_moe(hidden_states,
                         # self.w1s,
                         self.ws,
                         self.w2s,
                         # self.w3s,
                         routing_weights,
                         selected_experts,
                         inplace=True)

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

        final_hidden_states = self.fused_moe_infer(hidden_states,
                                                   selected_experts,
                                                   routing_weights)
        
        final_hidden_states = tensor_model_parallel_all_reduce(
            final_hidden_states)

        return final_hidden_states.view(batch_size, sequence_length,
                                        hidden_size)



@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_weight,
    stride_token_id,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using token and expert matrices.
    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can be any shape representing batches and K is the feature dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding, topk is the number of times each token is repeated,
        and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens, repeated topk times and arranged by the expert index they are assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A.
    This kernel performs the multiplication of a token by its corresponding expert matrix as determined by `expert_ids`. The sorting of `sorted_token_ids`
    by expert index and padding ensures divisibility by BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    #
    off_experts = tl.load(expert_ids_ptr + pid_m) * stride_be
    b_ptrs = b_ptr + off_experts + (offs_k[:, None] * stride_bk +
                                    offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_weight,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# TODO: rewrite in CPP
def alig_block_size(
        topk_ids: torch.Tensor, block_size: int,
        num_experts: int) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.
    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.
    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size. 
    Padding ensures that during block matrix multiplication, the dimensions align correctly.
    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [1, 2, 3, 4] at the end, resulting in [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3 | 1, 2, 3, 4].
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 13, 1, 7, 11, 14, 2, 5, 8, 15]. 
        Tokens 12-15 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    cnts = torch.zeros(topk_ids.shape[0],
                       num_experts,
                       dtype=topk_ids.dtype,
                       device=topk_ids.device)
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    tokens_per_expert_post_alig = torch.floor_divide(
        tokens_per_expert + block_size - 1, block_size) * block_size

    cumsum = tokens_per_expert_post_alig.cumsum(0)
    num_tokens_post_padded = cumsum[-1].clone()
    max_tokens_post_padded = (
        topk_ids.numel() + num_experts *
        (block_size - 1)) if topk_ids.numel() > num_experts else (
            topk_ids.numel() + 1) * block_size

    # we just store the expert id of each single block but each token,
    # as each token in the same block will be process by the same expert.
    expert_ids = torch.zeros(max(
        (max_tokens_post_padded + block_size - 1) // block_size + 1,
        num_experts),
                             dtype=topk_ids.dtype,
                             device=topk_ids.device)

    cumsum.div_(block_size, rounding_mode="floor")
    ones = torch.ones_like(expert_ids)
    expert_ids.scatter_add_(0, cumsum, ones)
    expert_ids = expert_ids.cumsum(0)

    cumsum = (tokens_per_expert_post_alig - tokens_per_expert).cumsum(0)

    padded_tokens = torch.zeros(max_tokens_post_padded - topk_ids.numel(),
                                dtype=topk_ids.dtype,
                                device=topk_ids.device)
    ones = torch.ones_like(padded_tokens)
    padded_tokens.scatter_add_(0, cumsum[:-1], ones)
    padded_tokens = padded_tokens.cumsum(0)

    sorted_token_ids = torch.cat([topk_ids.view(-1), padded_tokens]).argsort()

    return sorted_token_ids, expert_ids, num_tokens_post_padded


def fused_moe(hidden_states: torch.Tensor,
              w1: torch.Tensor,
              w2: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              inplace=False):
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of weights, w1 and w2, and top-k gating mechanism.
    We used three shared cache variables across all layers to save gpu memory, which is more effective in a static graph context.
    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - topk_weights (torch.Tensor): The weights for the top-k selected experts.
    - topk_ids (torch.Tensor): The indices of the top-k selected experts.
    - inplace (bool): If True, perform the operation in-place. Defaults to False.
    
    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Incompatible dimensions"
    assert hidden_states.is_contiguous(), "Matrix A must be contiguous"
    assert w1.is_contiguous(), "Matrix B must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    M, K = hidden_states.shape
    E, N, K = w1.shape

    config = {
        'BLOCK_SIZE_M': 64,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 32,
        'GROUP_SIZE_M': 8
    }

    if topk_ids.numel() <= w1.shape[0]:
        config = {
            'BLOCK_SIZE_M': 16,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 1
        }

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = alig_block_size(
        topk_ids, config['BLOCK_SIZE_M'], E)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    fused_moe_kernel[grid](
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        M,
        N,
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        hidden_states.stride(0),
        hidden_states.stride(1),
        w1.stride(0),
        w1.stride(2),
        w1.stride(1),
        intermediate_cache1.stride(1),
        intermediate_cache1.stride(2),
        topk_weights.stride(1),
        sorted_token_ids.stride(0),
        MUL_ROUTED_WEIGHT=False,
        top_k=topk_ids.shape[1],
        compute_type=tl.bfloat16
        if hidden_states.dtype == torch.bfloat16 else tl.float16,
        **config,
    )

    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(w2.shape[1], META['BLOCK_SIZE_N']), )
    fused_moe_kernel[grid](
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        M,
        w2.shape[1],
        w2.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        intermediate_cache2.stride(0),
        intermediate_cache2.stride(1),
        w2.stride(0),
        w2.stride(2),
        w2.stride(1),
        intermediate_cache3.stride(1),
        intermediate_cache3.stride(2),
        topk_weights.stride(1),
        sorted_token_ids.stride(0),
        MUL_ROUTED_WEIGHT=True,
        top_k=1,  #
        compute_type=tl.bfloat16
        if hidden_states.dtype == torch.bfloat16 else tl.float16,
        **config,
    )
    if inplace:
        return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                         dim=1,
                         out=hidden_states)
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                     dim=1)