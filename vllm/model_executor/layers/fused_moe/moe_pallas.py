# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
import torch_xla.experimental.custom_kernel  # noqa: F401


def _histogram(input: torch.Tensor, min: int, max: int) -> torch.Tensor:
    """
  Compute the histogram of a int32 tensor. The bin edges are defined by the
  min and max values, with step = 1.
  """
    assert input.dtype == torch.int32, "input must be of torch.int32 dtype."
    assert min <= max, "min must be less than or equal to max."

    def searchsorted(sorted_sequence: torch.Tensor,
                     values_to_search: torch.Tensor) -> torch.Tensor:
        return (sorted_sequence.unsqueeze(1) == values_to_search).sum(dim=1)

    bin_edges = torch.linspace(min, max, max - min + 1,
                               dtype=input.dtype).to(input.device)
    return searchsorted(bin_edges, input).to(torch.int32)


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
    renormalize: bool = False,
) -> torch.Tensor:
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
    """
    assert expert_map is None, "expert_map is not supported for pallas MoE."
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    device = hidden_states.device
    dtype = hidden_states.dtype
    assert (num_tokens * topk) % 16 == 0, (
        "The Pallas GMM kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens * topk}")

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, num_experts)
    topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    topk_weights, topk_indices = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    topk_indices = topk_indices.flatten()
    topk_argsort_indices = topk_indices.argsort()
    topk_argsort_revert_indices = topk_argsort_indices.argsort()
    token_indices = torch.arange(num_tokens,
                                 device=device).repeat_interleave(topk)
    token_indices = token_indices[topk_argsort_indices]
    group_sizes = _histogram(topk_indices.to(torch.int32), 0, num_experts - 1)

    x = hidden_states[token_indices]
    x = torch.ops.xla.gmm(x, w1, group_sizes, transpose_rhs=True)
    x = F.silu(x[..., :intermediate_size]) * x[..., intermediate_size:]
    x = torch.ops.xla.gmm(x, w2, group_sizes, transpose_rhs=True)
    x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)

    x = x * topk_weights.unsqueeze(dim=-1)
    x = x.sum(dim=-2)
    x = x.reshape(orig_shape)
    return x
