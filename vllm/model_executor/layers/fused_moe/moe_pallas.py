import torch
import torch.nn.functional as F
from torch_xla.experimental.custom_kernel import _histogram


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> torch.Tensor:
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
    """
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

    # NOTE(woosuk): The GMM Pallas kernel requires a different weight layout
    # from HF Transformers.
    w1 = w1.transpose(1, 2)
    w2 = w2.transpose(1, 2)

    x = hidden_states[token_indices]
    x = torch.ops.xla.gmm(x, w1, group_sizes)
    x = F.silu(x[..., :intermediate_size]) * x[..., intermediate_size:]
    x = torch.ops.xla.gmm(x, w2, group_sizes)
    x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)

    x = x * topk_weights.unsqueeze_(dim=-1)
    x = x.sum(dim=-2)
    x = x.reshape(orig_shape)
    return x
