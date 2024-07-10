import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from torch_xla.experimental.custom_kernel import _histogram, gmm


# @torch.compile(backend="openxla")
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
        hidden_states: [batch_size, seq_len, hidden_size]
        w1: [num_experts, hidden_size, intermediate_size * 2]
        w2: [num_experts, intermediate_size, hidden_size]
        gating_output: [batch_size, seq_len, num_experts]
    """
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

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
    x = gmm(x, w1, group_sizes)
    x = F.silu(x[..., :intermediate_size]) * x[..., intermediate_size:]
    x = gmm(x, w2, group_sizes)
    x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)

    x = x * topk_weights.unsqueeze_(dim=-1)
    x = x.sum(dim=-2)
    x = x.reshape(orig_shape)
    return x


if __name__ == "__main__":
    BATCH_SIZE = 1
    SEQ_LEN = 1024
    HIDDEN_SIZE = 1024
    INTERMEDIATE_SIZE = 1536
    NUM_EXPERTS = 8
    TOPK = 2
    DTYPE = torch.bfloat16
    device = xm.xla_device()

    x = torch.randn(BATCH_SIZE,
                    SEQ_LEN,
                    HIDDEN_SIZE,
                    dtype=DTYPE,
                    device=device)
    gating_output = torch.randn(BATCH_SIZE,
                                SEQ_LEN,
                                NUM_EXPERTS,
                                dtype=torch.float,
                                device=device)
    w1 = torch.randn(NUM_EXPERTS,
                     HIDDEN_SIZE,
                     INTERMEDIATE_SIZE * 2,
                     dtype=DTYPE,
                     device=device)
    w2 = torch.randn(NUM_EXPERTS,
                     INTERMEDIATE_SIZE,
                     HIDDEN_SIZE,
                     dtype=DTYPE,
                     device=device)
    output = fused_moe(
        x,
        w1,
        w2,
        gating_output,
        TOPK,
        renormalize=False,
    )
    output = output.cpu()
