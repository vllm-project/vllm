import pytest
import torch
import torch.nn as nn

from vllm import layernorm_ops

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [67, 768, 2048, 5120, 8192]  # Arbitrary values for testing
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]


class RefRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        weight = torch.empty(hidden_size)
        weight.normal_(mean=1.0, std=0.1)
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(hidden_size**-0.5)
    x = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x.uniform_(-scale, scale)
    ref = RefRMSNorm(hidden_size).to(dtype).cuda()

    out = torch.empty_like(x)
    layernorm_ops.rms_norm(
        out,
        x,
        ref.weight.data,
        ref.variance_epsilon,
    )
    ref_out = ref(x)
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-5)
