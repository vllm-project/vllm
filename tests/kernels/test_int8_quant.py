import pytest
import torch

from vllm._C import ops

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [16, 67, 768, 2048, 5120, 8192]  # Arbitrary values for testing
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]
SCALE = [0.1, 0.5, 0.8, 1.2, 2.1]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_quant(num_tokens: int, hidden_size: int, dtype: torch.dtype,
               seed: int, scale: float) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000

    out1 = (x / scale).round().clamp(
        torch.iinfo(torch.int8).min,
        torch.iinfo(torch.int8).max).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    scale_argument = torch.tensor([scale], dtype=torch.float32, device="cuda")

    ops.static_scaled_int8_quant(out2, x, scale_argument)
    assert torch.allclose(out1, out2,
                          atol=1)  # big atol to account for rounding errors
