import pytest
import torch

from vllm.model_executor.layers.activation import FastGELU, NewGELU, SiluAndMul

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 4096, 5120, 13824]  # Arbitrary values for testing
SEEDS = [0]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_silu_and_mul(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device="cuda")
    layer = SiluAndMul()
    out = layer(x)
    ref_out = layer._forward(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_gelu_new(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn(num_tokens, d, dtype=dtype, device="cuda")
    layer = NewGELU()
    out = layer(x)
    ref_out = layer._forward(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_gelu_fast(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randn(num_tokens, d, dtype=dtype, device="cuda")
    layer = FastGELU()
    out = layer(x)
    ref_out = layer._forward(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)
