import pytest
import torch

from vllm.model_executor.layers.activation import FastGELU, NewGELU, SiluAndMul
from vllm.utils import is_hip

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 4096, 5120, 13824]  # Arbitrary values for testing
SEEDS = [0]
DEVICES = [i for i in range(1 if torch.cuda.device_count() == 1 else 2)]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_silu_and_mul(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    gpu_id = f"cuda:{device}"
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device=gpu_id)
    layer = SiluAndMul()
    out = layer(x)
    ref_out = layer._forward(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_gelu_new(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    gpu_id = f"cuda:{device}"
    x = torch.randn(num_tokens, d, dtype=dtype, device=gpu_id)
    layer = NewGELU()
    out = layer(x)
    ref_out = layer._forward(x)
    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5)


# Reference default values of atol and rtol are from
# https://github.com/pytorch/pytorch/blob/6d96beb6bec24d73ee3f080bac54d2104068f675/test/test_transformers.py#L67
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
default_rtol = {
    torch.float16: 1e-3,
    torch.bfloat16: 1.6e-2,
    torch.float: 1.3e-6
}


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
def test_gelu_fast(
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: int,
) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    gpu_id = f"cuda:{device}"
    x = torch.randn(num_tokens, d, dtype=dtype, device=gpu_id)
    layer = FastGELU()
    out = layer(x)
    ref_out = layer._forward(x)
    atol = 1e-5 if not is_hip() else default_atol[out.dtype]
    rtol = 1e-5 if not is_hip() else default_rtol[out.dtype]
    assert torch.allclose(out, ref_out, atol=atol, rtol=rtol)
