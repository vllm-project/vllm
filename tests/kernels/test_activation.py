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

# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float: 1.3e-6}

def get_rtol(computed_value: torch.Tensor, ref_value: torch.Tensor) -> float:
    deviation = ref_value - computed_value
    deviation = torch.abs(deviation / ref_value)
    # Fill in the nans with the default rtol
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()


def get_atol(computed_value: torch.Tensor, ref_value: torch.Tensor) -> float:
    deviation = ref_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol


def get_tolerances(computed_value: torch.Tensor, ref_value: torch.Tensor):
    """Returns the absolute and relative tolerances for comparing two tensors."""
    atol = get_atol(ref_value, computed_value)
    rtol = get_rtol(ref_value, computed_value)

    atol = min(atol, default_atol[computed_value.dtype])
    rtol = min(rtol, default_rtol[computed_value.dtype])
    # torch.isclose() has weird behavior around see:
    # https://github.com/pytorch/pytorch/issues/102400
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    return atol, rtol


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
    atol = 1e-5
    rtol = 1e-5
    if is_hip():
        atol, rtol = get_tolerances(out, ref_out)
    assert torch.allclose(out, ref_out, atol=atol, rtol=rtol)