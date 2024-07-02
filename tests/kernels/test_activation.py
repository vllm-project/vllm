from typing import Type

import pytest
import torch

from vllm.model_executor.layers.activation import (FastGELU, GeluAndMul,
                                                   NewGELU, SiluAndMul)

from .allclose_default import get_default_atol, get_default_rtol

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 4096, 5120, 13824]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu":
        layer = SiluAndMul()
    elif activation == "gelu":
        layer = GeluAndMul(approximate="none")
    elif activation == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
    out = layer(x)
    ref_out = layer.forward_native(x)
    # The SiLU and GELU implementations are equivalent to the native PyTorch
    # implementations, so we can do exact comparison.
    assert torch.allclose(out, ref_out, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("activation", [FastGELU, NewGELU])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_activation(
    activation: Type[torch.nn.Module],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation()
    out = layer(x)
    ref_out = layer.forward_native(x)
    assert torch.allclose(out,
                          ref_out,
                          atol=get_default_atol(out),
                          rtol=get_default_rtol(out))
