import pytest
import torch
import torch.nn as nn

from vllm import fused_kernels

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [67, 768, 2048, 5120, 8192]  # Arbitrary values for testing
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]
SCALE = [0.1, 0.5, 0.8, 1.2, 2.1]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_dequant_add_residual(num_tokens: int, hidden_size: int,
                              dtype: torch.dtype, seed: int,
                              scale: float) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    s = float(hidden_size**-0.5)
    residual = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (num_tokens, hidden_size),
        dtype=torch.int32,
        device="cuda",
    )

    residual.uniform_(-s, s)
    out1 = (x * scale + residual).to(dtype)
    out2 = torch.empty_like(x, dtype=dtype)
    fused_kernels.invoke_dequant_add_residual(out2, x, residual, scale)
    assert torch.allclose(out1, out2,
                          atol=0.001), f"diff: {torch.max(out1 - out2)}"


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_per_token_dequant_add_residual(num_tokens: int, hidden_size: int,
                                        dtype: torch.dtype, seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    s = float(hidden_size**-0.5)
    residual = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    x = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (num_tokens, hidden_size),
        dtype=torch.int32,
        device="cuda",
    )
    scale = torch.rand(num_tokens, 1, dtype=torch.float32, device="cuda")

    residual.uniform_(-s, s)
    out1 = (x * scale + residual).to(dtype)
    scale = torch.squeeze(scale)
    out2 = torch.empty_like(x, dtype=dtype)
    fused_kernels.invoke_dequant_add_residual(out2, x, residual, scale)
    assert torch.allclose(out1, out2,
                          atol=0.001), f"diff: {torch.max(out1 - out2)}"


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_dequant(num_tokens: int, hidden_size: int, dtype: torch.dtype,
                 seed: int, scale: float) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (num_tokens, hidden_size),
        dtype=torch.int32,
        device="cuda",
    )

    out1 = (x * scale).to(dtype)
    out2 = torch.empty_like(x, dtype=dtype)
    fused_kernels.invoke_dequant(out2, x, scale)
    assert torch.allclose(out1, out2, atol=0.001)


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

    out1 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    fused_kernels.invoke_quant(out2, x, scale)
    assert torch.allclose(out1, out2, atol=1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_per_token_quant(num_tokens: int, hidden_size: int, dtype: torch.dtype,
                         seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000

    scale1 = torch.max(x, dim=1)[0].to(torch.float32) / 127.0
    out1 = (x / scale1.view(-1, 1)).round().clamp(-128, 127).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    scale2 = torch.empty(num_tokens, dtype=torch.float32, device="cuda")
    fused_kernels.invoke_quant(out2, x, scale2)
    assert torch.allclose(out1, out2, atol=1)
