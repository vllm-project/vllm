import pytest
import torch

# ruff: noqa: F401
import vllm._C

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [16, 67, 768, 2048, 5120, 5137, 8192,
                8193]  # Arbitrary values for testing
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]
SCALE = [0.1, 0.5, 0.8, 1.2, 2.1]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_scaled_int8_quant(num_tokens: int, hidden_size: int,
                                   dtype: torch.dtype, seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    int8_traits = torch.iinfo(torch.int8)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000

    x_token_max, _ = x.max(dim=1)
    x_token_max = x_token_max.to(dtype=torch.float32)
    scales = (x_token_max / float(127.0))[:, None].to(device="cuda",
                                                      dtype=torch.float32)
    torch_out = (x / scales).round().clamp(int8_traits.min,
                                           int8_traits.max).to(torch.int8)

    ops_out = torch.empty_like(x, dtype=torch.int8, device="cuda")
    scales_out = torch.empty_like(scales, dtype=torch.float32, device="cuda")
    torch.ops._C.dynamic_scaled_int8_quant(ops_out, x, scales_out)

    assert torch.allclose(scales_out, scales)
    assert torch.allclose(torch_out, ops_out,
                          atol=1)  # big atol to account for rounding errors


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_static_scaled_int8_quant(num_tokens: int, hidden_size: int,
                                  dtype: torch.dtype, seed: int,
                                  scale: float) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    int8_traits = torch.iinfo(torch.int8)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000

    out1 = (x / scale).round().clamp(int8_traits.min,
                                     int8_traits.max).to(torch.int8)
    out2 = torch.empty_like(x, dtype=torch.int8)
    scale_argument = torch.tensor([scale], dtype=torch.float32, device="cuda")

    torch.ops._C.static_scaled_int8_quant(out2, x, scale_argument)
    assert torch.allclose(out1, out2,
                          atol=1)  # big atol to account for rounding errors
