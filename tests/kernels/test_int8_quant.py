import pytest
import torch

from tests.kernels.quant_utils import ref_dynamic_per_token_quant
from vllm._custom_ops import scaled_int8_quant

DTYPES = [torch.float]
HIDDEN_SIZES = [16, 32, 64, 67, 128, 768, 2048, 5120, 5137, 8192,
                8193]  # Arbitrary values for testing
NUM_TOKENS = [1, 7, 83, 4096, 16384]  # Arbitrary values for testing
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

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000

    # reference
    ref_out, ref_scales = ref_dynamic_per_token_quant(x, torch.int8)
    # kernel
    ops_out, ops_scales = scaled_int8_quant(x)

    assert torch.allclose(ops_scales, ref_scales)
    assert torch.allclose(ops_out, ref_out,
                          atol=1)  # big atol to account for rounding errors

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_scaled_int8_azp_quant(num_tokens: int, hidden_size: int,
                                       dtype: torch.dtype, seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    int8_traits = torch.iinfo(torch.int8)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000 - 300

    x_token_max, _ = x.to(dtype=torch.float32).max(dim=1, keepdim=True)
    x_token_min, _ = x.to(dtype=torch.float32).min(dim=1, keepdim=True)

    # calculate scale and azp, and adjust the range
    scales = (x_token_max - x_token_min) / torch.tensor(255.0)
    azps = torch.round(x_token_min / scales + 128.0).to(torch.int32)

    torch_out = (x / scales - azps).round().to(torch.int8)
    assert torch_out.min() >= int8_traits.min and torch_out.max() <= int8_traits.max

    ops_out = torch.empty_like(x, dtype=torch.int8)
    scales_out = torch.empty_like(scales, dtype=torch.float32)
    azp_out = torch.empty_like(azps, dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(ops_out, x, scales_out, azp_out)

    if (not torch.allclose(scales_out, scales)):
        print(torch.argmax(torch.abs(scales_out - scales)))
    assert torch.allclose(scales_out, scales)
    assert torch.allclose(azp_out, azps, atol=1)  # azp rounding error
    assert torch.allclose(torch_out, ops_out, atol=1)  # azp rounding error


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
    scale = torch.tensor([scale], dtype=torch.float32, device="cuda")

    out1 = (x / scale).round().clamp(int8_traits.min,
                                     int8_traits.max).to(torch.int8)
    out2, _ = scaled_int8_quant(x, scale)

    assert torch.allclose(out1, out2,
                          atol=1)  # big atol to account for rounding errors
