import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import (ref_dynamic_per_tensor_fp8_quant,
                                       ref_dynamic_per_token_quant)

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [1, 2, 3, 4, 16, 67, 768, 2048, 5120, 5137, 8192,
                8193]  # Arbitrary values for testing
HIDDEN_SIZES += list(range(1024, 1033))  # vectorized conversion edge cases
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_token_fp8_quant(num_tokens: int, hidden_size: int,
                                     dtype: torch.dtype, seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype,
                   device="cuda") + 1e-6  # avoid nans

    ref_out, ref_scales = ref_dynamic_per_token_quant(x, torch.float8_e4m3fn)
    ops_out, ops_scales = ops.scaled_fp8_quant(x,
                                               use_per_token_if_dynamic=True)

    assert torch.allclose(ref_scales, ops_scales)
    assert torch.allclose(ref_out.to(dtype=torch.float32),
                          ops_out.to(dtype=torch.float32))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant(num_tokens: int, hidden_size: int,
                                      dtype: torch.dtype, seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")

    ref_out, ref_scale = ref_dynamic_per_tensor_fp8_quant(x)
    ops_out, ops_scale = ops.scaled_fp8_quant(x)

    assert torch.allclose(ref_scale, ops_scale)
    assert torch.allclose(ref_out.to(dtype=torch.float32),
                          ops_out.to(dtype=torch.float32))
