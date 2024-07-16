import pytest
import torch

import vllm._custom_ops as ops
from quant_utils import ref_dynamic_per_token_quant 

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [16, 67, 768, 2048, 5120, 5137, 8192,
                8193]  # Arbitrary values for testing
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]
SCALE = [0.1, 0.5, 0.8, 1.2, 2.1]

#@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
#@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
#@pytest.mark.parametrize("dtype", DTYPES)
#@pytest.mark.parametrize("seed", SEEDS)
#@torch.inference_mode()
#def test_dynamic_per_token_fp8_quant(num_tokens: int, hidden_size: int,
#                                  dtype: torch.dtype, seed: int) -> None:
#    torch.random.manual_seed(seed)
#    torch.cuda.manual_seed(seed)
#
#    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000
#
#    ref_out, ref_scales = ref_dynamic_per_token_quant(x, torch.float8_e4m3fn)
#    ops_out, ops_scales = ops.dynamic_per_token_fp8_quant(x)
#
#    assert torch.allclose(ref_scales, ops_scales)
#    assert torch.allclose(ref_out, ops_out)

@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant(num_tokens: int, hidden_size: int,
                                  dtype: torch.dtype, seed: int) -> None:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    fp8_traits = torch.iinfo(torch.float8_e4m3fn)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000

    # reference
    ref_scale = (x.abs().max() / float(fp8_traits.max))[:, None].to(device="cuda",
            dtype=torch.float32)
    ref_out = (x / ref_scale).round().clamp(fp8_traits.min, fp8_traits.max).to(torch.float8_e4m3fn)
    # kernel
    ops_out, ops_scale = ops.scaled_fp8_quant(x)

    assert torch.allclose(ref_scale, ops_scale)
    assert torch.allclose(ref_out, ops_scale)
