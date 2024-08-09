import pytest
import torch

from tests.kernels.quant_utils import ref_dynamic_per_token_quant
from tests.kernels.utils import opcheck
from vllm._custom_ops import scaled_int8_quant

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [16, 67, 768, 2048, 5120, 5137, 8192,
                8193]  # Arbitrary values for testing
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]
SCALE = [0.1, 0.5, 0.8, 1.2, 2.1]


def opcheck_int8_quant(output, input, scale=None):
    if scale is not None:
        opcheck(torch.ops._C.static_scaled_int8_quant, (output, input, scale))
    else:
        scale = torch.empty((input.numel() // input.shape[-1], 1),
                            device=input.device,
                            dtype=torch.float32)
        opcheck(torch.ops._C.dynamic_scaled_int8_quant, (output, input, scale))


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

    torch.testing.assert_close(ops_scales, ref_scales)
    torch.testing.assert_close(
        ops_out, ref_out, atol=1,
        rtol=0.0)  # big atol to account for rounding errors

    opcheck_int8_quant(ops_out, x)


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

    torch.testing.assert_close(
        out1, out2, atol=1,
        rtol=0.0)  # big atol to account for rounding errors

    opcheck_int8_quant(out2, x, scale)
