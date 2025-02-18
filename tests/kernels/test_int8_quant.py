# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.kernels.quant_utils import ref_dynamic_per_token_quant
from tests.kernels.utils import opcheck
from vllm._custom_ops import scaled_int8_quant
from vllm.platforms import current_platform

DTYPES = [torch.half, torch.bfloat16, torch.float]
HIDDEN_SIZES = [16, 67, 768, 5137, 8193]  # Arbitrary values for testing
NUM_TOKENS = [1, 7, 83, 4096]  # Arbitrary values for testing
SEEDS = [0]
SCALE = [0.1, 2.1]


def opcheck_int8_quant_static(output, input, scale, azp=None):
    if azp is None:
        opcheck(torch.ops._C.static_scaled_int8_quant,
                (output, input, scale, None))
    else:
        opcheck(torch.ops._C.static_scaled_int8_quant,
                (output, input, scale, azp))


def opcheck_int8_quant_dynamic(output, input, symmetric=True):
    scale = torch.empty((input.numel() // input.shape[-1], 1),
                        device=input.device,
                        dtype=torch.float32)
    if symmetric:
        opcheck(torch.ops._C.dynamic_scaled_int8_quant,
                (output, input, scale, None))
    else:
        azp = torch.empty((input.numel() // input.shape[-1], 1),
                          device=input.device,
                          dtype=torch.int32)
        opcheck(torch.ops._C.dynamic_scaled_int8_quant,
                (output, input, scale, azp))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_scaled_int8_quant(num_tokens: int, hidden_size: int,
                                   dtype: torch.dtype, seed: int) -> None:
    current_platform.seed_everything(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000

    # reference
    ref_out, ref_scales = ref_dynamic_per_token_quant(x, torch.int8)
    # kernel
    ops_out, ops_scales, _ = scaled_int8_quant(x)

    torch.testing.assert_close(ops_scales, ref_scales)
    # big atol to account for rounding errors
    torch.testing.assert_close(ops_out, ref_out, atol=1, rtol=0.0)

    opcheck_int8_quant_dynamic(ops_out, x)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_scaled_int8_azp_quant(num_tokens: int, hidden_size: int,
                                       dtype: torch.dtype, seed: int) -> None:
    current_platform.seed_everything(seed)
    int8_traits = torch.iinfo(torch.int8)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype,
                   device="cuda") * 1000 - 300

    x_token_max, _ = x.to(dtype=torch.float32).max(dim=1, keepdim=True)
    x_token_min, _ = x.to(dtype=torch.float32).min(dim=1, keepdim=True)

    # calculate scale and azp, and adjust the range
    scales = (x_token_max - x_token_min) / torch.tensor(255.0)
    azps = torch.round(torch.tensor(-128.0) - x_token_min / scales).to(
        torch.int32)

    torch_out = ((x / scales).round() + azps).clamp(
        int8_traits.min, int8_traits.max).to(torch.int8)
    assert torch_out.min() >= int8_traits.min and torch_out.max(
    ) <= int8_traits.max

    ops_out, scales_out, azp_out = scaled_int8_quant(x, symmetric=False)

    if (not torch.allclose(scales_out, scales)):
        print(torch.argmax(torch.abs(scales_out - scales)))
    torch.testing.assert_close(scales_out, scales)
    # big atol to account for rounding errors
    torch.testing.assert_close(azp_out, azps, atol=1, rtol=0.0)
    # if AZP is off by 1, after rounding-to-even, the output may be off by 2
    torch.testing.assert_close(ops_out, torch_out, atol=2, rtol=0.0)

    opcheck_int8_quant_dynamic(ops_out, x, False)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@torch.inference_mode()
def test_static_scaled_int8_quant(num_tokens: int, hidden_size: int,
                                  dtype: torch.dtype, seed: int,
                                  scale: float) -> None:
    current_platform.seed_everything(seed)
    int8_traits = torch.iinfo(torch.int8)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") * 1000
    scale_arg = torch.tensor([scale], dtype=torch.float32, device="cuda")

    out1 = (x / scale_arg).round().clamp(int8_traits.min,
                                         int8_traits.max).to(torch.int8)
    out2, scale2, _ = scaled_int8_quant(x, scale_arg)
    assert scale2 is scale_arg

    # big atol to account for rounding errors
    torch.testing.assert_close(out1, out2, atol=1, rtol=0.0)

    opcheck_int8_quant_static(out2, x, scale_arg)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("scale", SCALE)
@pytest.mark.parametrize("azp", [-255, 54])
@torch.inference_mode()
def test_static_scaled_int8_azp_quant(num_tokens: int, hidden_size: int,
                                      dtype: torch.dtype, seed: int,
                                      scale: float, azp: int) -> None:
    current_platform.seed_everything(seed)
    int8_traits = torch.iinfo(torch.int8)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype,
                   device="cuda") * 1000 - 300

    out1 = ((x / scale).round() + azp).clamp(int8_traits.min,
                                             int8_traits.max).to(torch.int8)
    scale_arg = torch.tensor([scale], dtype=torch.float32, device="cuda")
    azp_arg = torch.tensor([azp], dtype=torch.int32, device="cuda")

    out2, scale2, azp2 = scaled_int8_quant(x,
                                           scale_arg,
                                           azp_arg,
                                           symmetric=False)
    assert scale2 is scale_arg
    assert azp2 is azp_arg

    # big atol to account for rounding errors
    torch.testing.assert_close(out1, out2, atol=1, rtol=0.0)

    opcheck_int8_quant_static(out2, x, scale_arg, azp_arg)


@pytest.mark.parametrize("is_max", [True, False])
@torch.inference_mode()
def test_static_scaled_int8_azp_quant_saturating_cast(is_max: bool) -> None:
    # Test that the saturating cast works correctly for values near i32 max/min

    from numpy import inf, nextafter

    int32_traits = torch.iinfo(torch.int32)
    val = float(int32_traits.max if is_max else int32_traits.min)

    x_vals = [[
        nextafter(val, inf), val + 1, val, val - 1,
        nextafter(val, -inf)
    ]]
    x = torch.tensor(x_vals, dtype=torch.float32, device="cuda")

    # The calculation in the kernel is: cast<int8>(cast<int32>(x / scale) + azp)
    # where cast<T> is a saturating cast to type T.
    # Scale is set to 1.0 so that the input values are the ones that are cast.
    # AZP is set to 0 to make sure the int8 saturating cast is tested as well.
    scale = torch.scalar_tensor(1.0, dtype=torch.float32, device="cuda")
    azp = torch.scalar_tensor(0, dtype=torch.int32, device="cuda")

    int8_traits = torch.iinfo(torch.int8)
    val_i8 = int8_traits.max if is_max else int8_traits.min
    expected = torch.full((1, 5), val_i8, dtype=torch.int8, device="cuda")

    out, _, _ = scaled_int8_quant(x, scale, azp, symmetric=False)
    torch.testing.assert_close(expected, out, atol=0, rtol=0)
