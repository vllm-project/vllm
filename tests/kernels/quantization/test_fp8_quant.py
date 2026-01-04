# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import (
    FP8_DTYPE,
    ref_dynamic_per_tensor_fp8_quant,
    ref_dynamic_per_token_quant,
)
from tests.kernels.utils import opcheck
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float]
HIDDEN_SIZES = [17, 1024, 1025, 1026, 5137, 8193]
NUM_TOKENS = [1, 7, 4096]
SCALE_UBS = [True, False]
SEEDS = [0]


def opcheck_fp8_quant(
    output, input, scale=None, scale_ub=None, use_per_token_if_dynamic=False
):
    if scale is not None:
        opcheck(torch.ops._C.static_scaled_fp8_quant, (output, input, scale))
    elif use_per_token_if_dynamic:
        scale = torch.empty(
            (input.shape[0], 1), device=input.device, dtype=torch.float32
        )
        opcheck(
            torch.ops._C.dynamic_per_token_scaled_fp8_quant,
            (output, input, scale, scale_ub),
        )
    else:
        scale = torch.empty(
            (input.numel() // input.shape[-1], 1),
            device=input.device,
            dtype=torch.float32,
        )
        opcheck(torch.ops._C.dynamic_scaled_fp8_quant, (output, input, scale))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_ub", SCALE_UBS)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_token_fp8_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, scale_ub: bool, seed: int
) -> None:
    set_random_seed(seed)

    x = (
        torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda") + 1e-6
    )  # avoid nans

    scale_ub = (
        torch.mean(x).to(dtype=torch.float32, device="cuda") if scale_ub else None
    )
    ref_out, ref_scales = ref_dynamic_per_token_quant(x, FP8_DTYPE, scale_ub)
    ops_out, ops_scales = ops.scaled_fp8_quant(
        x, scale_ub=scale_ub, use_per_token_if_dynamic=True
    )

    torch.testing.assert_close(ref_scales, ops_scales)
    torch.testing.assert_close(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )

    opcheck_fp8_quant(ops_out, x, None, scale_ub, use_per_token_if_dynamic=True)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_dynamic_per_tensor_fp8_quant(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, seed: int
) -> None:
    set_random_seed(seed)

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")

    ref_out, ref_scale = ref_dynamic_per_tensor_fp8_quant(x)
    ops_out, ops_scale = ops.scaled_fp8_quant(x)

    torch.testing.assert_close(ref_scale, ops_scale)
    torch.testing.assert_close(
        ref_out.to(dtype=torch.float32), ops_out.to(dtype=torch.float32)
    )

    opcheck_fp8_quant(ops_out, x)


# Regression test for a case with large activations where an int32 index cannot
# represent the number of elements.
@torch.inference_mode()
@pytest.mark.parametrize("seed", SEEDS)
def test_fp8_quant_large(seed: int) -> None:
    set_random_seed(seed)

    num_tokens = 1024000  # Mistral-Nemo's max_position_embeddings
    hidden_size = 1152  # Smallest hidden_size to reproduce the error
    dtype = torch.bfloat16

    x = torch.rand(num_tokens, hidden_size, dtype=dtype, device="cuda")
    ref_out, scale = ref_dynamic_per_tensor_fp8_quant(x)
    ops_out, _ = ops.scaled_fp8_quant(x, scale)

    # Minimize memory footprint in this test by freeing x and upconverting
    # the outputs in place. (torch.allclose does not support fp8)
    del x
    ref_out = ref_out.to(dtype=dtype)
    ops_out = ops_out.to(dtype=dtype)

    torch.testing.assert_close(ref_out, ops_out)
