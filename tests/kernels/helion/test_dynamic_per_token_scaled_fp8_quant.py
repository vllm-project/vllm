# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic_per_token_scaled_fp8_quant helion kernel

Run `pytest tests/kernels/helion/test_dynamic_per_token_scaled_fp8_quant.py`.
"""

import pytest
import torch

from tests.kernels.quant_utils import (
    FP8_DTYPE,
    ref_dynamic_per_token_quant,
)
from vllm.kernels.helion.dynamic_per_token_scaled_fp8_quant import (
    dynamic_per_token_scaled_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

DTYPES = [torch.bfloat16, torch.float]
HIDDEN_SIZES = [17, 1024, 1025, 1026, 5137, 8193]
NUM_TOKENS = [1, 7, 4096]
SCALE_UBS = [True, False]
SEEDS = [0]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_ub", SCALE_UBS)
@pytest.mark.parametrize("seed", SEEDS)
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

    ops_out = torch.empty(x.shape, device="cuda", dtype=FP8_DTYPE)
    ops_scales = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
    dynamic_per_token_scaled_fp8_quant(ops_out, x, ops_scales, scale_ub)

    torch.testing.assert_close(ref_scales, ops_scales)
    # allow 1 ULP difference
    assert (
        ref_out.view(torch.uint8).to(torch.int16)
        - ops_out.view(torch.uint8).to(torch.int16)
    ).abs().max() <= 1
