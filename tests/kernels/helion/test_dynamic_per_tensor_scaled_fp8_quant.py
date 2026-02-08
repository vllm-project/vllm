# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the dynamic_per_token_scaled_fp8_quant helion kernel

Run `pytest tests/kernels/helion/test_dynamic_per_tensor_scaled_fp8_quant.py`.
"""

import pytest
import torch

from tests.kernels.quant_utils import (
    FP8_DTYPE,
    ref_dynamic_per_tensor_fp8_quant,
)
from vllm.kernels.helion.dynamic_per_tensor_scaled_fp8_quant import (
    dynamic_per_tensor_scaled_fp8_quant,
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
SEEDS = [0]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
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

    ops_out = torch.empty(x.shape, device="cuda", dtype=FP8_DTYPE)
    ops_scale = torch.empty(1, device=x.device, dtype=torch.float32)

    dynamic_per_tensor_scaled_fp8_quant(ops_out, x, ops_scale)

    torch.testing.assert_close(ref_scale, ops_scale)
    # allow 1 ULP difference
    assert (
        ref_out.view(torch.uint8).to(torch.int16)
        - ops_out.view(torch.uint8).to(torch.int16)
    ).abs().max() <= 1
