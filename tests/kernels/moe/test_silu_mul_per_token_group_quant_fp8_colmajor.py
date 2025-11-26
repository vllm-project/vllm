# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("T", [128, 256, 512])
@pytest.mark.parametrize("N", [128 * 2, 256 * 2, 768 * 2, 2048 * 2, 7168 * 2])
def test_silu_mul_fp8_quant_deep_gemm(T: int, N: int):
    current_platform.seed_everything(42)

    input = torch.rand((T, N), dtype=torch.bfloat16, device="cuda")

    # Test
    output, output_scales = silu_mul_per_token_group_quant_fp8_colmajor(input)

    # Reference
    ref_act_out = torch.empty((T, N // 2), dtype=torch.bfloat16, device="cuda")
    torch.ops._C.silu_and_mul(ref_act_out, input)
    ref_output, ref_output_scales = per_token_group_quant_fp8(
        ref_act_out, 128, column_major_scales=True
    )

    # massive atol to account for float8 rounding errors.
    torch.testing.assert_close(
        output.to(torch.float32), ref_output.to(torch.float32), atol=32, rtol=1e-3
    )
    torch.testing.assert_close(output_scales, ref_output_scales)
