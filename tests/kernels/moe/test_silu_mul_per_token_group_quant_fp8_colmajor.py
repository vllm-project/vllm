# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _per_token_group_quant_fp8_colmajor,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

FLOAT8_DTYPE = torch.float8_e4m3fn
GROUP_SIZE = 128


def reference_quant(x: torch.Tensor, use_ue8m0: bool):
    """
    Reference triton quant kernel from,
    vllm.model_executor.layers.quantization.utils.fp8_utils
    """

    x_q = torch.empty_like(x, device=x.device, dtype=FLOAT8_DTYPE)

    # Allocate the scale tensor in column-major format.
    shape = (x.shape[-1] // GROUP_SIZE,) + x.shape[:-1]
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32).permute(-1, -2)

    M = x.numel() // GROUP_SIZE
    N = GROUP_SIZE
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    finfo = torch.finfo(FLOAT8_DTYPE)
    fp8_min = finfo.min
    fp8_max = finfo.max

    _per_token_group_quant_fp8_colmajor[(M,)](
        x,
        x_q,
        x_s,
        GROUP_SIZE,
        x.shape[1],
        x.stride(0),
        x_s.stride(1),
        eps=1e-10,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=use_ue8m0,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x_q, x_s


def reference(x: torch.Tensor, use_ue8m0: bool) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = x.size()
    ref_act_out = torch.empty((T, N // 2), dtype=torch.bfloat16, device="cuda")
    torch.ops._C.silu_and_mul(ref_act_out, x)
    return reference_quant(ref_act_out, use_ue8m0)


@pytest.mark.parametrize("T", [128, 256, 512])
@pytest.mark.parametrize("N", [128 * 2, 256 * 2, 768 * 2, 2048 * 2, 7168 * 2])
def test_silu_mul_fp8_quant_deep_gemm(T: int, N: int):
    current_platform.seed_everything(42)

    input = torch.rand((T, N), dtype=torch.bfloat16, device="cuda")

    use_ue8m0 = is_deep_gemm_e8m0_used()

    # Test
    output, output_scales = silu_mul_per_token_group_quant_fp8_colmajor(
        input, use_ue8m0=use_ue8m0
    )

    # Reference
    ref_output, ref_output_scales = reference(input, use_ue8m0)

    torch.testing.assert_close(output.to(torch.float32), ref_output.to(torch.float32))
    torch.testing.assert_close(output_scales, ref_output_scales)
