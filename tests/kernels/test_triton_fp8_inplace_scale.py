# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.model_executor.layers.quantization.kernels.fp8_inplace_scale import (
    triton_fp8_inplace_scale)
from vllm.platforms import current_platform


def torch_fp8_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    print(f"x.dtype={x.dtype}, x.shape={x.shape}")
    finfo = torch.finfo(current_platform.fp8_dtype())
    x = (x.to(torch.float16) * (1.0 / scale)).clamp(finfo.min, finfo.max)
    x = x.to(current_platform.fp8_dtype())
    return x


@pytest.mark.parametrize("M", [1, 32, 65, 128, 1092])
@pytest.mark.parametrize("N", [1, 32, 63, 122, 1139])
def test_triton_fp8_inplace_scale_kernel(M, N):
    torch.manual_seed(0)
    device = "cuda"
    x = torch.rand((M, N), dtype=torch.float16,
                   device=device).to(current_platform.fp8_dtype())
    scale = torch.rand(1).item()
    golden = torch_fp8_scale(x, scale)
    check = triton_fp8_inplace_scale(x, scale)
    torch.testing.assert_close(golden, check)
