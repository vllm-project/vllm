# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 Gemma RMSNorm vs the fp32 reference: flashinfer kernels on SM80+,
``GemmaRMSNorm.forward_native`` below (no nvcc JIT on e.g. T4)."""

import pytest
import torch

from vllm import ir
from vllm.models.qwen3_5.nvidia.model import Qwen3_5RMSNorm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")


@torch.inference_mode()
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen3_5_rms_norm_matches_reference(
    default_vllm_config, dtype: torch.dtype, add_residual: bool
) -> None:
    if dtype not in current_platform.supported_dtypes:
        pytest.skip(f"{dtype} is not supported on this GPU")
    set_random_seed(0)
    hidden_size = 1024
    layer = Qwen3_5RMSNorm(hidden_size, eps=1e-6).to(device="cuda", dtype=dtype)
    layer.weight.data.normal_(mean=0.0, std=0.1)
    x = torch.randn(32, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if add_residual else None

    summed = x.float()
    if residual is not None:
        summed = summed + residual.float()
    variance = summed.pow(2).mean(dim=-1, keepdim=True)
    ref = (
        summed
        * torch.rsqrt(variance + layer.variance_epsilon)
        * (1.0 + layer.weight.float())
    ).to(dtype)

    if residual is not None:
        ref_residual = summed.to(dtype)
        out, out_residual = layer(x, residual)
        torch.testing.assert_close(out_residual, ref_residual, atol=0, rtol=0)
    else:
        out = layer(x)
    torch.testing.assert_close(out, ref, **ir.ops.rms_norm.get_tolerance(dtype))
