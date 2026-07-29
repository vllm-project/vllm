# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from vllm.model_executor.layers.lfm25_fused_silu_fp8 import (
    fused_lfm25_silu_fp8_quant,
    _fused_lfm25_silu_math_for_test,
)

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="LFM2.5 SiLU+FP8 fusion requires CUDA-alike",
)


@pytest.mark.parametrize("dim", [4096, 8192])
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32])
def test_silu_fp8_fusion(dim: int, num_tokens: int):
    set_random_seed(0)
    device = DEVICE
    dtype = torch.bfloat16

    gate_up = torch.randn(num_tokens, dim * 2, device=device, dtype=dtype)

    # Reference: kernel math + PyTorch quantization
    activated, _ = _fused_lfm25_silu_math_for_test(gate_up.clone())
    a32 = activated.float()
    am = a32.abs().max(dim=1, keepdim=True).values
    scale = torch.maximum(
        am * (1.0 / 448.0),
        torch.tensor(1.0 / (448.0 * 512.0), device=device),
    )
    ref_fp8 = (a32 / scale).round().clamp(-448, 448).to(torch.float8_e4m3fn)

    # Fused kernel
    fused_fp8, fused_s = fused_lfm25_silu_fp8_quant(gate_up.clone())

    assert (scale - fused_s).abs().max().item() < 1e-4
    assert (ref_fp8.float() - fused_fp8.float()).abs().max().item() <= 32
