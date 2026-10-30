# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-M3 input (fused add) Gemma RMSNorm with an MXFP8 output writes, bit
for bit, the residual and what quantizing today's BF16 output for qkv_proj gives:
E4M3 data plus 128x4-swizzled UE8M0 scales, including the zeroed padding rows."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="SM100-family only",
)

HIDDEN = 6144


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    return mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _rows(n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """Per-32-block magnitudes over 2^-20..2^20 with sprinkled NaN/Inf/0/denormals."""
    x = torch.randn(n, k, generator=gen, device="cuda")
    e = torch.randint(-20, 21, (n, k // 32, 1), generator=gen, device="cuda")
    x = (x.view(n, -1, 32) * torch.pow(2.0, e.float())).view(n, k)
    special = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, 448.0, 464.0],
        device="cuda",
    )
    mask = torch.rand(n, k, generator=gen, device="cuda") < 1e-3
    idx = torch.randint(0, len(special), (n, k), generator=gen, device="cuda")
    return torch.where(mask, special[idx], x).to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 129, 1024])
@pytest.mark.parametrize("fused_add", [True, False])
def test_input_norm_mxfp8_matches_norm_then_quantize(num_tokens, fused_add):
    from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

    from vllm.models.minimax_m3.nvidia.ops.norm_mxfp8 import (
        gemma_fused_add_rmsnorm_mxfp8,
        gemma_rmsnorm_mxfp8,
    )

    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = _rows(num_tokens, HIDDEN, gen)
    residual = _rows(num_tokens, HIDDEN, gen)
    weight = (torch.randn(HIDDEN, generator=gen, device="cuda") * 0.2).to(x.dtype)
    eps = 1e-6

    x_ref, res_ref = x.clone(), residual.clone()
    if fused_add:
        gemma_fused_add_rmsnorm(x_ref, res_ref, weight, eps)
        y_ref = x_ref
    else:
        y_ref = gemma_rmsnorm(x_ref, weight, eps)
    q_ref, s_ref = _quantize(y_ref)

    x_new, res_new = x.clone(), residual.clone()
    if fused_add:
        q, s = gemma_fused_add_rmsnorm_mxfp8(x_new, res_new, weight, eps)
    else:
        q, s = gemma_rmsnorm_mxfp8(x_new, weight, eps)

    assert torch.equal(_bits(q), _bits(q_ref))
    assert torch.equal(_bits(s), _bits(s_ref))
    assert torch.equal(_bits(res_new), _bits(res_ref))
    assert torch.equal(_bits(x_new), _bits(x))
