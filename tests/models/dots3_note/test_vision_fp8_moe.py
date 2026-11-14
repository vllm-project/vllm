# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots3 vision block-FP8 MoE weight cast."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

DEVICE = "xpu"


def _deepgemm_per_block_cast_to_fp8(x: torch.Tensor, gran_k: int):
    """DeepGEMM ``deep_gemm.utils.math.per_block_cast_to_fp8(use_ue8m0=False)``."""
    m, n = x.shape
    pm = (m + gran_k - 1) // gran_k * gran_k
    pn = (n + gran_k - 1) // gran_k * gran_k
    x_padded = torch.zeros((pm, pn), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, pn // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        sf.view(x_view.size(0), x_view.size(2)),
    )


@pytest.mark.parametrize("shape", [(2112, 1536), (1536, 2112), (100, 300)])
@pytest.mark.parametrize("block_size", [64, 128])
def test_per_block_cast_matches_deepgemm(shape, block_size):
    from vllm.models.dots3_note.nvidia.vision import _per_block_cast_to_fp8_padded

    torch.manual_seed(0)
    weight = torch.randn(shape, dtype=torch.bfloat16, device=DEVICE)
    weight[0, 0] = 0.0
    weight[-1] *= 1e-6

    q, s = _per_block_cast_to_fp8_padded(weight, block_size)

    padded = torch.zeros(
        (
            (shape[0] + block_size - 1) // block_size * block_size,
            (shape[1] + block_size - 1) // block_size * block_size,
        ),
        dtype=weight.dtype,
        device=DEVICE,
    )
    padded[: shape[0], : shape[1]] = weight
    ref_q, ref_s = _deepgemm_per_block_cast_to_fp8(padded, block_size)

    assert q.dtype == torch.float8_e4m3fn
    assert q.shape == ref_q.shape and s.shape == ref_s.shape
    torch.testing.assert_close(s, ref_s, atol=0, rtol=0)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
