# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm, has_flashinfer


@pytest.mark.skipif(
    not current_platform.is_cuda() or not has_flashinfer(),
    reason="FlashInfer FP8 scaled MM requires CUDA and FlashInfer.",
)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (1, 512, 3584),
        (8, 512, 3584),
        (1, 3584, 512),
        (8, 3584, 512),
    ],
)
def test_flashinfer_scaled_fp8_mm_matches_dequant_reference(m, n, k):
    torch.manual_seed(0)
    a_bf16 = (torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.02)
    b_bf16 = (torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02)

    scale_a = (a_bf16.abs().max().float().clamp_min(1e-8) / 448.0).view(1)
    scale_b = (b_bf16.abs().max().float().clamp_min(1e-8) / 448.0).view(1)
    a_fp8 = (a_bf16 / scale_a).to(torch.float8_e4m3fn).contiguous()
    b_fp8 = (b_bf16 / scale_b).to(torch.float8_e4m3fn).contiguous()

    expected = (a_fp8.float() * scale_a) @ (b_fp8.float() * scale_b)
    actual = flashinfer_scaled_fp8_mm(
        a_fp8,
        b_fp8,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=2e-2,
        atol=2e-3,
    )
