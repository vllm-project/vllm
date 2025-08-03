# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    silu_mul_fp8_quant_deep_gemm)
from vllm.platforms import current_platform

# (E, T, H, group_size, seed)
CASES = [
    (1, 1, 128, 64, 0),
    (1, 4, 128, 128, 0),
    (2, 4, 256, 128, 0),
    (32, 64, 256, 128, 0),
    (17, 31, 768, 128, 0),
]


@pytest.mark.parametrize("E,T,H,group_size,seed", CASES)
@torch.inference_mode()
def test_silu_mul_fp8_quant_deep_gemm(E, T, H, group_size, seed):
    current_platform.seed_everything(seed)

    # Input tensor of shape (E, T, 2*H)
    y = torch.randn((E, T, 2 * H), dtype=torch.float32, device="cuda")
    tokens_per_expert = torch.randint(
        low=0,
        high=T,
        size=(E, ),
        dtype=torch.int32,
        device="cuda",
    )

    # Run the Triton kernel
    y_q, y_s = silu_mul_fp8_quant_deep_gemm(y,
                                            tokens_per_expert,
                                            group_size=group_size,
                                            eps=1e-10)

    # Reference implementation
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max
    fp8_min = fp8_info.min
    eps = 1e-10

    # Compute silu activation and elementwise multiplication
    y1 = y[..., :H]
    y2 = y[..., H:]
    silu_x = y1 * torch.sigmoid(y1)
    merged = silu_x * y2

    # Compute reference scales and quantized output, skipping padded tokens
    for e in range(E):
        nt = tokens_per_expert[e].item()
        ref_s = torch.empty((T, H // group_size),
                            dtype=torch.float32,
                            device="cuda")
        ref_q = torch.empty((T, H), dtype=torch.float8_e4m3fn, device="cuda")
        for t in range(nt):
            data = merged[e, t]
            data_grp = data.view(H // group_size, group_size)
            amax = data_grp.abs().amax(dim=1).clamp(min=eps)
            scale = amax / fp8_max

            scaled = data / scale.repeat_interleave(group_size)
            clamped = scaled.clamp(fp8_min, fp8_max)
            q = clamped.to(torch.float8_e4m3fn)

            ref_s[t] = scale
            ref_q[t] = q

        y_se = y_s[e]
        y_qe = y_q[e]

        torch.testing.assert_close(y_se[:nt], ref_s[:nt])
        torch.testing.assert_close(
            y_qe[:nt].to(torch.float32),
            ref_q[:nt].to(torch.float32),
            atol=2,
            rtol=2e-1,
        )
