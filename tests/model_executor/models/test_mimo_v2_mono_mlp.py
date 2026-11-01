# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.parametrize("intermediate", [2048, 16384])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform.is_rocm(),
    reason="requires a ROCm GPU",
)
def test_mimo_mono_mlp_matches_blockscale_reference(intermediate: int):
    device = "cuda"
    gate_up = torch.randn((1, 2 * intermediate), device=device, dtype=torch.bfloat16)
    down_weight = torch.randn((6144, intermediate), device=device).to(
        torch.float8_e4m3fn
    )
    down_scale = (
        torch.rand((48, intermediate // 128), device=device, dtype=torch.float32) + 0.01
    )

    from vllm.model_executor.kernels.triton.mimo_mono_mlp import (
        can_use_mimo_fp8_mono_mlp,
        mimo_fp8_mono_mlp,
    )

    assert can_use_mimo_fp8_mono_mlp(
        gate_up,
        down_weight,
        down_scale,
        tensor_parallel_size=8 if intermediate == 2048 else 1,
        is_capturing=False,
    )
    actual = mimo_fp8_mono_mlp(gate_up, down_weight, down_scale)
    activation = torch.nn.functional.silu(gate_up[:, :intermediate].float())
    activation *= gate_up[:, intermediate:].float()
    activation_scale = activation.view(1, intermediate // 128, 128).abs().amax(-1)
    activation_scale = activation_scale.clamp_min(1e-10) / 448.0
    activation_q = (
        (activation.view(1, intermediate // 128, 128) / activation_scale[..., None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
    )
    expected = torch.zeros((1, 6144), device=device, dtype=torch.float32)
    for n in range(0, 6144, 128):
        for k in range(0, intermediate, 128):
            expected[:, n : n + 128] += (
                activation_q[:, k // 128].float()
                @ down_weight[n : n + 128, k : k + 128].float().T
                * activation_scale[:, k // 128 : k // 128 + 1]
                * down_scale[n // 128, k // 128]
            )
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
