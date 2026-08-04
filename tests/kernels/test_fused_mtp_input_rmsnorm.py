# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
    fused_mtp_input_rmsnorm,
)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def test_fused_mtp_input_rmsnorm() -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    num_tokens = 4
    hidden_size = 128
    hc_mult = 2
    eps = 1e-6

    inputs_embeds = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    previous_hidden_states = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=dtype, device=device
    )
    enorm_weight = torch.randn(hidden_size, dtype=dtype, device=device)
    hnorm_weight = torch.randn(hidden_size, dtype=dtype, device=device)

    enorm_output, hnorm_output = fused_mtp_input_rmsnorm(
        inputs_embeds,
        positions,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        eps,
        hc_mult,
    )

    masked_inputs = inputs_embeds.clone()
    masked_inputs[positions == 0] = 0
    expected_enorm = _rms_norm(masked_inputs, enorm_weight, eps)
    expected_hnorm = _rms_norm(previous_hidden_states, hnorm_weight, eps)

    torch.testing.assert_close(enorm_output, expected_enorm, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(hnorm_output, expected_hnorm, rtol=1e-3, atol=1e-3)
