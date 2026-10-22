# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.longspark import _LongSparkRMSNorm


def _rms_norm_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    normalized = hidden_states.float()
    if residual is not None:
        normalized = normalized + residual.float()
        residual = normalized.to(hidden_states.dtype)
    variance = normalized.pow(2).mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    output = (normalized.to(weight.dtype) * weight).to(hidden_states.dtype)
    return output, residual


@pytest.mark.parametrize(
    ("batch_size", "rows", "hidden_size"),
    [(1, 3, 64), (5, 7, 2560)],
)
@pytest.mark.parametrize("add_residual", [False, True])
def test_longspark_rms_norm_matches_native_bf16(
    batch_size: int,
    rows: int,
    hidden_size: int,
    add_residual: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(53)
    eps = 1e-6
    norm = _LongSparkRMSNorm(hidden_size, eps).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.normal_(generator=generator)
    hidden_states = torch.randn(
        batch_size,
        rows,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    residual = (
        torch.randn_like(hidden_states, generator=generator) if add_residual else None
    )
    expected, expected_residual = _rms_norm_reference(
        hidden_states, norm.weight, eps, residual
    )

    if residual is None:
        actual = norm(hidden_states)
        assert isinstance(actual, torch.Tensor)
    else:
        actual, actual_residual = norm(hidden_states.clone(), residual.clone())
        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)

    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1e-3)
