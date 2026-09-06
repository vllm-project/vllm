# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for NCP-OLMo mean-subtracting LayerNorm compatibility."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.envs as envs
from vllm.model_executor.models.ncp_olmo.hlm import (
    apply_ncp_layer_norm,
    apply_ncp_linear,
)
from vllm.model_executor.models.ncp_olmo.model import _chunk_mean


def test_batch_invariant_layer_norm_accepts_bf16_affine_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    layer_norm = nn.LayerNorm(16, eps=1e-5).to(dtype=torch.bfloat16)
    hidden_states = torch.randn(4, 3, 16, dtype=torch.bfloat16)

    output = apply_ncp_layer_norm(layer_norm, hidden_states)
    reference = F.layer_norm(
        hidden_states.float(),
        layer_norm.normalized_shape,
        layer_norm.weight.float(),
        layer_norm.bias.float(),
        layer_norm.eps,
    ).to(hidden_states.dtype)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


def test_batch_invariant_layer_norm_is_independent_of_row_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    layer_norm = nn.LayerNorm(32, eps=1e-5).to(dtype=torch.bfloat16)
    hidden_states = torch.randn(7, 32, dtype=torch.bfloat16)

    batched = apply_ncp_layer_norm(layer_norm, hidden_states)
    rowwise = torch.cat(
        [
            apply_ncp_layer_norm(layer_norm, row.unsqueeze(0))
            for row in hidden_states
        ],
        dim=0,
    )

    assert torch.equal(batched, rowwise)


def test_regular_layer_norm_path_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    layer_norm = nn.LayerNorm(8)
    hidden_states = torch.randn(2, 8)

    assert torch.equal(
        apply_ncp_layer_norm(layer_norm, hidden_states),
        layer_norm(hidden_states),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batch_invariant_native_linear_is_independent_of_row_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    linear = nn.Linear(32, 17, bias=False).to(device="cuda", dtype=torch.bfloat16)
    hidden_states = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)

    batched = apply_ncp_linear(linear, hidden_states)
    rowwise = torch.cat(
        [apply_ncp_linear(linear, row.unsqueeze(0)) for row in hidden_states],
        dim=0,
    )

    assert torch.equal(batched, rowwise)


def test_regular_native_linear_path_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    linear = nn.Linear(8, 5)
    hidden_states = torch.randn(2, 8)

    assert torch.equal(
        apply_ncp_linear(linear, hidden_states),
        linear(hidden_states),
    )


def test_chunk_mean_preserves_bf16_activation_dtype() -> None:
    hidden_states = torch.randn(12, 16, dtype=torch.bfloat16)

    pooled = _chunk_mean(hidden_states, 4)

    assert pooled.shape == (3, 16)
    assert pooled.dtype == torch.bfloat16
