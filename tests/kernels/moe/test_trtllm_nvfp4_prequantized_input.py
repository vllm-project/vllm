# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TRTLLM-Gen NVFP4 experts use an input quantized ahead of the MoE call
only for that very input, and quantize it themselves otherwise."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import trtllm_nvfp4_moe as m


@pytest.fixture
def quantize_calls(monkeypatch):
    calls = []

    def fake_quantize(hidden_states):
        calls.append(hidden_states)
        return ("quantized", hidden_states.shape)

    monkeypatch.setattr(m, "quantize_nvfp4_per_token_input", fake_quantize)
    return calls


def test_prequantized_input_used_for_the_same_tensor(quantize_calls):
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    pre = ("pre", "scales", "per_token")
    with m.prequantized_nvfp4_per_token_input(x, pre):
        assert m._quantize_per_token_input(x) is pre
        # A view of the same storage and shape is the same input.
        assert m._quantize_per_token_input(x.view(8, 64)) is pre
    assert not quantize_calls


def test_other_inputs_are_quantized(quantize_calls):
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    pre = ("pre", "scales", "per_token")
    other = torch.randn(8, 64, dtype=torch.bfloat16)
    with m.prequantized_nvfp4_per_token_input(x, pre):
        assert m._quantize_per_token_input(other) == ("quantized", other.shape)
        # Same storage, different shape: not the prequantized input.
        assert m._quantize_per_token_input(x[:4]) == ("quantized", (4, 64))
    # Outside the block the handoff is gone.
    assert m._quantize_per_token_input(x) == ("quantized", x.shape)
    assert len(quantize_calls) == 3


def test_handoff_cleared_on_error(quantize_calls):
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    with (
        pytest.raises(RuntimeError),
        m.prequantized_nvfp4_per_token_input(x, ("pre",)),
    ):
        raise RuntimeError
    assert m._prequantized_input is None
