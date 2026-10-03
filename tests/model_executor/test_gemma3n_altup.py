# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.gemma3n import Gemma3nAltUp


class PredictionCoefs(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, modalities: torch.Tensor) -> torch.Tensor:
        return self.output


def make_altup(all_coefs: torch.Tensor) -> Gemma3nAltUp:
    altup = Gemma3nAltUp.__new__(Gemma3nAltUp)
    nn.Module.__init__(altup)
    altup.altup_num_inputs = 4
    altup.altup_active_idx = 0

    def compute_router_modalities(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, :4]

    altup._compute_router_modalities = compute_router_modalities
    altup.prediction_coefs = PredictionCoefs(all_coefs)
    return altup


@pytest.mark.parametrize(
    ("num_tokens", "expected_baddbmm_calls"),
    [(32, 1), (33, 0)],
)
def test_predict_dispatches_baddbmm_at_token_threshold(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    expected_baddbmm_calls: int,
) -> None:
    generator = torch.Generator().manual_seed(55062 + num_tokens)
    hidden_states = torch.randn(4, num_tokens, 32, generator=generator)
    all_coefs_t = torch.randn(num_tokens, 4, 4, generator=generator)
    all_coefs = all_coefs_t.permute(0, 2, 1).reshape(num_tokens, 16)
    altup = make_altup(all_coefs)
    original_hidden = hidden_states.clone()
    original_baddbmm = torch.baddbmm
    baddbmm_calls = 0

    def counted_baddbmm(*args, **kwargs):
        nonlocal baddbmm_calls
        baddbmm_calls += 1
        return original_baddbmm(*args, **kwargs)

    monkeypatch.setattr(torch, "baddbmm", counted_baddbmm)
    actual = altup.predict(hidden_states)

    hidden_t = hidden_states.permute(1, 2, 0)
    expected = torch.matmul(hidden_t, all_coefs_t) + hidden_t
    expected = expected.permute(2, 0, 1).contiguous()

    assert baddbmm_calls == expected_baddbmm_calls
    assert actual.shape == hidden_states.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden_states, original_hidden, rtol=0, atol=0)
