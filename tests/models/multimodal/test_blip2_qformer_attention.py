# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F
from transformers import Blip2QFormerConfig

from vllm.model_executor.models.blip2 import Blip2QFormerMultiHeadAttention

pytestmark = pytest.mark.cpu_test


def _make_module(*, cross_attention: bool) -> Blip2QFormerMultiHeadAttention:
    config = Blip2QFormerConfig(
        hidden_size=32,
        encoder_hidden_size=48,
        num_attention_heads=4,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )
    return Blip2QFormerMultiHeadAttention(
        config,
        quant_config=None,
        cache_config=None,
        is_cross_attention=cross_attention,
    ).eval()


def _reference_forward(
    module: Blip2QFormerMultiHeadAttention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
) -> torch.Tensor:
    source = hidden_states if encoder_hidden_states is None else encoder_hidden_states
    query = module.transpose_for_scores(module.query(hidden_states))
    key = module.transpose_for_scores(module.key(source))
    value = module.transpose_for_scores(module.value(source))
    scores = torch.matmul(query, key.transpose(-1, -2))
    probabilities = torch.softmax(scores * module.scaling, dim=-1)
    context = torch.matmul(probabilities, value)
    context = context.permute(0, 2, 1, 3).contiguous()
    return context.view(*context.size()[:-2], module.all_head_size)


@pytest.mark.parametrize("cross_attention", [False, True])
def test_qformer_attention_uses_sdpa_and_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    cross_attention: bool,
) -> None:
    torch.manual_seed(7)
    module = _make_module(cross_attention=cross_attention)
    hidden_states = torch.randn(4, 17, 32)
    encoder_hidden_states = torch.randn(4, 37, 48) if cross_attention else None
    expected = _reference_forward(module, hidden_states, encoder_hidden_states)

    original_sdpa = F.scaled_dot_product_attention
    calls = 0

    def recording_sdpa(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    actual = module(hidden_states, encoder_hidden_states)

    assert calls == 1
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.shape == hidden_states.shape
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()


def test_qformer_attention_preserves_training_dropout_contract() -> None:
    torch.manual_seed(11)
    module = _make_module(cross_attention=True).train()
    module.dropout.p = 0.25
    hidden_states = torch.randn(2, 13, 32)
    encoder_hidden_states = torch.randn(2, 29, 48)

    first = module(hidden_states, encoder_hidden_states)
    second = module(hidden_states, encoder_hidden_states)

    assert first.shape == second.shape == hidden_states.shape
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert not torch.equal(first, second)
