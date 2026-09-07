# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE-3 / DFlash auxiliary hidden states on GLM-5.3 (Glm5Next)."""

from types import SimpleNamespace

import torch

import vllm.models.glm5next.nvidia.model as glm5next_model
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


def test_both_model_classes_advertise_eagle3():
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)


class _FakeMhcLayer:
    """Stands in for an mHC decoder layer: records the deferred post-mix call."""

    def __init__(self, n: int):
        self.n = n
        self.calls: list[tuple] = []

    def hc_post(self, x, residual, post, comb):
        self.calls.append((x, residual, post, comb))
        # Pretend the post-mix returns the widened residual stream.
        return residual


def _fake_model(layers, sequence_parallel: bool = False):
    return SimpleNamespace(layers=layers, is_sequence_parallel=sequence_parallel)


def test_aux_state_before_layer_zero_is_the_embedding():
    model = _fake_model([_FakeMhcLayer(n=4)])
    hidden = torch.randn(5, 8)
    out = Glm5NextModel._aux_hidden_state(model, 0, hidden, None, None, None, 5)
    assert out is hidden


def test_aux_state_inside_mhc_stack_applies_previous_post_mix_and_contracts(
    monkeypatch,
):
    n = 4
    layers = [_FakeMhcLayer(n) for _ in range(3)]
    model = _fake_model(layers)
    x = torch.randn(5, 8)
    residual = torch.randn(5, n * 8)
    post, comb = torch.randn(5, n), torch.randn(5, n)
    contracted = torch.randn(5, 8)
    seen = {}

    def fake_contract(stream, mult):
        seen["stream"], seen["mult"] = stream, mult
        return contracted

    monkeypatch.setattr(glm5next_model, "hc_contract", fake_contract)
    out = Glm5NextModel._aux_hidden_state(model, 2, x, residual, post, comb, 5)
    # The stream entering layer 2 is layer 1's deferred post-mix, contracted.
    assert layers[1].calls == [(x, residual, post, comb)]
    assert layers[0].calls == [] and layers[2].calls == []
    assert seen["stream"] is residual and seen["mult"] == n
    assert out is contracted


def test_aux_state_non_mhc_layer_is_the_summed_stream():
    model = _fake_model([_FakeMhcLayer(n=1), _FakeMhcLayer(n=1)])
    hidden = torch.randn(5, 8)
    # Non-mHC layers return (summed hidden_states, residual, None, None):
    # with post None the stream is hidden_states itself.
    out = Glm5NextModel._aux_hidden_state(model, 1, hidden, hidden, None, None, 5)
    assert out is hidden
