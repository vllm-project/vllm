# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EAGLE-3 / DFlash auxiliary hidden states on GLM-5.3 (Glm5Next).

The mHC layers defer the hyper-connection post-mix into the next layer's
fused pre-op, so the residual stream entering layer ``i`` is not materialized;
the aux capture applies the previous layer's post-mix and contracts the
streams. These tests use the torch-native mHC ops with the real tensor
shapes: x ``(tokens, hidden)``, residual ``(tokens, n, hidden)``,
post ``(tokens, n, 1)``, comb ``(tokens, n, n)``.
"""

from types import SimpleNamespace

import torch

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
from vllm.model_executor.layers.mhc import hc_contract
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


def test_both_model_classes_advertise_eagle3():
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)


class _MhcLayer:
    """A decoder layer stand-in whose post-mix is the torch-native mHC op."""

    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    def hc_post(self, x, residual, post, comb):
        self.calls += 1
        return mhc_post_torch(x, residual, post, comb)


def _fake_model(layers, sequence_parallel: bool = False):
    return SimpleNamespace(layers=layers, is_sequence_parallel=sequence_parallel)


def test_aux_state_before_layer_zero_is_the_embedding():
    model = _fake_model([_MhcLayer(n=4)])
    hidden = torch.randn(5, 8)
    out = Glm5NextModel._aux_hidden_state(model, 0, hidden, None, None, None, 5)
    assert out is hidden


def test_aux_state_inside_mhc_stack_is_previous_post_mix_contracted():
    torch.manual_seed(0)
    tokens, n, hidden = 5, 4, 8
    layers = [_MhcLayer(n) for _ in range(3)]
    model = _fake_model(layers)
    x = torch.randn(tokens, hidden)
    residual = torch.randn(tokens, n, hidden)
    post = torch.randn(tokens, n, 1)
    comb = torch.randn(tokens, n, n)

    out = Glm5NextModel._aux_hidden_state(model, 2, x, residual, post, comb, tokens)

    # Only the previous layer's post-mix is applied, then the streams are
    # averaged: what the last layer does before the final norm.
    expected = hc_contract(mhc_post_torch(x, residual, post, comb), n)
    assert out.shape == (tokens, hidden)
    torch.testing.assert_close(out, expected)
    assert [layer.calls for layer in layers] == [0, 1, 0]


def test_aux_state_non_mhc_layer_is_the_summed_stream():
    model = _fake_model([_MhcLayer(n=1), _MhcLayer(n=1)])
    hidden = torch.randn(5, 8)
    # Non-mHC layers return (summed hidden_states, residual, None, None):
    # with post None the stream is hidden_states itself.
    out = Glm5NextModel._aux_hidden_state(model, 1, hidden, hidden, None, None, 5)
    assert out is hidden
