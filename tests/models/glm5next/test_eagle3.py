# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.kernels.mhc import mhc_post_torch
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.common import model as glm

pytestmark = pytest.mark.cpu_test


def make_model(monkeypatch, layers):
    model = object.__new__(glm.Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = len(layers)
    model.layers = layers
    model._active_layers = layers
    model.is_sequence_parallel = False
    model.norm = torch.nn.Identity()
    monkeypatch.setattr(
        glm,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return model


def forward(model, hidden):
    return model(
        input_ids=None,
        positions=torch.arange(len(hidden)),
        intermediate_tensors=None,
        inputs_embeds=hidden,
    )


def test_capture_materializes_mhc_without_mutating_target(monkeypatch):
    hidden = torch.tensor([[3.0, 4.0]])
    residual = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    post = torch.tensor([[[0.5], [1.5]]])
    comb = torch.tensor([[[0.75, 0.25], [0.25, 0.75]]])
    state = (hidden, residual, post, comb)
    originals = [t.clone() for t in state]
    first = Mock(return_value=state)
    first.n = 2
    first.hc_post = Mock(side_effect=mhc_post_torch)

    def finish(positions, h, r, p, c):
        assert all(a is b for a, b in zip((h, r, p, c), state))
        return h, None, None, None

    model = make_model(monkeypatch, [first, Mock(side_effect=finish)])
    baseline = forward(model, hidden)
    first.hc_post.assert_not_called()
    model._set_aux_hidden_state_layers((1, 2))
    output, aux = forward(model, hidden)
    expected = (
        torch.einsum("sij,sih->sjh", comb.double(), residual.double())
        + post.double() * hidden.double().unsqueeze(1)
    ).mean(1)
    torch.testing.assert_close(aux[0].double(), expected)
    torch.testing.assert_close(aux[1], hidden)
    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    for value, original in zip(state, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_non_mhc_capture_uses_completed_decoder_output(monkeypatch):
    """The actual non-MTP decoder already adds its residual before returning."""
    layer = object.__new__(glm.Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.mhc = False
    layer.is_mtp_layer = False
    layer.input_layernorm = torch.nn.Identity()
    layer.self_attn = Mock(side_effect=lambda hidden_states, positions: hidden_states)
    layer.post_attention_layernorm = Mock(side_effect=lambda h, residual: (h, residual))
    layer.mlp = torch.nn.Identity()
    model = make_model(monkeypatch, [layer])
    model._set_aux_hidden_state_layers((0, 1))
    hidden = torch.tensor([[1.0, 2.0]])
    output, aux = forward(model, hidden)
    torch.testing.assert_close(output, 2 * hidden)
    torch.testing.assert_close(aux[0], hidden)
    torch.testing.assert_close(aux[1], output)


@pytest.mark.parametrize("capture_layer", [1, 6])
def test_sequence_parallel_capture_gathers_and_trims_padding(
    monkeypatch, capture_layer
):
    """Both early layers and the checkpoint's first tap need full token rows."""
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    shard = torch.cat([hidden, torch.zeros(1, 4)]).chunk(2)[0]
    layer = Mock(side_effect=lambda pos, h, r, p, c: (h, None, None, None))
    model = make_model(monkeypatch, [layer] * capture_layer)
    model.is_sequence_parallel = True
    model._set_aux_hidden_state_layers((0, capture_layer))
    monkeypatch.setattr(glm, "sp_shard", lambda _: shard)
    gather = Mock(return_value=torch.cat([hidden, torch.zeros(1, 4)]))
    monkeypatch.setattr(glm, "sp_all_gather", gather)
    output, aux = forward(model, hidden)
    assert gather.call_count == 2
    torch.testing.assert_close(output, hidden)
    for captured in aux:
        torch.testing.assert_close(captured, hidden)


def test_wrappers_configure_dflash_checkpoint_layers(monkeypatch):
    layers = (6, 15, 25, 34, 43)
    for wrapper in (glm.Glm5NextForCausalLM, glm.Glm5NextForConditionalGeneration):
        assert supports_eagle3(wrapper)
        target = object.__new__(wrapper)
        torch.nn.Module.__init__(target)
        model = make_model(monkeypatch, [None] * 45)
        if wrapper is glm.Glm5NextForCausalLM:
            target.model = model
        else:
            target.language_model = SimpleNamespace(
                embed_input_ids=lambda _: None,
                forward=lambda input_ids, positions: None,
                model=model,
            )
            target._language_model_names = ["language_model"]
        target.set_aux_hidden_state_layers(layers)
        assert model.aux_hidden_state_layers == layers
