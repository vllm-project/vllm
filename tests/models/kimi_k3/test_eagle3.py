# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearModel,
    _should_create_kimi_embedding,
)
from vllm.sequence import IntermediateTensors


def _make_kimi_linear_model() -> KimiLinearModel:
    model = object.__new__(KimiLinearModel)
    object.__setattr__(model, "aux_hidden_state_layers", (2,))
    object.__setattr__(model, "use_sequence_parallel", False)
    return model


def test_kimi_k3_advertises_eagle3_support():
    assert supports_eagle3(KimiK3ForConditionalGeneration)


def test_kimi_k3_replicates_embedding_on_last_pp_rank_for_dspark(monkeypatch):
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        speculative_config=SimpleNamespace(method="dspark"),
    )
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )

    assert _should_create_kimi_embedding(vllm_config)

    vllm_config.speculative_config = None
    assert not _should_create_kimi_embedding(vllm_config)


def test_kimi_k3_uses_shared_eagle3_layer_configuration():
    target = object.__new__(KimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(target)
    model = _make_kimi_linear_model()
    object.__setattr__(model, "layers", [None] * 93)
    language_model = SimpleNamespace(
        embed_input_ids=lambda _: None,
        model=model,
    )
    object.__setattr__(target, "language_model", language_model)
    object.__setattr__(target, "_language_model_names", ["language_model"])

    target.set_aux_hidden_state_layers((2, 46, 90))

    assert model.aux_hidden_state_layers == (2, 46, 90)
    assert target.get_eagle3_default_aux_hidden_state_layers() == (
        2,
        46,
        90,
    )


def test_kimi_linear_forward_extracts_standard_aux_hidden_states(monkeypatch):
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    layer_residual = torch.tensor([[5.0, 6.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, None, layer_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1))
    object.__setattr__(model, "use_attn_res", False)
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    expected_layer_output = layer_hidden_states + layer_residual
    torch.testing.assert_close(output, expected_layer_output)
    torch.testing.assert_close(aux_hidden_states[0], initial_hidden_states)
    torch.testing.assert_close(aux_hidden_states[1], expected_layer_output)


def test_kimi_linear_forward_extracts_attn_res_aux_hidden_states(monkeypatch):
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    prefix_sum = torch.tensor([[5.0, 6.0]])
    block_residual = torch.tensor([[[7.0, 8.0]]])
    final_hidden_states = torch.tensor([[9.0, 10.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, prefix_sum, block_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1))
    object.__setattr__(model, "use_attn_res", True)
    object.__setattr__(model, "num_attn_res_blocks", 1)
    object.__setattr__(
        model,
        "output_attn_res_norm",
        SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-5),
    )
    object.__setattr__(
        model,
        "output_attn_res_proj",
        SimpleNamespace(weight=torch.ones(1, 2)),
    )
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    final_attn_res = Mock(return_value=final_hidden_states)
    monkeypatch.setattr(kimi_model, "attn_res", final_attn_res)

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    torch.testing.assert_close(output, final_hidden_states)
    torch.testing.assert_close(aux_hidden_states[0], initial_hidden_states)
    torch.testing.assert_close(aux_hidden_states[1], prefix_sum + layer_hidden_states)
    assert final_attn_res.call_args.args[2] is block_residual


def test_kimi_linear_pp_intermediate_buffers_include_prior_aux_states():
    model = _make_kimi_linear_model()
    object.__setattr__(model, "config", SimpleNamespace(hidden_size=2))
    object.__setattr__(model, "start_layer", 24)
    object.__setattr__(model, "aux_hidden_state_layers", (3, 24, 48))
    object.__setattr__(model, "use_attn_res", False)

    intermediate_tensors = model.make_empty_intermediate_tensors(
        batch_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert set(intermediate_tensors.tensors) == {
        "hidden_states",
        "residual",
        "aux_hidden_states.3",
        "aux_hidden_states.24",
    }
    assert intermediate_tensors["aux_hidden_states.3"].shape == (4, 2)
    assert intermediate_tensors["aux_hidden_states.24"].shape == (4, 2)


def test_kimi_linear_pp_packs_local_aux_states_for_next_rank(monkeypatch):
    model = _make_kimi_linear_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    layer_residual = torch.tensor([[5.0, 6.0]])

    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", 1)
    object.__setattr__(
        model,
        "layers",
        [Mock(return_value=(layer_hidden_states, None, layer_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1, 2))
    object.__setattr__(model, "use_attn_res", False)
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    assert isinstance(output, IntermediateTensors)
    assert set(output.tensors) == {
        "hidden_states",
        "residual",
        "aux_hidden_states.0",
        "aux_hidden_states.1",
    }
    torch.testing.assert_close(output["aux_hidden_states.0"], initial_hidden_states)
    torch.testing.assert_close(
        output["aux_hidden_states.1"], layer_hidden_states + layer_residual
    )


def test_kimi_linear_pp_forwards_aux_states_to_last_rank(monkeypatch):
    model = _make_kimi_linear_model()
    incoming_hidden_states = torch.tensor([[1.0, 2.0]])
    incoming_residual = torch.tensor([[3.0, 4.0]])
    prior_aux_0 = torch.tensor([[5.0, 6.0]])
    prior_aux_1 = torch.tensor([[7.0, 8.0]])
    layer_hidden_states = torch.tensor([[9.0, 10.0]])
    layer_residual = torch.tensor([[11.0, 12.0]])

    object.__setattr__(model, "start_layer", 1)
    object.__setattr__(model, "end_layer", 2)
    object.__setattr__(
        model,
        "layers",
        [None, Mock(return_value=(layer_hidden_states, None, layer_residual))],
    )
    object.__setattr__(model, "aux_hidden_state_layers", (0, 1, 2))
    object.__setattr__(model, "use_attn_res", False)
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": incoming_hidden_states,
            "residual": incoming_residual,
            "aux_hidden_states.0": prior_aux_0,
            "aux_hidden_states.1": prior_aux_1,
        }
    )

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=intermediate_tensors,
    )

    expected_output = layer_hidden_states + layer_residual
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(aux_hidden_states[0], prior_aux_0)
    torch.testing.assert_close(aux_hidden_states[1], prior_aux_1)
    torch.testing.assert_close(aux_hidden_states[2], expected_output)
