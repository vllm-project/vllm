# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.model_executor.models import sarvam as sarvam_mod
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.sarvam import SarvamMLAForCausalLM, SarvamMLAModel
from vllm.sequence import IntermediateTensors


def _make_model(
    *,
    start_layer: int = 0,
    end_layer: int = 1,
    layers: list | None = None,
    aux_hidden_state_layers: tuple[int, ...] = (),
) -> SarvamMLAModel:
    """Build a SarvamMLAModel without running __init__.

    Constructing the real module needs a full VllmConfig plus an initialized
    distributed environment, neither of which the aux hidden state plumbing
    depends on.
    """
    model = object.__new__(SarvamMLAModel)
    object.__setattr__(model, "start_layer", start_layer)
    object.__setattr__(model, "end_layer", end_layer)
    object.__setattr__(model, "layers", layers if layers is not None else [])
    object.__setattr__(model, "aux_hidden_state_layers", aux_hidden_state_layers)
    object.__setattr__(model, "embedding_dropout", lambda x: x)
    # RMSNorm in fused-residual mode returns (norm(h + r), h + r); the scaling
    # is irrelevant here, so pass the sum through unchanged.
    object.__setattr__(
        model, "norm", lambda h, r=None: (h + r, h + r) if r is not None else h
    )
    return model


def _layer(hidden_states: torch.Tensor, residual: torch.Tensor) -> Mock:
    return Mock(return_value=(hidden_states, residual))


def _patch_pp_group(monkeypatch, *, is_first_rank=True, is_last_rank=True) -> None:
    monkeypatch.setattr(
        sarvam_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=is_first_rank, is_last_rank=is_last_rank),
    )


def test_sarvam_mla_advertises_eagle3_support():
    assert supports_eagle3(SarvamMLAForCausalLM)


def test_sarvam_mla_configures_aux_hidden_state_layers():
    # SarvamMLAForCausalLM inherits both hooks from the SupportsEagle3
    # protocol, so this pins that the protocol defaults actually reach the
    # inner SarvamMLAModel rather than silently doing nothing.
    target = object.__new__(SarvamMLAForCausalLM)
    torch.nn.Module.__init__(target)
    model = _make_model(layers=[None] * 12)
    object.__setattr__(target, "model", model)

    target.set_aux_hidden_state_layers((2, 6, 9))

    assert model.aux_hidden_state_layers == (2, 6, 9)
    assert target.get_eagle3_default_aux_hidden_state_layers() == (2, 6, 9)


def test_sarvam_mla_forward_captures_aux_hidden_states(monkeypatch):
    inputs_embeds = torch.tensor([[1.0, 2.0]])
    layer_hidden_states = torch.tensor([[3.0, 4.0]])
    layer_residual = torch.tensor([[5.0, 6.0]])
    model = _make_model(
        layers=[_layer(layer_hidden_states, layer_residual)],
        aux_hidden_state_layers=(0, 1),
    )
    _patch_pp_group(monkeypatch)

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )

    # Index 0 is the embedding output; index 1 is the output of layer 0, which
    # is only complete once the pending residual is added back.
    expected_layer_output = layer_hidden_states + layer_residual
    torch.testing.assert_close(output, expected_layer_output)
    assert len(aux_hidden_states) == 2
    torch.testing.assert_close(aux_hidden_states[0], inputs_embeds)
    torch.testing.assert_close(aux_hidden_states[1], expected_layer_output)


def test_sarvam_mla_forward_returns_bare_tensor_without_eagle3(monkeypatch):
    # Without a drafter the runner unpacks a single tensor, so returning a
    # tuple unconditionally would break every non-speculative request.
    model = _make_model(
        layers=[_layer(torch.tensor([[3.0, 4.0]]), torch.tensor([[5.0, 6.0]]))],
    )
    _patch_pp_group(monkeypatch)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.tensor([[1.0, 2.0]]),
    )

    assert isinstance(output, torch.Tensor)


def test_sarvam_mla_forward_uses_absolute_layer_indices(monkeypatch):
    # On a pipeline stage that does not start at layer 0, the capture points
    # are numbered by absolute layer index. With relative numbering the
    # requested layers would silently resolve to different tensors.
    stage_hidden_states = torch.tensor([[1.0, 2.0]])
    stage_residual = torch.tensor([[0.5, 0.5]])
    second_hidden_states = torch.tensor([[3.0, 4.0]])
    second_residual = torch.tensor([[5.0, 6.0]])
    model = _make_model(
        start_layer=1,
        end_layer=3,
        layers=[
            None,
            _layer(torch.tensor([[7.0, 8.0]]), torch.tensor([[9.0, 10.0]])),
            _layer(second_hidden_states, second_residual),
        ],
        aux_hidden_state_layers=(1, 3),
    )
    _patch_pp_group(monkeypatch, is_first_rank=False)

    _, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=IntermediateTensors(
            {"hidden_states": stage_hidden_states, "residual": stage_residual}
        ),
        inputs_embeds=None,
    )

    assert len(aux_hidden_states) == 2
    torch.testing.assert_close(
        aux_hidden_states[0], stage_hidden_states + stage_residual
    )
    torch.testing.assert_close(
        aux_hidden_states[1], second_hidden_states + second_residual
    )
