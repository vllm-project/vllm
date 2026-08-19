# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.gemma3 import Gemma3Model
from vllm.model_executor.models.recirculation import RecirculationConfig

pytestmark = pytest.mark.skip_global_cleanup


def test_recirculation_mix_matches_destination_norm() -> None:
    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.25,
    )
    source = torch.tensor([[3.0, 4.0]])
    destination = torch.tensor([[0.0, 10.0]])

    mixed = config.mix(source, destination, torch.tensor([8]))

    torch.testing.assert_close(mixed, torch.tensor([[1.5, 9.5]]))


def test_recirculation_mix_ramps_convex_coefficients_by_position() -> None:
    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.2,
        ramp_tokens=10,
    )
    source = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    destination = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    mixed = config.mix(source, destination, torch.tensor([0, 5, 10]))

    expected = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]])
    torch.testing.assert_close(mixed, expected)


def test_recirculation_mix_supports_nonconvex_beta() -> None:
    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.2,
        beta=1.0,
    )
    source = torch.tensor([[1.0, 0.0]])
    destination = torch.tensor([[0.0, 1.0]])

    mixed = config.mix(source, destination, torch.tensor([4]))

    torch.testing.assert_close(mixed, torch.tensor([[0.2, 1.0]]))


@pytest.mark.parametrize(
    "raw_config",
    [
        {"source_layer": 2, "destination_layer": 2},
        {"source_layer": 3, "destination_layer": 0},
        {"source_layer": 2, "destination_layer": 0, "alpha": 1.1},
        {"source_layer": 2, "destination_layer": 0, "ramp_tokens": -1},
        {"source_layer": 2, "destination_layer": 0, "unexpected": True},
    ],
)
def test_recirculation_config_rejects_invalid_values(raw_config: dict) -> None:
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        recirculation_config=raw_config,
    )

    with pytest.raises(ValueError):
        RecirculationConfig.from_hf_config(hf_config)


class _AdditiveLayer(nn.Module):
    def __init__(self, layer_idx: int, calls: list[int]) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.calls = calls

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(self.layer_idx)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = torch.full_like(residual, self.layer_idx + 1)
        return hidden_states, residual


class _FinalNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states + residual, residual


def test_gemma3_returns_first_pass_and_recirculates_upper_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(
        "vllm.model_executor.models.gemma3.get_pp_group", lambda: pp_group
    )
    calls: list[int] = []
    model = Gemma3Model.__new__(Gemma3Model)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.layers = nn.ModuleList([_AdditiveLayer(i, calls) for i in range(3)])
    model.norm = _FinalNorm()
    model.recirculation_config = RecirculationConfig(
        source_layer=1,
        destination_layer=0,
        alpha=0.2,
    )

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros(1, 2),
    )

    torch.testing.assert_close(output, torch.full((1, 2), 6.0))
    assert calls == [0, 1, 2, 1, 2]
