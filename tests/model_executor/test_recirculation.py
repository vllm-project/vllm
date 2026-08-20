# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from vllm.model_executor.models.interfaces import supports_recirculation
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
from vllm.model_executor.models.mistral import MistralModel
from vllm.model_executor.models.recirculation import RecirculationConfig

pytestmark = pytest.mark.skip_global_cleanup


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
        return torch.full_like(residual, self.layer_idx + 1), residual


class _FinalNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        assert residual is not None
        return hidden_states + residual, None


def _make_llama_model(
    monkeypatch: pytest.MonkeyPatch,
    model_type: type[LlamaModel] = LlamaModel,
) -> tuple[LlamaModel, list[int]]:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(
        "vllm.model_executor.models.llama.get_pp_group", lambda: pp_group
    )
    calls: list[int] = []
    model = cast(LlamaModel, object.__new__(model_type))
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.layers = nn.ModuleList([_AdditiveLayer(i, calls) for i in range(3)])
    model.norm = _FinalNorm()
    model.recirculation_config = RecirculationConfig(
        source_layer=1,
        destination_layer=0,
        alpha=0.2,
        wavefront=True,
    )
    return model, calls


def test_llama_uses_shared_wavefront_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_llama_model(monkeypatch)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 2),
        recirculation_wavefront_warmup=True,
    )

    torch.testing.assert_close(output[0:1], torch.full((1, 2), 6.0))
    torch.testing.assert_close(output[1:2], torch.ones(1, 2))
    assert calls == [0, 1, 2]


def test_llama_top_level_advertises_engine_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _make_llama_model(monkeypatch)
    causal_lm = LlamaForCausalLM.__new__(LlamaForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = model

    assert supports_recirculation(causal_lm)


def test_mistral_delegates_to_shared_wavefront_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_llama_model(monkeypatch, MistralModel)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 2),
        t_cond=None,
        recirculation_wavefront_warmup=True,
    )

    torch.testing.assert_close(output[0:1], torch.full((1, 2), 6.0))
    torch.testing.assert_close(output[1:2], torch.ones(1, 2))
    assert calls == [0, 1, 2]


def test_unvalidated_llama_subclass_does_not_inherit_adapter() -> None:
    class UnvalidatedLlamaModel(LlamaModel):
        pass

    model = UnvalidatedLlamaModel.__new__(UnvalidatedLlamaModel)

    assert not model.has_recirculation_adapter()


def test_engine_capability_rejects_incomplete_forward() -> None:
    class IncompleteModel:
        supports_recirculation = True

        def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            return input_ids

    assert not supports_recirculation(IncompleteModel())
