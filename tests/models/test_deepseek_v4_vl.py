# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VL wrapper finalization contract with the language model."""

import torch
from torch import nn

from vllm.model_executor.models.utils import WeightsMapper


class _FakeLanguageModel(nn.Module):
    finalizes_weights_during_load = False

    def __init__(self) -> None:
        super().__init__()
        self.tensor_a = nn.Parameter(torch.zeros(1))
        self.tensor_c = nn.Parameter(torch.zeros(1))
        self.finalized_values: list[tuple[float, float]] = []

    def process_weights_after_loading(self) -> None:
        self.finalized_values.append((self.tensor_a.item(), self.tensor_c.item()))

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1


def test_vl_wrapper_streams_then_delegates_finalization() -> None:
    from vllm.models.deepseek_v4.common.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    model = object.__new__(DeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = _FakeLanguageModel()
    model.vision = nn.Module()
    model.vision.tensor_b = nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = WeightsMapper()

    def interleaved_weights():
        yield "language_model.tensor_a", torch.tensor([1.0])
        assert model.language_model.tensor_a.item() == 1.0
        yield "vision.tensor_b", torch.tensor([2.0])
        assert model.vision.tensor_b.item() == 2.0
        yield "language_model.tensor_c", torch.tensor([3.0])

    loaded = model.load_weights(interleaved_weights())

    assert loaded == {
        "language_model.tensor_a",
        "vision.tensor_b",
        "language_model.tensor_c",
    }
    assert model.language_model.finalized_values == []

    model.process_weights_after_loading()

    assert model.language_model.finalized_values == [(1.0, 3.0)]
    assert torch.equal(
        model.compute_logits_local(torch.tensor([4.0])), torch.tensor([5.0])
    )
    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(1.0, 3.0)]


class _FakeFinalizingLanguageModel(_FakeLanguageModel):
    finalizes_weights_during_load = True

    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0

    def load_weights(self, weights) -> set[str]:
        self.load_calls += 1
        loaded = set()
        for name, value in weights:
            getattr(self, name).data.copy_(value)
            loaded.add(name)
        self.process_weights_after_loading()
        return loaded


def test_vl_wrapper_groups_child_that_finalizes_during_load() -> None:
    from vllm.models.deepseek_v4.common.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    model = object.__new__(DeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = _FakeFinalizingLanguageModel()
    model.vision = nn.Module()
    model.vision.tensor_b = nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = WeightsMapper()

    loaded = model.load_weights(
        iter(
            (
                ("language_model.tensor_a", torch.tensor([1.0])),
                ("vision.tensor_b", torch.tensor([2.0])),
                ("language_model.tensor_c", torch.tensor([3.0])),
            )
        )
    )

    assert loaded == {
        "language_model.tensor_a",
        "vision.tensor_b",
        "language_model.tensor_c",
    }
    assert model.language_model.load_calls == 1
    assert model.language_model.finalized_values == [(1.0, 3.0)]

    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(1.0, 3.0)]


def test_vl_wrapper_dummy_load_delegates_finalization() -> None:
    from vllm.models.deepseek_v4.common.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    model = object.__new__(DeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = _FakeFinalizingLanguageModel()

    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(0.0, 0.0)]
    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(0.0, 0.0)]
