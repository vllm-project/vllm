# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VL wrapper finalization contract with the language model."""

import importlib

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import WeightsMapper


class _FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_a = nn.Parameter(torch.zeros(1))
        self.tensor_c = nn.Parameter(torch.zeros(1))
        self.finalized_values: list[tuple[float, float]] = []

    def process_weights_after_loading(self) -> None:
        self.finalized_values.append((self.tensor_a.item(), self.tensor_c.item()))


@pytest.mark.parametrize(
    "module,cls_name",
    [
        (
            "vllm.models.deepseek_v4.common.vl_model",
            "DeepseekV4ForConditionalGeneration",
        ),
        ("vllm.models.deepseek_v41.nvidia.vl_model", "DeepseekV41ForCausalLM"),
        ("vllm.models.deepseek_v41.amd.vl_model", "DeepseekV41ForCausalLM"),
    ],
    ids=["v4", "v41-nvidia", "v41-amd"],
)
def test_vl_wrapper_delegates_finalization_to_hook(module, cls_name) -> None:
    wrapper_cls = getattr(importlib.import_module(module), cls_name)

    model = object.__new__(wrapper_cls)
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

    # A reload reruns the hook: no one-shot guard may skip it.
    model.process_weights_after_loading()
    assert model.language_model.finalized_values == [(1.0, 3.0)] * 2
