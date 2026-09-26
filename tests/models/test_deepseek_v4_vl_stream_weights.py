# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4/V4.1 VL wrappers stream the language model's weights to it
as one contiguous group."""

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import WeightsMapper
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA implementations"
)


class _FakeFinalizingLanguageModel(nn.Module):
    finalizes_weights_during_load = True

    def __init__(self) -> None:
        super().__init__()
        self.tensor_a = nn.Parameter(torch.zeros(1))
        self.tensor_c = nn.Parameter(torch.zeros(1))
        self.load_calls = 0
        self.finalized_values: list[tuple[float, float]] = []

    def load_weights(self, weights) -> set[str]:
        self.load_calls += 1
        loaded = set()
        for name, value in weights:
            getattr(self, name).data.copy_(value)
            loaded.add(name)
        self.finalized_values.append((self.tensor_a.item(), self.tensor_c.item()))
        return loaded


def _v4_wrapper():
    from vllm.models.deepseek_v4.common.vl_model import (
        DeepseekV4ForConditionalGeneration,
    )

    return DeepseekV4ForConditionalGeneration


def _v41_wrapper():
    from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM

    return DeepseekV41ForCausalLM


@pytest.mark.parametrize("wrapper", [_v4_wrapper, _v41_wrapper], ids=["v4", "v41"])
def test_load_weights_streams_language_model_group(wrapper) -> None:
    model = object.__new__(wrapper())
    nn.Module.__init__(model)
    model.language_model = _FakeFinalizingLanguageModel()
    model.vision = nn.Module()
    model.vision.tensor_b = nn.Parameter(torch.zeros(1))
    model.hf_to_vllm_mapper = WeightsMapper()

    def interleaved_weights():
        yield "vision.tensor_b", torch.tensor([2.0])
        yield "language_model.tensor_a", torch.tensor([1.0])
        # Loaded before the stream is drained, not after.
        assert model.language_model.tensor_a.item() == 1.0
        yield "language_model.tensor_c", torch.tensor([3.0])

    loaded = model.load_weights(interleaved_weights())

    assert loaded == {
        "language_model.tensor_a",
        "vision.tensor_b",
        "language_model.tensor_c",
    }
    assert model.language_model.load_calls == 1
    assert model.language_model.finalized_values == [(1.0, 3.0)]
    assert model.vision.tensor_b.item() == 2.0
