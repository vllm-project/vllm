# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader


def _loader(extra_config: dict) -> DummyModelLoader:
    return DummyModelLoader(
        LoadConfig(load_format="dummy", model_loader_extra_config=extra_config)
    )


@pytest.mark.parametrize(
    ("extra_config", "bound"), [({}, 1e-3), ({"dummy_weight_range": 0.1}, 0.1)]
)
def test_dummy_weight_range(extra_config: dict, bound: float):
    model = nn.Linear(256, 256, dtype=torch.float32)
    _loader(extra_config).load_weights(model, model_config=None)
    largest = model.weight.detach().abs().max().item()
    assert bound / 2 < largest <= bound


def test_dummy_weight_range_keeps_fp8_default():
    model = nn.Linear(256, 256, dtype=torch.bfloat16)
    model.weight.data = model.weight.data.to(torch.float8_e4m3fn)
    _loader({"dummy_weight_range": 0.1}).load_weights(model, model_config=None)
    assert model.weight.detach().float().abs().max().item() > 0.5


def test_dummy_rejects_unknown_extra_config():
    with pytest.raises(ValueError, match="dummy_weight_range"):
        _loader({"unknown": 1})
