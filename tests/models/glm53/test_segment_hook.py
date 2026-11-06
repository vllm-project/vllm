# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The optional decode-segment hook of Glm53Model.forward."""

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.models.glm53.nvidia import model as glm53_model


class _StockPath(Exception):
    """Raised where the stock forward starts (the pipeline-group lookup)."""


class _Segment:
    def __init__(self, result: torch.Tensor | None) -> None:
        self.result = result
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        self.calls.append((input_ids, positions))
        return self.result


class _Layers(nn.Module):
    def __iter__(self):
        raise AssertionError("a claimed forward must not run the layers")


def _model(monkeypatch: pytest.MonkeyPatch) -> glm53_model.Glm53Model:
    model = object.__new__(glm53_model.Glm53Model)
    nn.Module.__init__(model)
    model.norm = nn.Identity()
    model.layers = _Layers()

    def stock_path():
        raise _StockPath

    monkeypatch.setattr(deepseek_v32_model, "get_pp_group", stock_path)
    return model


def test_no_segment_is_the_default_and_runs_stock(monkeypatch):
    assert glm53_model.Glm53Model.glm53_segment is None
    model = _model(monkeypatch)
    with pytest.raises(_StockPath):
        model(torch.zeros(4, dtype=torch.int32), torch.arange(4))


def test_declining_segment_runs_stock(monkeypatch):
    model = _model(monkeypatch)
    segment = _Segment(None)
    model.glm53_segment = segment
    input_ids, positions = torch.zeros(4, dtype=torch.int32), torch.arange(4)
    with pytest.raises(_StockPath):
        model(input_ids, positions)
    assert len(segment.calls) == 1 and segment.calls[0][0] is input_ids


def test_claimed_forward_returns_the_final_norm_of_the_segment_output(monkeypatch):
    model = _model(monkeypatch)
    hidden = torch.arange(8, dtype=torch.float32).view(4, 2)
    model.norm = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.norm.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    model.glm53_segment = _Segment(hidden)
    with torch.no_grad():
        output = model(torch.zeros(4, dtype=torch.int32), torch.arange(4))
    torch.testing.assert_close(output, hidden * torch.tensor([2.0, 3.0]))
