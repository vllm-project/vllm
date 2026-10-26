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


# The drafter's hook: Glm53MultiTokenPredictor.forward.

from vllm.models.glm53.nvidia import mtp as glm53_mtp  # noqa: E402


class _DraftSegment:
    normalized = False
    claimed_tokens = None

    def __init__(self, result: torch.Tensor | None) -> None:
        self.result = result
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, input_ids, positions, hidden_states):
        self.calls.append((input_ids, positions, hidden_states))
        return self.result

    def after_stock_forward(self, input_ids, positions, hidden_states):
        pass


class _StockEmbedding(nn.Module):
    def forward(self, input_ids):
        raise _StockPath


def _predictor() -> glm53_mtp.Glm53MultiTokenPredictor:
    model = object.__new__(glm53_mtp.Glm53MultiTokenPredictor)
    nn.Module.__init__(model)
    layer = nn.Module()
    layer.shared_head = nn.Module()
    layer.shared_head.norm = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.shared_head.norm.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    model.layers = nn.ModuleDict({"78": layer})
    model.mtp_start_layer_idx = 78
    model.num_mtp_layers = 1
    model.replicated_embed = False
    model.embed_tokens = _StockEmbedding()
    return model


def test_drafter_without_segment_or_declined_runs_stock():
    assert glm53_mtp.Glm53MultiTokenPredictor.glm53_segment is None
    model = _predictor()
    args = (torch.zeros(4, dtype=torch.int32), torch.arange(4), torch.zeros(4, 2))
    with pytest.raises(_StockPath):
        model(*args)
    segment = _DraftSegment(None)
    model.glm53_segment = segment
    with pytest.raises(_StockPath):
        model(*args)
    assert len(segment.calls) == 1 and segment.calls[0][2] is args[2]


def test_drafter_claimed_rows_return_the_shared_head_norm_twice():
    model = _predictor()
    rows = torch.arange(8, dtype=torch.float32).view(4, 2)
    model.glm53_segment = _DraftSegment(rows)
    with torch.no_grad():
        logits_hidden, recycled = model(
            torch.zeros(4, dtype=torch.int32), torch.arange(4), torch.zeros(4, 2)
        )
    torch.testing.assert_close(logits_hidden, rows * torch.tensor([2.0, 3.0]))
    assert recycled is logits_hidden


def test_drafter_claimed_sampled_row_is_broadcast_over_the_rows():
    model = _predictor()
    row = torch.tensor([[1.0, 2.0]])
    model.glm53_segment = _DraftSegment(row)
    with torch.no_grad():
        hidden, _ = model(
            torch.zeros(4, dtype=torch.int32), torch.arange(4), torch.zeros(4, 2)
        )
    assert hidden.shape == (4, 2)
    # vLLM's draft prefill reads row last_token_indices[r] of the result.
    torch.testing.assert_close(hidden[torch.tensor([2])], torch.tensor([[2.0, 6.0]]))


def test_drafter_normalized_rows_skip_the_norm_and_hand_over_their_tokens():
    model = _predictor()
    row = torch.tensor([[1.0, 2.0]])
    tokens = torch.tensor([7], dtype=torch.int32)
    segment = _DraftSegment(row)
    segment.normalized = True
    segment.claimed_tokens = lambda rows=None: tokens
    model.glm53_segment = segment
    with torch.no_grad():
        hidden, _ = model(
            torch.zeros(1, dtype=torch.int32), torch.arange(1), torch.zeros(1, 2)
        )
    torch.testing.assert_close(hidden, row)
    # vLLM's local-argmax draft sampling (use_local_argmax_reduction) takes them.
    assert model.get_top_tokens(hidden) is tokens
