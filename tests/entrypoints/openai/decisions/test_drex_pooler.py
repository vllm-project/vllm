# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Drex pointer readout.

The readout math lives in a torch-only leaf module. It is loaded by path so
the test does not import the heavier ``vllm.model_executor`` package.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

_HEAD_PATH = (
    Path(__file__).parents[4] / "vllm" / "model_executor" / "models" / "drex_head.py"
)


def _load_head_module():
    assert _HEAD_PATH.is_file(), f"missing pointer head module: {_HEAD_PATH}"
    spec = importlib.util.spec_from_file_location("drex_head_under_test", _HEAD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drex_head = _load_head_module()


def _random_head(hidden: int, head_dim: int) -> "torch.nn.Module":
    head = drex_head.PointerHead(hidden, head_dim)
    with torch.no_grad():
        head.q.weight.copy_(torch.randn(head_dim, hidden))
        head.q.bias.copy_(torch.randn(head_dim))
        head.k.weight.copy_(torch.randn(head_dim, hidden))
        head.k.bias.copy_(torch.randn(head_dim))
    return head.eval()


def _reference_scores(
    head: "torch.nn.Module",
    h_decide: torch.Tensor,
    h_opts: torch.Tensor,
) -> np.ndarray:
    w_q = head.q.weight.detach().numpy().astype(np.float64)
    b_q = head.q.bias.detach().numpy().astype(np.float64)
    w_k = head.k.weight.detach().numpy().astype(np.float64)
    b_k = head.k.bias.detach().numpy().astype(np.float64)
    query = h_decide.numpy().astype(np.float64) @ w_q.T + b_q
    keys = h_opts.numpy().astype(np.float64) @ w_k.T + b_k
    return (keys @ query) / math.sqrt(head.head_dim)


def test_pointer_head_matches_numpy_reference():
    torch.manual_seed(0)
    hidden, head_dim, options = 8, 4, 5
    head = _random_head(hidden, head_dim)
    h_decide = torch.randn(hidden)
    h_opts = torch.randn(options, hidden)

    got = head(h_decide, h_opts).detach().numpy()
    want = _reference_scores(head, h_decide, h_opts)

    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_temperature_scales_logits_only_in_eval():
    torch.manual_seed(1)
    head = _random_head(8, 4)
    h_decide = torch.randn(8)
    h_opts = torch.randn(5, 8)

    raw = head(h_decide, h_opts)
    head.temperature = 2.0
    warmed = head(h_decide, h_opts)
    torch.testing.assert_close(warmed, raw / 2.0)

    head.train()
    training = head(h_decide, h_opts)
    torch.testing.assert_close(training, raw)


def test_read_pointer_scores_uses_marked_positions():
    torch.manual_seed(2)
    hidden, head_dim = 8, 4
    head = _random_head(hidden, head_dim)
    hidden_states = torch.randn(6, hidden)

    got = drex_head.read_pointer_scores(hidden_states, [1, 4], 5, head)
    want = _reference_scores(head, hidden_states[5], hidden_states[[1, 4]])

    np.testing.assert_allclose(got.detach().numpy(), want, rtol=0, atol=1e-6)


@pytest.mark.parametrize("options", [1, 2, 5])
def test_read_pointer_scores_length_tracks_options(options):
    torch.manual_seed(3)
    head = _random_head(8, 4)
    hidden_states = torch.randn(options + 2, 8)
    positions = list(range(options))

    got = drex_head.read_pointer_scores(hidden_states, positions, options + 1, head)

    assert got.shape == (options,)


@pytest.mark.parametrize(
    "opt_positions,decide_position",
    [
        ([], 1),
        ([0, 1], None),
        ([0, 1], 9),
        ([0, 9], 1),
    ],
)
def test_read_pointer_scores_empty_when_markers_are_missing(
    opt_positions, decide_position
):
    head = _random_head(8, 4)
    hidden_states = torch.randn(3, 8)

    got = drex_head.read_pointer_scores(
        hidden_states, opt_positions, decide_position, head
    )

    assert got.shape == (0,)
