# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.trough_utils import (
    compute_trough_layer_range,
    prepare_trough_layer_states,
    vectorized_entropy_select,
)


def test_compute_trough_layer_range_explicit_backtrack() -> None:
    config = {
        "trough_max_backtrack_layers": 10,
        "trough_backtrack_ratio": 0.0,
    }
    start, count = compute_trough_layer_range(32, config)
    assert start == 22
    assert count == 10


def test_compute_trough_layer_range_ratio() -> None:
    config = {
        "trough_max_backtrack_layers": 0,
        "trough_backtrack_ratio": 0.25,
    }
    start, count = compute_trough_layer_range(20, config)
    assert count == 5
    assert start == 15


def test_compute_trough_layer_range_all_layers_by_default() -> None:
    config = {
        "trough_max_backtrack_layers": 0,
        "trough_backtrack_ratio": 0.0,
    }
    start, count = compute_trough_layer_range(12, config)
    assert start == 0
    assert count == 12


def test_prepare_trough_layer_states_requires_indices_when_misaligned() -> None:
    buffers = {8: torch.randn(3, 8, 4)}
    states, reason = prepare_trough_layer_states(
        trough_buffers=buffers,
        last_seq_len=8,
        last_logits_indices=None,
        sample_batch_size=2,
    )
    assert states is None
    assert reason == "missing_logits_indices"


def test_prepare_trough_layer_states_selects_rows() -> None:
    layer_states = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    buffers = {4: layer_states}
    indices = torch.tensor([1, 0], dtype=torch.int64)
    states, reason = prepare_trough_layer_states(
        trough_buffers=buffers,
        last_seq_len=4,
        last_logits_indices=indices,
        sample_batch_size=2,
    )
    assert reason is None
    assert states is not None
    assert torch.equal(states[:, 0], layer_states[:, 1])
    assert torch.equal(states[:, 1], layer_states[:, 0])


class _LinearLmHead(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(vocab_size, hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states @ self.weight.T


def _identity_logits_processor(lm_head, hidden_states: torch.Tensor) -> torch.Tensor:
    return lm_head(hidden_states)


def test_vectorized_entropy_select_p_zero_uses_final_layer() -> None:
    # Two layers, one token. Layer 0 is sharper; layer 1 is flatter.
    hidden = torch.tensor(
        [
            [[2.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    fallback = hidden[-1, :, :]
    lm_head = _LinearLmHead(hidden_size=4, vocab_size=4)

    selected, _, all_logits, _ = vectorized_entropy_select(
        layer_states=hidden,
        fallback_hidden_states=fallback,
        logits_processor=_identity_logits_processor,
        lm_head=lm_head,
        select_method="trough",
        trough_p=0.0,
        trough_max_backtrack_layers=1,
        trough_backtrack_ratio=0.0,
        trough_start_layer=0,
        total_model_layers=2,
        trough_log_interval=0,
        trough_call_count=1,
    )

    final_logits = _identity_logits_processor(lm_head, fallback)
    assert torch.allclose(selected, final_logits)


def test_vectorized_entropy_select_last_m1() -> None:
    hidden = torch.randn(4, 2, 8)
    fallback = hidden[-1]
    lm_head = _LinearLmHead(hidden_size=8, vocab_size=8)

    selected, _, all_logits, _ = vectorized_entropy_select(
        layer_states=hidden,
        fallback_hidden_states=fallback,
        logits_processor=_identity_logits_processor,
        lm_head=lm_head,
        select_method="last-m1",
        trough_p=1.0,
        trough_max_backtrack_layers=4,
        trough_backtrack_ratio=0.0,
        trough_start_layer=0,
        total_model_layers=4,
        trough_log_interval=0,
        trough_call_count=1,
    )

    expected = all_logits[2]
    assert torch.allclose(selected, expected)
