# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.confidence_stop import DraftConfidenceStop

VOCAB = 20000  # Spans several kernel blocks.
NUM_STEPS = 4
THRESHOLD = 0.6

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="needs a CUDA device"
)


def _logits(top_probs: list[float]) -> torch.Tensor:
    """bf16 rows whose top-1 probability is approximately each value."""
    rows = torch.full((len(top_probs), VOCAB), -30.0)
    for row, p in enumerate(top_probs):
        # The remaining mass spreads over two tokens in different blocks.
        rows[row, 5] = 0.0
        rows[row, VOCAB - 3] = rows[row, VOCAB - 9000] = float(
            np.log((1 - p) / (2 * p))
        )
    return rows.to(device="cuda", dtype=torch.bfloat16)


def _run_step(stop, top_probs, idx_mapping, step):
    stop.update(
        _logits(top_probs),
        torch.tensor(idx_mapping, dtype=torch.int32, device="cuda"),
        torch.tensor(step, device="cuda"),
    )
    stop.step_launched()


@requires_cuda
def test_chain_state_follows_top_probabilities():
    stop = DraftConfidenceStop(THRESHOLD, 8, NUM_STEPS, torch.device("cuda"))
    # Row 2 is cudagraph padding and must never keep the batch drafting.
    idx_mapping = [3, 1, -1]
    stop.begin_round(np.array([3, 1]))

    _run_step(stop, [0.9, 0.3, 0.99], idx_mapping, 0)
    assert stop.alive[:3].tolist() == [1, 0, 0]
    assert stop.should_continue()
    # A dead chain stays dead even when the drafter becomes confident again.
    _run_step(stop, [0.7, 0.99, 0.99], idx_mapping, 1)
    assert stop.alive[:3].tolist() == [1, 0, 0]
    assert stop.should_continue()
    _run_step(stop, [0.4, 0.99, 0.99], idx_mapping, 2)
    assert stop.alive[:3].tolist() == [0, 0, 0]
    assert not stop.should_continue()
    # Steps 0 and 1 cleared the threshold for slot 3; step 2's draft did not.
    assert stop.num_verifiable_drafts(np.array([3, 1])).tolist() == [2, 2]

    # The first step of a new round resets every chain.
    stop.begin_round(np.array([1]))
    _run_step(stop, [0.99], [1], 0)
    assert stop.alive[0].item() == 1


@requires_cuda
@pytest.mark.parametrize(
    ("first_prob", "expected"),
    [
        # Nothing survives the first draft, which is still verified.
        (0.1, 1),
        (0.9, 1),
    ],
)
def test_stop_after_first_draft_keeps_it(first_prob, expected):
    stop = DraftConfidenceStop(THRESHOLD, 4, NUM_STEPS, torch.device("cuda"))
    stop.begin_round(np.array([0]))
    _run_step(stop, [first_prob], [0], 0)
    if first_prob >= THRESHOLD:
        assert stop.should_continue()
        _run_step(stop, [0.1], [0], 1)
    assert not stop.should_continue()
    assert stop.num_verifiable_drafts(np.array([0])).tolist() == [expected]


@requires_cuda
@pytest.mark.parametrize(("last_prob", "expected"), [(0.9, 4), (0.1, 3)])
def test_full_round_resolves_last_step_lazily(last_prob, expected):
    stop = DraftConfidenceStop(THRESHOLD, 4, NUM_STEPS, torch.device("cuda"))
    stop.begin_round(np.array([2]))
    for step in range(NUM_STEPS - 1):
        _run_step(stop, [0.9], [2], step)
        assert stop.should_continue()
    _run_step(stop, [last_prob], [2], NUM_STEPS - 1)
    stop.end_round()
    assert stop.num_verifiable_drafts(np.array([2])).tolist() == [expected]


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("decisions", "expected_launches", "expect_end_round"),
    [
        # Stop before the second decode step.
        ([True, False], 1, False),
        ([False], 0, False),
        # Never stopped: the round ends with its last step still pending.
        ([True, True, True], 3, True),
    ],
)
def test_multi_step_decode_stops_launching(
    decisions, expected_launches, expect_end_round
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = NUM_STEPS
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2), query_start_loc=torch.arange(3)
    )
    speculator.idx_mapping = torch.arange(2)
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    stop = Mock()
    stop.should_continue.side_effect = decisions

    speculator._multi_step_decode(
        num_reqs=2,
        skip_attn=True,
        batch_desc=BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL, num_tokens=2, num_reqs=2
        ),
        num_tokens_across_dp=None,
        seq_lens_cpu_upper_bound=None,
        confidence_stop=stop,
    )

    assert run_fullgraph.call_count == expected_launches
    assert stop.step_launched.call_count == expected_launches
    assert stop.end_round.called == expect_end_round
