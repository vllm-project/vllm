# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.confidence_stop import DraftConfidenceStop

VOCAB = 20000
NUM_STEPS = 4
FALLBACK = 2
THRESHOLD = 0.6

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="needs a CUDA device"
)


def _logits(top_probs: list[float]) -> torch.Tensor:
    """bf16 rows whose top-1 probability is approximately each value."""
    rows = torch.full((len(top_probs), VOCAB), -30.0)
    for row, p in enumerate(top_probs):
        rows[row, 5] = 0.0
        rows[row, VOCAB - 3] = rows[row, VOCAB - 9000] = float(
            np.log((1 - p) / (2 * p))
        )
    return rows.to(device="cuda", dtype=torch.bfloat16)


def _run_step(stop, top_probs, step):
    stop.update(_logits(top_probs), torch.tensor([step], device="cuda"))
    stop.step_launched()


@requires_cuda
def test_chain_state_follows_top_probabilities():
    stop = DraftConfidenceStop(THRESHOLD, FALLBACK, 8, NUM_STEPS, torch.device("cuda"))
    stop.begin_round(3)
    # Only the first row is the request; the second is cudagraph padding.
    _run_step(stop, [0.9, 0.1], 0)
    assert stop.should_continue()
    _run_step(stop, [0.7, 0.1], 1)
    assert stop.should_continue()
    _run_step(stop, [0.4, 0.99], 2)
    assert not stop.should_continue()
    # Steps 0 and 1 cleared the threshold; step 2's draft did not.
    assert stop.num_verifiable_drafts(np.array([3])).tolist() == [2]

    # A dead chain stays dead within a round, and step 0 starts a new chain.
    stop.update(_logits([0.99]), torch.tensor([3], device="cuda"))
    assert not stop.alive.item()
    stop.begin_round(1)
    _run_step(stop, [0.99], 0)
    assert stop.alive.item()


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
    stop = DraftConfidenceStop(THRESHOLD, FALLBACK, 4, NUM_STEPS, torch.device("cuda"))
    stop.begin_round(0)
    _run_step(stop, [first_prob], 0)
    if first_prob >= THRESHOLD:
        assert stop.should_continue()
        _run_step(stop, [0.1], 1)
    assert not stop.should_continue()
    assert stop.num_verifiable_drafts(np.array([0])).tolist() == [expected]


@requires_cuda
@pytest.mark.parametrize(("last_prob", "expected"), [(0.9, 4), (0.1, 3)])
def test_full_round_resolves_last_step_lazily(last_prob, expected):
    stop = DraftConfidenceStop(THRESHOLD, FALLBACK, 4, NUM_STEPS, torch.device("cuda"))
    stop.begin_round(2)
    for step in range(NUM_STEPS - 1):
        _run_step(stop, [0.9], step)
        assert stop.should_continue()
    _run_step(stop, [last_prob], NUM_STEPS - 1)
    stop.end_round()
    assert stop.num_verifiable_drafts(np.array([2])).tolist() == [expected]


@requires_cuda
def test_fixed_round_verifies_its_depth():
    """Multi-request rounds draft the fallback depth and skip the stop."""
    stop = DraftConfidenceStop(THRESHOLD, FALLBACK, 8, NUM_STEPS, torch.device("cuda"))
    stop.fixed_round(np.array([4, 6]), FALLBACK)
    assert stop.num_verifiable_drafts(np.array([4, 6])).tolist() == [FALLBACK] * 2
    # A later single-request round only updates its own slot.
    stop.begin_round(4)
    _run_step(stop, [0.1], 0)
    assert not stop.should_continue()
    assert stop.num_verifiable_drafts(np.array([4, 6])).tolist() == [1, FALLBACK]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("is_cuda", "capability", "fallback", "error"),
    [
        (True, 121, FALLBACK, None),
        (True, 120, FALLBACK, "SM121"),
        (True, 100, FALLBACK, "SM121"),
        (False, None, FALLBACK, "SM121"),
        (True, 121, None, "fallback_depth"),
        (True, 121, NUM_STEPS, "fallback_depth"),
    ],
)
def test_confidence_stop_config(monkeypatch, is_cuda, capability, fallback, error):
    """SM121 only (the per-step readback stalls faster GPUs), and a fallback
    depth below num_speculative_tokens is required."""
    from vllm.config import VllmConfig

    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(
            is_cuda=lambda: is_cuda, is_device_capability=lambda cap: cap == capability
        ),
    )
    spec = SimpleNamespace(
        draft_confidence_threshold=THRESHOLD,
        draft_confidence_fallback_depth=fallback,
        method="mtp",
        num_speculative_tokens=NUM_STEPS,
        use_multi_module_mtp=lambda: False,
        use_gemma4_mtp=lambda: False,
        uses_dynamic_speculative_decoding=lambda: False,
        enable_adaptive_verification=False,
        use_local_argmax_reduction=False,
        parallel_drafting=False,
    )
    config = SimpleNamespace(
        speculative_config=spec,
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    if error is None:
        VllmConfig._verify_draft_confidence_threshold(config)
    else:
        with pytest.raises(ValueError, match=error):
            VllmConfig._verify_draft_confidence_threshold(config)
