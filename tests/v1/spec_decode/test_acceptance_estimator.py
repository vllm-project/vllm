# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``predict`` writes one probability into two index spaces at once.

Adaptive verification reads the estimate in this step's batch order, while the
next step's refit reads it in persistent request-state slots. A mix-up between
the two is silent -- every value stays a plausible probability, it just belongs
to another request -- so it is asserted here rather than left to acceptance
metrics to hint at.
"""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for acceptance estimator tests", allow_module_level=True)

MAX_NUM_REQS = 16
NUM_STEPS = 4
VOCAB_SIZE = 2048


def _make_estimator(device: torch.device) -> OnlineAcceptanceEstimator:
    estimator = OnlineAcceptanceEstimator(MAX_NUM_REQS, NUM_STEPS, device)
    # Distinct per-position coefficients, so reading the wrong column shows up.
    estimator.coefficients[0] = torch.tensor([0.3, 0.5, 0.7, 0.9], device=device)
    estimator.coefficients[1] = torch.tensor([1.5, 1.0, 0.5, 0.0], device=device)
    return estimator


def _expected_probs(
    estimator: OnlineAcceptanceEstimator,
    logits: torch.Tensor,
    steps: torch.Tensor,
) -> torch.Tensor:
    top2 = logits.float().topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp(max=40.0)
    weight, bias = estimator.coefficients[0], estimator.coefficients[1]
    return torch.sigmoid(weight[steps] * margin + bias[steps])


@pytest.mark.parametrize("tokens_per_req", [1, NUM_STEPS])
def test_predict_agrees_across_slot_and_batch_order(tokens_per_req: int):
    """One row per request (per-step drafting) and a whole block at once.

    Which one it is is inferred from whether the draft step is per-row, so the
    two cases differ only in how ``draft_step`` is shaped.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    estimator = _make_estimator(device)
    num_reqs = 5

    # Batch position i holds a shuffled persistent slot, as a served batch would.
    slots = torch.randperm(MAX_NUM_REQS, device=device)[:num_reqs].to(torch.int32)
    if tokens_per_req == 1:
        idx_mapping = slots
        draft_step = torch.tensor(2, device=device, dtype=torch.int32)
        steps = draft_step.expand(num_reqs)
    else:
        idx_mapping = slots.repeat_interleave(NUM_STEPS)
        draft_step = torch.arange(NUM_STEPS, device=device, dtype=torch.int32).repeat(
            num_reqs
        )
        steps = draft_step
    batch_rows = torch.arange(num_reqs, device=device).repeat_interleave(tokens_per_req)
    logits = torch.randn(idx_mapping.shape[0], VOCAB_SIZE, device=device) * 4.0

    confidence_probs = torch.zeros(MAX_NUM_REQS, NUM_STEPS, device=device)
    estimator.predict(logits, idx_mapping, draft_step, confidence_probs)

    expected = _expected_probs(estimator, logits, steps.long())
    torch.testing.assert_close(
        estimator.predictions[idx_mapping.long(), steps.long()],
        expected,
        atol=2e-3,
        rtol=2e-3,
    )
    torch.testing.assert_close(
        confidence_probs[batch_rows, steps.long()], expected, atol=2e-3, rtol=2e-3
    )


def test_predict_skips_cudagraph_padded_rows():
    """Padded rows carry slot -1 and must not land on a live request."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    estimator = _make_estimator(device)
    num_reqs, num_padded = 3, 6

    slots = torch.tensor([7, 2, 11], device=device, dtype=torch.int32)
    idx_mapping = torch.cat(
        [slots, torch.full((num_padded,), -1, device=device, dtype=torch.int32)]
    )
    draft_step = torch.tensor(1, device=device, dtype=torch.int32)
    logits = torch.randn(num_reqs + num_padded, VOCAB_SIZE, device=device) * 4.0

    confidence_probs = torch.zeros(MAX_NUM_REQS, NUM_STEPS, device=device)
    estimator.predict(logits, idx_mapping, draft_step, confidence_probs)

    live = (slots.long(), torch.full((num_reqs,), 1, device=device))
    expected = _expected_probs(estimator, logits[:num_reqs], live[1])
    torch.testing.assert_close(
        estimator.predictions[live], expected, atol=2e-3, rtol=2e-3
    )
    # Nothing outside the live (request, position) cells was touched.
    untouched = torch.ones(MAX_NUM_REQS, NUM_STEPS, dtype=torch.bool, device=device)
    untouched[live] = False
    assert not estimator.predictions[untouched].any()
    assert not confidence_probs[num_reqs:].any()
