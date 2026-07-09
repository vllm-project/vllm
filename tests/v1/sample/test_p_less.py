# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric unit tests for the p-less logits processor.

These pin the order-k truncation threshold from
https://arxiv.org/abs/2509.23234 and run on CPU so no accelerator is needed.
"""

import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import PLessLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate

VOCAB_SIZE = 512


def _reference(logits_row: torch.Tensor, order: float) -> torch.Tensor:
    """Reference p-less truncation for a single row."""
    probs = torch.softmax(logits_row, dim=-1)
    moment = probs.pow(order).sum()
    threshold = moment.pow(1.0 / (order - 1.0))
    return logits_row.masked_fill(probs < threshold, -float("inf"))


def _build_processor() -> PLessLogitsProcessor:
    vllm_config = VllmConfig()
    return PLessLogitsProcessor(
        vllm_config, device=torch.device("cpu"), is_pin_memory=False
    )


def test_p_less_matches_reference_for_multiple_orders():
    """Enabled rows (orders 2 and 3) match the closed-form threshold, and a
    disabled row is left untouched, all within a single mixed batch."""
    torch.manual_seed(0)
    proc = _build_processor()

    params = [
        SamplingParams(temperature=1.0, p_less=True, p_less_order=2.0),
        SamplingParams(temperature=1.0, p_less=True, p_less_order=3.0),
        SamplingParams(temperature=1.0),  # p-less disabled
    ]
    batch_update = BatchUpdate(
        batch_size=len(params),
        removed=[],
        added=[(i, p, None, []) for i, p in enumerate(params)],
        moved=[],
    )
    proc.update_state(batch_update)

    logits = torch.randn(len(params), VOCAB_SIZE)
    out = proc.apply(logits.clone())

    # Order 2 (the paper's squared-sum threshold).
    torch.testing.assert_close(out[0], _reference(logits[0], 2.0))
    # Order 3 (k-order Renyi generalization).
    torch.testing.assert_close(out[1], _reference(logits[1], 3.0))
    # Disabled row is unchanged.
    torch.testing.assert_close(out[2], logits[2])


def test_p_less_keeps_modal_token_and_truncates_tail():
    """p-less is argmax invariant: the most likely token always survives, and
    a long near-uniform tail is removed."""
    proc = _build_processor()
    proc.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, SamplingParams(temperature=1.0, p_less=True), None, [])],
            moved=[],
        )
    )

    logits = torch.full((1, VOCAB_SIZE), 0.01)
    logits[0, 0] = 10.0  # dominant token
    out = proc.apply(logits.clone())

    assert out[0, 0] != -float("inf")
    assert torch.isinf(out[0, 1:]).all()
    assert out.argmax(dim=-1).item() == 0


def test_p_less_disabled_is_noop():
    """With no request enabling p-less the processor returns logits as-is."""
    proc = _build_processor()
    proc.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            added=[(0, SamplingParams(temperature=1.0), None, [])],
            moved=[],
        )
    )
    logits = torch.randn(1, VOCAB_SIZE)
    out = proc.apply(logits.clone())
    torch.testing.assert_close(out, logits)


def test_p_less_order_must_exceed_one():
    """Orders at or below 1 are rejected at sampling-param validation time."""
    for bad_order in (1.0, 0.5, 0.0):
        try:
            SamplingParams(temperature=1.0, p_less=True, p_less_order=bad_order)
        except ValueError:
            continue
        raise AssertionError(f"order {bad_order} should have been rejected")
