# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LogitBiasLogitsProcessor.apply_with_spec_decode()."""

import pytest
import torch

from vllm.v1.sample.logits_processor import BatchUpdate, LogitBiasLogitsProcessor
from vllm.sampling_params import SamplingParams

VOCAB_SIZE = 128
DEVICE = torch.device("cpu")


def _make_processor(
    biases: dict[int, dict[int, float]],
) -> LogitBiasLogitsProcessor:
    """Create a LogitBiasLogitsProcessor with pre-set biases.

    ``biases`` maps *batch index* → {token_id: bias_value}.
    We feed each entry through update_state so internal tensors are built.
    """
    proc = LogitBiasLogitsProcessor(None, DEVICE, False)

    # Build a BatchUpdate that adds one request per entry.
    # AddedRequest = (index, params, prompt_tok_ids, output_tok_ids)
    added = []
    batch_size = 0
    for req_idx, token_biases in biases.items():
        params = SamplingParams(logit_bias=token_biases)
        added.append((req_idx, params, None, []))
        batch_size = max(batch_size, req_idx + 1)

    if added:
        batch_update = BatchUpdate(
            batch_size=batch_size,
            added=added,
            removed=[],
            moved=[],
        )
        proc.update_state(batch_update)
    return proc


class TestLogitBiasApplyWithSpecDecode:
    """Tests for apply_with_spec_decode on LogitBiasLogitsProcessor."""

    def test_no_biases_returns_logits_unchanged(self):
        """When no biases are configured the logits tensor is untouched."""
        proc = LogitBiasLogitsProcessor(None, DEVICE, False)
        logits = torch.zeros(6, VOCAB_SIZE)
        result = proc.apply_with_spec_decode(logits, [2, 3, 1])
        assert torch.equal(result, torch.zeros(6, VOCAB_SIZE))

    def test_single_request_single_token_bias(self):
        """Bias for one request with one token is applied to all its rows."""
        biases = {0: {5: 1.5}}
        proc = _make_processor(biases)

        num_draft_tokens = [3]
        logits = torch.zeros(3, VOCAB_SIZE)
        logits = proc.apply_with_spec_decode(logits, num_draft_tokens)

        # All 3 rows should have token 5 biased by 1.5.
        for row in range(3):
            assert logits[row, 5].item() == pytest.approx(1.5), (
                f"row {row}, token 5 should be biased"
            )
            # Other tokens remain zero.
            assert logits[row, 0].item() == pytest.approx(0.0)

    def test_single_request_multiple_token_biases(self):
        """Multiple token biases for a single request."""
        biases = {0: {10: 2.0, 20: -1.0}}
        proc = _make_processor(biases)

        num_draft_tokens = [2]
        logits = torch.ones(2, VOCAB_SIZE)
        logits = proc.apply_with_spec_decode(logits, num_draft_tokens)

        for row in range(2):
            assert logits[row, 10].item() == pytest.approx(3.0)
            assert logits[row, 20].item() == pytest.approx(0.0)
            # Unbiased token stays at 1.0.
            assert logits[row, 0].item() == pytest.approx(1.0)

    def test_multiple_requests_correct_row_assignment(self):
        """Biases are applied to the correct row ranges per request."""
        # req 0 → rows 0-1, req 1 → rows 2-4 (no bias), req 2 → row 5
        biases = {0: {7: 0.5}, 2: {3: -2.0}}
        proc = _make_processor(biases)

        num_draft_tokens = [2, 3, 1]
        logits = torch.zeros(6, VOCAB_SIZE)
        logits = proc.apply_with_spec_decode(logits, num_draft_tokens)

        # Request 0: rows 0-1
        for row in range(2):
            assert logits[row, 7].item() == pytest.approx(0.5)
        # Request 1: rows 2-4 should be untouched (no bias configured)
        for row in range(2, 5):
            assert logits[row, 7].item() == pytest.approx(0.0)
            assert logits[row, 3].item() == pytest.approx(0.0)
        # Request 2: row 5
        assert logits[5, 3].item() == pytest.approx(-2.0)
        assert logits[5, 7].item() == pytest.approx(0.0)

    def test_request_with_zero_draft_tokens_skipped(self):
        """A request with 0 draft tokens should not cause errors."""
        biases = {0: {5: 1.0}, 1: {5: 2.0}}
        proc = _make_processor(biases)

        # Request 0 has 0 draft tokens; request 1 has 2.
        num_draft_tokens = [0, 2]
        logits = torch.zeros(2, VOCAB_SIZE)
        logits = proc.apply_with_spec_decode(logits, num_draft_tokens)

        # Request 0 contributes no rows; request 1 owns rows 0-1.
        for row in range(2):
            assert logits[row, 5].item() == pytest.approx(2.0)

    def test_bias_values_are_additive(self):
        """Biases add to existing logit values (not replace)."""
        biases = {0: {0: 3.0}}
        proc = _make_processor(biases)

        logits = torch.full((2, VOCAB_SIZE), 1.0)
        logits = proc.apply_with_spec_decode(logits, [2])

        assert logits[0, 0].item() == pytest.approx(4.0)
        assert logits[1, 0].item() == pytest.approx(4.0)

    def test_negative_bias(self):
        """Negative biases reduce logit values."""
        biases = {0: {10: -5.0}}
        proc = _make_processor(biases)

        logits = torch.zeros(1, VOCAB_SIZE)
        logits = proc.apply_with_spec_decode(logits, [1])

        assert logits[0, 10].item() == pytest.approx(-5.0)

    def test_all_requests_have_biases(self):
        """Every request in the batch has a bias configured."""
        biases = {0: {1: 0.1}, 1: {2: 0.2}, 2: {3: 0.3}}
        proc = _make_processor(biases)

        num_draft_tokens = [1, 2, 1]
        logits = torch.zeros(4, VOCAB_SIZE)
        logits = proc.apply_with_spec_decode(logits, num_draft_tokens)

        # req 0 → row 0
        assert logits[0, 1].item() == pytest.approx(0.1)
        # req 1 → rows 1-2
        assert logits[1, 2].item() == pytest.approx(0.2)
        assert logits[2, 2].item() == pytest.approx(0.2)
        # req 2 → row 3
        assert logits[3, 3].item() == pytest.approx(0.3)

    def test_consistency_with_apply(self):
        """apply_with_spec_decode with 1 draft token per request should
        produce the same result as apply() on a normal batch."""
        biases = {0: {5: 1.5, 10: -0.5}, 1: {20: 2.0}}
        proc = _make_processor(biases)

        batch_size = 2
        logits_spec = torch.randn(batch_size, VOCAB_SIZE)
        logits_normal = logits_spec.clone()

        # apply_with_spec_decode: 1 draft token per request = normal batch
        logits_spec = proc.apply_with_spec_decode(
            logits_spec, [1] * batch_size
        )
        # apply: standard path
        logits_normal = proc.apply(logits_normal)

        assert torch.allclose(logits_spec, logits_normal), (
            "apply_with_spec_decode with 1 draft token should match apply()"
        )
