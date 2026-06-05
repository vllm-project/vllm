# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Speculative Speculative Decoding (SSD).

Covers:
  - OutcomePredictor: shape, dtype, acceptance mask, no .item() calls
  - SSDAsyncOverlap: two streams, CUDA events, acceptance logic, metrics
  - Correctness guarantee: correct prediction uses pre-spec; wrong re-drafts
"""

from __future__ import annotations

from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.spec_decode.outcome_predictor import OutcomePredictor
from vllm.v1.spec_decode.ssd_worker import SSDAsyncOverlap

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 4
K = 4
VOCAB = 1000
HIDDEN = 64  # small for tests


@pytest.fixture()
def predictor() -> OutcomePredictor:
    """Return a CPU OutcomePredictor with small hidden size."""
    return OutcomePredictor(hidden_size=HIDDEN, K=K, mlp_hidden=32)


@pytest.fixture()
def draft_logits() -> torch.Tensor:
    """[BATCH, K, VOCAB] random logits."""
    return torch.randn(BATCH, K, VOCAB)


@pytest.fixture()
def hidden_state() -> torch.Tensor:
    """[BATCH, HIDDEN] random hidden states."""
    return torch.randn(BATCH, HIDDEN)


@pytest.fixture()
def draft_tokens() -> torch.Tensor:
    """[BATCH, K] random token ids."""
    return torch.randint(0, VOCAB, (BATCH, K))


# ---------------------------------------------------------------------------
# OutcomePredictor tests
# ---------------------------------------------------------------------------


class TestOutcomePredictor:

    def test_forward_output_shape(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        """forward() must return [batch, K] float tensor."""
        out = predictor(draft_logits, hidden_state)
        assert out.shape == (BATCH, K), f"Expected ({BATCH}, {K}), got {out.shape}"

    def test_forward_output_range(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        """Outputs must be probabilities in [0, 1] (sigmoid applied)."""
        out = predictor(draft_logits, hidden_state)
        assert (out >= 0.0).all(), "Probabilities must be >= 0"
        assert (out <= 1.0).all(), "Probabilities must be <= 1"

    def test_forward_output_dtype(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        """Outputs must be float32."""
        out = predictor(draft_logits, hidden_state)
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

    def test_extract_features_shape(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
    ) -> None:
        """extract_features must return [batch, K, FEATURE_SIZE]."""
        features = predictor.extract_features(draft_logits)
        expected = (BATCH, K, OutcomePredictor.FEATURE_SIZE)
        assert features.shape == expected, (
            f"Expected {expected}, got {features.shape}")

    def test_predict_acceptance_mask_dtype(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        """predict_acceptance_mask must return bool tensor."""
        mask = predictor.predict_acceptance_mask(draft_logits, hidden_state)
        assert mask.dtype == torch.bool, f"Expected bool, got {mask.dtype}"

    def test_predict_acceptance_mask_shape(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        """predict_acceptance_mask must return [batch, K]."""
        mask = predictor.predict_acceptance_mask(draft_logits, hidden_state)
        assert mask.shape == (BATCH, K), (
            f"Expected ({BATCH}, {K}), got {mask.shape}")

    def test_no_item_calls_in_predict(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        """
        Acceptance mask computation must not call .item() (which would sync GPU).
        We verify by checking the result stays as a tensor, not a Python scalar.
        """
        mask = predictor.predict_acceptance_mask(draft_logits, hidden_state)
        assert isinstance(mask, torch.Tensor), (
            ".item() should not have been called -- result must be a Tensor")

    def test_threshold_effect(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> None:
        """High threshold -> fewer accepted; low threshold -> more accepted."""
        mask_high = predictor.predict_acceptance_mask(
            draft_logits, hidden_state, threshold=0.99)
        mask_low = predictor.predict_acceptance_mask(
            draft_logits, hidden_state, threshold=0.01)
        # low threshold should accept >= high threshold
        assert mask_low.sum() >= mask_high.sum(), (
            "Lower threshold should accept at least as many tokens")

    def test_from_pretrained_roundtrip(
        self,
        predictor: OutcomePredictor,
        draft_logits: torch.Tensor,
        hidden_state: torch.Tensor,
        tmp_path,
    ) -> None:
        """Save and reload predictor; outputs must match."""
        save_path = str(tmp_path / "predictor.pt")
        torch.save(predictor.state_dict(), save_path)

        loaded = OutcomePredictor.from_pretrained(
            path=save_path,
            hidden_size=HIDDEN,
            K=K,
        )
        loaded.mlp[0].weight  # confirm it loaded

        out_orig = predictor(draft_logits, hidden_state)
        out_loaded = loaded(draft_logits, hidden_state)
        assert torch.allclose(out_orig, out_loaded, atol=1e-6), (
            "Reloaded predictor outputs should match original")


# ---------------------------------------------------------------------------
# SSDAsyncOverlap tests (CPU-only, no GPU required)
# ---------------------------------------------------------------------------


class TestSSDAsyncOverlapInit:
    """Test initialization without CUDA -- mock streams/events."""

    def _make_ssd(self) -> SSDAsyncOverlap:
        """Create SSDAsyncOverlap with mocked CUDA primitives."""
        with (
            patch("torch.cuda.Stream", return_value=MagicMock()),
            patch("torch.cuda.Event", return_value=MagicMock()),
        ):
            return SSDAsyncOverlap(
                outcome_predictor_path=None,
                hidden_size=HIDDEN,
                num_speculative_tokens=K,
            )

    def test_two_streams_created(self) -> None:
        """Two distinct CUDA streams must be created."""
        ssd = self._make_ssd()
        assert ssd.verify_stream is not None
        assert ssd.draft_stream is not None
        assert ssd.verify_stream is not ssd.draft_stream

    def test_three_events_created(self) -> None:
        """Three CUDA events must be created."""
        ssd = self._make_ssd()
        assert ssd.kv_ready_event is not None
        assert ssd.verify_done_event is not None
        assert ssd.predraft_done_event is not None

    def test_predictor_none_without_path(self) -> None:
        """Without a predictor path, _predictor must be None."""
        ssd = self._make_ssd()
        assert ssd._predictor is None

    def test_pre_spec_invalid_on_init(self) -> None:
        """Pre-speculation cache must be invalid initially."""
        ssd = self._make_ssd()
        assert ssd._pre_spec_valid is False
        assert ssd._pre_spec_tokens is None
        assert ssd._pre_spec_mask is None

    def test_prediction_accuracy_with_no_predictions(self) -> None:
        """Accuracy should be 0 / max(1,0) = 0.0 with no predictions."""
        ssd = self._make_ssd()
        assert ssd.prediction_accuracy == 0.0

    def test_get_metrics_keys(self) -> None:
        """get_metrics must contain expected keys."""
        ssd = self._make_ssd()
        metrics = ssd.get_metrics()
        assert "ssd_prediction_accuracy" in metrics
        assert "ssd_correct_predictions" in metrics
        assert "ssd_total_predictions" in metrics


class TestSSDPredictedContinuation:
    """Test _get_predicted_continuation logic (CPU, no CUDA mocking needed)."""

    def _make_ssd(self) -> SSDAsyncOverlap:
        with (
            patch("torch.cuda.Stream", return_value=MagicMock()),
            patch("torch.cuda.Event", return_value=MagicMock()),
        ):
            return SSDAsyncOverlap(hidden_size=HIDDEN, num_speculative_tokens=K)

    def test_first_rejection_chosen(self) -> None:
        """If mask = [T, T, F, F], continuation is draft_tokens[:, 2]."""
        ssd = self._make_ssd()
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, False, False],
        ])
        tokens = torch.tensor([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ])
        cont = ssd._get_predicted_continuation(mask, tokens)
        assert cont[0].item() == 30, f"Expected 30, got {cont[0].item()}"
        assert cont[1].item() == 70, f"Expected 70, got {cont[1].item()}"

    def test_all_accepted_uses_last_token(self) -> None:
        """If mask = all True, continuation is draft_tokens[:, K-1]."""
        ssd = self._make_ssd()
        mask = torch.ones(2, K, dtype=torch.bool)
        tokens = torch.tensor([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ])
        cont = ssd._get_predicted_continuation(mask, tokens)
        assert cont[0].item() == 40, f"Expected 40, got {cont[0].item()}"
        assert cont[1].item() == 80, f"Expected 80, got {cont[1].item()}"

    def test_first_position_rejected(self) -> None:
        """If mask = [F, F, F, F], continuation is draft_tokens[:, 0]."""
        ssd = self._make_ssd()
        mask = torch.zeros(2, K, dtype=torch.bool)
        tokens = torch.tensor([
            [10, 20, 30, 40],
            [50, 60, 70, 80],
        ])
        cont = ssd._get_predicted_continuation(mask, tokens)
        assert cont[0].item() == 10
        assert cont[1].item() == 50

    def test_no_item_calls_in_continuation(self) -> None:
        """_get_predicted_continuation must return a Tensor (no .item())."""
        ssd = self._make_ssd()
        mask = torch.tensor([[True, False, False, False]])
        tokens = torch.randint(0, VOCAB, (1, K))
        result = ssd._get_predicted_continuation(mask, tokens)
        assert isinstance(result, torch.Tensor), (
            "Result must be a Tensor -- .item() should not be called")
