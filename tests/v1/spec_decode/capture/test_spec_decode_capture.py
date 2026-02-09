# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for main capture orchestrator."""

import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.capture.spec_decode_capture import SpecDecodeCapture


@pytest.fixture
def mock_spec_config():
    """Create a mock SpeculativeConfig for testing."""
    config = Mock(spec=SpeculativeConfig)
    config.capture_enabled = True
    config.capture_dir = tempfile.mkdtemp()
    config.capture_percentile = 10.0
    config.capture_top_k = 50
    config.capture_window_size = 1000
    return config


@pytest.fixture
def mock_spec_config_disabled():
    """Create a disabled mock SpeculativeConfig."""
    config = Mock(spec=SpeculativeConfig)
    config.capture_enabled = False
    config.capture_dir = tempfile.mkdtemp()
    config.capture_percentile = 10.0
    config.capture_top_k = 50
    config.capture_window_size = 1000
    return config


class TestSpecDecodeCaptureInitialization:
    """Test SpecDecodeCapture initialization."""

    def test_initialization_with_config(self, mock_spec_config):
        """Test capture initializes correctly with config."""
        capture = SpecDecodeCapture(mock_spec_config)

        assert capture.config is not None
        assert capture.rate_limiter is not None
        assert capture.percentile_tracker is not None
        assert capture.transfer_handler is not None
        assert capture.use_percentile is True

    def test_initialization_creates_components(self, mock_spec_config):
        """Test that all components are created."""
        capture = SpecDecodeCapture(mock_spec_config)

        # Check components exist
        assert hasattr(capture, "config")
        assert hasattr(capture, "rate_limiter")
        assert hasattr(capture, "percentile_tracker")
        assert hasattr(capture, "transfer_handler")

        # Check percentile tracker has correct config
        assert capture.percentile_tracker.percentile == 10.0
        assert capture.percentile_tracker.window_size == 1000

    def test_initialization_auto_detects_tp_values(self, mock_spec_config):
        """Test capture auto-detects TP values (defaults to 1/0 when not in distributed mode)."""
        capture = SpecDecodeCapture(mock_spec_config)

        # When not in distributed mode, defaults to tp_size=1, tp_rank=0
        assert capture.tp_size == 1
        assert capture.tp_rank == 0
        assert capture.transfer_handler.tp_size == 1
        assert capture.transfer_handler.tp_rank == 0


class TestSpecDecodeCaptureCapture:
    """Test SpecDecodeCapture.maybe_capture method."""

    def test_early_exit_when_disabled(self, mock_spec_config_disabled):
        """Test that maybe_capture returns early when disabled."""
        capture = SpecDecodeCapture(mock_spec_config_disabled)

        # Create mock inputs with matching shapes
        logits = torch.randn(1, 3, 1000)
        spec_metadata = Mock()
        spec_metadata.target_logits_indices = [0, 1, 2]
        output_token_ids = torch.tensor([1, 2, 3, 4])

        # Mock transfer_handler to verify it's not called
        capture.transfer_handler.transfer_and_write = Mock()

        # Call maybe_capture
        capture.maybe_capture(logits, spec_metadata, output_token_ids, "test_model")

        # Verify transfer was not initiated
        capture.transfer_handler.transfer_and_write.assert_not_called()

    def test_calculates_acceptance_stats(self, mock_spec_config):
        """Test that acceptance stats are calculated correctly."""
        capture = SpecDecodeCapture(mock_spec_config)

        # Create mock inputs with matching shapes
        logits = torch.randn(1, 3, 1000)
        spec_metadata = Mock()
        spec_metadata.target_logits_indices = [0, 1, 2]  # 3 draft tokens
        output_token_ids = torch.tensor([1, 2, 3, 4])  # 4 tokens, so 3 accepted

        # Mock percentile tracker to always return False (don't capture)
        capture.percentile_tracker.observe_and_check_capture = Mock(return_value=False)

        # Call maybe_capture
        capture.maybe_capture(logits, spec_metadata, output_token_ids, "test_model")

        # Verify observe_and_check_capture was called with correct acceptance length
        # acceptance_length = 1.0 + (3 / 3) = 2.0
        capture.percentile_tracker.observe_and_check_capture.assert_called_once()
        call_args = capture.percentile_tracker.observe_and_check_capture.call_args[0]
        assert call_args[0] == pytest.approx(2.0)

    def test_respects_percentile_tracker_decision(self, mock_spec_config):
        """Test that capture respects percentile tracker decision."""
        capture = SpecDecodeCapture(mock_spec_config)

        logits = torch.randn(1, 3, 1000)
        spec_metadata = Mock()
        spec_metadata.target_logits_indices = [0, 1, 2]
        output_token_ids = torch.tensor([1, 2, 3, 4])

        # Mock percentile tracker to return False (don't capture)
        capture.percentile_tracker.observe_and_check_capture = Mock(return_value=False)
        capture.transfer_handler.transfer_and_write = Mock()

        capture.maybe_capture(logits, spec_metadata, output_token_ids, "test_model")

        # Verify transfer was not initiated
        capture.transfer_handler.transfer_and_write.assert_not_called()

    def test_respects_rate_limiter_decision(self, mock_spec_config):
        """Test that capture respects rate limiter decision."""
        capture = SpecDecodeCapture(mock_spec_config)

        logits = torch.randn(1, 3, 1000)
        spec_metadata = Mock()
        spec_metadata.target_logits_indices = [0, 1, 2]
        output_token_ids = torch.tensor([1, 2, 3, 4])

        # Mock percentile tracker to return True (should capture)
        capture.percentile_tracker.observe_and_check_capture = Mock(return_value=True)
        # Mock rate limiter to return False (rate limit exceeded)
        capture.rate_limiter.should_capture = Mock(return_value=False)
        capture.transfer_handler.transfer_and_write = Mock()

        capture.maybe_capture(logits, spec_metadata, output_token_ids, "test_model")

        # Verify transfer was not initiated
        capture.transfer_handler.transfer_and_write.assert_not_called()

    def test_initiates_transfer_when_approved(self, mock_spec_config):
        """Test that transfer is initiated when all checks pass."""
        capture = SpecDecodeCapture(mock_spec_config)

        # Use batch_size=1, seq_len=3 to match output_token_ids
        logits = torch.randn(1, 3, 1000)
        spec_metadata = Mock()
        spec_metadata.target_logits_indices = [0, 1, 2]
        # 4 tokens = 3 accepted + 1 bonus, need at least batch_size * seq_len = 3 tokens
        output_token_ids = torch.tensor([1, 2, 3, 4])

        # Mock all checks to pass
        capture.percentile_tracker.observe_and_check_capture = Mock(return_value=True)
        capture.rate_limiter.should_capture = Mock(return_value=True)
        capture.rate_limiter.record_captured = Mock()
        capture.transfer_handler.transfer_and_write = Mock()

        capture.maybe_capture(logits, spec_metadata, output_token_ids, "test_model")

        # Verify transfer was initiated
        capture.transfer_handler.transfer_and_write.assert_called_once()
        capture.rate_limiter.record_captured.assert_called_once()

    def test_handles_exceptions_gracefully(self, mock_spec_config):
        """Test that exceptions don't propagate."""
        capture = SpecDecodeCapture(mock_spec_config)

        logits = torch.randn(1, 3, 1000)
        spec_metadata = Mock()
        spec_metadata.target_logits_indices = [0, 1, 2]
        output_token_ids = torch.tensor([1, 2, 3, 4])

        # Mock percentile tracker to raise exception
        capture.percentile_tracker.observe_and_check_capture = Mock(
            side_effect=RuntimeError("Test error")
        )

        # Should not raise exception
        capture.maybe_capture(logits, spec_metadata, output_token_ids, "test_model")


class TestSpecDecodeCaptureGetStats:
    """Test SpecDecodeCapture.get_stats method."""

    def test_get_stats_returns_dict(self, mock_spec_config):
        """Test that get_stats returns a dictionary."""
        capture = SpecDecodeCapture(mock_spec_config)
        stats = capture.get_stats()

        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert "use_percentile" in stats
        assert "percentile_tracker" in stats
        assert "rate_limiter" in stats
        assert "writer" in stats

    def test_get_stats_enabled_status(self, mock_spec_config):
        """Test that get_stats reports correct enabled status."""
        capture = SpecDecodeCapture(mock_spec_config)
        stats = capture.get_stats()

        assert stats["enabled"] is True
        assert stats["use_percentile"] is True

    def test_get_stats_disabled_status(self, mock_spec_config_disabled):
        """Test that get_stats reports correct disabled status."""
        capture = SpecDecodeCapture(mock_spec_config_disabled)
        stats = capture.get_stats()

        assert stats["enabled"] is False


class TestSpecDecodeCaptureCalculateAcceptanceLength:
    """Test SpecDecodeCapture._calculate_acceptance_length method."""

    def test_calculate_acceptance_length_normal(self, mock_spec_config):
        """Test acceptance length calculation with normal values."""
        capture = SpecDecodeCapture(mock_spec_config)

        # 3 accepted out of 5 drafts: 1.0 + 3/5 = 1.6
        result = capture._calculate_acceptance_length(
            {"num_accepted_tokens": 3, "num_draft_tokens": 5}
        )
        assert result == pytest.approx(1.6)

    def test_calculate_acceptance_length_all_accepted(self, mock_spec_config):
        """Test acceptance length when all drafts accepted."""
        capture = SpecDecodeCapture(mock_spec_config)

        # 5 accepted out of 5 drafts: 1.0 + 5/5 = 2.0
        result = capture._calculate_acceptance_length(
            {"num_accepted_tokens": 5, "num_draft_tokens": 5}
        )
        assert result == pytest.approx(2.0)

    def test_calculate_acceptance_length_none_accepted(self, mock_spec_config):
        """Test acceptance length when no drafts accepted."""
        capture = SpecDecodeCapture(mock_spec_config)

        # 0 accepted out of 5 drafts: 1.0 + 0/5 = 1.0
        result = capture._calculate_acceptance_length(
            {"num_accepted_tokens": 0, "num_draft_tokens": 5}
        )
        assert result == pytest.approx(1.0)

    def test_calculate_acceptance_length_zero_drafts(self, mock_spec_config):
        """Test acceptance length with zero drafts."""
        capture = SpecDecodeCapture(mock_spec_config)

        # 0 drafts: should return 1.0
        result = capture._calculate_acceptance_length(
            {"num_accepted_tokens": 0, "num_draft_tokens": 0}
        )
        assert result == pytest.approx(1.0)


class TestSpecDecodeCaptureExtractTopK:
    """Test SpecDecodeCapture._extract_top_k method."""

    def test_extract_top_k_shape(self, mock_spec_config):
        """Test that extracted top-k has correct shape."""
        capture = SpecDecodeCapture(mock_spec_config)

        batch_size, seq_len, vocab_size = 2, 5, 1000
        k = 50
        logits = torch.randn(batch_size, seq_len, vocab_size)

        top_k_probs, top_k_indices = capture._extract_top_k(logits, k)

        assert top_k_probs.shape == (batch_size, seq_len, k)
        assert top_k_indices.shape == (batch_size, seq_len, k)

    def test_extract_top_k_probabilities_sum_to_one(self, mock_spec_config):
        """Test that extracted probabilities sum to approximately 1."""
        capture = SpecDecodeCapture(mock_spec_config)

        logits = torch.randn(1, 1, 100)
        k = 100  # Extract all

        top_k_probs, _ = capture._extract_top_k(logits, k)

        # Probabilities should sum to 1
        assert torch.allclose(top_k_probs.sum(dim=-1), torch.ones(1, 1), atol=1e-5)

    def test_extract_top_k_indices_are_correct(self, mock_spec_config):
        """Test that extracted indices correspond to highest logits."""
        capture = SpecDecodeCapture(mock_spec_config)

        # Create logits with known top values
        logits = torch.tensor([[[1.0, 5.0, 3.0, 2.0, 4.0]]])
        k = 3

        _, top_k_indices = capture._extract_top_k(logits, k)

        # Top 3 should be indices [1, 4, 2] (values 5.0, 4.0, 3.0)
        expected_indices = torch.tensor([[[1, 4, 2]]])
        assert torch.equal(top_k_indices, expected_indices)

    def test_extract_top_k_probabilities_are_sorted(self, mock_spec_config):
        """Test that extracted probabilities are in descending order."""
        capture = SpecDecodeCapture(mock_spec_config)

        logits = torch.randn(1, 1, 100)
        k = 10

        top_k_probs, _ = capture._extract_top_k(logits, k)

        # Check that probabilities are sorted in descending order
        sorted_probs, _ = torch.sort(top_k_probs, dim=-1, descending=True)
        assert torch.allclose(top_k_probs, sorted_probs)


class TestSpecDecodeCaptureShapeHandling:
    """Test SpecDecodeCapture handles different logits shapes."""

    def test_maybe_capture_2d_logits(self, mock_spec_config):
        """Test that maybe_capture handles 2D logits correctly."""
        mock_spec_config.capture_enabled = True
        mock_spec_config.capture_percentile = 100.0
        capture = SpecDecodeCapture(mock_spec_config)
        
        batch_size = 2
        vocab_size = 1000
        logits_2d = torch.randn(batch_size, vocab_size)
        
        spec_metadata = Mock()
        spec_metadata.target_logits_indices = [0, 1]
        # For 2D logits, seq_len=1, so we need batch_size * 1 = 2 tokens
        output_token_ids = torch.tensor([1, 2, 3])  # 2 accepted + 1 bonus
        
        capture.transfer_handler.transfer_and_write = Mock()
        capture.percentile_tracker.observe_and_check_capture = Mock(return_value=True)
        capture.rate_limiter.should_capture = Mock(return_value=True)
        capture.rate_limiter.record_captured = Mock()
        
        capture.maybe_capture(logits_2d, spec_metadata, output_token_ids, "test_model")

        assert capture.transfer_handler.transfer_and_write.called

    def test_maybe_capture_3d_logits(self, mock_spec_config):
        """Test that maybe_capture handles 3D logits correctly."""
        mock_spec_config.capture_enabled = True
        mock_spec_config.capture_percentile = 100.0
        capture = SpecDecodeCapture(mock_spec_config)
        
        batch_size = 2
        seq_len = 4
        vocab_size = 1000
        logits_3d = torch.randn(batch_size, seq_len, vocab_size)
        
        spec_metadata = Mock()
        spec_metadata.target_logits_indices = [0, 1, 2]
        # Need batch_size * seq_len = 8 tokens for reshape
        output_token_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])  # 8 accepted + 1 bonus
        
        capture.transfer_handler.transfer_and_write = Mock()
        capture.percentile_tracker.observe_and_check_capture = Mock(return_value=True)
        capture.rate_limiter.should_capture = Mock(return_value=True)
        capture.rate_limiter.record_captured = Mock()
        
        capture.maybe_capture(logits_3d, spec_metadata, output_token_ids, "test_model")

        assert capture.transfer_handler.transfer_and_write.called
