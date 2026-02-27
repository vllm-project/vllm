# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for determinism warmup functionality."""

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.batch_invariant as batch_invariant


class TestDeterminismWarmupIterations:
    """Tests for the warmup iteration configuration."""

    def test_default_iterations_when_batch_invariant_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Default should be 3 iterations when VLLM_BATCH_INVARIANT=1."""
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        monkeypatch.delenv("VLLM_DETERMINISM_WARMUP_ITERATIONS", raising=False)

        result = batch_invariant._read_determinism_warmup_iterations()
        assert result == 3

    def test_default_iterations_when_batch_invariant_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Default should be 0 iterations when VLLM_BATCH_INVARIANT=0."""
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        monkeypatch.delenv("VLLM_DETERMINISM_WARMUP_ITERATIONS", raising=False)

        result = batch_invariant._read_determinism_warmup_iterations()
        assert result == 0

    def test_explicit_iterations_override(self, monkeypatch: pytest.MonkeyPatch):
        """Explicit VLLM_DETERMINISM_WARMUP_ITERATIONS should override default."""
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        monkeypatch.setenv("VLLM_DETERMINISM_WARMUP_ITERATIONS", "5")

        result = batch_invariant._read_determinism_warmup_iterations()
        assert result == 5

    def test_zero_iterations_disables_warmup(self, monkeypatch: pytest.MonkeyPatch):
        """Setting iterations to 0 should disable warmup even with batch invariance."""
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        monkeypatch.setenv("VLLM_DETERMINISM_WARMUP_ITERATIONS", "0")

        result = batch_invariant._read_determinism_warmup_iterations()
        assert result == 0

    def test_negative_iterations_returns_zero(self, monkeypatch: pytest.MonkeyPatch):
        """Negative values should be clamped to 0."""
        monkeypatch.setenv("VLLM_DETERMINISM_WARMUP_ITERATIONS", "-5")

        result = batch_invariant._read_determinism_warmup_iterations()
        assert result == 0

    def test_invalid_value_returns_zero(self, monkeypatch: pytest.MonkeyPatch):
        """Invalid (non-integer) values should return 0."""
        monkeypatch.setenv("VLLM_DETERMINISM_WARMUP_ITERATIONS", "invalid")

        result = batch_invariant._read_determinism_warmup_iterations()
        assert result == 0


class TestRunDeterminismWarmup:
    """Tests for the run_determinism_warmup function."""

    def test_warmup_runs_correct_iterations(self, monkeypatch: pytest.MonkeyPatch):
        """Warmup should call dummy_run_fn the specified number of times."""
        monkeypatch.setattr(batch_invariant, "VLLM_BATCH_INVARIANT", True)

        dummy_run = MagicMock()

        result = batch_invariant.run_determinism_warmup(dummy_run, num_iterations=3)

        assert result is True
        assert dummy_run.call_count == 3

    def test_warmup_skipped_when_batch_invariant_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Warmup should be skipped when batch invariance is disabled."""
        monkeypatch.setattr(batch_invariant, "VLLM_BATCH_INVARIANT", False)

        dummy_run = MagicMock()

        result = batch_invariant.run_determinism_warmup(dummy_run, num_iterations=3)

        assert result is False
        dummy_run.assert_not_called()

    def test_warmup_skipped_with_zero_iterations(self, monkeypatch: pytest.MonkeyPatch):
        """Warmup should be skipped when iterations is 0."""
        monkeypatch.setattr(batch_invariant, "VLLM_BATCH_INVARIANT", True)

        dummy_run = MagicMock()

        result = batch_invariant.run_determinism_warmup(dummy_run, num_iterations=0)

        assert result is False
        dummy_run.assert_not_called()

    def test_warmup_continues_on_exception(self, monkeypatch: pytest.MonkeyPatch):
        """Warmup should continue even if an iteration fails."""
        monkeypatch.setattr(batch_invariant, "VLLM_BATCH_INVARIANT", True)

        dummy_run = MagicMock(side_effect=[RuntimeError("test"), None, None])

        result = batch_invariant.run_determinism_warmup(dummy_run, num_iterations=3)

        assert result is True
        assert dummy_run.call_count == 3

    def test_warmup_uses_default_iterations(self, monkeypatch: pytest.MonkeyPatch):
        """Warmup should use VLLM_DETERMINISM_WARMUP_ITERATIONS when not specified."""
        monkeypatch.setattr(batch_invariant, "VLLM_BATCH_INVARIANT", True)
        monkeypatch.setattr(batch_invariant, "VLLM_DETERMINISM_WARMUP_ITERATIONS", 2)

        dummy_run = MagicMock()

        result = batch_invariant.run_determinism_warmup(dummy_run)

        assert result is True
        assert dummy_run.call_count == 2


class TestGetDeterminismWarmupIterations:
    """Tests for the get_determinism_warmup_iterations function."""

    def test_returns_module_constant(self, monkeypatch: pytest.MonkeyPatch):
        """Should return the module-level constant."""
        monkeypatch.setattr(batch_invariant, "VLLM_DETERMINISM_WARMUP_ITERATIONS", 7)

        result = batch_invariant.get_determinism_warmup_iterations()

        assert result == 7
