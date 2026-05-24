# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.marlin_tuning.

Patches 17 + 18 migration: per-SM optimal Marlin kernel parameters
(block_size_m, num_warps, num_stages) with env overrides.

Measured on A5000 (Qwen3.6-35B-A3B-FP8, M≤4, topk=8, E=256):
  bsm=8   → +1.2% (winner)
  bsm=16  → baseline (upstream default)
  bsm=32  → −1.9%
  bsm=48  → −4.6%
  bsm=64  → −7.9%

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations



class TestGetOptimalBlockSizeM:
    """Group 1: Per-SM auto-selection."""

    def test_returns_none_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.kernels import marlin_tuning as mt
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        monkeypatch.delenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", raising=False)

        assert mt.get_optimal_block_size_m() is None

    def test_env_override_takes_precedence(self, monkeypatch):
        """VLLM_MARLIN_MOE_BLOCK_SIZE_M overrides arch table."""
        from vllm._genesis.kernels import marlin_tuning as mt
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 0))
        monkeypatch.setenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", "64")

        assert mt.get_optimal_block_size_m() == 64

    def test_env_override_ignored_if_invalid(self, monkeypatch):
        """Non-whitelisted values silently ignored (falls back to table)."""
        from vllm._genesis.kernels import marlin_tuning as mt
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 6))
        monkeypatch.setenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", "128")  # bad value

        # Falls back to table: SM 8.6 → 8
        assert mt.get_optimal_block_size_m() == 8

    def test_a5000_returns_8(self, monkeypatch):
        """SM 8.6 (A5000) → bsm=8 (measured +1.2%)."""
        from vllm._genesis.kernels import marlin_tuning as mt
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 6))
        monkeypatch.delenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", raising=False)

        assert mt.get_optimal_block_size_m() == 8

    def test_a100_returns_16(self, monkeypatch):
        """SM 8.0 (A100) → bsm=16 (defer to upstream heuristic)."""
        from vllm._genesis.kernels import marlin_tuning as mt
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 0))
        monkeypatch.delenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", raising=False)

        assert mt.get_optimal_block_size_m() == 16

    def test_hopper_returns_16(self, monkeypatch):
        """SM 9.0 (H100) → bsm=16 (upstream heuristic adequate)."""
        from vllm._genesis.kernels import marlin_tuning as mt
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (9, 0))
        monkeypatch.delenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", raising=False)

        assert mt.get_optimal_block_size_m() == 16

    def test_blackwell_returns_16(self, monkeypatch):
        """SM 10.0 (Blackwell) → bsm=16."""
        from vllm._genesis.kernels import marlin_tuning as mt
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (10, 0))
        monkeypatch.delenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", raising=False)

        assert mt.get_optimal_block_size_m() == 16

    def test_unknown_sm_returns_none(self, monkeypatch):
        """Unknown SM version → None (caller falls back to upstream)."""
        from vllm._genesis.kernels import marlin_tuning as mt
        from vllm._genesis import guards

        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (11, 5))
        monkeypatch.delenv("VLLM_MARLIN_MOE_BLOCK_SIZE_M", raising=False)

        assert mt.get_optimal_block_size_m() is None


class TestNumWarpsOverride:
    """Group 2: num_warps env override."""

    def test_returns_none_if_not_set(self, monkeypatch):
        """No env + non-NVIDIA → None (baseline behavior).

        NOTE: on NVIDIA CUDA with SM 8.6 the auto-select table returns 4.
        We force non-NVIDIA to exercise the pure-env branch.
        """
        from vllm._genesis.kernels.marlin_tuning import get_num_warps_override
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_WARPS", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert get_num_warps_override() is None

    def test_valid_values(self, monkeypatch):
        from vllm._genesis.kernels.marlin_tuning import get_num_warps_override

        for v in ["2", "4", "8"]:
            monkeypatch.setenv("VLLM_MARLIN_MOE_NUM_WARPS", v)
            assert get_num_warps_override() == int(v)

    def test_invalid_returns_none(self, monkeypatch):
        """Invalid env → fall through; non-NVIDIA → None."""
        from vllm._genesis.kernels.marlin_tuning import get_num_warps_override
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)

        for v in ["3", "16", "abc", ""]:
            monkeypatch.setenv("VLLM_MARLIN_MOE_NUM_WARPS", v)
            assert get_num_warps_override() is None


class TestNumStagesOverride:
    """Group 3: num_stages env override."""

    def test_returns_none_if_not_set(self, monkeypatch):
        """No env + non-NVIDIA → None (pure env branch).

        On NVIDIA SM 8.6, auto-select returns 3; we force non-NVIDIA here.
        """
        from vllm._genesis.kernels.marlin_tuning import get_num_stages_override
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_STAGES", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert get_num_stages_override() is None

    def test_valid_range_1_to_8(self, monkeypatch):
        from vllm._genesis.kernels.marlin_tuning import get_num_stages_override

        for v in range(1, 9):
            monkeypatch.setenv("VLLM_MARLIN_MOE_NUM_STAGES", str(v))
            assert get_num_stages_override() == v

    def test_out_of_range_returns_none(self, monkeypatch):
        from vllm._genesis.kernels.marlin_tuning import get_num_stages_override
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)

        for v in ["0", "9", "100"]:
            monkeypatch.setenv("VLLM_MARLIN_MOE_NUM_STAGES", v)
            assert get_num_stages_override() is None

    def test_non_digit_returns_none(self, monkeypatch):
        from vllm._genesis.kernels.marlin_tuning import get_num_stages_override
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)

        for v in ["abc", "-1", "1.5", ""]:
            monkeypatch.setenv("VLLM_MARLIN_MOE_NUM_STAGES", v)
            assert get_num_stages_override() is None


class TestPatch24AutoSelect:
    """P24: Per-SM auto-select for num_warps / num_stages."""

    def test_warps_autoselect_on_sm86_without_env(self, monkeypatch):
        """No env, SM 8.6 → returns 4 warps (measured optimal)."""
        from vllm._genesis.kernels import marlin_tuning as t
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_WARPS", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 6))
        assert t.get_num_warps_override() == 4

    def test_stages_autoselect_on_sm86_without_env(self, monkeypatch):
        """No env, SM 8.6 → returns 3 stages (measured optimal)."""
        from vllm._genesis.kernels import marlin_tuning as t
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_STAGES", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 6))
        assert t.get_num_stages_override() == 3

    def test_env_takes_precedence_over_autoselect(self, monkeypatch):
        from vllm._genesis.kernels import marlin_tuning as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 6))

        monkeypatch.setenv("VLLM_MARLIN_MOE_NUM_WARPS", "8")
        assert t.get_num_warps_override() == 8  # env wins

        monkeypatch.setenv("VLLM_MARLIN_MOE_NUM_STAGES", "5")
        assert t.get_num_stages_override() == 5  # env wins

    def test_unknown_sm_defers_to_upstream(self, monkeypatch):
        """SM with no tune entry (e.g. hypothetical 11.0) → None."""
        from vllm._genesis.kernels import marlin_tuning as t
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_WARPS", raising=False)
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_STAGES", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (11, 0))
        assert t.get_num_warps_override() is None
        assert t.get_num_stages_override() is None

    def test_a100_entries_are_none(self, monkeypatch):
        """A100 (SM 8.0): None = defer (no tuning data)."""
        from vllm._genesis.kernels import marlin_tuning as t
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_WARPS", raising=False)
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_STAGES", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "get_compute_capability", lambda: (8, 0))
        assert t.get_num_warps_override() is None
        assert t.get_num_stages_override() is None

    def test_non_nvidia_returns_none(self, monkeypatch):
        from vllm._genesis.kernels import marlin_tuning as t
        from vllm._genesis import guards
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_WARPS", raising=False)
        monkeypatch.delenv("VLLM_MARLIN_MOE_NUM_STAGES", raising=False)
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert t.get_num_warps_override() is None
        assert t.get_num_stages_override() is None


class TestLogSelectedTuning:
    """Group 4: Observability logging."""

    def test_log_selected_tuning_does_not_raise(self, caplog):
        """log_selected_tuning succeeds on any platform."""
        import logging
        from vllm._genesis.kernels.marlin_tuning import log_selected_tuning

        with caplog.at_level(logging.INFO, logger="genesis.marlin_tuning"):
            log_selected_tuning(
                num_experts=256, topk=8, selected_bsm=8
            )

        # Log line should contain our diagnostic info
        text = caplog.text
        assert "E=256" in text or "topk=8" in text or "bsm=8" in text
