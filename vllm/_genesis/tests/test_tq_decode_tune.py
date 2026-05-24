# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.tq_decode_tune (Patch 18b).

Validates env-driven tune for TurboQuant decode stage1 kernel with whitelisted
values, safe fallbacks, and platform guard.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations



class TestBlockKvOverride:
    """Group 1: BLOCK_KV env override."""

    def test_unset_returns_none(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_block_kv_override
        monkeypatch.delenv("VLLM_TQ_DECODE_BLOCK_KV", raising=False)
        assert get_block_kv_override() is None

    def test_valid_values_parsed(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_block_kv_override
        for v in [1, 2, 4, 8, 16, 32, 64]:
            monkeypatch.setenv("VLLM_TQ_DECODE_BLOCK_KV", str(v))
            assert get_block_kv_override() == v

    def test_invalid_values_rejected(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_block_kv_override
        for v in ["0", "3", "5", "128", "256", "abc", "-1", "", "  "]:
            monkeypatch.setenv("VLLM_TQ_DECODE_BLOCK_KV", v)
            assert get_block_kv_override() is None, f"value {v!r} should reject"

    def test_leading_trailing_whitespace_stripped(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_block_kv_override
        monkeypatch.setenv("VLLM_TQ_DECODE_BLOCK_KV", "  16  ")
        assert get_block_kv_override() == 16


class TestNumWarpsOverride:
    """Group 2: num_warps env override."""

    def test_unset_returns_none(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_num_warps_override
        monkeypatch.delenv("VLLM_TQ_DECODE_NUM_WARPS", raising=False)
        assert get_num_warps_override() is None

    def test_valid_values(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_num_warps_override
        for v in [1, 2, 4, 8]:
            monkeypatch.setenv("VLLM_TQ_DECODE_NUM_WARPS", str(v))
            assert get_num_warps_override() == v

    def test_invalid_values_rejected(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_num_warps_override
        for v in ["0", "3", "5", "16", "abc", "-2"]:
            monkeypatch.setenv("VLLM_TQ_DECODE_NUM_WARPS", v)
            assert get_num_warps_override() is None, f"value {v!r} should reject"


class TestNumStagesOverride:
    """Group 3: num_stages env override."""

    def test_unset_returns_none(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_num_stages_override
        monkeypatch.delenv("VLLM_TQ_DECODE_NUM_STAGES", raising=False)
        assert get_num_stages_override() is None

    def test_valid_range_1_to_8(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_num_stages_override
        for v in range(1, 9):
            monkeypatch.setenv("VLLM_TQ_DECODE_NUM_STAGES", str(v))
            assert get_num_stages_override() == v

    def test_out_of_range_rejected(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_num_stages_override
        for v in ["0", "9", "10", "100"]:
            monkeypatch.setenv("VLLM_TQ_DECODE_NUM_STAGES", v)
            assert get_num_stages_override() is None, f"value {v!r} should reject"

    def test_non_digit_rejected(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import get_num_stages_override
        for v in ["abc", "-1", "1.5", "", "   "]:
            monkeypatch.setenv("VLLM_TQ_DECODE_NUM_STAGES", v)
            assert get_num_stages_override() is None


class TestResolveDecodeTune:
    """Group 4: Full tuple resolution with fallback to upstream defaults."""

    def test_all_unset_returns_upstream_defaults(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import (
            resolve_decode_tune,
            UPSTREAM_BLOCK_KV, UPSTREAM_NUM_WARPS, UPSTREAM_NUM_STAGES,
        )
        monkeypatch.delenv("VLLM_TQ_DECODE_BLOCK_KV", raising=False)
        monkeypatch.delenv("VLLM_TQ_DECODE_NUM_WARPS", raising=False)
        monkeypatch.delenv("VLLM_TQ_DECODE_NUM_STAGES", raising=False)

        bkv, nw, ns = resolve_decode_tune()
        assert (bkv, nw, ns) == (UPSTREAM_BLOCK_KV, UPSTREAM_NUM_WARPS, UPSTREAM_NUM_STAGES)

    def test_all_set_valid(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import resolve_decode_tune
        monkeypatch.setenv("VLLM_TQ_DECODE_BLOCK_KV", "16")
        monkeypatch.setenv("VLLM_TQ_DECODE_NUM_WARPS", "4")
        monkeypatch.setenv("VLLM_TQ_DECODE_NUM_STAGES", "2")
        assert resolve_decode_tune() == (16, 4, 2)

    def test_partial_override(self, monkeypatch):
        """Only BLOCK_KV set — others keep upstream."""
        from vllm._genesis.kernels.tq_decode_tune import (
            resolve_decode_tune, UPSTREAM_NUM_WARPS, UPSTREAM_NUM_STAGES,
        )
        monkeypatch.setenv("VLLM_TQ_DECODE_BLOCK_KV", "8")
        monkeypatch.delenv("VLLM_TQ_DECODE_NUM_WARPS", raising=False)
        monkeypatch.delenv("VLLM_TQ_DECODE_NUM_STAGES", raising=False)

        bkv, nw, ns = resolve_decode_tune()
        assert bkv == 8
        assert nw == UPSTREAM_NUM_WARPS
        assert ns == UPSTREAM_NUM_STAGES

    def test_invalid_fallback_to_upstream(self, monkeypatch):
        """Bad env values must NOT corrupt the tuple — fall back to upstream."""
        from vllm._genesis.kernels.tq_decode_tune import (
            resolve_decode_tune,
            UPSTREAM_BLOCK_KV, UPSTREAM_NUM_WARPS, UPSTREAM_NUM_STAGES,
        )
        monkeypatch.setenv("VLLM_TQ_DECODE_BLOCK_KV", "3")   # not in whitelist
        monkeypatch.setenv("VLLM_TQ_DECODE_NUM_WARPS", "16")  # not in whitelist
        monkeypatch.setenv("VLLM_TQ_DECODE_NUM_STAGES", "0")  # out of range

        assert resolve_decode_tune() == (
            UPSTREAM_BLOCK_KV, UPSTREAM_NUM_WARPS, UPSTREAM_NUM_STAGES,
        )


class TestHasAnyOverride:
    """Group 5: Predicate for 'user opted into tuning'."""

    def test_false_when_all_unset(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import has_any_override
        for k in ("VLLM_TQ_DECODE_BLOCK_KV", "VLLM_TQ_DECODE_NUM_WARPS",
                  "VLLM_TQ_DECODE_NUM_STAGES"):
            monkeypatch.delenv(k, raising=False)
        assert has_any_override() is False

    def test_true_when_any_single_valid(self, monkeypatch):
        from vllm._genesis.kernels.tq_decode_tune import has_any_override
        monkeypatch.delenv("VLLM_TQ_DECODE_BLOCK_KV", raising=False)
        monkeypatch.delenv("VLLM_TQ_DECODE_NUM_WARPS", raising=False)
        monkeypatch.setenv("VLLM_TQ_DECODE_NUM_STAGES", "4")
        assert has_any_override() is True

    def test_false_when_all_invalid(self, monkeypatch):
        """Invalid env values don't count as an override."""
        from vllm._genesis.kernels.tq_decode_tune import has_any_override
        monkeypatch.setenv("VLLM_TQ_DECODE_BLOCK_KV", "999")
        monkeypatch.setenv("VLLM_TQ_DECODE_NUM_WARPS", "16")
        monkeypatch.setenv("VLLM_TQ_DECODE_NUM_STAGES", "0")
        assert has_any_override() is False


class TestShouldApplyPlatformGuard:
    """Group 6: Platform guard — NVIDIA CUDA + SM 8.0+ only."""

    def test_false_on_non_nvidia(self, monkeypatch):
        from vllm._genesis.kernels import tq_decode_tune as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        assert t.should_apply() is False

    def test_false_on_pre_ampere(self, monkeypatch):
        from vllm._genesis.kernels import tq_decode_tune as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: False)
        assert t.should_apply() is False

    def test_true_on_ampere_plus(self, monkeypatch):
        from vllm._genesis.kernels import tq_decode_tune as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        assert t.should_apply() is True


class TestLogSelectedTune:
    """Group 7: Observability — log does not raise."""

    def test_log_does_not_raise_when_skipped(self, monkeypatch, caplog):
        from vllm._genesis.kernels import tq_decode_tune as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: False)
        t.log_selected_tune()  # must not raise

    def test_log_does_not_raise_on_upstream_path(self, monkeypatch, caplog):
        from vllm._genesis.kernels import tq_decode_tune as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        for k in ("VLLM_TQ_DECODE_BLOCK_KV", "VLLM_TQ_DECODE_NUM_WARPS",
                  "VLLM_TQ_DECODE_NUM_STAGES"):
            monkeypatch.delenv(k, raising=False)
        t.log_selected_tune()  # must not raise

    def test_log_does_not_raise_on_override_path(self, monkeypatch, caplog):
        from vllm._genesis.kernels import tq_decode_tune as t
        from vllm._genesis import guards
        monkeypatch.setattr(guards, "is_nvidia_cuda", lambda: True)
        monkeypatch.setattr(guards, "is_sm_at_least",
                            lambda major, minor=0: True)
        monkeypatch.setenv("VLLM_TQ_DECODE_BLOCK_KV", "16")
        t.log_selected_tune()  # must not raise
