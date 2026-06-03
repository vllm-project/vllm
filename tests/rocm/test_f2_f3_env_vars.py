# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for PR1: registration of F2/F3 ROCm aiter env vars.

Env vars under test:
  VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT  (F2 gate)
  VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE (F3 gate)

These tests do NOT require a GPU and run on any platform.
"""

import pytest

import vllm.envs as envs
from vllm.envs import environment_variables

# ---------------------------------------------------------------------------
# F2 env var: VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT
# ---------------------------------------------------------------------------

F2_VAR = "VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT"
F3_VAR = "VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE"


class TestF2EnvVar:
    """Tests for VLLM_ROCM_USE_AITER_TRITON_FUSED_RMSNORM_FP4_QUANT."""

    def test_registered_in_environment_variables(self):
        """Env var must appear in the environment_variables registry."""
        assert F2_VAR in environment_variables, (
            f"{F2_VAR} not found in environment_variables; was it added to envs.py?"
        )

    def test_default_is_false(self, monkeypatch: pytest.MonkeyPatch):
        """Without the env var set the default must be False."""
        monkeypatch.delenv(F2_VAR, raising=False)
        assert getattr(envs, F2_VAR) is False

    @pytest.mark.parametrize("truthy_value", ["1", "true", "True", "TRUE"])
    def test_truthy_values_enable_feature(
        self, monkeypatch: pytest.MonkeyPatch, truthy_value: str
    ):
        """Setting the env var to a truthy string must yield True."""
        monkeypatch.setenv(F2_VAR, truthy_value)
        assert getattr(envs, F2_VAR) is True

    @pytest.mark.parametrize("falsy_value", ["0", "false", "False", "FALSE", ""])
    def test_falsy_values_keep_feature_disabled(
        self, monkeypatch: pytest.MonkeyPatch, falsy_value: str
    ):
        """Setting the env var to a falsy string must yield False."""
        monkeypatch.setenv(F2_VAR, falsy_value)
        assert getattr(envs, F2_VAR) is False

    def test_not_a_compile_factor(self):
        """F2 env var must NOT influence torch.compile cache keys."""
        compile_factors = envs.compile_factors()
        assert F2_VAR not in compile_factors, (
            f"{F2_VAR} should not be a compile factor; "
            "adding it would invalidate the cuda-graph cache unnecessarily."
        )


# ---------------------------------------------------------------------------
# F3 env var: VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE
# ---------------------------------------------------------------------------


class TestF3EnvVar:
    """Tests for VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE."""

    def test_registered_in_environment_variables(self):
        """Env var must appear in the environment_variables registry."""
        assert F3_VAR in environment_variables, (
            f"{F3_VAR} not found in environment_variables; was it added to envs.py?"
        )

    def test_default_is_false(self, monkeypatch: pytest.MonkeyPatch):
        """Without the env var set the default must be False."""
        monkeypatch.delenv(F3_VAR, raising=False)
        assert getattr(envs, F3_VAR) is False

    @pytest.mark.parametrize("truthy_value", ["1", "true", "True", "TRUE"])
    def test_truthy_values_enable_feature(
        self, monkeypatch: pytest.MonkeyPatch, truthy_value: str
    ):
        """Setting the env var to a truthy string must yield True."""
        monkeypatch.setenv(F3_VAR, truthy_value)
        assert getattr(envs, F3_VAR) is True

    @pytest.mark.parametrize("falsy_value", ["0", "false", "False", "FALSE", ""])
    def test_falsy_values_keep_feature_disabled(
        self, monkeypatch: pytest.MonkeyPatch, falsy_value: str
    ):
        """Setting the env var to a falsy string must yield False."""
        monkeypatch.setenv(F3_VAR, falsy_value)
        assert getattr(envs, F3_VAR) is False

    def test_not_a_compile_factor(self):
        """F3 env var must NOT influence torch.compile cache keys."""
        compile_factors = envs.compile_factors()
        assert F3_VAR not in compile_factors, (
            f"{F3_VAR} should not be a compile factor; "
            "it controls runtime dispatch only."
        )

    def test_independent_of_f2_var(self, monkeypatch: pytest.MonkeyPatch):
        """F3 and F2 env vars are independent; setting one must not affect the other."""
        monkeypatch.setenv(F3_VAR, "1")
        monkeypatch.delenv(F2_VAR, raising=False)
        assert getattr(envs, F3_VAR) is True
        assert getattr(envs, F2_VAR) is False


# ---------------------------------------------------------------------------
# TC-1.7  Both vars False when explicitly set to "0"
# ---------------------------------------------------------------------------


def test_tc1_7_both_false_when_set_to_zero(monkeypatch: pytest.MonkeyPatch):
    """TC-1.7: Both F2 and F3 must read False when set to '0'."""
    monkeypatch.setenv(F2_VAR, "0")
    monkeypatch.setenv(F3_VAR, "0")
    assert getattr(envs, F2_VAR) is False, f"{F2_VAR}='0' should be False"
    assert getattr(envs, F3_VAR) is False, f"{F3_VAR}='0' should be False"


def test_tc1_7_can_disable_after_enabling(monkeypatch: pytest.MonkeyPatch):
    """TC-1.7: Setting var back to '0' after '1' must disable the feature."""
    monkeypatch.setenv(F2_VAR, "1")
    monkeypatch.setenv(F3_VAR, "1")
    assert getattr(envs, F2_VAR) is True
    assert getattr(envs, F3_VAR) is True

    monkeypatch.setenv(F2_VAR, "0")
    monkeypatch.setenv(F3_VAR, "0")
    assert getattr(envs, F2_VAR) is False, "F2 should be False after setting to '0'"
    assert getattr(envs, F3_VAR) is False, "F3 should be False after setting to '0'"
