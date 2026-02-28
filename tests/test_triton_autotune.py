# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Triton autotuning control (VLLM_TRITON_AUTOTUNE)."""

import pytest

import vllm.envs as envs
from vllm.triton_utils.autotune import (
    disable_autotune_globally,
    restore_autotune_globally,
    vllm_autotune,
)
from vllm.triton_utils.importing import HAS_TRITON

if HAS_TRITON:
    from vllm.triton_utils import triton


@pytest.fixture(autouse=True)
def _cleanup_global_patch():
    """Ensure global autotune patch is restored after each test."""
    yield
    restore_autotune_globally()


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
class TestVllmAutotune:
    """Tests for the vllm_autotune wrapper."""

    @staticmethod
    def _make_dummy_kernel():
        """Create a minimal Triton JIT kernel for testing the autotune wrapper."""

        @triton.jit
        def _dummy_kernel(
            X,
            M: triton.language.constexpr,
            BLOCK_M: triton.language.constexpr,
        ):
            pass

        return _dummy_kernel

    def test_disabled_uses_first_config(self, monkeypatch: pytest.MonkeyPatch):
        """When VLLM_TRITON_AUTOTUNE=0, only configs[0] should be used."""
        monkeypatch.setattr(envs, "VLLM_TRITON_AUTOTUNE", False)

        configs = [
            triton.Config({"BLOCK_M": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 128}, num_warps=8),
            triton.Config({"BLOCK_M": 256}, num_warps=8),
        ]

        # Apply the decorator to a real Triton kernel to get an Autotuner
        autotuner = vllm_autotune(configs=configs, key=["M"])(self._make_dummy_kernel())
        # Verify the Autotuner was constructed with only the first config
        assert len(autotuner.configs) == 1
        assert autotuner.configs[0] == configs[0]

    def test_enabled_uses_all_configs(self, monkeypatch: pytest.MonkeyPatch):
        """When VLLM_TRITON_AUTOTUNE=1, all configs should be passed."""
        monkeypatch.setattr(envs, "VLLM_TRITON_AUTOTUNE", True)

        configs = [
            triton.Config({"BLOCK_M": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 128}, num_warps=8),
        ]

        autotuner = vllm_autotune(configs=configs, key=["M"])(self._make_dummy_kernel())
        # Verify the Autotuner was constructed with all configs
        assert len(autotuner.configs) == len(configs)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
class TestGlobalAutotunePatch:
    """Tests for disable_autotune_globally / restore_autotune_globally."""

    def test_disable_patches_triton_autotune(self):
        """disable_autotune_globally() should replace triton.autotune."""
        original = triton.autotune
        disable_autotune_globally()

        assert triton.autotune is not original

        # Cleanup
        restore_autotune_globally()

    def test_restore_undoes_patch(self):
        """restore_autotune_globally() should restore original autotune."""
        original = triton.autotune

        disable_autotune_globally()
        assert triton.autotune is not original

        restore_autotune_globally()
        assert triton.autotune is original

    def test_double_disable_is_idempotent(self):
        """Calling disable twice should not stack patches."""
        original = triton.autotune

        disable_autotune_globally()
        patched = triton.autotune
        disable_autotune_globally()  # second call

        assert triton.autotune is patched

        restore_autotune_globally()
        assert triton.autotune is original

    def test_restore_without_disable_is_noop(self):
        """Calling restore without disable should be a no-op."""
        original = triton.autotune
        restore_autotune_globally()
        assert triton.autotune is original


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")
class TestEnvVarIntegration:
    """Tests for env var integration."""

    def test_env_var_default_is_disabled(self):
        """VLLM_TRITON_AUTOTUNE should default to False (disabled)."""
        # The default in envs.py is False
        assert envs.VLLM_TRITON_AUTOTUNE is False
