# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the centralised Marlin CUDA driver/toolkit mismatch warning.

The guard lives in marlin_utils.warn_marlin_cuda_driver_mismatch() and is
called by every Marlin-based quantization config (awq_marlin, gptq_marlin)
so that the check covers all Marlin usage from a single place.
"""

import vllm.model_executor.layers.quantization.utils.marlin_utils as marlin_utils
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_warn_cache():
    """Clear the lru_cache on warn_marlin_cuda_driver_mismatch between tests."""
    marlin_utils.warn_marlin_cuda_driver_mismatch.cache_clear()


def _patch_mismatch(monkeypatch, *, driver=(12, 8), toolkit=(12, 9), compat=False):
    monkeypatch.setattr(marlin_utils, "_get_driver_cuda_version", lambda: driver)
    monkeypatch.setattr(marlin_utils, "_parse_cuda_version", lambda _: toolkit)
    monkeypatch.setattr(marlin_utils.envs, "VLLM_ENABLE_CUDA_COMPATIBILITY", compat)


# ---------------------------------------------------------------------------
# warn_marlin_cuda_driver_mismatch unit tests
# ---------------------------------------------------------------------------


def test_warning_emitted_on_driver_mismatch(monkeypatch):
    _reset_warn_cache()
    _patch_mismatch(monkeypatch, driver=(12, 8), toolkit=(12, 9), compat=False)

    from unittest.mock import patch

    with patch.object(marlin_utils.logger, "warning") as mock_warn:
        marlin_utils.warn_marlin_cuda_driver_mismatch()

    assert mock_warn.called
    assert any(
        "cudaErrorUnsupportedPtxVersion" in str(a) for a in mock_warn.call_args[0]
    )


def test_no_warning_when_compat_enabled(monkeypatch):
    _reset_warn_cache()
    _patch_mismatch(monkeypatch, driver=(12, 8), toolkit=(12, 9), compat=True)

    from unittest.mock import patch

    with patch.object(marlin_utils.logger, "warning") as mock_warn:
        marlin_utils.warn_marlin_cuda_driver_mismatch()

    assert not mock_warn.called


def test_no_warning_when_driver_matches_toolkit(monkeypatch):
    _reset_warn_cache()
    _patch_mismatch(monkeypatch, driver=(12, 9), toolkit=(12, 9), compat=False)

    from unittest.mock import patch

    with patch.object(marlin_utils.logger, "warning") as mock_warn:
        marlin_utils.warn_marlin_cuda_driver_mismatch()

    assert not mock_warn.called


def test_no_warning_when_driver_newer(monkeypatch):
    _reset_warn_cache()
    _patch_mismatch(monkeypatch, driver=(12, 9), toolkit=(12, 8), compat=False)

    from unittest.mock import patch

    with patch.object(marlin_utils.logger, "warning") as mock_warn:
        marlin_utils.warn_marlin_cuda_driver_mismatch()

    assert not mock_warn.called


# ---------------------------------------------------------------------------
# awq_marlin: marlin is still selected regardless of driver mismatch
# ---------------------------------------------------------------------------


def _force_awq_marlin_compatible(monkeypatch):
    monkeypatch.setattr(
        AWQMarlinConfig,
        "is_awq_marlin_compatible",
        classmethod(lambda cls, _: True),
    )


def test_awq_marlin_selected_despite_driver_mismatch(monkeypatch):
    """awq_marlin should still be returned even when driver < toolkit."""
    _reset_warn_cache()
    _force_awq_marlin_compatible(monkeypatch)
    _patch_mismatch(monkeypatch, driver=(12, 8), toolkit=(12, 9), compat=False)

    result = AWQMarlinConfig.override_quantization_method({}, user_quant=None)
    assert result == "awq_marlin"


def test_awq_marlin_selected_with_compat_enabled(monkeypatch):
    _reset_warn_cache()
    _force_awq_marlin_compatible(monkeypatch)
    _patch_mismatch(monkeypatch, driver=(12, 8), toolkit=(12, 9), compat=True)

    result = AWQMarlinConfig.override_quantization_method({}, user_quant=None)
    assert result == "awq_marlin"


# ---------------------------------------------------------------------------
# gptq_marlin: marlin is still selected regardless of driver mismatch
# ---------------------------------------------------------------------------


def _force_gptq_marlin_compatible(monkeypatch):
    monkeypatch.setattr(
        GPTQMarlinConfig,
        "is_gptq_marlin_compatible",
        classmethod(lambda cls, _: True),
    )


def test_gptq_marlin_selected_despite_driver_mismatch(monkeypatch):
    """gptq_marlin should still be returned even when driver < toolkit."""
    _reset_warn_cache()
    _force_gptq_marlin_compatible(monkeypatch)
    _patch_mismatch(monkeypatch, driver=(12, 8), toolkit=(12, 9), compat=False)

    assert (
        GPTQMarlinConfig.override_quantization_method({}, user_quant=None)
        == "gptq_marlin"
    )


def test_gptq_marlin_selected_with_compat_enabled(monkeypatch):
    _reset_warn_cache()
    _force_gptq_marlin_compatible(monkeypatch)
    _patch_mismatch(monkeypatch, driver=(12, 8), toolkit=(12, 9), compat=True)

    assert (
        GPTQMarlinConfig.override_quantization_method({}, user_quant=None)
        == "gptq_marlin"
    )
