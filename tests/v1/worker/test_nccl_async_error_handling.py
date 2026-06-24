# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING opt-in env
that controls whether vLLM preserves NCCL_ASYNC_ERROR_HANDLING /
TORCH_NCCL_ASYNC_ERROR_HANDLING (so torch's NCCL watchdog can abort
hung collectives) or pops them (the historical default that avoids
exceptions during CUDA graph capture).

See vllm-project/vllm#45094 for background.
"""

import os
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.v1.worker.gpu_worker import _apply_nccl_async_error_handling_policy


def _clear_envs_cache() -> None:
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot and restore the relevant env vars + envs cache so each
    test starts from a clean slate."""
    saved = {
        k: os.environ.get(k)
        for k in (
            "NCCL_ASYNC_ERROR_HANDLING",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
            "VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING",
        )
    }
    for k in saved:
        os.environ.pop(k, None)
    _clear_envs_cache()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _clear_envs_cache()


def test_default_pops_both_nccl_env_vars():
    """Default (env unset) preserves the historical behavior: both
    NCCL_ASYNC_ERROR_HANDLING and TORCH_NCCL_ASYNC_ERROR_HANDLING are
    popped to avoid CUDA-graph-capture exceptions."""
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    _apply_nccl_async_error_handling_policy()

    assert "NCCL_ASYNC_ERROR_HANDLING" not in os.environ
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ


def test_opt_in_preserves_both_nccl_env_vars():
    """When the user sets VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING=1, vLLM
    must NOT pop the NCCL env vars — torch's watchdog needs them set
    to abort hung collectives (the mitigation for issue #45094)."""
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING"] = "1"
    _clear_envs_cache()

    _apply_nccl_async_error_handling_policy()

    assert os.environ.get("NCCL_ASYNC_ERROR_HANDLING") == "1"
    assert os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") == "1"


def test_opt_in_accepts_true_string():
    """The opt-in env should accept 'True' / 'true' as well as '1'."""
    for truthy in ("True", "true", "1"):
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING"] = truthy
        _clear_envs_cache()

        _apply_nccl_async_error_handling_policy()

        assert os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") == "1", (
            f"opt-in via {truthy!r} should have preserved the env"
        )


def test_opt_in_rejects_unset_and_falsey_values():
    """An unset or explicitly-false env must NOT enable preservation."""
    for falsy in (None, "0", "False", "false", ""):
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        if falsy is None:
            os.environ.pop("VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING", None)
        else:
            os.environ["VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING"] = falsy
        _clear_envs_cache()

        _apply_nccl_async_error_handling_policy()

        assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ, (
            f"falsey value {falsy!r} should NOT have preserved the env"
        )


def test_opt_in_emits_warning_log():
    """When the operator opts in, a warning must be emitted documenting
    the tradeoff so it shows up in startup logs."""
    os.environ["VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING"] = "1"
    _clear_envs_cache()

    with patch("vllm.v1.worker.gpu_worker.logger") as mock_logger:
        _apply_nccl_async_error_handling_policy()

        mock_logger.warning.assert_called_once()
        msg = mock_logger.warning.call_args[0][0]
        # The warning must reference the env name and the tradeoff
        # so operators reading the log can understand what changed.
        assert "VLLM_NCCL_ENABLE_ASYNC_ERROR_HANDLING" in msg
        assert "watchdog" in msg.lower()
        assert "45094" in msg


def test_default_does_not_emit_warning():
    """Default (env unset) must NOT spam a warning on every worker init."""
    with patch("vllm.v1.worker.gpu_worker.logger") as mock_logger:
        _apply_nccl_async_error_handling_policy()

        mock_logger.warning.assert_not_called()


def test_default_is_idempotent_when_envs_already_absent():
    """If neither NCCL env is set, the default path must not raise."""
    # Sanity: both should be absent from the autouse fixture.
    assert "NCCL_ASYNC_ERROR_HANDLING" not in os.environ
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ

    # Should be a no-op (pop with default=None), not a KeyError.
    _apply_nccl_async_error_handling_policy()

    assert "NCCL_ASYNC_ERROR_HANDLING" not in os.environ
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" not in os.environ
