# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MoRIIO deferred-send reap timeout resolution.

``VLLM_MORIIO_DEFERRED_TIMEOUT_S`` was documented but never read, so an
operator setting it was silently given the default instead. These lock in the
precedence env > ``kv_connector_extra_config["defer_timeout"]`` > default, and
that a bad env value is ignored rather than crashing or being taken literally.

These are pure-python and need neither ROCm nor the ``mori`` package, so unlike
test_moriio_connector.py they are not platform-gated.
"""

from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_common
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConstants,
    log_resolved_defer_timeout,
    resolve_defer_timeout,
    resolve_defer_timeout_with_source,
)

ENV = MoRIIOConstants.ENV_DEFERRED_TIMEOUT_S


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Never inherit the operator's env; each test sets what it needs.

    The value is now read through ``vllm.envs``; disable its lazy cache so each
    ``monkeypatch.setenv``/``delenv`` is observed live and precedence tests stay
    deterministic regardless of prior cache state.
    """
    envs.disable_envs_cache()
    monkeypatch.delenv(ENV, raising=False)


def test_env_var_is_registered_in_vllm_envs():
    """Registered centrally so validate_environ() no longer flags it as unknown."""
    assert ENV in envs.environment_variables
    assert ENV == "VLLM_MORIIO_DEFERRED_TIMEOUT_S"


def test_env_wins_over_extra_config(monkeypatch):
    monkeypatch.setenv(ENV, "1800")
    assert resolve_defer_timeout({"defer_timeout": "300"}) == 1800.0


def test_extra_config_used_when_env_unset():
    assert resolve_defer_timeout({"defer_timeout": "300"}) == 300.0


def test_default_when_neither_set():
    assert resolve_defer_timeout({}) == MoRIIOConstants.DEFAULT_DEFER_TIMEOUT


def test_default_is_conservative():
    """The 60s default stranded blocks; 300s is the reviewed value."""
    assert MoRIIOConstants.DEFAULT_DEFER_TIMEOUT == 300.0


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "30s", ""])
def test_bad_env_falls_through_to_extra_config(monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    assert resolve_defer_timeout({"defer_timeout": "300"}) == 300.0


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "30s"])
def test_bad_env_warns(monkeypatch, bad):
    monkeypatch.setenv(ENV, bad)
    with patch.object(moriio_common.logger, "warning") as mock_warn:
        assert resolve_defer_timeout({}) == MoRIIOConstants.DEFAULT_DEFER_TIMEOUT
    assert mock_warn.call_count == 1
    assert ENV in mock_warn.call_args.args[0] % mock_warn.call_args.args[1:]


def test_empty_env_is_not_a_warning(monkeypatch):
    """An unset-looking value is indistinguishable from unset; stay quiet."""
    monkeypatch.setenv(ENV, "")
    with patch.object(moriio_common.logger, "warning") as mock_warn:
        assert resolve_defer_timeout({}) == MoRIIOConstants.DEFAULT_DEFER_TIMEOUT
    mock_warn.assert_not_called()


def test_env_accepts_float(monkeypatch):
    monkeypatch.setenv(ENV, "12.5")
    assert resolve_defer_timeout({}) == 12.5


@pytest.mark.parametrize(
    "env,extra,expected_source",
    [
        ("1800", {"defer_timeout": "300"}, f"env {ENV}"),
        (None, {"defer_timeout": "300"}, 'kv_connector_extra_config["defer_timeout"]'),
        (None, {}, "built-in default"),
        # A rejected env value must not be credited as the source.
        ("abc", {}, "built-in default"),
    ],
)
def test_reported_source(monkeypatch, env, extra, expected_source):
    if env is not None:
        monkeypatch.setenv(ENV, env)
    _, source = resolve_defer_timeout_with_source(extra)
    assert source == expected_source


def test_startup_log_records_value_and_source(monkeypatch):
    """The startup line must make the effective timeout readable from logs."""
    monkeypatch.setenv(ENV, "1800")
    with patch.object(moriio_common.logger, "info") as mock_info:
        resolved = log_resolved_defer_timeout({"defer_timeout": "300"}, "scheduler")
    assert resolved == 1800.0
    rendered = mock_info.call_args.args[0] % mock_info.call_args.args[1:]
    assert "scheduler" in rendered
    assert "1800.0s" in rendered
    assert ENV in rendered
