# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for migrating weight-offloading env vars to OffloadConfig (#25700)."""

import pytest

import vllm.envs as envs
from vllm.config.offload import OffloadConfig
from vllm.model_executor.offloader import base

pytestmark = pytest.mark.cpu_test

PIN_ENV = "VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY"
UVA_ENV = "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA"


def test_new_fields_default_to_none():
    cfg = OffloadConfig()
    assert cfg.disable_pin_memory is None
    assert cfg.uva.disable_uva is None


@pytest.mark.parametrize("config_value", [True, False])
def test_config_takes_precedence_over_env(monkeypatch, config_value):
    # Explicit config wins even when the deprecated env var disagrees.
    monkeypatch.setattr(envs, PIN_ENV, not config_value)
    assert base.resolve_offload_flag(config_value, PIN_ENV) is config_value


def test_env_fallback_when_config_unset_and_warns_once(monkeypatch):
    monkeypatch.setattr(envs, UVA_ENV, True)
    base._warned_deprecated_offload_envs.discard(UVA_ENV)

    assert base.resolve_offload_flag(None, UVA_ENV) is True
    assert UVA_ENV in base._warned_deprecated_offload_envs

    # A second resolution must not re-add / re-warn (idempotent).
    before = set(base._warned_deprecated_offload_envs)
    assert base.resolve_offload_flag(None, UVA_ENV) is True
    assert base._warned_deprecated_offload_envs == before


def test_env_unset_defaults_to_false(monkeypatch):
    monkeypatch.setattr(envs, PIN_ENV, False)
    assert base.resolve_offload_flag(None, PIN_ENV) is False
