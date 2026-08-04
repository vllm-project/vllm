# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MPCoordinatorConfig validation and env loading."""

# Standard
from unittest.mock import patch
import os

# Third Party
import pytest

# First Party
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def test_defaults_are_valid():
    config = MPCoordinatorConfig()
    assert config.instance_timeout > 0


def test_non_positive_intervals_rejected():
    with pytest.raises(ValueError):
        MPCoordinatorConfig(instance_timeout=0.0)
    with pytest.raises(ValueError):
        MPCoordinatorConfig(health_check_interval=-1.0)


def test_from_env_overrides_and_falls_back():
    env = {
        "LMCACHE_MP_COORDINATOR_HOST": "127.0.0.1",
        "LMCACHE_MP_COORDINATOR_PORT": "7777",
        "LMCACHE_MP_COORDINATOR_INSTANCE_TIMEOUT": "42",
    }
    with patch.dict(os.environ, env, clear=False):
        config = MPCoordinatorConfig.from_env()
    assert config.host == "127.0.0.1"
    assert config.port == 7777
    assert config.instance_timeout == 42.0
    # Unset variable keeps the default.
    assert config.health_check_interval == MPCoordinatorConfig.health_check_interval
