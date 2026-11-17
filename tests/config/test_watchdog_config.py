# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config import WatchdogConfig


def test_watchdog_config_defaults():
    """Verify WatchdogConfig defaults stay in sync with the WatchDog module
    defaults."""
    config = WatchdogConfig()
    assert config.timeout == 300
    assert config.check_interval == 10
    assert config.dump_dir == ""


def test_watchdog_config_custom_values():
    """Verify non-default watchdog parameters are honored."""
    config = WatchdogConfig(timeout=60, check_interval=5, dump_dir="/tmp/dumps")
    assert config.timeout == 60
    assert config.check_interval == 5
    assert config.dump_dir == "/tmp/dumps"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": -1},
        {"check_interval": 0},
        {"check_interval": -1},
    ],
)
def test_watchdog_config_rejects_non_positive_values(kwargs):
    """Verify timeout and check_interval must be greater than zero."""
    with pytest.raises(ValidationError):
        WatchdogConfig(**kwargs)


def test_watchdog_config_rejects_non_integral_check_interval():
    """Verify check_interval must be an int; non-integral floats are
    rejected."""
    with pytest.raises(ValidationError):
        WatchdogConfig(check_interval=0)


def test_watchdog_config_check_interval_is_stored_as_int():
    """Verify check_interval is coerced to and stored as an int."""
    config = WatchdogConfig(check_interval=5)
    assert config.check_interval == 5
    assert isinstance(config.check_interval, int)
