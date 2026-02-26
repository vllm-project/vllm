# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.device_communicators.all_reduce_utils import (
    get_capability_config,
)


@pytest.fixture
def sample_config():
    return {
        "9.0": {"hopper_data": True},
        "10.0": {"blackwell_data": True},
    }


def test_exact_match(sample_config):
    assert get_capability_config(sample_config, "9.0") == {"hopper_data": True}
    assert get_capability_config(sample_config, "10.0") == {"blackwell_data": True}


def test_family_fallback(sample_config):
    """SM 10.3 (B300/GB200) should fall back to 10.0 config."""
    assert get_capability_config(sample_config, "10.3") == {"blackwell_data": True}
    assert get_capability_config(sample_config, "9.5") == {"hopper_data": True}


def test_exact_override_takes_priority():
    """If a specific minor version config exists, it takes priority."""
    config = {
        "10.0": "base_blackwell",
        "10.3": "gb200_specific",
    }
    assert get_capability_config(config, "10.3") == "gb200_specific"
    assert get_capability_config(config, "10.0") == "base_blackwell"
    # Other 10.x still falls back to 10.0
    assert get_capability_config(config, "10.1") == "base_blackwell"


def test_unsupported_architecture(sample_config):
    """Truly unsupported architectures return None."""
    assert get_capability_config(sample_config, "12.0") is None
    assert get_capability_config(sample_config, "8.9") is None


def test_self_referencing_family_key(sample_config):
    """When capability_version_str is already a .0 family key, exact match
    succeeds on the first try."""
    assert get_capability_config(sample_config, "9.0") == {"hopper_data": True}
