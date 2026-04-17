# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config import FileUploadConfig


def test_defaults_are_off_and_conservative():
    c = FileUploadConfig()
    assert c.enabled is False
    assert c.dir == ""
    assert c.ttl_seconds == 3600
    assert c.max_size_mb == 512
    assert c.max_total_gb == 5
    assert c.max_concurrent == 4
    assert c.scope_header == ""
    assert c.disable_listing is False


def test_custom_values_accepted():
    c = FileUploadConfig(
        enabled=True,
        dir="/tmp/vllm-uploads",
        ttl_seconds=-1,
        max_size_mb=1024,
        max_total_gb=20,
        max_concurrent=8,
        scope_header="OpenAI-Project",
        disable_listing=True,
    )
    assert c.enabled is True
    assert c.ttl_seconds == -1
    assert c.scope_header == "OpenAI-Project"
    assert c.disable_listing is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_size_mb", 0),
        ("max_size_mb", -1),
        ("max_total_gb", 0),
        ("max_total_gb", -5),
        ("max_concurrent", 0),
        ("max_concurrent", -1),
    ],
)
def test_rejects_non_positive_sizes(field, value):
    with pytest.raises(ValidationError):
        FileUploadConfig(**{field: value})


def test_ttl_seconds_allows_negative_sentinel():
    """`-1` is the explicit 'disable time-based expiry' sentinel. Any negative
    value other than -1 is still accepted (no validator), but only -1 is
    documented. We intentionally do not restrict to {-1, >=0} at the config
    layer — the sweeper treats any value < 0 as disabled."""
    c = FileUploadConfig(ttl_seconds=-1)
    assert c.ttl_seconds == -1
    c2 = FileUploadConfig(ttl_seconds=0)
    assert c2.ttl_seconds == 0


def test_rejects_extra_fields():
    """@config forbids extra fields — surfaces typos at config-load time."""
    with pytest.raises(ValidationError):
        FileUploadConfig(enable=True)  # typo: missing 'd'
