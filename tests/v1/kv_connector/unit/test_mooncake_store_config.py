# SPDX-License-Identifier: Apache-2.0
"""Local pool validation for preallocated-buffer Store transfers."""

import pytest

from vllm.distributed.mooncake_store import MooncakeStoreConfig


def test_config_zero_local_buffer_disables_local_pool():
    config = MooncakeStoreConfig("", "localhost:50051", "tcp", "", local_buffer_size=0)
    assert config.local_buffer_size == 0


def test_config_rejects_negative_local_buffer():
    with pytest.raises(ValueError, match="local_buffer_size must be >= 0"):
        MooncakeStoreConfig("", "localhost:50051", "tcp", "", local_buffer_size=-1)
