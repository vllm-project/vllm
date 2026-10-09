# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem behavior of ECExampleConnector's shared-storage paths.

Regression tests for #60820: cache *lookups* must not create hash
directories in shared storage; only saves may.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnector,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def producer_connector(tmp_path):
    config = Mock(spec=VllmConfig)
    config.ec_transfer_config = Mock()
    config.ec_transfer_config.get_from_extra_config = Mock(return_value=str(tmp_path))
    config.ec_transfer_config.is_ec_producer = True
    config.ec_transfer_config.is_ec_consumer = False
    return ECExampleConnector(vllm_config=config, role=ECConnectorRole.WORKER)


def test_lookup_miss_creates_no_directory(producer_connector, tmp_path):
    """A cache-miss probe must not leave an empty hash directory behind."""
    assert not producer_connector.has_cache_item("deadbeef")
    assert list(tmp_path.iterdir()) == []


def test_save_creates_directory_and_lookup_roundtrip(producer_connector, tmp_path):
    """Saves still create the hash directory; lookups then find the entry."""
    producer_connector.save_caches(
        encoder_cache={"deadbeef": torch.zeros(2, 3)}, mm_hash="deadbeef"
    )

    assert (tmp_path / "deadbeef" / "encoder_cache.safetensors").is_file()
    assert producer_connector.has_cache_item("deadbeef")

    # A second, different miss must not create a directory either.
    assert not producer_connector.has_cache_item("cafef00d")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["deadbeef"]
