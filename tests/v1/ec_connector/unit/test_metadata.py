# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ECCPUConnectorMetadata."""

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
)


def test_metadata_is_ec_connector_metadata():
    """ECCPUConnectorMetadata must be a subclass of ECConnectorMetadata."""
    meta = ECCPUConnectorMetadata()
    assert isinstance(meta, ECConnectorMetadata)


def test_metadata_initialization():
    """ECCPUConnectorMetadata initializes with empty saves and loads dicts."""
    meta = ECCPUConnectorMetadata()
    assert hasattr(meta, "saves")
    assert hasattr(meta, "loads")
    assert meta.saves == {}
    assert meta.loads == {}


def test_metadata_saves_dict_operations():
    """The saves dict supports standard dict operations."""
    meta = ECCPUConnectorMetadata()

    # Add multiple entries
    meta.saves["mm_hash_1"] = [0, 1, 2]
    meta.saves["mm_hash_2"] = [10, 11]

    assert "mm_hash_1" in meta.saves
    assert "mm_hash_2" in meta.saves
    assert len(meta.saves) == 2
    assert meta.saves["mm_hash_1"] == [0, 1, 2]
    assert meta.saves["mm_hash_2"] == [10, 11]


def test_metadata_loads_carry_a_transfer_id_and_blocks():
    """Each load entry is a (transfer_id, block_ids) pair, so the worker can
    report the transfer rather than the mm_hash."""
    meta = ECCPUConnectorMetadata()

    meta.loads["mm_hash_1"] = (0, [5, 6, 7])
    meta.loads["mm_hash_2"] = (1, [100])

    assert len(meta.loads) == 2
    assert meta.loads["mm_hash_1"] == (0, [5, 6, 7])
    transfer_id, block_ids = meta.loads["mm_hash_2"]
    assert transfer_id == 1
    assert block_ids == [100]


def test_metadata_saves_and_loads_are_independent():
    """saves and loads dicts are independent."""
    meta = ECCPUConnectorMetadata()

    meta.saves["key"] = [1, 2]
    meta.loads["key"] = (0, [3, 4])

    # Same key but different dicts should not interfere
    assert meta.saves["key"] == [1, 2]
    assert meta.loads["key"] == (0, [3, 4])

    # Modifying one should not affect the other
    meta.saves["other"] = [99]
    assert "other" not in meta.loads


def test_metadata_block_indices_are_integers():
    """Block indices in saves/loads lists should be integers."""
    meta = ECCPUConnectorMetadata()

    meta.saves["hash"] = [0, 100, 999, 65535]
    meta.loads["hash"] = (0, [1, 2, 3])

    for idx in meta.saves["hash"]:
        assert isinstance(idx, int)
    for idx in meta.loads["hash"][1]:
        assert isinstance(idx, int)


def test_metadata_multiple_instances_are_independent():
    """Different ECCPUConnectorMetadata instances should not share state."""
    meta1 = ECCPUConnectorMetadata()
    meta2 = ECCPUConnectorMetadata()

    meta1.saves["x"] = [1]
    meta2.saves["y"] = [2]

    assert "x" in meta1.saves
    assert "x" not in meta2.saves
    assert "y" not in meta1.saves
    assert "y" in meta2.saves
