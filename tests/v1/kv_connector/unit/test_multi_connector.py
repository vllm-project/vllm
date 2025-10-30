# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import filecmp
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlKVConnectorStats,
)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

PROMPT_CONTEXT = "Hi " * 100
PROMPTS = [
    PROMPT_CONTEXT + "Hello, my name is",
    PROMPT_CONTEXT + "The capital of France is",
]

SAMPLING_PARAMS = SamplingParams(temperature=0, max_tokens=20)


# Test connector with custom stats for testing MultiConnector
class MockConnectorStats(KVConnectorStats):
    """Mock stats class for testing."""

    pass


class MockConnector(KVConnectorBase_V1):
    """Mock connector that implements build_kv_connector_stats for testing."""

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return MockConnectorStats(data=data) if data is not None else None


# Register the mock connector
KVConnectorFactory.register_connector("MockConnector", __name__, MockConnector.__name__)


# Helper function to compare directories recursively
def _compare_directories(dir1: Path, dir2: Path) -> bool:
    """Compares two directories recursively for identical content."""
    dcmp = filecmp.dircmp(dir1, dir2)
    if dcmp.left_only or dcmp.right_only or dcmp.diff_files:
        print(f"Differences found between {dir1} and {dir2}:")
        print(f"  Left only: {dcmp.left_only}")
        print(f"  Right only: {dcmp.right_only}")
        print(f"  Different files: {dcmp.diff_files}")
        return False
    for sub_dir in dcmp.common_dirs:
        if not _compare_directories(dir1 / sub_dir, dir2 / sub_dir):
            return False
    return True


def test_multi_shared_storage_connector_consistency():
    """
    Tests that MultiConnector with two SharedStorageConnectors saves
    identical KV cache data to separate storage locations.
    """
    storage_1_path = Path("storage_1/")
    storage_2_path = Path("storage_2/")
    shutil.rmtree(storage_1_path, ignore_errors=True)
    shutil.rmtree(storage_2_path, ignore_errors=True)
    storage_1_path.mkdir()
    storage_2_path.mkdir()

    # Configure MultiConnector with two SharedStorageConnectors
    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "TestSharedStorageConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_1_path),
                        "name": "storage1",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
                {
                    "kv_connector": "TestSharedStorageConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_2_path),
                        "name": "storage2",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
            ]
        },
    )

    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        kv_transfer_config=kv_transfer_config,
        disable_hybrid_kv_cache_manager=True,
    )
    # Run generation - this should trigger saving KV cache
    _ = llm.generate(PROMPTS, SAMPLING_PARAMS)

    # --- Verification ---

    # Check that both storage directories were populated
    local_subdirs = list(storage_1_path.iterdir())
    external_subdirs = list(storage_2_path.iterdir())

    assert len(local_subdirs) > 0, (
        f"Local storage path {storage_1_path} is empty after generation."
    )
    assert len(external_subdirs) > 0, (
        f"External storage path {storage_2_path} is empty after generation."
    )
    assert len(local_subdirs) == len(external_subdirs), (
        f"Mismatch in number of cache entries: "
        f"Local={len(local_subdirs)}, External={len(external_subdirs)}"
    )

    # The subdirectories should correspond to the prompt hashes
    # Since prompts are the same, the hash directories should be the same name
    local_subdir_names = sorted([d.name for d in local_subdirs])
    external_subdir_names = sorted([d.name for d in external_subdirs])
    assert local_subdir_names == external_subdir_names, (
        "Cache directory names do not match between local and external storage"
    )

    # Compare the contents of each corresponding cache directory
    for subdir_name in local_subdir_names:
        print(f"Comparing contents of cache directory: {subdir_name}")
        assert _compare_directories(
            storage_1_path / subdir_name, storage_2_path / subdir_name
        ), (
            f"Contents differ for cache directory '{subdir_name}' between "
            f"{storage_1_path} and {storage_2_path}"
        )

    events = get_connector_events()
    # get_num_new_matched_tokens and update_state_after_alloc will be called
    # on each connector in turn.
    assert events["storage1-SCHEDULER"][:3] == [
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[0] 0",
        "build_connector_meta",
    ]
    assert events["storage1-WORKER"][:5] == [
        "register_kv_caches",
        "bind_connector_metadata",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
    ]
    assert events["storage2-SCHEDULER"][:3] == [
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[0] 0",
        "build_connector_meta",
    ]
    assert events["storage2-WORKER"][:5] == [
        "register_kv_caches",
        "bind_connector_metadata",
        "start_load_kv",
        "wait_for_layer_load",
        "save_kv_layer",
    ]

    # Reset prefix cache or else we'll just get the tokens back from there.
    llm.reset_prefix_cache()

    # Run generation again - this should trigger loading from the first
    # connector.
    _ = llm.generate(PROMPTS, SAMPLING_PARAMS)

    events = get_connector_events()
    # get_num_new_matched_tokens will return new tokens from the first
    # connector so update_state_after_alloc will be with allocated blocks
    # on that one but with zero blocks for others (first nonzero match is
    # chosen).
    assert events["storage1-SCHEDULER"][:3] == [
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 96",
        "build_connector_meta",
    ]
    assert events["storage2-SCHEDULER"][:3] == [
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[0] 0",
        "build_connector_meta",
    ]

    # Delete storage1 connector state
    shutil.rmtree(storage_1_path)

    # Reset prefix cache or else we'll just get the tokens back from there.
    llm.reset_prefix_cache()

    # Run generation again - this should trigger loading from the first
    # connector.
    _ = llm.generate(PROMPTS, SAMPLING_PARAMS)

    events = get_connector_events()
    # get_num_new_matched_tokens will be called for both connectors but will
    # return 0 from the first connector, but the second connector should have
    # a hit, so update_state_after_alloc will only be called with allocated
    # blocks for the second connector.
    assert events["storage1-SCHEDULER"][:3] == [
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[0] 0",
        "build_connector_meta",
    ]
    assert events["storage2-SCHEDULER"][:3] == [
        "get_num_new_matched_tokens 0",
        "update_state_after_alloc num_blocks=[7] 96",
        "build_connector_meta",
    ]

    # Clean up
    shutil.rmtree(storage_1_path)
    shutil.rmtree(storage_2_path)


def get_connector_events() -> dict[str, list[str]]:
    # Read in connector events and reset the files.
    import glob

    event_files = glob.glob(tempfile.gettempdir() + "/connector_*_events.log")
    connector_events = {}
    for fname in event_files:
        name = fname.split("connector_")[1].split("_events.log")[0]
        try:
            with open(fname, "r+") as f:
                connector_events[name] = [line.strip() for line in f if line.strip()]
                f.truncate(0)
        except Exception as e:
            print(f"[ERROR] Could not read connector events for {name}: {e}")

    return connector_events


def test_engine_id_conflict():
    configs = [KVTransferConfig() for _ in range(2)]
    ids = [config.engine_id for config in configs]
    assert ids[0] != ids[1], (
        f"Engine IDs should be different for different configs. Got {ids}"
    )


class TestMultiConnectorStats:
    """Tests for MultiConnector stats reconstruction and operations."""

    def test_build_kv_connector_stats_with_none(self):
        """Test that build_kv_connector_stats returns empty stats when given None."""
        stats = MultiConnector.build_kv_connector_stats(data=None)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_with_empty_dict(self):
        """Test that build_kv_connector_stats returns empty stats with empty dict."""
        stats = MultiConnector.build_kv_connector_stats(data={})

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 0
        assert stats.is_empty()

    def test_build_kv_connector_stats_reconstructs_nixl_stats(self):
        """Test that NixlConnector stats are properly reconstructed with
        correct data."""
        serialized_data = {
            "NixlConnector": {
                "data": {
                    "transfer_duration": [1.5, 2.3],
                    "post_duration": [0.1, 0.2],
                    "bytes_transferred": [1024, 2048],
                    "num_descriptors": [10, 20],
                    "num_failed_transfers": [],
                    "num_failed_notifications": [],
                }
            }
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert "NixlConnector" in stats.data
        nixl_stats = stats.data["NixlConnector"]
        assert isinstance(nixl_stats, NixlKVConnectorStats)
        assert nixl_stats.data["transfer_duration"] == [1.5, 2.3]
        assert nixl_stats.data["post_duration"] == [0.1, 0.2]
        assert nixl_stats.data["bytes_transferred"] == [1024, 2048]
        assert nixl_stats.data["num_descriptors"] == [10, 20]

    def test_build_kv_connector_stats_with_multiple_connectors(self):
        """Test reconstruction with multiple connector types that have custom stats."""
        serialized_data = {
            "NixlConnector": {
                "data": {
                    "transfer_duration": [1.5],
                    "post_duration": [0.1],
                    "bytes_transferred": [1024],
                    "num_descriptors": [10],
                    "num_failed_transfers": [],
                    "num_failed_notifications": [],
                }
            },
            "MockConnector": {"data": {"mock_field": [1, 2, 3]}},
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        # Both connectors should be reconstructed
        assert len(stats.data) == 2
        assert "NixlConnector" in stats.data
        assert "MockConnector" in stats.data
        assert isinstance(stats.data["NixlConnector"], NixlKVConnectorStats)
        assert isinstance(stats.data["MockConnector"], MockConnectorStats)
        # Verify data is preserved
        assert stats.data["MockConnector"].data == {"mock_field": [1, 2, 3]}

    def test_build_kv_connector_stats_raises_error_for_unknown_connector(self):
        """Test that unknown connectors raise an error."""
        serialized_data = {
            "UnknownConnector": {"data": {"some_field": [1, 2, 3]}},
            "NixlConnector": {
                "data": {
                    "transfer_duration": [1.5],
                    "post_duration": [0.1],
                    "bytes_transferred": [1024],
                    "num_descriptors": [10],
                    "num_failed_transfers": [],
                    "num_failed_notifications": [],
                }
            },
        }

        with pytest.raises(
            ValueError, match="Connector 'UnknownConnector' is not registered."
        ):
            MultiConnector.build_kv_connector_stats(data=serialized_data)

    def test_build_kv_connector_stats_with_already_instantiated_objects(self):
        """Test that already-instantiated stats objects are preserved (same process)."""
        # This simulates the in-process case where stats are not serialized
        nixl_stats = NixlKVConnectorStats(
            data={
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
            }
        )
        mock_stats = MockConnectorStats(data={"mock_field": [1, 2, 3]})

        data_with_objects = {
            "NixlConnector": nixl_stats,
            "MockConnector": mock_stats,
        }

        stats = MultiConnector.build_kv_connector_stats(data=data_with_objects)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 2
        # Verify objects are preserved as-is
        assert stats.data["NixlConnector"] is nixl_stats
        assert stats.data["MockConnector"] is mock_stats

    def test_build_kv_connector_stats_with_mixed_objects_and_dicts(self):
        """Test handling mixed already-instantiated and serialized stats."""
        # This can happen during transition or partial serialization
        nixl_stats = NixlKVConnectorStats(
            data={
                "transfer_duration": [1.5],
                "post_duration": [0.1],
                "bytes_transferred": [1024],
                "num_descriptors": [10],
                "num_failed_transfers": [],
                "num_failed_notifications": [],
            }
        )

        mixed_data = {
            "NixlConnector": nixl_stats,  # Already instantiated
            "MockConnector": {"data": {"mock_field": [1, 2, 3]}},  # Serialized
        }

        stats = MultiConnector.build_kv_connector_stats(data=mixed_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        assert len(stats.data) == 2
        # Instantiated object preserved
        assert stats.data["NixlConnector"] is nixl_stats
        # Serialized object reconstructed
        assert isinstance(stats.data["MockConnector"], MockConnectorStats)
        assert stats.data["MockConnector"].data == {"mock_field": [1, 2, 3]}

    def test_build_kv_connector_stats_skips_connectors_without_custom_stats(self):
        """Test that connectors without custom stats (return None) are skipped."""
        # SharedStorageConnector doesn't override build_kv_connector_stats,
        # so it returns None and should be skipped
        serialized_data = {
            "NixlConnector": {
                "data": {
                    "transfer_duration": [1.5],
                    "post_duration": [0.1],
                    "bytes_transferred": [1024],
                    "num_descriptors": [10],
                    "num_failed_transfers": [],
                    "num_failed_notifications": [],
                }
            },
            "SharedStorageConnector": {"data": {"some_field": [1, 2, 3]}},
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        # Only NixlConnector should be reconstructed
        assert len(stats.data) == 1
        assert "NixlConnector" in stats.data
        assert isinstance(stats.data["NixlConnector"], NixlKVConnectorStats)
        # SharedStorageConnector should be skipped (returns None)
        assert "SharedStorageConnector" not in stats.data

    def test_build_kv_connector_stats_handles_malformed_data(self):
        """Test that malformed data raises appropriate errors."""
        serialized_data = {
            "NixlConnector": {"wrong_field": {"transfer_duration": [1.5]}}
        }

        with pytest.raises(AssertionError, match="Expected a dict with a 'data' field"):
            MultiConnector.build_kv_connector_stats(data=serialized_data)

    def test_aggregate_same_connector(self):
        """Test aggregating stats from the same connector type."""
        stats1 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0],
                        "post_duration": [0.1],
                        "bytes_transferred": [1024],
                        "num_descriptors": [10],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        stats2 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [2.0],
                        "post_duration": [0.2],
                        "bytes_transferred": [2048],
                        "num_descriptors": [20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        result = stats1.aggregate(stats2)

        assert result is stats1  # Should return self
        assert "NixlConnector" in result.data
        nixl_stats = result.data["NixlConnector"]
        assert nixl_stats.data["transfer_duration"] == [1.0, 2.0]
        assert nixl_stats.data["post_duration"] == [0.1, 0.2]
        assert nixl_stats.data["bytes_transferred"] == [1024, 2048]
        assert nixl_stats.data["num_descriptors"] == [10, 20]

    def test_aggregate_new_connector(self):
        """Test aggregating stats when a new connector type appears."""
        from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
            KVConnectorStats,
        )

        stats1 = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0],
                        "post_duration": [0.1],
                        "bytes_transferred": [1024],
                        "num_descriptors": [10],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        stats2 = MultiKVConnectorStats(
            data={"SharedStorageConnector": KVConnectorStats(data={"field": [1, 2]})}
        )

        result = stats1.aggregate(stats2)

        assert "NixlConnector" in result.data
        assert "SharedStorageConnector" in result.data

    def test_reduce(self):
        """Test that reduce() correctly reduces all nested connector stats."""
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0, 2.0],
                        "post_duration": [0.1, 0.2],
                        "bytes_transferred": [1024, 2048],
                        "num_descriptors": [10, 20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        reduced = stats.reduce()

        assert "NixlConnector" in reduced
        assert isinstance(reduced["NixlConnector"], dict)
        # Check that the stats were reduced (should have aggregated values)
        assert "Num successful transfers" in reduced["NixlConnector"]
        assert reduced["NixlConnector"]["Num successful transfers"] == 2

    def test_reset(self):
        """Test that reset() resets all nested connector stats."""
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(
                    data={
                        "transfer_duration": [1.0, 2.0],
                        "post_duration": [0.1, 0.2],
                        "bytes_transferred": [1024, 2048],
                        "num_descriptors": [10, 20],
                        "num_failed_transfers": [],
                        "num_failed_notifications": [],
                    }
                )
            }
        )

        assert not stats.is_empty()

        stats.reset()

        # After reset, stats should be empty
        assert stats.is_empty()
        nixl_stats = stats.data["NixlConnector"]
        assert len(nixl_stats.data["transfer_duration"]) == 0

    def test_is_empty_with_multiple_connectors(self):
        """Test is_empty() returns correct value with multiple connectors."""
        # All empty
        stats = MultiKVConnectorStats(
            data={
                "NixlConnector": NixlKVConnectorStats(data={}),
            }
        )
        # Initialize empty stats
        stats.data["NixlConnector"].reset()
        assert stats.is_empty()

        # One non-empty
        stats.data["NixlConnector"].data["transfer_duration"].append(1.0)
        assert not stats.is_empty()
