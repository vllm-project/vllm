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
from vllm.platforms import current_platform

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

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        pass

    def build_connector_meta(self, scheduler_output):
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_tokens) -> None:
        pass


class MockCrossLayerConnector(MockConnector):
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True


class MockKVCacheEvents:
    """Mock KVConnectorKVEvents for testing event merging."""

    def __init__(self, events: list | None = None):
        self._events = events or []

    def add_events(self, events: list):
        self._events.extend(events)

    def get_all_events(self) -> list:
        return self._events


class MockConnectorWithKVCacheEvents(MockConnector):
    """Mock connector for testing get_kv_connector_kv_cache_events."""

    def __init__(self):
        # Skip parent __init__ to avoid needing vllm_config
        self._kv_cache_events = None

    def get_kv_connector_kv_cache_events(self):
        return self._kv_cache_events


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


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason=(
        "hipErrorLaunchFailure when running this test, see issue:"
        "https://github.com/ROCm/pytorch/issues/2822"
    ),
)
def test_multi_example_connector_consistency():
    """
    Tests that MultiConnector with two ExampleConnectors saves
    identical KV cache data to separate storage locations.
    """
    storage_1_path = Path("storage_1/")
    storage_2_path = Path("storage_2/")
    shutil.rmtree(storage_1_path, ignore_errors=True)
    shutil.rmtree(storage_2_path, ignore_errors=True)
    storage_1_path.mkdir()
    storage_2_path.mkdir()

    # Configure MultiConnector with two ExampleConnectors
    kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "TestExampleConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "shared_storage_path": str(storage_1_path),
                        "name": "storage1",
                    },
                    "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                },
                {
                    "kv_connector": "TestExampleConnector",
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


def test_multi_connector_handle_preemptions_integration():
    """
    Integration test: verify MultiConnector delegates handle_preemptions
    to all sub-connectors.

    Uses TestExampleConnector which logs all method calls to temp files.
    This test directly calls handle_preemptions on a MultiConnector with
    TestExampleConnector sub-connectors and verifies the calls are logged.
    """
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )

    storage_path = Path(tempfile.mkdtemp())

    try:
        # Configure MultiConnector with two TestExampleConnectors
        kv_transfer_config = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "TestExampleConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "shared_storage_path": str(storage_path / "s1"),
                            "name": "preempt1",
                        },
                        "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                    },
                    {
                        "kv_connector": "TestExampleConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "shared_storage_path": str(storage_path / "s2"),
                            "name": "preempt2",
                        },
                        "kv_connector_module_path": "tests.v1.kv_connector.unit.utils",
                    },
                ]
            },
        )

        vllm_config = create_vllm_config(
            block_size=16,
            max_num_batched_tokens=100,
            kv_connector_extra_config=kv_transfer_config.kv_connector_extra_config,
        )
        vllm_config.kv_transfer_config = kv_transfer_config

        # Create scheduler - this initializes the MultiConnector with SCHEDULER role
        scheduler = create_scheduler(vllm_config, num_blocks=10)

        # Clear any events from initialization
        get_connector_events()

        # Directly call handle_preemptions on the scheduler's connector
        # Note: handle_preemptions is normally a worker-side method, but we're
        # testing the delegation behavior of MultiConnector here.
        # The connector attribute contains the KV connector.
        assert scheduler.connector is not None, "Scheduler should have a connector"
        preempted_req_ids = {"req-1", "req-2", "req-3"}
        scheduler.connector.handle_preemptions(preempted_req_ids)

        # Verify both connectors received the handle_preemptions call
        events = get_connector_events()

        # Both SCHEDULER-role connectors should have logged handle_preemptions
        assert "handle_preemptions" in events.get("preempt1-SCHEDULER", []), (
            f"preempt1-SCHEDULER should have handle_preemptions call. "
            f"Got events: {events}"
        )
        assert "handle_preemptions" in events.get("preempt2-SCHEDULER", []), (
            f"preempt2-SCHEDULER should have handle_preemptions call. "
            f"Got events: {events}"
        )

    finally:
        # Cleanup
        shutil.rmtree(storage_path, ignore_errors=True)


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
        # ExampleConnector doesn't override build_kv_connector_stats,
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
            "ExampleConnector": {"data": {"some_field": [1, 2, 3]}},
        }

        stats = MultiConnector.build_kv_connector_stats(data=serialized_data)

        assert stats is not None
        assert isinstance(stats, MultiKVConnectorStats)
        # Only NixlConnector should be reconstructed
        assert len(stats.data) == 1
        assert "NixlConnector" in stats.data
        assert isinstance(stats.data["NixlConnector"], NixlKVConnectorStats)
        # ExampleConnector should be skipped (returns None)
        assert "ExampleConnector" not in stats.data

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
            data={"ExampleConnector": KVConnectorStats(data={"field": [1, 2]})}
        )

        result = stats1.aggregate(stats2)

        assert "NixlConnector" in result.data
        assert "ExampleConnector" in result.data

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


class TestMultiConnectorPreferCrossLayerBlocks:
    def test_all_connectors_prefer_cross_layer_blocks(self):
        mc = MultiConnector.__new__(MultiConnector)
        mc._connectors = [
            MockCrossLayerConnector.__new__(MockCrossLayerConnector),
            MockCrossLayerConnector.__new__(MockCrossLayerConnector),
        ]
        assert mc.prefer_cross_layer_blocks is True

    def test_mixed_connectors_do_not_prefer_cross_layer_blocks(self):
        mc = MultiConnector.__new__(MultiConnector)
        mc._connectors = [
            MockCrossLayerConnector.__new__(MockCrossLayerConnector),
            MockConnector.__new__(MockConnector),  # default False
        ]
        assert mc.prefer_cross_layer_blocks is False


class TestMultiConnectorGetKVCacheEvents:
    """Tests for get_kv_connector_kv_cache_events merge logic."""

    def test_all_return_none(self):
        mc = MultiConnector.__new__(MultiConnector)
        conn1 = MockConnectorWithKVCacheEvents()
        conn2 = MockConnectorWithKVCacheEvents()
        conn1._kv_cache_events = None
        conn2._kv_cache_events = None
        mc._connectors = [conn1, conn2]

        result = mc.get_kv_connector_kv_cache_events()

        assert result is None

    def test_single_connector_returns_events(self):
        mc = MultiConnector.__new__(MultiConnector)
        conn1 = MockConnectorWithKVCacheEvents()
        conn2 = MockConnectorWithKVCacheEvents()
        events1 = MockKVCacheEvents(["event1", "event2"])
        conn1._kv_cache_events = events1
        conn2._kv_cache_events = None
        mc._connectors = [conn1, conn2]

        result = mc.get_kv_connector_kv_cache_events()

        assert result is events1
        assert result.get_all_events() == ["event1", "event2"]

    def test_merges_events_from_multiple_connectors(self):
        mc = MultiConnector.__new__(MultiConnector)
        conn1 = MockConnectorWithKVCacheEvents()
        conn2 = MockConnectorWithKVCacheEvents()
        conn3 = MockConnectorWithKVCacheEvents()
        conn1._kv_cache_events = MockKVCacheEvents(["a", "b"])
        conn2._kv_cache_events = MockKVCacheEvents(["c", "d"])
        conn3._kv_cache_events = None  # Should be skipped
        mc._connectors = [conn1, conn2, conn3]

        result = mc.get_kv_connector_kv_cache_events()

        # Events from conn1 and conn2 should be merged
        assert result.get_all_events() == ["a", "b", "c", "d"]

    def test_first_none_second_returns_value(self):
        mc = MultiConnector.__new__(MultiConnector)
        conn1 = MockConnectorWithKVCacheEvents()
        conn2 = MockConnectorWithKVCacheEvents()
        events2 = MockKVCacheEvents(["event"])
        conn1._kv_cache_events = None
        conn2._kv_cache_events = events2
        mc._connectors = [conn1, conn2]

        result = mc.get_kv_connector_kv_cache_events()

        assert result is events2

    def test_empty_connectors_list(self):
        mc = MultiConnector.__new__(MultiConnector)
        mc._connectors = []

        result = mc.get_kv_connector_kv_cache_events()

        assert result is None


def test_multi_connector_overrides_all_base_methods():
    """
    Ensure MultiConnector overrides all public methods from KVConnectorBase_V1.
    """
    # These are fine to inherit from KVConnectorBase_V1
    INHERITED_OK = {"role", "has_connector_metadata"}

    base_members = {
        name for name in dir(KVConnectorBase_V1) if not name.startswith("_")
    } - KVConnectorBase_V1.__abstractmethods__

    missing = [
        name
        for name in sorted(base_members)
        if name not in INHERITED_OK and name not in MultiConnector.__dict__
    ]

    if missing:
        pytest.fail(f"""
MultiConnector does not override these KVConnectorBase_V1 methods: {missing}

MultiConnector wraps other connectors and must delegate all methods.
Please add overrides that delegate to self._connectors.

Options:
  1. Add delegation in MultiConnector (preferred)
  2. Add to INHERITED_OK if the base implementation works correctly
""")
