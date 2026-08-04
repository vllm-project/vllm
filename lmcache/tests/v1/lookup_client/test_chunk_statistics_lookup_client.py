# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Optional, Union
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)
from lmcache.v1.lookup_client.record_strategies import _get_strategies
from lmcache.v1.utils.bloom_filter import BloomFilter


class BaseMockClient(LookupClientInterface):
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size

    def clear_lookup_status(self, lookup_id: str) -> None:
        pass

    def supports_producer_reuse(self) -> bool:
        return True

    def close(self) -> None:
        pass


class MockLookupClient(BaseMockClient):
    def __init__(self):
        super().__init__()
        self.lookup_calls = []

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        self.lookup_calls.append((token_ids, lookup_id, request_configs))
        return len(token_ids) // 2


class FastMissLookupClient(BaseMockClient):
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        time.sleep(0.008)
        return 0


class BaseTestCase:
    def create_stats_client(self, **kwargs):
        return ChunkStatisticsLookupClient(
            MockLookupClient(), LMCacheEngineConfig(**kwargs)
        )

    def setup_stats_client(self, **kwargs):
        default_kwargs = {
            "enable_chunk_statistics": True,
            "chunk_statistics_strategy": "memory_bloom_filter",
            "extra_config": {
                "chunk_statistics_mem_bf_expected_chunks": 1000,
                "chunk_statistics_mem_bf_false_positive_rate": 0.01,
            },
        }
        if "extra_config" in kwargs:
            default_kwargs["extra_config"].update(kwargs.pop("extra_config"))
        default_kwargs.update(kwargs)
        stats_client = self.create_stats_client(**default_kwargs)
        stats_client.start_statistics()
        return stats_client, stats_client.actual_lookup_client


class TestStrategyDiscovery:
    def test_get_strategies_discovers_all_strategies(self):
        strategies = _get_strategies()
        assert isinstance(strategies, dict) and len(strategies) >= 2
        assert "file_hash" in strategies and "memory_bloom_filter" in strategies
        assert strategies["file_hash"].__module__.split(".")[-1] == "file_hash"
        assert (
            strategies["memory_bloom_filter"].__module__.split(".")[-1]
            == "memory_bloom_filter"
        )


class TestBloomFilter:
    def test_bloom_filter_operations(self):
        bf = BloomFilter(expected_elements=1000, false_positive_rate=0.01)
        bf.add("test_item_1")
        bf.add("test_item_2")
        assert bf.contains("test_item_1") and bf.contains("test_item_2")
        assert not bf.contains("test_item_3")
        bf.clear()
        assert not bf.contains("test_item_1")

    def test_false_positive_rate(self):
        bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)
        for i in range(10000):
            bf.add(f"item_{i}")
        fp_rate = sum(1 for i in range(10000, 11000) if bf.contains(f"item_{i}")) / 1000
        assert fp_rate < 0.05

    def test_memory_metrics(self):
        bf = BloomFilter(expected_elements=10000, false_positive_rate=0.01)
        stats = bf.get_statistics()
        for metric in ["size_mb", "hash_count", "item_count", "bits_set", "fill_rate"]:
            assert metric in stats
        assert stats["size_mb"] > 0


class TestChunkStatisticsBasic(BaseTestCase):
    def test_preemption_handling(self):
        stats_client, _ = self.setup_stats_client()
        token_ids = list(range(256))
        stats_client.lookup(token_ids, "req_1")
        stats1 = stats_client.get_statistics()
        stats_client.lookup(token_ids, "req_1")
        stats2 = stats_client.get_statistics()
        assert stats1["total_requests"] == stats2["total_requests"]
        assert stats1["total_chunks"] == stats2["total_chunks"]

    def test_disabled_statistics(self):
        stats_client = self.create_stats_client()
        stats_client.lookup(list(range(256)), "req_1")
        stats = stats_client.get_statistics()
        assert stats["enabled"] is False and stats["total_requests"] == 0


class TestChunkStatisticsMetrics(BaseTestCase):
    def test_detailed_metrics(self):
        stats_client, _ = self.setup_stats_client(
            extra_config={
                "chunk_statistics_mem_bf_expected_chunks": 5000,
                "chunk_statistics_mem_bf_false_positive_rate": 0.01,
            }
        )
        stats_client.lookup(list(range(512)), "req_1")
        stats_client.lookup(list(range(256)), "req_2")
        stats = stats_client.get_statistics()
        for stat in [
            "enabled",
            "total_requests",
            "total_chunks",
            "unique_chunks",
            "duplicate_chunks",
            "reuse_rate",
            "bloom_filter",
            "timing",
        ]:
            assert stat in stats
        bf_stats = stats["bloom_filter"]
        for stat in [
            "size_mb",
            "hash_count",
            "item_count",
            "bits_set",
            "fill_rate",
            "expected_elements",
            "false_positive_rate",
        ]:
            assert stat in bf_stats
        assert stats["total_requests"] == 2 and stats["total_chunks"] == 3
        assert 0.0 <= stats["reuse_rate"] <= 1.0
        assert (
            bf_stats["expected_elements"] == 5000
            and bf_stats["false_positive_rate"] == 0.01
        )
        timing = stats["timing"]
        for field in [
            "lookup_time_seconds",
            "record_statistics_time_seconds",
            "check_exit_conditions_time_seconds",
            "total_time_seconds",
            "overhead_time_seconds",
            "overhead_percentage",
        ]:
            assert field in timing and timing[field] >= 0
        assert 0 <= timing["overhead_percentage"] <= 100

    def test_progressive_metrics(self):
        stats_client, _ = self.setup_stats_client()
        stats_client.lookup(list(range(256)), "req_1")
        assert stats_client.get_statistics()["total_requests"] == 1
        stats_client.lookup(list(range(256, 512)), "req_2")
        assert stats_client.get_statistics()["total_requests"] == 2
        stats_client.lookup(torch.arange(256), "req_3")
        stats3 = stats_client.get_statistics()
        assert stats3["total_requests"] == 3 and stats3["duplicate_chunks"] > 0

    def test_memory_efficiency(self):
        stats_client, _ = self.setup_stats_client(
            extra_config={
                "chunk_statistics_mem_bf_expected_chunks": 100000,
                "chunk_statistics_mem_bf_false_positive_rate": 0.01,
            }
        )
        for i in range(100):
            stats_client.lookup(list(range(i * 256, (i + 1) * 256)), f"req_{i}")
        stats = stats_client.get_statistics()
        assert stats["bloom_filter"]["size_mb"] < 1.0
        assert stats["total_requests"] == 100 and stats["total_chunks"] == 100


class TestChunkStatisticsLifecycle(BaseTestCase):
    def test_reset_statistics(self):
        stats_client, _ = self.setup_stats_client()
        stats_client.lookup(list(range(256)), "req_1")
        stats_client.reset_statistics()
        stats = stats_client.get_statistics()
        assert (
            stats["total_requests"] == 0
            and stats["total_chunks"] == 0
            and stats["unique_chunks"] == 0
        )

    def test_auto_exit_configuration(self):
        stats_client = self.create_stats_client(
            enable_chunk_statistics=True,
            chunk_statistics_auto_start_statistics=True,
            chunk_statistics_auto_exit_timeout_hours=1.0,
        )
        stats = stats_client.get_statistics()
        assert stats["enabled"] is True and stats["total_requests"] == 0
        assert (
            stats_client.enable_auto_exit is True and stats_client.timeout_hours == 1.0
        )
        stats_client2 = self.create_stats_client(
            enable_chunk_statistics=True, chunk_statistics_auto_exit_timeout_hours=0.0
        )
        assert stats_client2.enable_auto_exit is False


class TestChunkStatisticsPerformance:
    """Test suite for chunk statistics performance validation."""

    @pytest.mark.parametrize(
        "strategy_name,async_preprocess_chunks",
        [
            ("memory_bloom_filter", True),
            ("memory_bloom_filter", False),
            ("file_hash", True),
            ("file_hash", False),
        ],
    )
    def test_worst_case_overhead(self, strategy_name, async_preprocess_chunks):
        """Test worst case performance with different strategies and configs."""
        self._run_performance_test(
            strategy_name=strategy_name,
            async_preprocess_chunks=async_preprocess_chunks,
        )

    def _run_performance_test(self, strategy_name: str, async_preprocess_chunks: bool):
        if async_preprocess_chunks:
            mode_desc = f"{strategy_name} (Async Preprocessed)"
        else:
            mode_desc = f"{strategy_name} (Async Raw Tokens)"
        temp_dir: Optional[str] = (
            tempfile.mkdtemp(prefix="lmcache_test_")
            if strategy_name == "file_hash"
            else None
        )
        try:
            extra_config: dict[str, Any] = {
                "chunk_statistics_mem_bf_expected_chunks": 100000,
                "chunk_statistics_mem_bf_false_positive_rate": 0.01,
                "chunk_statistics_async_preprocess_chunks": async_preprocess_chunks,
            }
            if temp_dir:
                extra_config["chunk_statistics_file_output_dir"] = temp_dir
            config_kwargs = {
                "chunk_size": 256,
                "chunk_statistics_strategy": strategy_name,
                "extra_config": extra_config,
            }
            config = LMCacheEngineConfig(**config_kwargs)
            stats_client = ChunkStatisticsLookupClient(FastMissLookupClient(), config)
            stats_client.start_statistics()
            token_count, num_requests = 32 * 1024, 30
            token_ids = list(range(token_count))
            stats_client.lookup(token_ids, "warmup")
            stats_client.reset_statistics()
            stats_client.start_statistics()
            for i in range(num_requests):
                stats_client.lookup(token_ids, f"req_{i}")
            assert stats_client.wait_for_async_processing(timeout=10.0), "Async timeout"
            stats = stats_client.get_statistics()
            timing = stats["timing"]

            overhead_percentage = timing["overhead_percentage"]
            print(f"Overhead percentage({mode_desc}): {overhead_percentage:.2f}%")
            assert stats["total_requests"] == num_requests
            assert stats["total_chunks"] == num_requests * (token_count // 256)
            assert timing["overhead_percentage"] <= 40.0, (
                f"Overhead {timing['overhead_percentage']:.2f}% > 40%"
            )
            assert timing["record_statistics_time_seconds"] / num_requests < 0.01, (
                "Avg record time > 10ms"
            )
            async_queue = stats.get("async_queue", {})
            if async_queue:
                assert async_queue.get("capacity") == config.get_extra_config_value(
                    "chunk_statistics_async_queue_capacity", 100000
                )
                assert async_queue.get("full_blocks", 0) == 0
        finally:
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass


class TestFileHashStrategy:
    def test_file_hash_basic(self):
        # Standard
        from pathlib import Path
        import json

        temp_dir = tempfile.mkdtemp()
        try:
            config = LMCacheEngineConfig.from_dict(
                {
                    "chunk_statistics_enabled": True,
                    "chunk_statistics_strategy": "file_hash",
                    "extra_config": {
                        "chunk_statistics_file_output_dir": temp_dir,
                    },
                }
            )
            client = ChunkStatisticsLookupClient(MockLookupClient(), config)
            client.start_statistics()
            client.lookup(list(range(512)), "test_request_1")
            client.wait_for_async_processing(timeout=2.0)
            output_files = list(Path(temp_dir).glob("*.jsonl"))
            assert len(output_files) > 0
            with open(output_files[0], "r") as f:
                data = json.loads(f.readline())
                assert (
                    "chunk_hashes" in data
                    and "lookup_id" in data
                    and "timestamp" in data
                )
                assert len(data["chunk_hashes"]) == 2
            client.close()
        finally:
            shutil.rmtree(temp_dir)
