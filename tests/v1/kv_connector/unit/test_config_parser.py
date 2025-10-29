# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV cache offloading configuration."""

import pytest

from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.config_parser import (
    LMCacheOffloadingParser,
    NativeOffloadingParser,
    apply_extra_kv_connector_config,
    get_connector_config_parser,
)


class TestGetConnectorConfigParser:
    """Tests for get_connector_config_parser function."""

    def test_returns_native_parser(self):
        """Test that 'native' backend returns NativeOffloadingParser."""
        cache_config = CacheConfig(
            kv_offloading_backend="native", kv_offloading_size=4.0
        )
        vllm_config = VllmConfig(cache_config=cache_config)

        parser = get_connector_config_parser(vllm_config)

        assert parser is not None
        assert isinstance(parser, NativeOffloadingParser)

    def test_returns_lmcache_parser(self):
        """Test that 'lmcache' backend returns LMCacheOffloadingParser."""
        cache_config = CacheConfig(
            kv_offloading_backend="lmcache", kv_offloading_size=4.0
        )
        vllm_config = VllmConfig(cache_config=cache_config)

        parser = get_connector_config_parser(vllm_config)

        assert parser is not None
        assert isinstance(parser, LMCacheOffloadingParser)

    def test_returns_none_when_no_backend(self):
        """Test that None backend returns None."""
        cache_config = CacheConfig(kv_offloading_backend=None)
        vllm_config = VllmConfig(cache_config=cache_config)

        parser = get_connector_config_parser(vllm_config)

        assert parser is None

    def test_raises_error_for_unknown_backend(self):
        """Test that unknown backend raises ValueError."""
        cache_config = CacheConfig(
            kv_offloading_backend="unknown_backend", kv_offloading_size=4.0
        )
        vllm_config = VllmConfig(cache_config=cache_config)

        with pytest.raises(ValueError) as exc_info:
            get_connector_config_parser(vllm_config)

        assert "Unknown offloading backend: 'unknown_backend'" in str(exc_info.value)
        assert "native" in str(exc_info.value)
        assert "lmcache" in str(exc_info.value)


class TestNativeOffloadingParser:
    """Tests for NativeOffloadingParser class."""

    def test_configure_kv_transfer_single_rank(self):
        """Test native parser configuration with single rank."""
        # Setup
        cache_config = CacheConfig(
            kv_offloading_backend="native", kv_offloading_size=4.0
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        )
        vllm_config = VllmConfig(
            cache_config=cache_config, parallel_config=parallel_config
        )
        kv_transfer_config = KVTransferConfig()
        parser = NativeOffloadingParser()

        # Execute
        parser.configure_kv_transfer(kv_transfer_config, vllm_config)

        # Verify public attributes
        assert kv_transfer_config.kv_connector == "OffloadingConnector"
        assert kv_transfer_config.kv_role == "kv_both"
        assert kv_transfer_config.kv_connector_extra_config is not None
        assert "kv_bytes_per_rank" in kv_transfer_config.kv_connector_extra_config
        assert "num_cpu_blocks" in kv_transfer_config.kv_connector_extra_config

        # Verify expected kv_bytes_per_rank calculation
        # 4.0 GiB = 4.0 * (1 << 30) bytes, divided by 1 rank
        expected_bytes = 4.0 * (1 << 30)
        assert (
            kv_transfer_config.kv_connector_extra_config["kv_bytes_per_rank"]
            == expected_bytes
        )
        assert kv_transfer_config.kv_connector_extra_config["num_cpu_blocks"] == 0

    def test_configure_kv_transfer_multiple_ranks(self):
        """Test native parser configuration with multiple ranks."""
        # Setup - 2x2 tensor/pipeline parallel
        cache_config = CacheConfig(
            kv_offloading_backend="native", kv_offloading_size=8.0
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=2, pipeline_parallel_size=2
        )
        vllm_config = VllmConfig(
            cache_config=cache_config, parallel_config=parallel_config
        )
        kv_transfer_config = KVTransferConfig()
        parser = NativeOffloadingParser()

        # Execute
        parser.configure_kv_transfer(kv_transfer_config, vllm_config)

        # Verify
        assert kv_transfer_config.kv_connector == "OffloadingConnector"
        assert kv_transfer_config.kv_role == "kv_both"

        # Verify bytes per rank: 8.0 GiB / (2 * 2) = 2.0 GiB per rank
        expected_bytes = 8.0 * (1 << 30) / 4
        assert (
            kv_transfer_config.kv_connector_extra_config["kv_bytes_per_rank"]
            == expected_bytes
        )

    def test_configure_kv_transfer_preserves_existing_extra_config(self):
        """Test that native parser preserves existing extra config."""
        # Setup
        cache_config = CacheConfig(
            kv_offloading_backend="native", kv_offloading_size=4.0
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        )
        vllm_config = VllmConfig(
            cache_config=cache_config, parallel_config=parallel_config
        )
        kv_transfer_config = KVTransferConfig()
        kv_transfer_config.kv_connector_extra_config = {
            "existing_key": "existing_value"
        }
        parser = NativeOffloadingParser()

        # Execute
        parser.configure_kv_transfer(kv_transfer_config, vllm_config)

        # Verify that existing config is preserved and new config is added
        assert (
            kv_transfer_config.kv_connector_extra_config["existing_key"]
            == "existing_value"
        )
        assert "kv_bytes_per_rank" in kv_transfer_config.kv_connector_extra_config


class TestLMCacheOffloadingParser:
    """Tests for LMCacheOffloadingParser class."""

    def test_configure_kv_transfer_single_rank(self):
        """Test LMCache parser configuration with single rank."""
        # Setup
        cache_config = CacheConfig(
            kv_offloading_backend="lmcache", kv_offloading_size=4.0
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        )
        vllm_config = VllmConfig(
            cache_config=cache_config, parallel_config=parallel_config
        )
        kv_transfer_config = KVTransferConfig()
        parser = LMCacheOffloadingParser()

        # Execute
        parser.configure_kv_transfer(kv_transfer_config, vllm_config)

        # Verify public attributes
        assert kv_transfer_config.kv_connector == "LMCacheConnectorV1"
        assert kv_transfer_config.kv_role == "kv_both"
        assert kv_transfer_config.kv_connector_extra_config is not None
        assert "lmcache.local_cpu" in kv_transfer_config.kv_connector_extra_config
        assert (
            "lmcache.max_local_cpu_size" in kv_transfer_config.kv_connector_extra_config
        )

        # Verify expected values
        assert kv_transfer_config.kv_connector_extra_config["lmcache.local_cpu"] is True
        # 4.0 GiB / 1 rank = 4.0
        assert (
            kv_transfer_config.kv_connector_extra_config["lmcache.max_local_cpu_size"]
            == 4.0
        )

    def test_configure_kv_transfer_multiple_ranks(self):
        """Test LMCache parser configuration with multiple ranks."""
        # Setup - 2x2 tensor/pipeline parallel
        cache_config = CacheConfig(
            kv_offloading_backend="lmcache", kv_offloading_size=8.0
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=2, pipeline_parallel_size=2
        )
        vllm_config = VllmConfig(
            cache_config=cache_config, parallel_config=parallel_config
        )
        kv_transfer_config = KVTransferConfig()
        parser = LMCacheOffloadingParser()

        # Execute
        parser.configure_kv_transfer(kv_transfer_config, vllm_config)

        # Verify
        assert kv_transfer_config.kv_connector == "LMCacheConnectorV1"
        assert kv_transfer_config.kv_role == "kv_both"

        # Verify size per rank: 8.0 GiB / (2 * 2) = 2.0 GiB per rank
        assert (
            kv_transfer_config.kv_connector_extra_config["lmcache.max_local_cpu_size"]
            == 2.0
        )

    def test_configure_kv_transfer_replaces_extra_config(self):
        """Test that LMCache parser replaces the entire extra config."""
        # Setup
        cache_config = CacheConfig(
            kv_offloading_backend="lmcache", kv_offloading_size=4.0
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        )
        vllm_config = VllmConfig(
            cache_config=cache_config, parallel_config=parallel_config
        )
        kv_transfer_config = KVTransferConfig()
        kv_transfer_config.kv_connector_extra_config = {
            "existing_key": "existing_value"
        }
        parser = LMCacheOffloadingParser()

        # Execute
        parser.configure_kv_transfer(kv_transfer_config, vllm_config)

        # Verify that extra config is replaced (not merged) for LMCache
        assert "existing_key" not in kv_transfer_config.kv_connector_extra_config
        assert "lmcache.local_cpu" in kv_transfer_config.kv_connector_extra_config


class TestApplyKVConnectorConfig:
    """Tests for apply_extra_kv_connector_config function."""

    def test_apply_native_backend(self):
        """Test applying native backend configuration."""
        # Setup
        cache_config = CacheConfig(
            kv_offloading_backend="native", kv_offloading_size=4.0
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        )
        vllm_config = VllmConfig(
            cache_config=cache_config, parallel_config=parallel_config
        )
        kv_transfer_config = KVTransferConfig()

        # Execute
        apply_extra_kv_connector_config(vllm_config, kv_transfer_config)

        # Verify
        assert kv_transfer_config.kv_connector == "OffloadingConnector"
        assert kv_transfer_config.kv_role == "kv_both"
        assert kv_transfer_config.kv_connector_extra_config is not None

    def test_apply_lmcache_backend(self):
        """Test applying LMCache backend configuration."""
        # Setup
        cache_config = CacheConfig(
            kv_offloading_backend="lmcache", kv_offloading_size=4.0
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1
        )
        vllm_config = VllmConfig(
            cache_config=cache_config, parallel_config=parallel_config
        )
        kv_transfer_config = KVTransferConfig()

        # Execute
        apply_extra_kv_connector_config(vllm_config, kv_transfer_config)

        # Verify
        assert kv_transfer_config.kv_connector == "LMCacheConnectorV1"
        assert kv_transfer_config.kv_role == "kv_both"
        assert kv_transfer_config.kv_connector_extra_config is not None

    def test_no_configuration_when_backend_is_none(self):
        """Test that no configuration is applied when backend is None."""
        # Setup
        cache_config = CacheConfig(kv_offloading_backend=None)
        vllm_config = VllmConfig(cache_config=cache_config)
        kv_transfer_config = KVTransferConfig()

        # Execute
        apply_extra_kv_connector_config(vllm_config, kv_transfer_config)

        # Verify that config remains unchanged
        assert kv_transfer_config.kv_connector is None
        assert kv_transfer_config.kv_role is None

    def test_apply_with_different_parallel_configurations(self):
        """Test applying config with various parallelism settings."""
        test_cases = [
            # (tp_size, pp_size, total_size_gib, expected_bytes_per_rank)
            (1, 1, 4.0, 4.0 * (1 << 30)),
            (2, 1, 4.0, 2.0 * (1 << 30)),
            (1, 2, 4.0, 2.0 * (1 << 30)),
            (2, 2, 8.0, 2.0 * (1 << 30)),
            (4, 2, 16.0, 2.0 * (1 << 30)),
        ]

        for tp_size, pp_size, total_size_gib, expected_bytes in test_cases:
            # Setup
            cache_config = CacheConfig(
                kv_offloading_backend="native", kv_offloading_size=total_size_gib
            )
            parallel_config = ParallelConfig(
                tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size
            )
            vllm_config = VllmConfig(
                cache_config=cache_config, parallel_config=parallel_config
            )
            kv_transfer_config = KVTransferConfig()

            # Execute
            apply_extra_kv_connector_config(vllm_config, kv_transfer_config)

            # Verify
            actual_bytes = kv_transfer_config.kv_connector_extra_config[
                "kv_bytes_per_rank"
            ]
            assert actual_bytes == expected_bytes, (
                f"Failed for tp={tp_size}, pp={pp_size}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
