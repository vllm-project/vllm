# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV cache offloading configuration."""

import pytest

from vllm.config.cache import CacheConfig
from vllm.config.kv_offloading import apply_kv_offloading_config
from vllm.config.kv_transfer import KVTransferConfig
from vllm.config.parallel import ParallelConfig


def test_apply_kv_offloading_native():
    """Test applying native offloading configuration."""
    # Create configs
    cache_config = CacheConfig(
        block_size=16,
        kv_offloading_size=10.0,
        kv_offloading_backend="native",
    )
    kv_transfer_config = KVTransferConfig()
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=2,
        data_parallel_size=1,
    )

    # Apply offloading config
    apply_kv_offloading_config(
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        parallel_config=parallel_config,
    )

    # Verify configuration
    assert kv_transfer_config.kv_connector == "OffloadingConnector"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_transfer_config.kv_connector_extra_config is not None
    assert "kv_bytes_per_rank" in kv_transfer_config.kv_connector_extra_config

    kv_bytes_per_rank = kv_transfer_config.kv_connector_extra_config[
        "kv_bytes_per_rank"
    ]
    assert kv_bytes_per_rank == 10.0 * (1 << 30) / 2


def test_no_offloading_when_size_is_none():
    """Test that no offloading is applied when size is None."""
    cache_config = CacheConfig(
        block_size=16,
        kv_offloading_size=None,
        kv_offloading_backend=None,
    )
    kv_transfer_config = KVTransferConfig()
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    # Store original values
    original_connector = kv_transfer_config.kv_connector

    # Apply offloading config
    apply_kv_offloading_config(
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        parallel_config=parallel_config,
    )

    # Verify nothing changed
    assert kv_transfer_config.kv_connector == original_connector


def test_error_when_backend_missing():
    """Test error when backend is not specified."""
    cache_config = CacheConfig(
        block_size=16,
        kv_offloading_size=10.0,
        kv_offloading_backend=None,  # Missing backend
    )
    kv_transfer_config = KVTransferConfig()
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    with pytest.raises(ValueError, match="kv_offloading_backend must be specified"):
        apply_kv_offloading_config(
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
            parallel_config=parallel_config,
        )


def test_error_when_backend_unknown():
    """Test error when backend is not registered."""
    cache_config = CacheConfig(
        block_size=16,
        kv_offloading_size=10.0,
        kv_offloading_backend="nonexistent_backend",
    )
    kv_transfer_config = KVTransferConfig()
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    with pytest.raises(ValueError, match="Unknown offloading backend"):
        apply_kv_offloading_config(
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
            parallel_config=parallel_config,
        )


def test_error_when_size_negative():
    """Test error when offloading size is negative."""
    cache_config = CacheConfig(
        block_size=16,
        kv_offloading_size=-5.0,
        kv_offloading_backend="native",
    )
    kv_transfer_config = KVTransferConfig()
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    with pytest.raises(ValueError, match="kv_offloading_size must be positive"):
        apply_kv_offloading_config(
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
            parallel_config=parallel_config,
        )


def test_multiple_tp_ranks():
    """Test buffer size calculation with multiple TP ranks."""
    cache_config = CacheConfig(
        block_size=16,
        kv_offloading_size=40.0,
        kv_offloading_backend="native",
    )
    kv_transfer_config = KVTransferConfig()
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=8,
        data_parallel_size=1,
    )

    # Apply offloading config
    apply_kv_offloading_config(
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        parallel_config=parallel_config,
    )

    # Verify configuration
    assert kv_transfer_config.kv_connector == "OffloadingConnector"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_transfer_config.kv_connector_extra_config is not None
    assert "kv_bytes_per_rank" in kv_transfer_config.kv_connector_extra_config

    kv_bytes_per_rank = kv_transfer_config.kv_connector_extra_config[
        "kv_bytes_per_rank"
    ]
    assert kv_bytes_per_rank == 40.0 * (1 << 30) / 8
