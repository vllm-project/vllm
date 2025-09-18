# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import Mock, patch

import pytest

from vllm.config import CacheConfig
from vllm.engine.arg_utils import EngineArgs

# Using common test model
TEST_MODEL = "facebook/opt-125m"


def test_flag_requires_kv_cache_bytes():
    """Test that the flag requires kv_cache_memory_bytes to be set."""

    with pytest.raises(
            ValueError,
            match="--enable-pp-prop-kv-cache requires --kv-cache-memory-bytes"
    ):
        args = EngineArgs(
            model=TEST_MODEL,
            enable_pp_prop_kv_cache=True,
            # kv_cache_memory_bytes not set!
        )
        _ = args.create_engine_config()

    # Test that it works when both flags are provided
    args = EngineArgs(
        model=TEST_MODEL,
        enable_pp_prop_kv_cache=True,
        kv_cache_memory_bytes=1 << 30,
    )
    config = args.create_engine_config()
    assert config.cache_config.enable_pp_prop_kv_cache is True
    assert config.cache_config.kv_cache_memory_bytes == 1 << 30


def test_backward_compatibility():
    """Test that default behavior is unchanged."""
    cache_config = CacheConfig(gpu_memory_utilization=0.9, )
    assert cache_config.enable_pp_prop_kv_cache is False

    # Test with explicit kv_cache_memory_bytes but flag off
    cache_config = CacheConfig(
        kv_cache_memory_bytes=1 << 30,
        enable_pp_prop_kv_cache=False,  # disabled flag!
    )
    assert cache_config.enable_pp_prop_kv_cache is False
    assert cache_config.kv_cache_memory_bytes == 1 << 30

    args = EngineArgs(
        model=TEST_MODEL,
        kv_cache_memory_bytes=1 << 30,
    )
    config = args.create_engine_config()
    assert config.cache_config.enable_pp_prop_kv_cache is False
    assert config.cache_config.kv_cache_memory_bytes == 1 << 30


def test_flag_off_means_no_proportional_distribution():
    """Test that proportional distribution DOES NOT happen when flag is off."""
    from vllm.v1.worker.gpu_worker import Worker as V1Worker

    # Even with VLLM_PP_LAYER_PARTITION set
    with patch.dict(os.environ, {'VLLM_PP_LAYER_PARTITION': '56,8'}):
        # Create real config with flag OFF
        args = EngineArgs(
            model=TEST_MODEL,
            enable_pp_prop_kv_cache=False,  # disabled flag!
            kv_cache_memory_bytes=8 << 30,  # 8GB total
            pipeline_parallel_size=2,
        )
        vllm_config = args.create_engine_config()

        with patch('vllm.v1.worker.gpu_worker.get_pp_group') as mock_pp_group:
            # Mock PP group for 2 GPUs
            pp_group = Mock()
            pp_group.world_size = 2
            pp_group.rank_in_group = 0
            mock_pp_group.return_value = pp_group

            # Create real Worker with real config
            worker = V1Worker(vllm_config=vllm_config,
                              local_rank=0,
                              rank=0,
                              distributed_init_method=None)

            # Mock only what's necessary to avoid GPU operations
            worker.model_runner = Mock()
            worker.model_runner.profile_run = Mock()
            worker.init_snapshot = Mock(
                free_memory=10 << 30)  # For logging only

            # Should return FULL 8GB, NOT proportional
            result = worker.determine_available_memory()
            assert result == 8 << 30, \
                f"Expected 8GB, got {result/(1<<30):.2f}GB"


def test_proportional_distribution_calculation():
    """Test that memory splits proportionally based on layer distribution."""
    from vllm.v1.worker.gpu_worker import Worker as V1Worker

    # Your real setup: 56 layers on GPU1, 8 layers on GPU0
    with patch.dict(os.environ, {'VLLM_PP_LAYER_PARTITION': '56,8'}):
        # Create real config with flag ON
        args = EngineArgs(
            model=TEST_MODEL,
            enable_pp_prop_kv_cache=True,  # flag enabled!
            kv_cache_memory_bytes=8 << 30,  # 8GB total
            pipeline_parallel_size=2,
        )
        vllm_config = args.create_engine_config()

        with (patch('vllm.v1.worker.gpu_worker.get_pp_group') as
              mock_pp_group, patch('vllm.v1.worker.gpu_worker.get_pp_indices')
              as mock_pp_indices):

            # Mock PP group for 2 GPUs
            pp_group = Mock()
            pp_group.world_size = 2
            mock_pp_group.return_value = pp_group

            # Mock layer distribution: GPU0 gets 0-56, GPU1 gets 56-64
            mock_pp_indices.side_effect = lambda total, rank, size: (
                (0, 56) if rank == 0 else (56, 64))

            # Test GPU0 with 56 layers
            pp_group.rank_in_group = 0

            worker = V1Worker(vllm_config=vllm_config,
                              local_rank=0,
                              rank=0,
                              distributed_init_method=None)

            # Mock only what's necessary
            worker.model_runner = Mock()
            worker.model_runner.profile_run = Mock()
            worker.init_snapshot = Mock(
                free_memory=10 << 30)  # For logging only

            # Calculate memory for GPU0 (56 layers)
            result = worker.determine_available_memory()

            # Should be 8GB * (56/64) = 7GB
            expected = int((8 << 30) * 56 / 64)
            exp_gb = expected / (1 << 30)
            res_gb = result / (1 << 30)
            assert result == expected, \
                f"GPU0: Expected {exp_gb:.2f}GB, got {res_gb:.2f}GB"

            # Test GPU1 with 8 layers
            pp_group.rank_in_group = 1

            # Need to create a new worker for different rank
            worker2 = V1Worker(vllm_config=vllm_config,
                               local_rank=0,
                               rank=1,
                               distributed_init_method=None)
            worker2.model_runner = Mock()
            worker2.model_runner.profile_run = Mock()
            worker2.init_snapshot = Mock(free_memory=10 << 30)

            result = worker2.determine_available_memory()

            # Should be 8GB * (8/64) = 1GB
            expected = int((8 << 30) * 8 / 64)
            exp_gb = expected / (1 << 30)
            res_gb = result / (1 << 30)
            assert result == expected, \
                f"GPU1: Expected {exp_gb:.2f}GB, got {res_gb:.2f}GB"
