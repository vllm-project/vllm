# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import (check_enough_kv_cache_memory,
                                         get_kv_cache_configs,
                                         is_kv_cache_type_uniform)
from vllm.v1.kv_cache_interface import FullAttentionSpec


class TestKVCacheErrorHandlingFunctional:
    """Functional tests for KV cache error handling without using mocks."""

    def setup_method(self):
        """Setup common test configuration."""
        # Create actual KV cache specs without requiring model loading
        self.block_size = 16

        # Create a uniform KV cache spec
        self.uniform_spec = self._create_uniform_spec(
            2)  # 2 layers with same specs

        # Create a non-uniform KV cache spec
        self.non_uniform_spec = self._create_non_uniform_spec()

        # Create a basic cache config with required parameters
        self.cache_config = CacheConfig(block_size=self.block_size,
                                        gpu_memory_utilization=0.8,
                                        swap_space=1.0,
                                        cache_dtype="auto")
        # Add the min_required_gpu_blocks attribute since it's not a parameter
        self.cache_config.min_required_gpu_blocks = 100

    def _create_uniform_spec(self, num_layers):
        """Create uniform KV cache spec with the specified number of layers."""
        kv_cache_spec = {}
        for i in range(num_layers):
            kv_cache_spec[f"layer{i}"] = FullAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.float16,
                use_mla=False)
        return kv_cache_spec

    def _create_non_uniform_spec(self):
        """Create a non-uniform KV cache spec."""
        return {
            "layer0":
            FullAttentionSpec(block_size=self.block_size,
                              num_kv_heads=8,
                              head_size=128,
                              dtype=torch.float16,
                              use_mla=False),
            "layer1":
            FullAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.float16,
                use_mla=True  # Different type - MLA vs non-MLA
            )
        }

    def test_check_enough_kv_cache_memory_no_memory(self, monkeypatch):
        """Test error handling when no memory is available."""
        # Create a minimal config with just what's needed for the test
        from vllm.config import ModelConfig, SchedulerConfig, VllmConfig

        model_config = ModelConfig.__new__(ModelConfig)
        model_config.max_model_len = 2048

        scheduler_config = SchedulerConfig.__new__(SchedulerConfig)
        scheduler_config.gpu_memory_utilization = 0.9

        vllm_config = VllmConfig.__new__(VllmConfig)
        vllm_config.model_config = model_config
        vllm_config.scheduler_config = scheduler_config
        vllm_config.cache_config = self.cache_config

        with pytest.raises(ValueError) as excinfo:
            check_enough_kv_cache_memory(vllm_config,
                                         self.uniform_spec,
                                         available_memory=0)

        # Check error message contains helpful information
        error_msg = str(excinfo.value)
        assert "No available memory for the KV cache blocks" in error_msg
        assert "Try one of the following" in error_msg
        assert "Increase `gpu_memory_utilization`" in error_msg
        assert "Use a smaller model" in error_msg
        assert "Use a GPU with more memory" in error_msg
        assert "Reduce model_max_len" in error_msg

    def test_check_enough_kv_cache_memory_insufficient_memory(
            self, monkeypatch):
        """Test error handling if insufficient memory for KV cache."""
        # Create a minimal config with just what's needed for the test
        from vllm.config import ModelConfig, SchedulerConfig, VllmConfig

        model_config = ModelConfig.__new__(ModelConfig)
        model_config.max_model_len = 2048

        scheduler_config = SchedulerConfig.__new__(SchedulerConfig)
        scheduler_config.gpu_memory_utilization = 0.9

        vllm_config = VllmConfig.__new__(VllmConfig)
        vllm_config.model_config = model_config
        vllm_config.scheduler_config = scheduler_config
        vllm_config.cache_config = self.cache_config

        # Calculate needed memory based on the spec
        # For non-MLA, the page size is
        # 2 * block_size * num_kv_heads * head_size * dtype_size
        # With block_size=16, num_kv_heads=8, head_size=128, dtype=float16
        # Page size = 2 * 16 * 8 * 128 * 2 = 65,536 bytes per page per layer
        # With 2 layers: 2 * 65,536 = 131,072 bytes per page
        # For max_model_len=2048, we need ceil(2048/16) = 128 pages
        # Total = 128 * 131,072 = 16,777,216 bytes (16 MB)

        # Provide less than needed: 8 MB
        available_memory = 8 * 1024 * 1024

        with pytest.raises(ValueError) as excinfo:
            check_enough_kv_cache_memory(vllm_config,
                                         self.uniform_spec,
                                         available_memory=available_memory)

        # Check error message contains helpful information
        error_msg = str(excinfo.value)
        assert "Insufficient memory for KV cache allocation" in error_msg
        assert "Required:" in error_msg
        assert "Available:" in error_msg
        assert "deficit:" in error_msg
        assert "Approximate max possible sequence length" in error_msg
        assert "Solutions:" in error_msg
        assert "Increase `gpu_memory_utilization`" in error_msg
        assert "Decrease `max_model_len`" in error_msg
        assert "Use a smaller model or a GPU with more memory" in error_msg

    def test_is_kv_cache_type_uniform(self):
        """Test is_kv_cache_type_uniform function with actual KV cache specs."""
        # Test with uniform spec
        assert is_kv_cache_type_uniform(self.uniform_spec) is True

        # Test with non-uniform spec
        assert is_kv_cache_type_uniform(self.non_uniform_spec) is False

    def test_get_kv_cache_configs_non_uniform_kv_cache(self, monkeypatch):
        """Test error handling with non-uniform KV cache types."""
        # Create a minimal config with just what's needed for the test
        from vllm.config import ModelConfig, SchedulerConfig, VllmConfig

        model_config = ModelConfig.__new__(ModelConfig)
        model_config.max_model_len = 2048

        scheduler_config = SchedulerConfig.__new__(SchedulerConfig)
        scheduler_config.gpu_memory_utilization = 0.9

        vllm_config = VllmConfig.__new__(VllmConfig)
        vllm_config.model_config = model_config
        vllm_config.scheduler_config = scheduler_config
        vllm_config.cache_config = self.cache_config

        with pytest.raises(NotImplementedError) as excinfo:
            get_kv_cache_configs(
                vllm_config,
                [self.non_uniform_spec],
                available_memory=1024 * 1024 * 1024  # 1 GB (plenty of memory)
            )

        assert "Models with non-uniform KV cache types not supported" in str(
            excinfo.value)
