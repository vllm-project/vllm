"""
Test suite for TPU v6e Adaptive Attention Backend

This test suite validates the TPU v6e architecture-adaptive optimizations
including automatic architecture detection, MXU alignment, and performance
improvements.
"""
import os
import pytest
import torch

from vllm.v1.attention.backends.tpu_v6_adaptive_pallas import (
    TPUArchitectureDetector,
    TPUv6AdaptiveAttentionBackend,
    TPUv6AdaptiveAttentionBackendImpl,
    tpu_detector,
)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.config import VllmConfig, CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig


class TestTPUArchitectureDetector:
    """Test TPU architecture detection functionality"""

    def setup_method(self):
        """Set up test environment"""
        # Clean environment variables
        if 'TPU_VERSION' in os.environ:
            del os.environ['TPU_VERSION']

    def test_simulation_mode(self):
        """Test detection in simulation mode (no TPU)"""
        detector = TPUArchitectureDetector()
        assert detector.tpu_version == -1
        assert detector.is_simulated == True
        assert detector.config.version == 6  # Should default to v6 config
        assert detector.config.name == "TPU v6e (Trillium)"

    def test_v6_detection_via_env(self):
        """Test TPU v6 detection via environment variable"""
        os.environ['TPU_VERSION'] = '6'
        detector = TPUArchitectureDetector()
        assert detector.tpu_version == 6
        assert detector.config.version == 6
        assert detector.config.mxu_size == 256
        assert detector.config.name == "TPU v6e (Trillium)"

    def test_v5_detection_via_env(self):
        """Test TPU v5 detection via environment variable"""
        os.environ['TPU_VERSION'] = '5'
        detector = TPUArchitectureDetector()
        assert detector.tpu_version == 5
        assert detector.config.version == 5
        assert detector.config.mxu_size == 128
        assert detector.config.name == "TPU v5e"

    def test_head_dimension_optimization_v6(self):
        """Test head dimension optimization for v6"""
        os.environ['TPU_VERSION'] = '6'
        detector = TPUArchitectureDetector()

        # Test various head dimensions
        assert detector.optimize_head_dimension(128) == 256  # Pad up to 256
        assert detector.optimize_head_dimension(256) == 256  # Already aligned
        assert detector.optimize_head_dimension(100) == 256  # Pad up
        assert detector.optimize_head_dimension(
            300) == 512  # Pad up to next multiple

    def test_head_dimension_optimization_v5(self):
        """Test head dimension optimization for v5"""
        os.environ['TPU_VERSION'] = '5'
        detector = TPUArchitectureDetector()

        # Test various head dimensions
        assert detector.optimize_head_dimension(128) == 128  # Already aligned
        assert detector.optimize_head_dimension(100) == 128  # Pad up to 128
        assert detector.optimize_head_dimension(
            200) == 256  # Pad up to next multiple

    def test_attention_config_generation(self):
        """Test attention configuration generation"""
        os.environ['TPU_VERSION'] = '6'
        detector = TPUArchitectureDetector()

        config = detector.get_attention_config(2048)
        assert config["block_q"] <= 512  # Should not exceed optimal block size
        assert config["block_kv"] <= 1024
        assert config["memory_pipeline_stages"] == 4  # v6 has 4 stages
        assert config["mxu_size"] == 256
        assert config["is_v6_optimized"] == True


class TestTPUv6AdaptiveBackend:
    """Test TPU v6 adaptive backend functionality"""

    def setup_method(self):
        """Set up test environment"""
        os.environ['TPU_VERSION'] = '6'  # Force v6 for testing

    def test_backend_name(self):
        """Test backend naming"""
        assert TPUv6AdaptiveAttentionBackend.get_name(
        ) == "TPU_V6E_ADAPTIVE_PALLAS_VLLM_V1"

    def test_implementation_class(self):
        """Test implementation class registration"""
        impl_cls = TPUv6AdaptiveAttentionBackend.get_impl_cls()
        assert impl_cls == TPUv6AdaptiveAttentionBackendImpl

    def test_kv_cache_shape_v6(self):
        """Test KV cache shape calculation for v6"""
        shape = TPUv6AdaptiveAttentionBackend.get_kv_cache_shape(
            num_blocks=100, block_size=16, num_kv_heads=32, head_size=128)
        # Head size should be padded to 256 for v6
        expected_padded_head_size = 256
        expected_shape = (100, 16, 32 * 2, expected_padded_head_size)
        assert shape == expected_shape

    def test_page_size_optimization_v6(self):
        """Test page size optimization for v6"""

        # Mock vllm config
        class MockModelConfig:
            max_model_len = 4096

        class MockVllmConfig:
            model_config = MockModelConfig()

        config = MockVllmConfig()
        page_size = TPUv6AdaptiveAttentionBackend.get_page_size(config)
        # Should be larger than standard due to v6 optimizations
        assert page_size >= 32  # Minimum for v6

    def test_page_size_optimization_v5(self):
        """Test page size remains standard for v5"""
        os.environ['TPU_VERSION'] = '5'

        class MockModelConfig:
            max_model_len = 4096

        class MockVllmConfig:
            model_config = MockModelConfig()

        # Create new detector for v5
        from vllm.v1.attention.backends.tpu_v6_adaptive_pallas import TPUArchitectureDetector
        detector_v5 = TPUArchitectureDetector()
        assert detector_v5.config.version == 5


class TestTPUv6AdaptiveImplementation:
    """Test TPU v6 adaptive implementation functionality"""

    def setup_method(self):
        """Set up test environment"""
        os.environ['TPU_VERSION'] = '6'

    def test_initialization(self):
        """Test backend implementation initialization"""
        impl = TPUv6AdaptiveAttentionBackendImpl(
            num_heads=32,
            head_size=128,
            scale=0.125,
            num_kv_heads=32,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
        )

        assert impl.num_heads == 32
        assert impl.original_head_size == 128
        assert impl.head_size == 256  # Should be optimized for v6
        assert impl.scale == 0.125
        assert impl.call_count == 0
        assert impl.attention_config is not None

    def test_performance_tracking(self):
        """Test performance tracking functionality"""
        impl = TPUv6AdaptiveAttentionBackendImpl(
            num_heads=16,
            head_size=128,
            scale=0.125,
            num_kv_heads=16,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        # Test initial state
        report = impl.get_performance_report()
        assert report["backend"] == "TPUv6AdaptiveAttentionBackend"
        assert report["architecture"] == "TPU v6e (Trillium)"
        assert report["calls"] == 0
        assert report["mxu_size"] == "256x256"
        assert report["head_size_optimization"] == "128 -> 256"
        assert report["is_v6_optimized"] == True

    def test_applied_optimizations_v6(self):
        """Test applied optimizations for v6"""
        impl = TPUv6AdaptiveAttentionBackendImpl(
            num_heads=16,
            head_size=100,  # Will be optimized to 256
            scale=0.125,
            num_kv_heads=16,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        optimizations = impl._get_applied_optimizations()
        expected_optimizations = [
            "mxu_256x256_alignment", "4_stage_memory_pipeline",
            "enhanced_vmem_limits", "optimized_block_sizing",
            "head_dimension_padding"
        ]

        for opt in expected_optimizations:
            assert opt in optimizations

    def test_applied_optimizations_v5(self):
        """Test applied optimizations for v5"""
        os.environ['TPU_VERSION'] = '5'

        impl = TPUv6AdaptiveAttentionBackendImpl(
            num_heads=16,
            head_size=128,  # Already aligned for v5
            scale=0.125,
            num_kv_heads=16,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        optimizations = impl._get_applied_optimizations()
        expected_optimizations = [
            "mxu_128x128_alignment", "2_stage_memory_pipeline",
            "standard_block_sizing"
        ]

        for opt in expected_optimizations:
            assert opt in optimizations

        # Should not have head dimension padding since 128 is aligned for v5
        assert "head_dimension_padding" not in optimizations


class TestIntegration:
    """Test integration with vLLM components"""

    def test_global_detector_instance(self):
        """Test that global detector instance works correctly"""
        # Global detector should be accessible
        assert tpu_detector is not None
        assert hasattr(tpu_detector, 'config')
        assert hasattr(tpu_detector, 'tpu_version')

    def test_factory_function(self):
        """Test factory function for creating backends"""
        from vllm.v1.attention.backends.tpu_v6_adaptive_pallas import create_tpu_v6_adaptive_backend

        backend = create_tpu_v6_adaptive_backend(
            num_heads=16,
            head_size=128,
            scale=0.125,
            num_kv_heads=16,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )

        assert isinstance(backend, TPUv6AdaptiveAttentionBackendImpl)
        assert backend.original_head_size == 128

    def test_cross_version_compatibility(self):
        """Test compatibility across different TPU versions"""
        test_versions = ['4', '5', '6']

        for version in test_versions:
            os.environ['TPU_VERSION'] = version

            # Should not raise any errors
            detector = TPUArchitectureDetector()
            assert detector.config is not None

            impl = TPUv6AdaptiveAttentionBackendImpl(
                num_heads=16,
                head_size=128,
                scale=0.125,
                num_kv_heads=16,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="auto",
            )

            report = impl.get_performance_report()
            assert report["tpu_version"] == int(version)

    def teardown_method(self):
        """Clean up test environment"""
        if 'TPU_VERSION' in os.environ:
            del os.environ['TPU_VERSION']


if __name__ == "__main__":
    pytest.main([__file__])
