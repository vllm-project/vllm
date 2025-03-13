# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import current_platform
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)


# Unit tests for _initialize_kv_caches function
def test_initialize_kv_caches_no_gpu_memory():
    """Test handling of no available GPU memory."""
    model_executor = MagicMock()
    model_executor.get_kv_cache_specs.return_value = [
        KVCacheSpec(block_size=16, num_blocks=100)
    ]
    # No memory available
    model_executor.determine_available_memory.return_value = -10

    # Create a minimal config required for the test with mocked ModelConfig
    vllm_config = MagicMock()
    cache_config = MagicMock()
    cache_config.min_required_gpu_blocks = 16
    vllm_config.cache_config = cache_config

    engine_core = EngineCore.__new__(
        EngineCore)  # Create instance without calling __init__
    engine_core.model_executor = model_executor

    with pytest.raises(RuntimeError) as exc_info:
        engine_core._initialize_kv_caches(vllm_config)

    assert "Not enough GPU memory available to fit the model" in str(
        exc_info.value)


def test_initialize_kv_caches_zero_blocks():
    """Test handling of zero KV cache blocks allocation."""
    model_executor = MagicMock()
    model_executor.get_kv_cache_specs.return_value = [
        KVCacheSpec(block_size=16, num_blocks=100)
    ]

    # Some memory available
    model_executor.determine_available_memory.return_value = 1000

    # Mock the get_kv_cache_configs function to return an empty set of configs
    with patch('vllm.v1.engine.core.get_kv_cache_configs') as mock_get_configs:
        mock_get_configs.return_value = []  # No blocks allocated

        # Create a minimal config required for the test with mocked ModelConfig
        vllm_config = MagicMock()
        cache_config = MagicMock()
        cache_config.min_required_gpu_blocks = 16
        vllm_config.cache_config = cache_config

        engine_core = EngineCore.__new__(
            EngineCore)  # Create instance without calling __init__
        engine_core.model_executor = model_executor

        with pytest.raises(RuntimeError) as exc_info:
            engine_core._initialize_kv_caches(vllm_config)

        assert "Could not allocate any KV cache blocks" in str(exc_info.value)


def test_initialize_kv_caches_insufficient_blocks():
    """Test handling of insufficient KV cache blocks."""
    model_executor = MagicMock()
    model_executor.get_kv_cache_specs.return_value = [
        KVCacheSpec(block_size=16, num_blocks=100)
    ]

    # Some memory available
    model_executor.determine_available_memory.return_value = 1000

    # Create the KVCacheConfig with appropriate properties
    mock_kv_config = MagicMock(spec=KVCacheConfig)
    mock_kv_config.num_blocks = 10  # Fewer blocks than minimum required

    # Mock the get_kv_cache_configs to return configs with insufficient blocks
    with patch('vllm.v1.engine.core.get_kv_cache_configs') as mock_get_configs:
        mock_get_configs.return_value = [mock_kv_config]

        # Create a minimal config with minimum required blocks
        vllm_config = MagicMock()
        cache_config = MagicMock()
        cache_config.min_required_gpu_blocks = 20  #min required 20, but only 10
        vllm_config.cache_config = cache_config

        engine_core = EngineCore.__new__(
            EngineCore)  # Create instance without calling __init__
        engine_core.model_executor = model_executor

        with pytest.raises(RuntimeError) as exc_info:
            engine_core._initialize_kv_caches(vllm_config)

        assert "Could only allocate 10 KV cache blocks, " in str(
            exc_info.value)
        assert "but minimum required is 20" in str(exc_info.value)


def test_initialize_kv_caches_cuda_out_of_memory():
    """Test handling of CUDA out of memory during initialization."""
    model_executor = MagicMock()
    model_executor.get_kv_cache_specs.return_value = [
        KVCacheSpec(block_size=16, num_blocks=100)
    ]

    # Some memory available
    model_executor.determine_available_memory.return_value = 1000

    # Set up the model_executor to raise a CUDA OOM error during initialization
    cuda_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    model_executor.initialize_from_config.side_effect = cuda_error

    # Create the KVCacheConfig with appropriate properties
    mock_kv_config = MagicMock(spec=KVCacheConfig)
    mock_kv_config.num_blocks = 50

    # Mock the get_kv_cache_configs function to return valid configs
    with patch('vllm.v1.engine.core.get_kv_cache_configs') as mock_get_configs:
        mock_get_configs.return_value = [mock_kv_config]

        vllm_config = MagicMock()
        cache_config = MagicMock()
        cache_config.min_required_gpu_blocks = 16
        vllm_config.cache_config = cache_config

        engine_core = EngineCore.__new__(
            EngineCore)  # Create instance without calling __init__
        engine_core.model_executor = model_executor

        with pytest.raises(RuntimeError) as exc_info:
            engine_core._initialize_kv_caches(vllm_config)

        assert "CUDA out of memory during model initialization" in str(
            exc_info.value)


def test_initialize_kv_caches_operation_not_implemented():
    """Test 'operation not implemented' error during initialization."""
    model_executor = MagicMock()
    model_executor.get_kv_cache_specs.return_value = [
        KVCacheSpec(block_size=16, num_blocks=100)
    ]

    # Some memory available
    model_executor.determine_available_memory.return_value = 1000

    # Set up the model_executor to raise an 'operation not implemented' error
    not_implemented_error = RuntimeError("operation not implemented for dtype")
    model_executor.initialize_from_config.side_effect = not_implemented_error

    # Create the KVCacheConfig with appropriate properties
    mock_kv_config = MagicMock(spec=KVCacheConfig)
    mock_kv_config.num_blocks = 50

    # Mock the get_kv_cache_configs function to return valid configs
    with patch('vllm.v1.engine.core.get_kv_cache_configs') as mock_get_configs:
        mock_get_configs.return_value = [mock_kv_config]

        vllm_config = MagicMock()
        cache_config = MagicMock()
        cache_config.min_required_gpu_blocks = 16
        vllm_config.cache_config = cache_config

        engine_core = EngineCore.__new__(
            EngineCore)  # Create instance without calling __init__
        engine_core.model_executor = model_executor

        with pytest.raises(RuntimeError) as exc_info:
            engine_core._initialize_kv_caches(vllm_config)

        assert "Model operation not supported on this hardware" in str(
            exc_info.value)


def test_initialize_kv_caches_general_runtime_error():
    """Test handling of general runtime errors during initialization."""
    model_executor = MagicMock()
    model_executor.get_kv_cache_specs.return_value = [
        KVCacheSpec(block_size=16, num_blocks=100)
    ]

    # Some memory available
    model_executor.determine_available_memory.return_value = 1000

    # Set up the model_executor to raise a general error
    general_error = RuntimeError("Some unexpected error occurred")
    model_executor.initialize_from_config.side_effect = general_error

    # Create the KVCacheConfig with appropriate properties
    mock_kv_config = MagicMock(spec=KVCacheConfig)
    mock_kv_config.num_blocks = 50

    # Mock the get_kv_cache_configs function to return valid configs
    with patch('vllm.v1.engine.core.get_kv_cache_configs') as mock_get_configs:
        mock_get_configs.return_value = [mock_kv_config]

        vllm_config = MagicMock()
        cache_config = MagicMock()
        cache_config.min_required_gpu_blocks = 16
        vllm_config.cache_config = cache_config

        engine_core = EngineCore.__new__(
            EngineCore)  # Create instance without calling __init__
        engine_core.model_executor = model_executor

        with pytest.raises(RuntimeError) as exc_info:
            engine_core._initialize_kv_caches(vllm_config)

        # The original error should be re-raised
        assert "Some unexpected error occurred" in str(exc_info.value)


def test_initialize_kv_caches_success():
    """Test successful initialization of KV caches."""
    model_executor = MagicMock()
    model_executor.get_kv_cache_specs.return_value = [
        KVCacheSpec(block_size=16, num_blocks=100)
    ]
    # Some memory available
    model_executor.determine_available_memory.return_value = 1000

    # Create the KVCacheConfig with appropriate properties
    mock_kv_config = MagicMock(spec=KVCacheConfig)
    mock_kv_config.num_blocks = 50

    # Mock the get_kv_cache_configs function to return valid configs
    with patch('vllm.v1.engine.core.get_kv_cache_configs') as mock_get_configs:
        mock_get_configs.return_value = [mock_kv_config]

        vllm_config = MagicMock()
        cache_config = MagicMock()
        cache_config.min_required_gpu_blocks = 16
        vllm_config.cache_config = cache_config

        engine_core = EngineCore.__new__(
            EngineCore)  # Create instance without calling __init__
        engine_core.model_executor = model_executor
        engine_core.structured_output_manager = MagicMock(
        )  # Add this to avoid AttributeError

        # Patching the logger to check for success message
        with patch('vllm.v1.engine.core.logger') as mock_logger:
            num_gpu_blocks, num_cpu_blocks = engine_core._initialize_kv_caches(
                vllm_config)

            # Verify success
            assert num_gpu_blocks == 50
            assert num_cpu_blocks == 0
            # Verify that the info log was called with elapsed time message
            mock_logger.info.assert_called_with(
                ("init engine (profile, create kv cache, "
                 "warmup model) took %.2f seconds"),
                mock_logger.info.call_args[0][1])
