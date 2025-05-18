# SPDX-License-Identifier: Apache-2.0
"""Tests for the Pallas MOE implementation.

Run `pytest tests/platforms/test_tpu.py`.
"""
from unittest.mock import MagicMock, patch

import pytest

import vllm.config
from vllm.platforms.tpu import TpuPlatform


@pytest.mark.parametrize(
    "use_v1,initial_block_size,expected_block_size",
    [
        (True, 32, 32),  # Case 1: v1: block_size set, should remain unchanged
        (
            True, None, 128
        ),  # Case 2: v1: block_size None, should be set to get_page_size (128)
        (False, None, 16),  # Case 3: v0: block_size None, should be set to 16
        (False, 32, 32),  # Case 4: v0: block_size set, should remain unchanged
    ])
@patch(
    "vllm.v1.attention.backends.pallas.PallasAttentionBackend.get_page_size",
    return_value=128)
@patch(
    "vllm.v1.attention.backends.pallas.PallasAttentionBackend.get_min_page_size",
    return_value=8)
def test_tpu_platform_update_vllm_config_block_size_respect_passin_block_size(
        mock_get_min_page_size, mock_get_page_size, use_v1, initial_block_size,
        expected_block_size) -> None:
    """Test TPU platform updates VLLM config with block size."""
    # arrange
    mock_cached_config = MagicMock()
    mock_cached_config.block_size = initial_block_size

    mock_model_config = MagicMock()
    mock_model_config.dtype = "float16"

    mock_vllm_config = MagicMock()
    mock_vllm_config.cache_config = mock_cached_config
    mock_vllm_config.compilation_config = MagicMock()
    mock_vllm_config.compilation_config.level = (
        vllm.config.CompilationLevel.DYNAMO_ONCE)
    mock_vllm_config.compilation_config.backend = "openxla"
    mock_vllm_config.model_config = mock_model_config
    mock_vllm_config.speculative_config = None
    mock_vllm_config.parallel_config = MagicMock()
    mock_vllm_config.parallel_config.worker_cls = (
        "vllm.v1.worker.tpu_worker.TPUWorker")
    mock_vllm_config.scheduler_config = MagicMock()

    # act
    with patch("vllm.envs.VLLM_USE_V1", use_v1):
        TpuPlatform.check_and_update_config(mock_vllm_config)

    # assert
    assert mock_cached_config.block_size == expected_block_size
    if use_v1:
        mock_get_min_page_size.assert_called()
        if initial_block_size is None:
            mock_get_page_size.assert_called()
        else:
            mock_get_page_size.assert_not_called()
