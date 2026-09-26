from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn

from vllm.config import DeviceConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

@pytest.fixture
def should_do_global_cleanup_after_test():
    return False

class DummyModelLoader(BaseModelLoader):
    def download_model(self, model_config):
        pass

    def load_weights(self, model, model_config):
        pass

@patch("vllm.model_executor.model_loader.base_loader.initialize_model")
@patch("vllm.model_executor.model_loader.base_loader.process_weights_after_loading")
@patch("vllm.model_executor.model_loader.base_loader.torch.accelerator.empty_cache")
@patch("vllm.model_executor.model_loader.base_loader.gc.collect")
@patch("vllm.model_executor.model_loader.base_loader.torch.accelerator.max_memory_allocated")
def test_base_model_loader_memory_reclamation(
    mock_max_memory,
    mock_gc_collect,
    mock_empty_cache,
    mock_process_weights,
    mock_initialize,
):
    mock_model = MagicMock(spec=nn.Module)
    mock_initialize.return_value = mock_model

    mock_vllm_config = MagicMock(spec=VllmConfig)
    mock_vllm_config.device_config = MagicMock(spec=DeviceConfig)
    mock_vllm_config.device_config.device = torch.device("cpu")
    mock_vllm_config.load_config = MagicMock(spec=LoadConfig)
    mock_vllm_config.load_config.device = None
    mock_vllm_config.quant_config = None

    mock_model_config = MagicMock(spec=ModelConfig)
    mock_model_config.dtype = torch.float32

    loader = DummyModelLoader(load_config=mock_vllm_config.load_config)

    with patch(
        "vllm.model_executor.model_loader.base_loader.current_platform"
    ) as mock_platform:
        mock_platform.is_cuda_alike.return_value = True
        mock_platform.is_xpu.return_value = False
        mock_max_memory.return_value = 100

        loader.load_model(mock_vllm_config, mock_model_config)

        assert mock_gc_collect.call_count == 2
        assert mock_empty_cache.call_count == 2
