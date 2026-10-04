# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
    def __init__(self, load_config: LoadConfig, events: list[str]):
        super().__init__(load_config)
        self.events = events

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        self.events.append("load_weights")


@pytest.mark.cpu_test
@pytest.mark.parametrize("is_cuda_alike", [True, False])
def test_base_model_loader_memory_reclamation(is_cuda_alike: bool):
    events: list[str] = []
    model = nn.Linear(4, 4)

    mock_vllm_config = MagicMock(spec=VllmConfig)
    mock_vllm_config.device_config = MagicMock(spec=DeviceConfig)
    mock_vllm_config.device_config.device = torch.device("cpu")
    mock_vllm_config.load_config = MagicMock(spec=LoadConfig)
    mock_vllm_config.load_config.device = None
    mock_vllm_config.quant_config = None

    mock_model_config = MagicMock(spec=ModelConfig)
    mock_model_config.dtype = torch.float32

    loader = DummyModelLoader(load_config=mock_vllm_config.load_config, events=events)

    with (
        patch(
            "vllm.model_executor.model_loader.base_loader.initialize_model",
            return_value=model,
        ),
        patch(
            "vllm.model_executor.model_loader.base_loader.process_weights_after_loading",
            side_effect=lambda *args, **kwargs: events.append("process_weights"),
        ),
        patch(
            "vllm.model_executor.model_loader.base_loader.gc.collect",
            side_effect=lambda: events.append("gc.collect") or 0,
        ) as mock_gc_collect,
        patch(
            "vllm.model_executor.model_loader.base_loader.torch.accelerator.empty_cache",
            side_effect=lambda: events.append("empty_cache"),
        ) as mock_empty_cache,
        patch(
            "vllm.model_executor.model_loader.base_loader.torch.accelerator.max_memory_allocated",
            return_value=100,
        ),
        patch(
            "vllm.model_executor.model_loader.base_loader.current_platform"
        ) as mock_platform,
    ):
        mock_platform.is_cuda_alike.return_value = is_cuda_alike
        mock_platform.is_xpu.return_value = False

        loaded_model = loader.load_model(mock_vllm_config, mock_model_config)

    assert loaded_model is model
    assert mock_gc_collect.call_count == 2
    if is_cuda_alike:
        assert mock_empty_cache.call_count == 2
        assert events == [
            "load_weights",
            "gc.collect",
            "empty_cache",
            "process_weights",
            "gc.collect",
            "empty_cache",
        ]
    else:
        assert mock_empty_cache.call_count == 0
        assert events == [
            "load_weights",
            "gc.collect",
            "process_weights",
            "gc.collect",
        ]
