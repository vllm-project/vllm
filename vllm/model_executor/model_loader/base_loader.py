# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading, set_default_torch_dtype)

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows 
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else \
                      load_config.device
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)

            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()
