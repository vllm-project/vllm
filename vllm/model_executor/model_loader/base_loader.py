# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod

import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *, vllm_config: VllmConfig) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError
