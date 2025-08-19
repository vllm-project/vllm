# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, Optional

from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.bitsandbytes_loader import (
    BitsAndBytesModelLoader)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.model_executor.model_loader.runai_streamer_loader import (
    RunaiModelStreamerLoader)
from vllm.model_executor.model_loader.sharded_state_loader import (
    ShardedStateLoader)
from vllm.model_executor.model_loader.tensorizer_loader import TensorizerLoader
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture, get_model_cls)
from vllm.plugins import ExtensionManager

logger = init_logger(__name__)

# Reminder: Please update docstring in `LoadConfig`
# if a new load format is added here
LoadFormats = Literal[
    "auto",
    "bitsandbytes",
    "dummy",
    "fastsafetensors",
    "gguf",
    "mistral",
    "npcache",
    "pt",
    "runai_streamer",
    "runai_streamer_sharded",
    "safetensors",
    "sharded_state",
    "tensorizer",
]


def get_model(*,
              vllm_config: VllmConfig,
              model_config: Optional[ModelConfig] = None) -> nn.Module:
    loader = ExtensionManager.create(BaseModelLoader,
                                     vllm_config.load_config.load_format,
                                     vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(vllm_config=vllm_config,
                             model_config=model_config)


__all__ = [
    "get_model",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
]
