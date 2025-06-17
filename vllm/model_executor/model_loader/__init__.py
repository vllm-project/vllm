# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from torch import nn

from vllm.config import LoadConfig, LoadFormat, ModelConfig, VllmConfig
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
    get_architecture_class_name, get_model_architecture)


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.TENSORIZER:
        return TensorizerLoader(load_config)

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        return ShardedStateLoader(load_config)

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        return BitsAndBytesModelLoader(load_config)

    if load_config.load_format == LoadFormat.GGUF:
        return GGUFModelLoader(load_config)

    if load_config.load_format == LoadFormat.RUNAI_STREAMER:
        return RunaiModelStreamerLoader(load_config)

    if load_config.load_format == LoadFormat.RUNAI_STREAMER_SHARDED:
        return ShardedStateLoader(load_config, runai_model_streamer=True)

    return DefaultModelLoader(load_config)


def get_model(*,
              vllm_config: VllmConfig,
              model_config: Optional[ModelConfig] = None) -> nn.Module:
    loader = get_model_loader(vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(vllm_config=vllm_config,
                             model_config=model_config)


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
]
