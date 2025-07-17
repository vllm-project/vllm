# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Set
from typing import Union

from vllm.config import LoadConfig
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

logger = init_logger(__name__)

DEFAULT_MODEL_LOADERS = {
    "auto": DefaultModelLoader,
    "bitsandbytes": BitsAndBytesModelLoader,
    "dummy": DummyModelLoader,
    "fastsafetensors": DefaultModelLoader,
    "gguf": GGUFModelLoader,
    "mistral": DefaultModelLoader,
    "npcache": DefaultModelLoader,
    "pt": DefaultModelLoader,
    "runai_streamer": RunaiModelStreamerLoader,
    "runai_streamer_sharded": ShardedStateLoader,
    "safetensors": DefaultModelLoader,
    "sharded_state": ShardedStateLoader,
    "tensorizer": TensorizerLoader,
}


class ModelLoaderRegistry:
    _model_loaders: dict[str, type[BaseModelLoader]] = DEFAULT_MODEL_LOADERS

    @classmethod
    def get_supported_load_formats(cls) -> Set[str]:
        return cls._model_loaders.keys()

    @classmethod
    def get_model_loader(cls, load_config: LoadConfig) -> BaseModelLoader:
        load_format = load_config.load_format
        if load_format not in cls._model_loaders:
            raise ValueError(f"load_format: {load_format} is not supported")
        return cls._model_loaders[load_format](load_config)

    @classmethod
    def register(
        cls,
        load_format: str,
        loader_cls: Union[BaseModelLoader, str],
    ) -> None:
        """
        Register an external model loader to be used in vLLM.

        `loader_cls` can be either:

        - A class derived from `BaseModelLoader`
        - A string in the format `<module>:<class>` which can be used to
          lazily import the model loader.
        """
        if not isinstance(load_format, str):
            msg = f"`load_format` should be a string, not a {type(load_format)}"
            raise TypeError(msg)

        if load_format in cls._model_loaders:
            logger.warning(
                "Load format %s is already registered, and will be "
                "overwritten by the new loader class %s.", load_format,
                loader_cls)

        if isinstance(loader_cls, str):
            split_str = loader_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)
            module_name, class_name = split_str
            module = importlib.import_module(module_name)
            model_loader_cls = getattr(module, class_name)
        elif isinstance(loader_cls, type) and issubclass(
                loader_cls, BaseModelLoader):
            model_loader_cls = loader_cls
        else:
            msg = ("`loader_cls` should be a string or `BaseModelLoader`, "
                   f"not a {type(loader_cls)}")
            raise TypeError(msg)

        cls._model_loaders[load_format] = model_loader_cls
        logger.info("Registered `%s` with load format `%s`", model_loader_cls,
                    load_format)
