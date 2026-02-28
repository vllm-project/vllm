# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Literal

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name,
    get_model_architecture,
    get_model_cls,
)

logger = init_logger(__name__)

# Reminder: Please update docstring in `LoadConfig`
# if a new load format is added here
LoadFormats = Literal[
    "auto",
    "hf",
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

# Lazy attribute mapping: class_name -> module_name.
# Loader modules are imported on demand to avoid pulling in heavy GPU
# dependencies (e.g. bitsandbytes -> triton) when they aren't needed.
_LAZY_ATTR_TO_MODULE: dict[str, str] = {
    "BitsAndBytesModelLoader": "bitsandbytes_loader",
    "DefaultModelLoader": "default_loader",
    "DummyModelLoader": "dummy_loader",
    "GGUFModelLoader": "gguf_loader",
    "RunaiModelStreamerLoader": "runai_streamer_loader",
    "ShardedStateLoader": "sharded_state_loader",
    "TensorizerLoader": "tensorizer_loader",
}


def __getattr__(name: str):
    if name in _LAZY_ATTR_TO_MODULE:
        module_path = f"{__package__}.{_LAZY_ATTR_TO_MODULE[name]}"
        module = importlib.import_module(module_path)
        cls = getattr(module, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_LAZY_LOAD_FORMAT_TO_MODULE: dict[str, str] = {
    "auto": "DefaultModelLoader",
    "hf": "DefaultModelLoader",
    "bitsandbytes": "BitsAndBytesModelLoader",
    "dummy": "DummyModelLoader",
    "fastsafetensors": "DefaultModelLoader",
    "gguf": "GGUFModelLoader",
    "mistral": "DefaultModelLoader",
    "npcache": "DefaultModelLoader",
    "pt": "DefaultModelLoader",
    "runai_streamer": "RunaiModelStreamerLoader",
    "runai_streamer_sharded": "ShardedStateLoader",
    "safetensors": "DefaultModelLoader",
    "sharded_state": "ShardedStateLoader",
    "tensorizer": "TensorizerLoader",
}

# Holds loader classes registered at runtime via register_model_loader().
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {}


def _get_loader_cls(load_format: str) -> type[BaseModelLoader]:
    """Resolve a load format to its loader class, importing lazily."""
    if load_format in _LOAD_FORMAT_TO_MODEL_LOADER:
        return _LOAD_FORMAT_TO_MODEL_LOADER[load_format]
    if load_format in _LAZY_LOAD_FORMAT_TO_MODULE:
        cls_name = _LAZY_LOAD_FORMAT_TO_MODULE[load_format]
        cls = __getattr__(cls_name)
        _LOAD_FORMAT_TO_MODEL_LOADER[load_format] = cls
        return cls
    raise ValueError(f"Load format `{load_format}` is not supported")


def register_model_loader(load_format: str):
    """Register a customized vllm model loader.

    When a load format is not supported by vllm, you can register a customized
    model loader to support it.

    Args:
        load_format (str): The model loader format name.

    Examples:
        >>> from vllm.config.load import LoadConfig
        >>> from vllm.model_executor.model_loader import (
        ...     get_model_loader,
        ...     register_model_loader,
        ... )
        >>> from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        >>>
        >>> @register_model_loader("my_loader")
        ... class MyModelLoader(BaseModelLoader):
        ...     def download_model(self):
        ...         pass
        ...
        ...     def load_weights(self):
        ...         pass
        >>>
        >>> load_config = LoadConfig(load_format="my_loader")
        >>> type(get_model_loader(load_config))
        <class 'MyModelLoader'>
    """  # noqa: E501

    def _wrapper(model_loader_cls):
        if (
            load_format in _LOAD_FORMAT_TO_MODEL_LOADER
            or load_format in _LAZY_LOAD_FORMAT_TO_MODULE
        ):
            logger.warning(
                "Load format `%s` is already registered, and will be "
                "overwritten by the new loader class `%s`.",
                load_format,
                model_loader_cls,
            )
        if not issubclass(model_loader_cls, BaseModelLoader):
            raise ValueError(
                "The model loader must be a subclass of `BaseModelLoader`."
            )
        _LOAD_FORMAT_TO_MODEL_LOADER[load_format] = model_loader_cls
        logger.info(
            "Registered model loader `%s` with load format `%s`",
            model_loader_cls,
            load_format,
        )
        return model_loader_cls

    return _wrapper


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    return _get_loader_cls(load_config.load_format)(load_config)


def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    loader = get_model_loader(load_config or vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "register_model_loader",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
]
