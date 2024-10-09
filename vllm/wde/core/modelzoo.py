import functools
import importlib
from typing import Dict, List, Optional, Type

import torch.nn as nn

from vllm.logger import init_logger
from vllm.wde.encode_only.modelzoo import ENCODE_ONLY_MODELS

logger = init_logger(__name__)

_MODELS_LIST = [ENCODE_ONLY_MODELS]

_MODELS = dict()
for m in _MODELS_LIST:
    _MODELS.update(**m)

# Architecture -> type.
# out of tree models
_OOT_MODELS: Dict[str, Type[nn.Module]] = {}


class ModelRegistry:

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _get_model(model_arch: str):
        module_str, workflow = _MODELS[model_arch]
        module_name, model_cls_name = module_str.split(":")
        module = importlib.import_module(module_name)
        return getattr(module, model_cls_name, None)

    @staticmethod
    def load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch in _OOT_MODELS:
            return _OOT_MODELS[model_arch]
        if model_arch not in _MODELS:
            return None
        return ModelRegistry._get_model(model_arch)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys())

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def get_workflow(model_arch: str):
        module_str, workflow = _MODELS[model_arch]
        return workflow

    @staticmethod
    def register_model(model_arch: str, model_cls: Type[nn.Module]):
        if model_arch in _MODELS:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.", model_arch,
                model_cls.__name__)
        global _OOT_MODELS
        _OOT_MODELS[model_arch] = model_cls


__all__ = [
    "ModelRegistry",
]
