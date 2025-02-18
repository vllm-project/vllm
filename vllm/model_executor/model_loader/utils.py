# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import transformers
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vllm.config import ModelConfig, ModelImpl
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.adapters import (as_classification_model,
                                                 as_embedding_model,
                                                 as_reward_model)

logger = init_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def is_transformers_impl_compatible(
        arch: str,
        module: Optional[transformers.PreTrainedModel] = None) -> bool:
    mod = module or getattr(transformers, arch, None)
    if mod is None:
        return False
    if hasattr(mod, "supports_backend"):
        return mod.is_backend_compatible()
    else:
        return mod._supports_flex_attn


def resolve_transformers_fallback(model_config: ModelConfig,
                                  architectures: list[str]):
    for i, arch in enumerate(architectures):
        if arch == "TransformersModel":
            continue
        auto_map: dict[str, str] = getattr(model_config.hf_config, "auto_map",
                                           None) or dict()
        # Make sure that config class is always initialized before model class,
        # otherwise the model class won't be able to access the config class,
        # the expected auto_map should have correct order like:
        # "auto_map": {
        #     "AutoConfig": "<your-repo-name>--<config-name>",
        #     "AutoModel": "<your-repo-name>--<config-name>",
        #     "AutoModelFor<Task>": "<your-repo-name>--<config-name>",
        # },
        auto_modules = {
            name: get_class_from_dynamic_module(module, model_config.model)
            for name, module in sorted(auto_map.items(), key=lambda x: x[0])
        }
        custom_model_module = auto_modules.get("AutoModel")
        # TODO(Isotr0py): Further clean up these raises.
        # perhaps handled them in _ModelRegistry._raise_for_unsupported?
        if model_config.model_impl == ModelImpl.TRANSFORMERS:
            if not is_transformers_impl_compatible(arch, custom_model_module):
                raise ValueError(
                    f"The Transformers implementation of {arch} is not "
                    "compatible with vLLM.")
            architectures[i] = "TransformersModel"
        if model_config.model_impl == ModelImpl.AUTO:
            if not is_transformers_impl_compatible(arch, custom_model_module):
                raise ValueError(
                    f"{arch} has no vLLM implementation and the Transformers "
                    "implementation is not compatible with vLLM.")
            logger.warning(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.", arch)
            architectures[i] = "TransformersModel"
    return architectures


def get_model_architecture(
        model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = [
        "fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"
    ]

    if (model_config.quantization is not None
            and model_config.quantization not in mixtral_supported
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

    vllm_supported_archs = ModelRegistry.get_supported_archs()
    is_vllm_supported = any(arch in vllm_supported_archs
                            for arch in architectures)
    if (not is_vllm_supported
            or model_config.model_impl == ModelImpl.TRANSFORMERS):
        architectures = resolve_transformers_fallback(model_config,
                                                      architectures)

    model_cls, arch = ModelRegistry.resolve_model_cls(architectures)
    if model_config.task == "embed":
        model_cls = as_embedding_model(model_cls)
    elif model_config.task == "classify":
        model_cls = as_classification_model(model_cls)
    elif model_config.task == "reward":
        model_cls = as_reward_model(model_cls)

    return model_cls, arch


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their 
    constituent parts.
    """
    packed_mapping: Dict[str, List[str]]
    inverse_packed_mapping: Dict[str, Tuple[str,
                                            int]] = field(default_factory=dict)

    def __post_init__(self):
        for packed_name, sub_params in self.packed_mapping.items():
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )

    def get_sub_modules(self,
                        module_name: str) -> Optional[Tuple[str, List[str]]]:
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None


def configure_quant_config(quant_config: QuantizationConfig,
                           model_class: Type[nn.Module]):
    """
    Pass packed_modules_mapping by reference to quant_config so that
    quant_config can properly match fused modules

    Note that model attributes are passed by reference to quant_config,
    enabling them to be updated by model_class.__new__ (ex. chatglm, qwen)
    """
    packed_mapping = getattr(model_class, "packed_modules_mapping", None)
    if packed_mapping is not None:
        # pass packed_modules_mapping by reference to quant_config
        quant_config.packed_modules_mapping = packed_mapping
    else:
        logger.warning(
            "The model class %s has not defined `packed_modules_mapping`, "
            "this may lead to incorrect mapping of quantized or ignored "
            "modules", model_class.__name__)
