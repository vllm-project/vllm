# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""
import contextlib
import inspect
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from vllm.attention import Attention
from vllm.config import (ModelConfig, ModelImpl, VllmConfig,
                         set_current_vllm_config)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVCrossParallelLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.adapters import (as_classification_model,
                                                 as_embedding_model,
                                                 as_reward_model)
from vllm.utils import is_pin_memory_available

logger = init_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: Optional[type[nn.Module]] = None,
    model_config: Optional[ModelConfig] = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    if model_config is None:
        model_config = vllm_config.model_config
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config,
                                     check_compile=True,
                                     prefix=prefix):
            return model_class(vllm_config=vllm_config, prefix=prefix)

    msg = ("vLLM model class should accept `vllm_config` and `prefix` as "
           "input arguments. Possibly you have an old-style model class"
           " registered from out of tree and it is used for new vLLM version. "
           "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
           "for the design and update the model class accordingly.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
    # try to be compatible with old-style model class
    kwargs = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    with set_current_vllm_config(vllm_config,
                                 check_compile=True,
                                 prefix=prefix):
        return model_class(**kwargs)


def process_weights_after_loading(model: nn.Module, model_config: ModelConfig,
                                  target_device: torch.device) -> None:
    for _, module in model.named_modules():
        if isinstance(module, QKVCrossParallelLinear):
            # NOTE(Isotr0py): special case for cross QKV layer because
            # q and kv proj aren't registered as submodules intentionally
            module.process_weights_after_loading()
            continue
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Currently only used by MLA.
    # NOTE: This intentionally happens after other modules so we can easily
    # decompress the weights for MLA.
    for _, module in model.named_modules():
        if isinstance(module, Attention) and \
            hasattr(module, "process_weights_after_loading"):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading(model_config.dtype)


@contextmanager
def device_loading_context(module: torch.nn.Module,
                           target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                if original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


def resolve_transformers_arch(model_config: ModelConfig,
                              architectures: list[str]):
    for i, arch in enumerate(architectures):
        if arch == "TransformersForCausalLM":
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
            name:
            get_class_from_dynamic_module(module,
                                          model_config.model,
                                          revision=model_config.revision)
            for name, module in sorted(auto_map.items(), key=lambda x: x[0])
        }
        model_module = getattr(transformers, arch, None)
        if model_module is None:
            if "AutoModel" not in auto_map:
                raise ValueError(
                    f"Cannot find model module. '{arch}' is not a registered "
                    "model in the Transformers library (only relevant if the "
                    "model is meant to be in Transformers) and 'AutoModel' is "
                    "not present in the model config's 'auto_map' (relevant "
                    "if the model is custom).")
            model_module = auto_modules["AutoModel"]
        # TODO(Isotr0py): Further clean up these raises.
        # perhaps handled them in _ModelRegistry._raise_for_unsupported?
        if model_config.model_impl == ModelImpl.TRANSFORMERS:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"The Transformers implementation of {arch} is not "
                    "compatible with vLLM.")
            architectures[i] = "TransformersForCausalLM"
        if model_config.model_impl == ModelImpl.AUTO:
            if not model_module.is_backend_compatible():
                raise ValueError(
                    f"{arch} has no vLLM implementation and the Transformers "
                    "implementation is not compatible with vLLM. Try setting "
                    "VLLM_USE_V1=0.")
            logger.warning(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.", arch)
            architectures[i] = "TransformersForCausalLM"
    return architectures


def get_model_architecture(
        model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = [
        "fp8", "compressed-tensors", "gptq_marlin", "awq_marlin", "quark"
    ]

    vllm_supported_archs = ModelRegistry.get_supported_archs()
    vllm_not_supported = not any(arch in vllm_supported_archs
                                 for arch in architectures)
    if (model_config.model_impl == ModelImpl.TRANSFORMERS or
            model_config.model_impl != ModelImpl.VLLM and vllm_not_supported):
        architectures = resolve_transformers_arch(model_config, architectures)
    elif (model_config.quantization is not None
          and model_config.quantization not in mixtral_supported
          and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

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
    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str,
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
                        module_name: str) -> Optional[tuple[str, list[str]]]:
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None


def configure_quant_config(quant_config: QuantizationConfig,
                           model_class: type[nn.Module]):
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
