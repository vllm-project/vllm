"""Utilities for selecting and loading models."""
import contextlib
from typing import Tuple, Type

import torch
from torch import nn

from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.llava import LlavaForConditionalGeneration
from vllm.model_executor.tensorizer_loader import (
    ParameterizedLoadFormat, is_vllm_serialized_tensorizer,
    load_with_tensorizer)
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)

_VISION_MODEL_CLASSES = [
    LlavaForConditionalGeneration,
]


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(
        model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    if (model_config.quantization is not None
            and "MixtralForCausalLM" in architectures):
        architectures = ["QuantMixtralForCausalLM"]

    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return _get_model_architecture(model_config)[1]


def get_model(model_config: ModelConfig, device_config: DeviceConfig,
              **kwargs) -> nn.Module:
    lora_config = kwargs.get("lora_config", None)
    vision_language_config = kwargs.get("vision_language_config", None)
    tensorizer_config = kwargs.get("tensorizer_config", None)
    model_class = _get_model_architecture(model_config)[0]

    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")

        linear_method = quant_config.get_linear_method()

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        extra_kwargs = {}
        if hasattr(model_class, "supported_lora_modules"):
            extra_kwargs["lora_config"] = lora_config
        elif lora_config:
            raise ValueError(
                f"Model {model_class.__name__} does not support LoRA, "
                "but LoRA is enabled. Support for this model may "
                "be added in the future. If this is important to you, "
                "please open an issue on github.")
        elif model_class in _VISION_MODEL_CLASSES:
            extra_kwargs["vision_language_config"] = vision_language_config

        with torch.device(device_config.device):
            if (model_config.load_format == "tensorizer"
                    and is_vllm_serialized_tensorizer(tensorizer_config)):
                extra_kwargs["linear_method"] = linear_method
                tensorizer_config.model_class = model_class
                tensorizer_config.hf_config = model_config.hf_config
                tensorizer_config.dtype = model_config.dtype
                model = load_with_tensorizer(tensorizer_config, **extra_kwargs)
                return model.eval()
            model = model_class(config=model_config.hf_config,
                                linear_method=linear_method,
                                **extra_kwargs)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            if model_config.load_format == "tensorizer":
                # Provide a dynamic load format for `model.load_weights`
                # to retain tensorizer args from CLI.
                model_config.load_format = ParameterizedLoadFormat(
                    model_config.load_format)
                model_config.load_format.params = (
                    tensorizer_config._construct_tensorizer_args())

            model.load_weights(
                model_config.model,
                model_config.download_dir,
                model_config.load_format,
                model_config.revision,
            )
    return model.eval()
