"""Utilities for selecting and loading models."""
import contextlib
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import *
from vllm.model_executor.weight_utils import (get_quant_config,
                                              initialize_dummy_weights)
from vllm.utils import is_hip
from vllm.logger import init_logger

logger = init_logger(__name__)

# TODO(woosuk): Lazy-load the model classes.
_MODEL_REGISTRY = {
    "AquilaModel": AquilaForCausalLM,
    "AquilaForCausalLM": AquilaForCausalLM,  # AquilaChat2
    "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    "BloomForCausalLM": BloomForCausalLM,
    "ChatGLMModel": ChatGLMForCausalLM,
    "ChatGLMForConditionalGeneration": ChatGLMForCausalLM,
    "FalconForCausalLM": FalconForCausalLM,
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "MistralForCausalLM": MistralForCausalLM,
    # transformers's mpt class has lower case
    "MptForCausalLM": MPTForCausalLM,
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "PhiForCausalLM": PhiForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "RWForCausalLM": FalconForCausalLM,
    "YiForCausalLM": YiForCausalLM,
}

# Models to be disabled in ROCm
_ROCM_UNSUPPORTED_MODELS = []
if is_hip():
    for rocm_model in _ROCM_UNSUPPORTED_MODELS:
        del _MODEL_REGISTRY[rocm_model]

# Models partially supported in ROCm
_ROCM_PARTIALLY_SUPPORTED_MODELS = {
    "MistralForCausalLM":
    "Sliding window attention is not supported in ROCm's flash attention",
}


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            if is_hip() and arch in _ROCM_PARTIALLY_SUPPORTED_MODELS:
                logger.warning(
                    f"{arch} is not fully supported in ROCm. Reason: "
                    f"{_ROCM_PARTIALLY_SUPPORTED_MODELS[arch]}")
            return _MODEL_REGISTRY[arch]
        elif arch in _ROCM_UNSUPPORTED_MODELS:
            raise ValueError(
                f"Model architecture {arch} is not supported by ROCm for now. \n"
                f"Supported architectures {list(_MODEL_REGISTRY.keys())}")
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the (maybe quantized) linear method.
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization,
                                        model_config.model,
                                        model_config.hf_config,
                                        model_config.download_dir)
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
        with torch.device("cuda"):
            model = model_class(model_config.hf_config, linear_method)
        if model_config.load_format == "dummy":
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        else:
            # Load the weights from the cached or downloaded files.
            model.load_weights(model_config.model, model_config.download_dir,
                               model_config.load_format, model_config.revision)
    return model.eval()
