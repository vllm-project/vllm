import importlib
from typing import List, Optional, Type

import torch.nn as nn


_MODEL_REGISTRY = {
    "AquilaModel": ("aquila", "AquilaForCausalLM"),
    "AquilaForCausalLM": ("aquila", "AquilaForCausalLM"),  # AquilaChat2
    "BaiChuanForCausalLM": ("baichuan", "BaiChuanForCausalLM"),  # baichuan-7b
    "BaichuanForCausalLM": ("baichuan", "BaichuanForCausalLM"),  # baichuan-13b
    "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMForCausalLM"),
    "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "GPT2LMHeadModel"),
    "GPTBigCodeForCausalLM": ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    "GPTJForCausalLM": GPTJForCausalLM,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    "MistralForCausalLM": MistralForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
    # transformers's mpt class has lower case
    "MptForCausalLM": MPTForCausalLM,
    "MPTForCausalLM": MPTForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
    "PhiForCausalLM": PhiForCausalLM,
    "QWenLMHeadModel": QWenLMHeadModel,
    "RWForCausalLM": FalconForCausalLM,
    "YiForCausalLM": YiForCausalLM,
}

_MODELS_TO_MODULES = {
    "AquilaForCausalLM": "aquila",
    "BaiChuanForCausalLM": "baichuan",
    "BaichuanForCausalLM": "baichuan",
    "BloomForCausalLM": "bloom",
    "ChatGLMForCausalLM": "chatglm",
    "FalconForCausalLM": "falcon",
    "GPT2LMHeadModel": "gpt2",
    "GPTBigCodeForCausalLM": "gpt_bigcode",
    "GPTJForCausalLM": "gpt_j",
    "GPTNeoXForCausalLM": "gpt_neox",
    "InternLMForCausalLM": "internlm",
    "LlamaForCausalLM": "llama",
    "MPTForCausalLM": "mpt",
    "OPTForCausalLM": "opt",
    "PhiForCausalLM": "phi_1_5",
    "QWenLMHeadModel": "qwen",
    "MistralForCausalLM": "mistral",
    "MixtralForCausalLM": "mixtral",
    "YiForCausalLM": "yi",
}



class ModelRegistry:

    @staticmethod
    def load_model(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in _MODELS_TO_MODULES:
            return None
        module_name = _MODELS_TO_MODULES[model_arch]
        module = importlib.import_module(
            f"vllm.model_executor.models.{module_name}")
        return getattr(module, model_arch, None)

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS_TO_MODULES.keys())


__all__ = [
    "ModelRegistry",
]
