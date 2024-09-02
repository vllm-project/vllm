import itertools
from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
# from transformers import CLIPVisionConfig, LlavaConfig, SiglipVisionConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors, SamplerOutput
from .interfaces import SupportsMultiModal

logger = init_logger(__name__)

def get_max_llama_image_tokens(ctx: InputContext) -> int:
    logger.warning("need further check on max llama image tokens")
    print("ctx", type(ctx))
    print(ctx)
    return 1025 * 2

@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_llama_image_tokens)
class LlamaVLForCausalLM(nn.Module, SupportsMultiModal):
    def __init__(self, config,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        print("config", type(config))
        print(config)
        print("multimodal_config", type(multimodal_config))
        print(multimodal_config)
        print("cache_config", type(cache_config))
        print(cache_config)
        print("quant_config", type(quant_config))
        print(quant_config)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        for name, weight in weights:
            print(name, weight.shape)
            
