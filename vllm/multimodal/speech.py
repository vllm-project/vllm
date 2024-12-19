from functools import lru_cache
import lzma
from typing import List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.config import ModelConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.transformers_utils.image_processor import get_image_processor
from vllm.transformers_utils.tokenizer import get_tokenizer

from .base import MultiModalInputs, MultiModalPlugin
import base64
import pickle

class SpeechPlugin(MultiModalPlugin):

    def get_data_key(self) -> str:
        return "audio"

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:
        model_config = ctx.model_config
        if data is None:
            return MultiModalInputs({"audio": torch.zeros(16, model_config.hf_config.hidden_size)})
        else:
            return MultiModalInputs({"audio": data})

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 3000

    @staticmethod
    def sample_random_speaker() -> str:
        return None
