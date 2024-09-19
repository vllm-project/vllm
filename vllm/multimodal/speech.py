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

    def _decode_spk_emb(self, spk_emb: str) -> np.ndarray:
        n = base64.b64decode(spk_emb)
        return np.frombuffer(n, dtype=np.float16).copy()

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:
        model_config = ctx.model_config
        if isinstance(data, str):
            n =F.normalize(
                torch.from_numpy(self._decode_spk_emb(data)),
                p=2.0,
                dim=0,
                eps=1e-12,
                )

            return MultiModalInputs({"speech": n})
        elif isinstance(data, torch.Tensor):
            raise NotImplementedError("Embeddings input is not supported yet")

        raise TypeError(f"Invalid image type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 3000

    @staticmethod
    def sample_random_speaker() -> str:
        n = np.random.randn(768).astype(np.float16)
        s = base64.b64encode(n).decode("utf-8")
        return s
