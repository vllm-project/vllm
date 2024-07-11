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
import pybase16384 as b14

class SpeechPlugin(MultiModalPlugin):

    def get_data_key(self) -> str:
        return "speech"

    def _decode_spk_emb(self, spk_emb: str) -> np.ndarray:
        return np.frombuffer(
            lzma.decompress(
                b14.decode_from_string(spk_emb),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
            dtype=np.float16,
        ).copy()

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:
        model_config = ctx.model_config
        if isinstance(data, str):
            n =F.normalize(
                torch.from_numpy(self._decode_spk_emb(data)),
                p=2.0,
                dim=0,
                eps=1e-12,
                ).unsqueeze_(0)

            return MultiModalInputs({"speech": n})
        elif isinstance(data, torch.Tensor):
            raise NotImplementedError("Embeddings input is not supported yet")

        raise TypeError(f"Invalid image type: {type(data)}")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 3000

    @staticmethod
    def sample_random_speaker() -> str:
       return b14.encode_to_string(
           lzma.compress(
               np.random.randn(768).astype(np.float16).tobytes(),
               format=lzma.FORMAT_RAW,
               filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
               )
            )
