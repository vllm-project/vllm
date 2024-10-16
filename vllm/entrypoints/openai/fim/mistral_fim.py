from typing import List

from transformers_utils.tokenizer import AnyTokenizer
from transformers_utils.tokenizers import MistralTokenizer

from vllm.entrypoints.openai.fim.fim_encoder import (FIMEncoder,
                                                     FIMEncoderManager)


@FIMEncoderManager.register_module("mistral")
class MistralFIMEncoder(FIMEncoder):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError(
                "tokenizer incompatible with 'mistral' FIM encoder")

    def encode_with_suffix(self, prompt: str, suffix: str) -> List[int]:
        return self.tokenizer.encode_with_suffix(prompt=prompt, suffix=suffix)
