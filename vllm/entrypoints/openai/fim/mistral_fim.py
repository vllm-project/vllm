from typing import List

from mistral_common.tokens.tokenizers.sentencepiece import InstructTokenizerV1

from vllm.entrypoints.openai.fim.fim_encoder import FIMEncoder
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizers import MistralTokenizer


class MistralFIMEncoder(FIMEncoder):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if not isinstance(tokenizer, MistralTokenizer) \
            or isinstance(tokenizer.instruct, InstructTokenizerV1):
            raise ValueError(
                "tokenizer incompatible with 'mistral' FIM encoder")

    def encode_with_suffix(self, prefix: str, suffix: str) -> List[int]:
        return self.tokenizer.encode_with_suffix(prefix=prefix, suffix=suffix)
