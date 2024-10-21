from typing import List

from vllm.entrypoints.openai.fim.fim_encoder import FIMEncoder
from vllm.transformers_utils.tokenizer import AnyTokenizer


class CodeLlamaFIMEncoder(FIMEncoder):
    """
    FIM Encoder for Meta CodeLlama models

    Adapted from https://github.com/meta-llama/codellama/blob/e81b597e44dbecc2a0dedb9949fdf84adfc22395/llama/generation.py#L474
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)

        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            raise ValueError(
                "tokenizer incompatible with 'codellama' FIM encoder")

        self.bos_id = tokenizer.convert_tokens_to_ids("<s>")
        self.prefix_id = tokenizer.convert_tokens_to_ids("▁<PRE>")
        self.suffix_id = tokenizer.convert_tokens_to_ids("▁<SUF>")
        self.middle_id = tokenizer.convert_tokens_to_ids("▁<MID>")

        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if any(tid in
               {self.bos_id, self.prefix_id, self.suffix_id, self.middle_id}
               for tid in (None, unk_token_id)):
            raise ValueError(
                "tokenizer incompatible with 'codellama' FIM encoder")

    def encode_with_suffix(self, prefix: str, suffix: str) -> List[int]:
        return ([self.bos_id, self.prefix_id] +
                self.tokenizer(prefix, add_special_tokens=False) +
                [self.suffix_id] + self._encode_infilling(suffix) +
                [self.middle_id])

    def _encode_infilling(self, s: str) -> List[int]:
        """Encode a string without an implicit leading space."""
        return self.tokenizer("☺" + s, add_special_tokens=False)[2:]
