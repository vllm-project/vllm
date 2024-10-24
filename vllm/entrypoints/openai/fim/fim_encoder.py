from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from vllm.transformers_utils.tokenizer import AnyTokenizer


class FIMEncoder(ABC):
    """
    An encoder of fill-in-the-middle (FIM) prompts comprising prefix
    and suffix strings.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def encode_with_suffix(self, prefix: str, suffix: str) -> List[int]:
        """
        Encode the provided prompt prefix and suffix
        to a list of token ids
        """
        pass


class StringTemplateFIMEncoder(FIMEncoder):
    """FIMEncoder implementation using a simple string template
    with prefix and suffix variables."""

    def __init__(
        self,
        tokenizer: AnyTokenizer,
        name: str,
        template: str,
        special_tokens: Optional[Iterable[str]] = None,
    ):
        super().__init__(tokenizer)

        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            raise ValueError(
                "tokenizer incompatible with 'codellama' FIM encoder")

        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        for special_token in special_tokens or ():
            token_id = tokenizer.convert_tokens_to_ids(special_token)
            if token_id is None or token_id == unk_token_id:
                raise ValueError(
                    f"tokenizer incompatible with '{name}' FIM encoder")
        self.template = template

    def encode_with_suffix(self, prefix: str, suffix: str) -> List[int]:
        prompt = self.template.format(prefix=prefix, suffix=suffix)
        return self.tokenizer(prompt, add_special_tokens=False).input_ids
