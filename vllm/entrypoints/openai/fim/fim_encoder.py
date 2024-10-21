from abc import ABC, abstractmethod
from functools import partial
from inspect import isclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

from vllm.entrypoints.openai.fim.codellama_fim import CodeLlamaFIMEncoder
from vllm.entrypoints.openai.fim.mistral_fim import MistralFIMEncoder
from vllm.transformers_utils.tokenizer import AnyTokenizer

# Entries are either an FIMEncoder implementation class or
# tuple of (template, special_tokens_list).
_FIM_ENCODERS: Dict[str, Union[Type, Tuple[str, Iterable[str]]]] = {
    "mistral":
    MistralFIMEncoder,
    "codellama":
    CodeLlamaFIMEncoder,
    "deepseek": (
        "<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>",
        ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"),
    ),
    "starcoder": (
        "<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>",
        ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"),
    )
}


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
        name: str,
        tokenizer: AnyTokenizer,
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
        return self.tokenizer(prompt, add_special_tokens=False)


def get_supported_fim_encoders() -> Iterable[str]:
    """Return set of supported FIM encoder types."""
    return _FIM_ENCODERS.keys()


def get_fim_encoder_lookup(
        name: Optional[str]) -> Optional[Callable[[AnyTokenizer], FIMEncoder]]:
    """
    Get a function that returns a FIMEncoder instance for a given tokenizer.
    Raise a KeyError exception if the name is not recognized.
    """
    if name is None:
        return None

    if (encoder := _FIM_ENCODERS.get(name)) is None:
        raise KeyError(f"fim encoder '{name}' not recognized")

    factory: Callable[[AnyTokenizer], FIMEncoder]
    if isclass(encoder):
        assert issubclass(encoder, FIMEncoder)
        factory = encoder
    else:
        assert isinstance(encoder, tuple)
        template, special_tokens = encoder
        factory = partial(StringTemplateFIMEncoder,
                          name=name,
                          template=template,
                          special_tokens=special_tokens)

    def for_tokenizer(tokenizer: AnyTokenizer) -> FIMEncoder:
        fim_encoder = getattr(tokenizer, "fim_encoder", None)
        if fim_encoder is None:
            fim_encoder = factory(tokenizer)
            tokenizer.fim_encoder = fim_encoder  # type: ignore[union-attr]
        return fim_encoder

    return for_tokenizer
