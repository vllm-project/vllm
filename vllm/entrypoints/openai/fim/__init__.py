from functools import partial
from inspect import isclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Type, Union

from vllm.entrypoints.openai.fim.codellama_fim import CodeLlamaFIMEncoder
from vllm.entrypoints.openai.fim.fim_encoder import (FIMEncoder,
                                                     StringTemplateFIMEncoder)
from vllm.entrypoints.openai.fim.mistral_fim import MistralFIMEncoder
from vllm.transformers_utils.tokenizer import AnyTokenizer

__all__ = [
    "FIMEncoder", "get_supported_fim_encoders", "get_fim_encoder_lookup"
]

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
        raise ValueError(f"fim encoder '{name}' not recognized")

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
