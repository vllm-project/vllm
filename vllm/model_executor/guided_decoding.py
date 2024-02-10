from enum import Enum
from typing import Union
from types import SimpleNamespace
from pydantic import BaseModel
try:
    from outlines.serve.vllm import JSONLogitsProcessor, RegexLogitsProcessor
except ImportError as e:
    raise ValueError(
        "Please install 'outlines' (pip install outlines) to use guided generation."
    ) from e


class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    # TODO: add grammar, choice


def get_guided_decoding_logits_processor(guided_spec: Union[str, dict, BaseModel], mode: GuidedDecodingMode, tokenizer):
    def dummy_llm():
        x = SimpleNamespace()
        y = SimpleNamespace()
        x.tokenizer = tokenizer
        y.tokenizer = x
        return y

    if mode == GuidedDecodingMode.JSON:
        assert isinstance(guided_spec, (str, dict, BaseModel)), "JSON schema error"
        return [JSONLogitsProcessor(guided_spec, dummy_llm())]
    elif mode == GuidedDecodingMode.REGEX:
        assert isinstance(guided_spec, str), "Regex must be string"
        return [RegexLogitsProcessor(guided_spec, dummy_llm())]
    else:
        return None