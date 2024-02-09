from enum import Enum
from typing import Union
from pydantic import BaseModel
try:
    from outlines.serve.vllm import JSONLogitsProcessor, RegexLogitsProcessor
except ImportError:
    raise ValueError("Please install 'outlines' (pip install outlines) to use guided generation.")

from vllm.entrypoints.llm import LLM


class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    # TODO: add grammar, choice


def get_guided_decoding_logits_processor(guided_spec: Union[str, dict, BaseModel], mode: GuidedDecodingMode, tokenizer):
    def dummy_llm():
        return LLM(model="dummy", tokenizer=tokenizer)

    if mode == GuidedDecodingMode.JSON:
        assert type(guided_spec) in (str, dict, BaseModel), "JSON schema error"
        return [JSONLogitsProcessor(guided_spec, dummy_llm())]
    elif mode == GuidedDecodingMode.REGEX:
        assert type(guided_spec) is str, "Regex must be string"
        return [RegexLogitsProcessor(guided_spec, dummy_llm())]
    else:
        return None