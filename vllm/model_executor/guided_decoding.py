from enum import Enum
from typing import Union
from types import SimpleNamespace
from pydantic import BaseModel

from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
try:
    from outlines.serve.vllm import JSONLogitsProcessor, RegexLogitsProcessor
except ImportError as e:
    raise ValueError(
        "Please install 'outlines' (pip install outlines) to use guided generation."
    ) from e


def get_guided_decoding_logits_processor(
        request: Union[CompletionRequest, ChatCompletionRequest],
        tokenizer
    ) -> Union[JSONLogitsProcessor, RegexLogitsProcessor]:

    def dummy_llm():
        # outlines' logit processor takes in a vllm.LLM object
        # to grab the LLM's tokenizer
        x = SimpleNamespace()
        y = SimpleNamespace()
        x.tokenizer = tokenizer
        y.tokenizer = x
        return y

    if request.guided_json:
        if not isinstance(request.guided_json, (str, dict, BaseModel)):
            raise TypeError("JSON schema must be str, dict, or BaseModel")
        return [JSONLogitsProcessor(request.guided_json, dummy_llm())]
    elif request.guided_regex:
        if not isinstance(request.guided_regex, str):
            raise TypeError("Regex must be string")
        return [RegexLogitsProcessor(request.guided_regex, dummy_llm())]
    elif request.guided_choice:
        if not isinstance(request.guided_choice, list):
            raise TypeError("Choices must be a list")
        # create regex from choices
        choices = [str_with_escape(choice) for choice in request.guided_choice]
        choices_regex = "(" + "|".join(choices) + ")"
        return [RegexLogitsProcessor(choices_regex, dummy_llm())]
    else:
        return None
    

def str_with_escape(e: Union[str, int, float, bool]):
    s = str(e)
    a = []
    regex_reserved = set(".()[]{}|*+?^$-\\")
    for ch in s:
        if ch in regex_reserved:
            a.append("\\")
        a.append(ch)
    return "".join(a)