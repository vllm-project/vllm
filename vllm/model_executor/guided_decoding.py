from collections import defaultdict
from copy import copy
from functools import lru_cache
from json import dumps as json_dumps
from re import escape as regex_escape
from typing import Union
from types import SimpleNamespace
from pydantic import BaseModel

from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
try:
    from outlines.serve.vllm import JSONLogitsProcessor, RegexLogitsProcessor
except ImportError as e:
    raise ValueError(
        "Please install 'outlines' (pip install outlines) to use guided decoding."
    ) from e


def get_guided_decoding_logits_processor(
        request: Union[CompletionRequest, ChatCompletionRequest],
        tokenizer
    ) -> Union[JSONLogitsProcessor, RegexLogitsProcessor]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (json/regex, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying RegexFSM.
    """

    logits_processor = None
    if request.guided_json:
        if not isinstance(request.guided_json, (str, dict, BaseModel)):
            raise TypeError("JSON schema must be str, dict, or BaseModel")
        
        json = request.guided_json
        if isinstance(request.guided_json, dict):
            # turn dict into hashable string
            json = json_dumps(request.guided_json, sort_keys=True)
        elif isinstance(request.guided_json, BaseModel):
            # use pydantic signature so that different model classes
            # with the same fields will get hashed the same
            json = str(request.guided_json.__signature__)

        logits_processor = copy(get_cached_logits_processor(
            json, tokenizer, True))
    
    elif request.guided_regex:
        if not isinstance(request.guided_regex, str):
            raise TypeError("Regex must be string")
        
        logits_processor = copy(get_cached_logits_processor(
            request.guided_regex, tokenizer, False))
    
    elif request.guided_choice:
        if not isinstance(request.guided_choice, list):
            raise TypeError("Choices must be a list")
        
        # choice just uses regex
        choices = [regex_escape(str(choice)) 
                   for choice in request.guided_choice]
        choices_regex = "(" + "|".join(choices) + ")"

        logits_processor = copy(get_cached_logits_processor(
            choices_regex, tokenizer, False))
    
    if logits_processor:
        # reset logits processor's internal state
        logits_processor.fsm_state = defaultdict(int)
        return [logits_processor]
    else:
        return None
    

@lru_cache
def get_cached_logits_processor(guide: str, tokenizer, is_json: bool):
    # guide is guaranteed hashable (see above function)
    # tokenizer should be hashable right??

    def dummy_llm():
        # outlines' logit processor takes in a vllm.LLM object
        # to grab the LLM's tokenizer, may break in future
        x = SimpleNamespace()
        y = SimpleNamespace()
        x.tokenizer = tokenizer
        y.tokenizer = x
        return y
    
    if is_json:
        return JSONLogitsProcessor(guide, dummy_llm())
    else:
        return RegexLogitsProcessor(guide, dummy_llm())
