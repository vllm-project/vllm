import asyncio
import concurrent.futures
from copy import copy
from enum import Enum
from functools import lru_cache
from json import dumps as json_dumps
from re import escape as regex_escape
from typing import Union, Tuple
from types import SimpleNamespace
from pydantic import BaseModel

from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from vllm.model_executor.guided_logits_processors import JSONLogitsProcessor, RegexLogitsProcessor


class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    CHOICE = "choice"


async def get_guided_decoding_logits_processor(
        request: Union[CompletionRequest, ChatCompletionRequest],
        tokenizer
    ) -> Union[JSONLogitsProcessor, RegexLogitsProcessor]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    guide, mode = _get_guide_and_mode(request)
    if not guide:
        return None
    loop = asyncio.get_running_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result = await loop.run_in_executor(
            pool, _get_cached_logits_processor,
            guide, tokenizer, mode
        )
        logits_processor = copy(result)
        # reset logits processor's internal state
        logits_processor.init_state()
        return [logits_processor]
    

def _get_guide_and_mode(
        request: Union[CompletionRequest, ChatCompletionRequest]
    ) -> Tuple[str, GuidedDecodingMode]:
    # validate guided decoding parameters
    guide_count = sum([
        request.guided_json is not None,
        request.guided_regex is not None,
        request.guided_choice is not None
    ])
    if guide_count == 0:
        return None, None
    elif guide_count > 1:
        raise ValueError(
            "You can only use one kind of guided decoding "
            "('guided_json', 'guided_regex' or 'guided_choice')."
        )
    
    if request.guided_json:
        if not isinstance(request.guided_json, (str, dict, BaseModel)):
            raise TypeError("JSON schema must be str, dict, or BaseModel")
            
        json = request.guided_json
        if isinstance(json, dict):
            # turn dict into hashable string
            json = json_dumps(json, sort_keys=True)
        elif isinstance(json, BaseModel):
            # use pydantic signature so that different model classes
            # with the same fields will get hashed the same
            json = str(json.__signature__)
        return json, GuidedDecodingMode.JSON
        
    elif request.guided_regex:
        if not isinstance(request.guided_regex, str):
            raise TypeError("Regex must be string")
        return request.guided_regex, GuidedDecodingMode.REGEX
    
    elif request.guided_choice:
        if not isinstance(request.guided_choice, list):
            raise TypeError("Choices must be a list")
        
        # choice just uses regex
        choices = [regex_escape(str(choice)) 
                for choice in request.guided_choice]
        choices_regex = "(" + "|".join(choices) + ")"
        return choices_regex, GuidedDecodingMode.CHOICE


@lru_cache(maxsize=32)
def _get_cached_logits_processor(guide: str, tokenizer, mode: GuidedDecodingMode):
    def dummy_llm():
        # outlines' logit processor takes i"n a LLMEngine object
        # to grab the LLM's tokenizer, may break in future
        # NOTE: as of 2/17, outlines PR 541 gets this wrong"
        x = SimpleNamespace()
        y = SimpleNamespace()
        x.tokenizer = tokenizer
        y.tokenizer = x
        return y
    
    if mode == GuidedDecodingMode.JSON:
        return JSONLogitsProcessor(guide, dummy_llm())
    elif mode == GuidedDecodingMode.REGEX or mode == GuidedDecodingMode.CHOICE:
        return RegexLogitsProcessor(guide, dummy_llm())
    else:
        raise RuntimeError(f"Unknown guided decoding mode {mode}")
