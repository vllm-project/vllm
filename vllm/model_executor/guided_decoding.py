import asyncio
import concurrent.futures
from copy import copy
from enum import Enum
from functools import lru_cache
from json import dumps as json_dumps
from re import escape as regex_escape
from typing import Union, Tuple
from pydantic import BaseModel

from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest
from vllm.model_executor.guided_logits_processors import JSONLogitsProcessor, RegexLogitsProcessor


class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    CHOICE = "choice"


global_thread_pool = None  # used for generating logits processor fsm


async def get_guided_decoding_logits_processor(
        request: Union[CompletionRequest, ChatCompletionRequest],
        tokenizer) -> Union[JSONLogitsProcessor, RegexLogitsProcessor]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    global global_thread_pool
    guide, mode = _get_guide_and_mode(request)
    if not guide:
        return None

    if global_thread_pool is None:
        global_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2)
    loop = asyncio.get_running_loop()

    result = await loop.run_in_executor(global_thread_pool,
                                        _get_cached_logits_processor, guide,
                                        tokenizer, mode)

    logits_processor = copy(result)
    # reset logits processor's internal state
    logits_processor.init_state()
    return logits_processor


def _get_guide_and_mode(
    request: Union[CompletionRequest, ChatCompletionRequest]
) -> Tuple[str, GuidedDecodingMode]:

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
        choices = [
            regex_escape(str(choice)) for choice in request.guided_choice
        ]
        choices_regex = "(" + "|".join(choices) + ")"
        return choices_regex, GuidedDecodingMode.CHOICE

    else:
        return None, None


@lru_cache(maxsize=32)
def _get_cached_logits_processor(guide: str, tokenizer,
                                 mode: GuidedDecodingMode):
    if mode == GuidedDecodingMode.JSON:
        return JSONLogitsProcessor(guide, tokenizer)
    elif mode == GuidedDecodingMode.REGEX or mode == GuidedDecodingMode.CHOICE:
        return RegexLogitsProcessor(guide, tokenizer)
    else:
        raise ValueError(f"Unknown guided decoding mode {mode}")
