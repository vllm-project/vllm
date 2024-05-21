import asyncio
import concurrent.futures
from copy import copy
from enum import Enum
from functools import lru_cache
from json import dumps as json_dumps
from re import escape as regex_escape
from typing import Tuple, Union

from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest)
from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    CFGLogitsProcessor, JSONLogitsProcessor, RegexLogitsProcessor)


class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    CHOICE = "choice"
    GRAMMAR = "grammar"


# https://github.com/outlines-dev/outlines/blob/main/outlines/grammars/json.lark
# the main difference is that we changed the start: value to
# start: object | array, so we are denying scalar values as the root of the
# JSON. Starting with scalars as the root seems to cause llama to generate
# without stop.
JSON_GRAMMAR = r"""
?start: object | array

?value: object
| array
| UNESCAPED_STRING
| SIGNED_NUMBER      -> number
| "true"             -> true
| "false"            -> false
| "null"             -> null

array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : UNESCAPED_STRING ":" value

%import common.UNESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

%ignore WS
"""

global_thread_pool = None  # used for generating logits processor fsm


async def get_outlines_guided_decoding_logits_processor(
        request: Union[CompletionRequest, ChatCompletionRequest],
        tokenizer) -> Union[JSONLogitsProcessor, RegexLogitsProcessor, None]:
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
                                        tokenizer, mode,
                                        request.guided_whitespace_pattern)

    logits_processor = copy(result)
    # reset logits processor's internal state
    logits_processor.init_state()
    return logits_processor


def _get_guide_and_mode(
    request: Union[CompletionRequest, ChatCompletionRequest]
) -> Union[Tuple[str, GuidedDecodingMode], Tuple[None, None]]:

    if request.guided_json:
        json = request.guided_json
        if isinstance(json, dict):
            # turn dict into hashable string
            json = json_dumps(json)
        elif isinstance(json, BaseModel):
            # use pydantic signature so that different model classes
            # with the same fields will get hashed the same
            json = str(json.__signature__)
        return json, GuidedDecodingMode.JSON
    elif request.guided_regex:
        return request.guided_regex, GuidedDecodingMode.REGEX
    elif request.guided_choice:
        # choice just uses regex
        choices = [
            regex_escape(str(choice)) for choice in request.guided_choice
        ]
        choices_regex = "(" + "|".join(choices) + ")"
        return choices_regex, GuidedDecodingMode.CHOICE
    elif request.guided_grammar:
        return request.guided_grammar, GuidedDecodingMode.GRAMMAR
    elif (request.response_format is not None
          and request.response_format.type == "json_object"):
        return JSON_GRAMMAR, GuidedDecodingMode.GRAMMAR
    else:
        return None, None


@lru_cache(maxsize=32)
def _get_cached_logits_processor(guide: str,
                                 tokenizer: PreTrainedTokenizerBase,
                                 mode: GuidedDecodingMode,
                                 whitespace_pattern: Union[str, None]):
    if mode == GuidedDecodingMode.JSON:
        return JSONLogitsProcessor(guide, tokenizer, whitespace_pattern)
    elif mode == GuidedDecodingMode.REGEX or mode == GuidedDecodingMode.CHOICE:
        return RegexLogitsProcessor(guide, tokenizer)
    elif mode == GuidedDecodingMode.GRAMMAR:
        return CFGLogitsProcessor(guide, tokenizer)
    else:
        raise ValueError(f"Unknown guided decoding mode {mode}")
