import asyncio
import concurrent.futures
from enum import Enum
from json import dumps as json_dumps
from re import escape as regex_escape
from typing import Optional, Tuple, Union

from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest)
from vllm.model_executor.guided_decoding.outlines_logits_processors import (
    CFGLogitsProcessor, JSONLogitsProcessor, RegexLogitsProcessor)
from vllm.sampling_params import LogitsProcessor, LogitsProcessorFactory


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

class OutlinesDecodingLogitsProcessorFactory(LogitsProcessorFactory):

    def __init__(self, guide: str, tokenizer: PreTrainedTokenizerBase,
                 mode: GuidedDecodingMode, whitespace_pattern: Union[str,
                                                                     None]):
        self.guide = guide
        self.tokenizer = tokenizer
        self.mode = mode
        self.whitespace_pattern = whitespace_pattern

    def get_processor(self) -> LogitsProcessor:
        return _get_logits_processor(self.guide, self.tokenizer, self.mode,
                                         self.whitespace_pattern)

    async def get_processor_async(self) -> LogitsProcessor:
        global global_thread_pool
        if global_thread_pool is None:
            global_thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=2)
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(global_thread_pool,
                                            _get_logits_processor,
                                            self.guide, self.tokenizer,
                                            self.mode, self.whitespace_pattern)


async def get_outlines_guided_decoding_logits_processor(
    request: Union[CompletionRequest,
                   ChatCompletionRequest], tokenizer: PreTrainedTokenizerBase
) -> Optional[OutlinesDecodingLogitsProcessorFactory]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    global global_thread_pool
    guide, mode = _get_guide_and_mode(request)
    if not guide or not mode:
        return None
    
    return OutlinesDecodingLogitsProcessorFactory(
        guide, tokenizer, mode, request.guided_whitespace_pattern)


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


def _get_logits_processor(
    guide: str, tokenizer: PreTrainedTokenizerBase, mode: GuidedDecodingMode,
    whitespace_pattern: Union[str, None]
) -> Union[JSONLogitsProcessor, RegexLogitsProcessor, CFGLogitsProcessor]:
    if mode == GuidedDecodingMode.JSON:
        return JSONLogitsProcessor(guide, tokenizer, whitespace_pattern)
    elif mode == GuidedDecodingMode.REGEX or mode == GuidedDecodingMode.CHOICE:
        return RegexLogitsProcessor(guide, tokenizer)
    elif mode == GuidedDecodingMode.GRAMMAR:
        return CFGLogitsProcessor(guide, tokenizer)
    else:
        raise ValueError(f"Unknown guided decoding mode {mode}")
