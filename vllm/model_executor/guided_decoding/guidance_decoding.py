from enum import Enum
from re import escape as regex_escape
from typing import Union

from transformers import PreTrainedTokenizerBase

from vllm.model_executor.guided_decoding.guidance_logits_processors import (
    GuidanceLogitsProcessor)
from vllm.sampling_params import GuidedDecodingParams


class GuidedDecodingMode(Enum):
    JSON = "json"
    REGEX = "regex"
    CHOICE = "choice"
    GRAMMAR = "grammar"


def get_local_guidance_guided_decoding_logits_processor(
    guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizerBase
) -> Union[GuidanceLogitsProcessor, None]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """
    guide = None
    mode = None

    if guided_params.json:
        guide = guided_params.json
        mode = GuidedDecodingMode.JSON.value
    elif guided_params.regex:
        guide = guided_params.regex
        mode = GuidedDecodingMode.REGEX.value
    elif guided_params.choice:
        # choice just uses regex
        choices = (regex_escape(str(choice))
                   for choice in guided_params.choice)
        choices_regex = "(" + "|".join(choices) + ")"
        guide = choices_regex
        mode = GuidedDecodingMode.CHOICE.value
    elif guided_params.grammar:
        guide = guided_params.grammar
        mode = GuidedDecodingMode.GRAMMAR.value

    if not guide or not mode:
        return None

    return GuidanceLogitsProcessor(mode, guide, tokenizer,
                                   guided_params.whitespace_pattern)
