# SPDX-License-Identifier: Apache-2.0
from re import escape as regex_escape
from typing import Union

from transformers import PreTrainedTokenizerBase

from vllm.model_executor.guided_decoding.guidance_logits_processors import (
    ChoiceGuidanceLogitsProcessor, GrammarGuidanceLogitsProcessor,
    GuidanceLogitsProcessor, JsonGuidanceLogitsProcessor,
    RegexGuidanceLogitsProcessor)
from vllm.sampling_params import GuidedDecodingParams


def get_local_guidance_guided_decoding_logits_processor(
    guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizerBase
) -> Union[GuidanceLogitsProcessor, None]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """

    if guided_params.json:
        return JsonGuidanceLogitsProcessor(guided_params.json, tokenizer,
                                           guided_params.whitespace_pattern)
    elif guided_params.json_object:
        return JsonGuidanceLogitsProcessor('{"type": "object"}', tokenizer,
                                           guided_params.whitespace_pattern)
    elif guided_params.regex:
        return RegexGuidanceLogitsProcessor(guided_params.regex, tokenizer,
                                            None)
    elif guided_params.choice:
        # choice just uses regex
        choices = (regex_escape(str(choice))
                   for choice in guided_params.choice)
        choices_regex = "(" + "|".join(choices) + ")"
        return ChoiceGuidanceLogitsProcessor(choices_regex, tokenizer, None)
    elif guided_params.grammar:
        return GrammarGuidanceLogitsProcessor(guided_params.grammar, tokenizer,
                                              None)

    return None
