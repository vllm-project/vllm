# SPDX-License-Identifier: Apache-2.0
from re import escape as regex_escape

import llguidance
from transformers import PreTrainedTokenizerBase

from vllm.model_executor.guided_decoding.guidance_logits_processors import (
    GuidanceLogitsProcessor)
from vllm.sampling_params import GuidedDecodingParams


def get_local_guidance_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer: PreTrainedTokenizerBase) -> GuidanceLogitsProcessor:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    """

    grm = ""
    any_whitespace = 'disable-any-whitespace' not in \
        guided_params.backend_options()
    if guided_params.json:
        grm = llguidance.LLMatcher.grammar_from_json_schema(
            guided_params.json,
            overrides={"whitespace_pattern": guided_params.whitespace_pattern},
            defaults={
                "whitespace_flexible": any_whitespace,
            })
    elif guided_params.json_object:
        grm = llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            overrides={"whitespace_pattern": guided_params.whitespace_pattern},
            defaults={
                "whitespace_flexible": any_whitespace,
            })
    elif guided_params.regex:
        grm = llguidance.grammar_from("regex", guided_params.regex)
    elif guided_params.choice:
        # choice just uses regex
        choices = (regex_escape(str(choice))
                   for choice in guided_params.choice)
        choices_regex = "(" + "|".join(choices) + ")"
        grm = llguidance.grammar_from("regex", choices_regex)
    elif guided_params.grammar:
        # this supports Lark and GBNF
        grm = llguidance.grammar_from("grammar", guided_params.grammar)

    if grm:
        return GuidanceLogitsProcessor(grm, tokenizer)

    raise ValueError("Unknown guided decoding mode")
