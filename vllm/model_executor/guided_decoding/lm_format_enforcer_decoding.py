# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from json import loads as json_loads
from typing import Optional, Union

from lmformatenforcer import (CharacterLevelParser, JsonSchemaParser,
                              RegexParser, StringParser,
                              TokenEnforcerTokenizerData, UnionParser)
from lmformatenforcer.integrations.vllm import (
    build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data)
from transformers import PreTrainedTokenizerBase

from vllm.logits_process import LogitsProcessor
from vllm.sampling_params import GuidedDecodingParams


def get_local_lm_format_enforcer_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer) -> Optional[LogitsProcessor]:
    """
    Given an OpenAI-compatible request, check for guided decoding parameters
    and get the necessary logits processor for the given guide.
    We cache logit processors by (guide, tokenizer), and on cache hit
    we make a shallow copy to reuse the same underlying FSM.
    """

    tokenizer_data = _cached_build_vllm_token_enforcer_tokenizer_data(
        tokenizer)
    character_level_parser: CharacterLevelParser
    if guided_params.json:
        schema_dict = _normalize_json_schema_object(guided_params.json)
        character_level_parser = JsonSchemaParser(schema_dict)
    elif guided_params.choice:
        character_level_parser = UnionParser(
            [StringParser(choice) for choice in guided_params.choice])
    elif guided_params.regex:
        character_level_parser = RegexParser(guided_params.regex)
    elif guided_params.grammar:
        # CFG grammar not supported by LMFE
        raise ValueError("Cannot construct a guided decoding logits processor"
                         " using the grammar option with the"
                         " lm_format_enforcer backend.")
    elif guided_params.json_object:
        # None means any json object
        character_level_parser = JsonSchemaParser(None)
    else:
        return None

    logits_processor = build_vllm_logits_processor(tokenizer_data,
                                                   character_level_parser)
    return logits_processor


def _normalize_json_schema_object(schema: Union[str, dict]) -> dict:
    if isinstance(schema, str):
        return json_loads(schema)
    if isinstance(schema, dict):
        return schema
    raise AssertionError(f"Unsupported schema type {schema}")


@lru_cache
def _cached_build_vllm_token_enforcer_tokenizer_data(
        tokenizer: PreTrainedTokenizerBase) -> TokenEnforcerTokenizerData:
    return build_vllm_token_enforcer_tokenizer_data(tokenizer)
