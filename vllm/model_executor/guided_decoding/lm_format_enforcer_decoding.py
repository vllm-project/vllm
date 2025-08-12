from functools import lru_cache
from json import loads as json_loads
from typing import Optional, Union

from lmformatenforcer import (CharacterLevelParser, JsonSchemaParser,
                              RegexParser, StringParser,
                              TokenEnforcerTokenizerData, UnionParser)
from lmformatenforcer.integrations.vllm import (
    build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data)
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest)
from vllm.model_executor.guided_decoding.outlines_decoding import (
    get_outlines_guided_decoding_logits_processor)
from vllm.sampling_params import LogitsProcessor


async def get_lm_format_enforcer_guided_decoding_logits_processor(
        request: Union[CompletionRequest, ChatCompletionRequest],
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
    if request.guided_json:
        schema = _normalize_json_schema_object(request.guided_json)
        character_level_parser = JsonSchemaParser(schema)
    elif request.guided_choice:
        character_level_parser = UnionParser(
            [StringParser(choice) for choice in request.guided_choice])
    elif request.guided_regex:
        character_level_parser = RegexParser(request.guided_regex)
    elif request.guided_grammar:
        # CFG grammar not supported by LMFE, revert to outlines
        return await get_outlines_guided_decoding_logits_processor(
            request, tokenizer)
    elif (request.response_format is not None
          and request.response_format.type == "json_object"):
        character_level_parser = JsonSchemaParser(
            None)  # None means any json object
    else:
        return None

    logits_processor = build_vllm_logits_processor(tokenizer_data,
                                                   character_level_parser)
    return logits_processor


def _normalize_json_schema_object(schema: Union[str, dict, BaseModel]) -> dict:
    if isinstance(schema, str):
        return json_loads(schema)
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, BaseModel):
        return schema.model_json_schema()


@lru_cache
def _cached_build_vllm_token_enforcer_tokenizer_data(
        tokenizer: PreTrainedTokenizerBase) -> TokenEnforcerTokenizerData:
    return build_vllm_token_enforcer_tokenizer_data(tokenizer)
