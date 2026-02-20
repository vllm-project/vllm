# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
from transformers import PreTrainedTokenizerBase

from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

if TYPE_CHECKING:
    import lmformatenforcer
    import lmformatenforcer.integrations.vllm as lmfe_vllm
else:
    lmformatenforcer = LazyLoader("lmformatenforcer", globals(), "lmformatenforcer")
    lmfe_vllm = LazyLoader(
        "lmformatenforcer.integrations.vllm",
        globals(),
        "lmformatenforcer.integrations.vllm",
    )


@lru_cache
def _cached_build_vllm_token_enforcer_tokenizer_data(
    tokenizer: PreTrainedTokenizerBase, vocab_size: int
) -> "lmfe_vllm.TokenEnforcerTokenizerData":
    return lmfe_vllm.build_vllm_token_enforcer_tokenizer_data(
        tokenizer, use_bitmask=True, vocab_size=vocab_size
    )


@dataclass(slots=True)
class LMFormatEnforcerGrammar(StructuredOutputGrammar):
    token_enforcer: lmformatenforcer.TokenEnforcer
    current_tokens_prefix: list[int] = field(default_factory=list)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        original_len = len(self.current_tokens_prefix)
        for token in tokens:
            if not self.token_enforcer.get_allowed_tokens(
                self.current_tokens_prefix
            ).is_token_allowed(token):
                # Rollback partial updates to ensure atomicity.
                del self.current_tokens_prefix[original_len:]
                return False
            self.current_tokens_prefix.append(token)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        for prefix_length in range(len(tokens)):
            prefix = tokens[:prefix_length]
            next_token = tokens[prefix_length]
            if not self.token_enforcer.get_allowed_tokens(
                self.current_tokens_prefix + prefix
            ).is_token_allowed(next_token):
                break
        else:
            return tokens

        return tokens[:prefix_length]

    def rollback(self, num_tokens: int) -> None:
        self.current_tokens_prefix = self.current_tokens_prefix[:-num_tokens]

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        allowed_tokens = self.token_enforcer.get_allowed_tokens(
            self.current_tokens_prefix
        )
        bitmask[batch_index] = allowed_tokens.allowed_tokens

    def is_terminated(self) -> bool:
        # We are considered terminated if the prefix ends with eos_token_id
        return_value = (
            len(self.current_tokens_prefix) > 0
            and self.current_tokens_prefix[-1] == self.token_enforcer.eos_token_id
        )
        return return_value

    def reset(self):
        self.current_tokens_prefix = []


@dataclass(slots=True)
class LMFormatEnforcerBackend(StructuredOutputBackend):
    tokenizer_data: Any = field(init=False)

    def __post_init__(self):
        self.tokenizer_data = _cached_build_vllm_token_enforcer_tokenizer_data(
            self.tokenizer, self.vocab_size
        )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        character_level_parser: lmformatenforcer.CharacterLevelParser
        if request_type == StructuredOutputOptions.JSON:
            spec_dict = json.loads(grammar_spec)
            character_level_parser = lmformatenforcer.JsonSchemaParser(spec_dict)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            character_level_parser = lmformatenforcer.JsonSchemaParser(None)
        elif request_type == StructuredOutputOptions.REGEX:
            character_level_parser = lmformatenforcer.RegexParser(grammar_spec)
        elif request_type == StructuredOutputOptions.CHOICE:
            choices = ast.literal_eval(grammar_spec)
            character_level_parser = lmformatenforcer.UnionParser(
                [lmformatenforcer.StringParser(choice) for choice in choices]
            )
        else:
            raise ValueError(
                f"Invalid request type for LM Format Enforcer backend({request_type!s})"
            )
        max_rollback_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )

        if max_rollback_tokens > 0:
            raise ValueError(
                "LM Format Enforcer backend does not support speculative tokens"
            )

        token_enforcer = lmformatenforcer.TokenEnforcer(
            tokenizer_data=self.tokenizer_data,
            parser=character_level_parser,
        )
        return LMFormatEnforcerGrammar(token_enforcer)

    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor:
        return torch.full(
            (max_num_seqs, (self.vocab_size + 31) // 32),
            -1,
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )

    def destroy(self):
        pass


def validate_structured_output_request_lm_format_enforcer(params: SamplingParams):
    if params.structured_outputs is None:
        return

    so_params = params.structured_outputs

    if so_params.regex:
        return
    elif so_params.json:
        if isinstance(so_params.json, str):
            try:
                # make sure schema is valid json
                json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            try:
                json.dumps(so_params.json)
            except Exception as e:
                raise ValueError(
                    f"Error serializing structured outputs jsonschema: {e}"
                ) from e
        return
    elif so_params.choice:
        return
    elif so_params.grammar:
        raise ValueError(
            "LM Format Enforcer structured outputs backend "
            "does not support grammar specifications"
        )
