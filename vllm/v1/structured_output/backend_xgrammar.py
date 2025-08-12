# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer
from vllm.utils import LazyLoader
from vllm.v1.structured_output.backend_types import (StructuredOutputBackend,
                                                     StructuredOutputGrammar,
                                                     StructuredOutputOptions)
from vllm.v1.structured_output.utils import (choice_as_grammar,
                                             convert_lark_to_ebnf,
                                             grammar_is_likely_lark)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class XgrammarBackend(StructuredOutputBackend):

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.disable_any_whitespace = (
            "disable-any-whitespace"
            in vllm_config.decoding_config.guided_decoding_backend)
        tokenizer_group = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            lora_config=vllm_config.lora_config)  # type: ignore[arg-type]
        tokenizer_group.ping()

        tokenizer = tokenizer_group.get_lora_tokenizer(None)
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        if isinstance(tokenizer, MistralTokenizer):
            # NOTE: ideally, xgrammar should handle this accordingly.
            # refer to https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98
            try:
                if tokenizer.is_tekken:
                    encoded_vocab = tokenizer._vocab
                else:
                    encoded_vocab = [
                        token for token, _ in sorted(
                            tokenizer.get_vocab().items(),
                            key=lambda x: x[1],
                        )
                    ]
                stop_token_ids = None
                if hasattr(
                        tokenizer,
                        "eos_token_id",
                ) and tokenizer.eos_token_id is not None:
                    stop_token_ids = [tokenizer.eos_token_id]
            except AttributeError as e:
                raise ValueError(
                    f"Cannot get the vocabulary of the tokenizer "
                    f"{type(tokenizer)}. The tokenizer should have a "
                    "get_vocab method.") from e
            tokenizer_info = xgr.TokenizerInfo(  # type: ignore
                encoded_vocab=encoded_vocab,
                # NOTE: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43 # noqa: E501
                vocab_type=xgr.VocabType.RAW
                if tokenizer.is_tekken else xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        else:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer,
                vocab_size=self.vocab_size,
            )
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

    def compile_grammar(self, request_type: StructuredOutputOptions,
                        grammar_spec: str) -> StructuredOutputGrammar:
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}',
                any_whitespace=not self.disable_any_whitespace)
        elif request_type == StructuredOutputOptions.GRAMMAR:
            ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})")

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(ctx),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # TODO: support max_rollback_tokens
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(default_factory=lambda: 0,
                                      repr=False,
                                      hash=False,
                                      init=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.", request_id, token)
                return False
            self.num_processed_tokens += 1
        return True

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self.matcher.is_terminated()

    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for pattern restrictions
        if "pattern" in obj:
            return True

        # Check for numeric ranges
        if obj.get("type") in ("integer", "number") and any(
                key in obj
                for key in ("minimum", "maximum", "exclusiveMinimum",
                            "exclusiveMaximum", "multipleOf")):
            return True

        # Check for array unsupported keywords
        if obj.get("type") == "array" and any(
                key in obj
                for key in ("uniqueItems", "contains", "minContains",
                            "maxContains", "minItems", "maxItems")):
            return True

        # Unsupported keywords for strings
        if obj.get("type") == "string" and "format" in obj:
            return True

        # Unsupported keywords for objects
        if obj.get("type") == "object" and any(
                key in obj for key in ("minProperties", "maxProperties",
                                       "propertyNames", "patternProperties")):
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.guided_decoding is None:
        return

    gd_params = sampling_params.guided_decoding

    if gd_params.regex:
        try:
            xgr.Grammar.from_regex(gd_params.regex)
        except Exception as err:
            raise ValueError("Failed to transform regex into a grammar: "
                             f"{err}") from err

    if gd_params.choice:
        choice_grammar = choice_as_grammar(gd_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError("Failed to transform choices into a grammar: "
                             "{err}") from err
        gd_params.choice = None
        gd_params.grammar = choice_grammar
        return

    if gd_params.json:
        if isinstance(gd_params.json, str):
            try:
                schema = json.loads(gd_params.json)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = gd_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError("The provided JSON schema contains features not "
                             "supported by xgrammar.")
        return

    if gd_params.grammar:
        if grammar_is_likely_lark(gd_params.grammar):
            # xgrammar supports EBNF grammars only
            try:
                gd_params.grammar = convert_lark_to_ebnf(gd_params.grammar)
            except ValueError as e:
                raise ValueError(
                    "Failed to convert the grammar from Lark to EBNF. ") from e

        # Test parsing EBNF grammar, possibly already converted from Lark
        try:
            # parse the grammar, but we aren't compiling it.
            xgr.Grammar.from_ebnf(gd_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
