# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import LazyLoader
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    choice_as_grammar,
    compile_regex_with_timeout,
    grammar_is_likely_lark,
)

if TYPE_CHECKING:
    import xgrammar as xgr
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class XgrammarUnsupportedJsonFeaturesError(VLLMValidationError):
    """A JSON schema uses features the xgrammar backend does not support.

    Unlike a malformed schema, the request itself is valid: in `auto` backend
    mode the engine falls back to another structured-output backend instead
    of rejecting it.
    """


@dataclass
class XgrammarBackend(StructuredOutputBackend):
    def __post_init__(self):
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )

        if is_mistral_tokenizer(self.tokenizer):
            # NOTE: ideally, xgrammar should handle this accordingly.
            # refer to https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98
            stop_token_ids = [self.tokenizer.eos_token_id]

            # not self.tokenizer.vocab_size as self.tokenizer.vocab
            # collapses all decoded errors into a single token.
            self.vocab_size = len(self.tokenizer.vocab)
            tokenizer_info = xgr.TokenizerInfo(  # type: ignore
                encoded_vocab=self.tokenizer.vocab,
                # NOTE: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43 # noqa: E501
                vocab_type=xgr.VocabType.RAW
                if self.tokenizer.is_tekken
                else xgr.VocabType.BYTE_FALLBACK,
                vocab_size=self.vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
            )
        else:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
        self.compiler = xgr.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
            cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
        )

        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

    def compile_grammar(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        stop_token_ids: set[int] | None = None,
    ) -> StructuredOutputGrammar:
        # Note(arpera):
        # Our flag disable_any_whitespace does NOT map directly to
        # xgrammar's flag any_whitespace
        # To achieve desired behavior of disable_any_whitespace
        # we have to set not only any_whitespace
        # but also specify a list of separators after which
        # xgrammar must not insert spaces.
        # This is a requirement of xgrammar's API, so we must comply with it.
        #
        # FIXME(arpera):
        # Currently xgrammar v0.2.8 DOES emit spaces after comma
        # even if we specify it in separators list.
        # The bug has been reported to xgrammar team:
        # https://github.com/mlc-ai/xgrammar/issues/945
        # Please, track that issue, and once it is resolved remove this comment.
        # Upd. this bug was fixed in xgrammar main branch on Oct 8, 2026
        # and will be available in next release.
        # So, remove this comment once xgrammar updates to v0.2.9
        separators = (",", ":") if self.disable_any_whitespace else None
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec,
                any_whitespace=not self.disable_any_whitespace,
                separators=separators,
            )
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}',
                any_whitespace=not self.disable_any_whitespace,
                separators=separators,
            )
        elif request_type == StructuredOutputOptions.GRAMMAR:
            if grammar_is_likely_lark(grammar_spec):
                ctx = self.compiler.compile_lark(grammar_spec)
            else:
                ctx = self.compiler.compile_grammar(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = compile_regex_with_timeout(
                self.compiler.compile_regex,
                grammar_spec,
            )
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            s_tag = json.loads(grammar_spec)
            if "structures" in s_tag:
                # Falling back to deprecated method of compiling structural tag
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                ctx = self.compiler.compile_structural_tag(tags, s_tag["triggers"])
            else:
                ctx = self.compiler.compile_structural_tag(grammar_spec)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                override_stop_tokens=list(stop_token_ids) if stop_token_ids else None,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def allocate_token_bitmask(self, max_num_seqs: int):
        return xgr.allocate_token_bitmask(max_num_seqs, self.vocab_size)

    def destroy(self):
        del self.compiler


@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    # NOTE: This would be a generic-enough class for
    # supporting different backends, in the future.
    # For now, just xgrammar.
    #
    # https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.GrammarMatcher.find_jump_forward_string
    # for jump-forward decoding

    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(
        default_factory=lambda: 0, repr=False, hash=False, init=False
    )
    _is_terminated: bool = field(default=False, repr=False, hash=False)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored. Returns False if the FSM
        failed to advance.
        """
        if self._is_terminated:
            return True
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            self._is_terminated = self.matcher.is_terminated()
            if self._is_terminated:
                break
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if self._is_terminated:
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if self.matcher.is_terminated():
                    break
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens

    def rollback(self, num_tokens: int) -> None:
        self.matcher.rollback(num_tokens)
        self.num_processed_tokens -= num_tokens
        self._is_terminated = self.matcher.is_terminated()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(bitmask, idx)

    def is_terminated(self) -> bool:
        return self._is_terminated

    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        self._is_terminated = False


# cf https://github.com/mlc-ai/xgrammar/blob/a32ac892676d2eedc0327416105b9b06edfb94b2/cpp/json_schema_converter.cc
STRING_SUPPORTED_FORMATS = {
    "email",
    "date",
    "time",
    "date-time",
    "duration",
    "ipv4",
    "ipv6",
    "hostname",
    "uuid",
    "uri",
    "uri-reference",
    "uri-template",
    "json-pointer",
    "relative-json-pointer",
}


def _has_pattern_and_length_bounds(schema: dict[str, Any]) -> bool:
    return ("pattern" in schema or "format" in schema) and (
        "minLength" in schema or "maxLength" in schema
    )


# FIXME(arpera): The approach used here needs to be redesigned because of
# existing bugs: https://github.com/vllm-project/vllm/issues/57550
def _schema_types(schema: dict[str, Any]) -> set[str]:
    """Normalize a scalar or list-valued JSON Schema type."""
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return {schema_type}
    if isinstance(schema_type, list):
        return {item for item in schema_type if isinstance(item, str)}
    return set()


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        schema_types = _schema_types(obj)

        # integer/number + multipleOf is unsupported by xgrammar
        # This is known behavior and xgrammar emits warning in logs:
        #   [21:18:08] /project/cpp/json_schema_converter.cc:1053:
        #   Warning: multipleOf is not supported for type:number; ignoring multipleOf
        # This warning was added in PR
        # https://github.com/mlc-ai/xgrammar/pull/670
        # So, no need to track progress on this
        if (schema_types & {"integer", "number"}) and ("multipleOf" in obj):
            return True

        # array + some constraints is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/968
        if "array" in schema_types and any(
            key in obj
            for key in ("uniqueItems", "contains", "minContains", "maxContains")
        ):
            return True

        # string + format with unsupported keywords
        # is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/967
        # See tests on this in test_backend_xgrammar.py
        # unsupported_string_schemas
        if (
            "string" in schema_types
            and "format" in obj
            and obj["format"] not in STRING_SUPPORTED_FORMATS
        ):
            return True

        # string + format/pattern + length constraint is unsupported by xgrammar
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/966
        # See tests on this in test_backend_xgrammar.py
        # unsupported_string_schemas
        if "string" in schema_types and _has_pattern_and_length_bounds(obj):
            return True

        # propertyNames is not supported in pair with some constraints
        # in xgrammar
        # See tests on this in test_backend_xgrammar.py
        # unsupported_propertyNames_combinations
        if "object" in schema_types and "propertyNames" in obj:
            # propertyNames + maxLength is unsupported by xgrammar
            # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/960
            property_names = obj.get("propertyNames")
            if isinstance(property_names, dict) and _has_pattern_and_length_bounds(
                property_names
            ):
                return True
            # propertyNames + patternProperties is unsupported by xgrammar
            # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/959
            if "patternProperties" in obj:
                return True
            # propertyNames + properties is unsupported by xgrammar
            # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/958
            if "properties" in obj:
                return True
            # propertyNames + unevaluatedProperties is unsupported by xgrammar
            # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/961
            if obj.get("unevaluatedProperties", True) is not True:
                return True

        # patternProperties is not supported in pair with some constraints
        # in xgrammar
        # See tests on this in test_backend_xgrammar.py
        # unsupported_patternProperties_combinations
        if "object" in schema_types and isinstance(obj.get("patternProperties"), dict):
            # patternProperties + properties is unsupported by xgrammar
            # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/964
            if "properties" in obj:
                return True
            # patternProperties + patternProperties is unsupported by xgrammar
            # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/965
            if len(obj["patternProperties"]) > 1:
                return True

        # Note(arpera):
        # Xgrammar lacks support of multi-branch allOf
        # For instance, this schema:
        # {
        #   "allOf": [
        #     { "type": "string" },
        #     { "enum": ["yes", "no"] }
        #   ]
        # }
        # would accept any kind of json, such as
        # "maybe", "", 42, {}, [], {"a": 1}, etc.
        # which is NOT what is expected.
        # Tracking issue: https://github.com/mlc-ai/xgrammar/issues/937
        allof = obj.get("allOf")
        if isinstance(allof, list) and len(allof) >= 2:
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


def _structural_tag_json_schemas(s_tag: Any) -> list[dict[str, Any]]:
    """Extract the JSON schemas embedded in a structural tag payload.

    Handles both the new shape ({"type": "structural_tag", "format": ...}),
    where schemas appear as {"type": "json_schema", "json_schema": {...}}
    nodes nested anywhere inside "format", and the legacy shape
    ({"structures": [{"begin", "schema", "end"}], "triggers": [...]}).

    Args:
        s_tag: The parsed structural tag payload.

    Returns:
        The embedded JSON schemas. Malformed payloads yield an empty list;
        payload validity is checked separately by the grammar compilation.

    """
    schemas: list[dict[str, Any]] = []
    if not isinstance(s_tag, dict):
        return schemas

    if "structures" in s_tag:
        structures = s_tag["structures"]
        if isinstance(structures, list):
            for structure in structures:
                if isinstance(structure, dict):
                    schema = structure.get("schema")
                    if isinstance(schema, dict):
                        schemas.append(schema)
        return schemas

    def _collect(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "json_schema":
                json_schema = node.get("json_schema")
                if isinstance(json_schema, dict):
                    schemas.append(json_schema)
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for item in node:
                _collect(item)

    _collect(s_tag.get("format"))
    return schemas


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    Raises VLLMValidationError if the request is not supported.
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    if so_params.regex:
        # A NUL byte is never meaningful in a regex pattern and is not handled
        # by xgrammar's native regex converter. Reject it here, before the
        # pattern reaches that native code; the try/except below does not cover
        # this case.
        if "\x00" in so_params.regex:
            raise ValueError(
                "structured_outputs.regex must not contain a NUL character ('\\x00')"
            )
        try:
            compile_regex_with_timeout(
                xgr.Grammar.from_regex,
                so_params.regex,
            )
        except Exception as err:
            raise VLLMValidationError(
                f"Failed to transform regex into a grammar: {err}"
            ) from err

    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise VLLMValidationError(
                f"Failed to transform choices into a grammar: {err}"
            ) from err
        so_params.choice = None
        so_params.grammar = choice_grammar
        return

    if so_params.json:
        if isinstance(so_params.json, str):
            try:
                schema = json.loads(so_params.json)
            except json.JSONDecodeError as e:
                raise VLLMValidationError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise XgrammarUnsupportedJsonFeaturesError(
                "The provided JSON schema contains features not supported by xgrammar."
            )

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise VLLMValidationError(
                f"Failed to transform json schema into a grammar: {err}"
            ) from err
        return

    if so_params.grammar:
        # Parse the grammar with the same syntax `compile_grammar` will use,
        # but don't compile it. The grammar is passed on unchanged.
        try:
            if grammar_is_likely_lark(so_params.grammar):
                xgr.Grammar.from_lark(so_params.grammar)
            else:
                xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise VLLMValidationError("Invalid grammar specification.") from e
        return

    if so_params.structural_tag:
        check_json_nesting(so_params.structural_tag, structural_tag=True)
        try:
            s_tag = json.loads(so_params.structural_tag)
        except Exception as e:
            raise VLLMValidationError("Invalid structural tag specification.") from e

        # xgrammar silently ignores some JSON-schema keywords (e.g.
        # multipleOf) when it compiles a structural tag, so a nested schema
        # with unsupported features would validate successfully while its
        # constraint is dropped from the compiled grammar. Mirror the
        # plain-`json` branch: reject up front so backend="auto" falls back
        # to guidance/outlines instead of serving the wrong grammar.
        for schema in _structural_tag_json_schemas(s_tag):
            if has_xgrammar_unsupported_json_features(schema):
                raise XgrammarUnsupportedJsonFeaturesError(
                    "The provided JSON schema contains features not supported "
                    "by xgrammar."
                )

        try:
            # Using the deprecated method of compiling structural tag
            if "structures" in s_tag:
                tags = [
                    xgr.StructuralTagItem(
                        begin=s["begin"],
                        schema=json.dumps(s["schema"]),
                        end=s["end"],
                    )
                    for s in s_tag["structures"]
                ]
                xgr.Grammar.from_structural_tag(tags, s_tag["triggers"])
            else:
                xgr.Grammar.from_structural_tag(so_params.structural_tag)
        except Exception as e:
            raise VLLMValidationError("Invalid structural tag specification.") from e
