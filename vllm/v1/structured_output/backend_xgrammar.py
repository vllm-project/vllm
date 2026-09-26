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
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace
            )
        elif request_type == StructuredOutputOptions.JSON_OBJECT:
            ctx = self.compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=not self.disable_any_whitespace
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

        # Check for numeric ranges
        if (schema_types & {"integer", "number"}) and ("multipleOf" in obj):
            return True

        # Check for array unsupported keywords
        if "array" in schema_types and any(
            key in obj
            for key in ("uniqueItems", "contains", "minContains", "maxContains")
        ):
            return True

        # Unsupported keywords for strings
        if (
            "string" in schema_types
            and "format" in obj
            and obj["format"] not in STRING_SUPPORTED_FORMATS
        ):
            return True

        # A string mixing a generative constraint (pattern or format) with
        # explicit length bounds. xgrammar compiles the pattern/format side
        # and silently drops minLength/maxLength from the grammar, so output
        # can violate the bound without any error surfacing. Verified against
        # the compiled EBNF: pattern/format grammars come out byte-identical
        # with and without the length keywords, while maxLength alone lowers
        # to {0, N} correctly.
        if "string" in schema_types and _has_pattern_and_length_bounds(obj):
            return True

        # propertyNames validates names, so it is a string schema even when it
        # omits "type", which is the form that escapes the check above.
        if (
            "object" in schema_types
            and isinstance(obj.get("propertyNames"), dict)
            and _has_pattern_and_length_bounds(obj["propertyNames"])
        ):
            return True

        # FIXME: propertyNames conflicts with properties/patternProperties/
        # additionalProperties/unevaluatedProperties under xgrammar.
        # https://github.com/mlc-ai/xgrammar/issues/826
        if (
            "object" in schema_types
            and "propertyNames" in obj
            and (
                "properties" in obj
                or "patternProperties" in obj
                or isinstance(obj.get("additionalProperties"), dict)
                or obj.get("unevaluatedProperties", True) is not True
            )
        ):
            return True

        # FIXME: multiple patternProperties, or patternProperties alongside
        # properties, conflict under xgrammar.
        if (
            "object" in schema_types
            and isinstance(obj.get("patternProperties"), dict)
            and ("properties" in obj or len(obj["patternProperties"]) > 1)
        ):
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


def _resolve_local_json_ref(root: dict[str, Any], ref: str) -> Any:
    """Resolve a local JSON pointer (``#/$defs/foo``) against `root`.

    Returns None if the pointer is remote or dangling; both are left to
    xgrammar, which already reports them with a usable message.
    """
    if not ref.startswith("#"):
        return None
    pointer = ref[1:].strip("/")
    node: Any = root
    if not pointer:
        return node
    for part in pointer.split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def has_non_terminating_ref_cycle(schema: dict[str, Any]) -> bool:
    """Check whether a JSON schema contains a `$ref` cycle with no base case.

    A cycle such as ``{"$ref": "#/$defs/n", "$defs": {"n": {"$ref":
    "#/$defs/n"}}}`` describes a language with no finite member. xgrammar
    compiles it without complaint, but the resulting grammar allows no token
    at all, so the request is admitted, scheduled, and then killed on its
    first token with "Failed to advance FSM". Rejecting it here turns that
    mid-generation 500 into a request-time 400 (#57725).

    The check is deliberately narrow: it only rejects a schema whose `$ref`
    chain cycles without ever reaching a construct that emits a token, which
    is precisely the case xgrammar compiles into an empty language. Ordinary
    recursion - a self-referential object or array, with or without a base
    case - always emits a token before recursing and is left alone.
    """
    if not isinstance(schema, dict):
        return False

    refs: set[str] = set()

    def collect_refs(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                refs.add(ref)
            for value in node.values():
                collect_refs(value)
        elif isinstance(node, list):
            for value in node:
                collect_refs(value)

    collect_refs(schema)
    if not refs:
        return False

    # Least fixpoint: assume every `$ref` is non-terminating, then promote a
    # ref to terminating once its target is shown to be. Refs that never get
    # promoted are the ones with no base case.
    terminating: dict[str, bool] = dict.fromkeys(refs, False)

    def terminates(node: Any) -> bool:
        if not isinstance(node, dict):
            # Boolean schemas and non-schema values impose no constraint we
            # can reason about; assume they terminate rather than reject.
            return True

        ref = node.get("$ref")
        if isinstance(ref, str):
            # An unresolvable ref is xgrammar's to report, not ours.
            return terminating.get(ref, True)

        for key in ("anyOf", "oneOf"):
            branches = node.get(key)
            if (
                isinstance(branches, list)
                and branches
                and not any(terminates(branch) for branch in branches)
            ):
                return False

        # "allOf" is deliberately not inspected: xgrammar does not fully
        # compose multi-branch allOf (it warns "Support for allOf with
        # multiple options is still ongoing") and still admits a first token
        # for a cyclic branch, so treating it as non-terminating here would
        # reject a schema the engine currently serves.

        # Anything else - an object, an array, a scalar type - emits at least
        # one token before it can recurse, so the FSM has a legal first token
        # and this check does not apply. A schema like {"type": "object",
        # "properties": {"c": {"$ref": "#/$defs/n"}}, "required": ["c"]} is
        # also unsatisfiable, but it fails by running to max_tokens rather
        # than by rejecting token 0, so it is deliberately out of scope here.
        return True

    # Each pass promotes at least one ref or the set has stabilised.
    for _ in range(len(refs)):
        changed = False
        for ref in refs:
            if terminating[ref]:
                continue
            target = _resolve_local_json_ref(schema, ref)
            if target is None or terminates(target):
                terminating[ref] = True
                changed = True
        if not changed:
            break

    return not terminates(schema)


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
            raise VLLMValidationError(
                "The provided JSON schema contains features not supported by xgrammar."
            )

        # xgrammar compiles a `$ref` cycle with no base case into a grammar
        # that matches nothing, which only surfaces as an FSM failure on the
        # first token. Reject it here so the caller gets a 400 (#57725).
        if has_non_terminating_ref_cycle(schema):
            raise VLLMValidationError(
                "The provided JSON schema contains a '$ref' cycle with no base "
                "case, so no output can ever satisfy it. Give the recursion a "
                "terminating branch, for example by making the recursive "
                "property optional or adding a non-recursive 'anyOf' branch."
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
        try:
            s_tag = json.loads(so_params.structural_tag)

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
