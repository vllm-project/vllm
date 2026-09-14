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
    convert_lark_to_ebnf,
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


def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for numeric ranges
        if obj.get("type") in ("integer", "number") and ("multipleOf" in obj):
            return True

        # Check for array unsupported keywords
        if obj.get("type") == "array" and any(
            key in obj
            for key in ("uniqueItems", "contains", "minContains", "maxContains")
        ):
            return True

        # Unsupported keywords for strings
        if (
            obj.get("type") == "string"
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
        if (
            obj.get("type") == "string"
            and ("pattern" in obj or "format" in obj)
            and ("minLength" in obj or "maxLength" in obj)
        ):
            return True

        # Unsupported keywords for objects
        if obj.get("type") == "object" and any(
            key in obj for key in ("patternProperties", "propertyNames")
        ):
            return True

        # xgrammar enforces `allOf` only when it has a single branch. For
        # multiple branches it silently compiles to an "accept anything" rule,
        # dropping every constraint without surfacing an error.
        allof = obj.get("allOf")
        if isinstance(allof, list) and len(allof) > 1:
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


_ANNOTATION_KEYS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
    }
)


class _UnmergeableAllOf(Exception):
    """Raised when a multi-branch allOf cannot be flattened losslessly."""


def _lookup_local_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise _UnmergeableAllOf
    node: Any = root
    for raw_token in ref[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or token not in node:
            raise _UnmergeableAllOf
        node = node[token]
    if not isinstance(node, dict):
        raise _UnmergeableAllOf
    return node


def _is_object_schema(schema: dict[str, Any]) -> bool:
    return schema.get("type", "object") == "object"


def _merge_property_schemas(
    current: Any,
    incoming: Any,
    root: dict[str, Any],
    active_refs: frozenset[str],
) -> Any:
    if current == incoming:
        return current
    if not isinstance(current, dict) or not isinstance(incoming, dict):
        raise _UnmergeableAllOf
    if not _is_object_schema(current) or not _is_object_schema(incoming):
        raise _UnmergeableAllOf
    return _merge_object_schemas([current, incoming], root, active_refs)


def _merge_defs_map(dest: dict[str, Any], incoming: Any) -> None:
    if not isinstance(incoming, dict):
        raise _UnmergeableAllOf
    for name, defn in incoming.items():
        if name not in dest:
            dest[name] = defn
        elif dest[name] != defn:
            raise _UnmergeableAllOf


def _merge_object_schemas(
    schemas: list[dict[str, Any]],
    root: dict[str, Any],
    active_refs: frozenset[str],
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[Any] = []
    seen_required: set[Any] = set()
    defs: dict[str, Any] = {}
    definitions: dict[str, Any] = {}
    annotations: dict[str, Any] = {}

    for schema in schemas:
        if not isinstance(schema, dict):
            raise _UnmergeableAllOf
        for key, value in schema.items():
            if key == "type":
                if value != "object":
                    raise _UnmergeableAllOf
            elif key == "properties":
                if not isinstance(value, dict):
                    raise _UnmergeableAllOf
                for name, prop in value.items():
                    if name in properties:
                        properties[name] = _merge_property_schemas(
                            properties[name], prop, root, active_refs
                        )
                    else:
                        properties[name] = prop
            elif key == "required":
                if not isinstance(value, list):
                    raise _UnmergeableAllOf
                for item in value:
                    if item not in seen_required:
                        seen_required.add(item)
                        required.append(item)
            elif key == "$defs":
                _merge_defs_map(defs, value)
            elif key == "definitions":
                _merge_defs_map(definitions, value)
            elif key in _ANNOTATION_KEYS:
                if key not in annotations:
                    annotations[key] = value
            else:
                raise _UnmergeableAllOf

    result: dict[str, Any] = {"type": "object"}
    if properties:
        result["properties"] = {
            name: _flatten_node(prop, root, active_refs)
            for name, prop in properties.items()
        }
    if required:
        result["required"] = required
    if defs:
        result["$defs"] = defs
    if definitions:
        result["definitions"] = definitions
    result.update(annotations)
    return result


def _resolve_branch(
    root: dict[str, Any], branch: Any, active_refs: frozenset[str]
) -> tuple[dict[str, Any], frozenset[str]]:
    if not isinstance(branch, dict):
        raise _UnmergeableAllOf
    ref = branch.get("$ref")
    if ref is None:
        return branch, frozenset()
    if not isinstance(ref, str) or ref in active_refs:
        raise _UnmergeableAllOf
    target = _lookup_local_ref(root, ref)
    resolved, nested_refs = _resolve_branch(root, target, active_refs | {ref})
    inlined = nested_refs | {ref}
    siblings = {key: value for key, value in branch.items() if key != "$ref"}
    if not siblings:
        return resolved, inlined
    merged = _merge_object_schemas([resolved, siblings], root, active_refs | inlined)
    return merged, inlined


def _flatten_node(node: Any, root: dict[str, Any], active_refs: frozenset[str]) -> Any:
    if isinstance(node, dict):
        allof = node.get("allOf")
        if isinstance(allof, list) and len(allof) > 1:
            remaining = {key: value for key, value in node.items() if key != "allOf"}
            branches: list[dict[str, Any]] = []
            refs_inlined: frozenset[str] = frozenset()
            for branch in allof:
                resolved, refs = _resolve_branch(root, branch, active_refs)
                branches.append(resolved)
                refs_inlined |= refs
            if remaining:
                branches.append(remaining)
            child_refs = active_refs | refs_inlined
            node = _merge_object_schemas(branches, root, child_refs)
            return {
                key: _flatten_node(value, root, child_refs)
                for key, value in node.items()
            }
        return {
            key: _flatten_node(value, root, active_refs) for key, value in node.items()
        }
    if isinstance(node, list):
        return [_flatten_node(item, root, active_refs) for item in node]
    return node


def flatten_allof_branches(schema: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten multi-branch ``allOf`` combinators into a single object schema.

    Every object node with more than one ``allOf`` branch is merged into one
    object schema. Single-branch ``allOf`` is left as-is. ``$ref`` values are
    resolved only when they appear as an ``allOf`` branch; a bare ``$ref``
    node is not inlined. ``schema`` is not mutated.

    Args:
        schema: JSON schema to rewrite.

    Returns:
        A rewritten copy of ``schema``, or ``None`` if a multi-branch
        ``allOf`` cannot be merged losslessly.
    """
    try:
        flattened = _flatten_node(schema, schema, frozenset())
    except _UnmergeableAllOf:
        return None
    if not isinstance(flattened, dict):
        return None
    return flattened


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

        flattened = flatten_allof_branches(schema)
        if flattened is not None:
            schema = flattened
            if isinstance(so_params.json, str):
                so_params.json = json.dumps(flattened)
            else:
                so_params.json = flattened

        if has_xgrammar_unsupported_json_features(schema):
            raise VLLMValidationError(
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
        if grammar_is_likely_lark(so_params.grammar):
            # xgrammar supports EBNF grammars only
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise VLLMValidationError(
                    "Failed to convert the grammar from Lark to EBNF. "
                ) from e

        # Test parsing EBNF grammar, possibly already converted from Lark
        try:
            # parse the grammar, but we aren't compiling it.
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
