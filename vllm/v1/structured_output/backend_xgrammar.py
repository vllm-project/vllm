# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs
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
    get_xgrammar_disk_cache,
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

        # Optional persistent on-disk cache for compiled grammars. Off by
        # default; when enabled it lets cold starts deserialize grammars from
        # disk instead of recompiling them (see get_xgrammar_disk_cache).
        # tokenizer_info is kept because deserialize_json needs it on a hit.
        self.tokenizer_info = tokenizer_info
        self._disk_cache = get_xgrammar_disk_cache()
        self._tokenizer_key: str | None = None
        if self._disk_cache is not None:
            # Stable per-tokenizer discriminator for the cache key (computed
            # once). serialize_json() captures the full vocab + metadata, so
            # different tokenizers never share a key.
            self._tokenizer_key = hashlib.sha256(
                tokenizer_info.serialize_json().encode()
            ).hexdigest()

        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar:
        ctx = self._compile_grammar_ctx(request_type, grammar_spec)

        return XgrammarGrammar(
            matcher=xgr.GrammarMatcher(
                ctx,
                max_rollback_tokens=self.num_speculative_tokens,
            ),
            vocab_size=self.vocab_size,
            ctx=ctx,
        )

    def _compile_grammar_ctx(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> "xgr.CompiledGrammar":
        """Compile a grammar, consulting the optional persistent disk cache.

        When the disk cache is disabled (the default) this is exactly the
        original compile path with zero added overhead. When enabled, a hit
        deserializes the grammar from disk instead of recompiling; a miss
        recompiles and writes the serialized grammar back. A cache problem
        never fails the request -- it falls back to a fresh compile.
        """
        if self._disk_cache is None:
            return self._compile(request_type, grammar_spec)

        key = self._disk_key(request_type, grammar_spec)
        text = self._disk_cache.get(key, None)
        if text is not None:
            try:
                return xgr.CompiledGrammar.deserialize_json(text, self.tokenizer_info)
            except (
                xgr.InvalidJSONError,
                xgr.DeserializeFormatError,
                xgr.DeserializeVersionError,
            ):
                # Corrupt / format- or version-mismatched entry: recompile and
                # overwrite it below.
                logger.debug(
                    "Ignoring unusable xgrammar disk cache entry; recompiling.",
                    exc_info=True,
                )
            except Exception:
                # A cache problem must never fail a request.
                logger.warning(
                    "Unexpected error reading xgrammar disk cache; recompiling.",
                    exc_info=True,
                )

        ctx = self._compile(request_type, grammar_spec)
        try:
            self._disk_cache.set(key, ctx.serialize_json())
        except Exception:
            logger.warning("Failed to write xgrammar disk cache entry.", exc_info=True)
        return ctx

    def _disk_key(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> str:
        # INVARIANT: every input that changes the compiled grammar must be
        # folded into this key. deserialize_json re-validates only the
        # serialization version + tokenizer metadata, NOT compile flags, so a
        # missing factor would serve a valid-but-wrong grammar on a hit.
        # max_rollback_tokens is excluded on purpose: it parameterizes the
        # GrammarMatcher (rebuilt fresh every call), not the CompiledGrammar.
        return hashlib.sha256(
            "\x00".join(
                (
                    self._tokenizer_key,
                    request_type.name,
                    "1" if self.disable_any_whitespace else "0",
                    grammar_spec,
                )
            ).encode()
        ).hexdigest()

    def _compile(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> "xgr.CompiledGrammar":
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
        return ctx

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

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
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
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
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
        self.num_processed_tokens = 0
        self.matcher.reset()


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

        # Unsupported keywords for objects
        if obj.get("type") == "object" and any(
            key in obj for key in ("patternProperties", "propertyNames")
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


def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None:
    """Validate that the request is supported by structured output.

    Raises ValueError if the request is not supported.
    """
    if sampling_params.structured_outputs is None:
        return

    so_params = sampling_params.structured_outputs

    if so_params.regex:
        try:
            compile_regex_with_timeout(
                xgr.Grammar.from_regex,
                so_params.regex,
            )
        except Exception as err:
            raise ValueError(
                f"Failed to transform regex into a grammar: {err}"
            ) from err

    if so_params.choice:
        choice_grammar = choice_as_grammar(so_params.choice)
        try:
            xgr.Grammar.from_ebnf(choice_grammar)
        except Exception as err:
            raise ValueError(
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
                raise ValueError("Invalid JSON grammar specification.") from e
        else:
            schema = so_params.json

        if has_xgrammar_unsupported_json_features(schema):
            raise ValueError(
                "The provided JSON schema contains features not supported by xgrammar."
            )

        try:
            xgr.Grammar.from_json_schema(schema)
        except Exception as err:
            raise ValueError(
                f"Failed to transform json schema into a grammar: {err}"
            ) from err
        return

    if so_params.grammar:
        if grammar_is_likely_lark(so_params.grammar):
            # xgrammar supports EBNF grammars only
            try:
                so_params.grammar = convert_lark_to_ebnf(so_params.grammar)
            except ValueError as e:
                raise ValueError(
                    "Failed to convert the grammar from Lark to EBNF. "
                ) from e

        # Test parsing EBNF grammar, possibly already converted from Lark
        try:
            # parse the grammar, but we aren't compiling it.
            xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise ValueError("Invalid grammar specification.") from e
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
            raise ValueError("Invalid structural tag specification.") from e
