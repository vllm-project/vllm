# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers import MistralCommonBackend

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
from vllm.v1.structured_output.request import get_structured_output_key

if TYPE_CHECKING:
    import llguidance
    import llguidance.hf as llguidance_hf
    import llguidance.torch as llguidance_torch
else:
    llguidance = LazyLoader("llguidance", globals(), "llguidance")
    llguidance_hf = LazyLoader("llguidance.hf", globals(), "llguidance.hf")
    llguidance_torch = LazyLoader("llguidance.torch", globals(), "llguidance.torch")

logger = init_logger(__name__)


def _walk_json_for_additional_properties(data: object):
    if isinstance(data, dict):
        for value in data.values():
            _walk_json_for_additional_properties(value)
        if "additionalProperties" not in data and (
            "properties" in data or "patternProperties" in data
        ):
            data["additionalProperties"] = False
    elif isinstance(data, list):
        for item in data:
            _walk_json_for_additional_properties(item)


def has_guidance_unsupported_json_features(schema: dict[str, Any]) -> bool:
    """Check if JSON schema contains features unsupported by guidance/llguidance."""

    def check_object(obj: dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False

        # patternProperties is not supported by llguidance
        if "patternProperties" in obj:
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


def process_for_additional_properties(
    guide_json: str | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(guide_json, str):
        guide_json_obj = json.loads(guide_json)
    else:
        # copy for modifications
        guide_json_obj = copy.deepcopy(guide_json)
    _walk_json_for_additional_properties(guide_json_obj)
    return guide_json_obj


@dataclass
class GuidanceBackend(StructuredOutputBackend):
    def __post_init__(self):
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )
        self.disable_additional_properties = (
            self.vllm_config.structured_outputs_config.disable_additional_properties
        )

        if is_mistral_tokenizer(self.tokenizer):
            self.ll_tokenizer = self.tokenizer.llg_tokenizer
        elif isinstance(self.tokenizer, MistralCommonBackend):
            from mistral_common.guidance.tokenizer import from_mistral_tokenizer

            self.ll_tokenizer = from_mistral_tokenizer(self.tokenizer.tokenizer)
        else:
            self.ll_tokenizer = llguidance_hf.from_tokenizer(
                self.tokenizer, max(self.vocab_size, len(self.tokenizer))
            )

    def compile_grammar(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        stop_token_ids: set[int] | None = None,
    ) -> StructuredOutputGrammar:
        self.serialized_grammar = serialize_guidance_grammar(
            request_type,
            grammar_spec,
            self.disable_any_whitespace,
            self.disable_additional_properties,
        )

        ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer,
            self.serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )

        r = GuidanceGrammar(
            ll_matcher=ll_matcher,
            ll_tokenizer=self.ll_tokenizer,
            vocab_size=self.vocab_size,
        )

        r.check_error()
        return r

    def allocate_token_bitmask(self, max_num_seqs: int):
        return llguidance_torch.allocate_token_bitmask(
            max_num_seqs, self.ll_tokenizer.vocab_size
        )

    def destroy(self):
        pass


@dataclass
class GuidanceGrammar(StructuredOutputGrammar):
    ll_matcher: llguidance.LLMatcher
    ll_tokenizer: llguidance.LLTokenizer
    vocab_size: int
    printed_error: bool = False
    terminated: bool = False
    rollback_lag: int = 0

    def check_error(self):
        if not self.printed_error:
            err = self.ll_matcher.get_error()
            if err:
                self.printed_error = True
                logger.warning("LLMatcher error: %s", err)

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the parser.

        Returns True if the parser was advanced successfully.
        Returns False if the parser failed to advance.
        """

        if self.ll_tokenizer.eos_token in tokens:
            if self.ll_matcher.is_stopped() and not self.terminated:
                self.rollback_lag = 1
            self.terminated = True

        if self.ll_matcher.is_stopped():
            return True

        # TODO - Add jump decoding support in the future:
        # self.ll_matcher.compute_ff_bytes() - this should always work
        # self.ll_matcher.compute_ff_tokens() - this only works for
        #   "canonical" tokenizers
        # For conversion between the two, see
        # https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md

        r = self.ll_matcher.consume_tokens(tokens)

        self.check_error()

        return r

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the parser in sequence.
        Will not advance the parser.

        Returns the prefix list of tokens that are accepted by the parser.
        """
        if len(tokens) == 0:
            return []
        if self.ll_matcher.is_stopped():
            return []

        num_tokens = self.ll_matcher.validate_tokens(tokens)

        self.check_error()

        return tokens[:num_tokens]

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self.ll_matcher.rollback(num_tokens - self.rollback_lag)
            self.terminated = False
            self.rollback_lag = 0
            self.check_error()

    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None:
        # this will automatically return [EOS] mask if the matcher is stopped
        # or otherwise in an error state
        llguidance_torch.fill_next_token_bitmask(self.ll_matcher, bitmask, idx)
        self.check_error()

    def is_terminated(self) -> bool:
        return self.terminated

    def reset(self):
        # This method may be not needed anymore? TODO
        self.ll_matcher.reset()


def _unsupported_structural_tag(reason: str, node: Any) -> VLLMValidationError:
    def _shape(n: Any) -> Any:
        if isinstance(n, dict):
            return {k: "..." if k == "json_schema" else _shape(v) for k, v in n.items()}
        return [_shape(x) for x in n] if isinstance(n, list) else n

    try:
        desc = json.dumps(_shape(node))[:1000]
    except (TypeError, ValueError):
        desc = repr(node)[:1000]
    return VLLMValidationError(
        f"Invalid grammar specification: structural tag {reason} is not "
        f"supported by the guidance backend (got: {desc})"
    )


def _tag_schema(
    schema: Any,
    node: Any,
    disable_any_whitespace: bool,
    disable_additional_properties: bool,
) -> dict[str, Any]:
    """Prepare a tag's JSON schema for embedding into a llguidance grammar
    (``StructTag.grammar`` / ``%json``). These expect an actual JSON schema —
    not the serialized ``{"grammars": [...]}`` envelope returned by
    ``LLMatcher.grammar_from_json_schema``, which llguidance would silently
    read as an empty schema, dropping every constraint."""
    if schema is True:
        schema = {}  # boolean schema "true": any JSON value
    if not isinstance(schema, dict):
        raise _unsupported_structural_tag("non-object json_schema content", node)
    if disable_additional_properties:
        schema = process_for_additional_properties(schema)
    else:
        schema = copy.deepcopy(schema)
    schema.setdefault("x-guidance", {"whitespace_flexible": not disable_any_whitespace})
    return schema


def _tag_parts(
    t: Any,
    disable_any_whitespace: bool,
    disable_additional_properties: bool,
) -> tuple[str, str, dict[str, Any]]:
    """Validate one new-format ``tag`` entry -> ``(begin, end, schema)``."""
    if not isinstance(t, dict) or t.get("type", "tag") != "tag":
        raise _unsupported_structural_tag("entry that is not a 'tag'", t)
    begin, end, content = t.get("begin"), t.get("end"), t.get("content")
    if isinstance(end, list) and len(end) == 1:
        end = end[0]
    if not (isinstance(begin, str) and begin and isinstance(end, str)):
        raise _unsupported_structural_tag(
            "tag with token-level or non-string begin/end", t
        )
    if (
        not isinstance(content, dict)
        or content.get("type") != "json_schema"
        or content.get("style", "json") != "json"
        or content.get("any_order")
    ):
        raise _unsupported_structural_tag(
            "tag content that is not plain json_schema", t
        )
    schema = _tag_schema(
        content.get("json_schema"),
        content,
        disable_any_whitespace,
        disable_additional_properties,
    )
    return begin, end, schema


def _gtext(s: str) -> str:
    """Lark string literal (empty string -> no element), as emitted by
    ``llguidance.StructTag.to_grammar``."""
    return json.dumps(s) if s else ""


def _split_leading_special_token(begin: str) -> tuple[str, str]:
    """Split a leading ``<...>`` special-token marker off ``begin``
    (``StructTag.to_grammar``'s ``assume_special`` heuristic)."""
    if begin.startswith("<"):
        gt = begin.find(">")
        if gt > 0 and "<" not in begin[1:gt]:
            return begin[: gt + 1], begin[gt + 1 :]
    return "", begin


def _split_trailing_special_token(end: str) -> tuple[str, str]:
    """Split a trailing ``<...>`` special-token marker off ``end``, e.g.
    ``"}\\n</tool_call>"`` -> ``("}\\n", "</tool_call>")``."""
    if end.endswith(">"):
        lt = end.rfind("<")
        if lt != -1:
            marker = end[lt:]
            if len(marker) > 2 and ">" not in marker[:-1] and "<" not in marker[1:]:
                return end[:lt], marker
    return end, ""


def _tag_suffix(rest: str, schema: dict[str, Any], end: str) -> str:
    """Lark elements for a tag after its opener: remaining begin literal,
    ``%json`` body, and the end. Models emit closers like ``</tool_call>``
    either spelled in raw bytes or as the special token — a bytes-only end
    rejects the special token at runtime (llguidance maps unknown-special
    bytes to 0xFF), surfacing as "grammar rejected tokens" request
    termination — so when the end carries a ``<...>`` suffix both spellings
    are accepted, e.g. ``("}\\n</tool_call>" | "}\\n" </tool_call>)``."""
    elems = [_gtext(rest)] if rest else []
    elems.append("%json " + json.dumps(schema))
    prefix, marker = _split_trailing_special_token(end)
    if marker:
        both = f"{_gtext(prefix)} {marker}" if prefix else marker
        elems.append(f"({_gtext(end)} | {both})")
    elif end:
        elems.append(_gtext(end))
    return " ".join(elems)


def _triggered_tags_lark(parts: list[tuple[str, str, str, dict[str, Any]]]) -> str:
    """Free text with trigger-dispatched tags, zero or more times — the same
    skeleton ``llguidance.StructTag.to_grammar`` generates, except for the
    two-spelling end handling in ``_tag_suffix``."""
    tag_options = " | ".join(f"tag_{i}" for i in range(len(parts)))
    lark = (
        "%llguidance {}\n"
        f"start: ({tag_options})* tag_end\n"
        "tag_end: TAG_TEXT\n"
        "TAG_TEXT: /(.|\\n)*/\n"
    )
    for i, (trig, begin, end, schema) in enumerate(parts):
        body = _tag_suffix(begin[len(trig) :], schema, end)
        lark += "\n"
        if trig.startswith("<") and trig.endswith(">"):
            # trigger assumed to be a special token, as in to_grammar
            lark += f"tag_{i}: TAG_TEXT {trig} {body}\n"
        else:
            lark += f"tag_{i}_trig[lazy]: TAG_TEXT {_gtext(trig)}\n"
            lark += f"tag_{i}: tag_{i}_trig {body}\n"
    return lark


def _tags_with_separator_lark(
    parts: list[tuple[str, str, dict[str, Any]]],
    separator: str,
    stop_after_first: bool,
) -> str:
    """One or more tags separated by ``separator`` with NO free text
    (``at_least_one=true``); exactly one tag with ``stop_after_first``."""
    bodies = []
    for begin, end, schema in parts:
        special, rest = _split_leading_special_token(begin)
        suffix = _tag_suffix(rest, schema, end)
        bodies.append(f"{special} {suffix}" if special else suffix)
    alts = " | ".join(f"tag_{i}" for i in range(len(bodies)))
    if stop_after_first:
        start = f"start: {alts}"
    elif separator:
        start = f"start: ({alts}) ({_gtext(separator)} ({alts}))*"
    else:
        start = f"start: ({alts})+"
    rules = "\n".join(f"tag_{i}: {body}" for i, body in enumerate(bodies))
    return f"%llguidance {{}}\n{start}\n{rules}\n"


def _serialize_structural_tag_new_format(
    s_tag: Any,
    disable_any_whitespace: bool,
    disable_additional_properties: bool,
) -> str:
    """Translate a new-format (xgrammar-style) structural tag —
    ``{"type": "structural_tag", "format": {...}}`` — to a llguidance
    grammar. Covers the shapes ``vllm/tool_parsers/structural_tag_registry``
    emits for tool calling: ``triggered_tags`` (``tool_choice="auto"`` with a
    ``strict: true`` tool), ``tags_with_separator``
    (``"required"``/named), and ``any_text``. Anything else raises
    ``VLLMValidationError`` so the server surfaces a 400."""
    fmt = s_tag.get("format") if isinstance(s_tag, dict) else None
    if not isinstance(fmt, dict):
        raise _unsupported_structural_tag("without a 'format' object", s_tag)
    ftype = fmt.get("type")

    if ftype == "any_text":
        if fmt.get("excludes"):
            raise _unsupported_structural_tag("'any_text' with excludes", fmt)
        return llguidance.grammar_from("regex", "(?s:.*)")

    if ftype == "triggered_tags":
        triggers, tags = fmt.get("triggers"), fmt.get("tags")
        if (
            fmt.get("at_least_one")
            or fmt.get("stop_after_first")
            or fmt.get("excludes")
            or not triggers
            or not all(isinstance(t, str) for t in triggers)
            or not tags
        ):
            raise _unsupported_structural_tag(
                "'triggered_tags' with unsupported options", fmt
            )
        parts = []
        for t in tags:
            begin, end, schema = _tag_parts(
                t, disable_any_whitespace, disable_additional_properties
            )
            trig = next((tr for tr in triggers if begin.startswith(tr)), None)
            if trig is None:
                raise _unsupported_structural_tag(
                    "tag whose begin matches no trigger", fmt
                )
            parts.append((trig, begin, end, schema))
        return _triggered_tags_lark(parts)

    if ftype == "tags_with_separator":
        separator, tags = fmt.get("separator"), fmt.get("tags")
        if not fmt.get("at_least_one") or not isinstance(separator, str) or not tags:
            raise _unsupported_structural_tag(
                "'tags_with_separator' with unsupported options", fmt
            )
        parts = [
            _tag_parts(t, disable_any_whitespace, disable_additional_properties)
            for t in tags
        ]
        return _tags_with_separator_lark(
            parts, separator, bool(fmt.get("stop_after_first"))
        )

    raise _unsupported_structural_tag(f"format type {ftype!r}", fmt)


def serialize_guidance_grammar(
    request_type: StructuredOutputOptions,
    grammar_spec: str | dict[str, Any],
    disable_any_whitespace: bool = False,
    disable_additional_properties: bool = False,
) -> str:
    def _process_schema(
        grammar_spec: str | dict[str, Any],
    ) -> str:
        if disable_additional_properties:
            grammar_spec = process_for_additional_properties(grammar_spec)
        return llguidance.LLMatcher.grammar_from_json_schema(
            grammar_spec,
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )

    if request_type == StructuredOutputOptions.JSON:
        return _process_schema(grammar_spec)
    elif request_type == StructuredOutputOptions.JSON_OBJECT:
        return llguidance.LLMatcher.grammar_from_json_schema(
            '{"type": "object"}',
            defaults={
                "whitespace_flexible": not disable_any_whitespace,
            },
        )
    else:
        if request_type == StructuredOutputOptions.REGEX:
            tp = "regex"
        elif request_type == StructuredOutputOptions.GRAMMAR:
            tp = "grammar"
        elif request_type == StructuredOutputOptions.CHOICE:
            tp = "choice"
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            if isinstance(grammar_spec, str):
                s_tag = json.loads(grammar_spec)
            else:
                s_tag = grammar_spec
            if not (isinstance(s_tag, dict) and "structures" in s_tag):
                # New (xgrammar-style) structural tag format, e.g. produced by
                # vllm.tool_parsers.structural_tag_registry for hermes tool
                # calling (strict:true + tool_choice="auto", "required", or a
                # named tool). Same detection as backend_xgrammar: the legacy
                # format is {"triggers": [...], "structures": [...]}.
                return _serialize_structural_tag_new_format(
                    s_tag, disable_any_whitespace, disable_additional_properties
                )
            triggers: list[str] = s_tag["triggers"]
            tags: list[llguidance.StructTag] = []
            for s in s_tag["structures"]:
                begin: str = s["begin"]
                trig = next((t for t in triggers if begin.startswith(t)), None)
                if trig is None:
                    raise VLLMValidationError(
                        f"Trigger {begin} not found in triggers {triggers}"
                    )
                tags.append(
                    llguidance.StructTag(
                        trigger=trig,
                        begin=s["begin"],
                        # StructTag.grammar expects a JSON schema (or Lark),
                        # not the serialized {"grammars": [...]} envelope
                        # _process_schema returns — llguidance silently
                        # treats that envelope as an empty schema, dropping
                        # every constraint on the tag's content.
                        grammar=_tag_schema(
                            s["schema"],
                            s,
                            disable_any_whitespace,
                            disable_additional_properties,
                        ),
                        end=s["end"],
                    )
                )
            if not tags:
                raise VLLMValidationError(
                    "No structural tags found in the grammar spec."
                )
            return llguidance.StructTag.to_grammar(tags)
        else:
            logger.error(
                "Validation should have already occurred. Please file an issue."
            )
            raise ValueError(
                f"grammar is not of valid supported types. ({request_type!s})"
            )
        return llguidance.grammar_from(tp, grammar_spec)


def validate_guidance_grammar(
    sampling_params: SamplingParams, tokenizer: llguidance.LLTokenizer | None = None
) -> None:
    # if structured output is not enabled, there is nothing to validate
    if sampling_params.structured_outputs is None:
        return
    tp, grm = get_structured_output_key(sampling_params.structured_outputs)
    try:
        guidance_grm = serialize_guidance_grammar(tp, grm)
    except (ValueError, KeyError, TypeError) as e:
        raise VLLMValidationError(f"Invalid grammar specification: {e}") from e
    err = llguidance.LLMatcher.validate_grammar(guidance_grm, tokenizer)
    if err:
        raise VLLMValidationError(f"Grammar error: {err}")
