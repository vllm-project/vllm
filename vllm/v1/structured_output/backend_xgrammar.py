# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import weakref
from collections.abc import Callable, Iterator
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

# xgrammar 0.2.1 and 0.2.3 serialize grammar expressions with these enum
# values. Fail closed if a future xgrammar release changes the serialized
# schema instead of silently missing token-bearing expressions.
_XGRAMMAR_SUPPORTED_GRAMMAR_SERIALIZATION_VERSIONS = frozenset({"v13", "v14"})
_XGRAMMAR_EXPR_TOKEN = 9
_XGRAMMAR_EXPR_EXCLUDE_TOKEN = 10
_XGRAMMAR_EXPR_TOKEN_TAG_DISPATCH = 11
_XGRAMMAR_TOKENIZER_INFO_REQUIRED_ERROR = (
    "Token string resolution requires tokenizer_info"
)
_XGRAMMAR_UNRESOLVED_TOKEN_ID = -1
_XGRAMMAR_MAX_TOKEN_ID = (1 << 31) - 1
_XGRAMMAR_TOKEN_IDS_BY_TEXT_ATTR = "_vllm_token_ids_by_text"
_XGRAMMAR_DECODED_VOCAB_SIZE_ATTR = "_vllm_decoded_vocab_size"
_XGRAMMAR_TOKENIZER_INFO_CACHE: weakref.WeakKeyDictionary[Any, Any] = (
    weakref.WeakKeyDictionary()
)


def _validate_token_id(token_id: int, *, vocab_size: int) -> None:
    if not 0 <= token_id < vocab_size:
        raise ValueError(
            f"Structured output token ID {token_id} is outside the tokenizer "
            f"vocabulary range [0, {vocab_size})."
        )


def _iter_xgrammar_grammar_exprs(
    grammar: xgr.Grammar,
) -> Iterator[tuple[int, list[int]]]:
    """Yield xgrammar grammar-expression records from public serialization."""
    serialized = _load_xgrammar_serialized_grammar(grammar)
    offsets = serialized["grammar_expr_data"]
    data = serialized["grammar_expr_indptr"]

    if not isinstance(offsets, list) or not isinstance(data, list):
        raise ValueError("Invalid xgrammar grammar serialization.")

    for offset in offsets:
        if not isinstance(offset, int) or offset < 0 or offset + 2 > len(data):
            raise ValueError("Invalid xgrammar grammar serialization.")
        expr_type = data[offset]
        expr_len = data[offset + 1]
        if not isinstance(expr_type, int) or not isinstance(expr_len, int):
            raise ValueError("Invalid xgrammar grammar serialization.")
        if expr_len < 0 or offset + 2 + expr_len > len(data):
            raise ValueError("Invalid xgrammar grammar serialization.")
        expr_data = data[offset + 2 : offset + 2 + expr_len]
        if not all(isinstance(value, int) for value in expr_data):
            raise ValueError("Invalid xgrammar grammar serialization.")
        yield expr_type, expr_data


def _load_xgrammar_serialized_grammar(grammar: xgr.Grammar) -> dict[str, Any]:
    """Load a supported xgrammar grammar serialization."""
    try:
        serialized = json.loads(grammar.serialize_json())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid xgrammar grammar serialization.") from exc

    if not isinstance(serialized, dict):
        raise ValueError("Invalid xgrammar grammar serialization.")
    if (
        serialized.get("__VERSION__")
        not in _XGRAMMAR_SUPPORTED_GRAMMAR_SERIALIZATION_VERSIONS
    ):
        raise ValueError("Unsupported xgrammar grammar serialization version.")
    return serialized


def _iter_xgrammar_token_ids(grammar: xgr.Grammar) -> Iterator[int]:
    for expr_type, expr_data in _iter_xgrammar_grammar_exprs(grammar):
        if expr_type in (_XGRAMMAR_EXPR_TOKEN, _XGRAMMAR_EXPR_EXCLUDE_TOKEN):
            yield from expr_data
            continue
        if expr_type != _XGRAMMAR_EXPR_TOKEN_TAG_DISPATCH:
            continue
        if not expr_data:
            raise ValueError("Invalid xgrammar grammar serialization.")

        trigger_count = expr_data[0]
        trigger_end = 1 + trigger_count * 2
        exclude_count_index = trigger_end + 1
        if trigger_count < 0 or exclude_count_index >= len(expr_data):
            raise ValueError("Invalid xgrammar grammar serialization.")

        for index in range(1, trigger_end, 2):
            yield expr_data[index]

        exclude_count = expr_data[exclude_count_index]
        if exclude_count < 0 or exclude_count_index + 1 + exclude_count != len(
            expr_data
        ):
            raise ValueError("Invalid xgrammar grammar serialization.")
        yield from expr_data[exclude_count_index + 1 :]


def _validate_xgrammar_grammar_token_ids(
    grammar: xgr.Grammar,
    *,
    vocab_size: int,
    ignored_token_ids: frozenset[int] = frozenset(),
) -> None:
    """Reject token edges that would exceed the tokenizer-sized native state."""
    for token_id in _iter_xgrammar_token_ids(grammar):
        if token_id in ignored_token_ids:
            continue
        _validate_token_id(token_id, vocab_size=vocab_size)


def _xgrammar_grammar_has_unresolved_token_ids(grammar: xgr.Grammar) -> bool:
    """Return whether xgrammar left a string-token sentinel in the grammar."""
    return any(
        token_id == _XGRAMMAR_UNRESOLVED_TOKEN_ID
        for token_id in _iter_xgrammar_token_ids(grammar)
    )


def _load_structural_tag_payload(structural_tag: str) -> dict[str, Any]:
    try:
        payload = json.loads(structural_tag)
    except Exception as e:
        raise ValueError("Invalid structural tag specification.") from e
    if not isinstance(payload, dict):
        raise ValueError("Invalid structural tag specification.")
    return payload


def _iter_xgrammar_integral_json_values(value: Any) -> Iterator[int]:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, bool):
            continue
        if isinstance(current, int):
            yield current
            continue
        if isinstance(current, float):
            if current.is_integer():
                yield int(current)
            continue
        if isinstance(current, dict):
            stack.extend(current.values())
            continue
        if isinstance(current, list):
            stack.extend(current)


def _clone_xgrammar_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        cloned: Any = {}
    elif isinstance(value, list):
        cloned = []
    else:
        return value

    stack: list[tuple[Any, Any]] = [(value, cloned)]
    while stack:
        source, target = stack.pop()
        if isinstance(source, dict):
            for key, child in source.items():
                if isinstance(child, dict):
                    child_clone: Any = {}
                    target[key] = child_clone
                    stack.append((child, child_clone))
                elif isinstance(child, list):
                    child_clone = []
                    target[key] = child_clone
                    stack.append((child, child_clone))
                else:
                    target[key] = child
            continue

        for child in source:
            if isinstance(child, dict):
                child_clone = {}
                target.append(child_clone)
                stack.append((child, child_clone))
            elif isinstance(child, list):
                child_clone = []
                target.append(child_clone)
                stack.append((child, child_clone))
            else:
                target.append(child)
    return cloned


def _build_xgrammar_token_metadata(
    tokenizer_info: xgr.TokenizerInfo,
) -> tuple[dict[bytes, int], int]:
    decoded_vocab = tokenizer_info.decoded_vocab
    token_ids_by_text: dict[bytes, int] = {}
    for token_id, token in enumerate(decoded_vocab):
        token_bytes = token.encode("utf-8") if isinstance(token, str) else token
        token_ids_by_text.setdefault(token_bytes, token_id)
    return token_ids_by_text, len(decoded_vocab)


def _get_cached_xgrammar_token_metadata(
    tokenizer_info: xgr.TokenizerInfo,
) -> tuple[dict[bytes, int], int]:
    token_ids_by_text = getattr(
        tokenizer_info,
        _XGRAMMAR_TOKEN_IDS_BY_TEXT_ATTR,
        None,
    )
    decoded_vocab_size = getattr(
        tokenizer_info,
        _XGRAMMAR_DECODED_VOCAB_SIZE_ATTR,
        None,
    )
    if isinstance(token_ids_by_text, dict) and isinstance(decoded_vocab_size, int):
        return token_ids_by_text, decoded_vocab_size

    token_ids_by_text, decoded_vocab_size = _build_xgrammar_token_metadata(
        tokenizer_info
    )
    with contextlib.suppress(AttributeError, TypeError):
        setattr(
            tokenizer_info,
            _XGRAMMAR_TOKEN_IDS_BY_TEXT_ATTR,
            token_ids_by_text,
        )
        setattr(
            tokenizer_info,
            _XGRAMMAR_DECODED_VOCAB_SIZE_ATTR,
            decoded_vocab_size,
        )
    return token_ids_by_text, decoded_vocab_size


@dataclass
class _XgrammarTokenStringPlaceholders:
    reserved_ids: set[int]
    token_ids_by_text: dict[bytes, int] = field(default_factory=dict)
    replacements: dict[str, int] = field(default_factory=dict)
    synthetic_ids: set[int] = field(default_factory=set)
    next_id: int = 0

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        additional_reserved_ids: frozenset[int] = frozenset(),
        tokenizer_info: xgr.TokenizerInfo | None = None,
        minimum_synthetic_id: int = 0,
    ) -> "_XgrammarTokenStringPlaceholders":
        reserved_ids = {
            value
            for value in _iter_xgrammar_integral_json_values(payload)
            if 0 <= value <= _XGRAMMAR_MAX_TOKEN_ID
        }
        reserved_ids.update(additional_reserved_ids)
        token_ids_by_text: dict[bytes, int] = {}
        next_id = minimum_synthetic_id
        if tokenizer_info is not None:
            token_ids_by_text, decoded_vocab_size = _get_cached_xgrammar_token_metadata(
                tokenizer_info
            )
            next_id = max(
                next_id,
                tokenizer_info.vocab_size,
                decoded_vocab_size,
            )
        return cls(
            reserved_ids=reserved_ids,
            token_ids_by_text=token_ids_by_text,
            next_id=next_id,
        )

    def replace(self, value: Any, *, allow_empty: bool = False) -> Any:
        if not isinstance(value, str):
            return value
        if not value and not allow_empty:
            return value
        if value in self.replacements:
            return self.replacements[value]

        try:
            token_bytes = value.encode("utf-8")
        except UnicodeEncodeError:
            token_bytes = None
        if token_bytes is not None and token_bytes in self.token_ids_by_text:
            token_id = self.token_ids_by_text[token_bytes]
            self.replacements[value] = token_id
            return token_id

        placeholder = self._allocate()
        self.replacements[value] = placeholder
        self.synthetic_ids.add(placeholder)
        return placeholder

    def _allocate(self) -> int:
        while self.next_id in self.reserved_ids:
            self.next_id += 1
        if self.next_id > _XGRAMMAR_MAX_TOKEN_ID:
            raise ValueError("Invalid structural tag specification.")

        placeholder = self.next_id
        self.reserved_ids.add(placeholder)
        self.next_id += 1
        return placeholder

    @property
    def ids(self) -> frozenset[int]:
        return frozenset(self.synthetic_ids)


def _replace_xgrammar_token_string_list(
    value: Any,
    *,
    placeholders: _XgrammarTokenStringPlaceholders,
) -> Any:
    if not isinstance(value, list):
        return value
    return [placeholders.replace(item) for item in value]


def _xgrammar_rules_use_token_dispatch(format_payload: dict[str, Any]) -> bool:
    # Typeless xgrammar rules are token-dispatch rules once any trigger is
    # numeric; all-string typeless rules stay ordinary string dispatch rules.
    format_type = format_payload.get("type")
    if format_type == "dispatch":
        return False
    if format_type == "token_dispatch":
        return True

    rules = format_payload.get("rules")
    if not isinstance(rules, list):
        return False
    return any(
        isinstance(rule, list)
        and len(rule) == 2
        and isinstance(rule[0], (int, float))
        and not isinstance(rule[0], bool)
        for rule in rules
    )


def _replace_xgrammar_format_token_strings(
    format_payload: Any,
    *,
    placeholders: _XgrammarTokenStringPlaceholders,
    allow_typeless_token: bool = False,
    syntax_only: bool = False,
) -> Any:
    """Replace only token-bearing strings while preserving native format parsing."""
    if not isinstance(format_payload, dict):
        return format_payload

    replaced = _clone_xgrammar_json_value(format_payload)
    stack: list[tuple[dict[str, Any], dict[str, Any], bool]] = [
        (format_payload, replaced, allow_typeless_token)
    ]
    visited: list[tuple[dict[str, Any], dict[str, Any]]] = []
    while stack:
        source, target, allow_token = stack.pop()
        visited.append((source, target))

        # Native tag boundaries are parsed as TokenFormat objects even when a
        # caller includes an unrelated type field on the boundary object.
        if target.get("type") == "token" or (allow_token and "token" in target):
            target["token"] = placeholders.replace(target.get("token"))

        for field_name in ("trigger_tokens", "exclude_tokens"):
            if field_name in target:
                target[field_name] = _replace_xgrammar_token_string_list(
                    target[field_name],
                    placeholders=placeholders,
                )

        children: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
        source_content = source.get("content")
        target_content = target.get("content")
        if isinstance(source_content, dict) and isinstance(target_content, dict):
            children.append((source_content, target_content, False))

        for field_name in ("begin", "end"):
            source_child = source.get(field_name)
            target_child = target.get(field_name)
            if isinstance(source_child, dict) and isinstance(target_child, dict):
                children.append((source_child, target_child, True))

        for field_name in ("elements", "tags"):
            source_items = source.get(field_name)
            target_items = target.get(field_name)
            if not isinstance(source_items, list) or not isinstance(
                target_items,
                list,
            ):
                continue
            for source_item, target_item in zip(source_items, target_items):
                if isinstance(source_item, dict) and isinstance(target_item, dict):
                    children.append((source_item, target_item, False))

        source_rules = source.get("rules")
        target_rules = target.get("rules")
        if isinstance(source_rules, list) and isinstance(target_rules, list):
            token_dispatch = _xgrammar_rules_use_token_dispatch(source)
            for source_rule, target_rule in zip(source_rules, target_rules):
                if (
                    not isinstance(source_rule, list)
                    or len(source_rule) != 2
                    or not isinstance(target_rule, list)
                    or len(target_rule) != 2
                ):
                    continue
                if token_dispatch:
                    target_rule[0] = placeholders.replace(
                        source_rule[0],
                        allow_empty=True,
                    )
                source_child = source_rule[1]
                target_child = target_rule[1]
                if isinstance(source_child, dict) and isinstance(target_child, dict):
                    children.append((source_child, target_child, False))

        stack.extend(reversed(children))

    if syntax_only:
        for source, target in reversed(visited):
            _normalize_xgrammar_token_triggered_tags_for_syntax(source, target)

    return replaced


def _xgrammar_syntax_token_key(value: Any) -> tuple[str, int | str] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return ("id", value) if value >= 0 else None
    if isinstance(value, float):
        return ("id", int(value)) if value.is_integer() and value >= 0 else None
    if isinstance(value, str):
        return ("text", value) if value else None
    return None


def _xgrammar_token_triggered_tag_syntax_assignments(
    trigger_keys: list[tuple[str, int | str]],
    begin_keys: list[tuple[str, int | str]],
) -> dict[tuple[str, int | str], int] | None:
    trigger_counts: dict[tuple[str, int | str], int] = {}
    for key in trigger_keys:
        trigger_counts[key] = trigger_counts.get(key, 0) + 1

    if any(count > 1 for count in trigger_counts.values()):
        return None

    numeric_ids = {
        int(value) for kind, value in (*trigger_keys, *begin_keys) if kind == "id"
    }
    next_id = max(numeric_ids, default=-1) + 1
    assignments: dict[tuple[str, int | str], int] = {
        key: int(key[1]) for key in (*trigger_keys, *begin_keys) if key[0] == "id"
    }

    for key in (*trigger_keys, *begin_keys):
        if key[0] != "text" or key in assignments:
            continue
        while next_id in numeric_ids:
            next_id += 1
        assignments[key] = next_id
        numeric_ids.add(next_id)
        next_id += 1

    trigger_key_set = set(trigger_counts)
    begin_key_set = set(begin_keys)
    numeric_trigger_keys = {key for key in trigger_key_set if key[0] == "id"}
    text_trigger_keys = {key for key in trigger_key_set if key[0] == "text"}
    numeric_begin_keys = {key for key in begin_key_set if key[0] == "id"}
    text_begin_keys = {key for key in begin_key_set if key[0] == "text"}

    begin_only_text_keys = text_begin_keys - text_trigger_keys
    uncovered_numeric_trigger_keys = numeric_trigger_keys - numeric_begin_keys
    if not (
        len(uncovered_numeric_trigger_keys)
        <= len(begin_only_text_keys)
        <= len(numeric_trigger_keys)
    ):
        return None
    ordered_numeric_trigger_keys = sorted(
        uncovered_numeric_trigger_keys,
        key=repr,
    ) + sorted(
        numeric_trigger_keys - uncovered_numeric_trigger_keys,
        key=repr,
    )
    for begin_key, trigger_key in zip(
        sorted(begin_only_text_keys, key=repr),
        ordered_numeric_trigger_keys,
    ):
        assignments[begin_key] = assignments[trigger_key]

    unmatched_numeric_begin_keys = numeric_begin_keys - numeric_trigger_keys
    uncovered_text_trigger_keys = text_trigger_keys - text_begin_keys
    if not (
        len(uncovered_text_trigger_keys)
        <= len(unmatched_numeric_begin_keys)
        <= len(text_trigger_keys)
    ):
        return None
    ordered_text_trigger_keys = sorted(
        uncovered_text_trigger_keys,
        key=repr,
    ) + sorted(
        text_trigger_keys - uncovered_text_trigger_keys,
        key=repr,
    )
    for trigger_key, begin_key in zip(
        ordered_text_trigger_keys,
        sorted(unmatched_numeric_begin_keys, key=repr),
    ):
        assignments[trigger_key] = assignments[begin_key]

    trigger_ids = [assignments[key] for key in trigger_keys]
    if len(set(trigger_ids)) != len(trigger_ids):
        return None
    if {assignments[key] for key in begin_keys} != set(trigger_ids):
        return None
    return assignments


def _normalize_xgrammar_token_triggered_tags_for_syntax(
    format_payload: Any,
    replaced: Any,
) -> None:
    if not isinstance(format_payload, dict) or not isinstance(replaced, dict):
        return
    if format_payload.get("type") != "token_triggered_tags":
        return

    triggers = format_payload.get("trigger_tokens")
    tags = format_payload.get("tags")
    replaced_tags = replaced.get("tags")
    if not isinstance(triggers, list) or not isinstance(tags, list):
        return
    if not isinstance(replaced_tags, list) or len(tags) != len(replaced_tags):
        return

    trigger_keys: list[tuple[str, int | str]] = []
    for trigger in triggers:
        key = _xgrammar_syntax_token_key(trigger)
        if key is None:
            return
        trigger_keys.append(key)

    begin_keys: list[tuple[str, int | str]] = []
    replaced_begins: list[dict[str, Any]] = []
    for tag, replaced_tag in zip(tags, replaced_tags):
        if not isinstance(tag, dict) or not isinstance(replaced_tag, dict):
            return
        begin = tag.get("begin")
        replaced_begin = replaced_tag.get("begin")
        if not isinstance(begin, dict) or not isinstance(replaced_begin, dict):
            return
        key = _xgrammar_syntax_token_key(begin.get("token"))
        if key is None:
            return
        begin_keys.append(key)
        replaced_begins.append(replaced_begin)

    if not any(key[0] == "text" for key in (*trigger_keys, *begin_keys)):
        return
    assignments = _xgrammar_token_triggered_tag_syntax_assignments(
        trigger_keys,
        begin_keys,
    )
    if assignments is None:
        return

    replaced["trigger_tokens"] = [assignments[key] for key in trigger_keys]
    for replaced_begin, begin_key in zip(replaced_begins, begin_keys):
        replaced_begin["token"] = assignments[begin_key]


def _replace_xgrammar_structural_tag_token_strings(
    payload: dict[str, Any],
    *,
    additional_reserved_ids: frozenset[int] = frozenset(),
    tokenizer_info: xgr.TokenizerInfo | None = None,
    minimum_synthetic_id: int = 0,
    syntax_only: bool = False,
) -> tuple[str, frozenset[int]]:
    placeholders = _XgrammarTokenStringPlaceholders.from_payload(
        payload,
        additional_reserved_ids=additional_reserved_ids,
        tokenizer_info=tokenizer_info,
        minimum_synthetic_id=minimum_synthetic_id,
    )
    replaced = dict(payload)
    if "format" in replaced:
        replaced["format"] = _replace_xgrammar_format_token_strings(
            replaced["format"],
            placeholders=placeholders,
            syntax_only=syntax_only,
        )
    return json.dumps(replaced), placeholders.ids


@dataclass(frozen=True)
class _XgrammarStructuralTagParseResult:
    grammar: xgr.Grammar
    needs_tokenizer_info: bool
    token_id_validation_grammars: tuple[tuple[xgr.Grammar, frozenset[int]], ...]
    safe_structural_tag: str | None = None


def _xgrammar_requires_tokenizer_info(exc: Exception) -> bool:
    return _XGRAMMAR_TOKENIZER_INFO_REQUIRED_ERROR in str(exc)


def _parse_xgrammar_structural_tag_payload(
    payload: dict[str, Any],
    structural_tag: str,
) -> xgr.Grammar:
    if "structures" in payload:
        tags = [
            xgr.StructuralTagItem(
                begin=structure["begin"],
                schema=json.dumps(structure["schema"]),
                end=structure["end"],
            )
            for structure in payload["structures"]
        ]
        return xgr.Grammar.from_structural_tag(tags, payload["triggers"])
    return xgr.Grammar.from_structural_tag(structural_tag)


def _xgrammar_has_repeat_token_dispatch_candidate(
    value: Any,
    *,
    under_repeat: bool = False,
) -> bool:
    stack = [(value, under_repeat)]
    while stack:
        current, current_under_repeat = stack.pop()
        if isinstance(current, dict):
            if current_under_repeat and _xgrammar_rules_use_token_dispatch(current):
                return True
            child_under_repeat = current_under_repeat or current.get("type") == "repeat"
            stack.extend((child, child_under_repeat) for child in current.values())
            continue
        if isinstance(current, list):
            stack.extend((child, current_under_repeat) for child in current)
    return False


def _replace_xgrammar_repeats_with_optional(value: Any) -> Any:
    replaced = _clone_xgrammar_json_value(value)
    stack = [replaced]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if current.get("type") == "repeat":
                current["type"] = "optional"
            stack.extend(current.values())
            continue
        if isinstance(current, list):
            stack.extend(current)
    return replaced


def _build_xgrammar_repeat_token_dispatch_probe(
    value: Any,
    *,
    marker_id: int,
    under_repeat: bool = False,
) -> Any:
    replaced = _clone_xgrammar_json_value(value)
    stack = [(replaced, under_repeat)]
    while stack:
        current, current_under_repeat = stack.pop()
        if isinstance(current, dict):
            child_under_repeat = current_under_repeat or current.get("type") == "repeat"
            if current.get("type") == "repeat":
                current["type"] = "optional"
            if current_under_repeat and _xgrammar_rules_use_token_dispatch(current):
                rules = current.get("rules")
                if (
                    isinstance(rules, list)
                    and rules
                    and isinstance(rules[0], list)
                    and len(rules[0]) == 2
                ):
                    rules[0] = [marker_id, rules[0][1]]
            stack.extend((child, child_under_repeat) for child in current.values())
            continue
        if isinstance(current, list):
            stack.extend((child, current_under_repeat) for child in current)
    return replaced


def _parse_xgrammar_structural_tag_probe(
    payload: dict[str, Any],
    *,
    additional_reserved_ids: frozenset[int] = frozenset(),
) -> xgr.Grammar:
    structural_tag, _ = _replace_xgrammar_structural_tag_token_strings(
        payload,
        additional_reserved_ids=additional_reserved_ids,
    )
    return _parse_xgrammar_structural_tag_payload(payload, structural_tag)


def _xgrammar_has_repeat_token_dispatch(payload: dict[str, Any]) -> bool:
    """Detect active token-dispatch descendants under explicit repeat nodes."""
    if not _xgrammar_has_repeat_token_dispatch_candidate(payload):
        return False

    safe_payload = _replace_xgrammar_repeats_with_optional(payload)
    baseline_grammar = _parse_xgrammar_structural_tag_probe(safe_payload)
    marker_id = _XgrammarTokenStringPlaceholders.from_payload(
        payload,
        additional_reserved_ids=frozenset(_iter_xgrammar_token_ids(baseline_grammar)),
    )._allocate()
    probe_payload = _build_xgrammar_repeat_token_dispatch_probe(
        payload,
        marker_id=marker_id,
    )
    grammar = _parse_xgrammar_structural_tag_probe(
        probe_payload,
        additional_reserved_ids=frozenset({marker_id}),
    )
    return marker_id in _iter_xgrammar_token_ids(grammar)


def _reject_xgrammar_repeat_token_dispatch(payload: dict[str, Any]) -> None:
    if _xgrammar_has_repeat_token_dispatch(payload):
        raise ValueError("Invalid structural tag specification.")


def _parse_xgrammar_structural_tag_placeholder_grammars(
    payload: dict[str, Any],
) -> tuple[tuple[xgr.Grammar, frozenset[int]], ...]:
    # A real nested grammar token can collide with one synthetic placeholder.
    # Disjoint native parses keep that token visible in at least one pass
    # without mirroring xgrammar's branch selection over shadowed JSON fields.
    first_structural_tag, first_placeholder_ids = (
        _replace_xgrammar_structural_tag_token_strings(payload)
    )
    second_structural_tag, second_placeholder_ids = (
        _replace_xgrammar_structural_tag_token_strings(
            payload,
            additional_reserved_ids=first_placeholder_ids,
        )
    )
    return (
        (
            _parse_xgrammar_structural_tag_payload(payload, first_structural_tag),
            first_placeholder_ids,
        ),
        (
            _parse_xgrammar_structural_tag_payload(payload, second_structural_tag),
            second_placeholder_ids,
        ),
    )


def _parse_xgrammar_structural_tag_resolved_grammar(
    payload: dict[str, Any],
    *,
    tokenizer_info: xgr.TokenizerInfo,
    minimum_synthetic_id: int,
) -> tuple[str, xgr.Grammar]:
    _reject_xgrammar_repeat_token_dispatch(payload)
    # Resolve known token strings to the same IDs xgrammar would use. Unknown
    # active strings become synthetic out-of-range IDs so parse-only validation
    # can reject them without sending raw attacker-controlled IDs to compilation.
    structural_tag, _ = _replace_xgrammar_structural_tag_token_strings(
        payload,
        tokenizer_info=tokenizer_info,
        minimum_synthetic_id=minimum_synthetic_id,
    )
    return structural_tag, _parse_xgrammar_structural_tag_payload(
        payload,
        structural_tag,
    )


def _parse_xgrammar_structural_tag_grammar(
    structural_tag: str,
    *,
    allow_token_strings: bool = False,
    tokenizer_info: xgr.TokenizerInfo | None = None,
    tokenizer_info_factory: Callable[[], xgr.TokenizerInfo] | None = None,
    minimum_synthetic_id: int = 0,
) -> _XgrammarStructuralTagParseResult:
    payload = _load_structural_tag_payload(structural_tag)
    try:
        _reject_xgrammar_repeat_token_dispatch(payload)
        grammar = _parse_xgrammar_structural_tag_payload(payload, structural_tag)
        if allow_token_strings and _xgrammar_grammar_has_unresolved_token_ids(grammar):
            try:
                if tokenizer_info is None and tokenizer_info_factory is not None:
                    tokenizer_info = tokenizer_info_factory()
                if tokenizer_info is not None:
                    (
                        safe_structural_tag,
                        grammar,
                    ) = _parse_xgrammar_structural_tag_resolved_grammar(
                        payload,
                        tokenizer_info=tokenizer_info,
                        minimum_synthetic_id=minimum_synthetic_id,
                    )
                    return _XgrammarStructuralTagParseResult(
                        grammar=grammar,
                        needs_tokenizer_info=False,
                        token_id_validation_grammars=((grammar, frozenset()),),
                        safe_structural_tag=safe_structural_tag,
                    )
            except Exception as replacement_exc:
                raise ValueError("Invalid structural tag specification.") from (
                    replacement_exc
                )
        return _XgrammarStructuralTagParseResult(
            grammar=grammar,
            needs_tokenizer_info=False,
            token_id_validation_grammars=((grammar, frozenset()),),
        )
    except Exception as exc:
        if allow_token_strings and _xgrammar_requires_tokenizer_info(exc):
            try:
                if tokenizer_info is None and tokenizer_info_factory is not None:
                    tokenizer_info = tokenizer_info_factory()
                if tokenizer_info is not None:
                    (
                        safe_structural_tag,
                        grammar,
                    ) = _parse_xgrammar_structural_tag_resolved_grammar(
                        payload,
                        tokenizer_info=tokenizer_info,
                        minimum_synthetic_id=minimum_synthetic_id,
                    )
                    return _XgrammarStructuralTagParseResult(
                        grammar=grammar,
                        needs_tokenizer_info=False,
                        token_id_validation_grammars=((grammar, frozenset()),),
                        safe_structural_tag=safe_structural_tag,
                    )
                placeholder_grammars = (
                    _parse_xgrammar_structural_tag_placeholder_grammars(payload)
                )
                return _XgrammarStructuralTagParseResult(
                    grammar=placeholder_grammars[0][0],
                    needs_tokenizer_info=True,
                    token_id_validation_grammars=placeholder_grammars,
                )
            except Exception as replacement_exc:
                raise ValueError("Invalid structural tag specification.") from (
                    replacement_exc
                )
        raise ValueError("Invalid structural tag specification.") from exc


def validate_xgrammar_structural_tag_syntax(structural_tag: str) -> None:
    """Validate structural-tag syntax without requiring tokenizer context."""
    payload = _load_structural_tag_payload(structural_tag)
    try:
        _reject_xgrammar_repeat_token_dispatch(payload)
        _parse_xgrammar_structural_tag_payload(payload, structural_tag)
    except Exception as exc:
        if not _xgrammar_requires_tokenizer_info(exc):
            raise ValueError("Invalid structural tag specification.") from exc
        try:
            syntax_structural_tag, _ = _replace_xgrammar_structural_tag_token_strings(
                payload,
                syntax_only=True,
            )
            _parse_xgrammar_structural_tag_payload(payload, syntax_structural_tag)
        except Exception as replacement_exc:
            raise ValueError("Invalid structural tag specification.") from (
                replacement_exc
            )


def _validate_xgrammar_structural_tag_token_ids(
    parse_result: _XgrammarStructuralTagParseResult,
    *,
    vocab_size: int,
) -> None:
    for grammar, ignored_token_ids in parse_result.token_id_validation_grammars:
        _validate_xgrammar_grammar_token_ids(
            grammar,
            vocab_size=vocab_size,
            ignored_token_ids=ignored_token_ids,
        )


def _new_xgrammar_compiler(tokenizer_info: xgr.TokenizerInfo) -> xgr.GrammarCompiler:
    return xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=8,
        cache_enabled=True,
        cache_limit_bytes=vllm.envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
    )


def _build_xgrammar_tokenizer_info(
    tokenizer: Any,
    *,
    vocab_size: int | None = None,
) -> xgr.TokenizerInfo:
    if is_mistral_tokenizer(tokenizer):
        stop_token_ids = [tokenizer.eos_token_id]
        effective_vocab_size = (
            len(tokenizer.vocab) if vocab_size is None else vocab_size
        )
        return xgr.TokenizerInfo(  # type: ignore
            encoded_vocab=tokenizer.vocab,
            vocab_type=xgr.VocabType.RAW
            if tokenizer.is_tekken
            else xgr.VocabType.BYTE_FALLBACK,
            vocab_size=effective_vocab_size,
            stop_token_ids=stop_token_ids,
            add_prefix_space=True,
        )
    return xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)


def _get_cached_xgrammar_tokenizer_info(tokenizer: Any) -> xgr.TokenizerInfo:
    try:
        tokenizer_info = _XGRAMMAR_TOKENIZER_INFO_CACHE.get(tokenizer)
    except TypeError:
        return _build_xgrammar_tokenizer_info(tokenizer)

    if tokenizer_info is None:
        tokenizer_info = _build_xgrammar_tokenizer_info(tokenizer)
        with contextlib.suppress(TypeError):
            _XGRAMMAR_TOKENIZER_INFO_CACHE[tokenizer] = tokenizer_info
    return tokenizer_info


@dataclass
class XgrammarBackend(StructuredOutputBackend):
    def __post_init__(self):
        self.disable_any_whitespace = (
            self.vllm_config.structured_outputs_config.disable_any_whitespace
        )

        if is_mistral_tokenizer(self.tokenizer):
            # NOTE: ideally, xgrammar should handle this accordingly.
            # refer to https://github.com/mlc-ai/xgrammar/blob/d77c0a0173ef14779c918e3be7966ba852f7910f/python/xgrammar/tokenizer_info.py#L98
            # not self.tokenizer.vocab_size as self.tokenizer.vocab
            # collapses all decoded errors into a single token.
            self.vocab_size = len(self.tokenizer.vocab)
        tokenizer_info = _build_xgrammar_tokenizer_info(
            self.tokenizer,
            vocab_size=self.vocab_size,
        )
        self._tokenizer_info = tokenizer_info
        self.compiler = _new_xgrammar_compiler(tokenizer_info)
        self._structural_tag_validation_compiler = None
        self._structural_tag_validation_tokenizer_info = None

        self.num_speculative_tokens = 0
        if self.vllm_config.speculative_config is not None:
            self.num_speculative_tokens = (
                self.vllm_config.speculative_config.num_speculative_tokens
            )

    def _get_structural_tag_validation_compiler(self) -> xgr.GrammarCompiler:
        if self._structural_tag_validation_compiler is not None:
            return self._structural_tag_validation_compiler

        tokenizer_info = self._get_structural_tag_validation_tokenizer_info()
        if tokenizer_info is self._tokenizer_info:
            self._structural_tag_validation_compiler = self.compiler
            return self.compiler

        self._structural_tag_validation_compiler = _new_xgrammar_compiler(
            tokenizer_info
        )
        return self._structural_tag_validation_compiler

    def _get_structural_tag_validation_tokenizer_info(self) -> xgr.TokenizerInfo:
        if self._structural_tag_validation_tokenizer_info is not None:
            return self._structural_tag_validation_tokenizer_info

        if is_mistral_tokenizer(self.tokenizer):
            self._structural_tag_validation_tokenizer_info = self._tokenizer_info
            return self._tokenizer_info

        tokenizer_info = _get_cached_xgrammar_tokenizer_info(self.tokenizer)
        if tokenizer_info.vocab_size <= self.vocab_size:
            self._structural_tag_validation_tokenizer_info = self._tokenizer_info
        else:
            self._structural_tag_validation_tokenizer_info = tokenizer_info
        return self._structural_tag_validation_tokenizer_info

    def _compile_structural_tag_grammar(
        self,
        grammar_spec: str,
        grammar: xgr.Grammar,
        *,
        needs_tokenizer_info: bool,
        safe_structural_tag: str | None,
    ) -> xgr.CompiledGrammar:
        if safe_structural_tag is not None:
            return self.compiler.compile_structural_tag(safe_structural_tag)
        if not needs_tokenizer_info:
            return self.compiler.compile_grammar(grammar)

        validation_compiler = self._get_structural_tag_validation_compiler()
        validation_ctx = validation_compiler.compile_structural_tag(grammar_spec)
        resolved_grammar = validation_ctx.grammar
        _validate_xgrammar_grammar_token_ids(
            resolved_grammar,
            vocab_size=self.vocab_size,
        )
        if validation_compiler is self.compiler:
            return validation_ctx
        return self.compiler.compile_grammar(resolved_grammar)

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
            grammar = xgr.Grammar.from_ebnf(grammar_spec)
            _validate_xgrammar_grammar_token_ids(grammar, vocab_size=self.vocab_size)
            ctx = self.compiler.compile_grammar(grammar)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = compile_regex_with_timeout(
                self.compiler.compile_regex,
                grammar_spec,
            )
        elif request_type == StructuredOutputOptions.STRUCTURAL_TAG:
            tokenizer_info = getattr(
                self,
                "_structural_tag_validation_tokenizer_info",
                None,
            )
            if tokenizer_info is None and hasattr(self, "tokenizer"):
                tokenizer_info = self._get_structural_tag_validation_tokenizer_info()
            parse_result = _parse_xgrammar_structural_tag_grammar(
                grammar_spec,
                allow_token_strings=True,
                tokenizer_info=tokenizer_info,
                minimum_synthetic_id=self.vocab_size,
            )
            _validate_xgrammar_structural_tag_token_ids(
                parse_result,
                vocab_size=self.vocab_size,
            )
            ctx = self._compile_structural_tag_grammar(
                grammar_spec,
                parse_result.grammar,
                needs_tokenizer_info=parse_result.needs_tokenizer_info,
                safe_structural_tag=parse_result.safe_structural_tag,
            )
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


def _validate_xgrammar_structural_tag_grammar(
    structural_tag: str,
    *,
    vocab_size: int | None,
    tokenizer_info: xgr.TokenizerInfo | None = None,
    tokenizer_info_factory: Callable[[], xgr.TokenizerInfo] | None = None,
) -> None:
    parse_result = _parse_xgrammar_structural_tag_grammar(
        structural_tag,
        allow_token_strings=True,
        tokenizer_info=tokenizer_info,
        tokenizer_info_factory=tokenizer_info_factory,
        minimum_synthetic_id=0 if vocab_size is None else vocab_size,
    )
    if vocab_size is not None:
        _validate_xgrammar_structural_tag_token_ids(
            parse_result,
            vocab_size=vocab_size,
        )


def _validate_xgrammar_grammar(
    sampling_params: SamplingParams,
    *,
    vocab_size: int | None = None,
    tokenizer: Any | None = None,
) -> None:
    """Validate that the request is supported by structured output."""
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
            grammar = xgr.Grammar.from_ebnf(so_params.grammar)
        except Exception as e:
            raise VLLMValidationError("Invalid grammar specification.") from e
        if vocab_size is not None:
            _validate_xgrammar_grammar_token_ids(grammar, vocab_size=vocab_size)
        return

    if so_params.structural_tag:
        tokenizer_info_factory = (
            None
            if tokenizer is None
            else lambda: _get_cached_xgrammar_tokenizer_info(tokenizer)
        )
        _validate_xgrammar_structural_tag_grammar(
            so_params.structural_tag,
            vocab_size=vocab_size,
            tokenizer_info_factory=tokenizer_info_factory,
        )


def validate_xgrammar_grammar(
    sampling_params: SamplingParams,
    *,
    vocab_size: int | None = None,
    tokenizer: Any | None = None,
) -> None:
    """Validate xgrammar inputs and expose failures as client errors."""
    try:
        _validate_xgrammar_grammar(
            sampling_params,
            vocab_size=vocab_size,
            tokenizer=tokenizer,
        )
    except VLLMValidationError:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise VLLMValidationError(str(exc)) from exc
