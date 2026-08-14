# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm/renderers/cohere.py``.

The tests focus on the pure-Python helpers that produce the render-config
dicts passed to ``cohere_melody.render_cmd3`` / ``render_cmd4``. We also
include a class-level instantiation + async-non-blocking test that
mirrors the analogous ``test_mistral.py`` pattern, exercising the
:class:`CohereRenderer` end-to-end with mocked ``model_config`` /
tokenizer / melody bindings.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.renderers import ChatParams
from vllm.renderers.cohere import (
    CohereRenderer,
    MelodyContentType,
    _build_render_config,
    _content_blocks,
    _conversation_to_melody_messages,
    _document_to_melody,
    _normalize_tool_call,
    _role_to_melody,
    _tool_to_melody,
)
from vllm.tokenizers.hf import HfTokenizer

# ======================================================================
# _role_to_melody
# ======================================================================


class TestRoleToMelody:
    def test_assistant_maps_to_chatbot(self):
        # melody's templates use the legacy Cohere ``chatbot`` role name.
        assert _role_to_melody("assistant") == "chatbot"

    def test_developer_aliases_to_system(self):
        # OpenAI's ``developer`` role is documented as high-priority
        # instructions; map it onto the ``system`` slot rather than
        # letting the templates drop it on the floor.
        assert _role_to_melody("developer") == "system"

    @pytest.mark.parametrize("role", ["user", "system", "tool", "chatbot"])
    def test_recognized_roles_passthrough(self, role):
        assert _role_to_melody(role) == role

    @pytest.mark.parametrize(
        "role,expected",
        [
            ("ASSISTANT", "chatbot"),
            ("Developer", "system"),
            ("User", "user"),
            ("SYSTEM", "system"),
        ],
    )
    def test_role_normalization_is_case_insensitive(self, role, expected):
        # cmd3 / cmd4 templates lowercase the role before matching, so
        # accept any casing the caller provides.
        assert _role_to_melody(role) == expected

    @pytest.mark.parametrize("role", ["function", "moderator", "", "anything"])
    def test_unknown_roles_raise(self, role):
        # Silently dropping unknown roles produces malformed prompts
        # (the templates' role chain has no else branch).
        with pytest.raises(ValueError, match="Unsupported message role"):
            _role_to_melody(role)

    def test_non_string_role_rejected(self):
        # The function is typed ``role: str`` and the implementation
        # relies on Python's attribute lookup (``role.lower()``) to
        # reject non-strings — any exception type is acceptable as long
        # as we don't silently produce a malformed prompt.
        with pytest.raises((AttributeError, TypeError, ValueError)):
            _role_to_melody(None)  # type: ignore[arg-type]


# ======================================================================
# _normalize_tool_call
# ======================================================================


class TestNormalizeToolCall:
    def test_openai_dict_with_dict_arguments_json_encoded(self):
        # melody expects ``parameters`` as a JSON-encoded string even when
        # OpenAI delivers an already-parsed dict.
        out = _normalize_tool_call(
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": {"a": 1}},
            }
        )
        assert out == {"id": "c1", "name": "f", "parameters": '{"a": 1}'}

    def test_openai_dict_with_string_arguments_preserved(self):
        out = _normalize_tool_call(
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": '{"a":1}'},
            }
        )
        assert out["parameters"] == '{"a":1}'

    def test_flat_dict_without_function_wrapper(self):
        out = _normalize_tool_call({"id": "c1", "name": "f", "arguments": '{"k": 1}'})
        # Falls back to top-level ``name`` / ``arguments``.
        assert out == {"id": "c1", "name": "f", "parameters": '{"k": 1}'}

    def test_missing_id_becomes_empty_string(self):
        out = _normalize_tool_call({"function": {"name": "f", "arguments": "{}"}})
        assert out["id"] == ""

    def test_pydantic_model_dump_supported(self):
        class _Fake:
            def model_dump(self):
                return {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }

        out = _normalize_tool_call(_Fake())
        assert out == {"id": "c1", "name": "f", "parameters": "{}"}

    def test_invalid_type_rejected(self):
        with pytest.raises(TypeError, match="Unexpected tool_call value"):
            _normalize_tool_call(42)  # type: ignore[arg-type]


# ======================================================================
# _content_blocks
# ======================================================================


class TestContentBlocks:
    def test_none_returns_empty_list(self):
        assert _content_blocks(None) == []

    def test_string_wrapped_in_text_block(self):
        out = _content_blocks("hi")
        assert out == [{"type": MelodyContentType.TEXT, "text": "hi"}]

    def test_string_item_in_list_wrapped(self):
        out = _content_blocks(["a", "b"])
        assert out == [
            {"type": MelodyContentType.TEXT, "text": "a"},
            {"type": MelodyContentType.TEXT, "text": "b"},
        ]

    @pytest.mark.parametrize(
        "part_type",
        ["text", "input_text", "output_text", "refusal"],
    )
    def test_text_variants_normalized(self, part_type):
        out = _content_blocks([{"type": part_type, "text": "hello"}])
        assert out == [{"type": MelodyContentType.TEXT, "text": "hello"}]

    def test_thinking_block(self):
        out = _content_blocks([{"type": "thinking", "thinking": "thoughts"}])
        assert out == [{"type": MelodyContentType.THINKING, "thinking": "thoughts"}]

    def test_image_block_with_default_placeholder(self):
        out = _content_blocks([{"type": "image"}])
        assert out == [
            {
                "type": MelodyContentType.IMAGE,
                "image": {"template_placeholder": "<image>"},
            }
        ]

    def test_image_block_custom_placeholder(self):
        out = _content_blocks([{"type": "image", "template_placeholder": "[[IMG]]"}])
        assert out[0]["image"]["template_placeholder"] == "[[IMG]]"

    def test_document_block_dict_passthrough(self):
        out = _content_blocks(
            [{"type": "document", "document": {"data": {"text": "doc"}}}]
        )
        assert out == [
            {
                "type": MelodyContentType.DOCUMENT,
                "document": {"data": {"text": "doc"}},
            }
        ]

    def test_document_block_with_non_dict_falls_back_to_json_text(self):
        out = _content_blocks([{"type": "document", "document": "raw string doc"}])
        assert out[0]["type"] == MelodyContentType.TEXT
        # JSON-encoded for safety since melody expects a structured doc.
        assert out[0]["text"] == json.dumps("raw string doc")

    def test_tool_reference_emitted_as_text(self):
        out = _content_blocks([{"type": "tool_reference", "name": "calc"}])
        assert out == [{"type": MelodyContentType.TEXT, "text": "calc"}]

    def test_unknown_block_type_fallback_to_text(self):
        # Unknown block type with a string value is wrapped in a text block.
        out = _content_blocks([{"type": "custom", "custom": "value"}])
        assert out == [{"type": MelodyContentType.TEXT, "text": "value"}]

    def test_unknown_block_type_dict_value_json_encoded(self):
        out = _content_blocks([{"type": "custom", "custom": {"k": 1}}])
        assert out == [{"type": MelodyContentType.TEXT, "text": json.dumps({"k": 1})}]

    def test_non_string_non_dict_part_rejected(self):
        with pytest.raises(TypeError, match="Unexpected content part"):
            _content_blocks([42])  # type: ignore[list-item]


# ======================================================================
# _document_to_melody
# ======================================================================


class TestDocumentToMelody:
    def test_string_wrapped_in_text_dict(self):
        assert _document_to_melody("hello") == {"text": "hello"}

    def test_pure_dict_passthrough(self):
        inp = {"text": "x", "id": "d1"}
        out = _document_to_melody(inp)
        assert out == {"text": "x", "id": "d1"}
        # Must be a defensive copy so caller-side mutations of the
        # returned dict don't leak back into the input.
        out["new_key"] = "value"
        assert "new_key" not in inp

    def test_data_wrapper_flattened(self):
        # Cohere v2 documents use ``{id, data: {...}}``; melody expects
        # the flat shape with ``id`` merged into the payload.
        out = _document_to_melody({"id": "d1", "data": {"text": "hello", "title": "t"}})
        assert out == {"id": "d1", "text": "hello", "title": "t"}

    def test_data_wrapper_preserves_inner_id(self):
        # If the inner ``data`` already has an ``id``, it wins.
        out = _document_to_melody({"id": "outer", "data": {"id": "inner", "text": "x"}})
        assert out["id"] == "inner"

    def test_invalid_type_rejected(self):
        with pytest.raises(TypeError, match="Unsupported document type"):
            _document_to_melody(42)  # type: ignore[arg-type]


# ======================================================================
# _tool_to_melody
# ======================================================================


class TestToolToMelody:
    def test_openai_wrapper(self):
        out = _tool_to_melody(
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "calculate",
                    "parameters": {"type": "object"},
                },
            }
        )
        assert out == {
            "name": "calc",
            "description": "calculate",
            "parameters": {"type": "object"},
        }

    def test_flat_dict(self):
        out = _tool_to_melody({"name": "calc", "description": "d", "parameters": {}})
        assert out["name"] == "calc"
        assert out["parameters"] == {}

    def test_pydantic_like_model_dump(self):
        class _Fake:
            def model_dump(self):
                return {
                    "type": "function",
                    "function": {
                        "name": "calc",
                        "description": "x",
                        "parameters": {},
                    },
                }

        out = _tool_to_melody(_Fake())
        assert out["name"] == "calc"

    def test_missing_description_becomes_empty(self):
        out = _tool_to_melody({"name": "calc"})
        assert out["description"] == ""
        assert out["parameters"] == {}

    def test_invalid_type_rejected(self):
        with pytest.raises(TypeError, match="Unsupported tool type"):
            _tool_to_melody(42)  # type: ignore[arg-type]


# ======================================================================
# _conversation_to_melody_messages
# ======================================================================


class TestConversationToMelody:
    def test_basic_user_assistant_pair(self):
        conv = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = _conversation_to_melody_messages(conv)  # type: ignore[arg-type]
        assert out == [
            {
                "role": "user",
                "content": [{"type": MelodyContentType.TEXT, "text": "hi"}],
                "tool_calls": [],
            },
            {
                "role": "chatbot",
                "content": [{"type": MelodyContentType.TEXT, "text": "hello"}],
                "tool_calls": [],
            },
        ]

    def test_assistant_reasoning_prepended_as_thinking_block(self):
        # ``reasoning`` (or ``reasoning_content``) is prepended as a
        # ``thinking`` block on assistant turns, preserving multi-turn
        # chain-of-thought across the rendered prompt.
        conv = [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning": "thoughts",
            }
        ]
        out = _conversation_to_melody_messages(conv)  # type: ignore[arg-type]
        assert out[0]["content"] == [
            {"type": MelodyContentType.THINKING, "thinking": "thoughts"},
            {"type": MelodyContentType.TEXT, "text": "answer"},
        ]

    def test_assistant_reasoning_content_alias_accepted(self):
        conv = [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "thoughts",
            }
        ]
        out = _conversation_to_melody_messages(conv)  # type: ignore[arg-type]
        assert out[0]["content"][0] == {
            "type": MelodyContentType.THINKING,
            "thinking": "thoughts",
        }

    def test_user_reasoning_ignored(self):
        # Only assistant turns get reasoning-as-thinking lifting; user
        # turns with a ``reasoning`` key (which shouldn't happen in
        # practice) must not produce a phantom thinking block.
        conv = [
            {
                "role": "user",
                "content": "hi",
                "reasoning": "should be ignored",
            }
        ]
        out = _conversation_to_melody_messages(conv)  # type: ignore[arg-type]
        assert out[0]["content"] == [{"type": MelodyContentType.TEXT, "text": "hi"}]

    def test_tool_calls_normalized(self):
        conv = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a":1}'},
                    }
                ],
            }
        ]
        out = _conversation_to_melody_messages(conv)  # type: ignore[arg-type]
        assert out[0]["tool_calls"] == [
            {"id": "c1", "name": "f", "parameters": '{"a":1}'}
        ]

    def test_tool_call_id_preserved_on_tool_role(self):
        conv = [
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "c1",
            }
        ]
        out = _conversation_to_melody_messages(conv)  # type: ignore[arg-type]
        assert out[0]["tool_call_id"] == "c1"

    def test_messages_citations_attached_by_index(self):
        # ``messages_citations`` is a dict keyed by message index. Only
        # the message at the matching index should receive the
        # citations; other messages must be unaffected.
        conv = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        citations = {
            1: [
                {
                    "start_index": 0,
                    "end_index": 1,
                    "text": "a",
                    "sources": [
                        {
                            "tool_call_index": 0,
                            "tool_result_indices": [0],
                        }
                    ],
                    "is_thinking": False,
                }
            ]
        }
        out = _conversation_to_melody_messages(conv, citations)  # type: ignore[arg-type]
        assert "citations" not in out[0]
        assert out[1]["citations"] == citations[1]

    def test_messages_citations_none_is_a_no_op(self):
        conv = [{"role": "assistant", "content": "a"}]
        out = _conversation_to_melody_messages(conv, None)  # type: ignore[arg-type]
        assert "citations" not in out[0]

    def test_messages_citations_missing_index_is_a_no_op(self):
        # A ``messages_citations`` dict whose key doesn't hit any
        # message must not attach anything (and must not raise).
        conv = [{"role": "assistant", "content": "a"}]
        out = _conversation_to_melody_messages(conv, {5: [{"anything": 1}]})  # type: ignore[arg-type]
        assert "citations" not in out[0]


# ======================================================================
# _build_render_config
# ======================================================================


class TestBuildRenderConfig:
    def _conv(self):
        return [{"role": "user", "content": "hi"}]

    def test_default_format_is_cmd4(self):
        # Bare kwargs -> cmd4 (the current Command A+ prompt format).
        # Mirrors ``_DEFAULT_FORMAT`` in ``vllm/renderers/cohere.py`` and
        # the ``--cohere-format`` CLI default.
        fmt, cfg = _build_render_config(self._conv(), {})  # type: ignore[arg-type]
        assert fmt == "cmd4"
        assert cfg["use_jinja"] is True
        assert isinstance(cfg["messages"], list)
        # No additional_template_fields when no extra kwargs are set.
        assert "additional_template_fields" not in cfg

    def test_explicit_cmd3(self):
        fmt, cfg = _build_render_config(self._conv(), {"cohere_format": "cmd3"})  # type: ignore[arg-type]
        assert fmt == "cmd3"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid cohere_format"):
            _build_render_config(self._conv(), {"cohere_format": "cmd5"})  # type: ignore[arg-type]

    def test_documents_converted(self):
        _, cfg = _build_render_config(
            self._conv(),
            {
                "documents": [
                    "doc text",
                    {"id": "d1", "data": {"text": "wrapped"}},
                ]
            },
        )  # type: ignore[arg-type]
        assert cfg["documents"] == [
            {"text": "doc text"},
            {"id": "d1", "text": "wrapped"},
        ]

    def test_available_tools_take_precedence_over_tools(self):
        _, cfg = _build_render_config(
            self._conv(),
            {
                "tools": [{"type": "function", "function": {"name": "from_tools"}}],
                "available_tools": [
                    {"type": "function", "function": {"name": "preferred"}}
                ],
            },
        )  # type: ignore[arg-type]
        names = [t["name"] for t in cfg["available_tools"]]
        assert names == ["preferred"]

    def test_tools_used_when_no_available_tools(self):
        _, cfg = _build_render_config(
            self._conv(),
            {"tools": [{"type": "function", "function": {"name": "from_tools"}}]},
        )  # type: ignore[arg-type]
        assert [t["name"] for t in cfg["available_tools"]] == ["from_tools"]

    @pytest.mark.parametrize("value", ["enabled", "disabled"])
    def test_reasoning_type_direct(self, value):
        _, cfg = _build_render_config(self._conv(), {"reasoning_type": value})  # type: ignore[arg-type]
        assert cfg["reasoning_type"] == value

    def test_thinking_dict_shorthand_resolves_reasoning_type(self):
        _, cfg = _build_render_config(self._conv(), {"thinking": {"type": "enabled"}})  # type: ignore[arg-type]
        assert cfg["reasoning_type"] == "enabled"

    def test_thinking_shorthand_ignores_unknown_type(self):
        _, cfg = _build_render_config(self._conv(), {"thinking": {"type": "auto"}})  # type: ignore[arg-type]
        assert "reasoning_type" not in cfg

    def test_dev_instruction_forwarded(self):
        _, cfg = _build_render_config(self._conv(), {"dev_instruction": "be brief"})  # type: ignore[arg-type]
        assert cfg["dev_instruction"] == "be brief"

    def test_response_format_json_object_sets_json_mode(self):
        _, cfg = _build_render_config(
            self._conv(), {"response_format": {"type": "json_object"}}
        )  # type: ignore[arg-type]
        assert cfg["json_mode"] is True
        assert "json_schema" not in cfg

    def test_response_format_json_schema_sets_json_schema(self):
        schema = {"type": "object"}
        _, cfg = _build_render_config(
            self._conv(),
            {"response_format": {"type": "json_schema", "schema": schema}},
        )  # type: ignore[arg-type]
        # JSON-encoded for melody (string-only schema field).
        assert cfg["json_schema"] == json.dumps(schema)

    def test_response_format_nested_json_schema_unwrapped(self):
        # When the SDK shape is ``{type: json_schema, schema: {schema:
        # {...}}}``, the inner ``schema`` value is used.
        inner = {"type": "object"}
        _, cfg = _build_render_config(
            self._conv(),
            {
                "response_format": {
                    "type": "json_schema",
                    "schema": {"schema": inner},
                }
            },
        )  # type: ignore[arg-type]
        assert cfg["json_schema"] == json.dumps(inner)

    def test_json_schema_kwarg_direct(self):
        # Caller can also pass ``json_schema`` directly, both as dict and
        # as a pre-stringified value.
        _, cfg = _build_render_config(self._conv(), {"json_schema": {"a": 1}})  # type: ignore[arg-type]
        assert cfg["json_schema"] == '{"a": 1}'
        _, cfg = _build_render_config(
            self._conv(), {"json_schema": "raw-string-schema"}
        )  # type: ignore[arg-type]
        assert cfg["json_schema"] == "raw-string-schema"

    def test_json_mode_kwarg_overrides(self):
        _, cfg = _build_render_config(self._conv(), {"json_mode": True})  # type: ignore[arg-type]
        assert cfg["json_mode"] is True

    def test_cmd3_safety_mode_lowercased(self):
        _, cfg = _build_render_config(
            self._conv(),
            {"cohere_format": "cmd3", "safety_mode": "CONTEXTUAL"},
        )  # type: ignore[arg-type]
        assert cfg["safety_mode"] == "contextual"

    def test_cmd3_citation_quality_direct(self):
        _, cfg = _build_render_config(
            self._conv(),
            {"cohere_format": "cmd3", "citation_quality": "ACCURATE"},
        )  # type: ignore[arg-type]
        assert cfg["citation_quality"] == "accurate"

    def test_cmd3_citation_quality_derived_from_citation_options(self):
        # When ``citation_quality`` is unset, ``citation_options.mode`` is
        # collapsed to on/off so cmd3's binary toggle has a value.
        _, cfg = _build_render_config(
            self._conv(),
            {"cohere_format": "cmd3", "citation_options": {"mode": "accurate"}},
        )  # type: ignore[arg-type]
        assert cfg["citation_quality"] == "on"

        _, cfg = _build_render_config(
            self._conv(),
            {"cohere_format": "cmd3", "citation_options": {"mode": "off"}},
        )  # type: ignore[arg-type]
        assert cfg["citation_quality"] == "off"

    def test_cmd3_skip_preamble_forwarded(self):
        _, cfg = _build_render_config(
            self._conv(),
            {"cohere_format": "cmd3", "skip_preamble": True},
        )  # type: ignore[arg-type]
        assert cfg["skip_preamble"] is True

    def test_cmd3_no_grounding_field(self):
        # cmd3 should never emit a cmd4-only ``grounding`` field.
        _, cfg = _build_render_config(
            self._conv(),
            {"cohere_format": "cmd3", "grounding": "fast"},
        )  # type: ignore[arg-type]
        assert "grounding" not in cfg

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("FAST", "enabled"),
            ("ACCURATE", "enabled"),
            ("OFF", "disabled"),
            ("enabled", "enabled"),
            ("disabled", "disabled"),
            ("unknown", "unknown"),
        ],
    )
    def test_cmd4_grounding_direct(self, raw, expected):
        # melody's cmd4 only accepts ``unknown``/``enabled``/``disabled``,
        # so the renderer normalizes any of the v2-facing values into
        # that vocab.
        _, cfg = _build_render_config(
            self._conv(),
            {"cohere_format": "cmd4", "grounding": raw},
        )  # type: ignore[arg-type]
        assert cfg["grounding"] == expected

    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("ACCURATE", "enabled"),
            ("FAST", "enabled"),
            ("OFF", "disabled"),
        ],
    )
    def test_cmd4_grounding_from_citation_options_mode(self, mode, expected):
        _, cfg = _build_render_config(
            self._conv(),
            {
                "cohere_format": "cmd4",
                "citation_options": {"mode": mode},
            },
        )  # type: ignore[arg-type]
        assert cfg["grounding"] == expected

    def test_cmd4_grounding_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="Unrecognized cmd4 grounding"):
            _build_render_config(
                self._conv(),
                {"cohere_format": "cmd4", "grounding": "foobar"},
            )  # type: ignore[arg-type]

    def test_cmd4_platform_instruction(self):
        _, cfg = _build_render_config(
            self._conv(),
            {
                "cohere_format": "cmd4",
                "platform_instruction": "do this",
            },
        )  # type: ignore[arg-type]
        assert cfg["platform_instruction"] == "do this"

    def test_cmd4_no_safety_mode_field(self):
        # cmd4 should never carry cmd3-only ``safety_mode``/``citation_quality``.
        _, cfg = _build_render_config(
            self._conv(),
            {
                "cohere_format": "cmd4",
                "safety_mode": "contextual",
                "citation_quality": "on",
            },
        )  # type: ignore[arg-type]
        assert "safety_mode" not in cfg
        assert "citation_quality" not in cfg

    def test_extra_kwargs_become_additional_template_fields(self):
        # Anything not in the renderer's consumed-keys set is forwarded
        # verbatim under ``additional_template_fields`` so jinja templates
        # can resolve ``{{ var }}`` directly.
        _, cfg = _build_render_config(
            self._conv(),
            {
                "reasoning_effort": "low",
                "my_var": "x",
                "documents": ["doc"],  # consumed, must NOT leak through
            },
        )  # type: ignore[arg-type]
        extras = cfg["additional_template_fields"]
        assert extras == {"reasoning_effort": "low", "my_var": "x"}
        # Sanity: the consumed key still produced its dedicated config slot.
        assert cfg["documents"] == [{"text": "doc"}]

    def test_template_id_passthrough(self):
        # ``template_id`` is a safe selector for one of melody's built-in
        # template variants (not raw source) and is still accepted via
        # ``chat_template_kwargs``.
        _, cfg = _build_render_config(
            self._conv(),
            {"template_id": "tpl1"},
        )  # type: ignore[arg-type]
        assert cfg["template_id"] == "tpl1"
        # use_jinja is always True, regardless of caller input.
        assert cfg["use_jinja"] is True

    @pytest.mark.parametrize("cohere_only_key", ["template_jinja", "template"])
    def test_cohere_only_template_kwargs_are_rejected(self, cohere_only_key):
        # ``chat_template_kwargs.template_jinja`` / ``.template`` are
        # accepted at Cohere's own API surface but not at vLLM's -- raw
        # template source in vLLM must flow through the standard
        # ``chat_template`` request field so the
        # ``--trust-request-chat-template`` guard applies uniformly.
        # Silently dropping these keys would hide client misconfiguration,
        # so we reject them loudly instead.
        with pytest.raises(
            ValueError,
            match=f"chat_template_kwargs.{cohere_only_key!r}",
        ):
            _build_render_config(
                self._conv(),
                {cohere_only_key: "raw {{ jinja }}"},
            )  # type: ignore[arg-type]

    @pytest.mark.parametrize("cohere_only_key", ["template_jinja", "template"])
    def test_cohere_only_template_kwargs_none_is_tolerated(self, cohere_only_key):
        # An explicit ``None`` (e.g. from ``.model_dump(exclude_none=False)``
        # on an optional field) is treated as absent -- we neither raise
        # nor let it fall through as a Jinja variable.
        _, cfg = _build_render_config(
            self._conv(),
            {cohere_only_key: None},
        )  # type: ignore[arg-type]
        assert cohere_only_key not in cfg
        assert "additional_template_fields" not in cfg

    def test_chat_template_arg_populates_template_jinja(self):
        # The standard vLLM ``chat_template`` request field is the sole
        # supported channel for raw template source and is forwarded to
        # melody under its ``template_jinja`` config key.
        _, cfg = _build_render_config(
            self._conv(),
            {},
            "raw {{ jinja }}",
        )  # type: ignore[arg-type]
        assert cfg["template_jinja"] == "raw {{ jinja }}"
        assert cfg["use_jinja"] is True

    def test_chat_template_arg_none_leaves_template_jinja_unset(self):
        _, cfg = _build_render_config(
            self._conv(),
            {},
            None,
        )  # type: ignore[arg-type]
        assert "template_jinja" not in cfg


# ======================================================================
# End-to-end async rendering (mirrors ``test_mistral.py``)
# ======================================================================
#
# Verifies that the synchronous melody bindings run on the renderer's
# thread pool so the asyncio event loop stays responsive under
# concurrent load. Mirrors
# ``test_async_mistral_tokenizer_does_not_block_event_loop`` so future
# regressions in either path are caught uniformly.


@dataclass
class _MockHFConfig:
    model_type: str = "any"


@dataclass
class _MockModelConfig:
    runner_type = "generate"
    model: str = "cohere-test"
    tokenizer: str = "cohere-test"
    trust_remote_code: bool = False
    max_model_len: int = 100
    tokenizer_revision = None
    tokenizer_mode = "cohere"
    hf_config = _MockHFConfig()
    hf_text_config = _MockHFConfig()
    encoder_config: dict[str, Any] | None = None
    enable_prompt_embeds: bool = True
    skip_tokenizer_init: bool = True
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1


@dataclass
class _MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class _MockVllmConfig:
    model_config: _MockModelConfig
    parallel_config: _MockParallelConfig


@pytest.mark.asyncio
async def test_async_cohere_renderer_does_not_block_event_loop():
    expected_prompt = "MOCK_RENDERED_PROMPT"

    def slow_render(*_a, **_kw):
        time.sleep(2)
        return expected_prompt

    mock_tokenizer = Mock(spec=HfTokenizer)
    renderer = CohereRenderer(
        _MockVllmConfig(_MockModelConfig(), _MockParallelConfig()),
        tokenizer=mock_tokenizer,
    )

    # Replace the (already-imported) ``cohere_melody`` bindings with a
    # blocking mock. ``_render`` reads ``self._melody`` at call time, so
    # this works even though ``_render_async`` was bound at __init__.
    fake_melody = Mock()
    fake_melody.render_cmd3 = slow_render
    fake_melody.render_cmd4 = slow_render
    renderer._melody = fake_melody

    task = renderer.render_messages_async([], ChatParams())

    # Ensure the event loop is not blocked while the (blocking) render
    # call is in flight on the thread pool.
    blocked_count = 0
    for _ in range(20):  # ~2 seconds at 0.1s slices
        start = time.perf_counter()
        await asyncio.sleep(0)
        elapsed = time.perf_counter() - start
        if elapsed >= 0.5:
            blocked_count += 1
        await asyncio.sleep(0.1)

    _, prompt = await task
    assert prompt["prompt"] == expected_prompt, "Mocked blocking render was not called"
    assert blocked_count == 0, "Event loop blocked during rendering"


# ======================================================================
# End-to-end: request-level citations -> rendered prompt markup
# ======================================================================
#
# Ties the whole pipeline together: a Cohere v2 request carrying an
# assistant message with citations must produce a rendered prompt that
# contains melody's inline ``<co>...</co: <id>>`` markup around the
# cited span. Preserving this invariant is the whole point of the
# ``_messages_citations`` chat_template_kwargs entry.


class TestRequestCitationsReachRenderedPrompt:
    """End-to-end verification that assistant-message citations on the
    request survive the OpenAI-shape round-trip and land in the melody-
    rendered prompt as inline ``<co>...</co>`` markers.

    The chain covered here:

        CohereChatV2Request
            -> CohereServingChatV2._convert_v2_to_chat_completion
                -> ChatCompletionRequest.chat_template_kwargs["_messages_citations"]
                    -> _build_render_config (reads the entry)
                        -> cohere_melody.render_cmd4 (renders <co>...</co>)

    A regression that drops the citations or mangles the melody
    ``FilterCitation`` payload would remove the citation markers from
    the output, so a text-in / text-out assertion is enough.
    """

    @staticmethod
    def _melody():
        # Import locally so missing optional deps skip this test class
        # rather than failing collection. Two guards:
        # * ``cohere_melody`` -- the Rust binding this test drives.
        # * ``cohere`` -- required transitively by
        #   ``vllm.entrypoints.cohere.{protocol,serving}`` (both
        #   unconditionally ``from cohere.types import ...`` at module
        #   scope) which every test in this class imports locally.
        pytest.importorskip("cohere")
        pytest.importorskip("cohere_melody")
        import cohere_melody

        return cohere_melody

    @staticmethod
    def _openai_msgs_to_conversation(
        openai_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Cheap stand-in for ``parse_chat_messages`` on text-only inputs.

        ``_convert_v2_to_chat_completion`` emits OpenAI-shape assistant
        dicts with ``content`` as a plain string. The real chat-utils
        pipeline calls ``parse_chat_messages`` (which requires a full
        model config / mm tracker), but for text-only content it's
        essentially a passthrough -- we normalize ``content`` to the
        list-of-parts shape ``_conversation_to_melody_messages`` expects
        and preserve every other key the renderer reads.
        """
        conv: list[dict[str, Any]] = []
        for m in openai_messages:
            entry: dict[str, Any] = dict(m)
            content = entry.get("content")
            if isinstance(content, str):
                entry["content"] = [{"type": "text", "text": content}]
            elif content is None:
                entry["content"] = []
            conv.append(entry)
        return conv

    def test_document_citation_survives_to_prompt(self):
        # Prevent this test from being collected/run when the melody
        # extension isn't importable.
        melody = self._melody()

        from vllm.entrypoints.cohere.protocol import CohereChatV2Request
        from vllm.entrypoints.cohere.serving import CohereServingChatV2
        from vllm.renderers.cohere import _build_render_config

        request = CohereChatV2Request(
            model="m",
            messages=[
                {"role": "user", "content": "Who wrote Hamlet?"},
                {
                    "role": "assistant",
                    "content": "Shakespeare wrote it around 1600.",
                    "citations": [
                        {
                            "start": 0,
                            "end": 11,  # "Shakespeare"
                            "text": "Shakespeare",
                            "sources": [{"type": "document", "id": "doc_shakespeare"}],
                            "type": "TEXT_CONTENT",
                        }
                    ],
                },
                {"role": "user", "content": "and what year exactly?"},
            ],
            documents=[
                {
                    "id": "doc_shakespeare",
                    "data": {"text": "Hamlet was written by Shakespeare c. 1600."},
                }
            ],
        )

        # Step 1: v2 -> ChatCompletionRequest. Citations must land in
        # ``chat_template_kwargs["_messages_citations"]``.
        chat_req = CohereServingChatV2._convert_v2_to_chat_completion(request)
        assert chat_req.chat_template_kwargs is not None
        assert "_messages_citations" in chat_req.chat_template_kwargs

        # Step 2: build the melody render config. The renderer helper
        # folds the per-message citations onto the melody message
        # dicts.
        conversation = self._openai_msgs_to_conversation(chat_req.messages)
        fmt, config = _build_render_config(conversation, chat_req.chat_template_kwargs)

        assistant_msg = config["messages"][1]
        assert assistant_msg["role"] == "chatbot"
        assert "citations" in assistant_msg, (
            "citations were not attached to the melody assistant message dict"
        )

        # Step 3: hand the config to melody and check the rendered
        # prompt actually contains inline citation markup around the
        # cited span.
        #
        # The exact id is deterministic for this input. Melody builds
        # ``</co: <tool_call_index>:[<tool_result_indices>]>`` where
        # ``tool_call_index=0`` is the reserved bucket for the top-level
        # ``documents`` array and ``tool_result_indices`` are positions
        # inside it (see ``PromptRenderIds`` in melody/src/templating/
        # util.rs). ``doc_shakespeare`` sits at position 0 in the
        # request's ``documents`` list, so we expect ``0:[0]``. Two
        # historical regressions this pins:
        #   * ``0:[]`` -- the source id was never resolved to an index
        #     (documents didn't flow through) and melody had nothing to
        #     anchor the marker on.
        #   * ``1:[0]`` -- the citation was routed through the wrong
        #     tool-call bucket while documents were present, so it
        #     pointed at the wrong prompt slot.
        #
        # Note the same rendered prompt also contains an example
        # ``<co>span</co: 0:[1,2],1:[0]>`` marker baked into melody's
        # system-prompt boilerplate (placeholder text ``"span"``); the
        # substring below is specific enough to only match the marker
        # around the cited text.
        if fmt == "cmd4":
            rendered = melody.render_cmd4(config)
        else:
            rendered = melody.render_cmd3(config)

        assert "<co>Shakespeare</co: 0:[0]>" in rendered, (
            f"expected inline citation markup around the cited span; "
            f"tail of rendered prompt: {rendered[-400:]!r}"
        )

        # And the cited document's text itself must be in the prompt --
        # otherwise the model would have no way to satisfy the citation.
        assert "Hamlet" in rendered

    def test_no_markup_when_no_citations(self):
        # Control: the same request shape without any citations must
        # NOT contain ``<co>`` anywhere in the rendered prompt. Guards
        # against a false-positive where melody injects citation
        # markers regardless of what we passed in.
        melody = self._melody()

        from vllm.entrypoints.cohere.protocol import CohereChatV2Request
        from vllm.entrypoints.cohere.serving import CohereServingChatV2
        from vllm.renderers.cohere import _build_render_config

        request = CohereChatV2Request(
            model="m",
            messages=[
                {"role": "user", "content": "Who wrote Hamlet?"},
                {
                    "role": "assistant",
                    "content": "Shakespeare wrote it around 1600.",
                },
            ],
        )

        chat_req = CohereServingChatV2._convert_v2_to_chat_completion(request)
        assert (chat_req.chat_template_kwargs or {}).get("_messages_citations") is None

        conversation = self._openai_msgs_to_conversation(chat_req.messages)
        fmt, config = _build_render_config(
            conversation, chat_req.chat_template_kwargs or {}
        )

        if fmt == "cmd4":
            rendered = melody.render_cmd4(config)
        else:
            rendered = melody.render_cmd3(config)

        assert "<co>" not in rendered
        assert "</co:" not in rendered
