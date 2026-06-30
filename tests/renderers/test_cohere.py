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
        out = _document_to_melody({"text": "x", "id": "d1"})
        assert out == {"text": "x", "id": "d1"}
        # Must be a defensive copy (mutating output should not affect input).
        out["new_key"] = "value"

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


# ======================================================================
# _build_render_config
# ======================================================================


class TestBuildRenderConfig:
    def _conv(self):
        return [{"role": "user", "content": "hi"}]

    def test_default_format_is_cmd3(self):
        fmt, cfg = _build_render_config(self._conv(), {})  # type: ignore[arg-type]
        assert fmt == "cmd3"
        assert cfg["use_jinja"] is True
        assert isinstance(cfg["messages"], list)
        # No additional_template_fields when no extra kwargs are set.
        assert "additional_template_fields" not in cfg

    def test_explicit_cmd4(self):
        fmt, cfg = _build_render_config(self._conv(), {"cohere_format": "cmd4"})  # type: ignore[arg-type]
        assert fmt == "cmd4"

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
        _, cfg = _build_render_config(self._conv(), {"safety_mode": "CONTEXTUAL"})  # type: ignore[arg-type]
        assert cfg["safety_mode"] == "contextual"

    def test_cmd3_citation_quality_direct(self):
        _, cfg = _build_render_config(self._conv(), {"citation_quality": "ACCURATE"})  # type: ignore[arg-type]
        assert cfg["citation_quality"] == "accurate"

    def test_cmd3_citation_quality_derived_from_citation_options(self):
        # When ``citation_quality`` is unset, ``citation_options.mode`` is
        # collapsed to on/off so cmd3's binary toggle has a value.
        _, cfg = _build_render_config(
            self._conv(), {"citation_options": {"mode": "accurate"}}
        )  # type: ignore[arg-type]
        assert cfg["citation_quality"] == "on"

        _, cfg = _build_render_config(
            self._conv(), {"citation_options": {"mode": "off"}}
        )  # type: ignore[arg-type]
        assert cfg["citation_quality"] == "off"

    def test_cmd3_skip_preamble_forwarded(self):
        _, cfg = _build_render_config(self._conv(), {"skip_preamble": True})  # type: ignore[arg-type]
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

    def test_template_id_and_template_jinja_passthrough(self):
        _, cfg = _build_render_config(
            self._conv(),
            {
                "template_id": "tpl1",
                "template_jinja": "raw {{ jinja }}",
            },
        )  # type: ignore[arg-type]
        assert cfg["template_id"] == "tpl1"
        assert cfg["template_jinja"] == "raw {{ jinja }}"
        # use_jinja is always True, regardless of caller input.
        assert cfg["use_jinja"] is True


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
