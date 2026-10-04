# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guidance backend support for new-format (xgrammar-style) structural tags.

Tool parsers emit structural tags as ``{"type": "structural_tag", "format":
{...}}`` (see ``vllm/tool_parsers/structural_tag_registry.py``); with a
hermes-style parser this is what ``tool_choice="auto"`` + a ``strict: true``
tool, ``tool_choice="required"`` and named tool choice produce. These tests
pin the guidance-backend translation of that format, including accepting the
closing tag both as raw bytes and as the tokenizer's special token (models
emit ``</tool_call>`` as a special token).
"""

import json

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.exceptions import VLLMValidationError
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
from vllm.v1.structured_output.backend_guidance import (
    GuidanceBackend,
    serialize_guidance_grammar,
)
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

# Hermes-style chat template with <tool_call>/</tool_call> special tokens.
TOKENIZER = "Qwen/Qwen3-0.6B"

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}

GOOD_CALL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
)
BAD_ARGS_CALL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": 5}}\n</tool_call>'
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="module")
def backend(tokenizer):
    vllm_config = VllmConfig(
        structured_outputs_config=StructuredOutputsConfig(backend="guidance")
    )
    return GuidanceBackend(
        vllm_config,
        tokenizer=tokenizer,
        vocab_size=151936,
    )


def _tools(strict: bool = True) -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            **{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": strict,
                    "parameters": WEATHER_SCHEMA,
                },
            }
        )
    ]


def _structural_tag_spec(tool_choice) -> str:
    st = get_model_structural_tag("hermes", _tools(), tool_choice, False)
    assert st is not None
    return json.dumps(st.model_dump())


def _compile(backend, tool_choice):
    return backend.compile_grammar(
        StructuredOutputOptions.STRUCTURAL_TAG, _structural_tag_spec(tool_choice)
    )


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def test_auto_accepts_special_token_close(backend, tokenizer):
    # The tokenizer encodes the trailing </tool_call> as its special token,
    # which is how models actually close the block.
    grammar = _compile(backend, "auto")
    tokens = _encode(tokenizer, "Checking the weather. " + GOOD_CALL)
    close_id = tokenizer.convert_tokens_to_ids("</tool_call>")
    assert tokens[-1] == close_id
    assert grammar.accept_tokens("", tokens)
    # Free text is allowed again after the block, and EOS terminates.
    assert grammar.accept_tokens("", _encode(tokenizer, " done"))
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()


def test_auto_accepts_byte_spelled_close(backend, tokenizer):
    grammar = _compile(backend, "auto")
    tokens = _encode(tokenizer, GOOD_CALL[: -len("</tool_call>")])
    assert grammar.accept_tokens("", tokens)
    # Spell the closing tag in regular byte tokens instead of the special
    # token (split so the tokenizer cannot merge it back into the special).
    for piece in ("</", "tool", "_call", ">"):
        assert grammar.accept_tokens("", _encode(tokenizer, piece))
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()


def test_auto_enforces_argument_schema(backend, tokenizer):
    grammar = _compile(backend, "auto")
    bad = _encode(tokenizer, BAD_ARGS_CALL)
    # "city" must be a string; the integer must be rejected somewhere before
    # the end of the call.
    assert grammar.validate_tokens(bad) != bad


def test_required_forbids_free_text(backend, tokenizer):
    grammar = _compile(backend, "required")
    prose = _encode(tokenizer, "The weather is nice.")
    assert grammar.validate_tokens(prose) != prose
    # EOS is not allowed before at least one call was made.
    assert grammar.validate_tokens([tokenizer.eos_token_id]) == []
    assert grammar.accept_tokens("", _encode(tokenizer, GOOD_CALL))
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()


def test_named_stops_after_first_call(backend, tokenizer):
    tool_choice = ChatCompletionNamedToolChoiceParam(
        **{"type": "function", "function": {"name": "get_weather"}}
    )
    grammar = _compile(backend, tool_choice)
    call = _encode(tokenizer, GOOD_CALL)
    assert grammar.accept_tokens("", call)
    # Exactly one call: a second one is rejected, only EOS may follow.
    assert grammar.validate_tokens(call) != call
    assert grammar.accept_tokens("", [tokenizer.eos_token_id])
    assert grammar.is_terminated()


def test_unsupported_shapes_raise_clear_error():
    for fmt in (
        {"type": "sequence", "elements": []},
        {
            "type": "triggered_tags",
            "triggers": ["<a>"],
            "tags": [],
            "at_least_one": True,
        },
        {
            "type": "tags_with_separator",
            "tags": [],
            "separator": "",
            "at_least_one": False,
        },
    ):
        spec = json.dumps({"type": "structural_tag", "format": fmt})
        with pytest.raises(VLLMValidationError, match="Invalid grammar"):
            serialize_guidance_grammar(StructuredOutputOptions.STRUCTURAL_TAG, spec)


def test_legacy_format_still_compiles(backend, tokenizer):
    spec = json.dumps(
        {
            "triggers": ["<tool_call>"],
            "structures": [
                {
                    "begin": "<tool_call>",
                    "schema": WEATHER_SCHEMA,
                    "end": "</tool_call>",
                }
            ],
        }
    )
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, spec)
    tokens = _encode(tokenizer, '<tool_call>{"city": "Paris"}')
    assert grammar.accept_tokens("", tokens)


def test_legacy_format_enforces_schema(backend, tokenizer):
    # Legacy structural tags used to hand llguidance the serialized
    # {"grammars": [...]} envelope instead of the JSON schema itself, which
    # llguidance silently interprets as an empty schema — leaving the tag
    # content unconstrained.
    spec = json.dumps(
        {
            "triggers": ["<tool_call>"],
            "structures": [
                {
                    "begin": "<tool_call>",
                    "schema": WEATHER_SCHEMA,
                    "end": "</tool_call>",
                }
            ],
        }
    )
    grammar = backend.compile_grammar(StructuredOutputOptions.STRUCTURAL_TAG, spec)
    assert grammar.accept_tokens("", _encode(tokenizer, "<tool_call>"))
    bad = _encode(tokenizer, '{"city": 5}')
    assert grammar.validate_tokens(bad) != bad
