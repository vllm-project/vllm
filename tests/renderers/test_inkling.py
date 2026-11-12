# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden tests for the native Inkling renderer encoding.

The message fixtures mirror the Rust renderer's fixture tests
(`rust/src/chat/src/renderer/inkling/tests.rs`) so the two frontends stay
in token-level parity. Image blocks emit the bare `<|content_image|>` marker;
`InklingMultiModalProcessor` inserts the per-patch placeholder run.
"""

import pytest

from vllm.renderers.inkling import (
    InklingRenderer,
    _HfBackedTmlTokenizer,
    _resolve_inkling2_reasoning_effort,
    _resolve_reasoning_effort,
)
from vllm.renderers.inkling_encoding import (
    OPTIONAL_SPECIAL_TOKENS,
    SPECIAL_TOKEN_SPELLINGS,
    render_inkling2_messages,
    render_inkling_messages,
)
from vllm.renderers.params import ChatParams


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    # These tests touch no distributed or device state; the global
    # cleanup fixture is unnecessary (and trips a torch MPS allocator
    # assert on macOS dev machines).
    return False


# One id per special token, mirroring the real Inkling vocab layout; plain
# text encodes one token per character so decoded output is exact.
_SPECIAL_VOCAB = {
    "<|message_user|>": 200000,
    "<|message_model|>": 200001,
    "<|message_system|>": 200002,
    "<|message_tool|>": 200003,
    "<|content_text|>": 200004,
    "<|content_image|>": 200005,
    "<|content_model_end_sampling|>": 200006,
    "<|content_thinking|>": 200008,
    "<|end_message|>": 200010,
    "<|content_audio_input|>": 200020,
    # The HF vocab spells CONTENT_XML as an unused slot.
    "<|unused_200024|>": 200024,
    "<|audio_end|>": 200043,
    "<|content_invoke_tool_json|>": 200049,
}

_ID_TO_SPECIAL = {v: k for k, v in _SPECIAL_VOCAB.items()}


class FakeHfTokenizer:
    def get_vocab(self):
        return dict(_SPECIAL_VOCAB)

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(ch) for ch in text]


def decode(token_ids):
    return "".join(
        _ID_TO_SPECIAL.get(tid, chr(tid) if tid < 200000 else f"<{tid}>")
        for tid in token_ids
    )


@pytest.fixture
def inkling_tokenizer():
    return _HfBackedTmlTokenizer(FakeHfTokenizer())


def render_text(inkling_tokenizer, messages, **kwargs):
    return decode(render_inkling_messages(messages, inkling_tokenizer, **kwargs))


class TestRustFixtureParity:
    def test_tool_round_trip(self, inkling_tokenizer):
        messages = [
            {
                "role": "assistant",
                "reasoning_content": "think",
                "content": "answer",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"SF"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        assert render_text(
            inkling_tokenizer, messages, add_generation_prompt=False
        ) == (
            "<|message_model|><|content_thinking|>think<|end_message|>"
            "<|message_model|><|content_text|>answer<|end_message|>"
            "<|message_model|>get_weather<|content_invoke_tool_json|>"
            '{"name":"get_weather","args":{"city":"SF"}}<|end_message|>'
            "<|content_model_end_sampling|>"
            "<|message_tool|>get_weather<|content_text|>sunny<|end_message|>"
        )

    def test_tool_declare(self, inkling_tokenizer):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "required": ["city"],
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        messages = [
            {
                "role": "developer",
                "content": "rules",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "local_tool",
                            "parameters": {"z": 1, "a": {"b": 2}},
                        },
                    }
                ],
            },
            {"role": "user", "content": "hi"},
        ]
        assert render_text(inkling_tokenizer, messages, tools=tools) == (
            "<|message_system|>tool_declare<|unused_200024|>"
            '[{"description":"Get weather information","name":"get_weather",'
            '"parameters":{"properties":{"city":{"type":"string"}},'
            '"required":["city"],"type":"object"},"type":"function"},'
            '{"description":"","name":"local_tool",'
            '"parameters":{"a":{"b":2},"z":1},"type":"function"}]'
            "<|end_message|>"
            "<|message_system|><|content_text|>rules<|end_message|>"
            "<|message_user|><|content_text|>hi<|end_message|>"
            "<|message_model|>"
        )

    def test_text_image(self, inkling_tokenizer):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": "data:image/png;base64,"},
                ],
            }
        ]
        # Multimodal preprocessing expands this bare marker after rendering.
        assert render_text(inkling_tokenizer, messages) == (
            "<|message_user|><|content_text|>look<|end_message|>"
            "<|message_user|><|content_image|><|end_message|>"
            "<|message_model|>"
        )


class TestRenderingSemantics:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("none", 0.0),
            ("minimal", 0.1),
            ("low", 0.2),
            ("medium", 0.7),
            ("high", 0.9),
            ("xhigh", 0.99),
            ("max", 0.99),
            (None, 0.9),
            (0.8, 0.8),
            (True, None),
            ("invalid", None),
        ],
    )
    def test_resolve_reasoning_effort(self, value, expected):
        assert _resolve_reasoning_effort(value) == expected

    def test_generation_prompt_default(self, inkling_tokenizer):
        text = render_text(inkling_tokenizer, [{"role": "user", "content": "hi"}])
        assert text.endswith("<|end_message|><|message_model|>")

    def test_developer_folds_into_system(self, inkling_tokenizer):
        assert (
            render_text(
                inkling_tokenizer,
                [{"role": "developer", "content": "be nice"}],
                add_generation_prompt=False,
            )
            == "<|message_system|><|content_text|>be nice<|end_message|>"
        )

    def test_empty_string_content_skipped(self, inkling_tokenizer):
        assert (
            render_text(
                inkling_tokenizer,
                [{"role": "user", "content": ""}],
                add_generation_prompt=False,
            )
            == ""
        )

    def test_empty_reasoning_skipped(self, inkling_tokenizer):
        assert (
            render_text(
                inkling_tokenizer,
                [{"role": "assistant", "reasoning_content": "", "content": "hi"}],
                add_generation_prompt=False,
            )
            == "<|message_model|><|content_text|>hi<|end_message|>"
            "<|content_model_end_sampling|>"
        )

    def test_reasoning_field(self, inkling_tokenizer):
        messages = [
            {
                "role": "assistant",
                "reasoning": "think",
                "content": "answer",
            }
        ]
        assert render_text(
            inkling_tokenizer, messages, add_generation_prompt=False
        ) == (
            "<|message_model|><|content_thinking|>think<|end_message|>"
            "<|message_model|><|content_text|>answer<|end_message|>"
            "<|content_model_end_sampling|>"
        )

    def test_audio_part(self, inkling_tokenizer):
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_audio", "input_audio": {}}],
            }
        ]
        assert render_text(
            inkling_tokenizer, messages, add_generation_prompt=False
        ) == ("<|message_user|><|content_audio_input|><|audio_end|><|end_message|>")

    def test_tool_response_name_from_message(self, inkling_tokenizer):
        assert (
            render_text(
                inkling_tokenizer,
                [{"role": "tool", "name": "my_tool", "content": "ok"}],
                add_generation_prompt=False,
            )
            == "<|message_tool|>my_tool<|content_text|>ok<|end_message|>"
        )

    def test_tool_call_args_object_form(self, inkling_tokenizer):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "x",
                        "function": {
                            "name": "f",
                            # dict-form arguments, unsorted keys
                            "arguments": {"z": 1, "a": 2},
                        },
                    }
                ],
            }
        ]
        assert render_text(
            inkling_tokenizer, messages, add_generation_prompt=False
        ) == (
            "<|message_model|>f<|content_invoke_tool_json|>"
            '{"name":"f","args":{"a":2,"z":1}}<|end_message|>'
            "<|content_model_end_sampling|>"
        )

    def test_tool_call_empty_args(self, inkling_tokenizer):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": ""}}],
            }
        ]
        assert render_text(
            inkling_tokenizer, messages, add_generation_prompt=False
        ) == (
            "<|message_model|>f<|content_invoke_tool_json|>"
            '{"name":"f","args":{}}<|end_message|>'
            "<|content_model_end_sampling|>"
        )

    def test_tool_call_non_object_args_rejected(self, inkling_tokenizer):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "x", "function": {"name": "f", "arguments": "[1]"}}
                ],
            }
        ]
        with pytest.raises(TypeError, match="decode to an object"):
            render_inkling_messages(
                messages, inkling_tokenizer, add_generation_prompt=False
            )

    def test_unsupported_role_rejected(self, inkling_tokenizer):
        with pytest.raises(ValueError, match="unsupported Inkling message role"):
            render_inkling_messages(
                [{"role": "narrator", "content": "hi"}], inkling_tokenizer
            )


class TestReasoningEffort:
    def test_frontend_defaults_to_high(self, inkling_tokenizer):
        renderer = InklingRenderer.__new__(InklingRenderer)
        renderer._inkling_tokenizer = inkling_tokenizer
        renderer._is_inkling2 = False

        text = decode(
            renderer._render(
                [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "hi"},
                ],
                ChatParams(),
            )
        )

        assert text.startswith(
            "<|message_system|><|content_text|>rules<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
        )

    @pytest.mark.parametrize("value", ["none", 0, 0.0, -0.0])
    def test_zero_effort_has_one_canonical_spelling(self, inkling_tokenizer, value):
        renderer = InklingRenderer.__new__(InklingRenderer)
        renderer._inkling_tokenizer = inkling_tokenizer
        renderer._is_inkling2 = False

        text = decode(
            renderer._render(
                [{"role": "user", "content": "hi"}],
                ChatParams(chat_template_kwargs={"reasoning_effort": value}),
            )
        )

        assert text.startswith(
            "<|message_system|><|content_text|>Thinking effort level: 0.0"
            "<|end_message|>"
        )

    def test_emits_one_effort_after_initial_prefix(self, inkling_tokenizer):
        effort_block = (
            "<|message_system|><|content_text|>Thinking effort level: 0.7"
            "<|end_message|>"
        )
        text = render_text(
            inkling_tokenizer,
            [
                {"role": "system", "content": "rules"},
                {"role": "developer", "content": "policy"},
                {"role": "user", "content": "user1"},
                {"role": "assistant", "content": "assistant1"},
                {"role": "user", "content": "user2"},
            ],
            tools=[{"type": "function", "function": {"name": "f"}}],
            reasoning_effort=0.7,
        )

        assert text.count(effort_block) == 1
        assert (
            text.index("tool_declare")
            < text.index("rules")
            < text.index("policy")
            < text.index(effort_block)
            < text.index("user1")
            < text.index("assistant1")
            < text.index("user2")
        )

    def test_renders_after_tool_declare(self, inkling_tokenizer):
        tools = [{"type": "function", "function": {"name": "f"}}]
        text = render_text(
            inkling_tokenizer,
            [{"role": "user", "content": "hi"}],
            tools=tools,
            reasoning_effort=0.8,
        )
        declare_end = text.index("<|end_message|>") + len("<|end_message|>")
        assert text[declare_end:].startswith(
            "<|message_system|><|content_text|>Thinking effort level: 0.8"
            "<|end_message|>"
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.2, "0.2"), (0.7, "0.7"), (0.9, "0.9"), (0.99, "0.99")],
    )
    def test_formats_at_most_two_decimals(self, inkling_tokenizer, value, expected):
        text = render_text(
            inkling_tokenizer,
            [{"role": "user", "content": "hi"}],
            reasoning_effort=value,
        )
        assert f"Thinking effort level: {expected}<|end_message|>" in text

    def test_absent_emits_no_block(self, inkling_tokenizer):
        text = render_text(inkling_tokenizer, [{"role": "user", "content": "hi"}])
        assert "Thinking effort" not in text

    @pytest.mark.parametrize("value", [0.9900001, 1, 1.5, -0.1])
    def test_out_of_range_rejected(self, inkling_tokenizer, value):
        with pytest.raises(ValueError, match="must be in"):
            render_inkling_messages(
                [{"role": "user", "content": "hi"}],
                inkling_tokenizer,
                reasoning_effort=value,
            )


class TestSpecialTokenResolution:
    def test_missing_special_token_raises(self):
        class IncompleteTokenizer(FakeHfTokenizer):
            def get_vocab(self):
                vocab = dict(_SPECIAL_VOCAB)
                del vocab["<|content_invoke_tool_json|>"]
                return vocab

        with pytest.raises(ValueError, match="missing special tokens"):
            _HfBackedTmlTokenizer(IncompleteTokenizer())

    def test_semantic_spelling_preferred(self):
        class SemanticSpellingTokenizer(FakeHfTokenizer):
            def get_vocab(self):
                vocab = dict(_SPECIAL_VOCAB)
                del vocab["<|unused_200024|>"]
                vocab["<|content_xml|>"] = 200024
                return vocab

        inkling_tokenizer = _HfBackedTmlTokenizer(SemanticSpellingTokenizer())
        tools = [{"type": "function", "function": {"name": "f"}}]
        ids = render_inkling_messages(
            [{"role": "user", "content": "hi"}], inkling_tokenizer, tools=tools
        )
        assert 200024 in ids

    def test_all_spellings_covered(self):
        # Every semantic token must resolve from the reference vocab.
        for token, spellings in SPECIAL_TOKEN_SPELLINGS.items():
            if token not in OPTIONAL_SPECIAL_TOKENS:
                assert any(s in _SPECIAL_VOCAB for s in spellings), token


def test_render_inkling_messages_direct_protocol():
    """The encoding core only needs the structural tokenizer protocol."""

    class ProtocolTokenizer:
        def encode_text(self, text):
            return [ord(c) for c in text]

        def encode_special(self, token):
            return {
                "<|message_user|>": 200000,
                "<|message_model|>": 200001,
                "<|content_text|>": 200004,
                "<|content_model_end_sampling|>": 200006,
                "<|end_message|>": 200010,
            }[token]

    ids = render_inkling_messages(
        [{"role": "user", "content": "hi"}],
        ProtocolTokenizer(),
        add_generation_prompt=True,
    )
    assert ids == [200000, 200004, ord("h"), ord("i"), 200010, 200001]


# ── Inkling2 ──────────────────────────────────────────────────────────────
# Expected strings are the reference renderer's output for the same requests.

_INKLING2_VOCAB = {
    "<|endoftext|>": 199999,
    "<|message_user|>": 200000,
    "<|message_model|>": 200001,
    "<|message_system|>": 200002,
    "<|message_tool|>": 200003,
    "<|content_text|>": 200004,
    "<|content_image|>": 200005,
    "<|content_model_end_sampling|>": 200006,
    "<|content_thinking|>": 200008,
    "<|end_message|>": 200010,
    "<|content_tool_error|>": 200022,
    "<|content_xml|>": 200024,
    "<|content_invoke_tool_json|>": 200049,
    "<|content_invoke_tool_text|>": 200057,
}
_INKLING2_ID_TO_SPECIAL = {v: k for k, v in _INKLING2_VOCAB.items()}

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class FakeInkling2HfTokenizer:
    def get_vocab(self):
        return dict(_INKLING2_VOCAB)

    def encode(self, text, add_special_tokens=False, split_special_tokens=False):
        assert not add_special_tokens and split_special_tokens
        return [ord(ch) for ch in text]


def render_inkling2_text(messages, **kwargs):
    tokenizer = _HfBackedTmlTokenizer(
        FakeInkling2HfTokenizer(), split_special_tokens=True
    )
    return "".join(
        _INKLING2_ID_TO_SPECIAL.get(tid, chr(tid))
        for tid in render_inkling2_messages(messages, tokenizer, **kwargs)
    )


class TestInkling2Rendering:
    def test_system_and_user_without_generation_prefix(self):
        assert render_inkling2_text(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ]
        ) == (
            "<|message_system|><|content_text|>You are a concise assistant."
            "<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            "<|message_user|><|content_text|>What is 2 + 2?<|end_message|>"
        )

    def test_reasoning_history(self):
        assert render_inkling2_text(
            [
                {"role": "user", "content": "What is 2 + 2?"},
                {
                    "role": "assistant",
                    "reasoning_content": "Add the two numbers.",
                    "content": "4",
                },
                {"role": "user", "content": "Thanks!"},
            ],
            reasoning_effort=0.8,
        ) == (
            "<|message_system|><|content_text|>Thinking effort level: 0.8"
            "<|end_message|>"
            "<|message_user|><|content_text|>What is 2 + 2?<|end_message|>"
            "<|message_model|><|content_thinking|>Add the two numbers."
            "<|end_message|>"
            "<|message_model|><|content_text|>4<|end_message|>"
            "<|content_model_end_sampling|>"
            "<|message_user|><|content_text|>Thanks!<|end_message|>"
        )

    def test_tool_round_trip(self):
        messages = [
            {"role": "user", "content": "What should I wear in SF today?"},
            {
                "role": "assistant",
                "reasoning_content": "Check the current weather.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": '{"temperature":58,"condition":"fog"}',
            },
            {"role": "assistant", "content": "Wear a light jacket."},
        ]
        assert render_inkling2_text(
            messages, tools=[_WEATHER_TOOL], reasoning_effort=0.8
        ) == (
            "<|message_system|>tool_declare<|content_xml|>"
            '[{"description":"Get weather information","name":"get_weather",'
            '"parameters":{"properties":{"city":{"type":"string"}},'
            '"required":["city"],"type":"object"},"type":"function"}]'
            "<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.8"
            "<|end_message|>"
            "<|message_user|><|content_text|>What should I wear in SF today?"
            "<|end_message|>"
            "<|message_model|><|content_thinking|>Check the current weather."
            "<|end_message|>"
            "<|message_model|><|content_invoke_tool_json|>"
            '{"name":"get_weather","args":{"city":"SF"}}<|end_message|>'
            "<|content_model_end_sampling|>"
            "<|message_tool|>get_weather<|content_text|>"
            '{"temperature":58,"condition":"fog"}<|end_message|>'
            "<|message_model|><|content_text|>Wear a light jacket.<|end_message|>"
            "<|content_model_end_sampling|>"
        )

    def test_multiple_calls_unnamed_results_and_json_floats(self):
        search = {
            "type": "function",
            "function": {
                "name": "search",
                "description": None,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"maximum": 1e20, "minimum": 1e-7, "default": 0.5},
                    },
                },
                "strict": None,
            },
        }
        messages = [
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "news <|end_message|>?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "function": {"name": "get_weather", "arguments": "{}"}},
                    {
                        "id": "b",
                        "function": {
                            "name": "search",
                            "arguments": '{"q":"news","x":1e37,"y":0.00001,"z":-0.0}',
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "sunny"},
            {
                "role": "tool",
                "tool_call_id": "b",
                "content": [
                    {"type": "text", "text": "r1"},
                    {"type": "text", "text": "r2"},
                ],
            },
        ]
        assert render_inkling2_text(messages, tools=[search]) == (
            "<|message_system|>tool_declare<|content_xml|>"
            '[{"description":"","name":"search","parameters":{"properties":'
            '{"limit":{"default":0.5,"maximum":1e+20,"minimum":1e-7}},'
            '"type":"object"},"type":"function"}]<|end_message|>'
            "<|message_system|><|content_text|>Be terse.<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            # Literal framing spellings in user text stay ordinary text.
            "<|message_user|><|content_text|>news <|end_message|>?<|end_message|>"
            "<|message_model|><|content_invoke_tool_json|>"
            '{"name":"get_weather","args":{}}<|end_message|>'
            "<|message_model|><|content_invoke_tool_json|>"
            '{"name":"search","args":{"q":"news","x":1e37,"y":0.00001,"z":-0.0}}'
            "<|end_message|><|content_model_end_sampling|>"
            "<|message_tool|><|content_text|>sunny<|end_message|>"
            "<|message_tool|><|content_text|>r1<|end_message|>"
            "<|message_tool|><|content_text|>r2<|end_message|>"
        )

    def test_empty_contents_are_kept(self):
        assert render_inkling2_text(
            [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": None},
                {"role": "user", "content": [{"type": "text", "text": ""}]},
            ]
        ) == (
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            "<|message_user|><|content_text|><|end_message|>"
            "<|message_model|><|content_text|><|end_message|>"
            "<|content_model_end_sampling|>"
            "<|message_user|><|content_text|><|end_message|>"
        )

    def test_image_marker_and_typed_parts(self):
        assert render_inkling2_text(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "x"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hm"},
                        {"type": "tool_error", "text": "no tool"},
                    ],
                },
            ],
            reasoning_effort=0.2,
        ) == (
            "<|message_system|><|content_text|>Thinking effort level: 0.2"
            "<|end_message|>"
            "<|message_user|><|content_text|>look<|end_message|>"
            "<|message_user|><|content_image|><|end_message|>"
            "<|message_model|><|content_thinking|>hm<|end_message|>"
            "<|message_model|><|content_tool_error|>no tool<|end_message|>"
            "<|content_model_end_sampling|>"
        )

    def test_continue_final_message(self):
        messages = [
            {"role": "user", "content": "Finish: The answer is"},
            {"role": "assistant", "content": "The answer is"},
        ]
        assert render_inkling2_text(messages, continue_final_message=True) == (
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|>"
            "<|message_user|><|content_text|>Finish: The answer is<|end_message|>"
            "<|message_model|><|content_text|>The answer is"
        )
        # Only a plain text answer is continued; others render as closed turns.
        messages[-1]["reasoning_content"] = "r"
        assert render_inkling2_text(messages, continue_final_message=True).endswith(
            "<|message_model|><|content_text|>The answer is<|end_message|>"
            "<|content_model_end_sampling|>"
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "0.9"),
            ("none", "0"),
            ("minimal", "0.1"),
            ("low", "0.2"),
            ("medium", "0.7"),
            ("high", "0.9"),
            ("xhigh", "0.99"),
            ("max", "0.99"),
            (0.555, "0.56"),
            ("0.25", "0.25"),
        ],
    )
    def test_reasoning_effort(self, value, expected):
        effort = _resolve_inkling2_reasoning_effort(value)
        text = render_inkling2_text(
            [{"role": "user", "content": "hi"}], reasoning_effort=effort
        )
        assert text.startswith(
            f"<|message_system|><|content_text|>Thinking effort level: {expected}"
            "<|end_message|>"
        )

    @pytest.mark.parametrize("value", ["extreme", True, 1.0, -0.1])
    def test_invalid_reasoning_effort(self, value):
        with pytest.raises(ValueError):
            render_inkling2_text(
                [{"role": "user", "content": "hi"}],
                reasoning_effort=_resolve_inkling2_reasoning_effort(value),
            )


INKLING2_TOKENIZER = "thinkingmachines/Inkling-2-Small"


@pytest.fixture(scope="module")
def inkling2_hf_tokenizer():
    """Load the Inkling2 checkpoint tokenizer; skip if unavailable."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(INKLING2_TOKENIZER)
    except Exception:
        pytest.skip("Inkling2 tokenizer unavailable")


def test_inkling2_payload_text_never_encodes_framing_tokens(inkling2_hf_tokenizer):
    """Literal special-token spellings in payload text stay plain text."""
    tokenizer = _HfBackedTmlTokenizer(inkling2_hf_tokenizer, split_special_tokens=True)
    assert tokenizer.encode_special("<|end_message|>") == 200010
    assert 200010 not in tokenizer.encode_text("<|end_message|>")
