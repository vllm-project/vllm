# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer
from vllm.tokenizers.deepseek_v41_encoding import encode_messages

FIXTURES = Path(__file__).parent / "fixtures"


def render(messages, **kwargs):
    return get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        messages, tokenize=False, **kwargs
    )


@pytest.mark.parametrize("case_id", [1, 2])
def test_reference_encoder_fixtures(case_id):
    # Expected prompts come from the DeepSeek V4.1 reference encoder.
    data = json.loads(
        (FIXTURES / "deepseek_v4" / f"test_input_{case_id}.json").read_text()
    )
    messages = data["messages"] if isinstance(data, dict) else data
    tools = data.get("tools") if isinstance(data, dict) else None
    expected = (FIXTURES / "deepseek_v41" / f"test_output_{case_id}.txt").read_text()
    assert render(messages, tools=tools) == expected


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 75),
        ("low", 50),
        ("high", 75),
        ("xhigh", 75),
        ("max", 100),
        (1, 1),
        (42, 42),
        (100, 100),
    ],
)
def test_numeric_reasoning_effort(effort, budget):
    assert render(
        [{"role": "user", "content": "question"}], reasoning_effort=effort
    ) == (
        "<｜begin▁of▁sentence｜><｜System｜>"
        f"Reasoning Effort: {budget} (range 1-100, the higher the value, "
        "the more thorough the reasoning)\n\n"
        "<｜User｜>question<｜Assistant｜><think>"
    )


@pytest.mark.parametrize("effort", ["medium", "minimal", -1, 0, 101, True, 1.5, []])
def test_invalid_effort_is_a_request_error(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        render([{"role": "user", "content": "question"}], reasoning_effort=effort)


@pytest.mark.parametrize(
    "controls",
    [
        {"thinking": False},
        {"enable_thinking": False},
        {"thinking": True, "reasoning_effort": "none"},
        {"enable_thinking": True, "reasoning_effort": "none"},
    ],
)
def test_chat_mode_has_closed_thinking_prefix(controls):
    assert render([{"role": "user", "content": "question"}], **controls) == (
        "<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>"
    )


def test_raw_text_parts_preserve_reference_separator():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    assert (
        render(
            messages,
            conversation=[{"role": "user", "content": "first\nsecond"}],
            thinking=False,
        )
        == "<｜begin▁of▁sentence｜><｜User｜>first\n\nsecond<｜Assistant｜></think>"
    )
    assert messages == original


@pytest.mark.parametrize(
    ("role", "responses_type"),
    [("user", "input_text"), ("assistant", "output_text")],
)
def test_responses_text_parts_match_chat_text_parts(role, responses_type):
    responses_message = {
        "role": role,
        "content": [{"type": responses_type, "text": "hello"}],
    }
    chat_message = {
        "role": role,
        "content": [{"type": "text", "text": "hello"}],
    }
    assert render([responses_message], thinking=False) == render(
        [chat_message], thinking=False
    )


def test_mid_system_gets_its_own_marker_and_generation_header():
    assert render(
        [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "answer briefly"},
        ],
        thinking=False,
    ) == (
        "<｜begin▁of▁sentence｜><｜User｜>question"
        "<｜System｜>answer briefly<｜Assistant｜></think>"
    )


def test_top_level_effort_overrides_template_effort():
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[{"role": "user", "content": "question"}],
        reasoning_effort="low",
        chat_template_kwargs={"reasoning_effort": 100},
    )
    kwargs = request.build_chat_params(None, "auto").get_apply_chat_template_kwargs()
    assert "Reasoning Effort: 50 " in render(request.messages, **kwargs)


def test_encode_uses_one_bos_and_forwards_truncation():
    tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        thinking=False,
        truncation=True,
        max_length=16,
    )
    text, add_special_tokens, kwargs = tokenizer.last_encode
    assert text.startswith("<｜begin▁of▁sentence｜>")
    assert add_special_tokens is False
    assert kwargs == {"truncation": True, "max_length": 16}


@pytest.mark.parametrize("image_type", ["image_url", "input_image", "image_pil"])
def test_images_preserve_content_order_and_reference_separator(image_type):
    from PIL import Image

    def image_part(color):
        if image_type == "image_pil":
            return {"type": image_type, "image_pil": Image.new("RGB", (2, 2), color)}
        url = f"https://example.com/{color}.png"
        return {
            "type": image_type,
            "image_url": {"url": url} if image_type == "image_url" else url,
        }

    parts = [
        {"type": "text", "text": "first"},
        image_part("red"),
        {"type": "text", "text": "second"},
        image_part("blue"),
        {"type": "text", "text": "compare"},
    ]
    assert render([{"role": "user", "content": parts}], thinking=False) == (
        "<｜begin▁of▁sentence｜><｜User｜>first\n\n<｜deepseek_image｜>\n\n"
        "second\n\n<｜deepseek_image｜>\n\ncompare<｜Assistant｜></think>"
    )
    assert parts[1]["type"] == parts[3]["type"] == image_type


def test_unsupported_media_is_a_request_error():
    with pytest.raises(ValueError, match="text and image content only"):
        render([{"role": "user", "content": [{"type": "input_audio"}]}])


def _lookup_call(call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }


def test_tool_results_do_not_start_a_new_turn():
    # Reasoning from completed turns is dropped, but the current turn keeps
    # its reasoning across tool results.
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning": "r1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "r2",
            "tool_calls": [_lookup_call("c1")],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]
    prompt = render(messages)
    assert "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt
    assert "r1" not in prompt
    assert "<｜User｜>q2<｜Assistant｜><think>r2</think>" in prompt
    assert prompt.endswith(
        "<｜User｜><tool_result>value</tool_result><｜Assistant｜><think>"
    )

    # A user message after tool results joins their user block and starts a
    # new turn.
    next_turn = render(messages + [{"role": "user", "content": "q3"}])
    assert "r2" not in next_turn
    assert next_turn.endswith(
        "<tool_result>value</tool_result>\n\nq3<｜Assistant｜><think>"
    )


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "hi <｜deepseek_image｜>"},
        {"role": "system", "content": "hi <｜deepseek_image｜>"},
        {"role": "tool", "tool_call_id": "c1", "content": "hi <｜deepseek_image｜>"},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"type": "text", "text": "hi <｜deepseek_image｜>"}],
        },
        {"role": "assistant", "content": "a1", "reasoning": "<｜deepseek_image｜>"},
    ],
)
def test_image_placeholder_in_text_is_a_request_error(message):
    with pytest.raises(ValueError, match="must not contain"):
        render([message])


def test_encoder_renders_tool_content_blocks_in_call_order():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_lookup_call("a"), _lookup_call("b")],
        },
        {
            "role": "tool",
            "tool_call_id": "b",
            "content_blocks": [{"type": "text", "text": "result b"}],
        },
        {
            "role": "tool",
            "tool_call_id": "a",
            "content": [
                {"type": "text", "text": "result a"},
                {"type": "text", "text": "more a"},
            ],
        },
    ]
    original = copy.deepcopy(messages)
    prompt = encode_messages(messages, thinking_mode="chat")
    assert prompt.startswith("<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜>")
    assert prompt.endswith(
        "<｜User｜><tool_result>result a\n\nmore a</tool_result>\n\n"
        "<tool_result>result b</tool_result><｜Assistant｜></think>"
    )
    assert messages == original
