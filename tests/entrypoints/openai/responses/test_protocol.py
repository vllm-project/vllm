# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    serialize_message,
    serialize_messages,
)


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]


def test_responses_request_accepts_chat_template_kwargs() -> None:
    request = ResponsesRequest(
        input="test input",
        reasoning={"effort": "none"},
        chat_template_kwargs={"enable_thinking": False},
    )

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    )

    assert chat_params.chat_template_kwargs["enable_thinking"] is False
    assert chat_params.chat_template_kwargs["reasoning_effort"] == "none"


def test_responses_internal_chat_template_kwargs_take_precedence() -> None:
    request = ResponsesRequest(
        input="test input",
        chat_template_kwargs={
            "add_generation_prompt": False,
            "continue_final_message": True,
        },
    )

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    )

    assert chat_params.chat_template_kwargs["add_generation_prompt"] is True
    assert chat_params.chat_template_kwargs["continue_final_message"] is False
