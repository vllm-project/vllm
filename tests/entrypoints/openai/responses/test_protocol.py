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
        input="Hello",
        chat_template_kwargs={"enable_thinking": False},
    )

    assert request.chat_template_kwargs == {"enable_thinking": False}


def test_build_chat_params_merges_responses_chat_template_kwargs() -> None:
    request = ResponsesRequest(
        input="Hello",
        chat_template_kwargs={"enable_thinking": False},
        reasoning={"effort": "low"},
    )

    chat_params = request.build_chat_params(
        default_template=None,
        default_template_content_format="auto",
    )

    assert chat_params.chat_template_kwargs == {
        "enable_thinking": False,
        "add_generation_prompt": True,
        "continue_final_message": False,
        "reasoning_effort": "low",
    }
