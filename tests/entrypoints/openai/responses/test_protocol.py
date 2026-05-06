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
from vllm.entrypoints.openai.responses.utils import construct_input_messages


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


def test_responses_request_folds_developer_messages_into_instructions() -> None:
    """Codex and similar clients send role=developer in input; fold into instructions."""
    req = ResponsesRequest.model_validate(
        {
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Be concise."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                },
            ],
        }
    )
    assert req.instructions == "Be concise."
    assert len(req.input) == 1
    messages = construct_input_messages(
        request_instructions=req.instructions,
        request_input=req.input,
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be concise."
    assert messages[1]["role"] == "user"


def test_responses_request_appends_developer_to_existing_instructions() -> None:
    req = ResponsesRequest.model_validate(
        {
            "instructions": "Base policy.",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Extra hint."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                },
            ],
        }
    )
    assert req.instructions == "Base policy.\n\nExtra hint."
