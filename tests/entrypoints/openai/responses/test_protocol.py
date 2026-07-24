# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    serialize_message,
    serialize_messages,
)


@pytest.mark.parametrize("audio_format", ["mp3", "wav"])
def test_input_audio_in_message_content(audio_format: str) -> None:
    # Regression test for #47659: input_audio content parts inside a user
    # message must be accepted on /v1/responses just as they are on
    # /v1/chat/completions.
    req = ResponsesRequest.model_validate(
        {
            "model": "test",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": "AAAA",
                                "format": audio_format,
                            },
                        },
                        {"type": "input_text", "text": "what did I say?"},
                    ],
                }
            ],
        }
    )
    audio_part = req.input[0]["content"][0]
    assert audio_part["type"] == "input_audio"
    assert audio_part["input_audio"]["data"] == "AAAA"
    assert audio_part["input_audio"]["format"] == audio_format


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
