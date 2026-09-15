# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai.types.responses import ResponseCompactionItem
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesCompactRequest,
    ResponsesRequest,
    serialize_message,
    serialize_messages,
)
from vllm.entrypoints.openai.responses.utils import encode_compaction_summary


def test_compact_request_only_requires_model() -> None:
    request = ResponsesCompactRequest.model_validate({"model": "test-model"})

    assert request.model == "test-model"
    assert request.input is None
    assert request.service_tier == "auto"


def test_compaction_item_is_accepted_as_response_input() -> None:
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": [
                {
                    "id": "cmp_1",
                    "type": "compaction",
                    "encrypted_content": encode_compaction_summary("checkpoint"),
                }
            ],
        }
    )

    assert isinstance(request.input, list)
    assert isinstance(request.input[0], ResponseCompactionItem)


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
