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
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize("max_num_results", [-1, 0, 51, 10**12])
@pytest.mark.skip_global_cleanup
def test_file_search_rejects_invalid_max_num_results(max_num_results: int) -> None:
    with pytest.raises(VLLMValidationError, match="max_num_results"):
        ResponsesRequest(
            model="test-model",
            input="search",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": ["vs_test"],
                    "max_num_results": max_num_results,
                }
            ],
        )


@pytest.mark.parametrize("max_num_results", [1, 50])
@pytest.mark.skip_global_cleanup
def test_file_search_accepts_max_num_results_boundaries(
    max_num_results: int,
) -> None:
    request = ResponsesRequest(
        model="test-model",
        input="search",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": ["vs_test"],
                "max_num_results": max_num_results,
            }
        ],
    )

    assert request.tools[0].max_num_results == max_num_results


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
