# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.parser.harmony_utils import (
    parse_chat_input_to_harmony_message,
)
from vllm.entrypoints.openai.responses.harmony import (
    response_input_to_harmony,
    response_previous_input_to_harmony,
)
from vllm.exceptions import VLLMValidationError


@pytest.fixture()
def should_do_global_cleanup_after_test() -> bool:
    return False


def test_parse_chat_input_unsupported_content_type():
    chat_msg = {
        "role": "user",
        "content": [
            {
                "type": "input_file",
                "file": {
                    "filename": "test.pdf",
                    "file_data": "data:application/pdf;base64,dGVzdA==",
                },
            },
            {"type": "text", "text": "Summarize the document"},
        ],
    }

    with pytest.raises(VLLMValidationError) as exc_info:
        parse_chat_input_to_harmony_message(chat_msg)

    assert "Unsupported chat content part type: 'input_file'" in str(exc_info.value)
    assert exc_info.value.parameter == "type"
    assert exc_info.value.value == "input_file"


def test_response_previous_input_unsupported_content_type():
    chat_msg = {
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {
                    "filename": "test.pdf",
                    "file_data": "data:application/pdf;base64,dGVzdA==",
                },
            }
        ],
    }

    with pytest.raises(VLLMValidationError) as exc_info:
        response_previous_input_to_harmony(chat_msg)

    assert "Unsupported chat content part type: 'file'" in str(exc_info.value)
    assert exc_info.value.parameter == "type"
    assert exc_info.value.value == "file"


def test_response_input_unsupported_content_type():
    response_msg = {
        "type": "message",
        "role": "user",
        "content": [{"type": "invalid_type", "text": "test"}],
    }

    with pytest.raises(VLLMValidationError) as exc_info:
        response_input_to_harmony(response_msg, prev_responses=[])

    assert "Unsupported chat content part type: 'invalid_type'" in str(exc_info.value)
    assert exc_info.value.parameter == "type"
    assert exc_info.value.value == "invalid_type"
