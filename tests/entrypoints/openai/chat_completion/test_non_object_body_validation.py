# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Non-object JSON bodies must fail validation cleanly (4xx), not AttributeError (500).

mode=before validators that call data.get(...) without an isinstance(data, dict)
guard raise AttributeError for string/list/scalar bodies and surface as HTTP 500.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


@pytest.mark.parametrize(
    "payload",
    [
        "this is not valid json{{{",
        ["not", "an", "object"],
        42,
        None,
    ],
)
def test_chat_completion_request_rejects_non_object_body(payload):
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)


def test_chat_completion_request_cache_salt_still_validated_on_dict():
    with pytest.raises(ValidationError, match="cache_salt"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "cache_salt": "",
            }
        )
