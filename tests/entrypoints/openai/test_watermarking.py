# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


def test_chat_request_disables_watermarking():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}], watermarking=False
    )

    params = request.to_sampling_params(max_tokens=1, default_sampling_params={})

    assert not params.watermarking


def test_completion_request_disables_watermarking():
    request = CompletionRequest(prompt="hello", watermarking=False)

    params = request.to_sampling_params(max_tokens=1)

    assert not params.watermarking


def test_responses_request_disables_watermarking():
    request = ResponsesRequest(input="hello", watermarking=False)

    params = request.to_sampling_params(default_max_tokens=1)

    assert not params.watermarking
