# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncIterator

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_completions,
)


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self) -> None:
        self.status = 200
        self.reason = "OK"
        self.headers = {
            "x-vllm-backend-id": "test-model",
            "x-vllm-endpoint-pool-id": "single-endpoint:test-model",
            "x-vllm-backend-scope": "local",
            "x-vllm-route-outcome": "local_only",
        }
        self.content = _FakeContent(
            [
                b'data: {"choices":[{"text":"hi"}]}\n\n',
                b'data: {"usage":{"completion_tokens":1,"prompt_tokens":2}}\n\n',
                b"data: [DONE]\n\n",
            ]
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    def post(self, *, url, json, headers):
        return _FakeResponse()


async def _request_output():
    return await async_request_openai_completions(
        RequestFuncInput(
            prompt="hello",
            api_url="http://test/v1/completions",
            prompt_len=2,
            output_len=1,
            model="test-model",
            request_id="req-1",
        ),
        _FakeSession(),
    )


def test_async_request_openai_completions_captures_routing_headers() -> None:
    output = asyncio.run(_request_output())

    assert output.success is True
    assert output.generated_text == "hi"
    assert output.response_metadata == {
        "x-vllm-backend-id": "test-model",
        "x-vllm-endpoint-pool-id": "single-endpoint:test-model",
        "x-vllm-backend-scope": "local",
        "x-vllm-route-outcome": "local_only",
    }
