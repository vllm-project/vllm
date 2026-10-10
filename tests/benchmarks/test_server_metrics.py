# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import aiohttp
import pytest

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    _update_server_metrics,
    async_request_openai_chat_completions,
    async_request_openai_completions,
)
from vllm.benchmarks.serve import _get_server_metrics_results


class _ResponseContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _Response:
    status = 200
    reason = ""

    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _ResponseContent(chunks)

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _Session:
    def __init__(self, chunks: list[bytes], expected_url: str) -> None:
        self._chunks = chunks
        self._expected_url = expected_url

    def post(self, **kwargs: Any) -> _Response:
        assert kwargs["url"] == self._expected_url
        return _Response(self._chunks)


def _sse(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data)}".encode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_func", "endpoint", "choice"),
    [
        (
            async_request_openai_completions,
            "/v1/completions",
            {"choices": [{"text": "x"}]},
        ),
        (
            async_request_openai_chat_completions,
            "/v1/chat/completions",
            {"choices": [{"delta": {"content": "x"}}]},
        ),
    ],
)
async def test_openai_request_preserves_server_queue_time_and_ttft(
    request_func: Callable[..., Any],
    endpoint: str,
    choice: dict[str, Any],
) -> None:
    usage = {
        "choices": [],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        "metrics": {
            "queue_time_ms": 0.0,
            "time_to_first_token_ms": 1250.0,
        },
    }
    api_url = f"https://example.test{endpoint}"
    request = RequestFuncInput(
        prompt="test",
        api_url=api_url,
        prompt_len=4,
        output_len=1,
        model="test-model",
    )

    output = await request_func(
        request,
        cast(
            aiohttp.ClientSession,
            _Session(
                [_sse(choice), _sse(usage), b"data: [DONE]"],
                expected_url=api_url,
            ),
        ),
    )

    assert output.success
    assert output.server_queue_time == 0.0
    assert output.server_ttft == pytest.approx(1.25)


@pytest.mark.parametrize(
    "data",
    [
        {"metrics": None},
        {"metrics": {"queue_time_ms": None}},
        {"metrics": "invalid"},
        {},
    ],
)
def test_server_metrics_tolerate_unavailable_values(
    data: dict[str, Any],
) -> None:
    output = RequestFuncOutput()

    _update_server_metrics(output, data)

    assert output.server_queue_time is None
    assert output.server_ttft is None


def test_server_metrics_results_preserve_alignment() -> None:
    outputs = [
        RequestFuncOutput(
            server_queue_time=0.1,
            server_ttft=0.2,
        ),
        RequestFuncOutput(),
        RequestFuncOutput(
            server_ttft=0.3,
        ),
    ]

    assert _get_server_metrics_results(outputs) == {
        "server_queue_times": [0.1, None, None],
        "server_ttfts": [0.2, None, 0.3],
    }
    assert _get_server_metrics_results(
        [RequestFuncOutput(server_queue_time=0.0)]
    ) == {
        "server_queue_times": [0.0],
        "server_ttfts": [None],
    }
    assert _get_server_metrics_results(
        [RequestFuncOutput(server_ttft=0.0)]
    ) == {
        "server_queue_times": [None],
        "server_ttfts": [0.0],
    }
    assert _get_server_metrics_results([RequestFuncOutput()]) == {}
