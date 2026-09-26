# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_chat_completions,
    async_request_openai_completions,
)

pytestmark = pytest.mark.skip_global_cleanup

RequestCall = Callable[[RequestFuncInput, Any], Awaitable[RequestFuncOutput]]

_EXPECTED_TEXT = " leading 中 trailing "


class _FakeContent:
    def __init__(self, chunks: tuple[bytes, bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    status = 200
    reason = "OK"

    def __init__(self, chunks: tuple[bytes, bytes]) -> None:
        self.content = _FakeContent(chunks)

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *_: object) -> bool:
        return False


class _FakeSession:
    def __init__(self, chunks: tuple[bytes, bytes]) -> None:
        self._chunks = chunks

    def post(self, **_: Any) -> _FakeResponse:
        return _FakeResponse(self._chunks)


@pytest.mark.parametrize(
    ("request_func", "api_url", "payload"),
    [
        (
            async_request_openai_completions,
            "http://localhost:8000/v1/completions",
            {"choices": [{"text": _EXPECTED_TEXT}]},
        ),
        (
            async_request_openai_chat_completions,
            "http://localhost:8000/v1/chat/completions",
            {"choices": [{"delta": {"content": _EXPECTED_TEXT}}]},
        ),
    ],
)
def test_streamed_output_is_invariant_to_transport_chunk_boundaries(
    request_func: RequestCall,
    api_url: str,
    payload: dict[str, Any],
) -> None:
    """iter_any() yields arbitrary transport fragments, so parsing must
    give the same result no matter where the byte stream is split."""
    event = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    stream = f"data: {event}\n\ndata: [DONE]\n\n".encode()

    request = RequestFuncInput(
        prompt="test",
        api_url=api_url,
        prompt_len=1,
        output_len=8,
        model="test-model",
    )

    async def run_all_splits() -> list[tuple[int, bool, str, str]]:
        failures = []
        for split_at in range(1, len(stream)):
            session = _FakeSession((stream[:split_at], stream[split_at:]))
            output = await request_func(request, session)
            if not output.success or output.generated_text != _EXPECTED_TEXT:
                error = output.error.splitlines()[-1] if output.error else ""
                failures.append(
                    (split_at, output.success, output.generated_text, error)
                )
        return failures

    assert asyncio.run(run_all_splits()) == []
