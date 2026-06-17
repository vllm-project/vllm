# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator

import httpx
from fastapi import FastAPI
from httpx import ASGITransport

from vllm.entrypoints.openai.chat_completion import api_router as chat_api_router
from vllm.entrypoints.openai.completion import api_router as completion_api_router


class _StubCompletionHandler:
    async def create_completion(
        self,
        request,
        raw_request,
    ) -> AsyncGenerator[str, None]:
        async def _stream() -> AsyncGenerator[str, None]:
            yield 'data: {"id":"cmpl-test","choices":[]}' + "\n\n"
            yield "data: [DONE]\n\n"

        return _stream()

    def build_routing_headers(self, model_name: str | None = None) -> dict[str, str]:
        return {
            "x-vllm-backend-id": model_name or "test-model",
            "x-vllm-endpoint-pool-id": f"single-endpoint:{model_name or 'test-model'}",
            "x-vllm-backend-scope": "local",
            "x-vllm-route-outcome": "local_only",
        }


class _StubChatHandler:
    async def create_chat_completion(
        self,
        request,
        raw_request,
    ) -> AsyncGenerator[str, None]:
        async def _stream() -> AsyncGenerator[str, None]:
            yield 'data: {"id":"chatcmpl-test","choices":[]}' + "\n\n"
            yield "data: [DONE]\n\n"

        return _stream()

    def build_routing_headers(self, model_name: str | None = None) -> dict[str, str]:
        return {
            "x-vllm-backend-id": model_name or "test-model",
            "x-vllm-endpoint-pool-id": f"single-endpoint:{model_name or 'test-model'}",
            "x-vllm-backend-scope": "local",
            "x-vllm-route-outcome": "local_only",
        }


async def _post_completion() -> httpx.Response:
    app = FastAPI()
    completion_api_router.attach_router(app)
    app.state.openai_serving_completion = _StubCompletionHandler()

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": True,
            },
        )


async def _post_chat_completion() -> httpx.Response:
    app = FastAPI()
    chat_api_router.attach_router(app)
    app.state.openai_serving_chat = _StubChatHandler()

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "stream": True,
            },
        )


def test_completion_streaming_response_includes_routing_headers() -> None:
    response = asyncio.run(_post_completion())

    assert response.status_code == 200
    assert response.headers["x-vllm-backend-id"] == "test-model"
    assert response.headers["x-vllm-endpoint-pool-id"] == "single-endpoint:test-model"
    assert response.headers["x-vllm-backend-scope"] == "local"
    assert response.headers["x-vllm-route-outcome"] == "local_only"


def test_chat_streaming_response_includes_routing_headers() -> None:
    response = asyncio.run(_post_chat_completion())

    assert response.status_code == 200
    assert response.headers["x-vllm-backend-id"] == "test-model"
    assert response.headers["x-vllm-endpoint-pool-id"] == "single-endpoint:test-model"
    assert response.headers["x-vllm-backend-scope"] == "local"
    assert response.headers["x-vllm-route-outcome"] == "local_only"
