# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import httpx
import pytest
import pytest_asyncio
from openai import OpenAI

from vllm.entrypoints.openai.protocol import ResponsesResponse

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module")
def server(monkeypatch_module: pytest.MonkeyPatch):
    args = ["--enforce-eager", "--tool-server", "demo"]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


async def send_responses_request(server, data: dict) -> dict:
    """Helper function to send requests using HTTP."""
    async with httpx.AsyncClient(timeout=120.0) as http_client:
        response = await http_client.post(
            f"{server.url_root}/v1/responses",
            json=data,
            headers={"Authorization": f"Bearer {server.DUMMY_API_KEY}"})

        response.raise_for_status()
        return response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_output_messages_enabled(client: OpenAI, model_name: str,
                                       server):
    # It is necessary to not use the OpenAI cient as
    # enable_response_messages is not a valid param for OpenAI
    response_json = await send_responses_request(
        server, {
            "model": model_name,
            "input": "What is the capital of South Korea?",
            "enable_response_messages": True,
        })

    response = ResponsesResponse.model_validate(response_json)
    assert response is not None
    assert response.status == "completed"
    assert len(response.input_messages) > 0
    assert len(response.output_messages) > 0
