# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import pytest_asyncio
from openai import OpenAI

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module")
def mcp_disabled_server(monkeypatch_module: pytest.MonkeyPatch):
    args = ["--enforce-eager", "--tool-server", "demo"]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        m.setenv("PYTHON_EXECUTION_BACKEND", "dangerously_use_uv")
        # Helps the model follow instructions better
        m.setenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "1")
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.fixture(scope="function")
def mcp_enabled_server(monkeypatch_module: pytest.MonkeyPatch):
    args = ["--enforce-eager", "--tool-server", "demo"]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        m.setenv("PYTHON_EXECUTION_BACKEND", "dangerously_use_uv")
        m.setenv("VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "code_interpreter,container")
        # Helps the model follow instructions better
        m.setenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "1")
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest_asyncio.fixture
async def mcp_disabled_client(mcp_disabled_server):
    async with mcp_disabled_server.get_async_client() as async_client:
        yield async_client


@pytest_asyncio.fixture
async def mcp_enabled_client(mcp_enabled_server):
    async with mcp_enabled_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_mcp_tool_env_flag_enabled(mcp_enabled_client: OpenAI, model_name: str):
    response = await mcp_enabled_client.responses.create(
        model=model_name,
        input=(
            "Execute the following code: "
            "import random; print(random.randint(1, 1000000))"
        ),
        instructions=(
            "You must use the Python tool to execute code. Never simulate execution."
        ),
        tools=[
            {
                "type": "mcp",
                "server_label": "code_interpreter",
                # URL unused for DemoToolServer
                "server_url": "http://localhost:8888",
            }
        ],
        extra_body={"enable_response_messages": True},
    )
    assert response is not None
    assert response.status == "completed"
    # Verify output messages: Tool calls and responses on analysis channel
    tool_call_found = False
    tool_response_found = False
    for message in response.output_messages:
        recipient = message.get("recipient")
        if recipient and recipient.startswith("python"):
            tool_call_found = True
            assert message.get("channel") == "analysis", (
                "Tool call should be on analysis channel"
            )
        author = message.get("author", {})
        if (
            author.get("role") == "tool"
            and author.get("name")
            and author.get("name").startswith("python")
        ):
            tool_response_found = True
            assert message.get("channel") == "analysis", (
                "Tool response should be on analysis channel"
            )

    assert tool_call_found, "Should have found at least one Python tool call"
    assert tool_response_found, "Should have found at least one Python tool response"
    for message in response.input_messages:
        assert message.get("author").get("role") != "developer", (
            "No developer messages should be present with valid mcp tool"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
async def test_mcp_tool_env_flag_disabled(mcp_disabled_client: OpenAI, model_name: str):
    response = await mcp_disabled_client.responses.create(
        model=model_name,
        input=(
            "Execute the following code if the tool is present: "
            "import random; print(random.randint(1, 1000000))"
        ),
        tools=[
            {
                "type": "mcp",
                "server_label": "code_interpreter",
                # URL unused for DemoToolServer
                "server_url": "http://localhost:8888",
            }
        ],
        extra_body={"enable_response_messages": True},
    )
    assert response is not None
    assert response.status == "completed"
    # Verify output messages: No tool calls and responses
    tool_call_found = False
    tool_response_found = False
    for message in response.output_messages:
        recipient = message.get("recipient")
        if recipient and recipient.startswith("python"):
            tool_call_found = True
            assert message.get("channel") == "analysis", (
                "Tool call should be on analysis channel"
            )
        author = message.get("author", {})
        if (
            author.get("role") == "tool"
            and author.get("name")
            and author.get("name").startswith("python")
        ):
            tool_response_found = True
            assert message.get("channel") == "analysis", (
                "Tool response should be on analysis channel"
            )

    assert not tool_call_found, "Should not have a python call"
    assert not tool_response_found, "Should not have a tool response"
    for message in response.input_messages:
        assert message.get("author").get("role") != "developer", (
            "No developer messages should be present without a valid tool"
        )
