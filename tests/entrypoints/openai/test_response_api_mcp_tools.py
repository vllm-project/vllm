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
        with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
            yield remote_server


@pytest.fixture(scope="function")
def mcp_enabled_server(monkeypatch_module: pytest.MonkeyPatch):
    args = ["--enforce-eager", "--tool-server", "demo"]

    with monkeypatch_module.context() as m:
        m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
        m.setenv("PYTHON_EXECUTION_BACKEND", "dangerously_use_uv")
        m.setenv("GPT_OSS_SYSTEM_TOOL_MCP_LABELS",
                 "code_interpreter,container")
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
@pytest.mark.skip(reason="Code interpreter tool is not available in CI yet.")
async def test_mcp_tool_env_flag_enabled(mcp_enabled_client: OpenAI,
                                         model_name: str):
    response = await mcp_enabled_client.responses.create(
        model=model_name,
        # TODO: Ideally should be able to set max tool calls
        # to prevent multi-turn, but it is not currently supported
        # would speed up the test
        input=("What's the first 4 digits after the decimal point of "
               "cube root of `19910212 * 20250910`? "
               "Show only the digits. The python interpreter is not stateful "
               "and you must print to see the output."),
        tools=[{
            "type": "mcp",
            "server_label": "code_interpreter",
            # URL unused for DemoToolServer
            "server_url": "http://localhost:8888"
        }],
    )
    assert response is not None
    assert response.status == "completed"
    assert response.usage.output_tokens_details.tool_output_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.skip(reason="Code interpreter tool is not available in CI yet.")
async def test_mcp_tool_env_flag_disabled(mcp_disabled_client: OpenAI,
                                          model_name: str):
    response = await mcp_disabled_client.responses.create(
        model=model_name,
        # TODO: Ideally should be able to set max tool calls
        # to prevent multi-turn, but it is not currently supported
        # would speed up the test
        input=("What's the first 4 digits after the decimal point of "
               "cube root of `19910212 * 20250910`? "
               "Show only the digits. The python interpreter is not stateful "
               "and you must print to see the output."),
        tools=[{
            "type": "mcp",
            "server_label": "code_interpreter",
            # URL unused for DemoToolServer
            "server_url": "http://localhost:8888"
        }],
    )
    assert response is not None
    assert response.status == "completed"
    assert response.usage.output_tokens_details.tool_output_tokens == 0
