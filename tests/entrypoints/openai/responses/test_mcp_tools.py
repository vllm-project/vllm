# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import pytest_asyncio
from openai import OpenAI
from openai_harmony import ToolDescription, ToolNamespaceConfig

from vllm.entrypoints.tool_server import MCPToolServer

from ....utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


def test_get_tool_description():
    """Test MCPToolServer.get_tool_description filtering logic.

    Note: The wildcard "*" is normalized to None by
    _extract_allowed_tools_from_mcp_requests before reaching this layer,
    so we only test None and specific tool filtering here.
    See test_serving_responses.py for "*" normalization tests.
    """
    pytest.importorskip("mcp")

    server = MCPToolServer()
    tool1 = ToolDescription.new(
        name="tool1", description="First", parameters={"type": "object"}
    )
    tool2 = ToolDescription.new(
        name="tool2", description="Second", parameters={"type": "object"}
    )
    tool3 = ToolDescription.new(
        name="tool3", description="Third", parameters={"type": "object"}
    )

    server.harmony_tool_descriptions = {
        "test_server": ToolNamespaceConfig(
            name="test_server", description="test", tools=[tool1, tool2, tool3]
        )
    }

    # Nonexistent server
    assert server.get_tool_description("nonexistent") is None

    # None (no filter) - returns all tools
    result = server.get_tool_description("test_server", allowed_tools=None)
    assert len(result.tools) == 3

    # Filter to specific tools
    result = server.get_tool_description(
        "test_server", allowed_tools=["tool1", "tool3"]
    )
    assert len(result.tools) == 2
    assert result.tools[0].name == "tool1"
    assert result.tools[1].name == "tool3"

    # Single tool
    result = server.get_tool_description(
        "test_server",
        allowed_tools=["tool2"],
    )
    assert len(result.tools) == 1
    assert result.tools[0].name == "tool2"

    # No matching tools - returns None
    result = server.get_tool_description("test_server", allowed_tools=["nonexistent"])
    assert result is None

    # Empty list - returns None
    assert server.get_tool_description("test_server", allowed_tools=[]) is None


class TestMCPEnabled:
    """Tests that require MCP tools to be enabled via environment variable."""

    @pytest.fixture(scope="class")
    def monkeypatch_class(self):
        from _pytest.monkeypatch import MonkeyPatch

        mpatch = MonkeyPatch()
        yield mpatch
        mpatch.undo()

    @pytest.fixture(scope="class")
    def mcp_enabled_server(self, monkeypatch_class: pytest.MonkeyPatch):
        args = ["--enforce-eager", "--tool-server", "demo"]

        with monkeypatch_class.context() as m:
            m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
            m.setenv("PYTHON_EXECUTION_BACKEND", "dangerously_use_uv")
            m.setenv(
                "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS", "code_interpreter,container"
            )
            # Helps the model follow instructions better
            m.setenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "1")
            with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
                yield remote_server

    @pytest_asyncio.fixture
    async def mcp_enabled_client(self, mcp_enabled_server):
        async with mcp_enabled_server.get_async_client() as async_client:
            yield async_client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [MODEL_NAME])
    async def test_mcp_tool_env_flag_enabled(
        self, mcp_enabled_client: OpenAI, model_name: str
    ):
        response = await mcp_enabled_client.responses.create(
            model=model_name,
            input=(
                "Execute the following code: "
                "import random; print(random.randint(1, 1000000))"
            ),
            instructions=(
                "You must use the Python tool to execute code. "
                "Never simulate execution."
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
        assert tool_response_found, (
            "Should have found at least one Python tool response"
        )
        for message in response.input_messages:
            assert message.get("author").get("role") != "developer", (
                "No developer messages should be present with valid mcp tool"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [MODEL_NAME])
    async def test_mcp_tool_with_allowed_tools_star(
        self, mcp_enabled_client: OpenAI, model_name: str
    ):
        """Test MCP tool with allowed_tools=['*'] to select all available
        tools.

        This E2E test verifies that the "*" wildcard works end-to-end.
        See test_serving_responses.py for detailed unit tests of "*"
        normalization.
        """
        response = await mcp_enabled_client.responses.create(
            model=model_name,
            input=(
                "Execute the following code: "
                "import random; print(random.randint(1, 1000000))"
            ),
            instructions=(
                "You must use the Python tool to execute code. "
                "Never simulate execution."
            ),
            tools=[
                {
                    "type": "mcp",
                    "server_label": "code_interpreter",
                    "server_url": "http://localhost:8888",
                    # Using "*" to allow all tools from this MCP server
                    "allowed_tools": ["*"],
                }
            ],
            extra_body={"enable_response_messages": True},
        )
        assert response is not None
        assert response.status == "completed"
        # Verify tool calls work with allowed_tools=["*"]
        tool_call_found = False
        for message in response.output_messages:
            recipient = message.get("recipient")
            if recipient and recipient.startswith("python"):
                tool_call_found = True
                break
        assert tool_call_found, (
            "Should have found at least one Python tool call with '*'"
        )

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [MODEL_NAME])
    async def test_mcp_tool_calling_streaming_types(
        self, mcp_enabled_client: OpenAI, model_name: str
    ):
        pairs_of_event_types = {
            "response.completed": "response.created",
            "response.output_item.done": "response.output_item.added",
            "response.content_part.done": "response.content_part.added",
            "response.output_text.done": "response.output_text.delta",
            "response.reasoning_text.done": "response.reasoning_text.delta",
            "response.reasoning_part.done": "response.reasoning_part.added",
            "response.mcp_call_arguments.done": ("response.mcp_call_arguments.delta"),
            "response.mcp_call.completed": "response.mcp_call.in_progress",
        }

        tools = [
            {
                "type": "mcp",
                "server_label": "code_interpreter",
            }
        ]
        input_text = "What is 13 * 24? Use python to calculate the result."

        stream_response = await mcp_enabled_client.responses.create(
            model=model_name,
            input=input_text,
            tools=tools,
            stream=True,
            instructions=(
                "You must use the Python tool to execute code. "
                "Never simulate execution."
            ),
        )

        stack_of_event_types = []
        saw_mcp_type = False
        async for event in stream_response:
            if event.type == "response.created":
                stack_of_event_types.append(event.type)
            elif event.type == "response.completed":
                assert stack_of_event_types[-1] == pairs_of_event_types[event.type]
                stack_of_event_types.pop()
            elif (
                event.type.endswith("added")
                or event.type == "response.mcp_call.in_progress"
            ):
                stack_of_event_types.append(event.type)
            elif event.type.endswith("delta"):
                if stack_of_event_types[-1] == event.type:
                    continue
                stack_of_event_types.append(event.type)
            elif (
                event.type.endswith("done")
                or event.type == "response.mcp_call.completed"
            ):
                assert stack_of_event_types[-1] == pairs_of_event_types[event.type]
                if "mcp_call" in event.type:
                    saw_mcp_type = True
                stack_of_event_types.pop()

        assert len(stack_of_event_types) == 0
        assert saw_mcp_type, "Should have seen at least one mcp call"


class TestMCPDisabled:
    """Tests that verify behavior when MCP tools are disabled."""

    @pytest.fixture(scope="class")
    def monkeypatch_class(self):
        from _pytest.monkeypatch import MonkeyPatch

        mpatch = MonkeyPatch()
        yield mpatch
        mpatch.undo()

    @pytest.fixture(scope="class")
    def mcp_disabled_server(self, monkeypatch_class: pytest.MonkeyPatch):
        args = ["--enforce-eager", "--tool-server", "demo"]

        with monkeypatch_class.context() as m:
            m.setenv("VLLM_ENABLE_RESPONSES_API_STORE", "1")
            m.setenv("PYTHON_EXECUTION_BACKEND", "dangerously_use_uv")
            # Helps the model follow instructions better
            m.setenv("VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS", "1")
            with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
                yield remote_server

    @pytest_asyncio.fixture
    async def mcp_disabled_client(self, mcp_disabled_server):
        async with mcp_disabled_server.get_async_client() as async_client:
            yield async_client

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [MODEL_NAME])
    async def test_mcp_tool_env_flag_disabled(
        self, mcp_disabled_client: OpenAI, model_name: str
    ):
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
