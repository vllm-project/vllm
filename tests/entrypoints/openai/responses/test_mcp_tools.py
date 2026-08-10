# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for MCP tool support in the Responses API."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from openai import OpenAI
from openai_harmony import Message, ToolDescription, ToolNamespaceConfig

from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.mcp.result import MCPClientSessionAdapter, normalize_tool_result
from vllm.entrypoints.mcp.tool_server import MCPToolServer

from .conftest import (
    BASE_TEST_ENV,
    events_contain_type,
    log_response_diagnostics,
    retry_for_tool_call,
    retry_streaming_for,
    validate_streaming_event_stack,
)

MODEL_NAME = "openai/gpt-oss-20b"

_BASE_SERVER_ARGS = [
    "--enforce-eager",
    "--tool-server",
    "demo",
    "--max_model_len",
    "5000",
]

_PYTHON_TOOL_INSTRUCTION = (
    "You must use the Python tool to execute code. Never simulate execution."
)


class TestMCPToolServerUnit:
    """Test MCPToolServer.get_tool_description filtering logic.

    Note: The wildcard "*" is normalized to None by
    _extract_allowed_tools_from_mcp_requests before reaching this layer,
    so we only test None and specific tool filtering here.
    See responses/test_serving_responses.py for "*" normalization tests.
    """

    def test_get_tool_description(self):
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
                name="test_server",
                description="test",
                tools=[tool1, tool2, tool3],
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
        result = server.get_tool_description("test_server", allowed_tools=["tool2"])
        assert len(result.tools) == 1
        assert result.tools[0].name == "tool2"

        # No matching tools - returns None
        result = server.get_tool_description(
            "test_server", allowed_tools=["nonexistent"]
        )
        assert result is None

        # Empty list - returns None
        assert server.get_tool_description("test_server", allowed_tools=[]) is None

    def test_builtin_tools_consistency(self):
        """MCP_BUILTIN_TOOLS must match BUILTIN_TOOL_TO_MCP_SERVER_LABEL values."""
        from vllm.entrypoints.openai.parser.harmony_utils import (
            BUILTIN_TOOL_TO_MCP_SERVER_LABEL,
            MCP_BUILTIN_TOOLS,
        )

        assert set(BUILTIN_TOOL_TO_MCP_SERVER_LABEL.values()) == MCP_BUILTIN_TOOLS, (
            f"MCP_BUILTIN_TOOLS {MCP_BUILTIN_TOOLS} does not match "
            f"BUILTIN_TOOL_TO_MCP_SERVER_LABEL values "
            f"{set(BUILTIN_TOOL_TO_MCP_SERVER_LABEL.values())}"
        )

    @pytest.mark.asyncio
    async def test_duplicate_server_name_keeps_first_server(self, monkeypatch):
        responses = iter(
            [
                (
                    SimpleNamespace(
                        serverInfo=SimpleNamespace(name="duplicate"),
                        instructions="first server",
                    ),
                    SimpleNamespace(tools=[]),
                ),
                (
                    SimpleNamespace(
                        serverInfo=SimpleNamespace(name="duplicate"),
                        instructions="second server",
                    ),
                    SimpleNamespace(tools=[]),
                ),
            ]
        )

        async def mock_list_server_and_tools(_url):
            return next(responses)

        monkeypatch.setattr(
            "vllm.entrypoints.mcp.tool_server.list_server_and_tools",
            mock_list_server_and_tools,
        )

        server = MCPToolServer.__new__(MCPToolServer)
        await server.add_tool_server("first:8000,second:8000")

        description = server.get_tool_description("duplicate")
        assert description is not None
        assert description.description == "first server"
        assert server.urls["duplicate"] == "http://first:8000/sse"


class TestMCPToolResultNormalization:
    def test_plain_text_result_is_unchanged(self):
        from mcp.types import CallToolResult, TextContent

        result = CallToolResult(content=[TextContent(text="plain output")])

        assert normalize_tool_result(result) is result

    @pytest.mark.parametrize(
        "result_kwargs",
        [
            {"content": []},
            {
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ]
            },
            {
                "content": [{"type": "text", "text": "invalid input"}],
                "isError": True,
            },
            {
                "content": [{"type": "text", "text": "summary"}],
                "structuredContent": {"answer": 42},
            },
            {
                "content": [
                    {"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png"}
                ]
            },
        ],
        ids=["empty", "multiple", "error", "structured", "image"],
    )
    def test_rich_result_preserves_mcp_fields_as_json(self, result_kwargs):
        from mcp.types import CallToolResult

        result = CallToolResult.model_validate(result_kwargs)
        normalized = normalize_tool_result(result)

        assert len(normalized.content) == 1
        assert normalized.content[0].type == "text"
        payload = json.loads(normalized.content[0].text)
        expected = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        if result.content and result.content[0].type == "image":
            expected["content"][0]["data"] = "<omitted 8 base64 characters>"
        assert payload == expected

    def test_transport_metadata_is_not_exposed_to_model(self):
        from mcp.types import CallToolResult, TextContent

        result = CallToolResult(
            content=[TextContent(text="output", _meta={"content-secret": "value"})],
            _meta={"result-secret": "value"},
            isError=True,
        )

        payload = json.loads(normalize_tool_result(result).content[0].text)

        assert "_meta" not in payload
        assert "_meta" not in payload["content"][0]

    @pytest.mark.asyncio
    async def test_session_adapter_normalizes_call_tool_result(self):
        from mcp.types import CallToolResult, TextContent

        result = CallToolResult(
            content=[TextContent(text="failed")],
            isError=True,
        )
        session = SimpleNamespace(call_tool=AsyncMock(return_value=result))

        normalized = await MCPClientSessionAdapter(session).call_tool(
            "search", {"query": "vLLM"}
        )

        session.call_tool.assert_awaited_once_with("search", {"query": "vLLM"})
        assert json.loads(normalized.content[0].text)["isError"] is True


class TestMCPEnabled:
    """Tests that require MCP tools to be enabled via environment variable."""

    @pytest.fixture(scope="class")
    def mcp_enabled_server(self):
        env_dict = {
            **BASE_TEST_ENV,
            "VLLM_ENABLE_RESPONSES_API_STORE": "1",
            "PYTHON_EXECUTION_BACKEND": "dangerously_use_uv",
            "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": ("code_interpreter,container"),
            "VLLM_GPT_OSS_HARMONY_SYSTEM_INSTRUCTIONS": "1",
        }
        with RemoteOpenAIServer(
            MODEL_NAME, list(_BASE_SERVER_ARGS), env_dict=env_dict
        ) as remote_server:
            yield remote_server

    @pytest_asyncio.fixture
    async def client(self, mcp_enabled_server):
        async with mcp_enabled_server.get_async_client() as async_client:
            yield async_client

    @staticmethod
    def _mcp_tools_payload(*, allowed_tools: list[str] | None = None) -> list[dict]:
        tool: dict = {
            "type": "mcp",
            "server_label": "code_interpreter",
            "server_url": "http://localhost:8888",
        }
        if allowed_tools is not None:
            tool["allowed_tools"] = allowed_tools
        return [tool]

    @staticmethod
    def _python_exec_input(code: str = "") -> str:
        if not code:
            code = "import random; print(random.randint(1, 1000000))"
        return f"Execute the following code: {code}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [MODEL_NAME])
    async def test_mcp_tool_env_flag_enabled(self, client: OpenAI, model_name: str):
        response = await retry_for_tool_call(
            client,
            model=model_name,
            expected_tool_type="mcp_call",
            input=self._python_exec_input(),
            instructions=_PYTHON_TOOL_INSTRUCTION,
            tools=self._mcp_tools_payload(),
            temperature=0.0,
            extra_body={"enable_response_messages": True},
        )

        assert response.status == "completed"
        log_response_diagnostics(response, label="MCP Enabled")

        tool_call_found = False
        tool_response_found = False
        for message in response.output_messages:
            recipient = message.get("recipient")
            if recipient and recipient.startswith("python"):
                tool_call_found = True
                assert message.get("channel") == "commentary"
            parsed_message = Message.from_dict(message)
            if parsed_message.author.role == "tool" and (
                parsed_message.author.name or ""
            ).startswith("python"):
                tool_response_found = True
                assert message.get("channel") == "commentary"

        assert tool_call_found, (
            f"No Python tool call found. "
            f"Output types: "
            f"{[getattr(o, 'type', None) for o in response.output]}"
        )
        assert tool_response_found, "No Python tool response found"

        for message in response.input_messages:
            assert Message.from_dict(message).author.role != "developer"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [MODEL_NAME])
    async def test_mcp_tool_with_allowed_tools_star(
        self, client: OpenAI, model_name: str
    ):
        response = await retry_for_tool_call(
            client,
            model=model_name,
            expected_tool_type="mcp_call",
            input=self._python_exec_input(),
            instructions=_PYTHON_TOOL_INSTRUCTION,
            tools=self._mcp_tools_payload(allowed_tools=["*"]),
            temperature=0.0,
            extra_body={"enable_response_messages": True},
        )

        assert response.status == "completed"
        log_response_diagnostics(response, label="MCP Allowed Tools *")

        tool_call_found = any(
            (msg.get("recipient") or "").startswith("python")
            for msg in response.output_messages
        )
        assert tool_call_found, (
            f"No Python tool call with '*'. "
            f"Output types: "
            f"{[getattr(o, 'type', None) for o in response.output]}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_name", [MODEL_NAME])
    async def test_mcp_tool_calling_streaming_types(
        self,
        pairs_of_event_types: dict[str, str],
        client: OpenAI,
        model_name: str,
    ):
        def _has_mcp_events(events: list) -> bool:
            return events_contain_type(events, "mcp_call")

        events = await retry_streaming_for(
            client,
            model=model_name,
            validate_events=_has_mcp_events,
            input=("What is 123 * 456? Use Python to calculate the result."),
            tools=[{"type": "mcp", "server_label": "code_interpreter"}],
            instructions=_PYTHON_TOOL_INSTRUCTION,
            temperature=0.0,
        )

        validate_streaming_event_stack(events, pairs_of_event_types)
