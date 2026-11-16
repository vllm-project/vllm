# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for MCP tool support in the Responses API."""

from __future__ import annotations

import socket
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from openai import BadRequestError, OpenAI
from openai_harmony import Message, ToolDescription, ToolNamespaceConfig

import vllm.entrypoints.mcp.tool_server as tool_server_module
from tests.utils import RemoteOpenAIServer
from vllm.entrypoints.mcp.tool_server import (
    MCPToolServer,
    init_tool_server,
    post_process_tools_description,
)

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


def test_mcp_allowlist_rejects_model_generated_call() -> None:
    """A generated function call must not bypass the MCP execution policy."""
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    calls: list[str] = []
    mcp_server = MCPServer("python")

    @mcp_server.tool()
    def python(code: str) -> str:
        calls.append(code)
        return "2"

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        tool_server = uvicorn.Server(
            uvicorn.Config(mcp_server.sse_app(), log_level="warning")
        )
        thread = threading.Thread(
            target=tool_server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        thread.start()
        try:
            model = "Qwen/Qwen3-0.6B"
            args = [
                "--enforce-eager",
                "--max-model-len",
                "2048",
                "--max-num-seqs",
                "1",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "hermes",
                "--reasoning-parser",
                "qwen3",
                "--tool-server",
                f"127.0.0.1:{port}",
            ]
            env = {
                **BASE_TEST_ENV,
                "VLLM_USE_EXPERIMENTAL_PARSER_CONTEXT": "1",
                "VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS": "code_interpreter",
            }
            with RemoteOpenAIServer(model, args, env_dict=env) as server:
                client = server.get_client().with_options(max_retries=0)
                tools: list[dict[str, Any]] = [
                    {
                        "type": "mcp",
                        "server_label": "code_interpreter",
                        "allowed_tools": ["python"],
                    },
                    {
                        "type": "function",
                        "name": "code_interpreter",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "enum": ["print(1+1)"]},
                            },
                            "required": ["code"],
                            "additionalProperties": False,
                        },
                    },
                ]
                request = {
                    "model": model,
                    "input": "Use code_interpreter to calculate 1+1.",
                    "tools": tools,
                    "tool_choice": {"type": "function", "name": "code_interpreter"},
                    "temperature": 0.0,
                    "max_output_tokens": 128,
                    "max_tool_calls": 1,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                }
                response = client.responses.create(**request)
                outputs = [
                    item.output for item in response.output if item.type == "mcp_call"
                ]
                assert outputs == ["2"]
                assert calls == ["print(1+1)"]

                tools[0]["allowed_tools"] = []
                with pytest.raises(
                    BadRequestError, match="not permitted by allowed_tools"
                ):
                    client.responses.create(**request)
                assert calls == ["print(1+1)"]
        finally:
            tool_server.should_exit = True
            thread.join(timeout=10)


@pytest.mark.asyncio
async def test_init_tool_server_without_configuration():
    args = SimpleNamespace(tool_server=None)

    assert await init_tool_server(args) is None


@pytest.mark.asyncio
async def test_init_tool_server_with_mcp_url(monkeypatch: pytest.MonkeyPatch):
    server = MagicMock()
    server.add_tool_server = AsyncMock()
    monkeypatch.setattr(tool_server_module, "MCPToolServer", lambda: server)

    result = await init_tool_server(SimpleNamespace(tool_server="http://mcp.test"))

    assert result is server
    server.add_tool_server.assert_awaited_once_with("http://mcp.test")


@pytest.mark.skip_global_cleanup
class TestMCPToolServerUnit:
    """Test MCP tool description processing and filtering.

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
        assert result is not None
        assert len(result.tools) == 3

        # Filter to specific tools
        result = server.get_tool_description(
            "test_server", allowed_tools=["tool1", "tool3"]
        )
        assert result is not None
        assert len(result.tools) == 2
        assert result.tools[0].name == "tool1"
        assert result.tools[1].name == "tool3"

        # Single tool
        result = server.get_tool_description("test_server", allowed_tools=["tool2"])
        assert result is not None
        assert len(result.tools) == 1
        assert result.tools[0].name == "tool2"

        # No matching tools - returns None
        result = server.get_tool_description(
            "test_server", allowed_tools=["nonexistent"]
        )
        assert result is None

        # Empty list - returns None
        assert server.get_tool_description("test_server", allowed_tools=[]) is None

    def test_post_process_tools_description_uses_trim_schema_result(self, monkeypatch):
        mcp_types = pytest.importorskip("mcp.types")
        tool = mcp_types.Tool(
            name="tool",
            input_schema={"type": "object"},
        )
        result = mcp_types.ListToolsResult(tools=[tool])
        trimmed_schema = {"type": "object", "properties": {}}

        monkeypatch.setattr(
            "vllm.entrypoints.mcp.tool_server.trim_schema",
            lambda _schema: trimmed_schema,
        )

        post_process_tools_description(result)

        assert tool.input_schema is trimmed_schema

    def test_post_process_tools_description_honors_meta_opt_out(self):
        mcp_types = pytest.importorskip("mcp.types")
        included = mcp_types.Tool(
            name="included",
            input_schema={"type": "object"},
        )
        excluded = mcp_types.Tool(
            name="excluded",
            input_schema={"type": "object"},
            meta={"include_in_prompt": False},
        )
        result = mcp_types.ListToolsResult(tools=[included, excluded])

        post_process_tools_description(result)

        assert result.tools == [included]

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
