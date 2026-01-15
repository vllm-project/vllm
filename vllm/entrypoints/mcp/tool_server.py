# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

from openai_harmony import ToolDescription, ToolNamespaceConfig

from vllm.entrypoints.mcp.tool import HarmonyBrowserTool, HarmonyPythonTool, Tool
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from mcp.types import ListToolsResult


async def list_server_and_tools(server_url: str):
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with (
        sse_client(url=server_url) as streams,
        ClientSession(*streams) as session,
    ):
        initialize_response = await session.initialize()
        list_tools_response = await session.list_tools()
        return initialize_response, list_tools_response


def trim_schema(schema: dict) -> dict:
    # Turn JSON Schema from MCP generated into Harmony's variant.
    if "title" in schema:
        del schema["title"]
    if "default" in schema and schema["default"] is None:
        del schema["default"]
    if "anyOf" in schema:
        # Turn "anyOf": [{"type": "type-1"}, {"type": "type-2"}]
        # into "type": ["type-1", "type-2"]
        # if there's more than 1 types, also remove "null" type as Harmony will
        # just ignore it
        types = [
            type_dict["type"]
            for type_dict in schema["anyOf"]
            if type_dict["type"] != "null"
        ]
        schema["type"] = types
        del schema["anyOf"]
    if "properties" in schema:
        schema["properties"] = {
            k: trim_schema(v) for k, v in schema["properties"].items()
        }
    return schema


def post_process_tools_description(
    list_tools_result: "ListToolsResult",
) -> "ListToolsResult":
    # Adapt the MCP tool result for Harmony
    for tool in list_tools_result.tools:
        tool.inputSchema = trim_schema(tool.inputSchema)

    # Some tools schema don't need to be part of the prompt (e.g. simple text
    # in text out for Python)
    list_tools_result.tools = [
        tool
        for tool in list_tools_result.tools
        if getattr(tool.annotations, "include_in_prompt", True)
    ]

    return list_tools_result


class ToolServer(ABC):
    @abstractmethod
    def has_tool(self, tool_name: str) -> bool:
        """
        Return True if the tool is supported, False otherwise.
        """
        pass

    @abstractmethod
    def get_tool_description(
        self, tool_name: str, allowed_tools: list[str] | None = None
    ) -> ToolNamespaceConfig | None:
        """
        Return the tool description for the given tool name.
        If the tool is not supported, return None.
        """
        pass

    @abstractmethod
    def new_session(
        self, tool_name: str, session_id: str, headers: dict[str, str] | None = None
    ) -> AbstractAsyncContextManager[Any]:
        """
        Create a session for the tool.
        """
        ...


class MCPToolServer(ToolServer):
    def __init__(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            raise ImportError(
                "mcp is not installed. Please run `pip install mcp` to use "
                "MCPToolServer."
            ) from None
        self.harmony_tool_descriptions = {}

    async def add_tool_server(self, server_url: str):
        tool_urls = server_url.split(",")
        self.harmony_tool_descriptions = {}
        self.urls: dict[str, str] = {}
        for url in tool_urls:
            url = f"http://{url}/sse"
            initialize_response, list_tools_response = await list_server_and_tools(url)

            list_tools_response = post_process_tools_description(list_tools_response)

            tool_from_mcp = ToolNamespaceConfig(
                name=initialize_response.serverInfo.name,
                description=initialize_response.instructions,
                tools=[
                    ToolDescription.new(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema,
                    )
                    for tool in list_tools_response.tools
                ],
            )
            self.harmony_tool_descriptions[tool_from_mcp.name] = tool_from_mcp
            if tool_from_mcp.name not in self.urls:
                self.urls[tool_from_mcp.name] = url
            else:
                logger.warning(
                    "Tool %s already exists. Ignoring duplicate tool server %s",
                    tool_from_mcp.name,
                    url,
                )
        logger.info(
            "MCPToolServer initialized with tools: %s",
            list(self.harmony_tool_descriptions.keys()),
        )

    def has_tool(self, tool_name: str):
        return tool_name in self.harmony_tool_descriptions

    def get_tool_description(
        self,
        server_label: str,
        allowed_tools: list[str] | None = None,
    ) -> ToolNamespaceConfig | None:
        cfg = self.harmony_tool_descriptions.get(server_label)
        if cfg is None:
            return None

        # No restrictions: all tools from this MCP server
        if allowed_tools is None:
            return cfg

        filtered = [t for t in cfg.tools if t.name in allowed_tools]

        if not filtered:
            return None

        return ToolNamespaceConfig(
            name=cfg.name,
            description=cfg.description,
            tools=filtered,
        )

    @asynccontextmanager
    async def new_session(
        self, tool_name: str, session_id: str, headers: dict[str, str] | None = None
    ):
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        url = self.urls.get(tool_name)
        request_headers = {"x-session-id": session_id}
        if headers is not None:
            request_headers.update(headers)
        if not url:
            raise KeyError(f"Tool '{tool_name}' is not supported")
        async with (
            sse_client(url=url, headers=request_headers) as streams,
            ClientSession(*streams) as session,
        ):
            await session.initialize()
            yield session


class DemoToolServer(ToolServer):
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    async def init_and_validate(self):
        browser_tool = HarmonyBrowserTool()
        python_tool = HarmonyPythonTool()
        await python_tool.validate()
        if browser_tool.enabled:
            self.tools["browser"] = browser_tool
        if python_tool.enabled:
            self.tools["python"] = python_tool
        logger.info(
            "DemoToolServer initialized with tools: %s", list(self.tools.keys())
        )

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def get_tool_description(
        self, tool_name: str, allowed_tools: list[str] | None = None
    ) -> ToolNamespaceConfig | None:
        if tool_name not in self.tools:
            return None
        if tool_name == "browser":
            return ToolNamespaceConfig.browser()
        elif tool_name == "python":
            return ToolNamespaceConfig.python()
        else:
            raise ValueError(f"Unknown tool {tool_name}")

    @asynccontextmanager
    async def new_session(
        self, tool_name: str, session_id: str, headers: dict[str, str] | None = None
    ):
        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' is not supported")
        yield self.tools[tool_name]
