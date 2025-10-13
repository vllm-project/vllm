# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai_harmony import ToolDescription, ToolNamespaceConfig

from vllm.entrypoints.tool import HarmonyBrowserTool, HarmonyPythonTool, Tool
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from mcp.types import ListToolsResult


async def list_server_and_tools(server_url: str):
    async with (
        sse_client(url=server_url) as streams,
        ClientSession(*streams) as session,
    ):
        initialize_response = await session.initialize()
        list_tools_response = await session.list_tools()
        return initialize_response, list_tools_response


# TODO: This is a harmony specific change, migrate to harmony_utils
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
    def has_namespace(self, namespace: str) -> bool:
        """
        Return True if the namespace is supported, False otherwise.
        """
        pass

    @abstractmethod
    def get_tool_description(self, namespace: str) -> ToolNamespaceConfig | None:
        """
        Return the tool description for the given namespace.
        If the namespace is not supported, return None.
        """
        pass

    @abstractmethod
    def new_session(
        self, namespace: str, session_id: str, headers: dict[str, str] | None = None
    ) -> AbstractAsyncContextManager[Any]:
        """
        Create a session for the namespace.
        """
        ...


class MCPToolServer(ToolServer):
    def __init__(self):
        self.harmony_tool_descriptions = {}

    async def add_mcp_server(self, server_url: str):
        """
        Add an MCP server.

        Args:
            server_url: URL to connect to
        """
        tool_urls = server_url.split(",")
        self.harmony_tool_descriptions = {}
        self.urls: dict[str, str] = {}
        for url in tool_urls:
            url = f"http://{url}/sse"
            initialize_response, list_tools_response = await list_server_and_tools(url)

            server_name = initialize_response.serverInfo.name

            list_tools_response = post_process_tools_description(list_tools_response)

            tool_from_mcp = ToolNamespaceConfig(
                name=server_name,  # This is the namespace (== server_label)
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

            # Check for namespace collision (keep existing logic)
            if tool_from_mcp.name in self.urls:
                logger.warning(
                    "MCP server at %s provides namespace '%s' which is already "
                    "registered from %s. Ignoring duplicate registration.",
                    url,
                    tool_from_mcp.name,
                    self.urls[tool_from_mcp.name],
                )
                continue

            # Add to registry
            self.harmony_tool_descriptions[tool_from_mcp.name] = tool_from_mcp
            self.urls[tool_from_mcp.name] = url

        logger.info(
            "MCPToolServer initialized with tools: %s",
            list(self.harmony_tool_descriptions.keys()),
        )

    def has_namespace(self, namespace: str):
        return namespace in self.harmony_tool_descriptions

    def get_tool_description(self, namespace: str):
        return self.harmony_tool_descriptions.get(namespace)

    @asynccontextmanager
    async def new_session(
        self, namespace: str, session_id: str, headers: dict[str, str] | None = None
    ):
        url = self.urls.get(namespace)
        request_headers = {"x-session-id": session_id}
        if headers is not None:
            request_headers.update(headers)
        if not url:
            raise KeyError(f"Namespace '{namespace}' is not supported")
        async with (
            sse_client(url=url, headers=request_headers) as streams,
            ClientSession(*streams) as session,
        ):
            await session.initialize()
            yield session


# TODO: Move this as it is harmony specific, as the tools return harmony messages
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
            self.tools["code_interpreter"] = python_tool  # Use namespace, not "python"
        logger.info(
            "DemoToolServer initialized with tools: %s", list(self.tools.keys())
        )

    def has_namespace(self, namespace: str) -> bool:
        return namespace in self.tools

    def get_tool_description(self, namespace: str) -> ToolNamespaceConfig | None:
        if namespace not in self.tools:
            return None
        if namespace == "browser":
            return ToolNamespaceConfig.browser()
        elif namespace == "code_interpreter":
            return ToolNamespaceConfig.python()
        else:
            raise ValueError(f"Unknown namespace {namespace}")

    @asynccontextmanager
    async def new_session(
        self, namespace: str, session_id: str, headers: dict[str, str] | None = None
    ):
        if namespace not in self.tools:
            raise KeyError(f"Namespace '{namespace}' is not supported")
        yield self.tools[namespace]
