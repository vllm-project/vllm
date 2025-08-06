# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from openai_harmony import ToolNamespaceConfig

from vllm.entrypoints.tool import HarmonyBrowserTool, HarmonyPythonTool, Tool
from vllm.logger import init_logger

logger = init_logger(__name__)


class ToolServer(ABC):

    @abstractmethod
    def has_tool(self, tool_name: str) -> bool:
        pass

    @abstractmethod
    def get_tool_description(self, tool_name: str) -> ToolNamespaceConfig:
        pass

    @abstractmethod
    def get_tool_session(self,
                         tool_name: str) -> AbstractAsyncContextManager[Any]:
        ...


class DemoToolServer(ToolServer):

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        browser_tool = HarmonyBrowserTool()
        if browser_tool.enabled:
            self.tools["browser"] = browser_tool
        python_tool = HarmonyPythonTool()
        if python_tool.enabled:
            self.tools["python"] = python_tool
        logger.info("DemoToolServer initialized with tools: %s",
                    list(self.tools.keys()))

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def get_tool_description(self, tool_name: str) -> ToolNamespaceConfig:
        if tool_name not in self.tools:
            return None
        if tool_name == "browser":
            return ToolNamespaceConfig.browser()
        elif tool_name == "python":
            return ToolNamespaceConfig.python()
        else:
            raise ValueError(f"Unknown tool {tool_name}")

    @asynccontextmanager
    async def get_tool_session(self, tool_name: str):
        yield self.tools[tool_name]
