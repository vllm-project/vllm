# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.entrypoints.context import ConversationContext

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Abstract base class for tools."""

    @abstractmethod
    async def get_result(self, context: "ConversationContext") -> Any:
        """Get the result of the tool execution."""
        pass

    @property
    @abstractmethod
    def tool_config(self) -> Any:
        """Get the tool configuration."""
        pass


class HarmonyBrowserTool(Tool):
    """Browser tool for Harmony integration."""

    def __init__(self):
        self.enabled = True

        try:
            from gpt_oss.tools.browser.browser_tool import BrowserTool
        except ImportError:
            self.enabled = False
            logger.warning_once("gpt_oss is not installed, browser tool is disabled")
            return

        self.browser_tool = BrowserTool()
        logger.info_once("Browser tool initialized")

    async def get_result(self, context: "ConversationContext") -> Any:
        from vllm.entrypoints.context import HarmonyContext

        assert isinstance(context, HarmonyContext)
        last_msg = context.messages[-1]
        tool_output_msgs = []
        async for msg in self.browser_tool.process(last_msg):
            tool_output_msgs.append(msg)
        return tool_output_msgs

    @property
    def tool_config(self) -> Any:
        return self.browser_tool.tool_config if self.enabled else {}


class HarmonyPythonTool(Tool):
    """Python tool for Harmony integration."""

    def __init__(self):
        self.enabled = True

        try:
            from gpt_oss.tools.python_docker.docker_tool import PythonTool
        except ImportError:
            self.enabled = False
            logger.warning_once(
                "gpt_oss is not installed, code interpreter is disabled"
            )
            return

        # NOTE (Chen): as of gpt-oss 0.0.2, there is a bug in _make_response
        # and we do the following monkey patch to fix it.
        class PatchedGptOssPythonTool(PythonTool):

            def _make_response(self, output: str, channel: str = None) -> Any:
                return super()._make_response(output)

        self.python_tool = PatchedGptOssPythonTool()
        logger.info_once("Code interpreter tool initialized")

    async def get_result(self, context: "ConversationContext") -> Any:
        from vllm.entrypoints.context import HarmonyContext

        assert isinstance(context, HarmonyContext)
        last_msg = context.messages[-1]
        tool_output_msgs = []
        async for msg in self.python_tool.process(last_msg):
            tool_output_msgs.append(msg)
        return tool_output_msgs

    @property
    def tool_config(self) -> Any:
        return self.python_tool.tool_config if self.enabled else {}
