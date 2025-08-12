# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from openai_harmony import Message

from vllm.logger import init_logger

if TYPE_CHECKING:
    # Avoid circular import.
    from vllm.entrypoints.context import ConversationContext

logger = init_logger(__name__)


class Tool(ABC):

    @abstractmethod
    async def get_result(self, context: "ConversationContext") -> Any:
        pass


class HarmonyBrowserTool(Tool):

    def __init__(self):
        self.enabled = True
        exa_api_key = os.getenv("EXA_API_KEY")
        if not exa_api_key:
            self.enabled = False
            logger.warning_once("EXA_API_KEY is not set, browsing is disabled")
            return

        try:
            from gpt_oss.tools.simple_browser import SimpleBrowserTool
            from gpt_oss.tools.simple_browser.backend import ExaBackend
        except ImportError:
            self.enabled = False
            logger.warning_once(
                "gpt_oss is not installed, browsing is disabled")
            return

        browser_backend = ExaBackend(source="web", api_key=exa_api_key)
        self.browser_tool = SimpleBrowserTool(backend=browser_backend)
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
        return self.browser_tool.tool_config


class HarmonyPythonTool(Tool):

    def __init__(self):
        self.enabled = True

        try:
            from gpt_oss.tools.python_docker.docker_tool import PythonTool
        except ImportError:
            self.enabled = False
            logger.warning_once(
                "gpt_oss is not installed, code interpreter is disabled")
            return

        # NOTE (Chen): as of gpt-oss 0.0.2, there is a bug in _make_response
        # and we do the following monkey patch to fix it.
        class PatchedGptOssPythonTool(PythonTool):

            def _make_response(self,
                               output: str,
                               channel: Optional[str] = None) -> Message:
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
        return self.python_tool.tool_config
