# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from openai_harmony import Author, Message, Role, TextContent

from vllm.logger import init_logger

if TYPE_CHECKING:
    # Avoid circular import.
    from vllm.entrypoints.context import ConversationContext

logger = init_logger(__name__)


def validate_gpt_oss_install():
    """
    Check if the gpt-oss is installed and its version is at least 0.0.3.
    If not, raise an ImportError.
    """
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import InvalidVersion, Version

    try:
        pkg_version_str = version("gpt_oss")  # e.g., "0.0.5"
        pkg_version = Version(pkg_version_str)
    except PackageNotFoundError:
        raise ImportError("Package 'gpt_oss' is not installed.") from None
    except InvalidVersion as e:
        raise ImportError(
            f"Invalid version string for 'gpt_oss': {e}") from None

    if pkg_version < Version("0.0.3"):
        raise ImportError(
            f"gpt_oss >= 0.0.3 is required, but {pkg_version} is installed."
        ) from None


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
            validate_gpt_oss_install()
            from gpt_oss.tools.simple_browser import SimpleBrowserTool
            from gpt_oss.tools.simple_browser.backend import ExaBackend
        except ImportError as e:
            self.enabled = False
            logger.warning_once(
                "gpt_oss is not installed properly (%s), browsing is disabled",
                e)
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
            validate_gpt_oss_install()
            from gpt_oss.tools.python_docker.docker_tool import PythonTool
        except ImportError as e:
            self.enabled = False
            logger.warning_once(
                "gpt_oss is not installed properly (%s), code interpreter is "
                "disabled", e)
            return

        self.python_tool = PythonTool()

    async def validate(self):
        if not self.enabled:
            return
        try:
            message = Message(
                author=Author(role=Role.ASSISTANT),
                content=[TextContent(text="print('Hello, world!')")],
                channel="analysis",
                recipient="python",
                content_type="code",
            )
            msgs = []
            async for msg in self.python_tool.process(message):
                msgs.append(msg)
            assert msgs[0].content[0].text == "Hello, world!\n"
        except Exception as e:
            self.enabled = False
            logger.warning_once(
                "Code interpreter tool failed to initialize (%s), code "
                "interpreter is disabled", e)
            return
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
