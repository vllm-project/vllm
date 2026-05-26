# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai_harmony import Author, Message, Role, TextContent

from vllm.logger import init_logger
from vllm.utils import random_uuid

if TYPE_CHECKING:
    # Avoid circular import.
    from vllm.entrypoints.openai.responses.context import ConversationContext

logger = init_logger(__name__)

MIN_GPT_OSS_VERSION = "0.0.7"


def validate_gpt_oss_install():
    """
    Check if the gpt-oss is installed and its version is at least 0.0.7.
    If not, raise an ImportError.
    """
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import InvalidVersion, Version

    try:
        pkg_version_str = version("gpt_oss")
        pkg_version = Version(pkg_version_str)
    except PackageNotFoundError:
        raise ImportError("Package 'gpt_oss' is not installed.") from None
    except InvalidVersion as e:
        raise ImportError(f"Invalid version string for 'gpt_oss': {e}") from None

    if pkg_version < Version(MIN_GPT_OSS_VERSION):
        raise ImportError(
            f"gpt_oss >= {MIN_GPT_OSS_VERSION} is required, "
            f"but {pkg_version} is installed."
        ) from None


class Tool(ABC):
    @abstractmethod
    async def get_result(self, context: "ConversationContext") -> Any:
        pass

    @abstractmethod
    async def get_result_parsable_context(self, context: "ConversationContext") -> Any:
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
                "gpt_oss is not installed properly (%s), browsing is disabled", e
            )
            return

        browser_backend = ExaBackend(source="web", api_key=exa_api_key)
        self.browser_tool = SimpleBrowserTool(backend=browser_backend)
        logger.info_once("Browser tool initialized")

    async def get_result(self, context: "ConversationContext") -> Any:
        from vllm.entrypoints.openai.responses.context import HarmonyContext

        assert isinstance(context, HarmonyContext)
        last_msg = context.messages[-1]
        tool_output_msgs = []
        async for msg in self.browser_tool.process(last_msg):
            tool_output_msgs.append(msg)
        return tool_output_msgs

    async def get_result_parsable_context(self, context: "ConversationContext") -> Any:
        raise NotImplementedError("Not implemented yet")

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
                "gpt_oss is not installed properly (%s), code interpreter is disabled",
                e,
            )
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
                "interpreter is disabled",
                e,
            )
            return
        logger.info_once("Code interpreter tool initialized")

    async def get_result(self, context: "ConversationContext") -> Any:
        from vllm.entrypoints.openai.responses.context import HarmonyContext

        assert isinstance(context, HarmonyContext)
        last_msg = context.messages[-1]
        tool_output_msgs = []
        async for msg in self.python_tool.process(last_msg):
            tool_output_msgs.append(msg)
        return tool_output_msgs

    async def get_result_parsable_context(self, context: "ConversationContext") -> Any:
        """
        This function converts parsable context types to harmony and
        back so we can use GPTOSS demo python tool
        """
        from vllm.entrypoints.openai.responses.context import ParsableContext

        assert isinstance(context, ParsableContext)

        last_msg = context.parser.response_messages[-1]
        args = json.loads(last_msg.arguments)

        last_msg_harmony = Message(
            author=Author(role="assistant", name=None),
            content=[TextContent(text=args["code"])],
            channel="analysis",
            recipient="python",
            content_type="code",
        )

        tool_output_msgs = []
        async for msg in self.python_tool.process(last_msg_harmony):
            processed = ResponseFunctionToolCallOutputItem(
                id=f"fco_{random_uuid()}",
                type="function_call_output",
                call_id=f"call_{random_uuid()}",
                output=msg.content[0].text,
                status="completed",
            )
            tool_output_msgs.append(processed)
        return tool_output_msgs

    @property
    def tool_config(self) -> Any:
        return self.python_tool.tool_config
