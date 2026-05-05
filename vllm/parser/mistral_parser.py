# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

logger = init_logger(__name__)


class MistralParser(DelegatingParser):
    """
    Unified parser for Mistral models that handles both reasoning
    extraction and tool call parsing.

    This parser delegates to the existing implementations:
    - Any reasoning parser for reasoning extraction
    - MistralToolParser for tool call parsing

    The tool parser is always instantiated (even without tools) so the
    Mistral grammar streaming path has the parser instance it requires.
    """

    # Any reasoning parser is supported.
    tool_parser_cls = MistralToolParser

    @classmethod
    def specialize(
        cls,
        reasoning_parser_cls: type[ReasoningParser] | None,
        tool_parser_cls: type[ToolParser] | None,
        **kwargs: Any,
    ) -> tuple[bool, dict[str, Any]]:
        # Requires MistralToolParser but flexible reasoning parser
        if tool_parser_cls is None or not issubclass(
            tool_parser_cls, MistralToolParser
        ):
            return False, {}
        return True, {
            "reasoning_parser_cls": reasoning_parser_cls,
        }

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(tokenizer, *args, **kwargs)

        cls = type(self)
        if cls.reasoning_parser_cls:
            self._reasoning_parser = cls.reasoning_parser_cls(tokenizer, **kwargs)

        self._tool_parser = cls.tool_parser_cls(tokenizer, tools)

        logger.debug("vLLM Successfully initialized parser %s!", cls)
