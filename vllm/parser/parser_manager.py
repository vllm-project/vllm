# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.parser.abstract_parser import Parser
    from vllm.reasoning import ReasoningParser
    from vllm.tool_parsers import ToolParser

logger = init_logger(__name__)


class ParserManager:
    """
    Provides a unified Parser by composing individual reasoning and tool
    parsers from their respective registries.
    """

    @classmethod
    def get_tool_parser(
        cls,
        tool_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
    ) -> type[ToolParser] | None:
        """Get the tool parser based on the name."""
        from vllm.tool_parsers import ToolParserManager

        parser: type[ToolParser] | None = None
        if not enable_auto_tools or tool_parser_name is None:
            return parser
        logger.info_once('"auto" tool choice has been enabled.')

        try:
            if (
                tool_parser_name == "pythonic"
                and model_name
                and model_name.startswith("meta-llama/Llama-3.2")
            ):
                logger.warning(
                    "Llama3.2 models may struggle to emit valid pythonic tool calls"
                )
            parser = ToolParserManager.get_tool_parser(tool_parser_name)
        except Exception as e:
            raise TypeError(
                "Error: --enable-auto-tool-choice requires "
                f"tool_parser:'{tool_parser_name}' which has not "
                "been registered"
            ) from e
        return parser

    @classmethod
    def get_reasoning_parser(
        cls,
        reasoning_parser_name: str | None,
    ) -> type[ReasoningParser] | None:
        """Get the reasoning parser based on the name."""
        from vllm.reasoning import ReasoningParserManager

        parser: type[ReasoningParser] | None = None
        if not reasoning_parser_name:
            return None
        try:
            parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)
            assert parser is not None
        except Exception as e:
            raise TypeError(f"{reasoning_parser_name=} has not been registered") from e
        return parser

    @classmethod
    def get_parser(
        cls,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
        is_harmony: bool = False,
    ) -> type[Parser] | None:
        """
        Get a Parser that handles both reasoning and tool parsing.

        Composes individual reasoning and tool parsers into a single
        DelegatingParser subclass.

        Args:
            tool_parser_name: The name of the tool parser.
            reasoning_parser_name: The name of the reasoning parser.
            enable_auto_tools: Whether auto tool choice is enabled.
            model_name: The model name for parser-specific warnings.
            is_harmony: Whether the selected model uses the Harmony format.
                        If True, HarmonyParser is always returned.

        Returns:
            A Parser class, or None if neither parser is specified.
        """
        if not tool_parser_name and not reasoning_parser_name:
            return None

        reasoning_parser_cls = cls.get_reasoning_parser(reasoning_parser_name)
        tool_parser_cls = cls.get_tool_parser(
            tool_parser_name, enable_auto_tools, model_name
        )

        if reasoning_parser_cls is None and tool_parser_cls is None:
            return None

        from vllm.utils.mistral import is_mistral_tool_parser

        if is_harmony:
            from vllm.parser.harmony import HarmonyParser

            HarmonyParser.reasoning_parser_cls = reasoning_parser_cls
            HarmonyParser.tool_parser_cls = tool_parser_cls
            return HarmonyParser

        if is_mistral_tool_parser(tool_parser_cls):
            from vllm.parser.mistral import MistralParser

            MistralParser.reasoning_parser_cls = reasoning_parser_cls
            MistralParser.tool_parser_cls = tool_parser_cls
            return MistralParser

        from vllm.parser.abstract_parser import DelegatingParser

        r_cls = reasoning_parser_cls
        t_cls = tool_parser_cls

        class _Parser(DelegatingParser):
            reasoning_parser_cls = r_cls
            tool_parser_cls = t_cls

        return _Parser
