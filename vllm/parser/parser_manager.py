# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

if TYPE_CHECKING:
    from vllm.parser.abstract_parser import Parser
    from vllm.reasoning import ReasoningParser
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers import ToolParser

logger = init_logger(__name__)

HF_PARSER = "hf"


class ParserManager:
    """Provides a unified Parser by composing reasoning and tool parser adapters."""

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
    def validate_parser_tokenizer(
        cls,
        parser_cls: type[Parser],
        tokenizer: TokenizerLike,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Fail at startup if the tokenizer lacks what the parsers need."""
        try:
            parser_cls(tokenizer)
        except Exception as e:
            flags = []
            if parser_cls.tool_parser_cls is not None:
                flags.append(f"--tool-call-parser {tool_parser_name}")
            if parser_cls.reasoning_parser_cls is not None:
                flags.append(f"--reasoning-parser {reasoning_parser_name}")
            raise TypeError(
                f"{' and '.join(flags)} cannot be used with the tokenizer of "
                f"{model_name!r}: {e}"
            ) from e

    @classmethod
    def get_parser(
        cls,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
        enable_auto_tools: bool = False,
        model_name: str | None = None,
        is_harmony: bool = False,
        tool_strict_level: str = "auto",
        tokenizer: TokenizerLike | None = None,
    ) -> type[Parser] | None:
        """Get a Parser that handles both reasoning and tool parsing.

        Composes the individual parsers into a ``DelegatingParser`` subclass.

        Args:
            tool_parser_name: The name of the tool parser.
            reasoning_parser_name: The name of the reasoning parser.
            enable_auto_tools: Whether auto tool choice is enabled.
            model_name: The model name for parser-specific warnings.
            is_harmony: Whether the selected model uses the Harmony format.
                        If True, HarmonyParser is always returned.
            tool_strict_level: Server-side floor for tool-call structural
                tags (``--tool-strict-level``).
            tokenizer: Tokenizer the composed parser is validated against
                at startup (see :meth:`validate_parser_tokenizer`). ``None``
                skips the check (e.g. ``skip_tokenizer_init``).

        Returns:
            A Parser class, or None if neither parser is specified.

        """
        parser_cls = cls._compose_parser(
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
            enable_auto_tools=enable_auto_tools,
            model_name=model_name,
            is_harmony=is_harmony,
            tool_strict_level=tool_strict_level,
            tokenizer=tokenizer,
        )
        if parser_cls is not None and tokenizer is not None:
            cls.validate_parser_tokenizer(
                parser_cls,
                tokenizer,
                tool_parser_name=tool_parser_name,
                reasoning_parser_name=reasoning_parser_name,
                model_name=model_name,
            )
        return parser_cls

    @classmethod
    def _compose_parser(
        cls,
        tool_parser_name: str | None,
        reasoning_parser_name: str | None,
        enable_auto_tools: bool,
        model_name: str | None,
        is_harmony: bool,
        tool_strict_level: str,
        tokenizer: TokenizerLike | None,
    ) -> type[Parser] | None:
        if not tool_parser_name and not reasoning_parser_name:
            return None

        reasoning_parser_cls = cls.get_reasoning_parser(reasoning_parser_name)
        tool_parser_cls = cls.get_tool_parser(
            tool_parser_name, enable_auto_tools, model_name
        )

        if reasoning_parser_cls is None and tool_parser_cls is None:
            return None

        strict_level = ToolStrictLevel.from_name(tool_strict_level)

        if is_harmony:
            from vllm.parser.harmony import HarmonyParser

            HarmonyParser.reasoning_parser_cls = reasoning_parser_cls
            HarmonyParser.tool_parser_cls = tool_parser_cls
            HarmonyParser.tool_strict_level = strict_level
            return HarmonyParser

        if HF_PARSER in (reasoning_parser_name, tool_parser_name):
            if {reasoning_parser_name, tool_parser_name} - {
                HF_PARSER,
                None,
                "",
            }:
                raise TypeError(
                    "The hf parser cannot be combined with other "
                    "reasoning or tool call parsers"
                )
            from vllm.parser.response_template import (
                ResponseTemplateParser,
                validate_tokenizer_response_template,
            )

            if tokenizer is not None:
                validate_tokenizer_response_template(
                    tokenizer,
                    reasoning=reasoning_parser_cls is not None,
                    tools=tool_parser_cls is not None,
                )

            r_cls = reasoning_parser_cls
            t_cls = tool_parser_cls
            auto_tools = enable_auto_tools

            class _ResponseTemplateParser(ResponseTemplateParser):
                reasoning_parser_cls = r_cls
                tool_parser_cls = t_cls
                tool_strict_level = strict_level
                _enable_auto_tools = auto_tools

            return _ResponseTemplateParser

        if reasoning_parser_name == "kimi_k3" or tool_parser_name == "kimi_k3":
            from vllm.parser.kimi_k3 import KimiK3Parser

            r_cls = reasoning_parser_cls
            t_cls = tool_parser_cls

            class _KimiK3Parser(KimiK3Parser):
                reasoning_parser_cls = r_cls
                tool_parser_cls = t_cls
                tool_strict_level = strict_level

            return _KimiK3Parser

        if {reasoning_parser_name, tool_parser_name} & {
            "cohere_command3",
            "cohere_command4",
        }:
            from vllm.parser.cohere_command import CohereCommandParser

            r_cls = reasoning_parser_cls
            t_cls = tool_parser_cls

            class _CohereCommandParser(CohereCommandParser):
                reasoning_parser_cls = r_cls
                tool_parser_cls = t_cls
                tool_strict_level = strict_level

            return _CohereCommandParser

        if "minicpmv" in {reasoning_parser_name, tool_parser_name}:
            from vllm.parser.minicpmv import MiniCPMVUnifiedParser

            r_cls = reasoning_parser_cls
            t_cls = tool_parser_cls

            class _MiniCPMVParser(MiniCPMVUnifiedParser):
                reasoning_parser_cls = r_cls
                tool_parser_cls = t_cls
                tool_strict_level = strict_level

            return _MiniCPMVParser

        from vllm.parser.abstract_parser import DelegatingParser

        r_cls = reasoning_parser_cls
        t_cls = tool_parser_cls

        class _Parser(DelegatingParser):
            reasoning_parser_cls = r_cls
            tool_parser_cls = t_cls
            tool_strict_level = strict_level

        return _Parser
