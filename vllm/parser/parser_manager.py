# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path

if TYPE_CHECKING:
    from vllm.parser.abstract_parser import Parser
    from vllm.reasoning import ReasoningParser
    from vllm.tool_parsers import ToolParser

logger = init_logger(__name__)


class ParserManager:
    """
    Central registry for Parser implementations.

    Supports two registration modes:
      - Eager registration via `register_module`
      - Lazy registration via `register_lazy_module`
    """

    parsers: dict[str, type[Parser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def get_parser_internal(cls, name: str) -> type[Parser]:
        """
        Retrieve a registered or lazily registered Parser class.

        Args:
            name: The registered name of the parser.

        Returns:
            The Parser class.

        Raises:
            KeyError: If no parser is found under the given name.
        """
        if name in cls.parsers:
            return cls.parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        registered = ", ".join(cls.list_registered())
        raise KeyError(f"Parser '{name}' not found. Available parsers: {registered}")

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[Parser]:
        """Import and register a lazily loaded parser."""
        from vllm.parser.abstract_parser import Parser

        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, Parser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a Parser subclass."
                )
            cls.parsers[name] = parser_cls  # cache
            return parser_cls
        except Exception as e:
            logger.exception(
                "Failed to import lazy parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise

    @classmethod
    def _register_module(
        cls,
        module: type[Parser],
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        """Register a Parser class immediately."""
        from vllm.parser.abstract_parser import Parser

        if not issubclass(module, Parser):
            raise TypeError(
                f"module must be subclass of Parser, but got {type(module)}"
            )

        if module_name is None:
            module_names = [module.__name__]
        elif isinstance(module_name, str):
            module_names = [module_name]
        elif is_list_of(module_name, str):
            module_names = module_name
        else:
            raise TypeError("module_name must be str, list[str], or None.")

        for name in module_names:
            if not force and name in cls.parsers:
                existed = cls.parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.parsers[name] = module

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        Register a lazy module mapping for delayed import.

        Example:
            ParserManager.register_lazy_module(
                name="minimax_m2",
                module_path="vllm.parser.minimax_m2_parser",
                class_name="MiniMaxM2Parser",
            )
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[Parser] | None = None,
    ) -> type[Parser] | Callable[[type[Parser]], type[Parser]]:
        """
        Register a Parser class.

        Can be used as a decorator or called directly.

        Usage:
            @ParserManager.register_module("my_parser")
            class MyParser(Parser):
                ...

        Or:
            ParserManager.register_module(module=MyParser)
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # Immediate registration
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # Decorator usage
        def _decorator(obj: type[Parser]) -> type[Parser]:
            module_path = obj.__module__
            class_name = obj.__name__

            if isinstance(name, str):
                names = [name]
            elif is_list_of(name, str):
                names = name
            else:
                names = [class_name]

            for n in names:
                cls.lazy_parsers[n] = (module_path, class_name)

            return obj

        return _decorator

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all registered parsers."""
        return sorted(set(cls.parsers.keys()) | set(cls.lazy_parsers.keys()))

    @classmethod
    def import_parser(cls, plugin_path: str) -> None:
        """Import a user-defined parser from an arbitrary path."""
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]
        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception(
                "Failed to load module '%s' from %s.", module_name, plugin_path
            )

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
        logger.info('"auto" tool choice has been enabled.')

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
    ) -> type[Parser] | None:
        """
        Get a unified Parser that handles both reasoning and tool parsing.

        This method checks if a unified Parser exists that can handle both
        reasoning extraction and tool call parsing. If no unified parser
        exists, it creates a DelegatingParser that wraps the individual
        reasoning and tool parsers.

        Args:
            tool_parser_name: The name of the tool parser.
            reasoning_parser_name: The name of the reasoning parser.
            enable_auto_tools: Whether auto tool choice is enabled.
            model_name: The model name for parser-specific warnings.

        Returns:
            A Parser class, or None if neither parser is specified.
        """
        from vllm.parser.abstract_parser import _WrappedParser

        if not tool_parser_name and not reasoning_parser_name:
            return None

        # Strategy 1: If both names match, check for a unified parser with that name
        if tool_parser_name and tool_parser_name == reasoning_parser_name:
            try:
                parser = cls.get_parser_internal(tool_parser_name)
                logger.info(
                    "Using unified parser '%s' for both reasoning and tool parsing.",
                    tool_parser_name,
                )
                return parser
            except KeyError:
                pass  # No unified parser with this name

        # Strategy 2: Check for parser with either name
        for name in [tool_parser_name, reasoning_parser_name]:
            if name:
                try:
                    parser = cls.get_parser_internal(name)
                    logger.info(
                        "Using unified parser '%s' for both reasoning and tool parsing.",
                        name,
                    )
                    return parser
                except KeyError:
                    pass

        # Strategy 3: Create a DelegatingParser with the individual parser classes
        reasoning_parser_cls = cls.get_reasoning_parser(reasoning_parser_name)
        tool_parser_cls = cls.get_tool_parser(
            tool_parser_name, enable_auto_tools, model_name
        )

        if reasoning_parser_cls is None and tool_parser_cls is None:
            return None

        # Set the class-level attributes on the imported _WrappedParser
        _WrappedParser.reasoning_parser_cls = reasoning_parser_cls
        _WrappedParser.tool_parser_cls = tool_parser_cls

        return _WrappedParser
