# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path

logger = init_logger(__name__)


class Parser:
    """
    Abstract Parser class that unifies ReasoningParser and ToolParser into
    a single interface for parsing model output.

    This class provides a unified way to handle both reasoning extraction
    (e.g., chain-of-thought content in <think> tags) and tool call extraction
    (e.g., function calls in XML/JSON format) from model outputs.

    Subclasses can either:
    1. Override the abstract methods directly for custom parsing logic
    2. Set `reasoning_parser` and `tool_parser` properties to delegate to
       existing parser implementations

    Class Attributes:
        reasoning_parser_cls: The ReasoningParser class to use (for compatibility
            with code that needs the class, not instance).
        tool_parser_cls: The ToolParser class to use (for compatibility with
            code that needs the class, not instance).
    """

    # Class-level parser classes for compatibility with existing patterns
    # Subclasses should override these if they use specific parser classes
    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(self, tokenizer: TokenizerLike):
        """
        Initialize the Parser.

        Args:
            tokenizer: The tokenizer used by the model. This is required for
                token-based parsing operations.
        """
        self.model_tokenizer = tokenizer
        self._reasoning_parser: ReasoningParser | None = None
        self._tool_parser: ToolParser | None = None

    @cached_property
    def vocab(self) -> dict[str, int]:
        """Get the vocabulary mapping from tokens to IDs."""
        return self.model_tokenizer.get_vocab()

    @property
    def reasoning_parser(self) -> ReasoningParser | None:
        """The underlying reasoning parser, if any."""
        return self._reasoning_parser

    @reasoning_parser.setter
    def reasoning_parser(self, parser: ReasoningParser | None) -> None:
        self._reasoning_parser = parser

    @property
    def tool_parser(self) -> ToolParser | None:
        """The underlying tool parser, if any."""
        return self._tool_parser

    @tool_parser.setter
    def tool_parser(self, parser: ToolParser | None) -> None:
        self._tool_parser = parser

    # ========== Reasoning Parser Methods ==========

    @abstractmethod
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        Used by structured engines like `xgrammar` to check if the
        reasoning content ends in the model output.

        Args:
            input_ids: The token IDs of the model output.

        Returns:
            True if the reasoning content ends in the input_ids.
        """

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        """
        Check if the reasoning content ends during a decode step.

        Args:
            input_ids: The entire model output token IDs.
            delta_ids: The last few computed tokens at the current decode step.

        Returns:
            True if the reasoning content ends in the delta_ids.
        """
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token IDs from the input_ids.

        This extracts the non-reasoning content (e.g., everything after
        the </think> tag).

        Args:
            input_ids: The token IDs of the model output.

        Returns:
            The extracted content token IDs.
        """

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model
        response available before sending to the client.

        Args:
            model_output: The complete model-generated string.
            request: The request object used to generate the output.

        Returns:
            A tuple of (reasoning_content, response_content).
        """

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a streaming delta message.

        Args:
            previous_text: Text from all previous tokens.
            current_text: Text including the current delta.
            delta_text: The new text in this delta.
            previous_token_ids: Token IDs from previous generation.
            current_token_ids: All token IDs including current.
            delta_token_ids: The new token IDs in this delta.

        Returns:
            A DeltaMessage with reasoning and/or content fields, or None.
        """

    # ========== Tool Parser Methods ==========

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Adjust the request parameters for tool calling.

        Can be overridden by subclasses to modify request parameters
        (e.g., setting structured output schemas for tool calling).

        Args:
            request: The original request.

        Returns:
            The adjusted request.
        """
        return request

    @abstractmethod
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model-generated string.

        Used for non-streaming responses.

        Args:
            model_output: The complete model-generated string.
            request: The request object used to generate the output.

        Returns:
            ExtractedToolCallInformation containing the tool calls.
        """

    @abstractmethod
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """
        Extract tool calls from a streaming delta message.

        Args:
            previous_text: Text from all previous tokens.
            current_text: Text including the current delta.
            delta_text: The new text in this delta.
            previous_token_ids: Token IDs from previous generation.
            current_token_ids: All token IDs including current.
            delta_token_ids: The new token IDs in this delta.
            request: The request object.

        Returns:
            A DeltaMessage with tool_calls field, or None.
        """


class DelegatingParser(Parser):
    """
    A Parser implementation that delegates to separate ReasoningParser and
    ToolParser instances.

    This is the recommended base class for creating model-specific parsers
    that combine existing reasoning and tool parser implementations.
    Subclasses should set `self._reasoning_parser` and `self._tool_parser`
    in their `__init__` method.

    If either parser is None, the corresponding methods will return default
    values (no reasoning extraction, no tool calls).
    """

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        if self._reasoning_parser is None:
            return True  # No reasoning parser means reasoning is always "done"
        return self._reasoning_parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        if self._reasoning_parser is None:
            return True
        return self._reasoning_parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._reasoning_parser is None:
            return input_ids  # All content if no reasoning parser
        return self._reasoning_parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._reasoning_parser is None:
            return None, model_output
        return self._reasoning_parser.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._reasoning_parser is None:
            return DeltaMessage(content=delta_text)
        return self._reasoning_parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        if self._tool_parser is None:
            return request
        return self._tool_parser.adjust_request(request)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self._tool_parser is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        return self._tool_parser.extract_tool_calls(model_output, request)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self._tool_parser is None:
            return None
        return self._tool_parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )


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
    def get_parser(cls, name: str) -> type[Parser]:
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
