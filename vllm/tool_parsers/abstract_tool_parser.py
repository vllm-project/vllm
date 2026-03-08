# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
from collections.abc import Callable, Sequence
from functools import cached_property

from openai.types.responses.response_format_text_json_schema_config import (
    ResponseFormatTextJSONSchemaConfig,
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponseTextConfig,
)
from vllm.logger import init_logger
from vllm.sampling_params import (
    StructuredOutputsParams,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import get_json_schema_from_tools
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path

logger = init_logger(__name__)


class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    def __init__(self, tokenizer: TokenizerLike):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """
        Static method that used to adjust the request parameters.
        """
        if not request.tools:
            return request
        json_schema_from_tool = get_json_schema_from_tools(
            tool_choice=request.tool_choice, tools=request.tools
        )
        # Set structured output params for tool calling
        if json_schema_from_tool is not None:
            if isinstance(request, ChatCompletionRequest):
                # tool_choice: "Forced Function" or "required" will override
                # structured output json settings to make tool calling work correctly
                request.structured_outputs = StructuredOutputsParams(
                    json=json_schema_from_tool
                )
                request.response_format = None
            if isinstance(request, ResponsesRequest):
                request.text = ResponseTextConfig()
                request.text.format = ResponseFormatTextJSONSchemaConfig(
                    name="tool_calling_response",
                    schema=json_schema_from_tool,
                    type="json_schema",
                    description="Response format for tool calling",
                    strict=True,
                )

        return request

    def extract_tool_calls(
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Static method that should be implemented for extracting tool calls from
        a complete model-generated string.
        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        Static because it's stateless.
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls has not been implemented!"
        )

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
        Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls_streaming has not been implemented!"
        )


class ToolParserManager:
    """
    Central registry for ToolParser implementations.

    Supports two modes:
      - Eager (immediate) registration via `register_module`
      - Lazy registration via `register_lazy_module`
    """

    tool_parsers: dict[str, type[ToolParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def get_tool_parser(cls, name: str) -> type[ToolParser]:
        """
        Retrieve a registered or lazily registered ToolParser class.

        If the parser is lazily registered,
        it will be imported and cached on first access.
        Raises KeyError if not found.
        """
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        raise KeyError(f"Tool parser '{name}' not found.")

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ToolParser]:
        """Import and register a lazily loaded parser."""
        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ToolParser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a ToolParser subclass."
                )
            cls.tool_parsers[name] = parser_cls  # cache
            return parser_cls
        except Exception as e:
            logger.exception(
                "Failed to import lazy tool parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise

    @classmethod
    def _register_module(
        cls,
        module: type[ToolParser],
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        """Register a ToolParser class immediately."""
        if not issubclass(module, ToolParser):
            raise TypeError(
                f"module must be subclass of ToolParser, but got {type(module)}"
            )

        if module_name is None:
            module_name = module.__name__

        if isinstance(module_name, str):
            module_names = [module_name]
        elif is_list_of(module_name, str):
            module_names = module_name
        else:
            raise TypeError("module_name must be str, list[str], or None.")

        for name in module_names:
            if not force and name in cls.tool_parsers:
                existed = cls.tool_parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.tool_parsers[name] = module

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        Register a lazy module mapping.

        Example:
            ToolParserManager.register_lazy_module(
                name="kimi_k2",
                module_path="vllm.tool_parsers.kimi_k2_parser",
                class_name="KimiK2ToolParser",
            )
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[ToolParser] | None = None,
    ) -> type[ToolParser] | Callable[[type[ToolParser]], type[ToolParser]]:
        """
        Register module immediately or lazily (as a decorator).

        Usage:
            @ToolParserManager.register_module("kimi_k2")
            class KimiK2ToolParser(ToolParser):
                ...

        Or:
            ToolParserManager.register_module(module=SomeToolParser)
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # Immediate registration
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # Decorator usage
        def _decorator(obj: type[ToolParser]) -> type[ToolParser]:
            module_path = obj.__module__
            class_name = obj.__name__

            if isinstance(name, str):
                names = [name]
            elif name is not None and is_list_of(name, str):
                names = name
            else:
                names = [class_name]

            for n in names:
                # Lazy mapping only: do not import now
                cls.lazy_parsers[n] = (module_path, class_name)

            return obj

        return _decorator

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all eagerly and lazily registered tool parsers."""
        return sorted(set(cls.tool_parsers.keys()) | set(cls.lazy_parsers.keys()))

    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None:
        """Import a user-defined parser file from arbitrary path."""

        module_name = os.path.splitext(os.path.basename(plugin_path))[0]
        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception(
                "Failed to load module '%s' from %s.", module_name, plugin_path
            )
