# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from vllm.logger import init_logger
from vllm.utils import import_from_path, is_list_of

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                                  DeltaMessage,
                                                  ResponsesRequest)
    from vllm.transformers_utils.tokenizer import AnyTokenizer
else:
    ChatCompletionRequest = Any
    DeltaMessage = Any
    ResponsesRequest = Any
    AnyTokenizer = Any

logger = init_logger(__name__)


class ReasoningParser:
    """
    Abstract reasoning parser class that should not be used directly.
    Provided and methods should be used in derived classes.

    It is used to extract reasoning content from the model output.
    """

    def __init__(self, tokenizer: AnyTokenizer):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        It is used in structured engines like `xgrammar` to check if the
        reasoning content ends in the model output.

        Parameters:
        input_ids: list[int]
            The input_ids of the model output.

        Returns:
        bool
            True if the reasoning content ends in the input_ids.
        """

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        Parameters:
        input_ids: list[int]
            The input_ids of the model output.
        Returns:
        list[int]
            The extracted content from the input_ids.
        """

    @abstractmethod
    def extract_reasoning_content(
        self,
        model_output: str,
        request: Union[ChatCompletionRequest, ResponsesRequest],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract reasoning content from.

        request: ChatCompletionRequest
            The request object that was used to generate the model_output.

        Returns:
        tuple[Optional[str], Optional[str]]
            A tuple containing the reasoning content and the content.
        """

    @abstractmethod
    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """


class ReasoningParserManager:
    reasoning_parsers: dict[str, type] = {}

    @classmethod
    def get_reasoning_parser(cls, name: str | None) -> type[ReasoningParser]:
        """
        Get reasoning parser by name which is registered by `register_module`.

        Raise a KeyError exception if the name is not registered.
        """
        if name in cls.reasoning_parsers:
            return cls.reasoning_parsers[name]

        raise KeyError(
            f"reasoning helper: '{name}' not found in reasoning_parsers")

    @classmethod
    def _register_module(
        cls,
        module: type,
        module_name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
    ) -> None:
        if not issubclass(module, ReasoningParser):
            raise TypeError("module must be subclass of ReasoningParser, "
                            f"but got {type(module)}")
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.reasoning_parsers:
                existed_module = cls.reasoning_parsers[name]
                raise KeyError(f"{name} is already registered "
                               f"at {existed_module.__module__}")
            cls.reasoning_parsers[name] = module

    @classmethod
    def register_module(
        cls,
        name: Optional[Union[str, list[str]]] = None,
        force: bool = True,
        module: Union[type, None] = None,
    ) -> Union[type, Callable]:
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as None) or normal function(with module as not
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)
                or is_list_of(name, str)):
            raise TypeError(
                "name must be None, an instance of str, or a sequence of str, "
                f"but got {type(name)}")

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_reasoning_parser(cls, plugin_path: str) -> None:
        """
        Import a user-defined reasoning parser by the path
        of the reasoning parser define file.
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception("Failed to load module '%s' from %s.",
                             module_name, plugin_path)
            return
