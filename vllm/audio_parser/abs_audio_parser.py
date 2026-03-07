# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable, Sequence
from functools import cached_property

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path

logger = init_logger(__name__)


class AudioParser:
    """
    Abstract audio parser class that should not be used directly.
    Subclasses implement model-specific audio extraction logic while
    the serving layer only calls these generic interface methods.
    """

    def __init__(self, tokenizer: TokenizerLike):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def is_audio_output(self, prompt_token_ids: Sequence[int]) -> bool:
        """Check whether the model output should be treated as audio.

        Each subclass implements its own detection logic (e.g. checking
        for a special start token at the end of the prompt).  The serving
        layer calls this once per request and passes the result into the
        extraction methods.
        """
        raise NotImplementedError

    def extract_audio_content_nonstreaming(
        self,
        output_token_ids: Sequence[int],
        request: ChatCompletionRequest,
        is_audio_output: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        """Extract audio content from full output token ids.

        Returns:
            (text_token_ids, audio_token_ids, other_token_ids)
        """
        raise NotImplementedError

    def extract_audio_content_streaming(
        self,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        is_audio_output: bool = False,
    ) -> tuple[list[int], list[int], list[int]]:
        """Extract audio content from streaming delta token ids.

        Returns:
            (text_token_ids, audio_token_ids, other_token_ids)
        """
        raise NotImplementedError


class AudioParserManager:
    audio_parsers: dict[str, type] = {}

    @classmethod
    def get_audio_parser(cls, name) -> type:
        """
        Get audio parser by name which is registered by `register_module`.

        Raise a KeyError exception if the name is not registered.
        """
        if name in cls.audio_parsers:
            return cls.audio_parsers[name]

        raise KeyError(f"audio helper: '{name}' not found in audio_parsers")

    @classmethod
    def _register_module(
        cls,
        module: type,
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        if not issubclass(module, AudioParser):
            raise TypeError(
                f"module must be subclass of AudioParser, but got {type(module)}"
            )
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.audio_parsers:
                existed_module = cls.audio_parsers[name]
                raise KeyError(
                    f"{name} is already registered at {existed_module.__module__}"
                )
            cls.audio_parsers[name] = module

    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type | None = None,
    ) -> type | Callable:
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as None) or normal function(with module as not
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_list_of(name, str)):
            raise TypeError(
                "name must be None, an instance of str, or a sequence of str, "
                f"but got {type(name)}"
            )

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
    def import_audio_parser(cls, plugin_path: str) -> None:
        """
        Import a user-defined audio parser by the path
        of the audio parser define file.
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception(
                "Failed to load module '%s' from %s.", module_name, plugin_path
            )
            return
