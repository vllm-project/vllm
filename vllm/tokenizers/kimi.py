# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer for Kimi-Audio models using TikTokenTokenizer."""

import inspect
from pathlib import Path
from typing import Any, overload

from transformers import AutoTokenizer, BatchEncoding

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.logger import init_logger

from .protocol import TokenizerLike

logger = init_logger(__name__)


class KimiTokenizer(TokenizerLike):
    """Tokenizer wrapper for Kimi models using TikTokenTokenizer backend."""

    def __init__(self, hf_tokenizer: Any):
        """Initialize with a HuggingFace TikTokenTokenizer."""
        self._tokenizer = hf_tokenizer
        self._name_or_path = getattr(hf_tokenizer, "name_or_path", None)
        # Cache vocab for performance
        self._vocab = self.get_vocab()
        self._vocab_size = len(self._vocab)
        self._max_token_id = max(self._vocab.values()) if self._vocab else 0
        self._max_chars_per_token = max(
            (len(tok) for tok in self._vocab),
            default=0,
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "KimiTokenizer":
        """Load tokenizer from HuggingFace hub or local path."""
        hf_tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=download_dir,
            **kwargs,
        )
        return cls(hf_tokenizer)

    def num_special_tokens_to_add(self) -> int:
        return 0  # Kimi handles special tokens internally

    @property
    def all_special_tokens(self) -> list[str]:
        """Return all special tokens."""
        if hasattr(self._tokenizer, "special_tokens"):
            return list(self._tokenizer.special_tokens.keys())
        return []

    @property
    def all_special_ids(self) -> list[int]:
        """Return all special token IDs."""
        if hasattr(self._tokenizer, "special_tokens"):
            return list(self._tokenizer.special_tokens.values())
        return []

    @property
    def name_or_path(self) -> str | None:
        return self._name_or_path

    @property
    def bos_token_id(self) -> int:
        return getattr(self._tokenizer, "bos_id", 0)

    @property
    def eos_token_id(self) -> int:
        return getattr(self._tokenizer, "eos_id", 0)

    @property
    def pad_token_id(self) -> int:
        return getattr(self._tokenizer, "pad_id", 0)

    @property
    def is_fast(self) -> bool:
        return False  # TikTokenTokenizer is not a fast tokenizer

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    @property
    def max_chars_per_token(self) -> int:
        return self._max_chars_per_token

    @property
    def truncation_side(self) -> str:
        return getattr(self._tokenizer, "truncation_side", "right")

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        return self._vocab_size

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> BatchEncoding:
        """Tokenize text."""
        if text_pair is not None:
            return self._tokenizer(
                text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )

        input_ids: list[int] | list[list[int]]
        attention_mask: list[int] | list[list[int]]

        if isinstance(text, list):
            input_ids = [
                self.encode(
                    item,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )
                for item in text
            ]
            attention_mask = [[1] * len(ids) for ids in input_ids]
        else:
            input_ids = self.encode(
                text,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            )
            attention_mask = [1] * len(input_ids)

        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    def get_vocab(self) -> dict[str, int]:
        """Return vocabulary as token -> id mapping."""
        if hasattr(self._tokenizer, "vocab"):
            return self._tokenizer.vocab
        # Fallback: invert inv_vocab if that's all we have
        if hasattr(self._tokenizer, "inv_vocab"):
            return {v: k for k, v in self._tokenizer.inv_vocab.items()}
        return {}

    def get_added_vocab(self) -> dict[str, int]:
        """Return added vocabulary."""
        if hasattr(self._tokenizer, "special_tokens"):
            return dict(self._tokenizer.special_tokens)
        return {}

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode text to token IDs."""
        encode_kwargs: dict[str, Any] = {
            "truncation": truncation,
            "max_length": max_length,
            "add_special_tokens": add_special_tokens,
        }
        if "<|" in text and "|>" in text:
            encode_kwargs["allowed_special"] = "all"
            encode_kwargs["disallowed_special"] = ()

        supported_params = inspect.signature(self._tokenizer.encode).parameters
        filtered_kwargs = {
            key: value
            for key, value in encode_kwargs.items()
            if key in supported_params
        }
        if "bos" in supported_params:
            filtered_kwargs["bos"] = False
        if "eos" in supported_params:
            filtered_kwargs["eos"] = False

        try:
            return self._tokenizer.encode(text, **filtered_kwargs)
        except TypeError:
            fallback_kwargs = {
                key: value
                for key, value in filtered_kwargs.items()
                if key in {"allowed_special", "disallowed_special", "bos", "eos"}
            }
            return self._tokenizer.encode(text, **fallback_kwargs)

    def apply_chat_template(
        self,
        messages: list[ChatCompletionMessageParam] | None = None,
        tools: list[dict[str, Any]] | None = None,
        conversation: list[ChatCompletionMessageParam] | None = None,
        **kwargs,
    ) -> str | list[int]:
        """Apply chat template to messages."""
        if messages is None:
            messages = conversation
        if messages is None:
            raise ValueError("messages must be provided for chat templating")
        return self._tokenizer.apply_chat_template(messages, tools=tools, **kwargs)

    def get_chat_template(
        self,
        chat_template: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        if hasattr(self._tokenizer, "get_chat_template"):
            return self._tokenizer.get_chat_template(chat_template, tools=tools)
        if chat_template is not None:
            return chat_template
        return getattr(self._tokenizer, "chat_template", None)

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Convert tokens to string."""
        return self._tokenizer.convert_tokens_to_string(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text."""
        if isinstance(ids, int):
            ids = [ids]
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Convert IDs to tokens."""
        return self._tokenizer.convert_ids_to_tokens(
            ids, skip_special_tokens=skip_special_tokens
        )
