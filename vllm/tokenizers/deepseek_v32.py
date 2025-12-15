# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import Any

from transformers import BatchEncoding

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .deepseek_v32_encoding import encode_messages
from .hf import CachedHfTokenizer
from .protocol import TokenizerLike


class DeepseekV32Tokenizer(CachedHfTokenizer):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TokenizerLike":
        tokenizer = super().from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )
        return DeepseekV32Tokenizer(tokenizer)

    def __init__(self, tokenizer: TokenizerLike) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.name_or_path = getattr(tokenizer, "name_or_path", "")

        self._added_vocab = self.tokenizer.get_added_vocab()
        self._added_vocab_size = len(self._added_vocab)

    def apply_chat_template(
        self,
        messages: list["ChatCompletionMessageParam"],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str | list[int]:
        thinking = kwargs.get("thinking", False)
        thinking_mode = "thinking"
        if not thinking:
            thinking_mode = "chat"
        conversation = kwargs.get("conversation", messages)
        messages = conversation.copy()
        if tools is not None and len(tools) > 0:
            messages.insert(0, {"role": "system"})
            messages[0]["tools"] = tools  # type: ignore[typeddict-unknown-key]

        # Historical reasoning content is dropped when a new user message is introduced
        drop_thinking = messages[-1]["role"] == "user"

        encode_config = dict(thinking_mode=thinking_mode, drop_thinking=drop_thinking)
        prompt_str = encode_messages(messages, **encode_config)  # type: ignore

        if kwargs.get("tokenize", True):
            tokenizer_kwargs = {
                k: kwargs[k] for k in ("truncation", "max_length") if k in kwargs
            }
            return self.encode(
                prompt_str,
                add_special_tokens=False,
                **tokenizer_kwargs,
            )

        return prompt_str

    def num_special_tokens_to_add(self) -> int:
        return len(self.encode(""))

    @property
    def all_special_tokens(self) -> list[str]:
        return self.tokenizer.all_special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self.tokenizer.all_special_ids

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def is_fast(self) -> bool:
        return self.tokenizer.is_fast

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def max_token_id(self) -> int:
        return self.tokenizer.max_token_id

    @property
    def truncation_side(self) -> str:
        return self.tokenizer.truncation_side

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        # </think> is an added token in DeepseekV32 tokenizer
        return self.vocab_size + self._added_vocab_size

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> "BatchEncoding":
        return self.tokenizer(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def get_added_vocab(self) -> dict[str, int]:
        return self._added_vocab.copy()

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return self.tokenizer.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(
            ids, skip_special_tokens=skip_special_tokens
        )
