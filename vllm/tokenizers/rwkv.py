# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright BlinkDL and ChatRWKV contributors
#
# Adapted from BlinkDL/ChatRWKV tokenizer/rwkv_tokenizer.py.

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

from transformers import BatchEncoding
from transformers.utils import chat_template_utils as hf_chat_utils

from .protocol import TokenizerLike
from .rwkv_defaults import (
    RWKV_NATIVE_CHAT_TEMPLATE as _RWKV_NATIVE_CHAT_TEMPLATE,
)
from .rwkv_defaults import (
    render_rwkv_chat_template,
)

_VOCAB_FILE = Path(__file__).parent / "assets" / "rwkv_vocab_v20230424.txt"
_RWKV7_VOCAB_SIZE = 65536
_UNKNOWN_TOKEN_BYTES = "\ufffd".encode()


class _Trie:
    __slots__ = ("to", "token")

    def __init__(self) -> None:
        self.to: list[_Trie | None] = [None for _ in range(256)]
        self.token = 0

    def add(self, key: bytes, val: int) -> None:
        node = self
        for ch in key:
            child = node.to[ch]
            if child is None:
                child = _Trie()
                node.to[ch] = child
            node = child
        node.token = val + 1


class RWKVTokenizer(TokenizerLike):
    """RWKV World tokenizer using the ChatRWKV trie implementation."""

    def __init__(
        self,
        vocab_file: str | Path = _VOCAB_FILE,
        name_or_path: str | Path | None = None,
    ) -> None:
        idx2token: dict[int, bytes] = {}
        lines = Path(vocab_file).read_text(encoding="utf-8").splitlines()
        for line in lines:
            idx = int(line[: line.index(" ")])
            token = ast.literal_eval(line[line.index(" ") : line.rindex(" ")])
            token = token.encode("utf-8") if isinstance(token, str) else token
            assert isinstance(token, bytes)
            assert len(token) == int(line[line.rindex(" ") :])
            idx2token[idx] = token

        self.token2idx = {token: idx for idx, token in idx2token.items()}
        self.idx2token = [
            _UNKNOWN_TOKEN_BYTES
            for _ in range(max(max(idx2token) + 1, _RWKV7_VOCAB_SIZE))
        ]
        self.idx2token[0] = b""
        for idx, token in idx2token.items():
            self.idx2token[idx] = token
        self._vocab = {
            self._token_to_vocab_key(token): idx
            for token, idx in self.token2idx.items()
        }

        self.name_or_path = str(name_or_path or vocab_file)
        self._max_chars_per_token = max(len(token) for token in self.idx2token)

        self.root = _Trie()
        for token, idx in self.token2idx.items():
            self.root.add(token, val=idx)
        for ch in range(256):
            assert self.root.to[ch] is not None

        self._truncation_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "RWKVTokenizer":
        vocab_file = kwargs.pop("vocab_file", None)
        return cls(vocab_file or _VOCAB_FILE, name_or_path=path_or_repo_id)

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def all_special_tokens(self) -> list[str]:
        return ["<|endoftext|>"]

    @property
    def all_special_ids(self) -> list[int]:
        return [0]

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        return len(self.idx2token)

    @property
    def max_token_id(self) -> int:
        return len(self.idx2token) - 1

    @property
    def max_chars_per_token(self) -> int:
        return self._max_chars_per_token

    @property
    def truncation_side(self) -> str:
        return self._truncation_side

    def num_special_tokens_to_add(self) -> int:
        return 1

    def encode_bytes(self, src: bytes) -> list[int]:
        tokens: list[int] = []
        append = tokens.append
        root_to = self.root.to
        idx = 0
        src_len = len(src)
        while idx < src_len:
            node = root_to[src[idx]]
            assert node is not None
            j = idx + 1
            token = node.token
            end = j
            to = node.to
            while j < src_len:
                node = to[src[j]]
                if node is None:
                    break
                j += 1
                tok = node.token
                if tok:
                    token = tok
                    end = j
                to = node.to
            append(token - 1)
            idx = end
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return b"".join(map(self.idx2token.__getitem__, tokens))

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        tokens = self.encode_bytes(text.encode("utf-8"))
        if add_special_tokens:
            tokens = [self.bos_token_id, *tokens]
        if truncation and max_length is not None:
            if self.truncation_side == "left":
                tokens = tokens[-max_length:]
            else:
                tokens = tokens[:max_length]
        return tokens

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        if skip_special_tokens:
            ids = [idx for idx in ids if idx not in self.all_special_ids]
        return self.decode_bytes(list(ids)).decode("utf-8", errors="replace")

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> BatchEncoding:
        if text_pair is not None:
            raise NotImplementedError("text_pair is not supported for RWKVTokenizer.")
        if isinstance(text, list):
            batch_input_ids = [
                self.encode(
                    value,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )
                for value in text
            ]
            return BatchEncoding(
                {
                    "input_ids": batch_input_ids,
                    "attention_mask": [[1] * len(ids) for ids in batch_input_ids],
                }
            )
        single_input_ids = self.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return BatchEncoding(
            {
                "input_ids": single_input_ids,
                "attention_mask": [1] * len(single_input_ids),
            }
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        if skip_special_tokens:
            ids = [idx for idx in ids if idx not in self.all_special_ids]
        return [self._token_to_vocab_key(self.idx2token[idx]) for idx in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        chunks: list[bytes] = []
        for token in tokens:
            try:
                chunks.append(token.encode("latin-1"))
            except UnicodeEncodeError:
                chunks.append(token.encode("utf-8"))
        return b"".join(chunks).decode("utf-8", errors="replace")

    @staticmethod
    def _token_to_vocab_key(token: bytes) -> str:
        return token.decode("latin-1")

    def _convert_token_to_id(self, token: str) -> int:
        idx = self._vocab.get(token)
        if idx is not None:
            return idx
        return self.token2idx[token.encode("utf-8")]

    def get_chat_template(
        self,
        chat_template: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        del tools
        return chat_template or _RWKV_NATIVE_CHAT_TEMPLATE

    def apply_chat_template(
        self,
        messages: list[Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        chat_template: str | None = None,
        tokenize: bool = False,
        **kwargs,
    ) -> str | list[int]:
        conversation = (
            messages if messages is not None else kwargs.pop("conversation", None)
        )
        if conversation is None:
            raise ValueError("Either 'messages' or 'conversation' must be provided.")

        add_special_tokens = bool(kwargs.pop("add_special_tokens", True))
        template = self.get_chat_template(chat_template, tools=tools)
        if template is not None and template.strip() == _RWKV_NATIVE_CHAT_TEMPLATE:
            prompt = render_rwkv_chat_template(
                list(conversation),
                tools,
                add_generation_prompt=bool(kwargs.get("add_generation_prompt", False)),
                rwkv_generation_prompt=kwargs.get(
                    "rwkv_generation_prompt",
                    "open_think",
                ),
            )
        else:
            rendered, _ = hf_chat_utils.render_jinja_template(
                conversation,
                chat_template=template,
                tools=tools,
                **kwargs,
            )
            prompt = rendered[0] if rendered else ""

        if tokenize:
            return self.encode(prompt, add_special_tokens=add_special_tokens)
        return prompt
