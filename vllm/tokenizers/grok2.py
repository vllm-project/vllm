# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer for Grok-2 .tok.json format."""

import functools
import json
from collections.abc import Collection, Set
from pathlib import Path
from typing import Any, Literal, overload

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from transformers import BatchEncoding
from transformers.utils import chat_template_utils as hf_chat_utils

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.logger import init_logger

from .protocol import TokenizerLike

logger = init_logger(__name__)

PAD = "<|pad|>"
EOS = "<|eos|>"
SEP = "<|separator|>"
RESERVED_TOKEN_TEXTS = [f"<|reserved_{i}|>" for i in range(3, 128)]
CONTROL_TOKEN_TEXTS = [f"<|control{i}|>" for i in range(1, 705)]
DEFAULT_SPECIAL_TOKENS = [PAD, SEP, EOS]
DEFAULT_CONTROL_TOKENS = {"pad": PAD, "sep": SEP, "eos": EOS}
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'Human: ' + message['content'].strip() + '<|separator|>\\n\\n' }}"
    "{% elif message['role'] == 'system' %}"
    "{{ 'System: ' + message['content'].strip() + '<|separator|>\\n\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + '<|separator|>\\n\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Assistant:' }}"
    "{% endif %}"
)

# Default + separate each single digit.
PAT_STR_B = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


def _maybe_load_tokenizer_config(
    model_path: Path,
    *,
    repo_id: str | None,
    revision: str | None,
    download_dir: str | None,
) -> dict[str, Any]:
    config_path = model_path / "tokenizer_config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if repo_id is None:
        return {}

    try:
        config_file = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer_config.json",
            revision=revision,
            cache_dir=download_dir,
        )
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError):
        # If the repo, revision, or file does not exist, fall back silently.
        return {}
    except HfHubHTTPError as exc:
        logger.warning(
            "Failed to download tokenizer_config.json from %s. "
            "This may be due to a network or authentication issue. "
            "The default chat template will be used. Error: %s",
            repo_id,
            exc,
        )
        return {}

    try:
        with Path(config_file).open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse tokenizer_config.json. "
            "The default chat template will be used. Error: %s",
            exc,
        )
        return {}
    except OSError as exc:
        logger.warning(
            "Failed to open tokenizer_config.json. "
            "The default chat template will be used. Error: %s",
            exc,
        )
        return {}


def _load_tiktoken_encoding(
    vocab_file: Path,
) -> tuple[Any, dict[str, int]]:
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError("Grok-2 tokenizer requires the `tiktoken` package.") from exc

    with vocab_file.open("rb") as f:
        xtok_dict = json.load(f)

    mergeable_ranks = {
        bytes(item["bytes"]): item["token"]
        for item in xtok_dict.get("regular_tokens", [])
    }
    special_tokens = {
        bytes(item["bytes"]).decode("utf-8", errors="replace"): item["token"]
        for item in xtok_dict.get("special_tokens", [])
    }

    if xtok_dict.get("word_split") == "V1":
        pat_str = PAT_STR_B
    else:
        raise ValueError(f"Unknown word_split: {xtok_dict.get('word_split')!r}")

    pat_str = xtok_dict.get("pat_str", pat_str)

    kwargs = {
        "name": str(vocab_file),
        "pat_str": pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }

    if "vocab_size" in xtok_dict:
        kwargs["explicit_n_vocab"] = xtok_dict["vocab_size"]

    tokenizer = tiktoken.Encoding(**kwargs)

    default_allowed_special: set[str] | None = None
    if "default_allowed_special" in xtok_dict:
        default_allowed_special = {
            bytes(bytes_list).decode("utf-8", errors="replace")
            for bytes_list in xtok_dict["default_allowed_special"]
        }

    tokenizer._default_allowed_special = default_allowed_special or set()
    tokenizer._control_tokens = DEFAULT_CONTROL_TOKENS

    def encode_patched(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | Set[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        del disallowed_special
        if isinstance(allowed_special, set):
            allowed_special |= self._default_allowed_special
        return tiktoken.Encoding.encode(
            self,
            text,
            allowed_special=allowed_special,
            disallowed_special=(),
        )

    tokenizer.encode = functools.partial(encode_patched, tokenizer)
    tokenizer._default_allowed_special |= set(DEFAULT_CONTROL_TOKENS.values())
    tokenizer._default_allowed_special |= set(
        CONTROL_TOKEN_TEXTS + RESERVED_TOKEN_TEXTS
    )

    return tokenizer, special_tokens


class Grok2Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "Grok2Tokenizer":
        if args:
            logger.debug_once("Ignoring extra positional args for Grok2Tokenizer.")

        path = Path(path_or_repo_id)
        if path.is_file():
            vocab_file = path
            model_path = path.parent
            repo_id = None
        elif path.is_dir():
            vocab_file = path / "tokenizer.tok.json"
            model_path = path
            repo_id = None
        else:
            vocab_file = Path(
                hf_hub_download(
                    repo_id=str(path_or_repo_id),
                    filename="tokenizer.tok.json",
                    revision=revision,
                    cache_dir=download_dir,
                )
            )
            model_path = vocab_file.parent
            repo_id = str(path_or_repo_id)

        if not vocab_file.is_file():
            raise FileNotFoundError(f"tokenizer.tok.json not found at {vocab_file}.")

        config = _maybe_load_tokenizer_config(
            model_path,
            repo_id=repo_id,
            revision=revision,
            download_dir=download_dir,
        )

        return cls(
            vocab_file=vocab_file,
            name_or_path=str(path_or_repo_id),
            truncation_side=kwargs.get("truncation_side", "left"),
            chat_template=config.get("chat_template"),
            init_kwargs=config,
        )

    def __init__(
        self,
        *,
        vocab_file: Path,
        name_or_path: str,
        truncation_side: str,
        chat_template: str | None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.name_or_path = name_or_path
        self._truncation_side = truncation_side
        self.init_kwargs = init_kwargs or {}
        self._chat_template = chat_template or DEFAULT_CHAT_TEMPLATE

        self._tokenizer, self._special_tokens = _load_tiktoken_encoding(vocab_file)

        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token, token_id in self._tokenizer._mergeable_ranks.items():
            token_str = token.decode("utf-8", errors="replace")
            self._token_to_id[token_str] = token_id
            self._id_to_token[token_id] = token_str

        for token, token_id in self._special_tokens.items():
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token

        bos_token_id = self._special_tokens.get(SEP)
        if bos_token_id is None:
            bos_token_id = self._special_tokens.get(PAD)
        if bos_token_id is None:
            bos_token_id = self._special_tokens.get(EOS)
        if bos_token_id is None:
            bos_token_id = 0
        self._bos_token_id = bos_token_id

        self._eos_token_id = self._special_tokens.get(EOS, self._bos_token_id)
        self._pad_token_id = self._special_tokens.get(PAD, self._eos_token_id)
        self._unk_token_id = self._pad_token_id

        self._max_chars_per_token = max(len(tok) for tok in self._token_to_id)

    def num_special_tokens_to_add(self) -> int:
        return 0

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._special_tokens.keys())

    @property
    def all_special_ids(self) -> list[int]:
        return list(self._special_tokens.values())

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.n_vocab

    @property
    def max_token_id(self) -> int:
        return self._tokenizer.n_vocab - 1

    @property
    def max_chars_per_token(self) -> int:
        return self._max_chars_per_token

    @property
    def truncation_side(self) -> str:
        return self._truncation_side

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._special_tokens)

    def _maybe_truncate(self, tokens: list[int], max_length: int | None) -> list[int]:
        if max_length is None or len(tokens) <= max_length:
            return tokens
        if self.truncation_side == "left":
            return tokens[-max_length:]
        return tokens[:max_length]

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        del add_special_tokens
        tokens = self._tokenizer.encode(text)
        if truncation:
            tokens = self._maybe_truncate(tokens, max_length)
        return tokens

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        if skip_special_tokens:
            ids = [
                token_id
                for token_id in ids
                if token_id not in self._special_tokens.values()
            ]
        return self._tokenizer.decode(ids)

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._token_to_id.get(tokens, self._unk_token_id)
        return [self._token_to_id.get(token, self._unk_token_id) for token in tokens]

    def convert_ids_to_tokens(
        self, ids: list[int], skip_special_tokens: bool = False
    ) -> list[str]:
        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in self._special_tokens.values():
                continue
            tokens.append(self._id_to_token.get(token_id, "<|unk|>"))
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        token_ids = self.convert_tokens_to_ids(tokens)
        return self.decode(token_ids, skip_special_tokens=False)

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> BatchEncoding:
        if text_pair is not None:
            raise NotImplementedError("text_pair is not supported for Grok2Tokenizer.")

        if isinstance(text, list):
            input_ids_batch: list[list[int]] = [
                self.encode(
                    item,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )
                for item in text
            ]
            attention_mask_batch = [[1] * len(ids) for ids in input_ids_batch]
            return BatchEncoding(
                {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            )

        input_ids = self.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        attention_mask = [1] * len(input_ids)
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})

    def get_chat_template(
        self, chat_template: str | None, tools: list[dict[str, Any]] | None = None
    ) -> str | None:
        del tools
        return chat_template or self._chat_template

    def apply_chat_template(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]] | None = None,
        chat_template: str | None = None,
        tokenize: bool = False,
        **kwargs,
    ) -> str | list[int]:
        template = self.get_chat_template(chat_template, tools=tools)
        if template is None:
            raise ValueError(
                "No chat template available. Provide `chat_template` explicitly."
            )
        kwargs["return_dict"] = False
        prompt = hf_chat_utils.apply_chat_template(
            conversation=messages,
            chat_template=template,
            tools=tools,
            **kwargs,
        )
        if tokenize:
            return self.encode(prompt, add_special_tokens=False)
        return prompt
