# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer for Kimi-Audio using TikToken."""

import contextlib
import json
from pathlib import Path
from typing import Any, overload

import pybase64
import tiktoken
from huggingface_hub import hf_hub_download
from transformers import AddedToken, BatchEncoding
from transformers.utils import chat_template_utils as hf_chat_utils

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.logger import init_logger
from vllm.tokenizers.protocol import TokenizerLike

logger = init_logger(__name__)


def _load_tiktoken_encoding(
    vocab_file: Path, special_tokens: dict[str, int]
) -> tuple[Any, dict[str, int]]:
    """Load TikToken encoding from vocab file."""
    mergeable_ranks: dict[bytes, int] = {}
    with open(vocab_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                token_b64 = parts[0]
                rank = int(parts[1])
                token_bytes = pybase64.b64decode(token_b64)
                mergeable_ranks[token_bytes] = rank

    tokenizer = tiktoken.Encoding(
        name=str(vocab_file),
        pat_str=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    return tokenizer, special_tokens


class KimiAudioTokenizer(TokenizerLike):
    """TikToken tokenizer for Kimi-Audio."""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "KimiAudioTokenizer":
        if args:
            logger.debug_once("Ignoring extra positional args for KimiAudioTokenizer.")

        path = Path(path_or_repo_id)
        if path.is_file():
            vocab_file = path
        elif path.is_dir():
            vocab_file = path / "tiktoken.model"
            if not vocab_file.is_file():
                vocab_file = path / "tokenizer.model"
        else:
            # Download from HuggingFace Hub
            repo_id = str(path_or_repo_id)

            # Try to download tiktoken.model or tokenizer.model
            try:
                vocab_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="tiktoken.model",
                    revision=revision,
                    local_dir=download_dir,
                )
                vocab_file = Path(vocab_path)
            except Exception:
                try:
                    vocab_path = hf_hub_download(
                        repo_id=repo_id,
                        filename="tokenizer.model",
                        revision=revision,
                        local_dir=download_dir,
                    )
                    vocab_file = Path(vocab_path)
                except Exception as exc:
                    raise ValueError(
                        f"Could not find tiktoken.model or tokenizer.model in {repo_id}"
                    ) from exc

            # Also download tokenizer_config.json if available
            with contextlib.suppress(Exception):
                hf_hub_download(
                    repo_id=repo_id,
                    filename="tokenizer_config.json",
                    revision=revision,
                    local_dir=download_dir,
                )

        if not vocab_file.is_file():
            raise FileNotFoundError(f"tiktoken.model not found at {vocab_file}.")

        return cls(
            vocab_file=vocab_file,
            name_or_path=str(path_or_repo_id),
            truncation_side=kwargs.get("truncation_side", "left"),
        )

    def __init__(
        self,
        *,
        vocab_file: Path,
        name_or_path: str,
        truncation_side: str,
    ) -> None:
        super().__init__()
        self.name_or_path = name_or_path
        self._truncation_side = truncation_side
        self._vocab_file = vocab_file

        # Load special tokens from tokenizer_config.json
        special_tokens: dict[str, int] = {}
        tokenizer_config = vocab_file.parent / "tokenizer_config.json"
        if tokenizer_config.is_file():
            with open(tokenizer_config, encoding="utf-8") as f:
                config = json.load(f)
                # Extract special tokens from added_tokens_decoder
                added_tokens = config.get("added_tokens_decoder", {})
                for token_id_str, token_info in added_tokens.items():
                    token_id = int(token_id_str)
                    content = token_info.get("content", "")
                    if content:
                        special_tokens[content] = token_id

        self._tokenizer, self._special_tokens = _load_tiktoken_encoding(
            vocab_file, special_tokens
        )

        # Build token <-> ID mappings
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token_bytes, token_id in self._tokenizer._mergeable_ranks.items():
            token_str = token_bytes.decode("utf-8", errors="replace")
            self._token_to_id[token_str] = token_id
            self._id_to_token[token_id] = token_str

        # Initialize added_tokens_decoder before adding special tokens
        self._added_tokens_decoder: dict[int, Any] = {}

        # Add Kimi-Audio special tokens
        self._add_kimiaudio_special_tokens()

        # Set default special token IDs (will be updated when special tokens are added)
        self._bos_token_id = 151643  # Kimi-Audio BOS
        self._eos_token_id = 151644  # Kimi-Audio EOS
        self._pad_token_id = self._eos_token_id
        self._unk_token_id = self._pad_token_id

        self._max_chars_per_token = max(
            (len(tok) for tok in self._token_to_id), default=10
        )

    def _add_kimiaudio_special_tokens(self) -> None:
        """Add Kimi-Audio special tokens to the tokenizer."""
        # Tokens should already be in self._special_tokens from tokenizer_config.json
        # Just add them to added_tokens_decoder for compatibility
        kimiaudio_special_tokens = {
            "<|im_media_begin|>": 151661,
            "<|im_media_end|>": 151663,
            "<|im_kimia_text_blank|>": 151666,
            "<|im_msg_end|>": 151645,
            "<|im_kimia_user_msg_start|>": 151670,
            "<|im_kimia_assistant_msg_start|>": 151671,
        }

        for token_str, token_id in kimiaudio_special_tokens.items():
            # Only add if not already present
            if token_id not in self._added_tokens_decoder:
                self._added_tokens_decoder[token_id] = AddedToken(
                    token_str, single_word=True, normalized=False, special=True
                )
                # Also ensure it's in _token_to_id and _id_to_token
                if token_str not in self._token_to_id:
                    self._token_to_id[token_str] = token_id
                if token_id not in self._id_to_token:
                    self._id_to_token[token_id] = token_str

    def num_special_tokens_to_add(self) -> int:
        return 0

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._added_tokens_decoder.values())

    @property
    def all_special_ids(self) -> list[int]:
        return list(self._added_tokens_decoder.keys())

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

    @property
    def added_tokens_decoder(self) -> dict[int, Any]:
        return self._added_tokens_decoder

    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: dict[int, Any]) -> None:
        """Set added tokens decoder and update special token IDs."""
        self._added_tokens_decoder = value
        # Update special token IDs if known tokens are added
        for token_id, token in value.items():
            token_str = str(token) if hasattr(token, "__str__") else token
            if "<|im_kimia_user_msg_start|>" in token_str:
                self._bos_token_id = token_id
            elif "<|im_msg_end|>" in token_str or "<|im_end|>" in token_str:
                self._eos_token_id = token_id

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def __len__(self) -> int:
        """Return vocab size for compatibility with HF tokenizer interface."""
        return self._tokenizer.n_vocab

    def get_added_vocab(self) -> dict[str, int]:
        return {
            str(token): token_id
            for token_id, token in self._added_tokens_decoder.items()
        }

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
        **kwargs,
    ) -> list[int]:
        del add_special_tokens
        # Allow Kimi-Audio special tokens to be encoded
        tokens = self._tokenizer.encode(
            text,
            allowed_special={
                "<|im_media_begin|>",
                "<|im_media_end|>",
                "<|im_kimia_text_blank|>",
                "<|im_msg_end|>",
                "<|im_kimia_user_msg_start|>",
                "<|im_kimia_assistant_msg_start|>",
            },
        )
        if truncation:
            tokens = self._maybe_truncate(tokens, max_length)
        return tokens

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text, optionally skipping special tokens."""
        if isinstance(ids, int):
            ids = [ids]
        if skip_special_tokens:
            # Skip tokens that are in special_tokens (loaded from config)
            special_ids = set(self._special_tokens.values())
            ids = [token_id for token_id in ids if token_id not in special_ids]
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
            if skip_special_tokens and token_id in self._added_tokens_decoder:
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
        **kwargs,
    ) -> BatchEncoding:
        if text_pair is not None:
            raise NotImplementedError(
                "text_pair is not supported for KimiAudioTokenizer."
            )

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
        return chat_template

    def apply_chat_template(
        self,
        messages: list[ChatCompletionMessageParam] | None = None,
        tools: list[dict[str, Any]] | None = None,
        chat_template: str | None = None,
        tokenize: bool = False,
        **kwargs,
    ) -> str | list[int]:
        # Handle both 'messages' (protocol) and 'conversation' (caller) parameter names
        conversation = messages if messages is not None else kwargs.get("conversation")
        if conversation is None:
            raise ValueError("Either 'messages' or 'conversation' must be provided.")
        template = self.get_chat_template(chat_template, tools=tools)
        if template is None:
            raise ValueError(
                "No chat template available. Provide `chat_template` explicitly."
            )
        # Use render_jinja_template instead of apply_chat_template
        # Note: render_jinja_template returns ([prompts], [generation_indices])
        rendered, _ = hf_chat_utils.render_jinja_template(
            conversation,
            chat_template=template,
            tools=tools,
            **kwargs,
        )
        # Extract the first (and usually only) prompt
        prompt = rendered[0] if rendered else ""
        if tokenize:
            return self.encode(prompt, add_special_tokens=False)
        return prompt
