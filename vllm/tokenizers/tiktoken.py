# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Collection, Set
from pathlib import Path
from typing import Literal

from transformers import AutoTokenizer, PreTrainedTokenizer


class TikTokenTokenizer(PreTrainedTokenizer):
    _bytes_to_id: dict[bytes, int]

    def __init__(self, tokenizer_name: str, **kwargs):
        tokenizer_ = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True, **kwargs
        )
        self.tokenizer = tokenizer_
        self.model = tokenizer_.model

        self.eos_id = None

        # Expose special_tokens as instance attribute (not property) for compatibility
        # This ensures hasattr() checks work reliably in instantiate_extra_tokens
        self.special_tokens = self.tokenizer.special_tokens

        # adapt to kimi-audio
        if "<|im_kimia_text_eos|>" in self.special_tokens:
            self.eos_id = self.special_tokens["<|im_kimia_text_eos|>"]

        self.kimia_token_offset = kwargs.get("kimia_token_offset", 152064)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TikTokenTokenizer":
        if not Path(path_or_repo_id).exists():
            assert len(path_or_repo_id.split("/")) == 2, (
                "You have either provided a non-existent path: "
                "{path_or_repo_id} or an invalid HF Hub repo id."
            )
            return cls(path_or_repo_id, revision=revision)
        else:
            return cls(path_or_repo_id)

    # the following attributes are set to fit vLLM's design and are used
    # by the structured output backends.
    @property
    def all_special_ids(self) -> list[int]:
        return list(self.tokenizer.special_tokens.values())

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self.tokenizer.special_tokens.keys())

    @property
    def all_special_tokens_extended(self) -> list[str]:
        return self.all_special_tokens

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def eos_token_id(self) -> int:
        return self.eos_id if self.eos_id is not None else self.tokenizer.eos_id

    @property
    def inv_vocab(self):
        return {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_words

    @property
    def vocab(self) -> dict[str, int]:
        return self.tokenizer.vocab

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def pad_id(self) -> int:
        """Expose pad_id for KimiAudioProcessor compatibility."""
        # Kimi-Audio uses kimia_assistant_msg_start as pad token
        return self.special_tokens.get(
            "<|im_kimia_assistant_msg_start|>", 151666
        )

    @property
    def pad_token_id(self) -> int:
        return self.pad_id

    @property
    def truncation_side(self) -> str:
        raise NotImplementedError()

    # wrapper for online batch infering
    def __call__(self):
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Literal["all"] | Set[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = (),
    ) -> list[int]:
        return self.tokenizer.encode(s, bos=bos, eos=eos)

    def decode(self, ids: list[int] | int, **kwargs) -> str:
        assert self.kimia_token_offset, (
            "kimia_token_offset must be set for TikTokenTokenizer."
        )
        if isinstance(ids, int):
            ids = [ids]
        token_ids = [t for t in ids if t < self.kimia_token_offset]
        return self.tokenizer.decode(token_ids, **kwargs)

    def _get_bytes_to_id_map(self) -> dict[bytes, int]:
        _bytes_to_id = getattr(self, "_bytes_to_id", None)

        if _bytes_to_id is not None:
            return self._bytes_to_id

        self._bytes_to_id: dict[bytes, int] = {}
        for token_id in range(self.vocab_size):
            try:
                token_bytes = self.model.decode_single_token_bytes(token_id)
                self._bytes_to_id[token_bytes] = token_id
            except KeyError:
                continue

        return self._bytes_to_id

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        special_tokens = set(self.all_special_tokens)

        tokens = [t for t in tokens if t not in special_tokens]
        if not tokens:
            return ""

        has_bytes = any(isinstance(t, bytes) for t in tokens)
        if not has_bytes:
            return "".join(tokens)

        bytes_to_id: dict[bytes, int] = self._get_bytes_to_id_map()
        decoded_strs: list[str] = []
        fallback_ids: list[int] = []

        for t in tokens:
            if isinstance(t, bytes):
                try:
                    decoded_strs.append(t.decode("utf-8"))
                except UnicodeDecodeError:
                    # we convert the bytes to ids and decode them
                    fallback_id = bytes_to_id.get(t)
                    if fallback_id is not None:
                        fallback_ids.append(fallback_id)
            else:
                decoded_strs.append(t)

        if fallback_ids:
            return self.model.decode(fallback_ids)

        return "".join(decoded_strs)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        if isinstance(ids, int):
            ids = [ids]

        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            token = self.inv_vocab.get(token_id, "[UNK]")
            tokens.append(token)

        return tokens[0] if len(tokens) == 1 else tokens
