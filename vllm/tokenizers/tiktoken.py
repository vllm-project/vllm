# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Collection, Set
from pathlib import Path
from typing import Any, Literal, List, Union

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
        self.chat_template = getattr(self.tokenizer, "chat_template", None)

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
            assert len(str(path_or_repo_id).split("/")) == 2, (
                "You have either provided a non-existent path: "
                "{path_or_repo_id} or an invalid HF Hub repo id."
            )
            return cls(str(path_or_repo_id), revision=revision)
        else:
            return cls(str(path_or_repo_id))

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
        return self.tokenizer.pad_id

    @property
    def max_token_id(self) -> int:
        return self.vocab_size - 1

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

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_dict: bool = False,
        tokenizer_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> str | list[int]:
        return self.tokenizer.apply_chat_template(
            conversation=conversation,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )

    def encode(
        self,
        s: str,
        add_special_tokens: bool = True,
        *,
        bos: bool = None,
        eos: bool = None,
        allowed_special: Literal["all"] | Set[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = (),
    ) -> list[int]:
        final_bos = bos if bos is not None else add_special_tokens
        final_eos = eos if eos is not None else add_special_tokens
        
        return self.tokenizer.encode(s, bos=final_bos, eos=final_eos)

    def decode(self, ids: list[int] | int, **kwargs) -> str:
        # NOTE: TikToken is not support for these kwargs 
        kwargs.pop("skip_special_tokens", None) 
        kwargs.pop("clean_up_tokenization_spaces", None)
        assert self.kimia_token_offset, (
            "kimia_token_offset must be set for TikTokenTokenizer."
        )
        if isinstance(ids, int):
            ids = [ids]
        token_ids = [t for t in ids if t < self.kimia_token_offset]
        return self.tokenizer.decode(token_ids)

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

    def convert_ids_to_tokens(
        self,
        ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
    ) -> Union[str, bytes, List[Union[str, bytes]]]:
        is_single = isinstance(ids, int)
        if is_single:
            ids = [ids]

        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue

            try:
                raw_bytes = self.model.decode_single_token_bytes(token_id)
                tokens.append(raw_bytes)
            except KeyError:
                token_str = self.inv_vocab.get(token_id, "[UNK]")
                tokens.append(token_str)

        return tokens[0] if is_single else tokens

    def convert_tokens_to_string(self, tokens: List[Union[str, bytes]]) -> str:
        special_token_strs = set(self.special_tokens.keys())
        
        byte_array = bytearray()
        string_parts = []

        for t in tokens:
            if isinstance(t, str) and t in special_token_strs:
                continue

            if isinstance(t, bytes):
                byte_array.extend(t)
            elif isinstance(t, str):
                if byte_array:
                    string_parts.append(byte_array.decode("utf-8", errors="replace"))
                    byte_array.clear()
                string_parts.append(t)

        if byte_array:
            string_parts.append(byte_array.decode("utf-8", errors="replace"))

        return "".join(string_parts)
