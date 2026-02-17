# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
from pathlib import Path
from typing import TypeAlias

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config

from .protocol import TokenizerLike

HfTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


def _ensure_tokenizer_compat(tokenizer: HfTokenizer) -> None:
    # Limit the fallback to remote/custom tokenizers; avoid patching
    # stock transformers tokenizer behavior.
    if type(tokenizer).__module__.startswith("transformers."):
        return

    decoder = tokenizer.__dict__.get("_added_tokens_decoder")
    if decoder is None:
        with contextlib.suppress(Exception):
            public_decoder = tokenizer.added_tokens_decoder
            if isinstance(public_decoder, dict):
                decoder = dict(public_decoder)
    if decoder is None:
        decoder = {}
    tokenizer.__dict__["_added_tokens_decoder"] = decoder

    if tokenizer.__dict__.get("_added_tokens_encoder") is None:
        tokenizer.__dict__["_added_tokens_encoder"] = {
            str(token): token_id for token_id, token in decoder.items()
        }

    tokenizer.__dict__.setdefault("_extra_special_tokens", [])

    def _noop(*args, **kwargs):
        return None

    if (
        getattr(type(tokenizer), "_update_trie", None) is None
        and "_update_trie" not in tokenizer.__dict__
    ):
        tokenizer._update_trie = _noop
    if (
        getattr(type(tokenizer), "_update_total_vocab_size", None) is None
        and "_update_total_vocab_size" not in tokenizer.__dict__
    ):
        tokenizer._update_total_vocab_size = _noop


def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    _ensure_tokenizer_compat(tokenizer)
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)

    max_token_id = max(tokenizer_vocab.values())
    max_chars_per_token = max(len(tok) for tok in tokenizer_vocab)

    # Some tokenizers (e.g., QwenTokenizer) have special tokens that
    # are added and included in the implementation of the vocab_size
    # property, but not in get_vocab(); if there is an implementation
    # of vocab size, we should take the greater value.
    if hasattr(tokenizer, "vocab_size"):
        with contextlib.suppress(NotImplementedError):
            max_token_id = max(max_token_id, tokenizer.vocab_size)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        @property
        def all_special_ids(self) -> list[int]:
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self) -> list[str]:
            return tokenizer_all_special_tokens

        @property
        def max_token_id(self) -> int:
            return max_token_id

        @property
        def max_chars_per_token(self) -> int:
            return max_chars_per_token

        def get_vocab(self) -> dict[str, int]:
            return tokenizer_vocab

        def __len__(self) -> int:
            return tokenizer_len

        def __reduce__(self):
            return get_cached_tokenizer, (tokenizer,)

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer


class CachedHfTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> HfTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                cache_dir=download_dir,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        _ensure_tokenizer_compat(tokenizer)

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            path_or_repo_id, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        return get_cached_tokenizer(tokenizer)
