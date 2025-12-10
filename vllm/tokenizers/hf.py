# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
from pathlib import Path
from typing import TypeAlias

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config
from vllm.transformers_utils.gguf_utils import extract_eos_token_id_from_gguf

from .protocol import TokenizerLike

HfTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast

logger = init_logger(__name__)


def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)

    max_token_id = max(tokenizer_vocab.values())
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

        # Patch EOS token ID from GGUF metadata if available
        # GGUF files may have a different EOS token ID than HF tokenizer config
        # (e.g., Gemma uses <end_of_turn> ID 106 as EOS, but HF reports <eos> ID 1)
        gguf_file = kwargs.get("gguf_file")
        if gguf_file:
            gguf_path = Path(path_or_repo_id) / gguf_file
            gguf_eos_id = extract_eos_token_id_from_gguf(str(gguf_path))
            if gguf_eos_id is not None:
                hf_eos_id = tokenizer.eos_token_id
                if hf_eos_id != gguf_eos_id:
                    logger.info(
                        "Patching tokenizer eos_token_id from %d to %d "
                        "(using GGUF metadata)",
                        hf_eos_id,
                        gguf_eos_id,
                    )
                    tokenizer.eos_token_id = gguf_eos_id

        return get_cached_tokenizer(tokenizer)
