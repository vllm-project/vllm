# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``fastokens`` tokenizer mode.

Loads a Hugging Face fast tokenizer whose internal Rust tokenizer is replaced
by the fastokens shim. fastokens also rebinds
``tokenizers.decoders.DecodeStream`` so the streaming detokenizer accepts the
shim. Both patches are installed for the lifetime of the process —
``patch_transformers()`` is idempotent.
"""

from pathlib import Path

from .hf import CachedHfTokenizer, HfTokenizer
from .protocol import TokenizerLike


def _apply_fastokens_patch() -> None:
    try:
        import fastokens
    except ImportError as e:
        raise ImportError(
            "The 'fastokens' package is required for tokenizer_mode='fastokens'."
        ) from e
    fastokens.patch_transformers()


class FastokensTokenizer(TokenizerLike):
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
        _apply_fastokens_patch()
        return CachedHfTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )
