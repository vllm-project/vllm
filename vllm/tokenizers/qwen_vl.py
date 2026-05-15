# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import unicodedata
from collections.abc import Collection, Set

from transformers import AutoTokenizer

from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike


def get_qwen_vl_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """
    The logic of adding image pad tokens should only be applied in
    `QwenVLProcessor`, so they are patched out here.

    The definition of the wrapped tokenizer can be found here:
    https://huggingface.co/Qwen/Qwen-VL/blob/main/tokenization_qwen.py
    """
    new_tokenizer = copy.copy(tokenizer)

    class TokenizerWithoutImagePad(tokenizer.__class__):  # type: ignore
        def tokenize(
            self,
            text: str,
            allowed_special: Set[str] | str = "all",
            disallowed_special: Collection[str] | str = (),
            **kwargs,
        ) -> list[bytes | str]:
            text = unicodedata.normalize("NFC", text)

            return [
                self.decoder[t]
                for t in self.tokenizer.encode(
                    text,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            ]

        def _decode(
            self,
            token_ids: int | list[int],
            skip_special_tokens: bool = False,
            errors: str | None = None,
            **kwargs,
        ) -> str:
            if isinstance(token_ids, int):
                token_ids = [token_ids]

            return self.tokenizer.decode(
                token_ids,
                errors=errors or self.errors,
            )

    TokenizerWithoutImagePad.__name__ = f"{tokenizer.__class__.__name__}WithoutImagePad"

    new_tokenizer.__class__ = TokenizerWithoutImagePad
    return new_tokenizer


class QwenVLTokenizer(TokenizerLike):
    image_start_tag: str
    image_end_tag: str
    image_pad_tag: str

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_qwen_vl_tokenizer(tokenizer))
