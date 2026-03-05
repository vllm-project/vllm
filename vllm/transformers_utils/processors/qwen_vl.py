# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import unicodedata
from collections.abc import Collection, Set
from functools import lru_cache

from transformers import PreTrainedTokenizer
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_utils import PILImageResampling
from transformers.processing_utils import ProcessorMixin


@lru_cache(maxsize=1)
def _get_tokenizer_without_image_pad(
    tokenizer: PreTrainedTokenizer,
) -> PreTrainedTokenizer:
    """
    The logic of adding image pad tokens should only be applied in
    `QwenVLProcessor`, so they are patched out here.

    The definition of the wrapped tokenizer can be found here:
    https://huggingface.co/Qwen/Qwen-VL/blob/main/tokenization_qwen.py
    """
    new_tokenizer = copy.deepcopy(tokenizer)

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


class QwenVLImageProcessorFast(BaseImageProcessorFast):
    """
    Port of https://huggingface.co/Qwen/Qwen-VL/blob/main/visual.py#L354
    to HF Transformers.
    """

    resample = PILImageResampling.BICUBIC
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    do_normalize = True


class QwenVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_size: int,
    ) -> None:
        self.tokenizer = _get_tokenizer_without_image_pad(tokenizer)
        self.image_processor = QwenVLImageProcessorFast(
            size={"width": image_size, "height": image_size}
        )

    @property
    def image_start_tag(self) -> str:
        return self.tokenizer.image_start_tag

    @property
    def image_end_tag(self) -> str:
        return self.tokenizer.image_end_tag

    @property
    def image_pad_tag(self) -> str:
        return self.tokenizer.image_pad_tag
