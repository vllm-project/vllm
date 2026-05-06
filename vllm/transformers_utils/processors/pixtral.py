# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from PIL import Image
from transformers import BatchFeature, ProcessorMixin, TensorType
from transformers.image_utils import ImageInput

from vllm.tokenizers.mistral import MistralTokenizer


class MistralCommonImageProcessor:
    """
    Provide a HF-compatible interface for
    `mistral_common.tokens.tokenizers.multimodal.ImageEncoder`.
    """

    def __init__(self, mm_encoder: ImageEncoder) -> None:
        self.mm_encoder = mm_encoder

    def __call__(
        self,
        images: ImageInput,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        images_lst = [images] if not isinstance(images, list) else images

        images_processed = list[torch.Tensor]()

        for image in images_lst:
            image_inputs = self.mm_encoder(ImageChunk(image=image))
            image_processed = torch.tensor(image_inputs.image)

            images_processed.append(image_processed)

        return BatchFeature({"images": images_processed}, tensor_type=return_tensors)

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
    ) -> tuple[int, int, int]:
        image = Image.new("RGB", (width, height))
        ncols, nrows = self.mm_encoder._image_to_num_tokens(image)
        return ncols * nrows, nrows, ncols


class MistralCommonPixtralProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        tokenizer: MistralTokenizer,
        image_processor: MistralCommonImageProcessor,
    ) -> None:
        self.tokenizer = tokenizer.transformers_tokenizer

        # Back-compatibility for Transformers v4
        if not hasattr(self.tokenizer, "init_kwargs"):
            self.tokenizer.init_kwargs = {}

        self.image_processor = image_processor

        image_special_ids = self.image_processor.mm_encoder.special_ids
        self.image_break_id = image_special_ids.img_break
        self.image_token_id = image_special_ids.img
        self.image_end_id = image_special_ids.img_end
