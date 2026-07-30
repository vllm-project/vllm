# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from PIL import Image
from transformers import BatchFeature, ImageProcessingMixin, ProcessorMixin, TensorType
from transformers.image_utils import ImageInput

from vllm.tokenizers.mistral import MistralTokenizer


class MistralCommonImageProcessor(ImageProcessingMixin):
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

    # Copied from Transformers (Apache-2.0):
    # https://github.com/huggingface/transformers/blob/d20946079fd422335fbae3eeb98b7cd88334612f/src/transformers/image_processing_base.py#L473
    def fetch_images(self, image_url_or_urls):
        from transformers.image_utils import is_valid_image, load_image

        if isinstance(image_url_or_urls, (list, tuple)):
            return [self.fetch_images(x) for x in image_url_or_urls]
        if isinstance(image_url_or_urls, str):
            return load_image(image_url_or_urls)
        if is_valid_image(image_url_or_urls):
            return image_url_or_urls
        raise TypeError(
            "only a single or a list of entries is supported but got "
            f"type={type(image_url_or_urls)}"
        )


class MistralCommonPixtralProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        tokenizer: MistralTokenizer,
        image_processor: MistralCommonImageProcessor,
    ) -> None:
        self.tokenizer = tokenizer.transformers_tokenizer
        self.image_processor = image_processor

        image_special_ids = self.image_processor.mm_encoder.special_ids
        self.image_break_id = image_special_ids.img_break
        self.image_token_id = image_special_ids.img
        self.image_end_id = image_special_ids.img_end
