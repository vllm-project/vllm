# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from mistral_common.protocol.instruct.chunk import ImageChunk
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from PIL import Image
from transformers import BatchFeature, ProcessorMixin, TensorType
from transformers.audio_utils import AudioInput
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

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

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        self.tokenizer = tokenizer.transformers_tokenizer
        self.image_processor = MistralCommonImageProcessor(
            tokenizer.instruct.mm_encoder
        )

        self._image_special_ids = self.image_processor.mm_encoder.special_ids

    @property
    def image_break_id(self) -> int:
        return self._image_special_ids.img_break

    @property
    def image_token_id(self) -> int:
        return self._image_special_ids.img

    @property
    def image_end_id(self) -> int:
        return self._image_special_ids.img_end

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs,
    ):
        if images is None and text is None and videos is None and audio is None:
            raise ValueError(
                f"You need to provide at least one input to "
                f"call {self.__class__.__name__}"
            )

        kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs={},
            **kwargs,
        )
        kwargs["text_kwargs"]["return_tensors"] = "pt"
        kwargs["images_kwargs"]["return_tensors"] = None  # Avoid padding issue

        attribute_to_kwargs = {
            "tokenizer": (text, "text_kwargs"),
            "image_processor": (images, "images_kwargs"),
            "video_processor": (videos, "videos_kwargs"),
            "feature_extractor": (audio, "audio_kwargs"),
        }
        outputs = {}
        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name, None)
            input_data, input_kwargs = attribute_to_kwargs[attribute_name]
            if input_data is not None and attribute is not None:
                attribute_output = attribute(input_data, **kwargs[input_kwargs])
                outputs.update(attribute_output)

        return BatchFeature(outputs)
