# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil

import numpy as np
import torch
from mistral_common.tokens.tokenizers.audio import AudioEncoder
from transformers import BatchFeature, ProcessorMixin, TensorType
from transformers.audio_utils import AudioInput
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from vllm.tokenizers.mistral import MistralTokenizer


class MistralCommonFeatureExtractor:
    """
    Provide a HF-compatible interface for
    `mistral_common.tokens.tokenizers.multimodal.AudioEncoder`.
    """

    def __init__(self, audio_encoder: AudioEncoder) -> None:
        self.audio_encoder = audio_encoder

    @property
    def sampling_rate(self):
        return self.audio_encoder.audio_config.sampling_rate

    @property
    def frame_rate(self):
        return self.audio_encoder.audio_config.frame_rate

    def __call__(
        self,
        audios: AudioInput,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        audios_lst = [audios] if not isinstance(audios, list) else audios

        audios_processed = list[torch.Tensor]()

        for audio in audios_lst:
            audio = np.asarray(audio, dtype=np.float32).ravel()
            if not self.audio_encoder.audio_config.is_streaming:
                audio = self.audio_encoder.pad(audio, self.sampling_rate)

            audios_processed.append(torch.tensor(audio))

        return BatchFeature(
            {"audio_arrays": audios_processed}, tensor_type=return_tensors
        )

    def get_num_audio_tokens(self, audio_length: int) -> int:
        return ceil(audio_length / (self.sampling_rate // self.frame_rate))


class MistralCommonVoxtralProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        self.tokenizer = tokenizer.transformers_tokenizer
        self.feature_extractor = MistralCommonFeatureExtractor(
            tokenizer.instruct.audio_encoder
        )

        self._audio_special_ids = self.feature_extractor.audio_encoder.special_ids

    @property
    def audio_token_id(self) -> int:
        return self._audio_special_ids.audio

    @property
    def begin_audio_token_id(self) -> int:
        return self._audio_special_ids.begin_audio

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
        kwargs["audio_kwargs"]["return_tensors"] = None  # Avoid padding issue

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
