# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil

import numpy as np
import torch
from mistral_common.tokens.tokenizers.audio import AudioEncoder
from transformers import BatchFeature, ProcessorMixin, TensorType
from transformers.audio_utils import AudioInput
from transformers.processing_utils import ProcessingKwargs

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

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: type[ProcessingKwargs],
        tokenizer_init_kwargs: dict | None = None,
        **kwargs,
    ):
        return {
            "text_kwargs": tokenizer_init_kwargs,
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }
