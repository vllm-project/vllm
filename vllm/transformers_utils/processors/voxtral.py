# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import ceil

import numpy as np
import torch
from mistral_common.tokens.tokenizers.audio import AudioEncoder
from transformers import BatchFeature, ProcessorMixin, TensorType
from transformers.audio_utils import AudioInput

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

    def __init__(
        self,
        tokenizer: MistralTokenizer,
        feature_extractor: MistralCommonFeatureExtractor | None = None,
    ) -> None:
        self.tokenizer = tokenizer.transformers_tokenizer

        # Back-compatibility for Transformers v4
        if not hasattr(self.tokenizer, "init_kwargs"):
            self.tokenizer.init_kwargs = {}

        if feature_extractor is None:
            feature_extractor = MistralCommonFeatureExtractor(
                tokenizer.instruct.audio_encoder
            )

        self.feature_extractor = feature_extractor

        audio_special_ids = self.feature_extractor.audio_encoder.special_ids
        self.audio_token_id = audio_special_ids.audio
        self.begin_audio_token_id = audio_special_ids.begin_audio
