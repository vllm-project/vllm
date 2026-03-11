# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2026 The Moonshot AI team and the HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Processor for Kimi-Audio ASR model."""

import numpy as np
from transformers import BatchFeature, ProcessorMixin
from transformers.audio_utils import AudioInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class KimiAudioProcessor(ProcessorMixin):
    # Required for ProcessorMixin
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    # Special token IDs
    KIMIA_MEDIA_BEGIN: int = 151661
    KIMIA_MEDIA_END: int = 151663
    KIMIA_TEXT_BLANK: int = 151666

    # Audio processing constants
    AUDIO_SEQ_LEN: int = 376

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = None,
        audio: AudioInput | None = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        if text is not None:
            if not isinstance(text, list):
                text = [text]

            text_inputs = self.tokenizer(
                text, return_tensors=return_tensors, padding=True
            )

        if audio is not None:
            # Ensure audio is a list
            if isinstance(audio, np.ndarray):
                audio = [audio]

            # Pad audio to hop length (required by WhisperFeatureExtractor)
            hop_length = self.feature_extractor.hop_length
            padded_audio = []
            for aud in audio:
                length = aud.shape[-1]
                if length % hop_length != 0:
                    pad_length = hop_length - (length % hop_length)
                    aud = np.pad(
                        aud, (0, pad_length), mode="constant", constant_values=0
                    )
                padded_audio.append(aud)

            # Use feature_extractor directly like Qwen3ASR does
            audio_inputs = self.feature_extractor(
                padded_audio,
                sampling_rate=16000,
                padding=True,
                return_attention_mask=True,
                return_tensors=return_tensors,
            )
            # Rename to match Kimi-Audio expectations
            if "input_features" in audio_inputs:
                audio_inputs["whisper_input_features"] = audio_inputs.pop(
                    "input_features"
                )
            if "attention_mask" in audio_inputs:
                audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                    "attention_mask"
                )
        else:
            audio_inputs = {}

        return BatchFeature(
            data={**text_inputs, **audio_inputs},
            tensor_type=return_tensors,
        )
