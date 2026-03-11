# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# ruff: noqa
# mypy: ignore-errors
# coding=utf-8
# Copyright 2026 The Moonshot AI team and the HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor, BatchFeature, ProcessorMixin
from transformers.audio_utils import AudioInput
from transformers.tokenization_utils_base import TextInput

from vllm.tokenizers.kimi_audio import KimiAudioTokenizer


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute output lengths after Whisper feature extraction."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = (
        ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    )
    return output_lengths


class KimiAudioProcessor(ProcessorMixin):
    r"""
    Constructs a Kimi-Audio processor.

    [`KimiAudioProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`], and a tokenizer.
    See the [`~KimiAudioProcessor.__call__`] and [`~KimiAudioProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The text tokenizer.
    """

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
        # Pass feature_extractor and tokenizer to parent ProcessorMixin
        super().__init__(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            **kwargs,
        )

    def check_argument_for_proper_class(self, attribute_name: str, argument: Any):
        """Override to skip class validation for custom tokenizer."""
        # Skip validation for tokenizer since KimiAudioTokenizer doesn't inherit
        # from PreTrainedTokenizerBase but is compatible
        if attribute_name == "tokenizer" and argument is not None:
            return
        # For other attributes, use default validation
        super().check_argument_for_proper_class(attribute_name, argument)

    def __call__(
        self,
        text: TextInput = None,
        audio: AudioInput = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s).

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded.
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
            return_tensors (`str`):
                The type of tensors to return ("pt", "np", etc.)
        """
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        # Process audio if provided
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

        # Handle text input - can be string or token IDs from vLLM processor
        if isinstance(text, list) and len(text) > 0 and isinstance(text[0], int):
            # Text is already token IDs (from vLLM processor) - just wrap
            text_inputs = {"input_ids": torch.tensor([text], dtype=torch.long)}
        else:
            # Text is string - tokenize
            if not isinstance(text, list):
                text = [text]

            text_inputs = self.tokenizer(
                text, return_tensors=return_tensors, padding=True
            )

        return BatchFeature(
            data={**text_inputs, **audio_inputs},
            tensor_type=return_tensors,
        )
