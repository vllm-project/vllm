# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Horizon team, Xiaomi MiLM Plus.
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Optional, Union, cast

import numpy as np
import torch
from transformers import (AutoProcessor, Qwen2Tokenizer, Qwen2TokenizerFast,
                          Wav2Vec2FeatureExtractor)
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import (ProcessingKwargs, ProcessorMixin,
                                           Unpack)


class MiDashengLMProcessorKwargs(ProcessingKwargs):
    _defaults = {  # type: ignore
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
        },
        "audio_kwargs": {},
    }


def calculate_mel_frames_dasheng(
    audio_length_samples: int,
    n_fft: int = 512,
    hop_size: int = 160,
    dasheng_subsampling: int = 4,
    center=True,
    model_subsampling: int = 5,
) -> int:
    """Calculate the number of Mel-spectrogram frames."""
    if center:
        audio_length_samples = audio_length_samples + n_fft

    return (int(1 + ((audio_length_samples - n_fft) / hop_size)) //
            dasheng_subsampling // model_subsampling)


class MiDashengLMProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "audio_token",
        "audio_bos_token",
        "audio_eos_token",
    ]
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        feature_extractor: Wav2Vec2FeatureExtractor,
        tokenizer: Union[Qwen2Tokenizer, Qwen2TokenizerFast],
        model_subsampling: int = 5,
        chat_template: Optional[Union[str, dict[str, str]]] = None,
        audio_token: Optional[str] = None,
        audio_bos_token: Optional[str] = None,
        audio_eos_token: Optional[str] = None,
    ):
        assert audio_token is not None or hasattr(tokenizer, "audio_token"), (
            "Either `audio_token` must be provided "
            "or tokenizer must have `audio_token` attribute.")
        assert audio_bos_token is not None or hasattr(
            tokenizer, "audio_bos_token"), (
                "Either `audio_bos_token` must be provided "
                "or tokenizer must have `audio_bos_token` attribute.")
        assert audio_eos_token is not None or hasattr(
            tokenizer, "audio_eos_token"), (
                "Either `audio_eos_token` must be provided "
                "or tokenizer must have `audio_eos_token` attribute.")
        assert not feature_extractor.do_normalize, (
            "This model does not use normalization. "
            "Please set `do_normalize=False` in the feature extractor.")

        if chat_template is None:
            chat_template = tokenizer.chat_template

        def get_token(token_name: str) -> str:
            if not hasattr(tokenizer, token_name):
                raise ValueError(
                    f"Tokenizer does not have attribute `{token_name}`. ")
            token = getattr(tokenizer, token_name)
            if not isinstance(token, str):
                raise TypeError(f"Expected token {token_name} to be a string, "
                                f"but got {type(token)}.")
            return token

        self.audio_token = audio_token or get_token("audio_token")
        self.audio_bos_token = audio_bos_token or get_token("audio_bos_token")
        self.audio_eos_token = audio_eos_token or get_token("audio_eos_token")

        self.audio_token_id = cast(
            int, tokenizer.convert_tokens_to_ids(self.audio_token))
        self.model_subsampling = model_subsampling
        self.sampling_rate = feature_extractor.sampling_rate

        super().__init__(feature_extractor,
                         tokenizer,
                         chat_template=chat_template)
        self.feature_extractor: Wav2Vec2FeatureExtractor
        self.tokenizer: Union[Qwen2Tokenizer, Qwen2TokenizerFast]
        self.chat_template: Optional[Union[str, dict[str, str]]]

    def _process_messages_for_chat_template(
        self,
        conversation,
        batch_images,
        batch_videos,
        batch_video_metadata,
        **mm_load_kwargs,
    ):
        if (sr := mm_load_kwargs.get("sampling_rate")
            ) is not None and sr != self.sampling_rate:
            raise ValueError(
                f"This model is trained with "
                f"a sampling rate of {self.sampling_rate}, "
                f"but the sampling rate {sr} is used to load audio.")
        return super()._process_messages_for_chat_template(
            conversation,
            batch_images,
            batch_videos,
            batch_video_metadata,
            **mm_load_kwargs,
        )

    @classmethod
    def _validate_audio_sample(
        cls,
        sample: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        if isinstance(sample, torch.Tensor):
            if sample.ndim != 1:
                raise ValueError("Audio tensor must be 1D.")
            return sample.numpy()
        if isinstance(sample, np.ndarray):
            if sample.ndim != 1:
                raise ValueError("Audio array must be 1D.")
            return sample
        if isinstance(sample, str):
            raise TypeError(
                "Expected audio to be a numpy array or torch tensor.")
        raise TypeError(
            "Expected audio to be a numpy array, torch tensor, or string.")

    def __call__(
        self,
        text: Optional[list[str]] = None,
        audio: Optional[Union[list[np.ndarray], list[torch.Tensor]]] = None,
        **kwargs: Unpack[MiDashengLMProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text."
                             "Please provide a string, or a list of strings")

        if (kwargs.get("images") is not None
                or kwargs.get("videos") is not None):
            raise ValueError("This model does not support images or videos.")

        output_kwargs = self._merge_kwargs(
            MiDashengLMProcessorKwargs,  # type: ignore # Bad type hint in transformers
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            audio = [self._validate_audio_sample(sample) for sample in audio]
            # ensure we have as much audios as audio tokens
            num_audio_tokens = sum(
                sample.count(self.audio_token) for sample in text)
            num_audios = 1 if type(audio) is np.ndarray else len(audio)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token"
                    f"{ 's' if num_audio_tokens > 1 else '' } in provided text "
                    f"but received {num_audios} audio"
                    f"{ 's' if num_audios > 1 else '' }")

            output_kwargs["audio_kwargs"]["return_attention_mask"] = True
            output_kwargs["audio_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["return_tensors"] = "pt"

            # + Padding
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sampling_rate,
                **output_kwargs["audio_kwargs"],
            )

            # remove attention mask, dasheng uses lengths
            audio_feature_mask = audio_inputs.pop("attention_mask")

            expanded_text = []
            audio_lengths = audio_feature_mask.sum(-1).tolist()
            audio_inputs["audio_length"] = torch.tensor(audio_lengths).long()

            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    audio_length = audio_lengths.pop(0)
                    num_audio_tokens = calculate_mel_frames_dasheng(
                        audio_length, model_subsampling=self.model_subsampling)

                    expanded_audio_token = self.audio_token * num_audio_tokens

                    audio_token_start_idx = sample.find(self.audio_token)
                    audio_token_end_idx = audio_token_start_idx + len(
                        self.audio_token)

                    has_bos = (
                        sample[audio_token_start_idx -
                               len(self.audio_bos_token):audio_token_start_idx]
                        == self.audio_bos_token)
                    has_eos = (sample[audio_token_end_idx:audio_token_end_idx +
                                      len(self.audio_eos_token)] ==
                               self.audio_eos_token)

                    # Check if this audio token is surrounded by bos/eos tokens
                    if not has_bos and not has_eos:
                        expanded_audio_token = (self.audio_bos_token +
                                                expanded_audio_token +
                                                self.audio_eos_token)

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>",
                                            1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>",
                                            replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

        return_tensors = output_kwargs["text_kwargs"].pop(
            "return_tensors", "pt")
        inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(
            text,
            BatchFeature(inputs),  # type: ignore
            modalities=["audio"],
        )

        if audio is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(
            dict.fromkeys(tokenizer_input_names +
                          feature_extractor_input_names + ["audio_length"]))


AutoProcessor.register("MiDashengLMProcessor", MiDashengLMProcessor)
