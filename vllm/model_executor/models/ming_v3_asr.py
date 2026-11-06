# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/ac95e7b1678c08ea6f4a2581a051e432ed63e9b7/modeling_ming_v3_asr.py
# https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/ac95e7b1678c08ea6f4a2581a051e432ed63e9b7/processing_ming_v3_asr.py
# https://github.com/inclusionAI/Ming-3.0-asr/blob/6141e8c552815db724d8df0f72c9d865c547a1ea/examples/hf_runtime.py
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# The HF model/processor sources retain Apache-2.0 headers.
# ASR helpers from hf_runtime.py are covered by the MIT notice below.
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
#
# MIT License
#
# Copyright (c) 2026 inclusionAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Inference-only Ming V3 ASR with the native Bailing hybrid decoder."""

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from numbers import Integral
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import MultiModalDataDict, TokensPrompt
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.ming_v3_asr import (
    MingAudioConfig,
    MingV3ASRConfig,
)
from vllm.transformers_utils.processors.ming_audio import MingASRFeatureExtractor

from .bailing_moe_v3 import BailingMoeV3ForCausalLM
from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    StreamingTranscriptionPostProcessor,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from .ming_audio import MingAudioEncoder, MingAudioProjector, get_audio_token_count
from .module_mapping import MultiModelKeys
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

ASR_INSTRUCTION = (
    "Please recognize the language of this speech and transcribe it, "
    "placing a `\t` between the identified language and the transcribe."
)
_LANGUAGES = {"zh": "Chinese", "en": "English"}
_OUTPUT_LANGUAGES = ("Chinese", "English", "Canton", "Japanese", "Korean")
_AUDIO_FORMS = (
    ("<audio>", "<audioPatch>", "</audio>"),
    ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"),
)


def _audio_tokens(tokenizer) -> tuple[str, str, str]:
    vocab = tokenizer.get_vocab()
    return next(form for form in _AUDIO_FORMS if all(t in vocab for t in form))


def _get_audio_chunk_frames(
    config: MingAudioConfig, options: Mapping[str, object], samples: int
) -> int:
    chunk_frames = options.get("audio_chunk_frames")
    context_chunks = options.get("context_chunks", False)
    if context_chunks not in (False, True):
        raise ValueError("context_chunks must be a boolean or 0/1")
    if context_chunks:
        if chunk_frames is not None:
            raise ValueError("context_chunks and audio_chunk_frames are exclusive")
        if samples > 60 * 16000:
            raise ValueError("Context ASR chunk policy supports at most 60 seconds")
        chunk_frames = 6000 if samples > 30 * 16000 else 3000
    if chunk_frames is None:
        chunk_frames = config.chunk_frames
    if (
        isinstance(chunk_frames, bool)
        or not isinstance(chunk_frames, Integral)
        or not 0 < int(chunk_frames) <= 2 * config.n_audio_ctx
    ):
        raise ValueError(
            "audio_chunk_frames must be a positive integer no larger than "
            f"{2 * config.n_audio_ctx}"
        )
    return int(chunk_frames)


def _transcript(text: str) -> str:
    text = text.strip()
    if "\t" in text:
        return text.split("\t", 1)[1].strip()
    for language in _OUTPUT_LANGUAGES:
        if text.startswith(language):
            return text[len(language) :].strip()
    return text


class MingASRStreamingPostProcessor(StreamingTranscriptionPostProcessor):
    def __init__(self) -> None:
        self.raw_text = ""
        self.emitted_text = ""

    def process_delta(self, text_delta: str, finished: bool) -> str:
        self.raw_text += text_delta
        raw = self.raw_text.strip()
        if not finished and any(
            language.startswith(raw) for language in (*_OUTPUT_LANGUAGES, "Cantonese")
        ):
            return ""
        text = _transcript(self.raw_text)
        delta = text[len(self.emitted_text) :]
        self.emitted_text = text
        return delta


class MingASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> MingV3ASRConfig:
        return self.ctx.get_hf_config(MingV3ASRConfig)

    @cached_property
    def _feature_extractor(self) -> MingASRFeatureExtractor:
        return MingASRFeatureExtractor(self.get_hf_config().audio_config.n_mels)

    def get_feature_extractor(self, **kwargs: object) -> MingASRFeatureExtractor:
        return self._feature_extractor

    def get_supported_mm_processor_kwargs(self) -> dict[str, set[str]]:
        return {"audio_kwargs": {"audio_chunk_frames", "context_chunks"}}

    def get_audio_options(self, kwargs: Mapping[str, object]) -> Mapping[str, object]:
        resolved_kwargs = self._merge_and_resolve_mm_processor_kwargs(kwargs)
        audio_options = resolved_kwargs.get("audio_kwargs", {})
        if not isinstance(audio_options, Mapping):
            raise ValueError("audio_kwargs must be a mapping")

        return audio_options

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000, target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        tokenizer = self.get_tokenizer()
        start, _, end = _audio_tokens(tokenizer)
        overhead = len(tokenizer.encode(ASR_INSTRUCTION + start + end))
        return {"audio": max(1, seq_len - overhead)}


class MingASRDummyInputsBuilder(BaseDummyInputsBuilder[MingASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "".join(_audio_tokens(self.info.get_tokenizer())) * mm_counts.get(
            "audio", 0
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        audio_options = self.info.get_audio_options({})
        config = self.info.get_hf_config().audio_config
        chunk_frames = _get_audio_chunk_frames(config, audio_options, 60 * 16000)
        budget = self.info.get_mm_max_tokens_per_item(seq_len, mm_counts)["audio"]
        low, high = 1, budget * 4 + chunk_frames
        if audio_options.get("context_chunks"):
            high = min(high, 6000)
        while low < high:
            middle = (low + high + 1) // 2
            if get_audio_token_count(config, middle, chunk_frames) <= budget:
                low = middle
            else:
                high = middle - 1
        return {
            "audio": self._get_dummy_audios(
                length=max(201, low * 160),
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class MingASRMultiModalProcessor(BaseMultiModalProcessor[MingASRProcessingInfo]):
    def _call_hf_processor(
        self, hf_data: Mapping[str, object], hf_kwargs: Mapping[str, object]
    ) -> BatchFeature:
        config = self.info.get_hf_config().audio_config
        feature_extractor = self.info.get_feature_extractor()
        audio_options = self.info.get_audio_options(hf_kwargs)
        features, lengths, chunk_frames = [], [], []
        audios = hf_data.get("audio", [])
        assert isinstance(audios, list)
        for audio in audios:
            frames = _get_audio_chunk_frames(config, audio_options, len(audio))
            waveform = torch.as_tensor(audio)
            if waveform.ndim != 1 or waveform.numel() <= 200:
                raise ValueError(
                    "Ming audio requires more than 200 mono samples at 16 kHz"
                )
            if waveform.is_complex() or not torch.isfinite(waveform).all():
                raise ValueError("Audio must contain finite real samples")
            inputs = feature_extractor(
                waveform.detach().to(device="cpu", dtype=torch.float32).numpy(),
                sampling_rate=feature_extractor.sampling_rate,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            feature = inputs.input_features[0].T.to(torch.bfloat16)
            features.append(feature)
            lengths.append(len(feature))
            chunk_frames.append(frames)
        return BatchFeature(
            {
                "audio_feats": features,
                "audio_feats_lengths": torch.tensor(lengths),
                "audio_chunk_frames": torch.tensor(chunk_frames),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            name: MultiModalFieldConfig.batched(
                "audio", keep_on_cpu=name != "audio_feats"
            )
            for name in ("audio_feats", "audio_feats_lengths", "audio_chunk_frames")
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        config = self.info.get_hf_config()
        data = out_mm_kwargs.get_data()
        lengths = torch.as_tensor(data.get("audio_feats_lengths", [])).flatten()
        frames = torch.as_tensor(data.get("audio_chunk_frames", [])).flatten()
        counts = [
            get_audio_token_count(config.audio_config, int(length), int(chunk))
            for length, chunk in zip(lengths, frames, strict=True)
        ]
        return [
            PromptReplacement(
                modality="audio",
                target=[config.audio_token_id],
                replacement=lambda index: [config.audio_token_id] * counts[index],
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MingASRMultiModalProcessor,
    info=MingASRProcessingInfo,
    dummy_inputs=MingASRDummyInputsBuilder,
)
class MingV3ASRForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsTranscription,
    HasInnerState,
    IsHybrid,
    SupportsPP,
):
    supported_languages = _LANGUAGES
    supports_transcription_only = True
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.audio.": "audio_encoder.",
            "linear_proj_audio.": "audio_projector.proj.",
            "model.visual.": None,
            "linear_proj.": None,
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = MingAudioEncoder(
                vllm_config=vllm_config,
                config=self.config.audio_config,
                prefix=maybe_prefix(prefix, "audio_encoder"),
            )
            self.audio_projector = MingAudioProjector(
                self.config.audio_config, self.config.text_config.hidden_size
            )
        with self._mark_language_model(vllm_config):
            self.language_model = BailingMoeV3ForCausalLM(
                vllm_config=vllm_config.with_hf_config(self.config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if not modality.startswith("audio"):
            raise ValueError("Ming V3 ASR only supports audio")
        return "<audio><audioPatch></audio>"

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        features = cast(
            torch.Tensor | list[torch.Tensor] | None, kwargs.get("audio_feats")
        )
        if features is None:
            return ()
        lengths = torch.as_tensor(kwargs["audio_feats_lengths"]).flatten().tolist()
        chunk_frames = torch.as_tensor(kwargs["audio_chunk_frames"]).flatten().tolist()
        outputs = []
        for feature, length, frames in zip(
            features, lengths, chunk_frames, strict=True
        ):
            chunks = []
            feature = feature[:length].to(dtype=self.dtype)
            for chunk in feature.split(frames):
                encoded = self.audio_encoder(chunk)
                projected = self.audio_projector(encoded.unsqueeze(0))[0]
                chunks.append(projected)
            outputs.append(F.normalize(torch.cat(chunks), dim=-1))
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            tower_model=["audio_encoder.", "audio_projector."],
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_shape_from_config(
            vllm_config.with_hf_config(vllm_config.model_config.hf_config.text_config)
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config.with_hf_config(vllm_config.model_config.hf_config.text_config)
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return BailingMoeV3ForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is not None and language not in cls.supported_languages:
            raise ValueError("Ming V3 ASR language hint must be zh or en")
        return language

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        return SpeechToTextConfig(
            sample_rate=16000,
            audio_resample_method="torchaudio",
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> TokensPrompt:
        if stt_params.task_type != "transcribe" or stt_params.to_language is not None:
            raise ValueError("Ming V3 ASR does not support translation")
        language = cls.validate_language(stt_params.language)
        prompt = stt_params.request_prompt
        if not prompt.startswith(ASR_INSTRUCTION):
            prompt = ASR_INSTRUCTION + prompt
        reserved = ("\n", "\r", "<AUDIO>") + tuple(
            token for form in _AUDIO_FORMS for token in form
        )
        for text in (prompt, stt_params.hotwords or ""):
            if any(token in text for token in reserved):
                raise ValueError(
                    "ASR context and hotwords must be contiguous text "
                    "without audio placeholders"
                )
        if stt_params.hotwords:
            prompt += (
                "This audio may contains the following words or phrases:"
                + stt_params.hotwords
            )
        tokenizer = cached_tokenizer_from_config(stt_params.model_config)
        start, audio_token, end = _audio_tokens(tokenizer)
        ids = tokenizer.encode(
            prompt + start + audio_token + end, add_special_tokens=False
        )
        if language is not None:
            ids += tokenizer.encode(
                _LANGUAGES[language] + "\t", add_special_tokens=False
            )
        options = stt_params.extra_args or {}
        processor_options = {
            key: options[key]
            for key in ("audio_chunk_frames", "context_chunks")
            if key in options
        }
        _get_audio_chunk_frames(
            stt_params.model_config.hf_config.audio_config,
            processor_options,
            len(stt_params.audio),
        )
        return TokensPrompt(
            prompt_token_ids=ids,
            multi_modal_data={"audio": stt_params.audio},
            mm_processor_kwargs=processor_options,
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return _transcript(text)

    @classmethod
    def get_streaming_post_processor_cls(cls):
        return MingASRStreamingPostProcessor
