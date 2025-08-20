# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, TypedDict, Union, cast

import numpy as np
import torch
import torch.nn as nn
from mistral_common.protocol.instruct.messages import Audio, RawAudio
from mistral_common.protocol.transcription.request import TranscriptionRequest
from transformers import BatchFeature

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs.data import PromptType
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.models.whisper import WhisperForConditionalGeneration
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import (AudioProcessorItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, MultiModalHashes,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from ...transformers_utils.configs import KimiAudioConfig
from ...transformers_utils.processors import KimiAudioProcessor, WhisperEncoder
from .interfaces import (MultiModalEmbeddings, SupportsMultiModal, SupportsPP,
                         SupportsTranscription)
from .moonaudio import MoonshotKimiaModel
from .utils import AutoWeightsLoader, maybe_prefix


class KimiAudioMultiModalProjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim,
                      config.hidden_size,
                      bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size,
                         eps=config.rms_norm_eps,
                         bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class KimiAudioInputs(TypedDict):
    audio_input_ids: Optional[torch.Tensor]
    """Shape: `(num_audios, seq_len)`"""

    # text_input_ids: Optional[torch.Tensor]
    # """Shape: `(num_audios, seq_len)`"""

    is_continuous_mask: list[torch.Tensor]
    """Shape: `(num_audios, seq_len)`"""

    whisper_input_feature: list[torch.Tensor]
    """Shape: `(num_audios, seq_len, feature_dim)`"""


class KimiAudioProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> KimiAudioConfig:
        return self.ctx.get_hf_config(KimiAudioConfig)

    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        return self.ctx.get_hf_processor(KimiAudioProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_max_audio_len(self) -> int:
        processor: KimiAudioProcessor = self.get_hf_processor()
        audio_tokenizer = processor.audio_tokenizer
        sampling_rate = getattr(audio_tokenizer, "sampling_rate", 16000)
        chunk_length = getattr(audio_tokenizer, "chunk_length", 30)
        return int(sampling_rate * chunk_length)


class KimiAudioDummyInputsBuilder(
        BaseDummyInputsBuilder[KimiAudioProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        target_length = self.info.get_max_audio_len()
        return {
            "audio":
            self._get_dummy_audios(length=target_length, num_audios=num_audios)
        }


class KimiAudioMultiModalProcessor(
        BaseMultiModalProcessor[KimiAudioProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_input_ids=MultiModalFieldConfig.batched("audio"),
            is_continuous_mask=MultiModalFieldConfig.batched("audio"),
            whisper_input_feature=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        whisper_model_config = getattr(processor, "whisper_model_config", None)
        audio_id = processor.audio_token_id

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio = audios.get(item_idx)

            nb_audio_tokens = processor.get_num_audio_tokens(
                audio, whisper_model_config)

            return [audio_id] * nb_audio_tokens

        return [
            PromptReplacement(
                modality="audio",
                target="",  # Never match the prompt
                replacement=get_replacement,
            ),
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        return_mm_hashes: bool,
    ) -> tuple[list[int], MultiModalKwargs, Optional[MultiModalHashes], bool]:
        prompt_ids, mm_kwargs, mm_hashes, _ = super(
        )._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            return_mm_hashes=return_mm_hashes,
        )

        # NOTE: The tokens are already inserted by the chat template
        return prompt_ids, mm_kwargs, mm_hashes, True

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)


# TODO(HelloWorldU): Consider extends SupportsTranscription as well
@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder)
class KimiAudioForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP, SupportsTranscription):

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return "<|im_media_begin|><|im_media_end|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiAudioConfig = model_config.hf_config
        self.config = config
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end
        self.kimia_text_eos = config.kimia_text_eos_token_id
        self.kimia_text_blank = config.kimia_text_blank_token_id
        self.kimia_audio_eos = config.kimia_audio_eos_token_id

        mel_batch_size = getattr(config, "mel_batch_size", 20)
        encoder_path = os.path.join(model_config.model, "whisper-large-v3")
        self.audio_tower = WhisperEncoder(
            encoder_path,
            mel_batch_size=mel_batch_size,
        )
        self.multi_modal_projector = KimiAudioMultiModalProjector(self.config)
        self.language_model = MoonshotKimiaModel(
            vllm_config=config,
            prefix=maybe_prefix(prefix, "multi_modal_model"),
        )

        # text only
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                self.config.vocab_size,
                                                logit_scale)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. \
                             Got type: {type(mm_input)}")

        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[KimiAudioInputs]:
        audio_input_ids = kwargs.pop('audio_input_ids', None)
        # text_input_ids = kwargs.pop('text_input_ids', None)
        is_continuous_mask = kwargs.pop('is_continuous_mask', None)
        whisper_input_feature = kwargs.pop('whisper_input_feature', None)

        if whisper_input_feature is None:
            return None

        if is_continuous_mask is not None:
            is_continuous_mask = self._validate_and_reshape_mm_tensor(
                is_continuous_mask, 'is_continuous_mask')
        else:
            return None

        return KimiAudioInputs(
            audio_input_ids=audio_input_ids,
            # text_input_ids= text_input_ids,
            is_continuous_mask=is_continuous_mask,
            whisper_input_feature=whisper_input_feature,
        )

    def _process_audio_input(self,
                             audio_input: KimiAudioInputs) -> torch.Tensor:
        audio_input_ids = audio_input["audio_input_ids"]
        # text_input_ids = audio_input["text_input_ids"]
        whisper_input_feature = audio_input["whisper_input_feature"]
        is_continuous_mask = audio_input["is_continuous_mask"]
        is_continuous_mask = torch.tensor([is_continuous_mask],
                                          dtype=torch.bool)
        whisper_input_feature = self.audio_tower.tokenize_waveform(
            whisper_input_feature)
        whisper_input_feature = whisper_input_feature.reshape(
            whisper_input_feature.shape[0],
            int(whisper_input_feature.shape[1] // 4),
            whisper_input_feature.shape[2] * 4,
        )

        # shape: batch, seq_len, hidden_size
        device = self.language_model.embed_tokens.weight.device
        audio_input_ids = audio_input_ids.to(device)
        # text_input_ids = text_input_ids.to(device)
        audio_emb = self.language_model.get_input_embeddings(audio_input_ids)
        if self.config.use_whisper_feature:
            assert isinstance(whisper_input_feature, list)

            media_start_idx = (
                audio_input_ids == self.kimia_media_begin).nonzero()
            media_end_idx = (audio_input_ids == self.kimia_media_end).nonzero()
            # shape: batch, seq_len, hidden_size
            whisper_input_dim = whisper_input_feature[0].shape[-1]
            whisper_dtype = whisper_input_feature[0].dtype
            projector_device = self.multi_modal_projector.\
                                layers[0].weight.device
            expanded_whisper = torch.zeros(audio_emb.shape[1],
                                           whisper_input_dim,
                                           dtype=whisper_dtype,
                                           device=projector_device)
            for (seg_idx, start_idx), (_,
                                       end_idx) in zip(media_start_idx,
                                                       media_end_idx):
                feat_len = end_idx - (start_idx + 1)
                whisper_input_feature_i = whisper_input_feature[seg_idx].\
                                        squeeze(0)
                assert feat_len == is_continuous_mask[seg_idx].sum()
                expanded_whisper[start_idx + 1:end_idx, :] = (
                    whisper_input_feature_i[:feat_len, :])

            expanded_whisper = expanded_whisper.unsqueeze(0)
            whisper_emb = self.multi_modal_projector(expanded_whisper)
            whisper_emb = whisper_emb.to(device)
            is_continuous_mask = is_continuous_mask.to(device)
            whisper_emb = whisper_emb * is_continuous_mask[:, :, None]

            encoder_input_addwith_discrete_token = (
                audio_emb + whisper_emb) * torch.sqrt(
                    torch.tensor(2.0,
                                 dtype=whisper_emb.dtype,
                                 device=whisper_emb.device))
            audio_emb = (audio_emb * (~is_continuous_mask[:, :, None]) +
                         encoder_input_addwith_discrete_token *
                         is_continuous_mask[:, :, None])
        return audio_emb

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None

        processed_features = self._process_audio_input(audio_input)
        return processed_features

    def _merge_multimodal_embeddings(
            self, inputs_embeds: torch.Tensor,
            audio_emb: MultiModalEmbeddings) -> torch.Tensor:
        inputs_embeds += audio_emb
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None and \
            len(multimodal_embeddings) != 0:
            # customized merge
            inputs_embeds = self._merge_multimodal_embeddings(
                inputs_embeds,
                multimodal_embeddings,
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[tuple[torch.Tensor], IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        **kwargs: object,
    ) -> Optional[torch.Tensor]:
        # TODO(HelloWorldU): Since currently only text logits
        # are supported, we can add multimodal logits in the future.
        text_logits = self.logits_processor(self.lm_head, hidden_states,
                                            sampling_metadata, **kwargs)
        return text_logits

    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig,
                                  task_type: str) -> SpeechToTextConfig:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio_config = tokenizer.instruct.audio_encoder.audio_config
        max_audio_clip_s = audio_config.chunk_length_s
        sample_rate = audio_config.sampling_rate
        return SpeechToTextConfig(
            max_audio_clip_s=max_audio_clip_s,
            sample_rate=sample_rate,
            # mistral_common and whisper encoder take care of chunking
            min_energy_split_window_size=None,
        )

    @classmethod
    # for speech-to-text transcription
    def get_generation_prompt(cls, audio: np.ndarray,
                              model_config: ModelConfig,
                              stt_config: SpeechToTextConfig, language: str,
                              task_type: str,
                              request_prompt: str) -> PromptType:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio = Audio(audio, int(stt_config.sample_rate),
                      format="wav")  # lossless
        req = TranscriptionRequest(model=model_config.model,
                                   audio=RawAudio.from_audio(audio),
                                   language=language)

        tokenized = tokenizer.instruct.encode_transcription(req)
        audio = (tokenized.audios[0].audio_array, stt_config.sample_rate)
        prompts_dict = {"multi_modal_data": {"audio": audio}}
        prompts_dict["prompt_token_ids"] = tokenized.tokens
        return cast(PromptType, prompts_dict)

    @classmethod
    def validate_language(cls, language: str) -> bool:
        # same as whisper
        return WhisperForConditionalGeneration.validate_language(language)

    @classmethod
    def get_num_audio_tokens(cls, audio_duration_s: float,
                             stt_config: SpeechToTextConfig,
                             model_config: ModelConfig) -> Optional[int]:
        """
        Map from audio duration to number of audio tokens produced by the ASR 
        model, without running a forward pass.
        This is used for estimating the amount of processing for this audio.
        """
        tokenizer = cached_tokenizer_from_config(model_config)
        adapter = KimiAudioProcessor(tokenizer)
        return adapter.get_num_audio_tokens(
            int(audio_duration_s * stt_config.sample_rate))

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
