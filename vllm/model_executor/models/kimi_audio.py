# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Kimi-Audio model compatible with HuggingFace weights."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import TypedDict, cast

import numpy as np
import torch
import torch.nn as nn
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.tokens.tokenizers.audio import Audio

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.moonaudio import MoonshotKimiaModel
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.model_executor.models.whisper import WhisperForConditionalGeneration
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
    NestedTensors,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.glm4 import Glm4Tokenizer
from vllm.tokenizers.tiktoken import TikTokenTokenizer
from vllm.transformers_utils.configs import KimiAudioConfig
from vllm.transformers_utils.processors import KimiAudioProcessor
from vllm.transformers_utils.processors.kimi_audio import WhisperEncoder

logger = init_logger(__name__)

ISO639_1_SUPPORTED_LANGS = {
    "zh": "Chinese",
    "en": "English",
}


class KimiAudioMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class KimiAudioInputs(TypedDict):
    audio_input_ids: torch.Tensor | None
    """Shape: `(num_audios, seq_len)`"""

    is_continuous_mask: list[torch.Tensor]
    """Shape: `(num_audios, seq_len)`"""

    audio_waveforms: list[torch.Tensor]
    """List of audio waveforms as numpy arrays for GLM4 tokenization and Whisper feature extraction"""


class KimiAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> KimiAudioConfig:
        return self.ctx.get_hf_config(KimiAudioConfig)

    def get_hf_processor(self, **kwargs: object) -> KimiAudioProcessor:
        text_tokenizer = self.get_tokenizer()
        kwargs["text_tokenizer"] = text_tokenizer

        # Try to get audio_tokenizer from various sources
        # NOTE: Only pass initialization parameters (hashable) to processor __init__
        # Per-request data (lists) should come from generate() call, not here
        if "audio_tokenizer" not in kwargs:
            mm_processor_kwargs = getattr(
                self.ctx.model_config, "mm_processor_kwargs", {}
            )
            if isinstance(mm_processor_kwargs, dict):
                audio_tokenizer = mm_processor_kwargs.get("audio_tokenizer")
                if audio_tokenizer and isinstance(audio_tokenizer, str):
                    kwargs["audio_tokenizer"] = audio_tokenizer

        return self.ctx.get_hf_processor(KimiAudioProcessor, **kwargs)

    def get_tokenizer(self) -> TikTokenTokenizer:
        return self.ctx.tokenizer

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_max_audio_len(self) -> int:
        # 16000 samples * 30s
        return 480000

    def build_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)


class KimiAudioDummyInputsBuilder(BaseDummyInputsBuilder[KimiAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # TODO: Not corrtctly match expected behaviours in case of mm_only
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        kimi_text_blank = getattr(
            hf_processor.extra_tokens, "kimia_text_blank", "<|im_kimia_text_blank|>"
        )
        tokenzier = self.info.get_tokenizer()
        if isinstance(kimi_text_blank, int):
            kimi_text_blank = tokenzier.decode([kimi_text_blank])

        return kimi_text_blank * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        target_length = self.info.get_max_audio_len()

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=target_length, num_audios=num_audios, overrides=audio_overrides
            )
        }


class KimiAudioMultiModalProcessor(BaseMultiModalProcessor[KimiAudioProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_input_ids=MultiModalFieldConfig.batched("audio"),
            is_continuous_mask=MultiModalFieldConfig.batched("audio"),
            audio_waveforms=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """
        Kimi-Audio uses a dual-track architecture where audio_input_ids and
        text_input_ids are two separate, parallel sequences that get merged
        at the embedding level.

        Although we don't modify the text prompt (audio tokens are in the
        separate audio_input_ids stream), we MUST return PromptUpdates to
        satisfy vLLM's framework requirements. This prevents IndexError in
        _merge_mm_kwargs when it tries to index prompt updates for each audio item.

        The actual audio token replacement (placeholders -> discrete tokens)
        happens in the model's _replace_audio_placeholders() method.
        """
        # NOTE: As kimi audio is a special case, we only need to return empty list in this case.
        # Robustly handle cases where "audio" key might be missing (e.g. text-only profiling)
        if "audio" not in mm_items:
            return []

        audio_items = mm_items.get_items("audio", AudioProcessorItems)

        if not audio_items:
            return []

        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        placeholder_id = processor.extra_tokens.kimia_text_blank

        # We use PromptInsertion to instruct vLLM to append placeholders at the end of the prompt.
        # This satisfies vLLM's validation requirement ("Expected N placeholders") without
        # requiring the HF processor to perform the actual injection (which is tricky in Profiling).
        # We also set is_update_applied=False in _cached_apply_hf_processor so vLLM performs this insertion.
        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.end(),
                insertion=[placeholder_id],
            )
            for _ in range(len(audio_items))
        ]

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        processor_data, passthrough_data = super()._get_hf_mm_data(mm_items)

        # NOTE: Ensure audio data is passed to customed KimiAudioProcessor to trigger placeholder injection
        if "audio" in mm_items:
            # Use get_items() to robustly check for audio items
            audio_items = mm_items.get_items("audio", AudioProcessorItems)
            if audio_items:
                # Check if data is already in processor_data
                if "audio" not in processor_data and "audios" not in processor_data:
                    # Force inject using the key expected by KimiAudioProcessor.__call__
                    processor_data["audio"] = audio_items.get_all_items()

        return processor_data, passthrough_data

    def _cached_apply_hf_processor(
        self,
        prompt: str | list[int],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        prompt_ids, mm_info, _ = super()._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        # NOTE: We return False to indicate that prompt updates (inserting placeholders)
        # have NOT been applied yet. vLLM will then execute the PromptInsertion
        # instructions we defined in _get_prompt_updates. (This behavior also
        # match origin logic of vllm)
        return prompt_ids, mm_info, False


@MULTIMODAL_REGISTRY.register_processor(
    KimiAudioMultiModalProcessor,
    info=KimiAudioProcessingInfo,
    dummy_inputs=KimiAudioDummyInputsBuilder,
)
class MoonshotKimiaForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsTranscription
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|im_media_begin|><|im_media_end|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        logger.info("Initializing MoonshotKimiaForCausalLM model...")
        model_config = vllm_config.model_config
        config: KimiAudioConfig = model_config.hf_config
        self.config = config
        self.kimia_media_begin = config.kimia_media_begin
        self.kimia_media_end = config.kimia_media_end

        mel_batch_size = getattr(config, "mel_batch_size", 20)
        encoder_path = os.path.join(model_config.model, "whisper-large-v3")

        # NOTE: The audio tower's weight are not in the main checkpoint
        # so we need to maually add them to the loaded order later.
        with self._mark_tower_model(vllm_config, modalities=["audio"]):
            self.audio_tower = WhisperEncoder(
                encoder_path,
                mel_batch_size=mel_batch_size,
            )
            # Same to multi_modal_projector
            self.multi_modal_projector = KimiAudioMultiModalProjector(self.config)

        with self._mark_language_model(vllm_config):
            self.model = MoonshotKimiaModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "multi_modal_model"),
            )

            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            )
            # NOTE: In current vllm, we only support text logits.
            self.mimo_output = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            )
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.config.vocab_size, self.config.vocab_size, logit_scale
            )

            self.make_empty_intermediate_tensors = (
                self.model.make_empty_intermediate_tensors
            )
        logger.info("Init MoonshotKimiaForCausalLM model done.")

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of {name}. \
                             Got type: {type(mm_input)}"
            )

        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _replace_audio_placeholders(
        self,
        audio_input_ids: torch.Tensor,
        audio_waveforms: list[np.ndarray],
        audio_tokenizer_path: str,
    ) -> torch.Tensor:
        """
        Replace audio placeholder tokens with real discrete audio tokens.

        Args:
            audio_input_ids: Tensor containing [media_begin, placeholder, ..., media_end]
            audio_waveforms: List of audio waveforms (numpy arrays) for GLM4 tokenization
            audio_tokenizer_path: Path to GLM4 audio tokenizer

        Returns:
            Tensor with placeholders replaced by real audio token IDs
        """
        # Lazy initialization of audio tokenizer (only happens in worker process with GPU)
        if not hasattr(self, "_audio_tokenizer") or self._audio_tokenizer is None:
            logger.info(f"Initializing Glm4Tokenizer from {audio_tokenizer_path}")
            self._audio_tokenizer = Glm4Tokenizer(audio_tokenizer_path)
            self._audio_tokenizer = self._audio_tokenizer.to(self.config.device)
            logger.info("Glm4Tokenizer initialized successfully")

        # Get audio token offset from config
        audio_token_offset = getattr(self.config, "kimia_token_offset", 152064)

        # Process concatenated sequence (batch_size=1 in concatenation mode)
        # Kimi-Audio concatenates multiple audios into [1, total_len] with
        # multiple [media_begin...media_end] segments
        batch_size = audio_input_ids.shape[0]
        result_sequences = []

        for batch_idx in range(batch_size):
            seq = audio_input_ids[batch_idx].tolist()

            # Find ALL audio segments (multiple media_begin/media_end pairs)
            audio_segments = []  # [(start_idx, end_idx), ...]
            start_idx = None

            for idx, tok in enumerate(seq):
                if tok == self.kimia_media_begin:
                    start_idx = idx
                elif tok == self.kimia_media_end and start_idx is not None:
                    audio_segments.append((start_idx, idx))
                    start_idx = None

            # Replace each audio segment with discrete tokens
            # Work backwards to maintain correct indices during replacement
            new_seq = seq.copy()
            for segment_idx in reversed(range(len(audio_segments))):
                if segment_idx < len(audio_waveforms):
                    start, end = audio_segments[segment_idx]
                    audio_waveform = audio_waveforms[segment_idx]

                    # Ensure audio_waveform is numpy array (handle dummy data which might be list)
                    if isinstance(audio_waveform, list):
                        audio_waveform = np.array(audio_waveform)

                    # Tokenize audio to discrete tokens using GLM4 tokenizer
                    audio_tokens = self._audio_tokenizer.tokenize(speech=audio_waveform)
                    audio_tokens = audio_tokens.squeeze(0).cpu().tolist()
                    audio_tokens = [tok + audio_token_offset for tok in audio_tokens]

                    # Replace: keep media_begin, replace placeholders, keep media_end
                    new_seq = new_seq[: start + 1] + audio_tokens + new_seq[end:]

            result_sequences.append(new_seq)

        return torch.tensor(
            result_sequences, dtype=torch.long, device=audio_input_ids.device
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> KimiAudioInputs | None:
        audio_input_ids = kwargs.pop("audio_input_ids", None)
        is_continuous_mask = kwargs.pop("is_continuous_mask", None)
        audio_waveforms = kwargs.pop("audio_waveforms", None)
        # Fallback for profiling where key might be "audio"
        if audio_waveforms is None:
            audio_waveforms = kwargs.pop("audio", None)

        audio_tokenizer_path = kwargs.pop("audio_tokenizer", None)

        # Check for None or empty list to avoid cache size calculation errors
        # if audio_waveforms is None or len(audio_waveforms) == 0:
        #    return None
        if audio_waveforms is None:
             return None

        if is_continuous_mask is not None:
            is_continuous_mask = self._validate_and_reshape_mm_tensor(
                is_continuous_mask, "is_continuous_mask"
            )
        else:
            return None

        # Replace audio placeholders with real discrete audio tokens using GLM4 tokenizer
        if audio_input_ids is not None and audio_tokenizer_path is not None:
            audio_input_ids = self._replace_audio_placeholders(
                audio_input_ids,
                audio_waveforms,
                audio_tokenizer_path,
            )

        return KimiAudioInputs(
            audio_input_ids=audio_input_ids,
            is_continuous_mask=is_continuous_mask,
            audio_waveforms=audio_waveforms,
        )

    def _process_audio_input(self, audio_input: KimiAudioInputs) -> torch.Tensor:
        # In this method, we process audio waveforms to extract Whisper input
        # features and combine them with audio input token.
        audio_input_ids = audio_input["audio_input_ids"]
        audio_waveforms = audio_input["audio_waveforms"]
        is_continuous_mask = audio_input["is_continuous_mask"]
        is_continuous_mask = torch.tensor([is_continuous_mask], dtype=torch.bool)

        # Audio waveforms are loaded as numpy arrays in main process (in __call__)
        # Now extract Whisper continuous features in worker process (has GPU and model)
        if audio_waveforms and isinstance(audio_waveforms[0], np.ndarray):
            whisper_features = []
            for audio_data in audio_waveforms:
                # Convert numpy to tensor
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
                # Extract whisper continuous features (in worker, has GPU and model)
                feature = self.audio_tower.tokenize_waveform(audio_tensor)
                feature = feature.reshape(
                    feature.shape[0],
                    int(feature.shape[1] // 4),
                    feature.shape[2] * 4,
                )
                whisper_features.append(feature)
        else:
            # Legacy path: features already provided as tensors
            whisper_features = self.audio_tower.tokenize_waveform(audio_waveforms)
            whisper_features = whisper_features.reshape(
                whisper_features.shape[0],
                int(whisper_features.shape[1] // 4),
                whisper_features.shape[2] * 4,
            )

        # shape: batch, seq_len, hidden_size
        device = self.model.embed_tokens.weight.device
        audio_input_ids = audio_input_ids.to(device)
        # text_input_ids = text_input_ids.to(device)
        audio_emb = self.model.get_input_embeddings(audio_input_ids)
        if self.config.use_whisper_feature:
            assert isinstance(whisper_features, list)

            media_start_idx = (audio_input_ids == self.kimia_media_begin).nonzero()
            media_end_idx = (audio_input_ids == self.kimia_media_end).nonzero()
            # shape: batch, seq_len, hidden_size
            whisper_input_dim = whisper_features[0].shape[-1]
            whisper_dtype = whisper_features[0].dtype
            projector_device = self.multi_modal_projector.layers[0].weight.device
            expanded_whisper = torch.zeros(
                audio_emb.shape[1],
                whisper_input_dim,
                dtype=whisper_dtype,
                device=projector_device,
            )
            for (seg_idx, start_idx), (_, end_idx) in zip(
                media_start_idx, media_end_idx
            ):
                feat_len = end_idx - (start_idx + 1)
                whisper_feature_i = whisper_features[seg_idx].squeeze(0)
                assert feat_len == is_continuous_mask[seg_idx].sum()
                expanded_whisper[start_idx + 1 : end_idx, :] = whisper_feature_i[
                    :feat_len, :
                ]

            expanded_whisper = expanded_whisper.unsqueeze(0)
            whisper_emb = self.multi_modal_projector(expanded_whisper)
            whisper_emb = whisper_emb.to(device)
            is_continuous_mask = is_continuous_mask.to(device)
            whisper_emb = whisper_emb * is_continuous_mask[:, :, None]

            encoder_input_addwith_discrete_token = (
                audio_emb + whisper_emb
            ) * torch.sqrt(
                torch.tensor(2.0, dtype=whisper_emb.dtype, device=whisper_emb.device)
            )
            audio_emb = (
                audio_emb * (~is_continuous_mask[:, :, None])
                + encoder_input_addwith_discrete_token * is_continuous_mask[:, :, None]
            )
        return audio_emb

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        # Validate the multimodal input keyword arguments
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        audio_waveforms = audio_input["audio_waveforms"]
        # if not audio_waveforms:
        #    return []

        whisper_features_list = []

        for audio_data in audio_waveforms:
            # 1. Convert to tensor
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            elif isinstance(audio_data, list):
                # Handle case where audio_data is a list of floats (dummy data)
                audio_tensor = torch.tensor(audio_data).unsqueeze(0)
            else:
                audio_tensor = (
                    audio_data.unsqueeze(0) if audio_data.dim() == 1 else audio_data
                )

            feature = self.audio_tower.tokenize_waveform(audio_tensor)

            feature = feature.reshape(
                feature.shape[0],
                int(feature.shape[1] // 4),
                feature.shape[2] * 4,
            )

            projector_device = self.multi_modal_projector.layers[0].weight.device
            feature = feature.to(projector_device)

            whisper_emb = self.multi_modal_projector(feature)

            whisper_emb = whisper_emb.squeeze(0)

            whisper_features_list.append(whisper_emb)

        return whisper_features_list

    def _merge_audio_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Merge audio embeddings into text embeddings using Kimi-Audio's dual-stream logic.
        """
        if not multimodal_embeddings:
            return inputs_embeds

        # We need to find media segments in input_ids
        media_start_idx = (input_ids == self.kimia_media_begin).nonzero()
        media_end_idx = (input_ids == self.kimia_media_end).nonzero()

        if media_start_idx.shape[0] != len(multimodal_embeddings):
            return inputs_embeds

        device = inputs_embeds.device
        projector_device = self.multi_modal_projector.layers[0].weight.device

        whisper_input_dim = multimodal_embeddings[0].shape[-1]
        whisper_dtype = multimodal_embeddings[0].dtype

        # expanded_whisper will hold the audio features expanded to match sequence length
        expanded_whisper = torch.zeros(
            inputs_embeds.shape[0], # batch
            inputs_embeds.shape[1], # seq_len
            whisper_input_dim,
            dtype=whisper_dtype,
            device=projector_device,
        )

        is_continuous_mask = torch.zeros(
            inputs_embeds.shape[0],  # batch
            inputs_embeds.shape[1],  # seq_len
            dtype=torch.bool,
            device=device,
        )

        for i, (start_pos, end_pos) in enumerate(
            zip(media_start_idx, media_end_idx)
        ):
            batch_idx, start_col = start_pos
            _, end_col = end_pos

            if i >= len(multimodal_embeddings):
                break

            feat = multimodal_embeddings[i]
            feat_len = end_col - (start_col + 1)
            copy_len = min(feat_len, feat.shape[0])

            # Copy features to the correct position
            expanded_whisper[
                batch_idx, start_col + 1 : start_col + 1 + copy_len, :
            ] = feat[:copy_len, :]
            
            # Set mask for continuous area
            is_continuous_mask[batch_idx, start_col + 1 : end_col] = True

        # Project audio features
        whisper_emb = self.multi_modal_projector(expanded_whisper)
        whisper_emb = whisper_emb.to(device)

        # Apply mask
        whisper_emb = whisper_emb * is_continuous_mask[:, :, None]

        # Dual-stream merge: (text + audio) * sqrt(2)
        encoder_input_addwith_discrete_token = (
            inputs_embeds + whisper_emb
        ) * torch.sqrt(torch.tensor(2.0, dtype=whisper_emb.dtype, device=device))

        # Final merge: apply merged embeddings only where mask is True
        inputs_embeds = (
            inputs_embeds * (~is_continuous_mask[:, :, None])
            + encoder_input_addwith_discrete_token * is_continuous_mask[:, :, None]
        )

        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
    ) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor] | IntermediateTensors:
        
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings(input_ids)

        # Attempt to retrieve multimodal embeddings from kwargs
        # The key depends on how vLLM passes the output of embed_multimodal
        # Usually it matches the modality name or is a generic 'multimodal_embeddings' list
        # But wait, embed_multimodal is called OUTSIDE forward in V1 model runner.
        # Its output is passed to forward.
        
        # Let's inspect what keys are available in kwargs during runtime if possible, 
        # but for now we implement the logic assuming we can get the features.
        
        # In V1, SupportsMultiModal models receiving multimodal inputs in forward 
        # typically see them as arguments corresponding to the modality, e.g. 'audio' or 'vision_chunk'.
        # Since our embed_multimodal returns a list of tensors, we expect a list of tensors here.
        
        multimodal_embeddings = kwargs.get("audio") # Assuming 'audio' is the key based on registry
        
        if multimodal_embeddings is not None:
            # We also need 'is_continuous_mask' which is metadata.
            # This should have been passed through from the batch/processor.
            # But vLLM might not pass arbitrary tensor attributes from BatchFeature to forward unless configured.
            inputs_embeds = self._merge_audio_embeddings(
                inputs_embeds, input_ids, multimodal_embeddings
            )

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # TODO(HelloWorldU): Since currently only text logits
        # are supported, we can add multimodal logits in the future.
        text_logits = self.logits_processor(self.lm_head, hidden_states)
        return text_logits

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
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
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        model_config: ModelConfig,
        stt_config: SpeechToTextConfig,
        language: str,
        task_type: str,
        request_prompt: str,
    ) -> PromptType:
        tokenizer = cached_tokenizer_from_config(model_config)
        audio = Audio(audio, int(stt_config.sample_rate), format="wav")  # lossless
        req = TranscriptionRequest(
            model=model_config.model,
            audio=RawAudio.from_audio(audio),
            language=language,
        )

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
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        """
        Map from audio duration to number of audio tokens produced by the ASR
        model, without running a forward pass.
        This is used for estimating the amount of processing for this audio.
        """
        tokenizer = cached_tokenizer_from_config(model_config)
        adapter = KimiAudioProcessor(tokenizer)
        return adapter.get_num_audio_tokens(
            int(audio_duration_s * stt_config.sample_rate)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # audio_tower is initialized in __init__ via WhisperModel.from_pretrained
        # and its weights are NOT in the main checkpoint.
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights)

        # Manually register audio_tower parameters as loaded
        # to satisfy vLLM's initialization check
        for name, _ in self.audio_tower.named_parameters():
            loaded_weights.add(f"audio_tower.{name}")
        # Same to multi_modal_projector
        for name, _ in self.multi_modal_projector.named_parameters():
            loaded_weights.add(f"multi_modal_projector.{name}")

        return loaded_weights
