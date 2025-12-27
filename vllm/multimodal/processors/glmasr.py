# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import BatchFeature
from transformers.models.glmasr import GlmAsrConfig, GlmAsrProcessor
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal.glmasr_utils import (
    DEFAULT_CONV_PARAMS,
    DEFAULT_MAX_AUDIO_LEN_S,
    DEFAULT_MERGE_FACTOR,
    _get_audio_output_lengths_from_mask,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder


class GlmAsrProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> GlmAsrConfig:
        return self.ctx.get_hf_config(GlmAsrConfig)

    def get_hf_processor(self, **kwargs: object) -> GlmAsrProcessor:
        return self.ctx.get_hf_processor(GlmAsrProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class GlmAsrDummyInputsBuilder(BaseDummyInputsBuilder[GlmAsrProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        hf_processor = self.info.get_hf_processor()
        return hf_processor.audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        max_audio_len = getattr(
            self.info.get_hf_processor(), "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S
        )
        audio_len = int(max_audio_len * sampling_rate)

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


def _glmasr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    chunk_counts = hf_inputs.get("chunk_counts")
    if chunk_counts is not None:
        return dict(
            audio_embeds=MultiModalFieldConfig.batched("audio"),
            input_features=MultiModalFieldConfig.flat_from_sizes(
                "audio", chunk_counts, dim=0
            ),
            feature_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "audio", chunk_counts, dim=0
            ),
            chunk_counts=MultiModalFieldConfig.batched("audio"),
        )
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
        chunk_counts=MultiModalFieldConfig.batched("audio"),
    )


class GlmAsrMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_glmasr_field_config,
            )
        return super()._parse_audio_data(data)


class GlmAsrMultiModalProcessor(BaseMultiModalProcessor[GlmAsrProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return GlmAsrMultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _calculate_chunk_counts(
        self,
        audio_list: list[Any],
        feature_extractor: WhisperFeatureExtractor,
        processor: GlmAsrProcessor,
    ) -> list[int]:
        """Calculate chunk counts for each audio."""
        sampling_rate = feature_extractor.sampling_rate
        chunk_length = feature_extractor.chunk_length
        max_audio_len = getattr(processor, "max_audio_len", DEFAULT_MAX_AUDIO_LEN_S)
        window_size = int(sampling_rate * chunk_length)
        max_windows = int(max_audio_len // chunk_length)

        chunk_counts = []
        for audio in audio_list:
            n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]
            n_chunks = max(1, (n_samples + window_size - 1) // window_size)
            chunk_counts.append(min(n_chunks, max_windows))
        return chunk_counts

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Handle deprecated "audios" key
        if "audios" in mm_data:
            mm_data["audio"] = mm_data.pop("audios")

        # Text-only input
        audio_list = mm_data.get("audio", [])
        if not audio_list:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Normalize audio_list to list format
        if not isinstance(audio_list, list):
            audio_list = [audio_list]

        processor = self.info.get_hf_processor(**mm_kwargs)
        feature_extractor = processor.feature_extractor
        mm_kwargs = dict(**mm_kwargs, sampling_rate=feature_extractor.sampling_rate)

        # Calculate chunk counts
        chunk_counts = self._calculate_chunk_counts(
            audio_list, feature_extractor, processor
        )

        outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        # Rename mask key and add chunk_counts
        if "input_features_mask" in outputs:
            outputs["feature_attention_mask"] = outputs.pop("input_features_mask")
        outputs["chunk_counts"] = torch.tensor(chunk_counts, dtype=torch.long)

        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _glmasr_field_config(hf_inputs)

    def _normalize_to_tensor(
        self, mask: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        """Convert mask to tensor, handling both list and tensor formats."""
        if isinstance(mask, list):
            return (
                torch.stack(mask)
                if mask and isinstance(mask[0], torch.Tensor)
                else torch.tensor(mask)
            )
        return mask

    def _extract_mask_for_item(
        self,
        feature_attention_mask: torch.Tensor | list[torch.Tensor],
        chunk_counts: torch.Tensor | list[int] | None,
        item_idx: int,
    ) -> torch.Tensor:
        """Extract attention mask for a specific audio item."""
        if chunk_counts is None:
            # Single item per audio
            mask = feature_attention_mask[item_idx]
            return (
                mask.unsqueeze(0)
                if isinstance(feature_attention_mask, torch.Tensor)
                else self._normalize_to_tensor(mask)
            )

        # Multiple chunks per audio: calculate slice indices
        counts = (
            chunk_counts.tolist()
            if isinstance(chunk_counts, torch.Tensor)
            else chunk_counts
        )
        start_idx = sum(counts[:item_idx])
        end_idx = start_idx + counts[item_idx]

        # Extract slice
        if isinstance(feature_attention_mask, torch.Tensor):
            return feature_attention_mask[start_idx:end_idx]
        return self._normalize_to_tensor(feature_attention_mask[start_idx:end_idx])

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        config = self.info.get_hf_config()

        audio_token = getattr(processor, "audio_token", "<|pad|>")
        audio_token_id = vocab.get(audio_token)
        if audio_token_id is None:
            audio_token_id = processor.audio_token_id

        merge_factor = getattr(config, "merge_factor", DEFAULT_MERGE_FACTOR)
        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        chunk_counts = out_mm_data.get("chunk_counts")

        def get_replacement_glmasr(item_idx: int):
            if feature_attention_mask is not None:
                mask = self._extract_mask_for_item(
                    feature_attention_mask, chunk_counts, item_idx
                )
                # Get conv params from config if available
                conv_params = getattr(config, "conv_params", DEFAULT_CONV_PARAMS)
                audio_output_lengths = _get_audio_output_lengths_from_mask(
                    mask, merge_factor, conv_params
                )
                num_features = audio_output_lengths.sum().item()
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                raise ValueError("Audio is too short")

            audio_tokens = [audio_token_id] * int(num_features)
            return PromptUpdateDetails.select_token_id(
                audio_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_glmasr,
            )
        ]
