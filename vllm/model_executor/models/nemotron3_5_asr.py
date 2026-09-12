# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline input processing and encoder for NVIDIA Nemotron 3.5 ASR."""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    BatchFeature,
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrProcessor,
    NemotronAsrStreamingEncoder,
)
from transformers.models.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrPromptProjector,
)

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams
from vllm.transformers_utils.processors.nemotron3_5_asr import (
    NemotronAsrStreamingFeatureExtractor,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

_MAX_AUDIO_CLIP_SECONDS = 30


def _get_subsampling_output_lengths(
    input_lengths: torch.Tensor,
    *,
    subsampling_factor: int,
    subsampling_conv_kernel_size: int,
    subsampling_conv_stride: int,
) -> torch.Tensor:
    num_layers = int(math.log2(subsampling_factor))
    all_paddings = (subsampling_conv_kernel_size - 1) + (subsampling_conv_stride - 1)
    add_pad = all_paddings - subsampling_conv_kernel_size

    output_lengths = input_lengths
    for _ in range(num_layers):
        output_lengths = (
            torch.div(
                output_lengths + add_pad,
                subsampling_conv_stride,
                rounding_mode="floor",
            )
            + 1
        )

    return output_lengths


def _get_max_subsampling_input_length(
    output_length: int,
    *,
    subsampling_factor: int,
    subsampling_conv_kernel_size: int,
    subsampling_conv_stride: int,
) -> int:
    num_layers = int(math.log2(subsampling_factor))
    all_paddings = (subsampling_conv_kernel_size - 1) + (subsampling_conv_stride - 1)
    add_pad = all_paddings - subsampling_conv_kernel_size

    input_length = output_length
    for _ in range(num_layers):
        input_length = subsampling_conv_stride * input_length - add_pad - 1

    return input_length


class Nemotron3_5AsrProcessingInfo(BaseProcessingInfo):
    """vLLM metadata and preprocessing information for Nemotron ASR."""

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_hf_config(self) -> Nemotron3_5AsrConfig:
        return self.ctx.get_hf_config(Nemotron3_5AsrConfig)

    def get_hf_processor(self, **kwargs: object) -> Nemotron3_5AsrProcessor:
        del kwargs
        if not hasattr(self, "_cached_hf_processor"):
            revision = self.ctx.model_config.revision
            processor_config = get_hf_file_to_dict(
                "processor_config.json",
                self.model_id,
                revision=revision,
            )
            if processor_config is None:
                raise ValueError(
                    f"processor_config.json was not found for {self.model_id}."
                )
            processor_config = dict(processor_config)
            feature_config = dict(processor_config.pop("feature_extractor"))
            feature_config.pop("feature_extractor_type", None)
            processor_config.pop("processor_class", None)
            feature_extractor = NemotronAsrStreamingFeatureExtractor(**feature_config)
            self._cached_hf_processor = Nemotron3_5AsrProcessor(
                feature_extractor,
                self.get_tokenizer(),
                **processor_config,
            )
        return self._cached_hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
        )

    def get_feature_extractor(
        self, **kwargs: object
    ) -> NemotronAsrStreamingFeatureExtractor:
        processor = self.get_hf_processor(**kwargs)
        return processor.feature_extractor

    @property
    def skip_prompt_length_check(self) -> bool:
        return True


class Nemotron3_5AsrDummyInputsBuilder(
    BaseDummyInputsBuilder[Nemotron3_5AsrProcessingInfo]
):
    """Build a bounded audio input for multimodal profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        encoder_config = self.info.get_hf_config().encoder_config
        num_audios = mm_counts.get("audio", 0)

        max_mel_frames = _get_max_subsampling_input_length(
            encoder_config.max_position_embeddings,
            subsampling_factor=encoder_config.subsampling_factor,
            subsampling_conv_kernel_size=encoder_config.subsampling_conv_kernel_size,
            subsampling_conv_stride=encoder_config.subsampling_conv_stride,
        )
        max_encoder_audio_len = max_mel_frames * feature_extractor.hop_length - 1
        max_clip_audio_len = _MAX_AUDIO_CLIP_SECONDS * feature_extractor.sampling_rate
        audio_len = min(max_encoder_audio_len, max_clip_audio_len)
        audio_overrides = mm_options.get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class Nemotron3_5AsrMultiModalProcessor(
    EncDecMultiModalProcessor[Nemotron3_5AsrProcessingInfo]
):
    """Convert vLLM audio items into Nemotron encoder inputs."""

    skip_decoder_start_token: bool = True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [self.info.get_hf_config().blank_token_id]

    def _preprocess_hf_mm_data(
        self,
        mm_data: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        feature_extractor = self.info.get_feature_extractor(**hf_processor_mm_kwargs)

        mm_data = dict(mm_data)
        mm_data["audio"] = mm_data.pop("audios")
        hf_processor_mm_kwargs = dict(
            **hf_processor_mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        return mm_data, hf_processor_mm_kwargs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_audios = hf_inputs["input_features"].shape[0]
        return {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "attention_mask": MultiModalFieldConfig.batched("audio"),
            "prompt_ids": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            "num_lookahead_tokens": MultiModalFieldConfig.shared(
                "audio", num_audios, keep_on_cpu=True
            ),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        attention_mask = out_mm_kwargs.get_data()["attention_mask"]
        assert isinstance(attention_mask, torch.Tensor)

        encoder_config = self.info.get_hf_config().encoder_config
        output_lengths = _get_subsampling_output_lengths(
            attention_mask.sum(-1),
            subsampling_factor=encoder_config.subsampling_factor,
            subsampling_conv_kernel_size=encoder_config.subsampling_conv_kernel_size,
            subsampling_conv_stride=encoder_config.subsampling_conv_stride,
        ).tolist()

        def replacement(item_idx: int) -> list[int]:
            return [0] * output_lengths[item_idx]

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=replacement,
            )
        ]


class Nemotron3_5AsrAudioEncoder(nn.Module):
    """Encode mel features and apply language-prompt conditioning."""

    def __init__(self, config: Nemotron3_5AsrConfig):
        super().__init__()
        self.config = config
        self.encoder = NemotronAsrStreamingEncoder(config.encoder_config)
        self.encoder_projector = nn.Linear(
            config.encoder_config.hidden_size,
            config.decoder_hidden_size,
        )
        self.prompt_projector = Nemotron3_5AsrPromptProjector(config)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prompt_ids: torch.LongTensor | None = None,
        num_lookahead_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            num_lookahead_tokens=num_lookahead_tokens,
            use_cache=False,
        )
        hidden_states = encoder_outputs.last_hidden_state
        output_mask = encoder_outputs.attention_mask
        if output_mask is not None:
            output_mask = output_mask.bool()

        if prompt_ids is None:
            prompt_ids = torch.full(
                (hidden_states.shape[0],),
                self.config.default_prompt_id,
                dtype=torch.long,
                device=hidden_states.device,
            )
        prompt_ids = prompt_ids.to(hidden_states.device)
        prompt = F.one_hot(
            prompt_ids,
            num_classes=self.config.num_prompts,
        ).to(hidden_states.dtype)
        prompt = prompt[:, None, :].expand(-1, hidden_states.shape[1], -1)
        hidden_states = self.prompt_projector(
            torch.cat((hidden_states, prompt), dim=-1)
        )
        return self.encoder_projector(hidden_states), output_mask
