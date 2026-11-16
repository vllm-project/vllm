# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LFM3-Audio model: speech in, text out.

The speech output head (codebook embedding, depth decoder) is not served and its
weights are skipped on load.
"""

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

import torch
from torch import nn
from transformers import BatchFeature, Lfm2Model

from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.lfm3_audio import (
    Lfm3AudioConfig,
    Lfm3AudioEncoderConfig,
)
from vllm.transformers_utils.processors.lfm3_audio import (
    Lfm3AudioFeatureExtractor,
    Lfm3AudioProcessor,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from .lfm3 import _short_conv_state_dtype, _short_conv_state_shape
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

# Bounds the profiling run and the encoder cache (one token per 80 ms); the
# encoder itself is causal with a bounded window, so any length runs.
MAX_AUDIO_SECONDS = 600


class Lfm3AudioFeatureInputs(TensorSchema):
    """Dimensions:
    - na: Number of audios
    - nmb: Number of mel bins
    - nf: Number of spectrogram frames
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "nmb", "nf", dynamic_dims={"nf"}),
    ]


class Lfm3AudioEmbeddingInputs(TensorSchema):
    """Dimensions:
    - na: Number of audios
    - nt: Number of audio tokens
    - hs: Hidden size of the language model
    """

    type: Literal["audio_embeds"]
    audio_embeds: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "nt", "hs", dynamic_dims={"nt"}),
    ]


Lfm3AudioInputs: TypeAlias = Lfm3AudioFeatureInputs | Lfm3AudioEmbeddingInputs


def split_audio_features(processed_data: BatchFeature) -> BatchFeature:
    """Keeps each clip's own frames: the features are padded to the longest clip
    of the call, so cached and freshly processed items would differ."""
    if (masks := processed_data.pop("feature_attention_mask", None)) is not None:
        processed_data["input_features"] = [
            features[..., :length]
            for features, length in zip(
                processed_data["input_features"], masks.sum(-1).tolist()
            )
        ]
    return processed_data


def suppress_speech_mode(
    config: Lfm3AudioConfig, logits: torch.Tensor | None
) -> torch.Tensor | None:
    # These tokens switch generation to speech, whose output head is not served.
    if logits is not None:
        logits[..., config.audio_start_token_id] = float("-inf")
        logits[..., config.mixed_start_token_id] = float("-inf")
    return logits


class Lfm3AudioDownsampler(nn.Module):
    """Three stride-2 convolutions over the (mel, time) plane: 128 mel bins and
    8 frames become one vector."""

    def __init__(self, config: Lfm3AudioEncoderConfig) -> None:
        super().__init__()
        channels = config.conv_channels
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, channels, **conv)
        self.conv_mid = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_mid = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_end = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_end = nn.Conv2d(channels, channels, kernel_size=1)
        self.lin_out = nn.Conv1d(
            config.num_mel_bins * channels // 8,
            config.text_config.hidden_size,
            kernel_size=1,
        )
        self.act = nn.SiLU()

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.conv_in(input_features.unsqueeze(1)))
        hidden_states = self.act(self.lin_mid(self.conv_mid(hidden_states)))
        hidden_states = self.act(self.lin_end(self.conv_end(hidden_states)))
        return self.lin_out(hidden_states.flatten(1, 2)).transpose(1, 2)


class Lfm3AudioEncoder(nn.Module):
    def __init__(self, config: Lfm3AudioEncoderConfig) -> None:
        super().__init__()
        self.sliding_window = config.sliding_window
        self.downsample = Lfm3AudioDownsampler(config)
        lfm_config = copy.deepcopy(config.text_config)
        lfm_config._attn_implementation = "sdpa"
        self.lfm = Lfm2Model(lfm_config)
        # The encoder consumes spectrogram features only
        self.lfm.embed_tokens = None

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.downsample(input_features)
        positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        distance = positions[:, None] - positions[None, :]
        mask = (distance >= 0) & (distance < self.sliding_window)
        return self.lfm(
            inputs_embeds=hidden_states,
            attention_mask={"full_attention": mask[None, None], "conv": None},
            use_cache=False,
        ).last_hidden_state


class Lfm3AudioProjector(nn.Module):
    def __init__(self, config: Lfm3AudioConfig, prefix: str = "") -> None:
        super().__init__()
        in_features = config.audio_encoder_config.text_config.hidden_size
        out_features = config.text_config.hidden_size
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear_1 = ReplicatedLinear(
            in_features,
            out_features,
            return_bias=False,
            prefix=maybe_prefix(prefix, "linear_1"),
        )
        self.act = nn.GELU()
        self.linear_2 = ReplicatedLinear(
            out_features,
            out_features,
            return_bias=False,
            prefix=maybe_prefix(prefix, "linear_2"),
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(self.layer_norm(audio_features))
        return self.linear_2(self.act(hidden_states))


class Lfm3AudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Lfm3AudioConfig:
        return self.ctx.get_hf_config(Lfm3AudioConfig)

    def get_hf_processor(self, **kwargs: object) -> Lfm3AudioProcessor:
        return self.ctx.get_hf_processor(Lfm3AudioProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> Lfm3AudioFeatureExtractor:
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.get_feature_extractor().sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_max_audio_samples(self) -> int:
        return MAX_AUDIO_SECONDS * self.get_feature_extractor().sampling_rate

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int] | None = None
    ) -> Mapping[str, int]:
        extractor = self.get_feature_extractor()
        num_frames = extractor.num_frames(self.get_max_audio_samples())
        return {"audio": extractor.num_encoder_tokens(num_frames)}


class Lfm3AudioDummyInputsBuilder(BaseDummyInputsBuilder[Lfm3AudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return self.info.get_hf_processor().audio_token * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        return {
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_samples(),
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class Lfm3AudioMultiModalProcessor(BaseMultiModalProcessor[Lfm3AudioProcessingInfo]):
    def _postprocess_hf_mm_data(
        self,
        mm_data: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
        processed_data: BatchFeature,
    ) -> BatchFeature:
        return split_audio_features(processed_data)

    def _get_mm_fields_config(
        self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "audio_embeds": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        extractor = processor.feature_extractor

        def audio_tokens(item_idx: int) -> list[int]:
            item = out_mm_kwargs["audio"][item_idx]
            if (embeds := item.get("audio_embeds")) is not None:
                count = len(embeds.data)
            else:
                features = item["input_features"].data
                assert isinstance(features, torch.Tensor)
                count = extractor.num_encoder_tokens(features.shape[-1])
            return [processor.audio_token_id] * count

        return [PromptReplacement("audio", [processor.audio_token_id], audio_tokens)]


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3AudioMultiModalProcessor,
    info=Lfm3AudioProcessingInfo,
    dummy_inputs=Lfm3AudioDummyInputsBuilder,
)
class Lfm3AudioForConditionalGeneration(
    nn.Module, HasInnerState, IsHybrid, SupportsMultiModal, SupportsPP
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Training-export layout of the hub checkpoints
            "lfm.model.": "language_model.model.",
            "lfm.lm_head.": "language_model.lm_head.",
            "encoder.lfm.lfm.": "audio_encoder.lfm.",
            "encoder.downsample.": "audio_encoder.downsample.",
            "mlp.model.0.": "audio_projector.layer_norm.",
            "mlp.model.1.": "audio_projector.linear_1.",
            "mlp.model.3.": "audio_projector.linear_2.",
            # Transformers layout
            "model.language_model.": "language_model.model.",
            "model.audio_encoder.": "audio_encoder.",
            "model.audio_projector.": "audio_projector.",
            "lm_head.": "language_model.lm_head.",
            # Speech output head
            "codebook_embedding.": None,
            "depth_linear.": None,
            "depth_embeddings.": None,
            "depthformer.": None,
            "mask_embedding": None,
            "model.codebook_embedding.": None,
            "model.depth_decoder.": None,
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio|>"
        raise ValueError("Only audio modality is supported")

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return _short_conv_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        return _short_conv_state_shape(
            vllm_config, vllm_config.model_config.hf_config.text_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Lfm3AudioConfig = vllm_config.model_config.hf_config
        self.config = config
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = Lfm3AudioEncoder(config.audio_encoder_config)
            self.audio_projector = Lfm3AudioProjector(
                config, prefix=maybe_prefix(prefix, "audio_projector")
            )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Lfm3ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Lfm3AudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        if audio_embeds is not None:
            return Lfm3AudioEmbeddingInputs(
                type="audio_embeds", audio_embeds=audio_embeds
            )
        if input_features is not None:
            return Lfm3AudioFeatureInputs(
                type="audio_features", input_features=input_features
            )
        return None

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        if audio_input["type"] == "audio_embeds":
            return list(audio_input["audio_embeds"])
        dtype = self.audio_encoder.downsample.conv_in.weight.dtype
        return [
            self.audio_projector(self.audio_encoder(features[None].to(dtype))[0])
            for features in audio_input["input_features"]
        ]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return suppress_speech_mode(
            self.config, self.language_model.compute_logits(hidden_states)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
