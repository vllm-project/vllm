# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn as nn
from transformers.models.glmasr import GlmAsrConfig, GlmAsrEncoder

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.glmasr_utils import (
    DEFAULT_CONV_PARAMS,
    DEFAULT_MERGE_FACTOR,
    _flatten_audio_features_by_length,
    _get_audio_output_lengths_for_tower,
    _group_audio_embeddings,
    _normalize_chunk_counts,
)
from vllm.multimodal.processors.glmasr import (
    GlmAsrDummyInputsBuilder,
    GlmAsrMultiModalProcessor,
    GlmAsrProcessingInfo,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix


class GlmAsrFeatureInputs(TensorSchema):
    """
    Dimensions:
        - num_chunks: Number of audio chunks (flattened)
        - nmb: Number of mel bins
        - num_audios: Number of original audio files
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("num_chunks", "nmb", "chunk_length", dynamic_dims={"chunk_length"}),
    ]
    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("num_chunks", "chunk_length", dynamic_dims={"chunk_length"}),
    ]
    chunk_counts: Annotated[
        torch.Tensor,
        TensorShape("num_audios"),
    ]


class GlmAsrEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"
    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs", dynamic_dims={"naf"}),
    ]


GlmAsrInputs: TypeAlias = GlmAsrFeatureInputs | GlmAsrEmbeddingInputs


class GlmAsrMultiModalProjector(nn.Module):
    def __init__(self, config: GlmAsrConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.intermediate_size,
            config.text_config.hidden_size * 2,
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size * 2,
            config.text_config.hidden_size,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    GlmAsrMultiModalProcessor,
    info=GlmAsrProcessingInfo,
    dummy_inputs=GlmAsrDummyInputsBuilder,
)
class GlmAsrForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.audio_tower = GlmAsrEncoder(config.audio_config)
        self.multi_modal_projector = GlmAsrMultiModalProjector(config)
        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["LlamaForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|begin_of_audio|><|pad|><|end_of_audio|>"

        raise ValueError("Only audio modality is supported")

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    def _parse_and_validate_audio_input(self, **kwargs: object) -> GlmAsrInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        chunk_counts = kwargs.pop("chunk_counts", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            return GlmAsrEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        if input_features is not None:
            return GlmAsrFeatureInputs(
                type="audio_features",
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                chunk_counts=chunk_counts,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: GlmAsrInputs
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            return tuple(audio_embeds)

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]
        chunk_counts = audio_input.get("chunk_counts")

        if isinstance(input_features, list):
            input_features = torch.cat(input_features, dim=0)
            feature_attention_mask = torch.cat(feature_attention_mask, dim=0)

        chunk_counts = _normalize_chunk_counts(
            chunk_counts, num_chunks=input_features.shape[0]
        )

        audio_outputs = self.audio_tower(input_features)
        audio_hidden_states = audio_outputs.last_hidden_state
        audio_hidden_states = audio_hidden_states.reshape(
            input_features.shape[0],
            -1,
            self.config.audio_config.intermediate_size,
        )
        audio_features = self.multi_modal_projector(audio_hidden_states)

        # Calculate output lengths using encoder's method if available,
        # otherwise use our calculation
        merge_factor = getattr(self.config, "merge_factor", DEFAULT_MERGE_FACTOR)
        conv_params = getattr(self.config, "conv_params", DEFAULT_CONV_PARAMS)

        audio_lengths = feature_attention_mask.sum(-1)
        audio_output_lengths = _get_audio_output_lengths_for_tower(
            self.audio_tower, audio_lengths, merge_factor, conv_params
        )

        masked_audio_features = _flatten_audio_features_by_length(
            audio_features, audio_output_lengths
        )

        chunk_embeddings = torch.split(
            masked_audio_features, audio_output_lengths.flatten().tolist()
        )
        return _group_audio_embeddings(chunk_embeddings, chunk_counts)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["audio_tower.embed_positions"]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights)
