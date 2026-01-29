# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig
from transformers.models.audioflamingo3 import (
    AudioFlamingo3Config,
    AudioFlamingo3Processor,
)
from transformers.models.qwen2_audio import Qwen2AudioEncoder

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
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
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)

MAX_AUDIO_LEN = 10 * 60


# === Audio Inputs === #
class AudioFlamingo3FeatureInputs(TensorSchema):
    """
    Dimensions:
        - num_chunks: Number of audio chunks (flattened)
        - nmb: Number of mel bins
        - num_audios: Number of original audio files
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("num_chunks", "nmb", 3000),
    ]

    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("num_chunks", 3000),
    ]

    chunk_counts: Annotated[
        torch.Tensor,
        TensorShape("num_audios"),
    ]


class AudioFlamingo3EmbeddingInputs(TensorSchema):
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


AudioFlamingo3Inputs: TypeAlias = (
    AudioFlamingo3FeatureInputs | AudioFlamingo3EmbeddingInputs
)


class AudioFlamingo3Encoder(Qwen2AudioEncoder):
    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__(config)
        self.avg_pooler = nn.AvgPool1d(kernel_size=2, stride=2)
        # self.layer_norm is already initialized in super().__init__

    def forward(
        self,
        input_features: torch.Tensor | list[torch.Tensor],
        attention_mask: torch.Tensor = None,
    ):
        # input_features: (batch, num_mel_bins, seq_len)
        if isinstance(input_features, list):
            input_features = torch.stack(input_features)

        hidden_states = nn.functional.gelu(self.conv1(input_features))
        hidden_states = nn.functional.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.transpose(-1, -2)
        hidden_states = (
            hidden_states + self.embed_positions.weight[: hidden_states.size(-2), :]
        ).to(hidden_states.dtype)

        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        # AvgPool (time/2) + LayerNorm
        # hidden_states: (batch, seq_len, hidden_size)
        hidden_states = hidden_states.permute(0, 2, 1)  # (batch, hidden_size, seq_len)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(
            0, 2, 1
        )  # (batch, seq_len/2, hidden_size)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor):
        """
        Computes the output length of the convolutional layers and the output length
        of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class AudioFlamingo3MultiModalProjector(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, audio_features):
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class AudioFlamingo3MultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_audioflamingo3_field_config,
            )

        return super()._parse_audio_data(data)


class AudioFlamingo3ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(AudioFlamingo3Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(AudioFlamingo3Processor, **kwargs)

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor
        return feature_extractor

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()

        return AudioFlamingo3MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class AudioFlamingo3DummyInputsBuilder(
    BaseDummyInputsBuilder[AudioFlamingo3ProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token
        return audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        audio_len = MAX_AUDIO_LEN * sampling_rate
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


def _audioflamingo3_field_config(hf_inputs: Mapping[str, torch.Tensor]):
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


class AudioFlamingo3MultiModalProcessor(
    BaseMultiModalProcessor[AudioFlamingo3ProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: dict[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios

        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        # Calculate chunk counts
        audio_list = mm_data.get("audio")
        if not isinstance(audio_list, list):
            audio_list = [audio_list]

        chunk_counts = []
        sampling_rate = feature_extractor.sampling_rate
        chunk_length = feature_extractor.chunk_length
        window_size = int(sampling_rate * chunk_length)
        # MAX_AUDIO_LEN is 10 * 60 in HF processor.
        max_windows = int(MAX_AUDIO_LEN // chunk_length)

        for audio in audio_list:
            # audio is numpy array or list
            n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]

            n_win = max(1, (n_samples + window_size - 1) // window_size)
            if n_win > max_windows:
                n_win = max_windows
            chunk_counts.append(n_win)

        outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if "input_features_mask" in outputs:
            outputs["feature_attention_mask"] = outputs.pop("input_features_mask")

        outputs["chunk_counts"] = torch.tensor(chunk_counts, dtype=torch.long)

        return outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _audioflamingo3_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        audio_token = getattr(processor, "audio_token", "<sound>")
        audio_token_id = vocab.get(audio_token)
        if audio_token_id is None:
            # Fallback if not found, though it should be there
            audio_token_id = processor.audio_token_id

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        chunk_counts = out_mm_data.get("chunk_counts")

        def get_replacement_audioflamingo3(item_idx: int):
            if feature_attention_mask is not None:
                if chunk_counts is not None:
                    counts = (
                        chunk_counts.tolist()
                        if isinstance(chunk_counts, torch.Tensor)
                        else chunk_counts
                    )
                    start_idx = sum(counts[:item_idx])
                    count = counts[item_idx]
                    end_idx = start_idx + count

                    if isinstance(feature_attention_mask, list):
                        mask_list = feature_attention_mask[start_idx:end_idx]
                        if len(mask_list) > 0 and isinstance(
                            mask_list[0], torch.Tensor
                        ):
                            mask = torch.stack(mask_list)
                        else:
                            mask = torch.tensor(mask_list)
                    else:
                        mask = feature_attention_mask[start_idx:end_idx]
                else:
                    # feature_attention_mask is list[Tensor] or Tensor
                    if isinstance(feature_attention_mask, list):
                        mask = feature_attention_mask[item_idx]
                    else:
                        mask = feature_attention_mask[item_idx].unsqueeze(0)

                # mask shape: (num_chunks, 3000)
                input_lengths = mask.sum(-1)
                conv_lengths = (input_lengths - 1) // 2 + 1
                audio_output_lengths = (conv_lengths - 2) // 2 + 1
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
                replacement=get_replacement_audioflamingo3,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    AudioFlamingo3MultiModalProcessor,
    info=AudioFlamingo3ProcessingInfo,
    dummy_inputs=AudioFlamingo3DummyInputsBuilder,
)
class AudioFlamingo3ForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA
):
    """
    AudioFlamingo3 model for conditional generation.

    This model integrates a Whisper-based audio encoder with a Qwen2 language model.
    It supports multi-chunk audio processing.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = AudioFlamingo3Encoder(
                config.audio_config,
            )
            self.multi_modal_projector = AudioFlamingo3MultiModalProjector(config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> AudioFlamingo3Inputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        chunk_counts = kwargs.pop("chunk_counts", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            return AudioFlamingo3EmbeddingInputs(
                type="audio_embeds", audio_embeds=audio_embeds
            )

        if input_features is not None:
            return AudioFlamingo3FeatureInputs(
                type="audio_features",
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                chunk_counts=chunk_counts,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: AudioFlamingo3Inputs
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

        if chunk_counts is None:
            chunk_counts = [1] * input_features.shape[0]
        elif isinstance(chunk_counts, torch.Tensor):
            chunk_counts = chunk_counts.tolist()
        elif (
            isinstance(chunk_counts, list)
            and chunk_counts
            and isinstance(chunk_counts[0], torch.Tensor)
        ):
            chunk_counts = [c.item() for c in chunk_counts]

        # Calculate output lengths
        input_lengths = feature_attention_mask.sum(-1)
        # Conv downsampling
        conv_lengths = (input_lengths - 1) // 2 + 1
        # AvgPool downsampling
        audio_output_lengths = (conv_lengths - 2) // 2 + 1

        batch_size, _, max_mel_seq_len = input_features.shape

        # Calculate max_seq_len after convs (before pooling) for attention mask
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1

        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0,
                max_seq_len,
                dtype=conv_lengths.dtype,
                device=conv_lengths.device,
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = conv_lengths.unsqueeze(-1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device,
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        # Forward pass
        audio_features = self.audio_tower(
            input_features, attention_mask=audio_attention_mask
        )

        # Project
        audio_features = self.multi_modal_projector(audio_features)

        # Masking after pooling
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_output_lengths = audio_output_lengths.unsqueeze(1)
        audio_features_mask = (
            torch.arange(max_audio_tokens)
            .expand(num_audios, max_audio_tokens)
            .to(audio_output_lengths.device)
            < audio_output_lengths
        )
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

        # Split to tuple of embeddings for individual audio input.
        chunk_embeddings = torch.split(
            masked_audio_features, audio_output_lengths.flatten().tolist()
        )

        grouped_embeddings = []
        current_idx = 0
        for count in chunk_counts:
            audio_chunks = chunk_embeddings[current_idx : current_idx + count]
            grouped_embeddings.append(torch.cat(audio_chunks, dim=0))
            current_idx += count
        return tuple(grouped_embeddings)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
