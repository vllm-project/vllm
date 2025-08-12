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
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""
from functools import cached_property
from typing import (Any, Iterable, List, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature, ProcessorMixin
from transformers.models.qwen2_audio import (Qwen2AudioConfig,
                                             Qwen2AudioEncoder,
                                             Qwen2AudioProcessor)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import InputContext
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems, ProcessorInputs,
                                        PromptReplacement)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)


# # === Audio Inputs === #
class Qwen2AudioInputs(TypedDict):
    input_features: torch.Tensor
    """Shape: `(num_audios, num_mel_bins, 3000)`"""

    feature_attention_mask: torch.Tensor
    """Shape: `(num_audios, 3000)`"""


# === Audio Encoder === #


class Qwen2AudioMultiModalProjector(nn.Module):

    def __init__(self, audio_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(audio_hidden_size, text_hidden_size, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


# From Qwen2AudioEncoder._get_feat_extract_output_lengths
def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    feat_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (feat_lengths - 2) // 2 + 1
    return feat_lengths, output_lengths


def get_max_qwen2_audio_audio_tokens(ctx: InputContext) -> int:
    hf_config = ctx.get_hf_config(Qwen2AudioConfig)
    max_source_position = hf_config.audio_config.max_source_positions
    output_lengths = (max_source_position - 2) // 2 + 1
    return output_lengths


class Qwen2AudioMultiModalProcessor(BaseMultiModalProcessor):

    def _get_hf_processor(self) -> Qwen2AudioProcessor:
        return self.ctx.get_hf_processor(Qwen2AudioProcessor)

    def _get_feature_extractor(self) -> WhisperFeatureExtractor:
        return self._get_hf_processor().feature_extractor  # type: ignore

    def _get_processor_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # resample audio to the model's sampling rate
        feature_extractor = self._get_feature_extractor()
        mm_items.resample_audios(feature_extractor.sampling_rate)

        return super()._get_processor_data(mm_items)

    def _call_hf_processor(
        self,
        hf_processor: ProcessorMixin,
        prompt: str,
        processor_data: Mapping[str, object],
        mm_processor_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processor_data = dict(processor_data)
        audios = processor_data.pop("audios", [])

        if audios:
            processor_data["audios"] = audios

            feature_extractor = self._get_feature_extractor()
            mm_processor_kwargs = dict(
                **mm_processor_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        else:
            # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
            pass

        return super()._call_hf_processor(
            hf_processor,
            prompt=prompt,
            processor_data=processor_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[PromptReplacement]:
        hf_config = self.ctx.get_hf_config(Qwen2AudioConfig)
        placeholder = hf_config.audio_token_index

        feature_attention_mask = hf_inputs.get("feature_attention_mask")
        if feature_attention_mask is None:
            audio_output_lengths = []
        else:
            _, audio_output_lengths = _get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1))

        def get_replacement_qwen2_audio(item_idx: int):
            return [placeholder] * audio_output_lengths[item_idx]

        return [
            PromptReplacement(
                modality="audio",
                target=[placeholder],
                replacement=get_replacement_qwen2_audio,
            )
        ]

    def _get_dummy_mm_inputs(
        self,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        feature_extractor = self._get_feature_extractor()
        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate

        audio_count = mm_counts["audio"]
        audio = np.zeros(audio_len)
        data = {"audio": [audio] * audio_count}

        return ProcessorInputs(
            prompt_text="<|AUDIO|>" * audio_count,
            mm_data=data,
            mm_processor_kwargs={},
        )


@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "audio", get_max_qwen2_audio_audio_tokens)
@MULTIMODAL_REGISTRY.register_processor(Qwen2AudioMultiModalProcessor)
class Qwen2AudioForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.audio_tower = Qwen2AudioEncoder(config.audio_config)
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(
            config.audio_config.d_model, config.text_config.hidden_size)

        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[Qwen2AudioInputs]:
        input_features = kwargs.pop('input_features', None)
        feature_attention_mask = kwargs.pop('feature_attention_mask', None)
        if input_features is None:
            return None
        input_features = self._validate_and_reshape_mm_tensor(
            input_features, 'input_features')
        feature_attention_mask = self._validate_and_reshape_mm_tensor(
            feature_attention_mask, 'feature_attention_mask')
        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of audio input features. "
                             f"Got type: {type(input_features)}")
        return Qwen2AudioInputs(input_features=input_features,
                                feature_attention_mask=feature_attention_mask)

    def _process_audio_input(self,
                             audio_input: Qwen2AudioInputs) -> torch.Tensor:

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)))

        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (torch.arange(
            0,
            max_seq_len,
            dtype=audio_feat_lengths.dtype,
            device=audio_feat_lengths.device).unsqueeze(0).expand(
                batch_size, max_seq_len))
        lengths_expand = audio_feat_lengths.unsqueeze(-1).expand(
            batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(
            batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len,
                                                  max_seq_len)
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device)
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(input_features,
                                         attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.multi_modal_projector(selected_audio_feature)
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(
            num_audios, max_audio_tokens
        ).to(audio_output_lengths.device) < audio_output_lengths.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(
            -1, embed_dim)

        return masked_audio_features

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.audio_token_index)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
