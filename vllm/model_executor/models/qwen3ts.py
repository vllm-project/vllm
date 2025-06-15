# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Tsinghua University and ByteDance.
# Copyright 2025 Alexander Chemeris <Alexander.Chemeris@gmail.com>
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Reference: vLLM (https://github.com/vllm-project/vllm)
"""Inference-only Qwen3TS model compatible with HuggingFace weights."""
from collections.abc import Iterable, Mapping
from typing import Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig, ProcessorMixin

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.interfaces import (SupportsLoRA,
                                                   SupportsMultiModal,
                                                   SupportsPP)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              init_vllm_registered_model,
                                              maybe_prefix,
                                              merge_multimodal_embeddings)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, MultiModalDataDict,
                                        MultiModalDataItems,
                                        MultiModalFieldConfig,
                                        MultiModalKwargs, PromptReplacement,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

# === Time Series Inputs === #


class Qwen3TSTimeSeriesInput(TypedDict):
    timeseries: torch.Tensor
    """Shape: `(num_time_series, max_length, num_features)`
    
    Concatenated time series data from all time series in the batch.
    Each time series may have different lengths, padded to max_length.
    The last feature dimension contains mask information.
    """


# === TimeSeriesEmbedding ===
class TimeSeriesEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.patch_size = config['patch_size']
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']
        self.num_features = config['num_features']

        layers = []
        input_size = 1 * self.patch_size

        for _ in range(self.num_layers - 1):
            layers.append(
                ReplicatedLinear(input_size,
                                 self.hidden_size,
                                 return_bias=False))
            layers.append(nn.GELU())
            input_size = self.hidden_size
        layers.append(
            ReplicatedLinear(input_size, self.hidden_size, return_bias=False))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, self.num_features)

        mask = x[:, :, -1]
        valid_lengths = mask.sum(dim=1).long()  # Shape: (batch_size)

        patch_cnt = (valid_lengths + self.patch_size -
                     1) // self.patch_size  # 向上取整

        patches_list = []
        for i in range(batch_size):
            vl = valid_lengths[i].item()
            pc = patch_cnt[i].item()
            if pc == 0:
                continue
            xi = x[i, :vl, :1]
            total_padded_length = pc * self.patch_size
            padding_length = total_padded_length - vl
            if padding_length > 0:
                padding = torch.zeros(padding_length,
                                      1,
                                      device=x.device,
                                      dtype=x.dtype)
                xi = torch.cat([xi, padding], dim=0)
            xi = xi.reshape(pc, self.patch_size * 1)
            patches_list.append(xi)

        if patches_list:
            x_patches = torch.cat(
                patches_list,
                dim=0)  # Shape: (total_patch_cnt, patch_size * num_features)
            x = self.mlp(x_patches)
        else:
            x = torch.empty(0, self.hidden_size, device=x.device)

        return x, patch_cnt


# === TS Encoder === #
# get_patch_cnt: From Time Series Embedding
def get_patch_cnt(x: torch.Tensor, ts_config: PretrainedConfig):
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1, ts_config['num_features'])

    mask = x[:, :, -1]
    valid_lengths = mask.sum(1).long()  # Shape: (batch_size)

    patch_cnt = (valid_lengths + int(ts_config['patch_size']) - 1) // int(
        ts_config['patch_size'])
    return patch_cnt


class Qwen3TSProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(PretrainedConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(ProcessorMixin, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"timeseries": 50}  # Allow up to 50 time series per prompt


class Qwen3TSDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3TSProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<ts><ts/>" * mm_counts.get("timeseries", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        hf_config = self.info.get_hf_config()
        max_ts_length = hf_config.ts['max_length']
        ts_count = mm_counts.get("timeseries", 0)
        return {
            "timeseries":
            self._get_dummy_timeseries(length=max_ts_length,
                                       num_timeseries=ts_count)
        }


class Qwen3TSMultiModalProcessor(BaseMultiModalProcessor[Qwen3TSProcessingInfo]
                                 ):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        ts = mm_data.pop("timeseries", [])

        if ts:
            mm_data["timeseries"] = ts

        mm_kwargs = dict(mm_kwargs)
        mm_kwargs['vllm_flag'] = True
        result = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        return result

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # Define the field name and configuration for time series data
        return {
            "timeseries": MultiModalFieldConfig.batched("timeseries"),
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        hf_config = self.info.get_hf_config()
        placeholder = hf_config.ts_token_start_index

        if 'timeseries' not in mm_items:
            return []

        if 'timeseries' not in out_mm_kwargs:
            return []

        # ChatTS processor returns a list of tuples
        # (ts_tokens, encoded_ts_arrays)
        ts_tokens, encoded_ts_arrays = zip(*out_mm_kwargs["timeseries"])

        # patch_cnt = get_patch_cnt(concatenated_ts, hf_config.ts)
        patch_size = hf_config.ts['patch_size']
        # encoded_ts_arrays: list[torch.Tensor]
        # encoded_ts_arrays[i].shape: (num_rows, num_elements*2, num_features)
        # num_elements*2 is used because each element is a pair of (value, mask)
        patch_cnt = [
            (encoded_ts_arrays[i].shape[1] // 2 + patch_size - 1) // patch_size
            for i in range(len(encoded_ts_arrays))
        ]

        def get_replacement_qwen3_ts(item_idx: int):
            # Use the pre-tokenized replacements
            tokens = ts_tokens[item_idx].copy()
            # Extend the tokens with placeholders to match the patch_cnt
            num_placeholders = sum(1 for t in tokens if t == placeholder)
            if num_placeholders < patch_cnt[item_idx]:
                tokens.extend([placeholder] *
                              (patch_cnt[item_idx] - num_placeholders))
            # return tokens
            return PromptUpdateDetails.select_token_id(
                tokens,
                embed_token_id=placeholder,
            )

        return [
            PromptReplacement(
                modality="timeseries",
                target=[placeholder, placeholder + 1],
                replacement=get_replacement_qwen3_ts,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3TSMultiModalProcessor,
    info=Qwen3TSProcessingInfo,
    dummy_inputs=Qwen3TSDummyInputsBuilder,
)
class Qwen3TSForCausalLM(nn.Module, SupportsMultiModal, SupportsPP,
                         SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: PretrainedConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.model_dtype = vllm_config.model_config.dtype

        self.ts_encoder = TimeSeriesEmbedding(config.ts)
        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen3ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_ts_input(
            self, **kwargs: object) -> Optional[Qwen3TSTimeSeriesInput]:
        timeseries = kwargs.pop('timeseries', None)
        if timeseries is None:
            return None

        # ChatTS processor returns a list of tuples
        # (ts_tokens, encoded_ts_arrays)
        encoded_ts_arrays = [ts[0][1] for ts in timeseries]
        # ts_tokens, encoded_ts_arrays = zip(*timeseries)

        device = encoded_ts_arrays[0].device

        max_length = max(ts.shape[1] for ts in encoded_ts_arrays)
        total_rows = sum(ts.shape[0] for ts in encoded_ts_arrays)
        feature_dim = encoded_ts_arrays[0].shape[2] if encoded_ts_arrays else 0

        # Pre-allocate the tensor with the right size
        concatenated_ts = torch.zeros((total_rows, max_length, feature_dim),
                                      dtype=self.model_dtype,
                                      device=device)

        # Copy each array to the right position
        row_offset = 0
        for ts in encoded_ts_arrays:
            ts_tensor = torch.tensor(ts, dtype=self.model_dtype,
                                     device=device) if isinstance(
                                         ts, np.ndarray) else ts
            concatenated_ts[row_offset:row_offset +
                            ts.shape[0], :ts.shape[1], :] = ts_tensor
            row_offset += ts.shape[0]

        input_features = concatenated_ts

        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of ts input features. "
                             f"Got type: {type(input_features)}")

        return Qwen3TSTimeSeriesInput(timeseries=input_features)

    def _process_ts_input(
            self, ts_input: Qwen3TSTimeSeriesInput) -> list[torch.Tensor]:
        """Process time series input and return a list of 2D tensors."""
        ts_features, patch_cnt = self.ts_encoder(ts_input["timeseries"])

        # Reshape ts_features into a list of 2D tensors
        if ts_features.size(0) > 0:
            features_list = []
            start_idx = 0
            for count in patch_cnt:
                if count > 0:
                    end_idx = start_idx + count
                    features_list.append(ts_features[start_idx:end_idx])
                    start_idx = end_idx
                else:
                    # Add empty tensor for consistency when count is 0
                    # This ensures consistent behavior with prefix caching
                    features_list.append(
                        torch.zeros((0, ts_features.size(1)),
                                    device=ts_features.device,
                                    dtype=ts_features.dtype))
            return features_list
        else:
            return []

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        ts_input = self._parse_and_validate_ts_input(**kwargs)
        if ts_input is None:
            return None

        return self._process_ts_input(ts_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.ts_token_start_index)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            ts_features = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, ts_features)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
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

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)

        autoloaded_weights = loader.load_weights(weights,
                                                 mapper=self.hf_to_vllm_mapper)

        # The HF config doesn't specify whether these are tied,
        # so we detect it this way
        if "embed_tokens.weight" not in autoloaded_weights:
            self.embed_tokens = self.language_model.model.embed_tokens
            autoloaded_weights.add("embed_tokens.weight")

        return autoloaded_weights
