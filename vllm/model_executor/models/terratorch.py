# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 IBM.
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
"""Wrapper around `Terratorch` models"""

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from terratorch.vllm import (
    DummyDataGenerator,
    InferenceRunner,
    InputDefinition,
    InputTypeEnum,
)
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import DispatchPooler, DummyPooler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import (
    ImageItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalKwargsItems,
    MultiModalUUIDDict,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from .interfaces import IsAttentionFree, MultiModalEmbeddings, SupportsMultiModal
from .interfaces_base import default_pooling_type

logger = init_logger(__name__)


def _terratorch_field_names(pretrained_cfg: dict):
    input_definition = InputDefinition(**pretrained_cfg["input"])
    return set(input_definition.data.keys())


def _terratorch_field_factory(
    pretrained_cfg: dict,
) -> Callable[
    [Mapping[str, torch.Tensor]],
    Mapping[str, MultiModalFieldConfig],
]:
    def _terratorch_field_config(hf_inputs: Mapping[str, torch.Tensor]):
        input_definition = InputDefinition(**pretrained_cfg["input"])
        fields = {}
        for input_name, input in input_definition.data.items():
            if input.type == InputTypeEnum.tensor:
                fields[input_name] = "image"

        return {
            field_name: MultiModalFieldConfig.batched(modality=field_modality)
            for field_name, field_modality in fields.items()
        }

    return _terratorch_field_config


class TerratorchProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class TerratorchInputBuilder(BaseDummyInputsBuilder[TerratorchProcessingInfo]):
    def __init__(self, info: TerratorchProcessingInfo):
        super().__init__(info)
        self.dummy_data_generator = DummyDataGenerator(
            self.info.get_hf_config().to_dict()["pretrained_cfg"]
        )

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        # Dummy data is generated based on the 'input' section
        # defined in the HF configuration file

        if mm_options:
            logger.warning(
                "Configurable multimodal profiling "
                "options are not supported for Terratorch. "
                "They are ignored for now."
            )

        return self.dummy_data_generator.get_dummy_mm_data()


class TerratorchMultiModalDataParser(MultiModalDataParser):
    def __init__(self, pretrained_cfg: dict, *args, **kwargs):
        self._pretrained_cfg = pretrained_cfg
        super().__init__(*args, **kwargs)

    def _parse_image_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[ImageItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            terratorch_fields = _terratorch_field_names(self._pretrained_cfg)

            return DictEmbeddingItems(
                data,
                modality="image",
                required_fields=terratorch_fields,
                fields_factory=_terratorch_field_factory(self._pretrained_cfg),
            )

        return super()._parse_image_data(data)


class TerratorchMultiModalProcessor(BaseMultiModalProcessor):
    def __init__(
        self,
        info: TerratorchProcessingInfo,
        dummy_inputs: "BaseDummyInputsBuilder[TerratorchProcessingInfo]",
        *,
        cache: MultiModalProcessorOnlyCache | None = None,
    ) -> None:
        self.pretrained_cfg = info.get_hf_config().to_dict()["pretrained_cfg"]
        super().__init__(info=info, dummy_inputs=dummy_inputs, cache=cache)

    def _get_data_parser(self) -> MultiModalDataParser:
        return TerratorchMultiModalDataParser(pretrained_cfg=self.pretrained_cfg)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _terratorch_field_factory(self.pretrained_cfg)(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    def apply(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        if "image" in mm_data:
            image_data = mm_data["image"]
            image_data = {k: v.unsqueeze(0) for k, v in image_data.items()}
        else:
            image_data = mm_data
            image_data = {k: v.unsqueeze(0) for k, v in image_data.items()}

        mm_data = {"image": image_data}

        mm_items = self._to_mm_items(mm_data)
        tokenization_kwargs = tokenization_kwargs or {}
        mm_hashes = self._hash_mm_items(
            mm_items, hf_processor_mm_kwargs, tokenization_kwargs, mm_uuids=mm_uuids
        )
        mm_placeholders = {"image": [PlaceholderRange(offset=0, length=0)]}

        mm_processed_data = BatchFeature(image_data)

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed_data,
            self._get_mm_fields_config(mm_processed_data, hf_processor_mm_kwargs),
        )

        return MultiModalInputs(
            type="multimodal",
            prompt_token_ids=[1],
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


@default_pooling_type("All")
@MULTIMODAL_REGISTRY.register_processor(
    TerratorchMultiModalProcessor,
    info=TerratorchProcessingInfo,
    dummy_inputs=TerratorchInputBuilder,
)
class Terratorch(nn.Module, IsAttentionFree, SupportsMultiModal):
    merge_by_field_config = True
    supports_multimodal_raw_input_only = True
    is_pooling_model = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None

        raise ValueError("Only image modality is supported")

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"]

        self.inference_runner = InferenceRunner(config)
        self.model = self.inference_runner.model

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler({"plugin": DummyPooler()})

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # We do not really use any input tokens and therefore no embeddings
        # to be calculated. However, due to the mandatory token ids in
        # the input prompt we pass one token and the size of the dummy
        # embedding tensors must reflect that.
        return torch.empty((input_ids.shape[0], 0))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        model_output = self.inference_runner.forward(**kwargs)

        return model_output.output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_list = []
        model_buffers = dict(self.named_buffers())
        loaded_buffers = []
        for key, value in weights:
            if isinstance(value, (dict, OrderedDict)):
                if key == "state_dict":
                    weights_to_parse = value
                    for name, weight in weights_to_parse.items():
                        name = f"inference_runner.{name}"

                        if "pos_embed" in name:
                            continue

                        if "_timm_module." in name:
                            name = name.replace("_timm_module.", "")

                        # this model requires a couple of buffers to be loaded
                        # that are not loadable with the AutoWeightsLoader
                        if name in model_buffers:
                            if "_timm_module." in name:
                                name = name.replace("_timm_module.", "")
                            buffer = model_buffers[name]
                            weight_loader = getattr(
                                buffer, "weight_loader", default_weight_loader
                            )
                            weight_loader(buffer, weight)
                            loaded_buffers.append(name)
                        else:
                            params_list.append((name, weight))
                    break

            elif isinstance(value, torch.Tensor):
                params_list.append((f"inference_runner.model.{key}", value))

        # Load the remaining model parameters
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(params_list)

        return autoloaded_weights.union(set(loaded_buffers))
