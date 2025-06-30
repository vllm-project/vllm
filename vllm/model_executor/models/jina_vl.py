# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors, PoolerOutput

from .interfaces import SupportsCrossEncoding, SupportsMultiModal
from .qwen2_vl import (Qwen2VLDummyInputsBuilder,
                       Qwen2VLForConditionalGeneration,
                       Qwen2VLMultiModalProcessor, Qwen2VLProcessingInfo)
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

logger = init_logger(__name__)


class JinaVLScorer(nn.Module):

    # To ensure correct weight loading and mapping.
    score_mapper = WeightsMapper(orig_to_new_prefix={
        "score.0.": "dense.",
        "score.2.": "out_proj."
    })

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = torch.relu(x)
        x = self.out_proj(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.score_mapper)


@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class JinaVLForSequenceClassification(nn.Module, SupportsCrossEncoding,
                                      SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        pooler_config = vllm_config.model_config.pooler_config

        config.num_labels = 1
        self.LOGIT_BIAS = 2.65

        self.qwen2_vl = Qwen2VLForConditionalGeneration(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "qwen2_vl"))
        self.score = JinaVLScorer(config)

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.LAST,
            normalize=False,
            softmax=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.qwen2_vl(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        hidden_states = self._pooler.extract_states(hidden_states,
                                                    pooling_metadata)

        logits = self.score(hidden_states) - self.LOGIT_BIAS

        pooled_data = self._pooler.head(logits, pooling_metadata)

        pooled_outputs = [
            self._pooler.build_output(data.squeeze(-1)) for data in pooled_data
        ]
        return PoolerOutput(outputs=pooled_outputs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        qwen2vl_weights, score_weights = split_weights(weights)

        self.qwen2_vl.load_weights(qwen2vl_weights)

        self.score.load_weights(score_weights)


def split_weights(
    all_weights: Iterable[tuple[str, torch.Tensor]]
) -> tuple[Iterable[tuple[str, torch.Tensor]], Iterable[tuple[str,
                                                              torch.Tensor]]]:
    all_weights1, all_weights2 = itertools.tee(all_weights)

    return ((n, w) for n, w in all_weights1
            if not n.startswith("score.")), ((n, w) for n, w in all_weights2
                                             if n.startswith("score."))
