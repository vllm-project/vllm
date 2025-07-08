# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping
from typing import Optional

import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
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

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense = ColumnParallelLinear(config.hidden_size,
                                          config.hidden_size,
                                          bias=True)
        self.out_proj = RowParallelLinear(config.hidden_size,
                                          config.num_labels,
                                          bias=True)

    def forward(self, x, **kwargs):
        x, _ = self.dense(x)
        x = torch.relu(x)
        x, _ = self.out_proj(x)
        return x


class JinaVLMultiModalProcessor(Qwen2VLMultiModalProcessor):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:

        # NOTE: We should reverse the order of the mm_data because the query prompt is
        # placed after the document prompt in the score template for JinaVLForRanking model,
        # but in mm_data they are stored in the opposite order (query first, then document).
        for _, value in mm_data.items():
            value.reverse()
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs,
                                          tok_kwargs)


@MULTIMODAL_REGISTRY.register_processor(JinaVLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class JinaVLForSequenceClassification(nn.Module, SupportsCrossEncoding,
                                      SupportsMultiModal):
    weight_mapper = WeightsMapper(
        orig_to_new_prefix={
            "score.0.": "score.dense.",
            "score.2.": "score.out_proj.",
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "qwen2_vl.language_model.model.",
            "visual.": "qwen2_vl.visual.",
            # mapping for original checkpoint
            "lm_head.": "qwen2_vl.language_model.lm_head.",
            "model.": "qwen2_vl.language_model.model.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        pooler_config = vllm_config.model_config.pooler_config

        self.LOGIT_BIAS = 2.65

        self.qwen2_vl = Qwen2VLForConditionalGeneration(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "qwen2_vl"))
        self.score = JinaVLScorer(config)

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.LAST,
            normalize=False,
            softmax=True)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"

        raise ValueError("Only image modality is supported")

    @classmethod
    def get_score_template(cls, query: str, document: str) -> Optional[str]:
        return f"**Document**:\n{document}\n**Query**:\n{query}"

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = self.qwen2_vl(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        logits = self.score(hidden_states) - self.LOGIT_BIAS
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.weight_mapper)
