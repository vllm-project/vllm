# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.cohere_reward import CohereRewardConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import (
    SequencePooler,
    SequencePoolerHead,
    SequencePoolerHeadOutput,
    SequencePoolingMethodOutput,
    get_seq_pooling_method,
)
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify
from vllm.sequence import IntermediateTensors
from vllm.v1.pool.metadata import PoolingMetadata

from .cohere2_vision import (
    Cohere2VisionForConditionalGeneration,
)
from .interfaces import default_pooling_type
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


class CohereRewardPoolerHead(SequencePoolerHead):
    """Identity pooler head for reward models.

    Returns pooled data as-is without any transformation.
    """

    def get_supported_tasks(self):
        return {"token_classify", "classify", "score"}

    def forward(
        self,
        pooled_data: SequencePoolingMethodOutput,
        pooling_metadata: PoolingMetadata,
    ) -> SequencePoolerHeadOutput:
        return pooled_data


@default_pooling_type("LAST")
class Cohere2VisionForRewardModel(Cohere2VisionForConditionalGeneration):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        delattr(self.language_model, "logits_processor")

        config = vllm_config.model_config.hf_config
        self.ranking = RowParallelLinear(
            config.text_config.hidden_size,
            1,
            bias=False,
            input_is_parallel=False,
            prefix=maybe_prefix(prefix, "ranking"),
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        if getattr(config, "pooling_type", "LAST") == "ALL":
            self.pooler = DispatchPooler(
                {"token_classify": pooler_for_token_classify(pooler_config)},
            )
        else:
            pooler = SequencePooler(
                pooling=get_seq_pooling_method(pooler_config.get_seq_pooling_type()),
                head=CohereRewardPoolerHead(),
            )
            self.pooler = DispatchPooler(
                {"token_classify": pooler},
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )
        logits, _ = self.ranking(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["ranking.bias"])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


# UNIFIED IMPLEMENTATION
# TODO: Resolve the MULTIMODAL_REGISTRY
# @MULTIMODAL_REGISTRY.register_processor(
#     Cohere2VisionMultiModalProcessor,
#     info=Cohere2VisionProcessingInfo,
#     dummy_inputs=Cohere2VisionDummyInputsBuilder)
@default_pooling_type("LAST")
class CohereForRewardModel(nn.Module):
    is_pooling_model = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: CohereRewardConfig = vllm_config.model_config.hf_config
        self.config = config
        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.base_config,
            prefix=maybe_prefix(prefix, "model"),
            architectures=config.base_config.architectures,
        )
        self.is_vision = (
            config.base_config.architectures[0]
            == "Cohere2VisionForConditionalGeneration"
        )

        # vision model base
        if self.is_vision:
            delattr(self.model.language_model, "logits_processor")
            hidden_size = config.base_config.text_config.hidden_size
            self.hf_to_vllm_mapper = WeightsMapper(
                orig_to_new_prefix={
                    "model.language_model.": "model.language_model.model.",
                }
            )
        else:
            delattr(self.model, "logits_processor")
            hidden_size = config.base_config.hidden_size
            self.hf_to_vllm_mapper = WeightsMapper(
                orig_to_new_prefix={
                    "model.": "model.model.",
                }
            )

        self.ranking = RowParallelLinear(
            hidden_size,
            1,
            bias=False,
            input_is_parallel=False,
            prefix=maybe_prefix(prefix, "ranking"),
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        if getattr(config, "pooling_type", "LAST") == "ALL":
            self.pooler = DispatchPooler(
                {"token_classify": pooler_for_token_classify(pooler_config)},
            )
        else:
            pooler = SequencePooler(
                pooling=get_seq_pooling_method(pooler_config.get_seq_pooling_type()),
                head=CohereRewardPoolerHead(),
            )
            self.pooler = DispatchPooler(
                {"token_classify": pooler},
            )

    # MULTIMODAL_REGISTRY
    # def get_multimodal_embeddings(self,
    #                               **kwargs: object) -> MultiModalEmbeddings:
    #     if self.is_vision:
    #         image_input = self.model._parse_and_validate_image_input(**kwargs)
    #         if image_input is None:
    #             return []

    #         return self.model._process_image_input(image_input, **kwargs)
    #     else:
    #         return None

    # def get_input_embeddings(
    #     self,
    #     input_ids: torch.Tensor,
    #     multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    # ) -> torch.Tensor:
    #     if self.is_vision:
    #         return self.model.get_input_embeddings(input_ids, multimodal_embeddings)
    #     else:
    #         return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        logits, _ = self.ranking(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["ranking.bias"])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
