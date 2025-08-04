# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import (ClassifierPooler, CLSPool,
                                               DispatchPooler, Pooler)
from vllm.model_executor.models.bert import BertEmbeddingModel, BertModel
from vllm.model_executor.models.roberta import (RobertaClassificationHead,
                                                RobertaEmbedding,
                                                replace_roberta_positions)
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              maybe_prefix)
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsCrossEncoding


class XLMRobertaEmbeddingModel(BertEmbeddingModel):
    """A model that uses XLM-RoBERTa to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.padding_idx = vllm_config.model_config.hf_config.pad_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Fix RoBERTa positions here outside of the CUDA graph.
        # Because we need the to extract the sequences from
        # input_ids the control flow is data dependent.
        replace_roberta_positions(input_ids=input_ids,
                                  position_ids=positions,
                                  padding_idx=self.padding_idx)

        return self.model(input_ids=input_ids,
                          position_ids=positions,
                          token_type_ids=token_type_ids,
                          inputs_embeds=inputs_embeds,
                          intermediate_tensors=intermediate_tensors)

    def _build_model(self,
                     vllm_config: VllmConfig,
                     prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config,
                         prefix=prefix,
                         embedding_class=RobertaEmbedding)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights_list = list(weights)
        has_roberta_prefix = any(
            name.startswith("roberta.") for name, _ in weights_list)
        if has_roberta_prefix:
            # For models with the `roberta.` prefix
            mapper = WeightsMapper(orig_to_new_prefix={"roberta.": "model."})
        else:
            # For models without the `roberta.` prefix
            mapper = WeightsMapper(orig_to_new_prefix={"": "model."})

        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head."])
        return loader.load_weights(weights_list, mapper=mapper)


class XLMRobertaForSequenceClassification(nn.Module, SupportsCrossEncoding):
    """A model that uses XLM-RoBERTa to provide sequence classification functionality.

   This class encapsulates the BertModel and provides an interface for
   classification operations and cross-encoding.

   Attributes:
       roberta: An instance of BertModel used for forward operations.
       pooler: An instance of DispatchPooler used for pooling operations.
   """

    is_pooling_model = True
    # Weight mapping for Jina-style checkpoints that might be used with XLM-RoBERTa
    jina_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            'emb_ln': "embeddings.LayerNorm",
            'layers': "layer",
            'mixer.Wqkv': "attention.self.qkv_proj",
            'mixer.out_proj': "attention.output.dense",
            'norm1': "attention.output.LayerNorm",
            'mlp.fc1': "intermediate.dense",
            'mlp.fc2': "output.dense",
            'norm2': "output.LayerNorm",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx = vllm_config.model_config.hf_config.pad_token_id

        self.num_labels = config.num_labels
        self.roberta = BertModel(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "bert"),
                                 embedding_class=RobertaEmbedding)
        self.classifier = RobertaClassificationHead(config)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler({
            "encode":
            Pooler.for_encode(pooler_config),
            "classify":
            ClassifierPooler(
                pooling=CLSPool(),
                classifier=self.classifier,
                act_fn=ClassifierPooler.act_fn_for_seq_cls(
                    vllm_config.model_config),
            ),
            "score":
            ClassifierPooler(
                pooling=CLSPool(),
                classifier=self.classifier,
                act_fn=ClassifierPooler.act_fn_for_cross_encoder(
                    vllm_config.model_config),
            ),
        })

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.jina_to_vllm_mapper)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        replace_roberta_positions(input_ids=input_ids,
                                  position_ids=positions,
                                  padding_idx=self.padding_idx)
        return self.roberta(input_ids=input_ids,
                            position_ids=positions,
                            inputs_embeds=inputs_embeds,
                            intermediate_tensors=intermediate_tensors,
                            token_type_ids=token_type_ids)
