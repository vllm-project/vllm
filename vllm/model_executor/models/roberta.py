# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import RobertaConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import CrossEncodingPooler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bert import BertEmbeddingModel, BertModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.transformers_utils.config import (
    get_cross_encoder_activation_function)

from .interfaces import SupportsCrossEncoding


def roberta_task_weights_filter(
    all_weights: Iterable[Tuple[str, torch.Tensor]]
) -> Tuple[Iterable[Tuple[str, torch.Tensor]], Iterable[Tuple[str,
                                                              torch.Tensor]]]:
    """
    Separate task-specific weights that are applied on top
    of the encoder-decoder bert base.
    To do so, return two generators over the original iterator.
    Also, remove the "roberta." prefix to make it loadable
    from vanilla BertModel.
    """
    # Copy of a lazy iterator without in-memory overhead so both
    # iterators can be iterated upon independently.
    all_weights1, all_weights2 = itertools.tee(all_weights)

    def encoder_decoder_weights():
        for name, weight in all_weights1:
            if name.startswith("roberta."):
                yield (name[len("roberta."):], weight)

    return encoder_decoder_weights(), ((n, w) for n, w in all_weights2
                                       if not n.startswith("roberta."))


class RobertaEmbedding(nn.Module):

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size,
                                                padding_idx=self.padding_idx)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.position_ids = nn.Parameter(
            torch.empty((1, config.max_position_embeddings)), )

        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type != "absolute":
            raise ValueError("Only 'absolute' position_embedding_type" +
                             " is supported")

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        position_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()
        inputs_embeds = self.word_embeddings(input_ids)

        # Replace position ids because in RoBERTa models
        # they have to start at padding_idx + 1 and ignore
        # existing padding tokens
        # References:
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L133
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L1669
        pos_list = []
        token_list = []
        offset = 0
        for seq_len in seq_lens:
            pos_list.append(position_ids[offset:offset + seq_len])
            token_list.append(input_ids[offset:offset + seq_len])
            offset += seq_len

        new_pos_list = []
        for positions, tokens in zip(pos_list, token_list):
            # Verify assumption that incoming position are
            # always a sequence from 0 to N.
            expected_pos = torch.arange(positions.size()[0],
                                        dtype=torch.long,
                                        device=inputs_embeds.device)
            assert torch.equal(positions, expected_pos)
            new_pos_list.append(
                create_position_ids_from_input_ids(tokens, self.padding_idx))
        position_ids = torch.cat(new_pos_list)

        # Position embeddings.
        position_embeddings = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=inputs_embeds.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


# Adapted from transformers
def create_position_ids_from_input_ids(input_ids,
                                       padding_idx,
                                       past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()

    incremental_indices = (torch.cumsum(mask, dim=0).type_as(mask) +
                           past_key_values_length) * mask

    return incremental_indices.long() + padding_idx


# Adapted from transformers
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class RobertaEmbeddingModel(BertEmbeddingModel):
    """A model that uses Roberta to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def _build_model(self,
                     vllm_config: VllmConfig,
                     prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config,
                         prefix=prefix,
                         embedding_class=RobertaEmbedding)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        # Separate weights in "roberta"-prefixed and all else (not in memory).
        # For use with models like FacebookAI/roberta-base.
        bert_weights, task_weights = roberta_task_weights_filter(weights)
        loaded = self.model.load_weights(bert_weights)
        if not len(loaded):
            # Fix for models like `sentence-transformers/stsb-roberta-base-v2`
            # which use the same architecture, but have no "roberta" prefix.
            loaded = self.model.load_weights(task_weights)
        assert len(loaded), "Unable to load RobertaEmbeddingModel"


class RobertaForSequenceClassification(nn.Module, SupportsCrossEncoding):
    """A model that uses Roberta to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       roberta: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.default_activation_function = \
            get_cross_encoder_activation_function(config)

        self.num_labels = config.num_labels
        self.roberta = BertModel(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "bert"),
                                 embedding_class=RobertaEmbedding,
                                 add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self._pooler = CrossEncodingPooler(config, self.classifier)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        bert_weights, task_weights = roberta_task_weights_filter(weights)
        self.roberta.load_weights(bert_weights)

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in task_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.roberta(input_ids=input_ids,
                            position_ids=positions,
                            inputs_embeds=inputs_embeds,
                            intermediate_tensors=intermediate_tensors,
                            token_type_ids=token_type_ids)
