from typing import List, Optional

import torch
from torch import nn
from transformers import RobertaConfig

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.models.bert import BertEmbeddingModel, BertModel
from vllm.sequence import IntermediateTensors


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
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_shape = input_ids.size()

        # Input embeddings.
        inputs_embeds = self.word_embeddings(input_ids)

        # TODO: figure out if there is a better way
        # to make to make position ids start at padding_idx + 1
        # References:
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L133
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L1669
        position_ids += self.padding_idx + 1

        # Position embeddings.
        position_embeddings = self.position_embeddings(position_ids)

        # Token type embeddings. (TODO: move off hotpath?)
        token_type_embeddings = self.token_type_embeddings(
            torch.zeros(input_shape,
                        dtype=torch.long,
                        device=inputs_embeds.device))

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


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

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Verify assumption that position are always a sequence from
        # 0 to N. (Actually here we just check 0 and N to simplify).
        # This is important to fix the position which are assumed to
        # start from padding_idx + 1 instead of 0 in the Roberta models.
        assert hasattr(attn_metadata, "seq_lens_tensor")
        cumulative = attn_metadata.seq_lens_tensor.cumsum(dim=0)
        start_pos = torch.cat(
            (torch.tensor([0], device=attn_metadata.seq_lens_tensor.device),
             cumulative[:-1]))
        assert len(torch.nonzero(positions[start_pos])) == 0
        end_pos = cumulative - 1
        last_tokens = attn_metadata.seq_lens_tensor - 1
        assert len(torch.nonzero(positions[end_pos] - last_tokens)) == 0

        return super().forward(input_ids=input_ids,
                               positions=positions,
                               kv_caches=kv_caches,
                               attn_metadata=attn_metadata,
                               intermediate_tensors=intermediate_tensors,
                               inputs_embeds=inputs_embeds)
