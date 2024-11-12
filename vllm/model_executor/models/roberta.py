from typing import Optional

import torch
from torch import nn
from transformers import RobertaConfig

from vllm.config import CacheConfig, PoolerConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.models.bert import (BertEmbedding, BertEmbeddingModel,
                                             BertEncoder, BertModel)


class RobertaModel(BertModel):

    def __init__(
        self,
        config: RobertaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        # Skip BertModel.__init__()
        nn.Module.__init__(self)
        self.embeddings = RobertaEmbedding(config)
        self.encoder = BertEncoder(config, cache_config, quant_config)


class RobertaEmbedding(BertEmbedding):

    def __init__(self, config: RobertaConfig):
        # Skip BertEmbedding.__init__()
        nn.Module.__init__(self)
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


class RobertaEmbeddingModel(BertEmbeddingModel):
    """A model that uses Roberta to provide embedding functionalities.

   This class encapsulates the RobertaModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of RobertaModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(self,
                 config: RobertaConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 pooler_config: Optional[PoolerConfig] = None) -> None:
        nn.Module.__init__(self)
        self.model = RobertaModel(config, cache_config, quant_config)
        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.CLS,
            normalize=True,
            softmax=False)

    def _build_model(self,
                     config: RobertaConfig,
                     cache_config: Optional[CacheConfig] = None,
                     quant_config: Optional[QuantizationConfig] = None):
        return BertModel(config, cache_config, quant_config, RobertaEmbedding)
