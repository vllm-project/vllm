from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import BertConfig

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolerOutput


class BertEmbeddingModel(nn.Module):
    """A model that uses Bert to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    stacked_params_mapping = {
        "query": {
            "param_name": "qkv_proj",
            "shard_id": "q",
        },
        "key": {
            "param_name": "qkv_proj",
            "shard_id": "k",
        },
        "value": {
            "param_name": "qkv_proj",
            "shard_id": "v",
        },
    }

    params_mapping = {
        "beta": "bias",
        "gamma": "weight",
        "LayerNorm": "layernorm",
    }

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.base_model_prefix = "bert"
        self.model = BertModel(config=kwargs["config"],
                               cache_config=kwargs.get("cache_config", None),
                               quant_config=kwargs.get("quant_config", None))
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids,
                          position_ids=positions,
                          kv_caches=kv_caches,
                          inputs_embeds=inputs_embeds,
                          attn_metadata=attn_metadata)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        params_dict = dict(self.model.named_parameters())

        for name, loaded_weight in weights:
            name = self._rename_key(name)
            name, shard_id = self._rename_stacked_param(name)

            # Skip the specific downstream task weight.
            if name.startswith('cls.'):
                continue
            # use Pooler instead.
            if name.startswith('pooler.'):
                continue
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            if shard_id:
                weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader(param, loaded_weight)

    def _rename_key(self, key: str):
        prefix = f"{self.base_model_prefix}."
        key = key[len(prefix):] if key.startswith(prefix) else key

        for src, dst in self.params_mapping.items():
            key = key.replace(src, dst)

        return key

    def _rename_stacked_param(
        self,
        name: str,
    ) -> Tuple[str, Optional[str]]:
        for key, mapping in self.stacked_params_mapping.items():
            if key in name:
                name = name.replace(key, mapping["param_name"])
                return name, mapping["shard_id"]
        return name, None


class BertModel(nn.Module):

    def __init__(
        self,
        config: BertConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(config, cache_config, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids=input_ids,
                                        position_ids=position_ids,
                                        inputs_embeds=inputs_embeds)
        output = self.encoder(hidden_states, kv_caches, attn_metadata)
        return output


class BertEmbedding(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type != "absolute":
            raise ValueError("Only 'absolute' position_embedding_type" +
                             " is supported")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[0]

        # input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_length,
                                        dtype=torch.long,
                                        device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings += position_embeddings
        embeddings = self.layernorm(embeddings)
        return embeddings


class BertEncoder(nn.Module):

    def __init__(
        self,
        config: BertConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(config=config,
                      cache_config=cache_config,
                      quant_config=quant_config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        for i in range(len(self.layer)):
            layer = self.layer[i]
            hidden_states = layer(
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )
        return hidden_states


class BertLayer(nn.Module):

    def __init__(
        self,
        config: BertConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config=config,
                                       cache_config=cache_config,
                                       quant_config=quant_config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            kv_cache,
            attn_metadata,
        )

        output = self.feed_forward(self_attention_outputs)
        return output

    def feed_forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):

    def __init__(
        self,
        config: BertConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.self = BertSelfAttention(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        self_outputs = self.self(hidden_states, kv_cache, attn_metadata)
        attn_output = self.output(self_outputs[0], hidden_states)
        return attn_output


class BertSelfAttention(nn.Module):

    def __init__(
        self,
        config: BertConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        assert self.head_dim * self.total_num_heads == self.hidden_size

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        output = self.attn(q,
                           k,
                           v,
                           kv_cache,
                           attn_metadata,
                           attn_type=AttentionType)
        return output


class BertSelfOutput(nn.Module):

    def __init__(self, config: BertConfig):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states
