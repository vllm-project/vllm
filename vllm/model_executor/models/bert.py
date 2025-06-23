# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn
from transformers import BertConfig

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, PoolerConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import (ClassifierPooler, Pooler,
                                               PoolingType)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.transformers_utils.config import (
    get_cross_encoder_activation_function)

from .interfaces import SupportsCrossEncoding, SupportsQuant, SupportsV0Only
from .utils import WeightsMapper, maybe_prefix


class BertEmbedding(nn.Module):

    def __init__(self, config: BertConfig):

        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        self.position_embeddings = VocabParallelEmbedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = VocabParallelEmbedding(
            config.type_vocab_size, config.hidden_size)
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

        # Input embeddings.
        inputs_embeds = self.word_embeddings(input_ids)

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


class BertPooler(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@support_torch_compile
class BertEncoder(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layer = nn.ModuleList([
            BertLayer(config=config,
                      cache_config=cache_config,
                      quant_config=quant_config,
                      prefix=f"{prefix}.layer.{layer_idx}")
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self,
                 config: BertConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.attention = BertAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention")

        self.intermediate = BertIntermediate(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.intermediate")

        self.output = BertOutput(hidden_size=config.hidden_size,
                                 intermediate_size=config.intermediate_size,
                                 layer_norm_eps=config.layer_norm_eps,
                                 quant_config=quant_config,
                                 prefix=f"{prefix}.output")

    def forward(self, hidden_states: torch.Tensor):
        attn_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attn_output)
        output = self.output(intermediate_output, attn_output)
        return output


class BertAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.self = BertSelfAttention(hidden_size=hidden_size,
                                      num_attention_heads=num_attention_heads,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.output")

        self.output = BertSelfOutput(hidden_size=hidden_size,
                                     layer_norm_eps=layer_norm_eps,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.output")

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self_output = self.self(hidden_states)
        return self.output(self_output, hidden_states)


class BertSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_attention_heads
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
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj")

        self.attn = Attention(num_heads=self.num_heads,
                              head_size=self.head_dim,
                              scale=self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=AttentionType.ENCODER_ONLY)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        output = self.attn(q, k, v)
        return output


class BertSelfOutput(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 layer_norm_eps: float,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.dense = RowParallelLinear(input_size=hidden_size,
                                       output_size=hidden_size,
                                       bias=True,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.dense")
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()
        self.dense = ColumnParallelLinear(input_size=hidden_size,
                                          output_size=intermediate_size,
                                          bias=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.dense")
        self.intermediate_act_fn = get_act_fn(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 layer_norm_eps: float,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__()

        self.dense = RowParallelLinear(input_size=intermediate_size,
                                       output_size=hidden_size,
                                       bias=True,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.dense")

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertModel(nn.Module, SupportsQuant):
    packed_modules_mapping = {"qkv_proj": ["query", "key", "value"]}

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 embedding_class: type = BertEmbedding,
                 add_pooling_layer: bool = False):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embeddings = embedding_class(config)
        self.encoder = BertEncoder(vllm_config=vllm_config,
                                   prefix=f"{prefix}.encoder")
        self.pooler = BertPooler(config) if add_pooling_layer else None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            attn_metadata = get_forward_context().attn_metadata
            assert hasattr(attn_metadata, "seq_lens_tensor")
            hidden_states = self.embeddings(
                input_ids=input_ids,
                seq_lens=attn_metadata.seq_lens_tensor,
                position_ids=position_ids,
                token_type_ids=token_type_ids)
        return self.encoder(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.pooler is None and "pooler" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class BertEmbeddingModel(nn.Module, SupportsV0Only, SupportsQuant):
    """A model that uses Bert to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        pooler_config = vllm_config.model_config.pooler_config
        self.model = self._build_model(vllm_config=vllm_config,
                                       prefix=maybe_prefix(prefix, "model"))
        self._pooler = self._build_pooler(pooler_config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids,
                          position_ids=positions,
                          inputs_embeds=inputs_embeds,
                          intermediate_tensors=intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        self.model.load_weights(weights)

    def _build_model(self,
                     vllm_config: VllmConfig,
                     prefix: str = "") -> BertModel:
        return BertModel(vllm_config=vllm_config,
                         prefix=prefix,
                         embedding_class=BertEmbedding)

    def _build_pooler(self, pooler_config: PoolerConfig) -> Pooler:
        return Pooler.from_config_with_defaults(pooler_config,
                                                pooling_type=PoolingType.CLS,
                                                normalize=True,
                                                softmax=False)


class BertForSequenceClassification(nn.Module, SupportsV0Only,
                                    SupportsCrossEncoding, SupportsQuant):
    """A model that uses Bert to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.default_activation_function = \
            get_cross_encoder_activation_function(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(vllm_config=vllm_config,
                              prefix=maybe_prefix(prefix, "bert"),
                              embedding_class=BertEmbedding,
                              add_pooling_layer=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self._pooler = ClassifierPooler(vllm_config.model_config,
                                        self.classifier, self.bert.pooler)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        self_weights = []

        def weight_filter():
            for name, weight in weights:
                if name.startswith("bert."):
                    yield (name[len("bert."):], weight)
                else:
                    self_weights.append((name, weight))

        self.bert.load_weights(weight_filter())

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in self_weights:
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
        return self.bert(input_ids=input_ids,
                         position_ids=positions,
                         inputs_embeds=inputs_embeds,
                         intermediate_tensors=intermediate_tensors,
                         token_type_ids=token_type_ids)
