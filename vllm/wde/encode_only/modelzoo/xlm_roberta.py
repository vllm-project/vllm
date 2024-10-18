# Derived from XLM-RoBERTa implementation posted on HuggingFace; license below:
# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch XLM-RoBERTa model."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import XLMRobertaConfig
from transformers.utils import logging

from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.wde.core.layers.attention import (Attention, AttentionBackend,
                                            AttentionMetadata)

logger = logging.get_logger(__name__)


class LoadWeightsMixin:

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "query", "q"),
            ("qkv_proj", "key", "k"),
            ("qkv_proj", "value", "v")
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if hasattr(self, "prefix"):
                name = self.prefix + name

            if name in self._ignore_weights_keys:
                continue

            if name == "roberta.embeddings.token_type_embeddings.weight":
                # token_type_ids is all zero,
                # so we only need token_type_embeddings[0]
                self.roberta.embeddings.init_token_type_embeddings0()
                default_weight_loader(
                    self.roberta.embeddings.token_type_embeddings0,
                    loaded_weight[0])
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

        if hasattr(self, "tie_weights"):
            self.tie_weights()


class XLMRobertaEmbeddings(nn.Module):

    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        self.config = config
        self.position_embedding_type = getattr(config,
                                               "position_embedding_type",
                                               "absolute")
        assert self.position_embedding_type == "absolute"

        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.token_type_embeddings0 = None
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=config.pad_token_id)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

    def init_token_type_embeddings0(self):
        del self.token_type_embeddings0
        self.register_buffer(
            "token_type_embeddings0",
            torch.zeros(self.config.hidden_size,
                        dtype=self.word_embeddings.weight.dtype,
                        device=self.word_embeddings.weight.device))

    def forward(self, input_ids, positions):
        embeddings = self.word_embeddings(input_ids)

        # token_type_embeddings is all zero in FacebookAI/xlm-roberta,
        # so we don't need it.
        # token_type_ids is all zero in BGEM3,
        # so we only need token_type_embeddings[0]
        if self.token_type_embeddings0 is not None:
            token_type_embeddings = self.token_type_embeddings0
            embeddings += token_type_embeddings

        embeddings += self.position_embeddings(positions)
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class XLMRobertaSelfAttention(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 attn_backend: AttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_attention_heads

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=True,
            quant_config=quant_config,
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              quant_config=quant_config,
                              attn_backend=attn_backend)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, attn_metadata)
        return attn_output


class XLMRobertaSelfOutput(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = ColumnParallelLinear(config.hidden_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XLMRobertaAttention(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 attn_backend: AttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.self = XLMRobertaSelfAttention(config, attn_backend)
        self.output = XLMRobertaSelfOutput(config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        self_outputs = self.self(hidden_states, attn_metadata)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class XLMRobertaIntermediate(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = RowParallelLinear(config.hidden_size,
                                       config.intermediate_size,
                                       bias=True,
                                       quant_config=quant_config)
        self.intermediate_act_fn = get_act_fn(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class XLMRobertaOutput(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = RowParallelLinear(config.intermediate_size,
                                       config.hidden_size,
                                       bias=True,
                                       quant_config=quant_config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XLMRobertaLayer(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 attn_backend: AttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.attention = XLMRobertaAttention(config, attn_backend,
                                             quant_config)
        self.intermediate = XLMRobertaIntermediate(config, quant_config)
        self.output = XLMRobertaOutput(config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attn_metadata)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class XLMRobertaEncoder(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 attn_backend: AttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.layer = nn.ModuleList([
            XLMRobertaLayer(config, attn_backend, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attn_metadata)
        return hidden_states


class XLMRobertaModel(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 attn_backend: AttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config, attn_backend, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        positions += self.config.pad_token_id + 1

        embedding_output = self.embeddings(
            input_ids=input_ids,
            positions=positions,
        )

        encoder_outputs = self.encoder(embedding_output, attn_metadata)

        return encoder_outputs


class XLMRobertaLMHead(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = ColumnParallelLinear(config.hidden_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        self.layer_norm = nn.LayerNorm(config.hidden_size,
                                       eps=config.layer_norm_eps)

        self.decoder = ColumnParallelLinear(config.hidden_size,
                                            config.vocab_size,
                                            quant_config=quant_config)
        self.gelu = get_act_fn("gelu")

    def forward(self, features):
        x, _ = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x, _ = self.decoder(x)
        return x


class XLMRobertaForMaskedLM(nn.Module, LoadWeightsMixin):
    _ignore_weights_keys = [
        "roberta.pooler.dense.weight",
        "roberta.pooler.dense.bias",
        # token_type_embeddings is all zero
        "roberta.embeddings.token_type_embeddings.weight"
    ]

    def __init__(self,
                 config: XLMRobertaConfig,
                 attn_backend: AttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None,
                 *args,
                 **kwargs):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.roberta = XLMRobertaModel(config, attn_backend, quant_config)
        self.lm_head = XLMRobertaLMHead(config, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        sequence_output = self.roberta(
            input_ids,
            positions,
            attn_metadata,
        )
        logits = self.lm_head(sequence_output)
        return logits

    def tie_weights(self):
        self.lm_head.decoder.weight = (
            self.roberta.embeddings.word_embeddings.weight)
        self.lm_head.decoder.bias.zero_()


class XLMRobertaClassificationHead(nn.Module):

    def __init__(self,
                 config: XLMRobertaConfig,
                 quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.dense = ColumnParallelLinear(config.hidden_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        self.out_proj = ColumnParallelLinear(config.hidden_size,
                                             config.num_labels,
                                             quant_config=quant_config)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x, _ = self.dense(features)
        x = torch.tanh(x)
        x, _ = self.out_proj(x)
        return x


class XLMRobertaForSequenceClassification(nn.Module, LoadWeightsMixin):
    _ignore_weights_keys = [
        "roberta.pooler.dense.weight", "roberta.pooler.dense.bias"
    ]

    def __init__(self,
                 config: XLMRobertaConfig,
                 attn_backend: AttentionBackend,
                 quant_config: Optional[QuantizationConfig] = None,
                 *args,
                 **kwargs):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(config, attn_backend, quant_config)
        self.classifier = XLMRobertaClassificationHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        sequence_output = self.roberta(
            input_ids,
            positions,
            attn_metadata,
        )

        seq_start_loc = attn_metadata.seq_start_loc

        # take <s> token (equiv. to [CLS])
        cls_features = sequence_output[seq_start_loc[:-1]]

        logits = self.classifier(cls_features)
        return logits
