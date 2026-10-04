# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn
from transformers import DistilBertConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import EncoderOnlyAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise import get_seq_pooling_method
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsCrossEncoding, SupportsQuant
from .interfaces_base import default_pooling_type
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix


class DistilBertEmbedding(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.size = config.dim
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size, config.dim)
        self.position_embeddings = VocabParallelEmbedding(
            config.max_position_embeddings, config.dim
        )
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class DistilBertAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_attention_heads
        assert self.total_num_heads % tp_size == 0

        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = hidden_size // self.total_num_heads
        assert self.head_dim * self.total_num_heads == hidden_size

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_lin = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_lin",
        )
        self.attn = EncoderOnlyAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.out_lin(attn_output)
        return output


class DistilBertFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.lin1 = ColumnParallelLinear(
            input_size=dim,
            output_size=hidden_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.lin1",
        )
        self.lin2 = RowParallelLinear(
            input_size=hidden_dim,
            output_size=dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.lin2",
        )
        self.activation = get_act_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lin1(x)
        x = self.activation(x)
        x, _ = self.lin2(x)
        return x


class DistilBertLayer(nn.Module):
    def __init__(
        self,
        config: DistilBertConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attention = DistilBertAttention(
            hidden_size=config.dim,
            num_attention_heads=config.n_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.sa_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.ffn = DistilBertFFN(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            activation=config.activation,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.output_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(hidden_states)
        attn_output = self.sa_layer_norm(attn_output + hidden_states)
        ffn_output = self.ffn(attn_output)
        ffn_output = self.output_layer_norm(ffn_output + attn_output)
        return ffn_output


class DistilBertTransformer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layer = nn.ModuleList(
            [
                DistilBertLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layer.{i}",
                )
                for i in range(config.n_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


@support_torch_compile
@default_pooling_type(seq_pooling_type="CLS")
class DistilBertModel(nn.Module, SupportsQuant):
    is_pooling_model = True

    packed_modules_mapping = {"qkv_proj": ["q_lin", "k_lin", "v_lin"]}

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".attention.q_lin": (".attention.qkv_proj", "q"),
            ".attention.k_lin": (".attention.qkv_proj", "k"),
            ".attention.v_lin": (".attention.qkv_proj", "v"),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.embeddings = DistilBertEmbedding(self.config)
        self.transformer = DistilBertTransformer(
            vllm_config=vllm_config, prefix=f"{prefix}.transformer"
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.word_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=positions,
            inputs_embeds=inputs_embeds,
        )
        return self.transformer(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class DistilBertClassificationHead(nn.Module):
    def __init__(self, config: DistilBertConfig, head_dtype: torch.dtype | None):
        super().__init__()
        self.pre_classifier = nn.Linear(config.dim, config.dim, dtype=head_dtype)
        self.classifier = nn.Linear(config.dim, config.num_labels, dtype=head_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF also applies dropout here, but it is a no-op at inference time.
        x = self.pre_classifier(x)
        x = torch.relu(x)
        return self.classifier(x)


@default_pooling_type(seq_pooling_type="CLS")
class DistilBertForSequenceClassification(
    nn.Module, SupportsCrossEncoding, SupportsQuant
):
    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: DistilBertConfig = vllm_config.model_config.hf_config
        head_dtype = vllm_config.model_config.head_dtype

        self.distilbert = DistilBertModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "distilbert"),
        )
        self.head = DistilBertClassificationHead(config, head_dtype)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler.for_seq_cls(
            pooler_config,
            pooling=get_seq_pooling_method("CLS"),
            classifier=self.head,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.distilbert.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # pre_classifier.* and classifier.* in the checkpoint are top-level
        # (not under distilbert.*) and must be routed into self.head.
        mapper = WeightsMapper(
            orig_to_new_prefix={
                "pre_classifier.": "head.pre_classifier.",
                "classifier.": "head.classifier.",
            }
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distilbert(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
