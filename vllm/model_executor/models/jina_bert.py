# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from jinaai/jina-bert-implementation (Apache-2.0).

import torch
from torch import nn
from transformers import BertConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from .bert import BertAttention, BertEncoder, BertModel, _decode_token_type_ids
from .bloom import _get_alibi_slopes
from .interfaces_base import default_pooling_type
from .utils import WeightsMapper


class JinaBertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.token_type_embeddings = VocabParallelEmbedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_type_ids = _decode_token_type_ids(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return self.LayerNorm(
            inputs_embeds + self.token_type_embeddings(token_type_ids)
        )


class JinaBertMLP(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.gated_layers = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gated_layers",
        )
        self.act = GeluAndMul()
        self.wo = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wo",
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.gated_layers(hidden_states)
        hidden_states, _ = self.wo(self.act(hidden_states))
        return self.layernorm(hidden_states + residual)


class JinaBertLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        slopes = _get_alibi_slopes(config.num_attention_heads)
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        slopes = slopes.chunk(tp_size)[tp_rank].tolist()
        self.attention = BertAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attention",
            alibi_slopes=slopes,
        )
        self.mlp = JinaBertMLP(vllm_config, prefix=f"{prefix}.mlp")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attention(hidden_states))


class JinaBertEncoder(BertEncoder):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.layer = nn.ModuleList(
            JinaBertLayer(vllm_config, prefix=f"{prefix}.layer.{index}")
            for index in range(config.num_hidden_layers)
        )


@support_torch_compile
@default_pooling_type(seq_pooling_type="MEAN")
class JinaBertModel(BertModel):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"bert.": "", "cls.": None, "pooler.": None},
        orig_to_new_stacked={
            ".self.query": (".self.qkv_proj", "q"),
            ".self.key": (".self.qkv_proj", "k"),
            ".self.value": (".self.qkv_proj", "v"),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        if self.config.position_embedding_type != "alibi":
            raise ValueError("JinaBertModel requires ALiBi position embeddings")
        if self.config.feed_forward_type != "geglu":
            raise ValueError("JinaBertModel requires GeGLU feed-forward layers")
        self.embeddings = JinaBertEmbeddings(self.config)
        self.encoder = JinaBertEncoder(vllm_config, prefix=f"{prefix}.encoder")
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None
        self.pooler = DispatchPooler.for_embedding(pooler_config)
