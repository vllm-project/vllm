# coding=utf-8
# Adapted from
# https://github.com/THUDM/ChatGLM2-6B
"""Inference-only ChatGLM model compatible with THUDM weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import LayerNorm

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (
    hf_model_weights_iterator,
    load_tensor_parallel_weights,
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.parallel_utils.layers import VocabParallelEmbedding
from vllm.model_executor.parallel_utils.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.sequence import SequenceOutputs

from vllm.transformers_utils.configs import ChatGLMConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class GLMAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = (config.multi_query_group_num
                                   if config.multi_query_attention else
                                   config.num_attention_heads)
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = ColumnParallelLinear(
            config.hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) *
            self.head_dim,
            bias=config.add_qkv_bias,
            gather_output=False,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.add_bias_linear,
            input_is_parallel=True,
        )

        self.attn = PagedAttentionWithRoPE(
            self.num_heads,
            self.head_dim,
            self.scaling,
            rotary_dim=self.head_dim // 2,
            num_kv_heads=self.num_kv_heads,
            is_neox_style=False,
            # is_glm_style=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        key_cache, value_cache = kv_cache

        context_layer = self.attn(
            position_ids,
            q,
            k,
            v,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )

        attn_output, _ = self.dense(context_layer)

        return attn_output


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=config.add_bias_linear,
            gather_output=False,
        )

        self.activation_func = SiluAndMul()

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            input_is_parallel=True,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)

        self.fp32_residual_connection = config.fp32_residual_connection

        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size,
                                               eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = GLMAttention(config)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = GLMMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.self_attention(
            hidden_states=layernorm_output,
            position_ids=position_ids,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = residual + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.mlp(layernorm_output) + residual

        return output


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super().__init__()
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        self.layers = nn.ModuleList(
            [GLMBlock(config) for i in range(self.num_layers)])

        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(
                config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_caches[i],
                input_metadata=input_metadata,
                cache_event=cache_event,
            )
        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class ChatGLMModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embedding = VocabParallelEmbedding(config.padded_vocab_size,
                                                config.hidden_size)

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config)

        self.output_layer = ColumnParallelLinear(
            config.hidden_size,
            config.padded_vocab_size,
            bias=False,
            gather_output=False,
            params_dtype=config.torch_dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ):
        inputs_embeds = self.embedding(input_ids)

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states=inputs_embeds,
            position_ids=position_ids,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )

        return hidden_states


class ChatGLMForCausalLM(nn.Module):

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.config: ChatGLMConfig = config
        self.transformer = ChatGLMModel(config)
        self.lm_head_weight = self.transformer.output_layer.weight
        self.sampler = Sampler(config.padded_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = [
        "output_layer.weight",
        "embedding.weight",
    ]
    _row_parallel_weights = ["dense_4h_to_h", "self_attention.dense"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        q_proj_shard_size = self.config.hidden_size // tp_size
        kv_proj_shard_size = (self.config.hidden_size //
                              self.config.num_attention_heads *
                              self.config.multi_query_group_num // tp_size)

        mlp_hidden_shard_size = self.config.ffn_hidden_size // tp_size

        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if "word_embeddings" in name:
                name = name.replace(".word_embeddings", "")

            if name in state_dict:
                param = state_dict[name]
                if "query_key_value" in name:
                    q_offset = q_proj_shard_size * tp_rank
                    k_offset = (q_proj_shard_size * tp_size +
                                kv_proj_shard_size * tp_rank)
                    v_offset = (q_proj_shard_size * tp_size +
                                kv_proj_shard_size * (tp_size + tp_rank))
                    wq = loaded_weight[q_offset:q_offset + q_proj_shard_size]
                    wk = loaded_weight[k_offset:k_offset + kv_proj_shard_size]
                    wv = loaded_weight[v_offset:v_offset + kv_proj_shard_size]
                    loaded_weight = torch.cat([wq, wk, wv], dim=0)
                    param.data.copy_(loaded_weight)
                    continue

                if "dense_h_to_4h" in name:
                    w_gate = loaded_weight[mlp_hidden_shard_size *
                                           tp_rank:mlp_hidden_shard_size *
                                           (tp_rank + 1)]
                    w_proj = loaded_weight[mlp_hidden_shard_size *
                                           (tp_size +
                                            tp_rank):mlp_hidden_shard_size *
                                           (tp_size + tp_rank + 1)]
                    loaded_weight = torch.cat([w_gate, w_proj], dim=0)
                    param.data.copy_(loaded_weight)
                    continue

                load_tensor_parallel_weights(
                    param,
                    loaded_weight,
                    name,
                    self._column_parallel_weights,
                    self._row_parallel_weights,
                    tp_rank,
                )
            elif name == "transformer.rotary_pos_emb.inv_freq":
                continue
            else:
                print("Warning never found tensor's name:", name)
