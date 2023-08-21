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
import torch.nn.functional as F

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RMSNorm(torch.nn.Module):

    def __init__(self,
                 normalized_shape,
                 eps=1e-5,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1,
                                                               keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class GLMAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0
        self.num_attention_heads_per_partition = config.num_attention_heads // tp_size

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (self.projection_size +
                                    2 * self.hidden_size_per_attention_head *
                                    config.multi_query_group_num)
        self.query_key_value = ColumnParallelLinear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            gather_output=False,
            perform_initialization=False)
        rotary_dim = (config.hidden_size // config.num_attention_heads
                      if config.kv_channels is None else config.kv_channels)
        tp_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % tp_size == 0
        self.attn = PagedAttentionWithRoPE(
            self.num_attention_heads_per_partition,
            config.kv_channels,
            scale=config.kv_channels**-0.5,
            rotary_dim=rotary_dim // 2,
            max_position=config.seq_length,
            num_kv_heads=self.num_multi_query_groups_per_partition,
            glm=True)

        # Output.
        self.dense = RowParallelLinear(self.projection_size,
                                       config.hidden_size,
                                       bias=config.add_bias_linear,
                                       input_is_parallel=True,
                                       perform_initialization=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Attention heads [num_toknens, h] --> [num_toknens, (np * 3 * hn)]
        qkv, _ = self.query_key_value(hidden_states)
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = qkv.split(
                [
                    self.num_attention_heads_per_partition *
                    self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition *
                    self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition *
                    self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()
        else:
            query_layer, key_layer, value_layer = qkv.chunk(chunks=3, dim=-1)

        key_cache, value_cache = kv_cache
        context_layer = self.attn(position_ids, query_layer, key_layer,
                                  value_layer, key_cache, value_cache,
                                  input_metadata, cache_event)
        attn_output, _ = self.dense(context_layer)
        return attn_output


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config):
        super(GLMMLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width,
        #  see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = ColumnParallelLinear(config.hidden_size,
                                                  config.ffn_hidden_size * 2,
                                                  bias=config.add_bias_linear,
                                                  gather_output=False,
                                                  perform_initialization=False)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(config.ffn_hidden_size,
                                               config.hidden_size,
                                               bias=config.add_bias_linear,
                                               input_is_parallel=True,
                                               perform_initialization=False)

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
        super(GLMBlock, self).__init__()
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size,
                                             eps=config.layernorm_epsilon,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = GLMAttention(config)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.torch_dtype)

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
        super(GLMTransformer, self).__init__()
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        self.layers = nn.ModuleList(
            [GLMBlock(config) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size,
                                                 eps=config.layernorm_epsilon,
                                                 dtype=config.torch_dtype)

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
                                                config.hidden_size,
                                                perform_initialization=False)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config)

        self.output_layer = ColumnParallelLinear(
            config.hidden_size,
            config.padded_vocab_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            params_dtype=config.torch_dtype)

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
        hidden_states = self.encoder(hidden_states=inputs_embeds,
                                     position_ids=position_ids,
                                     kv_caches=kv_caches,
                                     input_metadata=input_metadata,
                                     cache_events=cache_events)

        return hidden_states


class ChatGLMForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
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
        "dense_h_to_4h", "query_key_value", "output_layer.weight",
        "embedding.weight"
    ]
    _row_parallel_weights = ["dense_4h_to_h", "dense.weight", 'dense.bias']

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):

        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "word_embeddings" in name:
                name = name.replace(".word_embeddings", "")

            if name in state_dict:
                load_tensor_parallel_weights(state_dict[name], loaded_weight,
                                             name,
                                             self._column_parallel_weights,
                                             self._row_parallel_weights,
                                             tensor_model_parallel_rank)
            else:
                print("Warning never found tensor's name:", name)
