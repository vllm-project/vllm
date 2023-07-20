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
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class GLMAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias
                                         )
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.attn = PagedAttentionWithRoPE(config.num_attention_heads,
                                   config.kv_channels ,
                                   scale=config.kv_channels**-0.5,
                                   rotary_dim = rotary_dim // 2,
                                   max_position = config.seq_length,
                                   glm = True)

        # Output.
        self.dense = nn.Linear(self.projection_size,
                                config.hidden_size, 
                                bias=config.add_bias_linear
                               )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Attention heads [num_toknens, h] --> [num_toknens, (np * 3 * hn)]
        qkv = self.query_key_value(hidden_states)
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = qkv.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.contiguous()
            head_nums = self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition
            num_tokens = key_layer.shape[0]
            key_layer = key_layer.reshape(-1, self.num_multi_query_groups_per_partition, 
                                        self.hidden_size_per_attention_head
                                        ).unsqueeze(-2).expand(-1,-1,head_nums,-1).contiguous().view(num_tokens,-1)
            value_layer = value_layer.reshape(-1, self.num_multi_query_groups_per_partition, 
                                            self.hidden_size_per_attention_head
                                            ).unsqueeze(-2).expand(-1,-1,head_nums,-1).contiguous().view(num_tokens,-1)
        else:
            query_layer, key_layer, value_layer = qkv.chunk(chunks=3, dim=-1)

        key_cache, value_cache = kv_cache
        context_layer = self.attn(position_ids, query_layer, key_layer, value_layer, key_cache, value_cache,
                                input_metadata, cache_event)
        attn_output = self.dense(context_layer)
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

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config, ):
        super(GLMBlock, self).__init__()
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = GLMAttention(config)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon,
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
        self.layers = nn.ModuleList([GLMBlock(config) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon,
                                                 dtype=config.torch_dtype)

    def forward(
            self, 
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            cache_events: Optional[List[torch.cuda.Event]],
    )->torch.Tensor:
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


class Embedding(nn.Module):
    """Language model embeddings."""

    def __init__(self, config, ):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        # embeddings = embeddings.transpose(0, 1).contiguous()
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = Embedding(config)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config)
        self.output_layer = nn.Linear(config.hidden_size,
                                      config.padded_vocab_size, 
                                      bias=False,
                                      dtype=config.torch_dtype
                                    )
        # TODO(fengxi) promptlearning 
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    # def get_prompt(self, batch_size, device=None, dtype=torch.half):
    #     prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
    #     .unsqueeze(0).expand(batch_size, -1)
    #     past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
    #     past_key_values = past_key_values.view(
    #         batch_size,
    #         self.pre_seq_len,
    #         self.num_layers * 2,
    #         self.multi_query_group_num,
    #         self.kv_channels
    #     )
    #     # seq_len, b, nh, hidden_size
    #     past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
    #     return past_key_values

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
            cache_events=cache_events
        )

        return hidden_states


class ChatGLMForConditionalGeneration(nn.Module):

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

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache):
            if name in state_dict:
                state_dict[name].copy_(loaded_weight)
            else:
                print("Warning never found tensor's name:", name)

