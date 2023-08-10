# coding=utf-8
# Adapted from
# https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py
"""Inference-only ChatGLM model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.

"""

import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.nn import LayerNorm
from typing import Optional, Tuple, List, Dict

from transformers.utils import logging

# vllm
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs

KVCache = Tuple[torch.Tensor, torch.Tensor]

# flags required to enable jit fusion kernels

if sys.platform != "darwin":
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm2-6b",
    # See all ChatGLM models at https://huggingface.co/models?filter=chatglm
]


def build_rotary_pos(seq_len,
                     n_elem,
                     dtype,
                     device,
                     base: int = 10000,
                     rope_ratio: int = 1):
    theta = 1.0 / (base**(
        torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device) / rope_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will
    # get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
    return cache


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor,
                         rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, np, hn]
    x = x.unsqueeze(1)
    sq, _, np, _ = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] -
            xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] +
            xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width,
        #  see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            gather_output=False,
            perform_initialization=False,
            params_dtype=config.torch_dtype)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(config.ffn_hidden_size,
                                               config.hidden_size,
                                               bias=self.add_bias,
                                               input_is_parallel=True,
                                               perform_initialization=False,
                                               params_dtype=config.torch_dtype)

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


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


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, layer_number):
        super().__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = (config.kv_channels *
                                config.num_attention_heads)

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = (self.projection_size //
                                               config.num_attention_heads)
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.norm_factor = self.hidden_size_per_attention_head**-0.5

        self.rope_ratio = 1 if ("rope_ratio"
                                not in config.to_dict()) else config.rope_ratio
        self.multi_query_attention = config.multi_query_attention

        self.seq_length = config.seq_length
        self.rotary_dim = (config.hidden_size // config.num_attention_heads if
                           config.kv_channels is None else config.kv_channels)

        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = (
                config.multi_query_group_num)
            self.qkv_hidden_size = (self.projection_size +
                                    2 * self.hidden_size_per_attention_head *
                                    config.multi_query_group_num)
        self.query_key_value = ColumnParallelLinear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            gather_output=False,
            perform_initialization=False,
            params_dtype=config.torch_dtype)

        # num_kv_heads=self.num_multi_query_groups_per_partition
        # if self.multi_query_attention else config.num_attention_heads
        self.atten = PagedAttention(config.num_attention_heads,
                                    self.hidden_size_per_attention_head,
                                    self.norm_factor)
        # Output.
        self.dense = RowParallelLinear(self.projection_size,
                                       config.hidden_size,
                                       bias=config.add_bias_linear,
                                       input_is_parallel=True,
                                       perform_initialization=False,
                                       params_dtype=config.torch_dtype)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
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
            query_layer = query_layer.contiguous().view(
                query_layer.size()[:-1] +
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head))
            key_layer = key_layer.contiguous().view(key_layer.size()[:-1] + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head))
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:-1] +
                (self.num_multi_query_groups_per_partition,
                 self.hidden_size_per_attention_head))
        else:
            new_tensor_shape = (mixed_x_layer.size()[:-1] +
                                (self.num_attention_heads_per_partition,
                                 3 * self.hidden_size_per_attention_head))
            mixed_x_layer = mixed_x_layer.contiguous().view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer,
             value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        k_cache, v_cache = kv_cache
        rotary_pos = build_rotary_pos(self.seq_length,
                                      self.rotary_dim // 2,
                                      dtype=query_layer.dtype,
                                      device=query_layer.device,
                                      rope_ratio=self.rope_ratio)
        if positions is not None:
            rotary_pos = rotary_pos[positions]
        else:
            rotary_pos = rotary_pos[:query_layer.shape[0]]

        rotary_pos = rotary_pos.unsqueeze(0).transpose(0, 1).contiguous()

        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos).reshape(
            -1, self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos).reshape(
            -1, self.num_multi_query_groups_per_partition,
            self.hidden_size_per_attention_head)

        if self.multi_query_attention:
            key_layer = torch.repeat_interleave(
                key_layer,
                self.num_attention_heads_per_partition //
                self.num_multi_query_groups_per_partition,
                dim=1)
            value_layer = torch.repeat_interleave(
                value_layer,
                self.num_attention_heads_per_partition //
                self.num_multi_query_groups_per_partition,
                dim=1)

        # ==================================
        # core attention computation
        # ==================================

        query_layer = query_layer.reshape(
            -1, self.num_attention_heads_per_partition *
            self.hidden_size_per_attention_head)
        key_layer = key_layer.reshape(
            -1, self.num_attention_heads_per_partition *
            self.hidden_size_per_attention_head)
        value_layer = value_layer.reshape(
            -1, self.num_attention_heads_per_partition *
            self.hidden_size_per_attention_head)

        context_layer = self.atten(query_layer, key_layer, value_layer,
                                   k_cache, v_cache, input_metadata,
                                   cache_event)
        # =================
        # Output. [sq, h]
        # =================
        output, _ = self.dense(context_layer)

        return output


class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config, layer_number):
        super().__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size,
                                             eps=config.layernorm_epsilon)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(
            config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = MLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # hidden_states: [s, b, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        # Self attention.
        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=layernorm_output,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )

        layernorm_input = residual + hidden_states

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = residual + mlp_output

        return output


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super().__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number)

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size,
                                                 eps=config.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:

        for index in range(self.num_layers):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[index]
            layer = self._get_layer(index)
            hidden_states = layer(positions, hidden_states, kv_caches[index],
                                  input_metadata, cache_event)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class ChatGLMModel(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            perform_initialization=False,
            params_dtype=config.torch_dtype)

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.fp32_residual_connection = config.fp32_residual_connection

        # Rotary positional embeddings
        self.seq_length = config.seq_length

        self.encoder = GLMTransformer(config)

        self.output_layer = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            params_dtype=config.torch_dtype)

        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:

        inputs_embeds = self.embedding(input_ids)
        # If the input flag for fp32 residual connection is set,
        # convert for float.
        if self.fp32_residual_connection:
            inputs_embeds = inputs_embeds.float()
        # Run encoder.
        hidden_states = self.encoder(positions, inputs_embeds, kv_caches,
                                     input_metadata, cache_events)

        next_tokens = self.sampler(self.output_layer.weight, hidden_states,
                                   input_metadata)

        return next_tokens

    _column_parallel_weights = [
        "dense_h_to_4h.weight", "query_key_value.weight",
        "output_layer.weight", "embedding.weight"
    ]
    _row_parallel_weights = ["dense_4h_to_h.weight", "dense.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):

        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "rotary_pos_emb.inv_freq" in name:
                continue

            name = name.replace("transformer.", "")

            if name.startswith("embedding"):
                name = name.replace(".word_embeddings", "")

            param = state_dict[name]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)
