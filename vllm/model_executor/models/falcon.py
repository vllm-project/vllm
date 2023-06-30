
"""Inference-only Falcon model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

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


class Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size

        self.query_key_value = ColumnParallelLinear(
            self.hidden_size,
            3 * self.hidden_size,  # FIXME(woosuk)
            bias=config.bias,
            gather_output=False,
            perform_initialization=False,
        )
        self.dense = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=config.bias,
            input_is_parallel=True,
            perform_initialization=False,
        )

        assert config.rotary
        rotary_dim = int(self.head_dim * 0.5)
        assert rotary_dim % 2 == 0
        scaling = self.head_dim ** -0.5
        self.attn = PagedAttentionWithRoPE(self.num_heads, self.head_dim,
                                           scaling, rotary_dim)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv = self.query_key_value(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            position_ids, q, k, v, k_cache, v_cache, input_metadata, cache_event)
        output = self.dense(attn_output)
        return output


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4 * hidden_size,
                                                  bias=config.bias,
                                                  gather_output=False,
                                                  perform_initialization=False)
        self.act = get_act_fn("gelu")
        self.dense_4h_to_h = RowParallelLinear(4 * hidden_size, hidden_size,
                                               bias=config.bias,
                                               input_is_parallel=True,
                                               perform_initialization=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_h_to_4h(x)
        x = self.act(x)
        x = self.dense_4h_to_h(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.num_heads = config.n_head

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.self_attention = Attention(config)
        if not config.parallel_attn:
            # unused if parallel attn
            self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        layernorm_output = self.input_layernorm(hidden_states)
        residual = hidden_states

        # Self attention.
        attention_output = self.self_attention(
            position_ids=position_ids,
            hidden_states=layernorm_output,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        if not self.config.parallel_attn:
            residual = attention_output + residual
            layernorm_output = self.post_attention_layernorm(residual)

        # MLP.
        mlp_output = self.mlp(layernorm_output)
        if self.config.parallel_attn:
            mlp_output += attention_output

        output = mlp_output + residual
        return output


class RWModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        assert not config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.word_embeddings(input_ids)
        for i in range(len(self.h)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class RWForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = RWModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.transformer(
            input_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(
            self.lm_head.weight, hidden_states, input_metadata)
        return next_tokens

    _column_parallel_weights = ["word_embeddings.weight", "lm_head.weight",
                                "dense_h_to_4h.weight", "dense_h_to_4h.bias"]
    _row_parallel_weights = ["dense.weight", "dense_4h_to_h.weight"]

    def load_weights(self, model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache):
            param = state_dict[name]

            def _expand_mqa_mha(qkv_array, n_head, head_dim):
                """manipulates along axis=0 from MQA to MHA
                inputs: qkv_array.shape=((n_heads + 2) * head_dim, hidden_dim)
                    with n_heads for q, then 1 for k, 1 for 1 v, times head dim
                return: qkv_array.shape=(3 * n_heads * head_dim, hidden_dim)

                TODO: this function is no longer needed once vllm supports MQA.
                """
                qkv_array = qkv_array.numpy()
                
                dims_q = n_head * head_dim
                q, k, v = np.split(qkv_array, (dims_q, dims_q + head_dim), axis=0)
                # q is fine, but k & v have not replicated shape along the first axis
                # as long as MQA is not nativly supported, increase memory and replicated
                # (head_dim, hidden_dim) to (n_heads * head_dim, hidden_dim)
                if k.ndim == 2 and v.ndim == 2:
                    replication = (n_head, 1)  # weights
                else:
                    replication = n_head  # biases
                # replicate n_head times for q, v
                k, v = np.tile(k, replication), np.tile(v, replication)
                # concat q, k, v along the first axis (n_heads * head_dim, hidden_dim)
                # to (3 * n_heads * head_dim, hidden_dim)
                qkv_array = np.concatenate((q, k, v), axis=0)
                return torch.from_numpy(qkv_array)

            # For the fused QKV linear layer, manually shard the weights.
            if "query_key_value" in name:
                # The fused QKV weight has the shape of [3 * num_heads * head_size, hidden_size].
                # When tensor parallelism is used, we shard the weights along the head dimension.
                total_num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // total_num_heads
                num_heads = total_num_heads // tensor_model_parallel_world_size
                head_start = tensor_model_parallel_rank * num_heads
                head_end = (tensor_model_parallel_rank + 1) * num_heads

                if name.endswith(".weight"):
                    # FIXME
                    orig_dtype = loaded_weight.dtype
                    loaded_weight = loaded_weight.float()
                    loaded_weight = _expand_mqa_mha(loaded_weight, n_head=total_num_heads, head_dim=head_size)
                    loaded_weight = loaded_weight.view(3, total_num_heads, head_size, hidden_size)
                    loaded_weight = loaded_weight[:, head_start:head_end, :, :]
                    loaded_weight = loaded_weight.reshape(-1, hidden_size)
                    loaded_weight = loaded_weight.to(orig_dtype)
                elif name.endswith(".bias"):
                    loaded_weight = _expand_mqa_mha(loaded_weight, n_head=total_num_heads, head_dim=head_size)
                    loaded_weight = loaded_weight.view(3, total_num_heads, head_size)
                    loaded_weight = loaded_weight[:, head_start:head_end, :]
                    loaded_weight = loaded_weight.reshape(-1)
                else:
                    raise ValueError(f"Unexpected parameter name {name}")
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights,
                                         tensor_model_parallel_rank)
