
"""Inference-only Falcon model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
from typing import Dict, List, Optional, Tuple

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
        assert config.multi_query
        self.total_num_query_heads = config.n_head
        self.head_dim = self.hidden_size // self.total_num_query_heads
        assert self.head_dim * self.total_num_query_heads == self.hidden_size

        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_query_heads % tensor_model_parallel_world_size == 0
        self.num_query_heads = (self.total_num_query_heads //
                                tensor_model_parallel_world_size)

        self.total_num_kv_heads = getattr(config, "n_head_kv", 1)
        assert self.total_num_kv_heads % tensor_model_parallel_world_size == 0
        self.num_kv_heads = (self.total_num_kv_heads //
                             tensor_model_parallel_world_size)

        # Grouped MQA.
        assert self.num_query_heads % self.num_kv_heads == 0
        self.query_per_kv = self.num_query_heads // self.num_kv_heads

        self.query = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=config.bias,
            gather_output=False,
            perform_initialization=False,
        )
        self.key_value = ColumnParallelLinear(
            self.hidden_size,
            2 * self.total_num_kv_heads * self.head_dim,
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
        self.attn = PagedAttentionWithRoPE(self.num_query_heads, self.head_dim,
                                           scaling, rotary_dim)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        q, _ = self.query(hidden_states)
        kv, _ = self.key_value(hidden_states)
        k, v = kv.chunk(chunks=2, dim=-1)

        # FIXME(woosuk): For now, we regard MQA as MHA and replicate the keys
        # and values for each head. This is not efficient and should be fixed
        # in the future.
        k = k.repeat(1, self.query_per_kv)
        v = v.repeat(1, self.query_per_kv)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            position_ids, q, k, v, k_cache, v_cache, input_metadata, cache_event)
        output, _ = self.dense(attn_output)
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
        x, _ = self.dense_h_to_4h(x)
        x = self.act(x)
        x, _ = self.dense_4h_to_h(x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.num_heads = config.n_head

        self.input_layernorm = nn.LayerNorm(hidden_size,
                                            eps=config.layer_norm_epsilon)
        self.self_attention = Attention(config)
        if not config.parallel_attn:
            # unused if parallel attn
            self.post_attention_layernorm = nn.LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon)
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
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size, self.embed_dim, perform_initialization=False)

        # Transformer blocks
        self.h = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)])

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

            # For the fused QKV linear layer, we split the weight into query and key_value
            # and shard them along the head dimension.
            if "query_key_value" in name:
                if not name.endswith(".weight"):
                    # Falcon does not use bias for linear layers.
                    raise ValueError(f"Unexpected parameter name {name}")

                # The fused QKV weight has the shape of
                # [(num_q_heads + 2 * num_kv_heads) * head_size, hidden_size].
                total_num_query_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // total_num_query_heads

                num_query_heads = (total_num_query_heads //
                                   tensor_model_parallel_world_size)
                query_head_start = tensor_model_parallel_rank * num_query_heads
                query_head_end = (tensor_model_parallel_rank + 1) * num_query_heads

                total_num_kv_heads = getattr(self.config, "n_head_kv", 1)
                num_kv_heads = (total_num_kv_heads //
                                tensor_model_parallel_world_size)
                kv_head_start = tensor_model_parallel_rank * num_kv_heads
                kv_head_end = (tensor_model_parallel_rank + 1) * num_kv_heads
                loaded_weight = loaded_weight.view(
                    num_query_heads + 2 * num_kv_heads, head_size, hidden_size)

                # Load query weights.
                query_weight = loaded_weight[:num_query_heads, :, :]
                query_weight = loaded_weight[query_head_start:query_head_end, :, :]
                query_weight = query_weight.reshape(-1, hidden_size)
                param = state_dict[name.replace("query_key_value", "query")]
                assert param.shape == query_weight.shape
                param.data.copy_(query_weight)

                # Load key and value weights.
                key_value_weight = loaded_weight[num_query_heads:, :, :]
                key_value_weight = key_value_weight.reshape(
                    2, num_kv_heads, head_size, hidden_size)
                key_value_weight = key_value_weight[:, kv_head_start:kv_head_end, :, :]
                key_value_weight = key_value_weight.reshape(-1, hidden_size)
                param = state_dict[name.replace("query_key_value", "key_value")]
                assert param.shape == key_value_weight.shape
                param.data.copy_(key_value_weight)
            else:
                param = state_dict[name]
                load_tensor_parallel_weights(param, loaded_weight, name,
                                             self._column_parallel_weights,
                                             self._row_parallel_weights,
                                             tensor_model_parallel_rank)
