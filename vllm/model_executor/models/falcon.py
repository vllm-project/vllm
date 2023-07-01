# Adapted from https://huggingface.co/tiiuae/falcon-7b/blob/main/modelling_RW.py
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
from vllm.transformers_utils.configs import RWConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class Attention(nn.Module):

    def __init__(self, config: RWConfig):
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

        self.total_num_kv_heads = config.n_head_kv
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
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        k = k.repeat(1, 1, self.query_per_kv)
        k = k.reshape(-1, self.num_kv_heads * self.query_per_kv * self.head_dim)

        v = v.view(-1, self.num_kv_heads, self.head_dim)
        v = v.repeat(1, 1, self.query_per_kv)
        v = v.reshape(-1, self.num_kv_heads * self.query_per_kv * self.head_dim)

        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            position_ids, q, k, v, k_cache, v_cache, input_metadata, cache_event)
        output, _ = self.dense(attn_output)
        return output


class MLP(nn.Module):

    def __init__(self, config: RWConfig):
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

    def __init__(self, config: RWConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.num_heads = config.n_head

        # NOTE(woosuk): Falcon-7B uses the same layernorm for attention and
        # MLP. However, Falcon-40B uses separate layernorms for them.
        if _is_falcon_7b(config):
            # Falcon-7B
            self.input_layernorm = nn.LayerNorm(hidden_size,
                                                eps=config.layer_norm_epsilon)
        elif _is_falcon_40b(config):
            # Falcon-40B
            self.ln_attn = nn.LayerNorm(hidden_size,
                                        eps=config.layer_norm_epsilon)
            self.ln_mlp = nn.LayerNorm(hidden_size,
                                       eps=config.layer_norm_epsilon)
        else:
            raise ValueError("Only 7B and 40B models are supported for now.")
        assert config.parallel_attn
        self.self_attention = Attention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        residual = hidden_states
        if _is_falcon_7b(self.config):
            # Falcon-7B
            ln_attn = self.input_layernorm(hidden_states)
            ln_mlp = ln_attn
        elif _is_falcon_40b(self.config):
            # Falcon-40B
            ln_attn = self.ln_attn(hidden_states)
            ln_mlp = self.ln_mlp(hidden_states)
        else:
            assert False

        # Self attention.
        attention_output = self.self_attention(
            position_ids=position_ids,
            hidden_states=ln_attn,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        # MLP.
        mlp_output = self.mlp(ln_mlp)
        output = attention_output + mlp_output + residual
        return output


class RWModel(nn.Module):

    def __init__(self, config: RWConfig):
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

    def __init__(self, config: RWConfig):
        super().__init__()
        self.config = config

        self.transformer = RWModel(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size,
                                            bias=False, gather_output=False,
                                            perform_initialization=False)
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
                                "dense_h_to_4h.weight"]
    _row_parallel_weights = ["dense.weight", "dense_4h_to_h.weight"]

    def load_weights(self, model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     use_np_cache: bool = False):
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, use_np_cache):
            if "query_key_value" not in name:
                param = state_dict[name]
                load_tensor_parallel_weights(param, loaded_weight, name,
                                             self._column_parallel_weights,
                                             self._row_parallel_weights,
                                             tensor_model_parallel_rank)
                continue

            # For the fused QKV linear layer, we split the weight into
            # query and key_value and shard them along the head dimension.
            if not name.endswith(".weight"):
                # Falcon does not use bias for linear layers.
                raise ValueError(f"Unexpected parameter name {name}")

            total_num_query_heads = self.config.num_attention_heads
            hidden_size = self.config.hidden_size
            head_size = hidden_size // total_num_query_heads

            assert total_num_query_heads % tensor_model_parallel_world_size == 0
            num_query_heads = (total_num_query_heads //
                               tensor_model_parallel_world_size)
            query_head_start = tensor_model_parallel_rank * num_query_heads
            query_head_end = (tensor_model_parallel_rank + 1) * num_query_heads

            total_num_kv_heads = self.config.n_head_kv
            assert total_num_kv_heads % tensor_model_parallel_world_size == 0
            num_kv_heads = (total_num_kv_heads //
                            tensor_model_parallel_world_size)
            kv_head_start = tensor_model_parallel_rank * num_kv_heads
            kv_head_end = (tensor_model_parallel_rank + 1) * num_kv_heads

            if _is_falcon_7b(self.config):
                # Falcon-7B (MQA)
                # The fused QKV weight has the shape of
                # [(num_q_heads + 2 * num_kv_heads) * head_size, hidden_size].
                loaded_weight = loaded_weight.view(-1, head_size, hidden_size)

                # Load query weights.
                query_weight = loaded_weight[query_head_start:query_head_end, :, :]
                query_weight = query_weight.reshape(-1, hidden_size)
                param = state_dict[name.replace("query_key_value", "query")]
                assert param.shape == query_weight.shape
                param.data.copy_(query_weight)

                # Load key_value weights.
                key_value_weight = loaded_weight[total_num_query_heads:, :, :]
                key_value_weight = key_value_weight.reshape(
                    2, total_num_kv_heads, head_size, hidden_size)
                key_value_weight = key_value_weight[:, kv_head_start:kv_head_end, :, :]
                key_value_weight = key_value_weight.reshape(-1, hidden_size)
                param = state_dict[name.replace("query_key_value", "key_value")]
                assert param.shape == key_value_weight.shape
                param.data.copy_(key_value_weight)
            elif _is_falcon_40b(self.config):
                # Falcon-40B (Grouped MQA)
                # The fused QKV weight has the shape of
                # [total_num_kv_heads, num_query_per_kv + 2, head_size, hidden_size].
                num_query_per_kv = total_num_query_heads // total_num_kv_heads
                loaded_weight = loaded_weight.view(
                    total_num_kv_heads, num_query_per_kv + 2, head_size, hidden_size)

                # Load query weights.
                query_weight = loaded_weight[kv_head_start:kv_head_end, :-2, :, :]
                query_weight = query_weight.reshape(-1, hidden_size)
                param = state_dict[name.replace("query_key_value", "query")]
                assert param.shape == query_weight.shape
                param.data.copy_(query_weight)

                # Load key_value weights.
                key_value_weight = loaded_weight[kv_head_start:kv_head_end, -2:, :, :]
                key_value_weight = key_value_weight.permute(1, 0, 2, 3)
                key_value_weight = key_value_weight.reshape(-1, hidden_size)
                param = state_dict[name.replace("query_key_value", "key_value")]
                assert param.shape == key_value_weight.shape
                param.data.copy_(key_value_weight)
            else:
                assert False


def _is_falcon_7b(config: RWConfig) -> bool:
    # FIXME(woosuk): This is hacky.
    return config.hidden_size == 4544


def _is_falcon_40b(config: RWConfig) -> bool:
    # FIXME(woosuk): This is hacky.
    return config.hidden_size == 8192
