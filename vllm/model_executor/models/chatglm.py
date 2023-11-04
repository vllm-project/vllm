from typing import List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor import InputMetadata
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_rank
from vllm.model_executor.weight_utils import hf_model_weights_iterator, load_tensor_parallel_weights, \
    load_padded_tensor_parallel_vocab
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.chatglm import ChatGLMConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MLP(nn.Module):

    def __init__(self, config: ChatGLMConfig):
        super().__init__()

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return torch.nn.functional.silu(x[0]) * x[1]

        self.activation_func = swiglu

        self.dense_h_to_4h = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=config.add_bias_linear,
            gather_output=False,
        )
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            input_is_parallel=True,
        )

    def forward(self, hidden_states: torch.Tensor):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


def compute_tp_num_heads(config, tp_world_size):
    total_num_heads = config.num_attention_heads
    assert total_num_heads % tp_world_size == 0
    num_heads = total_num_heads // tp_world_size
    return total_num_heads, num_heads


def compute_tp_num_kv_heads(config, tp_world_size):
    total_num_kv_heads = config.multi_query_group_num
    if total_num_kv_heads >= tp_world_size:
        # Number of KV heads is greater than TP size, so we partition
        # the KV heads across multiple tensor parallel GPUs.
        assert total_num_kv_heads % tp_world_size == 0
    else:
        # Number of KV heads is less than TP size, so we replicate
        # the KV heads across multiple tensor parallel GPUs.
        assert tp_world_size % total_num_kv_heads == 0
    num_kv_heads = max(1, total_num_kv_heads // tp_world_size)
    num_kv_heads_replicas = max(1, tp_world_size // total_num_kv_heads)
    return total_num_kv_heads, num_kv_heads, num_kv_heads_replicas


class Attention(nn.Module):

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.rope_theta = 10000

        tp_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads, self.num_heads = compute_tp_num_heads(
            config, tp_world_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.total_num_kv_heads, self.num_kv_heads, num_kv_heads_replicas = \
            compute_tp_num_kv_heads(config, tp_world_size)

        self.attn = PagedAttentionWithRoPE(num_heads=self.num_heads,
                                           head_size=self.head_dim,
                                           scale=self.head_dim**-0.5,
                                           rotary_dim=self.head_dim,
                                           base=self.rope_theta,
                                           is_glm_style=True,
                                           is_neox_style=False,
                                           num_kv_heads=self.num_kv_heads,
                                           max_position=config.seq_length)

        self.qkv_hidden_size = (
            (self.total_num_heads +
             2 * num_kv_heads_replicas * self.total_num_kv_heads) *
            self.head_dim)
        self.query_key_value = ColumnParallelLinear(
            config.hidden_size,
            self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            gather_output=False,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.add_bias_linear,
            input_is_parallel=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        hidden_states, _ = self.query_key_value(hidden_states)
        query, key, value = hidden_states.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )

        k_cache, v_cache = kv_cache
        attn_output = self.attn(positions, query, key, value, k_cache, v_cache,
                                input_metadata, cache_event)

        output, _ = self.dense(attn_output)
        return output


class DecoderLayer(nn.Module):

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.layernorm_epsilon)
        self.self_attention = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.layernorm_epsilon)
        self.mlp = MLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attention(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ChatGLMModel(nn.Module):

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)])
        self.final_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.layernorm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.word_embeddings(input_ids)
        for i in range(len(self.layers)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def name_mapping(name: str):
    if name.startswith('transformer.encoder.layers.'):
        prefix = 'model.layers.'
        arr = name.split('.')
        return prefix + '.'.join(arr[3:])
    if name == 'transformer.output_layer.weight':
        return 'lm_head.weight'
    if name == 'transformer.encoder.final_layernorm.weight':
        return 'model.final_layernorm.weight'
    if name == 'transformer.embedding.word_embeddings.weight':
        return 'model.word_embeddings.weight'
    assert False, f'unknow param {name}'


class ChatGLMForCausalLM(nn.Module):

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.config = config
        self.model = ChatGLMModel(config)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.padded_vocab_size,
            bias=False,
            gather_output=False,
        )
        self.sampler = Sampler(config.padded_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = []
    _row_parallel_weights = ['dense_4h_to_h.weight', 'dense.weight']

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = 'auto',
        revision: Optional[str] = None,
    ):
        tp_world_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if 'rotary_pos_emb.inv_freq' in name:
                continue

            vname = name_mapping(name)
            param = state_dict[vname]
            if 'dense_h_to_4h' in vname:
                shard_size = param.size(0) // 2
                base = 0
                weight1 = loaded_weight[base + shard_size * tp_rank:base +
                                        shard_size * (tp_rank + 1)]
                param.data[:shard_size].copy_(weight1)
                base = loaded_weight.size(0) // 2
                weight2 = loaded_weight[base + shard_size * tp_rank:base +
                                        shard_size * (tp_rank + 1)]
                param.data[shard_size:].copy_(weight2)
                continue

            if 'query_key_value' in vname:
                total_num_heads, num_heads = compute_tp_num_heads(
                    self.config, tp_world_size)
                head_dim = self.config.hidden_size // total_num_heads
                total_num_kv_heads, num_kv_heads, num_kv_heads_replicas = \
                    compute_tp_num_kv_heads(self.config, tp_world_size)
                query_shard_size = num_heads * head_dim
                kv_proj_shard_size = num_kv_heads * head_dim

                base = tp_rank * num_heads * head_dim
                param_query = param.data[:query_shard_size]
                weight_query = loaded_weight[base:base + query_shard_size]
                param_query.copy_(weight_query)

                base = (total_num_heads +
                        (tp_rank // num_kv_heads_replicas)) * head_dim
                param_key = param.data[query_shard_size:query_shard_size +
                                       kv_proj_shard_size]
                weight_key = loaded_weight[base:base + kv_proj_shard_size]
                param_key.copy_(weight_key)

                base = (total_num_heads + total_num_kv_heads +
                        (tp_rank // num_kv_heads_replicas)) * head_dim
                param_value = param.data[query_shard_size +
                                         kv_proj_shard_size:]
                weight_value = loaded_weight[base:base + kv_proj_shard_size]
                param_value.copy_(weight_value)
                continue

            if 'word_embeddings' in vname or 'lm_head' in vname:
                load_padded_tensor_parallel_vocab(param, loaded_weight,
                                                  tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                vname,
                self._column_parallel_weights,
                self._row_parallel_weights,
                tp_rank,
            )
