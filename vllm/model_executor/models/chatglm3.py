from typing import List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor import InputMetadata
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.sampler import Sampler
# from vllm.model_executor.parallel_utils.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
# from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_rank
from vllm.model_executor.weight_utils import hf_model_weights_iterator, load_tensor_parallel_weights
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.chatglm3 import ChatGLM3Config

KVCache = Tuple[torch.Tensor, torch.Tensor]


class ChatGLM3MLP(nn.Module):

    def __init__(
            self,
            config: ChatGLM3Config
    ):
        super().__init__()

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return torch.nn.functional.silu(x[0]) * x[1]

        self.activation_func = swiglu

        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=config.add_bias_linear,
        )
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
        )

    def forward(self, hidden_states: torch.Tensor):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class ChatGLM3Attention(nn.Module):
    def __init__(
            self,
            config: ChatGLM3Config
    ):
        super().__init__()
        self.rope_theta = 10000

        self.head_dim = config.hidden_size // config.num_attention_heads
        scaling = self.head_dim ** -0.5
        self.attn = PagedAttentionWithRoPE(
            num_heads=config.num_attention_heads,
            head_size=self.head_dim,
            scale=scaling,
            rotary_dim=self.head_dim,
            base=self.rope_theta,
            is_glm_style=True,
            num_kv_heads=config.multi_query_group_num,
            max_position=config.seq_length)

        self.num_attention_heads_per_partition = config.num_attention_heads
        self.num_multi_query_groups_per_partition = config.multi_query_group_num
        self.projection_size = config.kv_channels * config.num_attention_heads
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.qkv_hidden_size = (
                self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
        )

        self.query_key_value = nn.Linear(
            config.hidden_size, self.qkv_hidden_size,
            bias=config.add_bias_linear or config.add_qkv_bias)
        self.dense = nn.Linear(
            self.projection_size, config.hidden_size,
            bias=config.add_bias_linear)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
            cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        mixed_x_layer = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )

        k_cache, v_cache = kv_cache
        attn_output = self.attn(
            positions, query_layer, key_layer, value_layer, k_cache, v_cache,
            input_metadata, cache_event)

        output = self.dense(attn_output)
        return output


class ChatGLM3DecoderLayer(nn.Module):

    def __init__(self, config: ChatGLM3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attention = ChatGLM3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.mlp = ChatGLM3MLP(config)

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


class ChatGLM3Model(nn.Module):

    def __init__(self, config: ChatGLM3Config):
        super().__init__()
        # self.embedding = VocabParallelEmbedding(
        self.embedding = nn.Embedding(
            config.padded_vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            ChatGLM3DecoderLayer(config) for _ in range(config.num_layers)
        ])

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
            cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
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
    if name.startswith("transformer.encoder.layers."):
        prefix = "model.layers."
        arr = name.split('.')
        return prefix + '.'.join(arr[3:])
    if name == 'transformer.output_layer.weight':
        return 'lm_head.weight'
    if name == 'transformer.encoder.final_layernorm.weight':
        return 'model.final_layernorm.weight'
    if name == 'transformer.embedding.word_embeddings.weight':
        return 'model.embedding.weight'
    assert False, f"unknow param {name}"


class ChatGLM3ForCausalLM(nn.Module):

    def __init__(self, config: ChatGLM3Config):
        super().__init__()
        self.config = config
        self.model = ChatGLM3Model(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.padded_vocab_size,
            bias=False,
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
    _row_parallel_weights = []

    def load_weights(
            self,
            model_name_or_path: str,
            cache_dir: Optional[str] = None,
            load_format: str = "auto",
            revision: Optional[str] = None,
    ):
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_pos_emb.inv_freq" in name:
                continue

            vname = name_mapping(name)
            param = state_dict[vname]
            load_tensor_parallel_weights(
                param,
                loaded_weight,
                vname,
                self._column_parallel_weights,
                self._row_parallel_weights,
                tensor_model_parallel_rank,
            )
