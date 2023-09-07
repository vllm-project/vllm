# coding=utf-8
# Adapted from https://huggingface.co/mosaicml/mpt-7b/tree/main
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttentionWithALiBi
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (convert_pyslice_to_tensor,
                                              hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.mpt import MPTConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _get_alibi_slopes(
    total_num_heads: int,
    alibi_bias_max: int,
) -> torch.Tensor:
    next_power_of_2 = 2**math.ceil(math.log2(total_num_heads))
    m = torch.arange(1, next_power_of_2 + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / next_power_of_2)
    slopes = 1.0 / torch.pow(2, m)
    if next_power_of_2 != total_num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:total_num_heads]
    return slopes


class MPTAttention(nn.Module):

    def __init__(self, config: MPTConfig):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.clip_qkv = config.attn_config["clip_qkv"]
        self.qk_ln = config.attn_config["qk_ln"]
        self.alibi_bias_max = config.attn_config["alibi_bias_max"]
        assert not config.attn_config["prefix_lm"]
        assert config.attn_config["alibi"]

        self.qkv_proj = ColumnParallelLinear(
            self.d_model,
            3 * self.d_model,
            bias=not config.no_bias,
            gather_output=False,
            perform_initialization=False,
        )
        if self.qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model)
            self.k_ln = nn.LayerNorm(self.d_model)
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=not config.no_bias,
            input_is_parallel=True,
            perform_initialization=False,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        # Create the alibi slopes and slice them.
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.total_num_heads,
                                         self.alibi_bias_max)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()

        self.head_dim = self.d_model // self.total_num_heads
        scaling = self.head_dim**-0.5
        self.attn = PagedAttentionWithALiBi(self.num_heads, self.head_dim,
                                            scaling, alibi_slopes)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        del position_ids  # unused.
        qkv, _ = self.qkv_proj(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        if self.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata,
                                cache_event)
        output, _ = self.out_proj(attn_output)
        return output


class MPTMLP(nn.Module):

    def __init__(self, config: MPTConfig):
        super().__init__()
        hidden_size = config.d_model
        expansion_ratio = config.expansion_ratio
        intermediate_size = expansion_ratio * hidden_size
        self.up_proj = ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=not config.no_bias,
                                            gather_output=False,
                                            perform_initialization=False)
        self.act = get_act_fn("gelu")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=not config.no_bias,
                                           input_is_parallel=True,
                                           perform_initialization=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act(x)
        x, _ = self.down_proj(x)
        return x


class MPTBlock(nn.Module):

    def __init__(self, config: MPTConfig):
        super().__init__()
        hidden_size = config.d_model
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn = MPTAttention(config)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ffn = MPTMLP(config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        x = self.norm_1(hidden_states)
        x = self.attn(
            position_ids=position_ids,
            hidden_states=x,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        hidden_states = hidden_states + x
        x = self.norm_2(hidden_states)
        x = self.ffn(x)
        hidden_states = hidden_states + x
        return hidden_states


class MPTModel(nn.Module):

    def __init__(self, config: MPTConfig):
        super().__init__()
        assert config.embedding_fraction == 1.0
        assert config.norm_type == "low_precision_layernorm"

        self.wte = VocabParallelEmbedding(config.vocab_size,
                                          config.d_model,
                                          perform_initialization=False)
        self.blocks = nn.ModuleList(
            [MPTBlock(config) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, "bias"):
                    if isinstance(module.bias, nn.Parameter):
                        # Remove the bias term in Linear and LayerNorm.
                        module.register_parameter("bias", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.blocks)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            block = self.blocks[i]
            hidden_states = block(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class MPTForCausalLM(nn.Module):

    def __init__(self, config: MPTConfig):
        super().__init__()
        self.config = config
        assert config.tie_word_embeddings

        self.transformer = MPTModel(config)
        # TODO(zhuohan): create a new weight after implementing pipeline
        #                parallelism
        self.lm_head_weight = self.transformer.wte.weight
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = ["wte.weight", "up_proj.weight", "up_proj.bias"]
    _row_parallel_weights = ["out_proj.weight", "down_proj.weight"]

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto"):
        tp_world_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format):
            if "Wqkv" in name:
                # NOTE(woosuk): MPT's fused QKV has the shape of
                # [3 * num_heads * head_size, hidden_size].
                # When tensor model parallelism is used, we need to shard
                # the weight along the hidden dimension.
                total_num_heads = self.config.num_attention_heads
                hidden_size = self.config.hidden_size
                head_size = hidden_size // total_num_heads
                num_heads = total_num_heads // tp_world_size
                head_start = tp_rank * num_heads
                head_end = (tp_rank + 1) * num_heads
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                if name.endswith(".weight"):
                    loaded_weight = loaded_weight.view(3, total_num_heads,
                                                       head_size, hidden_size)
                    loaded_weight = loaded_weight[:, head_start:head_end, :, :]
                    loaded_weight = loaded_weight.reshape(-1, hidden_size)
                elif name.endswith(".bias"):
                    loaded_weight = loaded_weight.view(3, total_num_heads,
                                                       head_size)
                    loaded_weight = loaded_weight[:, head_start:head_end, :]
                    loaded_weight = loaded_weight.reshape(-1)
                else:
                    raise ValueError(f"Unexpected parameter name {name}")
                name = name.replace("Wqkv", "qkv_proj")
            param = state_dict[name]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights, tp_rank)
