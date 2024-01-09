# coding=utf-8
# Adapted from https://huggingface.co/mosaicml/mpt-7b/tree/main
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
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

    def __init__(
        self,
        config: MPTConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.total_num_heads = config.n_heads
        self.head_dim = self.d_model // self.total_num_heads
        self.clip_qkv = config.attn_config["clip_qkv"]
        self.qk_ln = config.attn_config["qk_ln"]
        self.alibi_bias_max = config.attn_config["alibi_bias_max"]
        if "kv_n_heads" in config.attn_config:
            self.total_num_kv_heads = config.attn_config['kv_n_heads']
        else:
            self.total_num_kv_heads = self.total_num_heads
        assert not config.attn_config["prefix_lm"]
        assert config.attn_config["alibi"]

        # pylint: disable=invalid-name
        self.Wqkv = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=not config.no_bias,
            linear_method=linear_method,
        )
        if self.qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model)
            self.k_ln = nn.LayerNorm(self.d_model)
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=not config.no_bias,
            linear_method=linear_method,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Create the alibi slopes and slice them.
        tp_rank = get_tensor_model_parallel_rank()
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.total_num_heads,
                                         self.alibi_bias_max)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()

        self.head_dim = self.d_model // self.total_num_heads
        scaling = self.head_dim**-0.5
        self.attn = PagedAttention(self.num_heads,
                                   self.head_dim,
                                   scaling,
                                   alibi_slopes=alibi_slopes,
                                   num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        del position_ids  # unused.
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.out_proj(attn_output)
        return output


class MPTMLP(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        hidden_size = config.d_model
        expansion_ratio = config.expansion_ratio
        intermediate_size = expansion_ratio * hidden_size
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=not config.no_bias,
            linear_method=linear_method,
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn("gelu", quant_config, intermediate_size)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=not config.no_bias,
            linear_method=linear_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act(x)
        x, _ = self.down_proj(x)
        return x


class MPTBlock(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        hidden_size = config.d_model
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn = MPTAttention(config, linear_method)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ffn = MPTMLP(config, linear_method)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        x = self.norm_1(hidden_states)
        x = self.attn(
            position_ids=position_ids,
            hidden_states=x,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        hidden_states = hidden_states + x
        x = self.norm_2(hidden_states)
        x = self.ffn(x)
        hidden_states = hidden_states + x
        return hidden_states


class MPTModel(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        assert config.embedding_fraction == 1.0
        assert config.norm_type == "low_precision_layernorm"

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.blocks = nn.ModuleList(
            [MPTBlock(config, linear_method) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, "bias") and isinstance(
                        module.bias, nn.Parameter):
                    # Remove the bias term in Linear and LayerNorm.
                    module.register_parameter("bias", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            hidden_states = block(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
            )
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class MPTForCausalLM(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        assert config.tie_word_embeddings
        self.linear_method = linear_method

        self.transformer = MPTModel(config, linear_method)
        self.lm_head_weight = self.transformer.wte.weight
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head_weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
