# coding=utf-8
# Adapted from https://huggingface.co/mosaicml/mpt-7b/tree/main
import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.mpt import MPTConfig


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
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
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
            quant_config=quant_config,
        )
        if self.qk_ln:
            self.q_ln = nn.LayerNorm(self.d_model)
            self.k_ln = nn.LayerNorm(self.d_model)
        self.out_proj = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=not config.no_bias,
            quant_config=quant_config,
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
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scaling,
                              alibi_slopes=alibi_slopes,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        del position_ids  # unused.
        qkv, _ = self.Wqkv(hidden_states)
        if self.clip_qkv is not None:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_ln:
            q = self.q_ln(q)
            k = self.k_ln(k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.out_proj(attn_output)
        return output


class MPTMLP(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.d_model
        expansion_ratio = config.expansion_ratio
        intermediate_size = expansion_ratio * hidden_size
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=not config.no_bias,
            quant_config=quant_config,
        )
        self.act = get_act_fn("gelu", quant_config, intermediate_size)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=not config.no_bias,
            quant_config=quant_config,
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
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        hidden_size = config.d_model
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn = MPTAttention(config, cache_config, quant_config)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ffn = MPTMLP(config, quant_config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        x = self.norm_1(hidden_states)
        x = self.attn(
            position_ids=position_ids,
            hidden_states=x,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
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
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        assert config.embedding_fraction == 1.0
        assert config.norm_type == "low_precision_layernorm"

        self.wte = VocabParallelEmbedding(
            config.vocab_size,
            config.d_model,
        )
        self.blocks = nn.ModuleList([
            MPTBlock(config, cache_config, quant_config)
            for _ in range(config.n_layers)
        ])
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
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            hidden_states = block(
                position_ids,
                hidden_states,
                kv_caches[i],
                attn_metadata,
            )
        hidden_states = self.norm_f(hidden_states)
        return hidden_states


class MPTForCausalLM(nn.Module):

    def __init__(
        self,
        config: MPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        assert config.tie_word_embeddings
        self.quant_config = quant_config

        self.transformer = MPTModel(config, cache_config, quant_config)
        self.lm_head = self.transformer.wte
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         attn_metadata)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
