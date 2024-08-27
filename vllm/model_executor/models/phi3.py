# coding=utf-8
# Adapted from llama.py

"""Inference-only Phi3 model code inherit from Llama.py"""

from typing import Optional

import torch

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

from .utils import make_layers

from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from transformers import Phi3Config

class Phi3Attention(LlamaAttention):
    def __init__(
        self,
        config: Phi3Config,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config, 
            hidden_size=config.hidden_size, 
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            cache_config=cache_config,
            prefix=prefix)
        
        self.rope_scaling = config.rope_scaling


    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k =  self.rotary_emb(positions, q, k) \
            if self.rope_scaling is None \
            else self.rotary_emb(positions, q, k, num_orig_input_tokens_tensor=attn_metadata.num_orig_input_tokens_tensor)
            
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Phi3DecoderLayer(LlamaDecoderLayer):

    def __init__(
        self,
        config: Phi3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix
        )
        self.self_attn = Phi3Attention(
            config=config,
            quant_config=quant_config,
            bias=getattr(config, "attention_bias", False) or getattr(
            config, "bias", False),
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )



class Phi3Model(LlamaModel):

    def __init__(
        self,
        config: Phi3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            lora_config=lora_config,
            prefix=prefix
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Phi3DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers")


class Phi3ForCausalLM(LlamaForCausalLM):

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Phi3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            lora_config=lora_config
        )

        self.model = Phi3Model(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                prefix="model")
        
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
                