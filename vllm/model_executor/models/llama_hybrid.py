# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LLaMA model with optional Hybrid SSM + Sliding-Window Attention support.

This module extends the standard LLaMA model to optionally use
HybridAttentionLayer, which combines sliding-window KV cache attention
with an SSM history branch for improved memory efficiency on long contexts.

To enable hybrid attention, set `use_hybrid_attention: true` in the model's
HuggingFace config or pass it via config override.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.hybrid_attn_layer import HybridAttentionLayer
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope

from .llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)
from .utils import extract_layer_index


class HybridLlamaAttention(nn.Module):
    """LLaMA attention that can use either standard or hybrid attention.

    When `use_hybrid_attention` is True in the config, this module uses
    HybridAttentionLayer which combines sliding-window KV cache with an
    SSM history branch. Otherwise, it falls back to standard Attention.
    """

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        use_hybrid_attention: bool = False,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # LLaMA 4 scaling support
        llama_4_scaling_config = getattr(config, "llama_4_scaling", None)
        self.do_llama_4_scaling = llama_4_scaling_config is not None
        if self.do_llama_4_scaling:
            self.llama_4_scaling_original_max_position_embeddings = (
                llama_4_scaling_config["original_max_position_embeddings"]
            )
            self.llama_4_scaling_beta = llama_4_scaling_config["beta"]

        # QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Initialize rotary embeddings
        self._init_rotary_emb(config, quant_config=quant_config)

        # Determine sliding window from config
        sliding_window = getattr(config, "sliding_window", None)
        if layer_types := getattr(config, "layer_types", None):
            if hasattr(config, "target_layer_count"):
                effective_layer_idx = layer_idx - config.target_layer_count
            else:
                effective_layer_idx = layer_idx
            if effective_layer_idx < len(layer_types):
                is_sliding = layer_types[effective_layer_idx] == "sliding_attention"
                if not is_sliding:
                    sliding_window = None

        # Choose attention implementation
        self.use_hybrid_attention = use_hybrid_attention

        if use_hybrid_attention:
            # SSM hyperparameters - can be tuned
            ssm_state_size = self.head_dim
            ssm_conv_kernel_size = 4
            ssm_intermediate_size = self.hidden_size // 2

            self.attn = HybridAttentionLayer(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                num_kv_heads=self.num_kv_heads,
                ssm_state_size=ssm_state_size,
                ssm_conv_kernel_size=ssm_conv_kernel_size,
                ssm_intermediate_size=ssm_intermediate_size,
                cache_config=cache_config,
                prefix=f"{prefix}.attn",
                per_layer_sliding_window=sliding_window,
            )
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                per_layer_sliding_window=sliding_window,
                attn_type=attn_type,
                prefix=f"{prefix}.attn",
            )

    def _get_llama_4_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        scaling = 1 + self.llama_4_scaling_beta * torch.log(
            1
            + torch.floor(
                positions / self.llama_4_scaling_original_max_position_embeddings
            )
        )
        return scaling.unsqueeze(-1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        if self.do_llama_4_scaling:
            attn_scale = self._get_llama_4_attn_scale(positions)
            q = (q * attn_scale).to(q.dtype)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(
        self,
        config: LlamaConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            is_neox_style=is_neox_style,
            partial_rotary_factor=self.partial_rotary_factor,
        )


class HybridLlamaDecoderLayer(LlamaDecoderLayer):
    """LLaMA decoder layer with optional hybrid attention support."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
    ) -> None:
        # Skip parent __init__ to customize attention
        nn.Module.__init__(self)

        config = config or vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = self.get_quant_config(vllm_config)

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        # Check if hybrid attention is enabled
        use_hybrid_attention = getattr(config, "use_hybrid_attention", False)

        # Use HybridLlamaAttention instead of LlamaAttention
        self.self_attn = HybridLlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            use_hybrid_attention=use_hybrid_attention,
        )

        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )

        from vllm.model_executor.layers.layernorm import RMSNorm

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class HybridLlamaModel(LlamaModel):
    """LLaMA model with hybrid attention layer support."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        # Use HybridLlamaDecoderLayer instead of LlamaDecoderLayer
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=HybridLlamaDecoderLayer,
        )


class HybridLlamaForCausalLM(LlamaForCausalLM):
    """LLaMA for causal LM with optional hybrid attention.

    This model can be loaded with standard LLaMA weights. To enable hybrid
    attention, set `use_hybrid_attention: true` in the model config or via:

        --override-neuron-config '{"use_hybrid_attention": true}'

    The hybrid attention combines sliding-window KV cache with an SSM history
    branch for improved memory efficiency on long context sequences.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        # Use HybridLlamaDecoderLayer instead of LlamaDecoderLayer
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=HybridLlamaDecoderLayer,
        )

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = HybridLlamaDecoderLayer,
    ):
        return HybridLlamaModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Reuse parent's weight loading - hybrid layers have the same
        # weight structure for attention, the SSM adapter weights are
        # initialized randomly (for benchmarking without pretrained SSM weights)
        return super().load_weights(weights)

