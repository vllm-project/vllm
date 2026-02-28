# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen2.5-VL model with Hybrid SSM + Sliding-Window Attention support.

This module extends the Qwen2.5-VL model to use HybridAttentionLayer in its
language model backbone, combining sliding-window KV cache attention with an
SSM history branch for improved memory efficiency on long contexts.

To enable hybrid attention, set `use_hybrid_attention: true` in the model's
HuggingFace config or pass it via config override.

Usage:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --override-neuron-config '{"use_hybrid_attention": true}'
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.hybrid_attn_layer import HybridAttentionLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.config import set_default_rope_theta

from .qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2MLP, Qwen2Model
from .qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from .utils import extract_layer_index, maybe_prefix


class HybridQwen2Attention(nn.Module):
    """Qwen2 attention that can use either standard or hybrid attention.

    When `use_hybrid_attention` is True in the config, this module uses
    HybridAttentionLayer which combines sliding-window KV cache with an
    SSM history branch. Otherwise, it falls back to standard Attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        use_hybrid_attention: bool = False,
    ) -> None:
        super().__init__()
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
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.use_hybrid_attention = use_hybrid_attention

        if use_hybrid_attention:
            # SSM hyperparameters - proportional to attention dimensions
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
            )
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                attn_type=attn_type,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class HybridQwen2DecoderLayer(Qwen2DecoderLayer):
    """Qwen2 decoder layer with optional hybrid attention support."""

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # Skip parent __init__ to customize attention
        nn.Module.__init__(self)

        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        # Check if hybrid attention is enabled
        use_hybrid_attention = getattr(config, "use_hybrid_attention", False)

        # Use HybridQwen2Attention instead of Qwen2Attention
        self.self_attn = HybridQwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            use_hybrid_attention=use_hybrid_attention,
        )

        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class HybridQwen2Model(Qwen2Model):
    """Qwen2 model with hybrid attention layer support."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        # Use HybridQwen2DecoderLayer instead of Qwen2DecoderLayer
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=HybridQwen2DecoderLayer,
        )


class HybridQwen2ForCausalLM(Qwen2ForCausalLM):
    """Qwen2 for causal LM with optional hybrid attention.

    This model can be loaded with standard Qwen2 weights. To enable hybrid
    attention, set `use_hybrid_attention: true` in the model config.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        # Skip Qwen2ForCausalLM __init__ to use our HybridQwen2Model
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        # Use HybridQwen2Model instead of Qwen2Model
        self.model = HybridQwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        from vllm.distributed import get_pp_group

        from .utils import PPMissingLayer

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                from vllm.model_executor.layers.vocab_parallel_embedding import (
                    ParallelLMHead,
                )

                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Reuse parent's weight loading - hybrid layers have the same
        # weight structure for attention, the SSM adapter weights are
        # initialized randomly (for benchmarking without pretrained SSM weights)
        return super().load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class HybridQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with Hybrid SSM + Sliding-Window Attention.

    This model extends Qwen2_5_VLForConditionalGeneration to use hybrid
    attention in the language model backbone. The vision encoder remains
    unchanged, while the text decoder uses HybridAttentionLayer for
    improved memory efficiency on long video/image contexts.

    To enable hybrid attention, set `use_hybrid_attention: true` in the
    model's config or via override:

        --override-neuron-config '{"use_hybrid_attention": true}'

    The hybrid attention combines:
    1. Sliding-window KV cache for local context
    2. SSM (State Space Model) for history/long-range dependencies
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Call grandparent's __init__ to set up basic attributes
        nn.Module.__init__(self)

        from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
            Qwen2_5_VLConfig,
        )

        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        # Set up vision encoder (same as parent)
        if multimodal_config.get_limit_per_prompt(
            "image"
        ) or multimodal_config.get_limit_per_prompt("video"):
            from vllm.attention.backends.registry import AttentionBackendEnum

            from .qwen2_5_vl import Qwen2_5_VisionTransformer

            attn_backend_override = (
                multimodal_config.mm_encoder_attn_backend
                if multimodal_config is not None
                else None
            )
            self.visual = Qwen2_5_VisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
                attn_backend_override=attn_backend_override,
            )
        else:
            self.visual = None

        # Use HybridQwen2ForCausalLM instead of standard Qwen2ForCausalLM
        # First, ensure use_hybrid_attention is set in the text config
        text_config = config.get_text_config()
        if not hasattr(text_config, "use_hybrid_attention"):
            text_config.use_hybrid_attention = getattr(
                config, "use_hybrid_attention", True
            )

        self.language_model = HybridQwen2ForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @property
    def quant_config(self):
        return self.vllm_config.quant_config

