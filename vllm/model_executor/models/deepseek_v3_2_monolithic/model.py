# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Monolithic DeepSeek V3.2 model for SM100 (Blackwell).
No PP, TP only, same checkpoint compatibility.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform

from .decoder_layer import MonolithicDecoderLayer
from .ops import fused_add_rms_norm


@support_torch_compile
class MonolithicDeepseekV32Model(nn.Module):
    """Transformer backbone."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        topk_tokens = config.index_topk
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList(
            [
                MonolithicDecoderLayer(
                    vllm_config=vllm_config,
                    config=config,
                    layer_idx=i,
                    topk_indices_buffer=self.topk_indices_buffer,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm_weight = nn.Parameter(
            torch.ones(config.hidden_size, dtype=torch.get_default_dtype())
        )
        self.rms_norm_eps = config.rms_norm_eps

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = fused_add_rms_norm(
            hidden_states, residual, self.norm_weight, self.rms_norm_eps
        )
        return hidden_states


class DeepseekV32MonolithicForCausalLM(nn.Module):
    """
    Monolithic DeepSeek V3.2 CausalLM for SM100.
    """

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.tp_size = get_tensor_model_parallel_world_size()

        self.model = MonolithicDeepseekV32Model(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.num_redundant_experts = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds=None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states)
        logits = tensor_model_parallel_all_gather(logits)
        logits = logits[..., : self.config.vocab_size]
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Delegate to the original DeepSeek V2 weight loader.

        Our module structure matches the original for all weights that
        need special loading (fused_qkv_a_proj, experts, gate_up_proj).
        Only layernorm weights and indexer paths differ.
        """
        from vllm.model_executor.models.deepseek_v2 import (
            DeepseekV2ForCausalLM,
        )

        def _remap_weights():
            for name, w in weights:
                yield self._remap(name), w

        self.use_mha = False
        self.fuse_qkv_a_proj = True
        return DeepseekV2ForCausalLM.load_weights(self, _remap_weights())

    def _remap(self, name: str) -> str:
        """Remap only names that differ from original model structure."""
        # Only remap layernorms (raw params) and indexer (underscore prefix).
        # Everything else (fused_qkv_a_proj, experts, gate, etc.) uses the
        # same module paths as the original model.
        replacements = [
            ("input_layernorm.weight", "input_layernorm_weight"),
            (
                "post_attention_layernorm.weight",
                "post_attention_layernorm_weight",
            ),
            (
                "self_attn.q_a_layernorm.weight",
                "attn.q_a_layernorm_weight",
            ),
            (
                "self_attn.kv_a_layernorm.weight",
                "attn.kv_a_layernorm_weight",
            ),
            ("self_attn.q_b_proj", "attn.q_b_proj"),
            ("self_attn.kv_b_proj", "attn.kv_b_proj"),
            ("self_attn.o_proj", "attn.o_proj"),
            ("self_attn.indexer.", "attn.indexer_"),
            ("model.norm.weight", "model.norm_weight"),
        ]
        for old, new in replacements:
            if old in name:
                return name.replace(old, new)
        return name
