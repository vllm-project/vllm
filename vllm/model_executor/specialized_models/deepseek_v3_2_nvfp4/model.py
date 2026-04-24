# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 NVFP4 model for SM100 (Blackwell)."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform

from .layer import DeepseekV32DecoderLayer, remap_weight_name

logger = init_logger(__name__)


@support_torch_compile
class DeepseekV32Model(nn.Module):
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
                DeepseekV32DecoderLayer(
                    vllm_config=vllm_config,
                    config=config,
                    layer_idx=i,
                    topk_indices_buffer=self.topk_indices_buffer,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=torch.get_default_dtype(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV32ForCausalLM(nn.Module):
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

        self.model = DeepseekV32Model(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
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
        return self.logits_processor(self.lm_head, hidden_states)

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
                yield remap_weight_name(name), w

        self.use_mha = False
        self.fuse_qkv_a_proj = True
        self.is_fp4_ckpt = False
        loaded = DeepseekV2ForCausalLM.load_weights(self, _remap_weights())

        # Fuse indexer linear weights after loading.
        for layer in self.model.layers:
            layer.fuse_indexer_weights()

        return loaded
