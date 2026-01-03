# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from mimo_v2_flash.py and SGLang's mimo_v2_flash_nextn.py
# Copyright 2025 Xiaomi Corporation.
# Copyright 2023 The vLLM team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiMo-V2-Flash MTP model."""

from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors

from .mimo_v2_flash import MiMoV2Attention, MiMoV2MLP
from .utils import maybe_prefix

logger = init_logger(__name__)


class MiMoV2FlashMTPBlock(nn.Module):
    """MTP decoder block using SWA attention and dense MLP."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 1000000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        v_scale = getattr(config, "attention_value_scale", None)

        # MTP layers use SWA (sliding window attention) configuration
        self.self_attn = MiMoV2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.swa_num_attention_heads,
            num_kv_heads=config.swa_num_key_value_heads,
            head_dim=config.swa_head_dim,
            v_head_dim=getattr(config, "swa_v_head_dim", None),
            v_scale=v_scale,
            sliding_window_size=config.sliding_window_size,
            attention_bias=config.attention_bias,
            add_swa_attention_sink_bias=getattr(
                config, "add_swa_attention_sink_bias", False
            ),
            layer_id=layer_id,
            rope_theta=getattr(config, "swa_rope_theta", rope_theta),
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
            prefix=f"{prefix}.self_attn",
        )

        # MTP layers use dense MLP (not MoE)
        self.mlp = MiMoV2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiMoV2FlashMTPLayer(nn.Module):
    """Complete MTP layer with embedding norms, projection, and decoder block."""

    def __init__(self, vllm_config: VllmConfig, layer_idx: int, prefix: str) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Normalization layers for embedding and hidden state fusion
        self.enorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        # Projection to combine embedded tokens with previous hidden states
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # The MTP decoder block (contains attention + MLP)
        self.mtp_block = MiMoV2FlashMTPBlock(
            config=config,
            layer_id=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mtp_block",
        )

        # Final normalization before logits
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert inputs_embeds is not None

        # Fuse embedded tokens with previous hidden states
        hidden_states = self.eh_proj(
            torch.cat(
                [
                    self.enorm(inputs_embeds),
                    self.hnorm(previous_hidden_states),
                ],
                dim=-1,
            )
        )

        # Pass through the MTP decoder block
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )

        # Final layer norm with residual
        if residual is not None:
            hidden_states, _ = self.final_layernorm(hidden_states, residual)
        else:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class MiMoV2FlashMultiTokenPredictor(nn.Module):
    """MTP predictor model following DeepSeek MTP architecture pattern.

    Uses layers dict with indexed MTP layers for KV cache binding compatibility.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        # MTP layer index starts at num_hidden_layers (0 after override)
        self.mtp_start_layer_idx = config.num_hidden_layers
        # Number of MTP layers (set by override, default to 1)
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        # Shared embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Create MTP layers indexed by their layer number (for KV cache binding)
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): MiMoV2FlashMTPLayer(
                    vllm_config, idx, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Use the single MTP layer (spec_step_idx is ignored since we share)
        layer_idx = self.mtp_start_layer_idx
        return self.layers[str(layer_idx)](
            input_ids, positions, previous_hidden_states, inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        logits = self.logits_processor(lm_head, hidden_states)
        return logits


class MiMoV2FlashMTP(nn.Module):
    """MTP model for MiMo-V2-Flash."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = MiMoV2FlashMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, self.lm_head, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        mtp_start_layer_idx = self.model.mtp_start_layer_idx

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue

            # Check if this is an MTP weight
            if "model.mtp.layers" not in name:
                continue

            # Get the original MTP layer index from checkpoint
            orig_layer_idx = self._get_checkpoint_layer_idx(name)
            if orig_layer_idx is None:
                continue

            # Rewrite weight name to match model parameter names
            name = self._map_weight_name(name, orig_layer_idx, mtp_start_layer_idx)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Only process mtp_block weights for stacking
                if "mtp_block" not in name:
                    break
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Try to remap scale names for FP8 quantization
                mapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if mapped_name is not None:
                    name = mapped_name

                if name not in params_dict:
                    continue

                param = params_dict[name]

                if "attention_sink_bias" in name:
                    # Handle attention sink bias sharding
                    start = tp_rank * param.numel()
                    param.data.copy_(loaded_weight[start : start + param.numel()])
                else:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

                loaded_params.add(name)

        return loaded_params

    def _get_checkpoint_layer_idx(self, name: str) -> int | None:
        """Extract MTP layer index from checkpoint weight name."""
        pattern = r"model\.mtp\.layers\.(\d+)\."
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
        return None

    def _map_weight_name(
        self, name: str, orig_layer_idx: int, mtp_start_layer_idx: int
    ) -> str:
        """
        Map checkpoint weight names to model parameter names.

        Checkpoint format: model.mtp.layers.{0,1,2}.*
        Model format: model.layers.{mtp_start_layer_idx}.*

        Since we use a single shared MTP layer, all checkpoint layer weights
        map to the same model layer (layer 0, the last checkpoint loaded wins).
        """
        # Handle pre_mlp_layernorm -> post_attention_layernorm renaming
        if "pre_mlp_layernorm" in name:
            name = name.replace("pre_mlp_layernorm", "post_attention_layernorm")

        # Weight names that go at layer level (not in mtp_block)
        layer_level_weights = [
            "enorm",
            "hnorm",
            "eh_proj",
            "final_layernorm",
        ]

        # Target layer index (all checkpoint layers map to this single layer)
        target_layer_idx = mtp_start_layer_idx  # Should be 0

        # Pattern to match checkpoint format
        pattern = r"model\.mtp\.layers\.(\d+)\."
        match = re.match(pattern, name)
        if match:
            # Check for layer-level weights
            for layer_weight in layer_level_weights:
                if layer_weight in name:
                    # Map model.mtp.layers.X.enorm -> model.layers.0.enorm
                    name = name.replace(
                        match.group(), f"model.layers.{target_layer_idx}."
                    )
                    return name
            # All other weights go to mtp_block
            # Map model.mtp.layers.X.self_attn -> model.layers.0.mtp_block.self_attn
            name = name.replace(
                match.group(), f"model.layers.{target_layer_idx}.mtp_block."
            )
        return name
