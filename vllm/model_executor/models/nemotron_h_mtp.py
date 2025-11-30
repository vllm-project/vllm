# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NemotronH-MTP model with attention layers."""

import copy
import typing
from collections.abc import Callable, Iterable
from typing import cast

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import NemotronHConfig

from .interfaces import SupportsPP
from .nemotron_h import (
    NemotronHAttentionDecoderLayer,
    NemotronHMoEDecoderLayer,
)


class NemotronHMTPAttentionDecoderLayer(NemotronHAttentionDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Fusion layer to combine embeddings with target hidden states
        self.eh_proj = ColumnParallelLinear(
            input_size=config.hidden_size * 2,
            output_size=config.hidden_size,
            bias=False,
            gather_output=True,
            params_dtype=config.dtype if hasattr(config, "dtype") else torch.bfloat16,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Normalize both inputs before fusion
        inputs_embeds_normed = self.enorm(inputs_embeds)
        previous_hidden_states_normed = self.hnorm(previous_hidden_states)

        # Fuse via concatenation and linear projection
        fused = torch.cat([inputs_embeds_normed, previous_hidden_states_normed], dim=-1)
        hidden_states, _ = self.eh_proj(fused)

        # Call parent forward (Attention)
        # Parent forward expects: hidden_states, residual
        # It applies self.norm(hidden_states) then self.mixer(hidden_states)
        # We pass our fused hidden_states as input.
        hidden_states, residual = super().forward(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        return hidden_states, residual


class NemotronHMTPMoEDecoderLayer(NemotronHMoEDecoderLayer):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=prefix,
        )

        self.final_layernorm = RMSNorm(
            config.hidden_size,
            eps=getattr(config, "layer_norm_epsilon", 1e-5),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Call parent forward (MoE)
        hidden_states, residual = super().forward(
            hidden_states=hidden_states,
            residual=residual,
            **kwargs,
        )

        # Collapse residual before final norm if necessary
        if residual is not None:
            hidden_states = hidden_states + residual

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class NemotronHMTPLayer(nn.Module):
    """
    Container layer for one MTP step.
    Contains an Attention layer (with fusion) and an MoE layer (with final norm).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = cast(NemotronHConfig, config)
        self.config_mtp = copy.deepcopy(self.config)

        self.attn = NemotronHMTPAttentionDecoderLayer(
            config=self.config_mtp,
            layer_idx=layer_idx,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            parallel_config=vllm_config.parallel_config,
            prefix=f"{prefix}.attn",
        )

        self.moe = NemotronHMTPMoEDecoderLayer(
            config=self.config_mtp,
            layer_idx=layer_idx,
            model_config=vllm_config.model_config,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            parallel_config=vllm_config.parallel_config,
            prefix=f"{prefix}.moe",
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        hidden_states, residual = self.attn(
            inputs_embeds=inputs_embeds,
            positions=positions,
            previous_hidden_states=previous_hidden_states,
            residual=None,
        )

        hidden_states = self.moe(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
        )

        return hidden_states


@support_torch_compile
class NemotronHMultiTokenPredictor(nn.Module):
    """MTP predictor with NemotronH layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): NemotronHMTPLayer(
                    vllm_config=vllm_config,
                    layer_idx=idx,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(self.num_mtp_layers)
            }
        )

        self.make_empty_intermediate_tensors: Callable[..., IntermediateTensors] = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None, (
            "embed_tokens not initialized - must be shared from target model"
        )
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        # Use the MTP layer (cycling for multi-step)
        layer_idx_str = str(spec_step_idx % self.num_mtp_layers)
        hidden_states = self.layers[layer_idx_str](
            inputs_embeds,
            positions,
            hidden_states,
            spec_step_idx,
        )

        return hidden_states


@support_torch_compile
class NemotronHMTP(nn.Module, SupportsPP):
    """NemotronH MTP model."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = vllm_config.quant_config

        # Needed for load_weights mapping
        self.mtp_start_layer_idx = config.num_hidden_layers

        # EPLB config for experts
        self.num_redundant_experts = 0
        if vllm_config.parallel_config and vllm_config.parallel_config.eplb_config:
            self.num_redundant_experts = (
                vllm_config.parallel_config.eplb_config.num_redundant_experts
            )

        assert not cache_config.enable_prefix_caching, (
            "NemotronHMTP currently does not support prefix caching"
        )

        # MTP predictor
        self.model = NemotronHMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        # LM head for generating logits
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward - applies attention-based MTP."""
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """Compute logits for DRAFT token generation."""
        assert self.lm_head is not None, (
            "lm_head not initialized - must be shared from target model"
        )
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load MTP weights with proper name remapping."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        if self.config.mtp_hybrid_override_pattern == "*E":
            layers_unravel_mapping = {
                ".eh_proj": ".attn.eh_proj",
                ".enorm": ".attn.enorm",
                ".hnorm": ".attn.hnorm",
                ".final_layernorm": ".moe.final_layernorm",
                "layers.0.layers.0": "layers.0.attn",
                "layers.0.layers.1": "layers.0.moe",
                "layers.1.layers.0": "layers.1.attn",
                "layers.1.layers.1": "layers.1.moe",
            }
        else:
            raise RuntimeError(
                f"SMOR: unsupported mtp_hybrid_override_pattern: {self.config.mtp_hybrid_override_pattern}"
            )

        expert_params_mapping = []
        if hasattr(self.config, "n_routed_experts") and self.config.n_routed_experts:
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="",  # Empty - non-gated MoE
                num_experts=self.config.n_routed_experts,
                num_redundant_experts=self.num_redundant_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # if get_tensor_model_parallel_rank() == 0:
        #     print(f"SMOR: params_dict.keys: {params_dict.keys()}")

        for name, loaded_weight in weights:
            # Only process MTP weights - skip all non-MTP weights
            if (
                not name.startswith("mtp.")
                and "embeddings" not in name
                and "lm_head" not in name
            ):
                continue
            # Skip rotary embeddings (computed, not loaded)
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                name = (
                    "model." + name[4:]
                )  # Remove "mtp." prefix, now "model.layers.X..."

            # TODO smor- a truly ugly solution, replace it
            for k, v in layers_unravel_mapping.items():
                if k in name:
                    name = name.replace(k, v)
                    break

            if "embeddings" in name:
                name = name.replace("embeddings", "embed_tokens")
                if name.startswith("backbone."):
                    name = name.replace("backbone.", "model.")

            # Handle stacked parameters (qkv_proj) for attention layers
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Must be in a mixer (attention layer)
                # Now it is inside attn.mixer
                if ".mixer." not in name:
                    continue

                is_stacked = True
                stacked_name = name.replace(weight_name, param_name)

                if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                    continue

                if stacked_name not in params_dict:
                    # Might be that mapping failed or param doesn't exist
                    # print(f"SMOR: Warning stacked param {stacked_name} not in dict")
                    continue

                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(stacked_name)
                break

            if is_stacked:
                continue

            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                # weight_name is like "experts.0.up_proj."
                if weight_name not in name:
                    continue

                is_expert_weight = True

                # Replace the expert-specific weight name with fused parameter name
                # e.g., "experts.0.up_proj." -> "experts.w13_"
                name_mapped = name.replace(weight_name, param_name)

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                success = weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    loaded_params.add(name_mapped)
                break

            if is_expert_weight:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue

            if name not in params_dict:
                # print(f"SMOR: Warning param {name} not in dict")
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
