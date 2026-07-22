# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Bailing MoE v2.5 MTP model."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.bailing_moe_linear import (
    BailingMoeV25,
    BailingMoeV25MLAAttention,
)
from vllm.sequence import IntermediateTensors

from .utils import PPMissingLayer, is_pp_missing_parameter, maybe_prefix


def _get_draft_hf_config(vllm_config: VllmConfig) -> PretrainedConfig:
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None:
        draft_model_config = speculative_config.draft_model_config
        if draft_model_config is not None:
            return draft_model_config.hf_config
    return vllm_config.model_config.hf_config


class BailingMTPSharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class BailingMoeV25MultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        layer_id: int,
    ) -> None:
        super().__init__()
        config = _get_draft_hf_config(vllm_config)
        self.config = config
        self.layer_id = layer_id
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = BailingMoeV25MLAAttention(
            config,
            quant_config=vllm_config.quant_config,
            layer_id=layer_id,
            prefix=maybe_prefix(prefix, "self_attn"),
            cache_config=vllm_config.cache_config,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = BailingMoeV25(
            config,
            quant_config=vllm_config.quant_config,
            layer_id=layer_id,
            prefix=maybe_prefix(prefix, "mlp"),
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.shared_head = BailingMTPSharedHead(
            config,
            maybe_prefix(prefix, "shared_head"),
            vllm_config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states.to(residual.device)
        return self.final_layernorm(hidden_states)


class BailingMoeV25MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = _get_draft_hf_config(vllm_config)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): BailingMoeV25MultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                    idx,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = PPMissingLayer()
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
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
        lm_head: nn.Module | None = None,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        head = lm_head if lm_head is not None else mtp_layer.shared_head.head
        return self.logits_processor(
            head,
            mtp_layer.shared_head(hidden_states),
        )


@support_torch_compile
class BailingMoeV25MTPModel(nn.Module):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = _get_draft_hf_config(vllm_config)
        self.lm_head: nn.Module | None = None
        self.model = BailingMoeV25MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states, spec_step_idx, self.lm_head)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        expert_params_mapping = list(self.get_expert_mapping())
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        loaded_mtp_layers: set[int] = set()

        def load_param(
            name: str,
            loaded_weight: torch.Tensor,
            shard_id=None,
        ) -> bool:
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                return False
            if name not in params_dict or is_pp_missing_parameter(name, self):
                return False

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            elif isinstance(shard_id, int):
                weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader(
                    param,
                    loaded_weight,
                    name,
                    expert_id=shard_id[0],
                    shard_id=shard_id[1],
                )
            loaded_params.add(name)
            return True

        def get_spec_layer_idx(name: str) -> int | None:
            if not name.startswith("model.layers."):
                return None
            try:
                layer_idx = int(name.split("model.layers.", 1)[1].split(".", 1)[0])
            except (IndexError, ValueError):
                return None
            mtp_idx = layer_idx - self.config.num_hidden_layers
            if 0 <= mtp_idx < self.config.num_nextn_predict_layers:
                return layer_idx
            return None

        def normalize_name(name: str) -> str:
            name = name.replace(".attention.dense", ".self_attn.o_proj")
            name = name.replace(".attention.", ".self_attn.")
            return name.replace(
                "mlp.gate.e_score_correction_bias",
                "mlp.gate.expert_bias",
            )

        def load_lm_head(loaded_weight: torch.Tensor) -> None:
            for layer_idx in range(
                self.model.mtp_start_layer_idx,
                self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
            ):
                name = f"model.layers.{layer_idx}.shared_head.head.weight"
                load_param(name, loaded_weight)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name == "model.word_embeddings.weight":
                load_param("model.embed_tokens.weight", loaded_weight)
                continue
            if name == "lm_head.weight":
                load_lm_head(loaded_weight)
                continue

            spec_layer = get_spec_layer_idx(name)
            if spec_layer is None:
                continue
            name = normalize_name(name)

            loaded = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts." in name and name not in params_dict:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if load_param(mapped_name, loaded_weight, shard_id):
                    loaded = True
                    break
            if loaded:
                loaded_mtp_layers.add(spec_layer)
                continue

            if "mlp.experts" in name:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if load_param(
                        mapped_name,
                        loaded_weight,
                        (expert_id, shard_id),
                    ):
                        loaded = True
                        break
                if loaded:
                    loaded_mtp_layers.add(spec_layer)
                continue

            if load_param(name, loaded_weight):
                loaded_mtp_layers.add(spec_layer)

        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if (
                layer_idx not in loaded_mtp_layers
                and is_mtp_completeness_check_enabled()
            ):
                raise ValueError(
                    f"Bailing MTP speculative decoding layer {layer_idx} "
                    "weights are missing from checkpoint. Use a checkpoint "
                    "that includes MTP layer weights, or disable speculative "
                    "decoding."
                )
        return loaded_params
