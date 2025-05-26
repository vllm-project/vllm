# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .deepseek_v2 import (DeepseekV2DecoderLayer,
                          get_spec_layer_idx_from_weight_name)
from .interfaces import SupportsPP
from .utils import maybe_prefix


class SharedHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(config.vocab_size,
                                   config.hidden_size,
                                   quant_config=quant_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DeepSeekMultiTokenPredictorLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2,
                                 config.hidden_size,
                                 bias=False)
        self.shared_head = SharedHead(config=config, quant_config=quant_config)
        self.mtp_block = DeepseekV2DecoderLayer(config, prefix, model_config,
                                                cache_config, quant_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds[positions == 0] = 0
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        hidden_states, residual = self.mtp_block(positions=positions,
                                                 hidden_states=hidden_states,
                                                 residual=None)
        hidden_states = residual + hidden_states
        return hidden_states


class DeepSeekMultiTokenPredictor(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict({
            str(idx):
            DeepSeekMultiTokenPredictorLayer(
                config,
                f"{prefix}.layers.{idx}",
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
            )
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
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
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        mtp_layer = self.layers[str(self.mtp_start_layer_idx +
                                    current_step_idx)]
        logits = self.logits_processor(mtp_layer.shared_head.head,
                                       mtp_layer.shared_head(hidden_states),
                                       sampling_metadata)
        return logits


class DeepSeekMTP(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DeepSeekMultiTokenPredictor(vllm_config=vllm_config,
                                                 prefix=maybe_prefix(
                                                     prefix, "model"))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions,
                                   previous_hidden_states, inputs_embeds,
                                   spec_step_idx)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        return self.model.compute_logits(hidden_states, sampling_metadata,
                                         spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        """
        spec_layer_weight_names = [
            "embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head"
        ]
        spec_layer_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(f"model.layers.{spec_layer}.",
                                f"model.layers.{spec_layer}.mtp_block.")
        return name
