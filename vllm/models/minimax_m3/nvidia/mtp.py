# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import (
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .model import (
    MiniMAXGemmaRMSNorm,
    MiniMaxM3DecoderLayer,
)


class MiniMaxM3MultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config

        self.enorm = MiniMAXGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = MiniMAXGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.transformer_layer = MiniMaxM3DecoderLayer(
            vllm_config=vllm_config,
            prefix=prefix,
            force_sparse_attn=True,
            force_moe=True,
            is_mtp_block=True,
        )
        self.final_layernorm = MiniMAXGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
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
        # Mask out inputs at position 0, as not needed by MTP.
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)

        # Combine the normalized token embeddings with the normalized
        # previous hidden states.
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states, _ = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        # Apply transformer layer.
        hidden_states, residual = self.transformer_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )

        hidden_states += residual
        return hidden_states


class MiniMaxM3MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        assert vllm_config.speculative_config is not None
        # Use the draft (MTP) config, not the target model's. This is flat for a
        # standalone checkpoint, and the promoted text_config for a bundled one.
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.num_mtp_layers = config.num_mtp_modules
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): MiniMaxM3MultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(self.num_mtp_layers)
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

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
        return self.layers[str(current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )


class MiniMaxM3MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = MiniMaxM3MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

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
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        current_step_idx = spec_step_idx % self.model.num_mtp_layers
        mtp_layer = self.model.layers[str(current_step_idx)]
        return self.logits_processor(
            self.lm_head, mtp_layer.final_layernorm(hidden_states)
        )

    def _get_mtp_layer_idx_from_weight_name(self, name: str) -> int | None:
        """Return the MTP layer index in *.mtp.layers.{idx}.*, else None."""
        match = re.search(r"\.mtp\.layers\.(\d+)\.", name)
        return int(match.group(1)) if match else None

    def _map_checkpoint_name(self, name: str) -> str | None:
        """Map a full checkpoint key to this MTP module's parameter name.

        The MTP module only owns the *.mtp.layers.* weights plus the token
        embedding and LM head, which the checkpoint shares with the main model.
        Everything else belongs to other modules and is ignored here by returning
        None.
        """
        # In the bundled checkpoint, the MTP weights are prefixed with
        # "language_model". The standalone MTP checkpoint has no such prefix.
        # Strip it if present.
        name = name.removeprefix("language_model.")

        if name == "model.embed_tokens.weight":
            return "model.embed_tokens.weight"
        if name == "lm_head.weight":
            return "lm_head.weight"
        if "model.mtp.layers" in name:
            if "weight_scale_inv" in name:
                # The checkpoint stores block scales as "weight_scale_inv".
                # The ModelOpt MXFP8 layers expose them as "weight_scale".
                name = name.replace("weight_scale_inv", "weight_scale")
            # Strip "mtp" from prefix.
            return name.replace(".mtp.", ".")
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Map q/k/v projections to qkv_proj, and gate/up projections to gate_up_proj.
        stacked_params_mapping: list[tuple[str, str, int | str]] = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".qkv_proj", ".index_q_proj", "index_q"),
            (".qkv_proj", ".index_k_proj", "index_k"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        # Map expert weights w1/w2/w3 to gate/down/up.
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_mtp_layers: set[int] = set()
        for name, loaded_weight in weights:
            mtp_layer = self._get_mtp_layer_idx_from_weight_name(name)
            mapped_name = self._map_checkpoint_name(name)
            if mapped_name is None:
                # This weight does not belong to the MTP module, so skip it.
                continue
            name = mapped_name

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # Routed experts (w1/w2/w3) are handled below. Don't let the
                # stacked mapping rewrite them.
                if ("block_sparse_moe.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue

                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None or remapped_name not in params_dict:
                        continue
                    name = remapped_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

            loaded_params.add(name)
            if mtp_layer is not None:
                loaded_mtp_layers.add(mtp_layer)

        # Validate that weights were loaded for each MTP layer.
        for layer_idx in range(self.model.num_mtp_layers):
            if layer_idx not in loaded_mtp_layers:
                raise ValueError(
                    f"Failed to load MTP layer {layer_idx} weights from checkpoint."
                )

        return loaded_params
