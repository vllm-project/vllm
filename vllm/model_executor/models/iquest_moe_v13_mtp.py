# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .iquest_moe_v13 import (
    IquestMoeAttention,
    IquestMoEBlock,
    IquestMoeRMSNorm,
)
from .utils import is_pp_missing_parameter, maybe_prefix

logger = init_logger(__name__)


def get_spec_layer_idx_from_name(weight_name: str) -> int:
    spec_layer_idx = weight_name.split(".")[1]
    return int(spec_layer_idx)


class IquestMoeV13MTPInnerLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sandwich_norm: bool = False,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size

        self.self_attn = IquestMoeAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = IquestMoEBlock(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.attention_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_out_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.use_sandwich_norm = use_sandwich_norm
        if self.use_sandwich_norm:
            self.ffn_out_norm = IquestMoeRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.attn_out_scale = getattr(config, "first_layer_attn_out_scale", 1.0)
            self.ffn_out_scale = getattr(config, "first_layer_ffn_out_scale", 1.0)
        else:
            self.attn_out_scale = getattr(config, "attn_out_scale", 1.0)
            self.ffn_out_scale = getattr(config, "ffn_out_scale", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        # NOTE(yxing): post-norm is different for first layer and non-first layers
        if self.use_sandwich_norm:
            norm_hidden_states = self.attention_norm(hidden_states)
            attn_output = self.self_attn(
                positions=positions, hidden_states=norm_hidden_states
            )
            h = hidden_states + self.attn_out_norm(attn_output) * self.attn_out_scale

            # fully connected
            hidden_states = self.mlp(self.feed_forward_norm(h))
            output = h + self.ffn_out_norm(hidden_states) * self.ffn_out_scale
            return output
        else:
            x = self.attention_norm(hidden_states)
            attn_output = self.self_attn(positions=positions, hidden_states=x)
            h = x + self.attn_out_norm(attn_output) * self.attn_out_scale

            # fully connected
            ffn_out = self.mlp(self.feed_forward_norm(h))
            output = h + ffn_out * self.ffn_out_scale
            return output


class IquestMoeV13MTPLayer(nn.Module):
    """One MTP module: enorm/hnorm + eh_proj + inner decoder block + final LN."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sandwich_norm: bool = False,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.enorm = IquestMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = IquestMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_model_layer = IquestMoeV13MTPInnerLayer(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mtp_model_layer",
            use_sandwich_norm=use_sandwich_norm,
        )
        self.final_layernorm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states = self.mtp_model_layer(positions, hidden_states)
        return self.final_layernorm(hidden_states)


@support_torch_compile
class IquestMoeV13MTPFirstLayer(IquestMoeV13MTPLayer):
    """Compiled entry point for the structurally distinct first MTP layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            use_sandwich_norm=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(positions, previous_hidden_states, inputs_embeds)


@support_torch_compile
class IquestMoeV13MTPNextLayer(IquestMoeV13MTPLayer):
    """Compiled entry point shared by non-first MTP layer instances."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            use_sandwich_norm=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(positions, previous_hidden_states, inputs_embeds)


class IquestMoeV13MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_mtp_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        from vllm.compilation.backends import set_model_tag

        layers: dict[str, nn.Module] = {}
        for idx in range(
            self.mtp_start_layer_idx,
            self.num_mtp_layers + self.mtp_start_layer_idx,
        ):
            relative_idx = idx - self.mtp_start_layer_idx
            layer_cls = (
                IquestMoeV13MTPFirstLayer
                if relative_idx == 0
                else IquestMoeV13MTPNextLayer
            )
            # Each independently compiled MTP layer needs a distinct backend
            # cache namespace. The outer draft model is tagged as eagle_head.
            with set_model_tag(f"iquest_mtp_layer_{relative_idx}"):
                layers[str(idx)] = layer_cls(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
        self.layers = nn.ModuleDict(layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.forward_mtp_layer(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def forward_mtp_layer(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None, (
                "IquestMoeV13 MTP requires input_ids when inputs_embeds is None"
            )
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            positions, previous_hidden_states, inputs_embeds
        )


@support_torch_compile
class IquestMoeV13MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.model = IquestMoeV13MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.mtp_start_layer_idx = self.config.num_hidden_layers

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

    def forward_mtp_layer(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Run one fixed MTP layer without entering the outer compiled graph."""
        return self.model.forward_mtp_layer(
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
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mtp_prefix = "mtp_layers."
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        enable_sink_attention = getattr(self.config, "enable_sink_attention", False)

        def _load_into(param_name: str, weight: torch.Tensor) -> None:
            param = params_dict[param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded_params.add(param_name)

        for name, loaded_weight in weights:
            # Shared weights loaded into the draft head.
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                _load_into(name, loaded_weight)
                continue

            if not name.startswith(mtp_prefix):
                continue

            # Rewrite mtp_layers.0.<rest> -> model.layers.0.<rest>
            spec_layer_idx = get_spec_layer_idx_from_name(name)
            name = name.replace(
                f"mtp_layers.{spec_layer_idx}",
                f"model.layers.{spec_layer_idx + self.mtp_start_layer_idx}",
            )

            # Fused QKV: match on the exact ".<component>.weight" suffix.
            for param_name, weight_name, shard_id in stacked_params_mapping:
                suffix = f".{weight_name}.weight"
                if not name.endswith(suffix):
                    continue
                mapped = name[: -len(suffix)] + f".{param_name}.weight"
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                # Sonic-MoE fused expert tensors: match exact ".experts.fc" /
                # ".experts.proj" component suffixes.
                if name.endswith(".mlp.experts.fc"):
                    mapped = name[: -len(".fc")] + ".w13_weight"
                    param = params_dict[mapped]
                    weight_loader = param.weight_loader
                    for expert_id in range(self.config.num_experts):
                        weight_loader(
                            param,
                            loaded_weight[expert_id][: self.config.intermediate_size],
                            mapped,
                            shard_id="w1",
                            expert_id=expert_id,
                        )
                        weight_loader(
                            param,
                            loaded_weight[expert_id][self.config.intermediate_size :],
                            mapped,
                            shard_id="w3",
                            expert_id=expert_id,
                        )
                    loaded_params.add(mapped)
                    continue
                if name.endswith(".mlp.experts.proj"):
                    mapped = name[: -len(".proj")] + ".w2_weight"
                    param = params_dict[mapped]
                    weight_loader = param.weight_loader
                    for expert_id in range(self.config.num_experts):
                        weight_loader(
                            param,
                            loaded_weight[expert_id],
                            mapped,
                            shard_id="w2",
                            expert_id=expert_id,
                        )
                    loaded_params.add(mapped)
                    continue

                if not enable_sink_attention and name.endswith(".sink_k"):
                    logger.warning_once("sink attention feature is disabled")
                    continue

                # Sonic-MoE router naming: ".mlp.router.weight" -> ".mlp.gate.weight".
                if name.endswith(".mlp.router.weight"):
                    name = name[: -len(".router.weight")] + ".gate.weight"

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning_once("Unexpected MTP weight skipped: %s", name)
                    continue
                _load_into(name, loaded_weight)

        return loaded_params
