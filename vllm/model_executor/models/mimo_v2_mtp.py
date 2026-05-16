# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only MiMo-V2 MTP (Multi-Token Prediction) draft model.

Supports both MiMo-V2-Pro and MiMo-V2-Flash checkpoints.

Checkpoint weight layout (model.mtp.layers.{idx}.*):
  enorm            - RMSNorm for token embeddings
  hnorm            - RMSNorm for previous hidden states
  eh_proj          - ReplicatedLinear(hidden*2 -> hidden)
  input_layernorm  - pre-attention RMSNorm
  self_attn.*      - attention weights; format differs by variant:
                       Pro:   fused qkv_proj  [Q;K;V] concatenated
                       Flash: separate q_proj, k_proj, v_proj
  pre_mlp_layernorm - post-attention / pre-MLP RMSNorm
  mlp.*            - dense MLP (gate_proj / up_proj / down_proj)
  final_layernorm  - norm applied before logit computation
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from .mimo_v2 import MiMoV2Attention, MiMoV2MLP
from .utils import _merge_multimodal_embeddings, maybe_prefix

# MiMo-V2 checkpoints contain multiple MTP layers, but vLLM currently supports
# only the first layer and only one speculative token.
_MIMO_V2_PRO_NUM_MTP_LAYERS = 1
_MIMO_V2_FLASH_NUM_MTP_LAYERS = 1


class MiMoV2MTPLayer(nn.Module):
    """Single MTP predictor layer for MiMo-V2 (Pro and Flash).

    Mirrors the single-layer MiMo-V2 nextn reference implementation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()

        # Predictor head components
        self.enorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        # MTP uses the SWA attention configuration
        # implementation.
        swa_rope_theta = getattr(
            config,
            "swa_rope_theta",
            getattr(config, "rope_theta", 1000000),
        )
        sliding_window_size = getattr(config, "sliding_window_size", -1)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attn = MiMoV2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.swa_num_attention_heads,
            num_kv_heads=config.swa_num_key_value_heads,
            head_dim=config.swa_head_dim,
            v_head_dim=getattr(config, "swa_v_head_dim", None),
            v_scale=getattr(config, "attention_value_scale", None),
            sliding_window_size=sliding_window_size,
            attention_bias=config.attention_bias,
            add_swa_attention_sink_bias=getattr(
                config, "add_swa_attention_sink_bias", False
            ),
            layer_id=0,
            rope_theta=swa_rope_theta,
            max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
            quant_config=quant_config,
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
            prefix=f"{prefix}.self_attn",
        )
        self.pre_mlp_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.mlp = MiMoV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Combine token embedding and previous hidden state
        h, _ = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1
            )
        )

        # Transformer block with fused residual norms
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn(positions=positions, hidden_states=h)
        h, residual = self.pre_mlp_layernorm(h, residual)
        h = self.mlp(h)
        h = h + residual

        return self.final_layernorm(h)


class _MiMoV2MTPLayers(nn.Module):
    """Thin wrapper so parameter paths match checkpoint: model.mtp.layers.*"""

    def __init__(
        self,
        config: PretrainedConfig,
        num_mtp_layers: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                str(i): MiMoV2MTPLayer(
                    config=config,
                    prefix=f"{prefix}.{i}",
                    quant_config=quant_config,
                )
                for i in range(num_mtp_layers)
            }
        )


class MiMoV2MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        spec_cfg = vllm_config.speculative_config
        assert spec_cfg is not None
        if spec_cfg.num_speculative_tokens != 1:
            raise ValueError(
                "MiMo-V2 MTP in vLLM only supports num_speculative_tokens=1."
            )
        num_mtp_layers = 1

        self.num_mtp_layers = num_mtp_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.mtp = _MiMoV2MTPLayers(
            config=config,
            num_mtp_layers=num_mtp_layers,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "mtp.layers"),
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
        assert spec_step_idx == 0, "MiMo-V2 MTP only supports one speculative token."
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.mtp.layers[str(spec_step_idx)](
            inputs_embeds, positions, previous_hidden_states
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        assert spec_step_idx == 0, "MiMo-V2 MTP only supports one speculative token."
        return self.logits_processor(lm_head, hidden_states)


class MiMoV2MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = MiMoV2MultiTokenPredictor(
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
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        assert spec_step_idx == 0, "MiMo-V2 MTP only supports one speculative token."
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        assert spec_step_idx == 0, "MiMo-V2 MTP only supports one speculative token."
        return self.model.compute_logits(hidden_states, self.lm_head, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # Flash format: separate projections → fused qkv_proj
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Only load MTP-related weights, shared embeddings, and lm_head
            if (
                "model.mtp" not in name
                and "model.embed_tokens" not in name
                and not name.startswith("lm_head")
            ):
                continue

            # Support fused qkv_proj checkpoint (Pro format).
            # The checkpoint is stored pre-sharded for TP=8 as
            # [Q_rank0, K_rank0, V_rank0, Q_rank1, ...], so splitting along
            # dim 0 with chunk(tp_size) gives each rank its Q+K+V slice for
            # both the FP8 weight and the block weight_scale_inv. This matches
            # how the main model loads the same layout.
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    default_weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue

            # gate_proj/up_proj → gate_up_proj stacking (both formats);
            # Flash: q_proj/k_proj/v_proj → qkv_proj merging.
            stacked_matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_rewritten = name.replace(weight_name, param_name)
                if (
                    name_rewritten.endswith(".bias")
                    and name_rewritten not in params_dict
                ):
                    continue
                if name_rewritten not in params_dict:
                    continue
                param = params_dict[name_rewritten]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_rewritten)
                stacked_matched = True
                break

            if stacked_matched:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            # attention_sink_bias is head-parallel; slice by tp
            if "attention_sink_bias" in name:
                total_heads = loaded_weight.shape[0]
                heads_per_rank = total_heads // tp_size
                loaded_weight = loaded_weight.narrow(
                    0, tp_rank * heads_per_rank, heads_per_rank
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class MiMoV2OmniMTP(MiMoV2MTP, SupportsMultiModal):
    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds
