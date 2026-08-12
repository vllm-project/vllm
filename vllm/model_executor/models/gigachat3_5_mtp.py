# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GigaChat 3.5 Multi-Token Prediction (NextN) draft model.

Mirrors the DeepSeek-V3 NextN layout (``deepseek_mtp.py``) so the checkpoint's
``model.layers.{num_hidden_layers + i}.*`` weights map cleanly, but uses
GigaChat's gated norms, a dense MLP (``nextn_is_sparse=false``), and a
full-attention (MLA) draft block. The draft carries no GDN/mamba state, so this
model is neither ``IsHybrid`` nor ``MixtureOfExperts``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

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
from vllm.transformers_utils.configs.gigachat3_5 import GigaChat35Config

from .deepseek_v2 import get_spec_layer_idx_from_weight_name
from .gigachat3_5 import (
    GigaChat35DecoderLayer,
    _build_norm,
)
from .utils import maybe_prefix

logger = init_logger(__name__)


class GigaChat35SharedHead(nn.Module):
    def __init__(
        self,
        config: GigaChat35Config,
        prefix: str,
        quant_config=None,
    ) -> None:
        super().__init__()
        self.norm = _build_norm(config, config.hidden_size)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class GigaChat35MultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = _build_norm(config, config.hidden_size)
        self.hnorm = _build_norm(config, config.hidden_size)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = GigaChat35SharedHead(
            config=config, prefix=f"{prefix}.shared_head", quant_config=quant_config
        )
        # The NextN block is always a full-attention (MLA) layer with a dense
        # MLP (nextn_is_sparse=false); it carries no GDN/mamba state.
        if getattr(config, "nextn_is_sparse", False):
            raise NotImplementedError(
                "GigaChat 3.5 MTP with nextn_is_sparse=True (a MoE NextN "
                "block) is not supported: expert weights are not loaded for "
                "the draft model."
            )
        self.mtp_block = GigaChat35DecoderLayer(
            vllm_config,
            prefix=prefix,
            layer_type="full_attention",
            is_nextn=True,
            config=config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_embeds is not None
        # Position 0 has no previous token to predict from.
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        # GigaChat35DecoderLayer returns the residual-added hidden (single
        # tensor), i.e. the pre-final-norm hidden used for logits.
        hidden_states = self.mtp_block(positions=positions, hidden_states=hidden_states)
        # Return (pre-final-norm, post-final-norm); compute_logits applies the
        # final norm again to the pre-norm element so each path gets one norm.
        return hidden_states, self.shared_head(hidden_states)


class GigaChat35MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): GigaChat35MultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "hidden_states": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class GigaChat35MTP(nn.Module):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = GigaChat35MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """Add ``.mtp_block.`` for transformer-block modules and hoist shared
        weights (embed_tokens) to top level, matching the module hierarchy."""
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        elif shared_weight:
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                name = name_mapped
                break
            else:
                # Shared weights (embed_tokens) live under model.* and are only
                # present once; skip duplicates from later NextN layers.
                if (
                    spec_layer != self.model.mtp_start_layer_idx
                    and ".layers" not in name
                ):
                    continue
                if name not in params_dict:
                    logger.warning_once(
                        "MTP parameter %s not found in params_dict, skip loading",
                        name,
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Validate that weights were loaded for each expected NextN layer, so a
        # checkpoint that declares N MTP layers but ships fewer (e.g. a config
        # with num_nextn_predict_layers=2 that only exports layer 40) fails
        # loudly instead of silently under-loading the draft model.
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers:
                raise ValueError(
                    f"GigaChat 3.5 MTP layer {layer_idx} weights are missing "
                    f"from the checkpoint. Re-export the checkpoint with all "
                    f"NextN layers, set num_nextn_predict_layers to the number "
                    f"actually present, or reduce num_speculative_tokens."
                )
        return loaded_params


EntryClass = [GigaChat35MTP]
