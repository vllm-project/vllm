# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors

from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        config: LlamaConfig | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__(vllm_config, prefix=prefix, config=config)

        config = config or vllm_config.model_config.hf_config
        quant_config = self.get_quant_config(vllm_config)

        # First layer uses 2*hidden_size (embeds + hidden_states concatenated)
        # Subsequent layers use hidden_size (only hidden_states, no embeds)
        qkv_input_size = 2 * self.hidden_size if layer_idx == 0 else self.hidden_size

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            qkv_input_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "qkv_proj"),
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

        if getattr(config, "norm_before_residual", False):
            self._residual_norm = self._norm_before_residual
        else:
            self._residual_norm = self._norm_after_residual

    def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
        """Use drafter's quantization config instead of verifier's."""
        draft_model_config = vllm_config.speculative_config.draft_model_config
        draft_load_config = vllm_config.load_config

        return (
            VllmConfig.get_quantization_config(draft_model_config, draft_load_config)
            if draft_model_config
            else None
        )

    def _norm_before_residual(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.hidden_norm(hidden_states)
        residual = hidden_states
        return hidden_states, residual

    def _norm_after_residual(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layer_idx == 0:
            # First layer: concatenate embeds with hidden_states
            embeds = self.input_layernorm(embeds)
            hidden_states, residual = self._residual_norm(hidden_states=hidden_states)
            hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        else:
            # Subsequent layers: process hidden_states and residuals only
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile(
    # torch.compile is disabled for multimodal EAGLE3 models due to constraint
    # violations with dynamic shapes during tensor concatenation operations.
    # See: https://github.com/vllm-project/vllm/pull/22872/files#r2362028132
    # Non-multimodal EAGLE3 models can still use torch.compile safely.
    enable_if=lambda vllm_config: not MULTIMODAL_REGISTRY.supports_multimodal_inputs(
        vllm_config.model_config
    ),
)
class LlamaModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if hasattr(self.config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                self.config.target_hidden_size * 3, self.config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                self.config.hidden_size * 3, self.config.hidden_size, bias=False
            )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_embeds is None:
            input_embeds = self.get_input_embeddings(input_ids)
        assert hidden_states.shape[-1] == input_embeds.shape[-1]

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                embeds=input_embeds,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Eagle3LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        # Ensure draft_vocab_size is set
        # default to the base vocab size when absent
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )

        # Store target layer count in draft config for
        # proper layer_types indexing in draft models
        self.config.target_layer_count = target_layer_num
        self.model = LlamaModel(
            vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                "Expected logits to have shape "
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # combine multiple auxiliary hidden states returned by eagle3
        return self.model.fc(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
