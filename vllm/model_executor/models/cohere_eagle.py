# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import CohereConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.commandr import (
    CohereDecoderLayer,
    CohereForCausalLM,
    LayerNorm,
)

from .utils import AutoWeightsLoader, get_draft_quant_config, maybe_prefix

logger = init_logger(__name__)


class CohereEagleDecoderLayer(CohereDecoderLayer):
    """Eagle draft variant of CohereDecoderLayer.

    Optionally replaces ``input_layernorm`` with ``nn.Identity`` on the very
    first draft layer to mirror the original EAGLE implementation.

    Reference:
    https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
    """

    def __init__(
        self,
        config: CohereConfig,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        disable_input_layernorm: bool = False,
    ) -> None:
        super().__init__(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        if disable_input_layernorm:
            del self.input_layernorm
            self.input_layernorm = nn.Identity()


@support_torch_compile
class CohereEagleModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)

        # Cohere2-targeted EAGLE drafts inherit the target's sliding-window
        # attention pattern. ``CohereAttention`` resolves per-layer behavior
        # via ``config.layer_types[layer_idx]`` and the eagle layers use
        # absolute indices (target_layer_num + i), so prepend the target's
        # ``layer_types`` to the draft's so the lookup succeeds.
        target_text_config = vllm_config.model_config.hf_text_config
        if hasattr(target_text_config, "layer_types") and hasattr(
            self.config, "layer_types"
        ):
            self.config.layer_types = list(target_text_config.layer_types) + list(
                self.config.layer_types
            )

        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                CohereEagleDecoderLayer(
                    self.config,
                    cache_config=vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )

        # Cohere EAGLE checkpoints include a bias term on the input fusion
        # projection (unlike LLaMA EAGLE which uses bias=False).
        self.fc = ReplicatedLinear(
            input_size=self.config.hidden_size * 2,
            output_size=self.config.hidden_size,
            bias=True,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )

        # Cohere EAGLE applies an explicit final LayerNorm to the draft
        # hidden states before they are consumed by the logits processor.
        self.norm = LayerNorm(
            param_shape=(self.config.hidden_size),
            eps=self.config.layer_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states

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
            if "rotary_emb.inv_freq" in name:
                continue

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # When PP is disabled the draft model shares its token
                # embeddings with the target model, so skip loading any
                # embed_tokens weights coming from the EAGLE checkpoint.
                if get_pp_group().world_size == 1 and "embed_tokens." in name:
                    continue

                # lm_head is tied with embed_tokens; skip to avoid duplicate
                # loads.
                if "lm_head.weight" in name:
                    continue

                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class EagleCohereForCausalLM(CohereForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = CohereEagleModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support multimodal inputs yet."
            )
        return self.model(input_ids, positions, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head.", "model.embed_tokens."]
                if self.config.tie_word_embeddings
                else None
            ),
        )

        loaded_weight_names = loader.load_weights(weights)

        # Embed tokens are tied with the target model and therefore not
        # present in the EAGLE checkpoint; mark them as loaded explicitly to
        # avoid a spurious "weight not found" warning from the default
        # weight loader.
        loaded_weight_names.add("model.embed_tokens.weight")
        return loaded_weight_names
