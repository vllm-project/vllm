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
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.commandr import (
    CohereDecoderLayer,
    CohereForCausalLM,
    select_norm_impl,
)

from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class CohereDecoderLayer(CohereDecoderLayer):
    def __init__(
        self,
        config: CohereConfig,
        quant_config: QuantizationConfig | None = None,
        # disable_input_layernorm: bool,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
        )

        # # Skip the input_layernorm
        # # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        # if disable_input_layernorm:
        #     del self.input_layernorm
        #     self.input_layernorm = nn.Identity()


@support_torch_compile
class CohereModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        # COHERE: extend config.layer_types with eagle layers,
        # since sliding window pattern is decided by config.layer_types
        # in CohereAttention.
        # COHERE STARTS
        self.config.layer_types = (
            vllm_config.model_config.hf_text_config.layer_types
            + self.config.layer_types
        )
        assert len(self.config.layer_types) == (
            vllm_config.model_config.hf_text_config.num_hidden_layers
            + self.config.num_hidden_layers
        )
        # COHERE ENDS
        self.quant_config = quant_config

        self.vocab_size = self.config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList(
            [
                CohereDecoderLayer(
                    self.config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )

        self.fc = ReplicatedLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            bias=True,  # COHERE: use bias
            prefix=maybe_prefix(prefix, "fc"),
            quant_config=quant_config,
        )

        self.use_last_layernorm = True
        if self.use_last_layernorm:
            norm_cls, norm_eps = select_norm_impl(self.config)
            self.norm = norm_cls(param_shape=(self.config.hidden_size), eps=norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_embeds = self.embed_tokens(input_ids)
        hidden_states, _ = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        if self.use_last_layernorm:
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states, hidden_states
        else:
            return hidden_states + residual

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
            # Skip loading rotary embeddings since vLLM has its own
            if "rotary_emb.inv_freq" in name:
                continue

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
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
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # if PP disabled then draft will share embed with target
                if get_pp_group().world_size == 1 and "embed_tokens." in name:
                    continue

                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue

                # Skip loading extra bias for GPTQ models.
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
        # draft model quantization config may differ from target model
        quant_config = VllmConfig.get_quantization_config(
            vllm_config.speculative_config.draft_model_config, vllm_config.load_config
        )
        self.model = CohereModel(
            vllm_config=vllm_config,
            prefix="eagle_draft_model",  # cohere
            start_layer_id=target_layer_num,
            quant_config=quant_config,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        model_weights = {}
        # COHERE: our weights doesn't skip "model."" prefix
        # target model has "model." prefix but vLLM skipped
        # it for llama draft model
        for name, loaded_weight in weights:
            # if "lm_head" not in name:
            #     name = "model." + name
            model_weights[name] = loaded_weight

        loaded_weight_name_list = loader.load_weights(model_weights.items())

        # we dont load embed tokens for draft model because
        # it will share embed tokens from target model.
        # However, non quant weight loading will complain
        # that embed_tokens is defined but not found in
        # model checkpoint so forcefully adding to the list
        # to avoid mismatch error in default_loader.py load_weights()
        loaded_weight_name_list.add("model.embed_tokens.weight")
        return loaded_weight_name_list
