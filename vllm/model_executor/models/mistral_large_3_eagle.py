# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from functools import partial

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2Model,
)
from vllm.model_executor.models.mistral_large_3 import MistralLarge3ForCausalLM

from .interfaces import SupportsMultiModal
from .utils import make_empty_intermediate_tensors_factory, maybe_prefix

logger = init_logger(__name__)


@support_torch_compile
class EagleMistralLarge3Model(DeepseekV2Model):
    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", start_layer_id: int = 0
    ):
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config

        self.vocab_size = config.vocab_size

        assert get_pp_group().world_size == 1
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.start_layer = 0
        self.end_layer = self.config.num_hidden_layers

        self.fc = RowParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            bias=False,
            input_is_parallel=False,
            quant_config=quant_config,
            return_bias=False,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        inputs_embeds = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        output = super().forward(
            input_ids, positions, intermediate_tensors=None, inputs_embeds=inputs_embeds
        )
        assert isinstance(output, torch.Tensor)
        return output


class EagleMistralLarge3ForCausalLM(MistralLarge3ForCausalLM):
    remapping = MistralLarge3ForCausalLM.remapping | {
        r"eagle_linear\.weight": r"model.fc.weight",
        r"eagle_linear\.qscale_act": r"model.fc.input_scale",
        r"eagle_linear\.qscale_weight": r"model.fc.weight_scale",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        vllm_config.model_config = vllm_config.speculative_config.draft_model_config
        # draft model quantization config may differ from target model
        self.quant_config = VllmConfig.get_quantization_config(
            vllm_config.speculative_config.draft_model_config, vllm_config.load_config
        )
        vllm_config.quant_config = self.quant_config
        self.model_cls = partial(
            EagleMistralLarge3Model, start_layer_id=target_layer_num
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    embed_input_ids = SupportsMultiModal.embed_input_ids  # type: ignore

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.model(input_ids, positions, hidden_states, inputs_embeds)
        return hidden_states, hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Pretend we've loaded the embedding and lm_head weights
        # (later copied from target model)
        return super().load_weights(weights) | {
            "model.embed_tokens.weight",
            "lm_head.weight",
        }
