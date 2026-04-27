# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.llama import (
    LlamaDecoderLayer,
    LlamaModel,
)
from vllm.model_executor.models.mistral import MistralForCausalLM
from vllm.model_executor.models.utils import (
    _merge_multimodal_embeddings,
    maybe_prefix,
)

logger = init_logger(__name__)


@support_torch_compile
class EagleMistralModel(LlamaModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
    ) -> None:
        # Bypass LlamaModel.__init__ to avoid creating duplicate attention
        # layer entries in the global context.
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = vllm_config.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
            quant_config=self.quant_config,
        )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
                    config=self.config,
                )
                for i in range(self.config.num_hidden_layers)
            ]
        )
        self.fc = RowParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            bias=False,
            input_is_parallel=False,
            quant_config=self.quant_config,
            return_bias=False,
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Store weight scales for bf16->fp8 conversion on the fly.
        # Needs to persist across multiple `load_weights` calls.
        self._loaded_weight_scales: dict[str, torch.Tensor] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
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
        # Pretend embed_tokens is loaded; the actual weight is shared
        # from the target model at runtime by `load_eagle_model`.
        return super().load_weights(weights) | {"embed_tokens.weight"}


class EagleMistralForCausalLM(MistralForCausalLM):
    mistral_mapping = MistralForCausalLM.mistral_mapping | {
        "eagle_linear": "model.fc",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Bypass MistralForCausalLM.__init__ to use the draft model config
        # and to avoid creating an lm_head.
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        # Draft model quantization config may differ from the target model.
        self.quant_config = VllmConfig.get_quantization_config(
            vllm_config.speculative_config.draft_model_config, vllm_config.load_config
        )
        vllm_config.quant_config = self.quant_config
        self.model = EagleMistralModel(
            vllm_config=vllm_config, prefix="model", start_layer_id=target_layer_num
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = super().embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        assert is_multimodal is not None

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
