# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.model_executor.models.gemma2 import Gemma2Model
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors


class MyGemma2Embedding(nn.Module):

    is_pooling_model = True

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.model = Gemma2Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None

        self.pooler = DispatchPooler({
            "encode": Pooler.for_encode(pooler_config),
            "embed": Pooler.for_embed(pooler_config),
        })

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states

        # Return all-zero embeddings
        return torch.zeros_like(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        return self.model.load_weights(weights)
