# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformers modeling backend mixin for legacy models."""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.models.utils import WeightsMapper
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class LegacyMixin:
    hf_to_vllm_mapper = WeightsMapper(
        # These are applied in order, so the order matters!
        orig_to_new_prefix={
            # Handle BERT-like models
            "roberta": "model",
            "bert": "model",
        },
        orig_to_new_suffix={
            # Replace legacy suffixes used for norms
            ".gamma": ".weight",
            ".beta": ".bias",
        },
    )

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Skip unsupported/unwanted output embeddings layers
        self.skip_prefixes.extend(
            [
                "model.lm_head.",
                "model.predictions.",
                "model.qa_outputs.",
                "model.embeddings_project.",
                "model.discriminator_predictions.",
            ]
        )

        # Some encoder models have the position_ids buffer in the checkpoint.
        # vLLM will always pass position_ids as an argument, so we skip loading
        # the buffer if it exists
        self.skip_substrs.append("position_ids")

        # Some encoder models have the bias of the final classifier layer
        # in the checkpoint. vLLM does not use this bias, so we skip loading
        # it if it exists
        self.skip_substrs.append("score.bias")

        # roberta-like models an extra padding in positions.
        # FIXME(Isotr0py): This is quite hacky for roberta edge case,
        # we should find a better way to handle this.
        self.is_roberta = "roberta" in self.text_config.model_type
        self.padding_idx = self.text_config.pad_token_id

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self.is_roberta:
            # RoBERTa-specific positions padding
            positions += self.padding_idx + 1
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
