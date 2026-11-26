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

from .base import Base

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class LegacyMixin(Base):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.hf_to_vllm_mapper |= WeightsMapper(
            # Drop unsupported/unwanted output embeddings layers.
            orig_to_new_prefix={
                "model.lm_head.": None,
                "model.predictions.": None,
                "model.qa_outputs.": None,
                "model.embeddings_project.": None,
                "model.discriminator_predictions.": None,
            },
            orig_to_new_substr={
                # Some encoder models have the position_ids buffer in the checkpoint.
                # vLLM always passes position_ids as an argument, so drop the buffer.
                "position_ids": None,
                # Some encoder models have the bias of the final classifier layer in
                # the checkpoint. vLLM does not use this bias, so drop it.
                "score.bias": None,
            },
        )

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
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if self.is_roberta:
            # RoBERTa positions start at padding_idx + 1.
            # Non-in-place add to avoid mutating the persistent GPU buffer --
            # in-place += would accumulate on CUDA graph padding slots.
            positions = positions + self.padding_idx + 1
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
