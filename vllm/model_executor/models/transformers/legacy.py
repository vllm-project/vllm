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

from inspect import signature
from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.interfaces_base import get_score_type
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class LegacyMixin:
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Drop unsupported/unwanted output embeddings layers.
        self.hf_to_vllm_mapper.orig_to_new_prefix.update(
            {
                "model.lm_head.": None,
                "model.predictions.": None,
                "model.qa_outputs.": None,
                "model.embeddings_project.": None,
                "model.discriminator_predictions.": None,
            }
        )

        # Some encoder models have the position_ids buffer in the checkpoint. vLLM will
        # always pass position_ids as an argument, so we drop the  buffer if it exists.
        self.hf_to_vllm_mapper.orig_to_new_substr["position_ids"] = None

        # Some encoder models have the bias of the final classifier layer in the
        # checkpoint. vLLM does not use this bias, so we drop it if it exists.
        self.hf_to_vllm_mapper.orig_to_new_substr["score.bias"] = None

        # roberta-like models an extra padding in positions.
        # FIXME(Isotr0py): This is quite hacky for roberta edge case,
        # we should find a better way to handle this.
        self.is_roberta = "roberta" in self.text_config.model_type
        self.padding_idx = self.text_config.pad_token_id
        self.register_buffer(
            "_token_type_ids",
            torch.zeros(
                vllm_config.scheduler_config.max_num_batched_tokens,
                dtype=torch.int32,
                device=self.device_config.device,
            )
            if getattr(self.text_config, "type_vocab_size", 0) > 0
            and "token_type_ids" in signature(self.model.forward).parameters
            else None,
            persistent=False,
        )
        if (
            self._token_type_ids is not None
            and get_score_type(self) == "cross-encoder"
            and envs.VLLM_USE_BREAKABLE_CUDAGRAPH
            and vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            raise ValueError(
                "Transformers cross-encoders with token type IDs do not support "
                "breakable CUDA graphs. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 or "
                "use enforce_eager=True."
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        kwargs = {}
        if self._token_type_ids is not None:
            # Warmup and requests must use the same persistent graph input.
            padded_token_type_ids = self._token_type_ids[: positions.shape[-1]]
            padded_token_type_ids.zero_()
            if token_type_ids is not None:
                padded_token_type_ids[: token_type_ids.shape[-1]].copy_(token_type_ids)
            kwargs["token_type_ids"] = padded_token_type_ids.unsqueeze(0)
        elif token_type_ids is not None:
            raise ValueError(
                "Token type IDs require a Transformers model with a dedicated "
                "token type vocabulary and token_type_ids input."
            )
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
